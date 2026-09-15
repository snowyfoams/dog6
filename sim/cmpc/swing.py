"""Swing-leg control: where to put the foot, and how to get it there.

    p_des = p_ref + v_CoM * dt / 2                                       (33)

    tau_i = J' [Kp (Bp_ref - Bp) + Kd (Bv_ref - Bv)] + tau_ff            (1)
    tau_ff = J' Lambda (Ba_ref - Jdot qdot) + C qdot + G                 (2,3)

THIS IS THE PART OF THE METHOD THAT IS NOT OPTIMISED
    The MPC is convex GIVEN where the feet are.  Deciding where to put the
    next one is eq (33), a heuristic, and it sits entirely outside the QP.  So
    the convexity of the whole controller is convexity of a subproblem: the
    placement law chooses the contact geometry, and the optimiser then makes
    the best of it.  When this controller falls over, look here first.

WHAT EQUATION (33) IS SAYING
    Put the foot where the hip will be standing over it at the MIDDLE of the
    coming stance.  The body travels v_CoM * dt over a stance of length dt, so
    landing half that distance ahead of the hip makes the foot start stance
    ahead of the hip and finish it behind, symmetric about the middle.  A foot
    placed under the hip instead would spend the entire stance behind it and
    the body would decelerate.

    It carries NO velocity-error feedback.  The full Raibert heuristic adds a
    term in (v - v_cmd) that actively corrects a velocity the robot has failed
    to reach; (33) alone only sustains the velocity it already has.  The paper
    writes (33), so (33) is what is here -- but it means the MPC's wrench is
    the only thing correcting a velocity error, and a large enough one has to
    be corrected by pushing rather than by stepping.

THE FRAMES, WHICH ARE THE EASY THING TO GET WRONG
    The placement is computed in WORLD axes -- it involves the ground, and the
    CoM velocity, neither of which mean anything in a rotating frame.  The
    control law (1) is written in BODY axes, because that is where the leg's
    Jacobian and mass matrix live.  So the reference is generated in world and
    converted per control step:

        Bp_ref = R' (p_world_ref - p_trunk)
        Bv_ref = R' (v_world_ref - v_trunk)

    The velocity conversion DROPS a term: strictly, a point fixed in world has
    body-frame velocity R'(v_w - v_trunk) - w x Bp.  The omega cross term is
    left out, which is the same trunk-as-inertial approximation `leg_dynamics`
    makes for Lambda and C qdot + G, and is consistent with it.  At 1 rad/s
    and a 0.2 m leg it is 0.2 m/s, which is not nothing -- it is absorbed by
    Kd, and it is the first thing to add if swing tracking degrades in a fast
    turn.

    Note also that these are relative to the TRUNK ORIGIN, not the CoM.  The
    leg kinematics are written in the trunk frame; the MPC's body model uses
    the CoM.  They are 29 mm apart in z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import kinematics as K
from .. import leg_dynamics as LD
from .. import params as P
from . import config as cfg


# ===========================================================================
# where the foot goes
# ===========================================================================
def foot_placement(hip_world, velocity_world, stance_duration=None,
                   ground_height: float = 0.0) -> np.ndarray:
    """Equation (33).  Where this foot should touch down, in world axes.

    `hip_world` is this leg's hip hinge, in world coordinates.  The paper's
    `p_ref` is the hip's projection onto the ground, so only its xy is used.

    WHICH VELOCITY GOES IN HERE IS THE WHOLE QUESTION -- see
    `hip_velocity` below and `cfg.YAW_PLACEMENT_CORRECTION`.  Written with the
    CoM velocity this is (33) literally, and it cannot turn.

    The z target is the ground plus the foot ball's radius, because the foot
    SITE is the ball's centre -- aiming the site at the ground would bury the
    foot by 15 mm.
    """
    stance_duration = (cfg.STANCE_DURATION if stance_duration is None
                       else float(stance_duration))
    hip_world = np.asarray(hip_world, dtype=float).reshape(3)
    velocity = np.asarray(velocity_world, dtype=float).reshape(3)

    target = np.empty(3)
    target[:2] = hip_world[:2] + velocity[:2] * stance_duration * 0.5
    target[2] = ground_height + P.FOOT_RADIUS
    return target


def hip_velocity(com_velocity, omega, hip_world, com_world) -> np.ndarray:
    """The velocity of the HIP, not of the CoM: ``v_com + w x r_hip``.

    OFF BY DEFAULT -- see `cfg.YAW_PLACEMENT_CORRECTION`.  The argument behind
    (33) is that a foot should land where its hip will be at the MIDDLE of the
    coming stance, so the stance is symmetric about the hip and the body
    neither accelerates nor decelerates across it.  For pure translation the
    hip travels with the CoM and v_CoM is exactly the right velocity.

    Under a yaw rate it is not: the hip sits 0.17 m off the turn axis and
    moves at w x r_hip even when the CoM is stationary.  At 90 deg/s that is
    0.26 m/s, and over half a stance it displaces the target by 33 mm.
    Replacing v_CoM with the hip's own velocity is the first-order form of the
    same "under the hip at mid-stance" argument in a rotating frame, and
    reduces EXACTLY to (33) when w = 0.  It is what the Cheetah
    implementation does by rotating the hip location by half the stance's yaw.

    AND ON DOG6 IT CHANGES NOTHING MEASURABLE.  10 s at 20/40/60/90 deg/s
    tracks 95-96 % of the commanded yaw with or without it, at under 0.3 deg
    rms of roll either way.  That is the paper's own claim appearing in the
    data rather than in its abstract: a 33 mm error in a heuristic foot
    placement is absorbed by an optimiser that is already planning a horizon
    of wrenches and has yaw-moment authority to spare.  Worth knowing before
    reaching for this knob to fix a turn -- when a turn here diverged, the
    cause was the yaw branch cut in `controller.solve`, not the placement.
    """
    lever = np.asarray(hip_world, float) - np.asarray(com_world, float)
    return np.asarray(com_velocity, float) + np.cross(
        np.asarray(omega, float), lever)


def _smoothstep(s):
    """Quintic 6s^5 - 15s^4 + 10s^3 and its first two derivatives.

    Quintic, not cubic: it has zero FIRST AND SECOND derivative at both ends.
    The second derivative matters because `a_ref` feeds the Lambda
    feedforward directly -- a cubic's acceleration steps discontinuously at
    liftoff and touchdown, and that step is multiplied by an apparent mass of
    4.5 kg and injected straight into the knee.
    """
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    value = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
    first = 30.0 * s * s * (1.0 + s * (-2.0 + s))
    second = 60.0 * s * (1.0 + s * (-3.0 + 2.0 * s))
    return value, first, second


@dataclass
class SwingTrajectory:
    """The arc one foot follows, from where it left to where it will land.

    Generated in WORLD axes and evaluated by swing progress in [0, 1].  The
    horizontal and vertical profiles are separate because they want different
    shapes: xy interpolates once, z goes up and comes back down.
    """

    start: np.ndarray
    end: np.ndarray
    height: float = cfg.SWING_HEIGHT
    duration: float = cfg.SWING_DURATION

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float).reshape(3).copy()
        self.end = np.asarray(self.end, dtype=float).reshape(3).copy()

    def at(self, progress: float):
        """``(p, v, a)`` in world axes at a swing progress in [0, 1]."""
        s = float(np.clip(progress, 0.0, 1.0))
        rate = 1.0 / self.duration                       # ds/dt

        # -- horizontal: one smooth interpolation start -> end -------------
        h, dh, ddh = _smoothstep(s)
        delta = self.end - self.start
        pos = self.start + delta * h
        vel = delta * dh * rate
        acc = delta * ddh * rate * rate

        # -- vertical: up to the apex, then down to the landing ------------
        # Two half-arcs.  Both ends of each half have zero slope and zero
        # curvature, so the apex joins smoothly and the foot leaves and
        # arrives with no vertical velocity.
        #
        # THE APEX IS ABSOLUTE, NOT "height ABOVE EACH END", AND THAT MATTERS
        # WHEN THE TWO ENDS DIFFER IN z.  Measuring the rise from each end
        # separately gives the two halves different apex values and the
        # trajectory jumps at mid-swing -- a step of (end_z - start_z)
        # straight into a 4.5 kg apparent mass.  Both halves are anchored to
        # one apex instead, so the arc clears `height` above the HIGHER end
        # and more above the lower one.
        apex = max(self.start[2], self.end[2]) + self.height
        if s < 0.5:
            base, seg, dseg = self.start[2], 2.0 * s, 2.0
        else:
            base, seg, dseg = self.end[2], 2.0 - 2.0 * s, -2.0
        rise = apex - base
        h, dh, ddh = _smoothstep(seg)
        pos[2] = base + rise * h
        vel[2] = rise * dh * dseg * rate
        acc[2] = rise * ddh * dseg * dseg * rate * rate
        return pos, vel, acc


# ===========================================================================
# the control law
# ===========================================================================
def to_body(p_world, v_world, trunk_position, trunk_velocity, rotation):
    """World reference -> the trunk frame the leg's Jacobian works in.

    See the module docstring for the term this drops.
    """
    rot_t = np.asarray(rotation, dtype=float).T
    p_body = rot_t @ (np.asarray(p_world, float) - np.asarray(trunk_position, float))
    v_body = rot_t @ (np.asarray(v_world, float) - np.asarray(trunk_velocity, float))
    return p_body, v_body


def swing_torque(leg, q, qd, p_ref_body, v_ref_body, a_ref_body,
                 kp=None, kd=None, feedforward: bool = True) -> np.ndarray:
    """Equations (1)-(3) for one leg.  Returns three joint torques.

    `feedforward=False` drops the Lambda and bias terms, leaving a plain
    Cartesian PD.  That switch exists so the self-test can MEASURE what the
    feedforward is worth on this robot rather than assert it.
    """
    kp = cfg.KP_SWING if kp is None else np.asarray(kp, dtype=float)
    kd = cfg.KD_SWING if kd is None else np.asarray(kd, dtype=float)

    q = np.asarray(q, dtype=float).reshape(3)
    qd = np.asarray(qd, dtype=float).reshape(3)

    position, jacobian = K.leg_state(leg, q)
    velocity = jacobian @ qd

    force = (kp @ (np.asarray(p_ref_body, float) - position)
             + kd @ (np.asarray(v_ref_body, float) - velocity))
    tau = jacobian.T @ force

    if feedforward:
        # (2): the operational-space inertia turns a desired foot
        # ACCELERATION into the force that produces it, having removed the
        # part the chain produces on its own at constant joint rate.
        lam = LD.operational_inertia(leg, q)
        bias_accel = LD.jdot_qdot(leg, q, qd)
        tau = tau + jacobian.T @ (lam @ (np.asarray(a_ref_body, float) - bias_accel))
        # (3): gravity and Coriolis for the leg itself.  Without this the PD
        # has to hold the leg's own weight as a steady-state error.
        tau = tau + LD.bias(leg, q, qd)
    return tau


def body_frame_force(force_world, rotation) -> np.ndarray:
    """``fb = R' f`` -- the MPC's world-frame force, in body axes.

    SPLIT OUT OF `stance_torque` BECAUSE IT RUNS AT A DIFFERENT RATE.  R comes
    from the IMU at `cfg.IMU_HZ` (200 Hz); the Jacobian comes from the
    encoders at `cfg.CONTROL_HZ` (250 Hz).  Fusing them into one call would
    force both to the faster rate and quietly invent IMU samples that the
    hardware never produced.  The controller calls this on IMU ticks and holds
    the result; `stance_torque` consumes the held value on every sweep.

    Takes and returns the GROUND REACTION force -- the force on the robot --
    with the sign flip left to `stance_torque`, so that both halves of the
    chain speak about the same physical quantity.

    `force_world` may be (3,) or (4, 3); the rotation applies to each row.
    """
    force = np.asarray(force_world, dtype=float)
    rot_t = np.asarray(rotation, dtype=float).T
    return force @ rot_t.T if force.ndim == 2 else rot_t @ force


def stance_torque(leg, q, force_body) -> np.ndarray:
    """Joint torques that deliver an already-rotated ground reaction force.

    THE SIGN IS THE WHOLE CONTENT OF THIS FUNCTION.  `force_body` is the force
    the GROUND applies to the robot, in body axes -- that is what the MPC
    solves for, and for a standing robot its z component is positive.  To
    receive it, the leg has to push DOWN on the ground with the opposite
    force, so the endpoint force the actuators generate is -fb, and

        tau = J' (-fb)

    Getting this backwards produces a robot that drives itself into the floor
    at exactly the rate it should be holding itself up, which looks like a
    gain problem and is not.

    `q` is fresh encoder data every sweep; `force_body` is whatever
    `body_frame_force` last produced, up to one IMU period old.
    """
    jacobian = K.foot_jacobian(leg, np.asarray(q, dtype=float).reshape(3))
    return jacobian.T @ (-np.asarray(force_body, dtype=float).reshape(3))


def joint_pd(q, qd, q_ref, qd_ref=None, kp=None, kd=None) -> np.ndarray:
    """The joint-space floor under everything.

    A pure force law says nothing about where a joint should BE, so a leg that
    loses its contact -- or a swing whose Cartesian error is momentarily
    small in a direction the Jacobian cannot see -- has no restoring term.
    Small enough not to fight the MPC.
    """
    kp = cfg.KP_JOINT if kp is None else kp
    kd = cfg.KD_JOINT if kd is None else kd
    qd_ref = np.zeros(3) if qd_ref is None else np.asarray(qd_ref, dtype=float)
    return (kp * (np.asarray(q_ref, float) - np.asarray(q, float))
            + kd * (qd_ref - np.asarray(qd, float)))


def describe() -> str:
    from .. import coordinates as C

    q = C.Q_STAND[0]
    start = np.array([0.182, 0.065, P.FOOT_RADIUS])
    traj = SwingTrajectory(start, start + np.array([0.05, 0.0, 0.0]))
    rows = ["DOG6 swing leg",
            "  placement       p_des = p_hip + v_CoM * %.3f s / 2   (eq 33)"
            % cfg.STANCE_DURATION,
            "  arc             %.0f mm apex over %.3f s"
            % (1000 * cfg.SWING_HEIGHT, cfg.SWING_DURATION),
            "  gains           Kp %.0f N/m, Kd %.0f N s/m"
            % (cfg.KP_SWING[0, 0], cfg.KD_SWING[0, 0]),
            "  foot mass at stance  %.2f kg vertical, %.2f kg forward"
            % (LD.foot_apparent_mass(0, q, (0, 0, 1)),
               LD.foot_apparent_mass(0, q, (1, 0, 0))),
            "",
            "  s      z (mm)   vz (m/s)   az (m/s^2)"]
    for s in np.linspace(0.0, 1.0, 11):
        p, v, a = traj.at(s)
        rows.append("  %.1f   %6.1f   %8.3f   %9.2f"
                    % (s, 1000 * (p[2] - start[2]), v[2], a[2]))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
