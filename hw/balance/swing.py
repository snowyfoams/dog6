"""The swing leg: straight up and back down on the spot it left.  TRUNK frame.

    rest_feet_b(h, foot_xy)                   -> (4, 3) where each foot rests
    swing_reference(rest_b, s, duration)      -> (p, v) on the arc
    swing_reference_pva(rest_b, s, duration)  -> (p, v, a) on the arc
    swing_torque(state, leg, p_ref, v_ref)    -> (3,) the Cartesian impedance
    swing_feedforward(leg, q, v_ref, a_ref)   -> (3,) M0 J^+ (a_ref - Jdot qd_ref)
    joint_swing_reference(leg, rest_b, s, duration, q_seed) -> (q, qd)
    joint_swing_torque(state, leg, q_ref, qd_ref)           -> (3,) joint PD

    tau_i = J_i^T [ Kp (p_ref - x_i) + Kd (v_ref - J_i qd_i) ]   + leg gravity
            + M0 J_i^+ (a_ref - Jdot_i qd_ref)          (--swing-ff, 2026-09-25)

cMPC's swing law, eq (1), with two things taken out on the operator's word and
on a measurement -- and nothing else changed:

    NO PLACEMENT (eq 33).  The operator's decision, 2026-09-16.  The arc
    starts and ends at the same trunk-frame point: the foot's resting site in
    the posture being held, which is the `foot_xy` the tracking reference
    already pins and the height the reference is commanding.  This is exactly
    DOG5's `swing_foot_body` with its step at zero, and it needs no velocity
    estimate, no world position and no rotation -- DOG6 has none of the three.

    NO FEEDFORWARD (eq 2-3's Lambda and bias).  Measured on this Pi on
    2026-09-16: `sim.cmpc.swing.swing_torque` with feedforward is 2.3 ms PER
    LEG -- two swing legs are more than the whole 4 ms sweep, against a law
    that already runs slot 0 at ~440 us.  The PD alone is what DOG5 flew,
    at DOG5's gains.  Leg gravity is NOT added here: `torque.stance_torque`
    already gives every leg its own weight, and a swing leg's allocated force
    is exactly zero, so what it gets there is gravity and nothing else.

THE INERTIAL TERM IS BACK, IN THE FORM THAT COSTS 80 us A LEG, 2026-09-25
    The operator's diagnosis, on watching a 15 mm apex not lift and a 40 mm
    one lift and shake: the swing law feeds forward gravity and nothing else,
    and what the arc actually asks for is M qdd.  Measured on the leg's own
    dynamics (`sim.leg_dynamics`, RNEA over the CAD inertials, armature in)
    along a 140 ms / 15 mm arc at the lift pose:

        M qdd       2.3 pitch / 2.4 knee N*m   the term that was missing
        C qd        0.01 N*m                    not worth a multiply
        g           0.36 N*m                    already fed forward
        the PD      ~0.5 N*m at most, with the WHOLE apex as its error

    So the PD was being asked to make 90 % of the torque out of a tracking
    error it could not build in 140 ms, and the apex was working as a force
    gain, not a height.  The Coriolis term is nothing because the reflected
    rotor -- 61 / 96 % of the pitch / knee diagonal of M -- has no velocity
    term, and the links that do are light.

    WHY IT IS CHEAP WHERE cMPC's WAS NOT.  Lambda = (J M^-1 J^T)^-1 and the
    RNEA's Jdot qdot cost 2.2 ms a leg on this Pi.  But M at the lift pose
    is 0.0116 / 0.0139 / 0.0089 on the diagonal with nothing above 0.0006
    off it, and it barely moves along an in-place z arc, so the CONSTANT
    `JOINT_INERTIA_M0` is M to a few percent.  qdd_ref is the arc's own
    acceleration, less Jdot qd_ref, through the pitch and knee columns of the
    MEASURED Jacobian -- the least-squares `joint_swing_reference` forms qd
    with, on the reference's Jacobian; the two differ by the tracking error,
    which is the PD's business.  Jdot qd_ref is one more closed-form Jacobian
    at q + qd_ref * eps, and it is NOT optional: the first cut left it out
    and the operator watched the swing foot leave the z line fore-aft and
    come back on the PD.  One leg in its own dynamics, 20 mm / 140 ms: 4.3 mm
    of x at touchdown without it, 0.8 with it, same apex.  Two 2x2 solves.
    MEASURED in `law.update` under the robot's venv python, two legs
    swinging: +163 us in the Cartesian mode (729 -> 891 us p50), +153 in the
    joint (1149 -> 1302) -- on a swing sweep that is already past the 333 us
    slot in either mode.  Against the arc it reproduces `swing_demand` to a
    few percent and pushes the same way the PD does through the first
    quarter of the swing.  `swing_feedforward` is the whole of it.

    WHAT THE PD IS FOR ONCE THE FEEDFORWARD LIFTS, measured the same way
    (one leg, RNEA, slew 150, delay, filter, 20 mm / 140 ms): Kp_z is NOT
    the knob.  A Cartesian PD without Lambda couples a z force into x through
    M^-1 J^T, so Kp_z 180 -> 800 took the x drift from 2 to 10 mm and 1500
    to 19 for a few mm of apex.  Kd_z is: 15 -> 30 (zeta 0.28 -> 0.57 on the
    foot's 3.9 kg) took the z rms from 3.5 to 2.8 mm and the apex from 16.4
    to 17.7 for 1 mm of x; past 40 the x cost outruns the gain.  And the
    landing speed belongs to neither -- see `swing_demand`.  `--kp-swing`,
    `--kd-swing`, defaults DOG5's.

    WHAT FLEW ON THE BENCH, 2026-09-25: `--kd-swing 20 20 40` WITH THE
    FEEDFORWARD OFF tracked the apex well (the operator's reading, robot hung
    up, Cartesian swing).  The arithmetic behind it: Kd (v_ref - v) carries
    Kd v_ref, a velocity feedforward -- 40 N s/m x 0.78 m/s at the peak of a
    20 mm / 140 ms arc is 31 N at the foot, 1.6 N*m on the pitch, the same
    order as the inertial term and a quarter-cycle behind it, and unlike M0
    qdd_ref it carries no armature number to be wrong about.  Kd_z 40 is
    zeta 0.75 on the foot's 3.9 kg.  That is the setting to take to the trot
    first; the feedforward waits on a measured armature.

    IT IS OPEN LOOP, AND THAT IS THE POINT: the torque the arc needs arrives
    when the arc needs it, not after the error has grown.  It does not buy
    anything from the slew -- the feedforward IS the tau_peak / (swing/4)
    curve the banner holds against `--tau-slew`.  Off by default
    (`--swing-ff`), identical in both swing modes, and its share of tau is
    logged as `tau_ff` so a run can say what it did.

THE ARC IS `sim.cmpc.swing.SwingTrajectory`, IMPORTED, NOT COPIED
    Two quintic half-arcs to one absolute apex: zero velocity AND zero
    acceleration at liftoff, apex and touchdown.  DOG5's was two cubic
    smoothsteps, zero velocity only.  With start == end the horizontal
    component is identically zero and the foot moves in trunk z alone.

THE JOINT SWING, 2026-09-17: THE FOLD'S OWN, NOT SHARED
    The operator watched the fold stand's swing leg go round the abduction
    axis instead of lifting.  The arc is not why: along it, in the fold stance,
    abd moves 0.7 deg at the 40 mm apex.  THE CARTESIAN PD IS WHY.  Mapped to
    the joints, J^T Kp J at the fold's lift pose is 3.8 / 4.0 / 1.8 N*m/rad
    (abd / pitch / knee) with 0.22 / 0.23 / 0.13 N*m*s/rad of damping -- a
    165 mm lever turns 140 N/m in y into almost no abduction stiffness, so
    the leg is close to limp about abd and anything that pushes it swings it.

    So the fold gets `joint_swing_*`: the SAME z-only arc, solved through the
    closed-form IK every sweep, with ABD HELD at the resting foot's IK angle,
    and a joint PD on that.  Z IN THE TRUNK FRAME AND NOTHING ELSE: no x, no
    y, no abduction.  Gains in `config.KP_SWING_JOINT`.  The nominal and wide
    trots keep the Cartesian swing they flew.

    AND THE NOMINAL TROT KEEPS IT FOR A REASON, 2026-09-25: `--swing joint`
    on that posture failed every run it was tried on.  What lifts the nominal
    trot's foot is the Cartesian PD WITH `--swing-ff` (below), at 20 mm of
    apex.  At 40 mm the feedforward brought the foot down hard enough to
    bounce the robot: with the inertial term in, the apex is real and so is
    the descent, whose speed scales with the apex.  20 mm is the flown number.

WHY THE TRUNK FRAME AND NOT THE WORLD
    cMPC generates the reference in the world because placement involves the
    ground.  With placement off nothing does: the reference is a fixed point
    under the hip plus a bump, the Jacobian is trunk-frame, and so is the
    measured foot.  No R appears anywhere in this file, so nothing here can be
    wrong about the attitude -- which was DOG5's argument for the same choice.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/swing.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import leg_dynamics as LD   # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402
from sim.cmpc.swing import SwingTrajectory   # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402

__all__ = ["rest_feet_b", "swing_reference", "swing_reference_pva",
           "swing_torque", "swing_feedforward", "feedforward_inertia",
           "joint_swing_reference", "joint_swing_torque", "swing_demand",
           "SWING_MODES"]

#: `law.BalanceLaw.swing`: the Cartesian impedance, or the fold's joint PD.
SWING_MODES = ("cartesian", "joint")


def rest_feet_b(h: float, foot_xy=None) -> np.ndarray:
    """(4, 3) TRUNK-frame foot sites with the feet at `foot_xy` and the trunk
    bottom at `h` -- the same point `law.ik_reference` solves the IK for.

    `foot_xy` is HIP-frame, as everywhere in this package; None is the
    nominal `sim.stand.FOOT_XY`.
    """
    foot_xy = ST.FOOT_XY if foot_xy is None else foot_xy
    height = float(h) + cfg.TRUNK_BOTTOM_OFFSET
    rest = np.empty((C.N_LEGS, 3))
    rest[:, :2] = np.asarray(P.HIP_OFFSET, dtype=float)[:, :2] + np.asarray(
        foot_xy, dtype=float)
    rest[:, 2] = -(height - P.FOOT_RADIUS)
    return rest


def swing_reference_pva(rest_b, progress: float, duration: float,
                        height: float = cfg.SWING_HEIGHT, land_b=None):
    """``(p, v, a)`` for one foot, trunk frame, at swing `progress` in [0, 1].

    `land_b` None lands on `rest_b`, the in-place trot.  Given, the arc
    lands THERE instead -- the hold's foot step (`law.BalanceLaw.begin_step`),
    not placement: a fixed trunk-frame point, no velocity term.  `a` is what
    `swing_feedforward` turns into torque.
    """
    land_b = rest_b if land_b is None else land_b
    arc = SwingTrajectory(rest_b, land_b, height=height, duration=duration)
    return arc.at(progress)


def swing_reference(rest_b, progress: float, duration: float,
                    height: float = cfg.SWING_HEIGHT, land_b=None):
    """``(p, v)`` -- `swing_reference_pva` without the acceleration."""
    p, v, _ = swing_reference_pva(rest_b, progress, duration, height, land_b)
    return p, v


def swing_torque(state, leg: int, p_ref, v_ref, kp=None, kd=None) -> np.ndarray:
    """(3,) the swing impedance for `leg`, from this sweep's encoders.

    Uses the Jacobian `state.read` already computed -- no second chain walk.
    """
    kp = cfg.KP_SWING if kp is None else np.asarray(kp, dtype=float)
    kd = cfg.KD_SWING if kd is None else np.asarray(kd, dtype=float)
    jac = state.jac[leg]
    qd = C.unflat(state.qd)[leg]
    force = (kp * (np.asarray(p_ref, dtype=float) - state.x_b[leg])
             + kd * (np.asarray(v_ref, dtype=float) - jac @ qd))
    return jac.T @ force


def joint_swing_reference(leg: int, rest_b, progress: float, duration: float,
                          q_seed, height: float = cfg.SWING_HEIGHT):
    """``(q, qd)`` for one leg: the z-only arc through the IK, abd HELD.

    `rest_b` is the TRUNK-frame resting site (`rest_feet_b`); `q_seed` picks
    the IK branch -- pass the leg's measured joints.  Abd is fixed at the
    resting site's IK angle for the whole swing, and pitch and knee alone
    make the lift: their rates are the least-squares solve of the arc's z
    velocity through the Jacobian's pitch and knee columns.
    """
    hip = np.asarray(P.HIP_OFFSET[leg], dtype=float)
    rest_b = np.asarray(rest_b, dtype=float)
    q_rest = HK.leg_ik(leg, rest_b - hip, q_seed=q_seed)
    p, v = swing_reference(rest_b, progress, duration, height)
    q = HK.leg_ik(leg, p - hip, q_seed=q_rest)
    q[0] = q_rest[0]
    jac = HK.foot_jacobian(leg, q)
    qd = np.zeros(3)
    qd[1:] = np.linalg.lstsq(jac[:, 1:], v, rcond=None)[0]
    return q, qd


#: kg m^2 about the abd, pitch and knee axes: the DIAGONAL of the leg's
#: joint-space mass matrix at the lift pose -- `leg_dynamics.mass_matrix`,
#: RNEA over `params.LINK_INERTIALS` with the armature in -- 0.0116 / 0.0139 /
#: 0.0089.  The reflected rotor (`params.ARMATURE`, 0.0085) is 74 / 61 / 96 %
#: of it and no off-diagonal term is above 0.0006, so a constant diagonal IS
#: the matrix to within a few percent along an in-place z arc.  FOR THE
#: BANNER ONLY: no torque is computed from it; it turns the arc's joint
#: acceleration into the N*m the swing would need, to hold against the cap
#: and the slew before T.  (Until 2026-09-25 this was a hand estimate,
#: 0.0219 / 0.0133, that overstated the pitch by 60 %.)
#: The full 3x3 M at the lift pose is what `swing_feedforward` multiplies
#: by; the diagonal below it is the banner's arithmetic.  The off-diagonal
#: signs are the robot's: `hw.kinematics` and `sim.kinematics` agree to
#: 1e-12 over random poses (`sim.selftest`), so the joint coordinates are one
#: convention.
def feedforward_inertia(armature: float | None = None) -> np.ndarray:
    """The 3x3 M0 the feedforward multiplies by: the links at the lift pose
    plus `armature` on the diagonal -- `params.ARMATURE` (DOG5's 8.5e-5 x
    10^2, INHERITED, NOT MEASURED ON DOG6) unless a measured one is given.

    THE ARMATURE IS 61-96 % OF M0, SO IT IS THE FEEDFORWARD'S GAIN.  On the
    bench, 2026-09-25, with the feedforward on, a 15 mm arc reached 26 mm and
    the foot drifted 15-21 mm in x, at a tau_ff peak of 2.2 N*m and a PD peak
    of 0.5: the leg answered the model's torque with 1.7x the model's motion.
    The operator's reading; the inference is that the real reflected rotor is
    smaller than the inherited number, and `hw.swing_bench --analyse` fits it
    from a log's own q and tau (`--ff-armature` then puts it here).
    """
    q_lift = np.asarray(cfg.NOMINAL_POSE, dtype=float)[0]
    links = LD.mass_matrix(0, q_lift, armature=False)
    a = P.ARMATURE if armature is None else float(armature)
    return links + a * np.eye(3)


JOINT_INERTIA_M0 = feedforward_inertia()
JOINT_INERTIA_EST = np.diag(JOINT_INERTIA_M0)

#: Seconds along qd_ref for the finite-difference Jdot qdot in
#: `swing_feedforward`: 6 rad/s x 1e-4 s is 0.6 mrad of joint, far inside
#: the closed form's accuracy and far outside float noise.
_JDOT_EPS = 1.0e-4


def swing_demand(leg: int, rest_b, duration: float, height: float,
                 q_seed, n: int = 200) -> dict:
    """What one swing ASKS of the pitch and knee motors, from the arc alone.

    The z-only arc through the IK (`joint_swing_reference`), sampled and
    differentiated: peak joint speed, peak inertial torque on
    `JOINT_INERTIA_EST`, and the torque rate the gate's slew limiter must
    allow for the torque to reach its peak inside a quarter of the swing --
    ``tau_peak / (duration / 4)``.  THAT LAST NUMBER SAYS WHETHER THE FOOT
    CAN FOLLOW THE ARC -- NOT WHETHER IT LIFTS.  Past it the foot still
    leaves the floor (it does at 80 ms of swing on the robot, 2026-09-25),
    but on the PD's terms behind the limiter rather than the arc's: late and
    short, or wound up and over.  MuJoCo, 2026-09-17 (`hw.fold_trot`): 160,
    240 and 260 ms swings at 40 mm all diverged with the knee overshooting
    ~90 deg, 300 ms tracked to 2.9 mm.  Which of the two a run did is in the
    log -- `x_b` against `p_swing` while `swing_s > 0`.

    Returns ``{"qd": (3,), "tau": (3,), "slew": float}`` -- peak |qd| rad/s,
    peak |tau| N*m per joint, and the slew in N*m/s.

    THE SLEW THAT LETS THE TORQUE RISE IS NOT THE SLEW THAT LETS THE FOOT
    LAND.  With the feedforward on the braking half of the arc is a torque
    reversal the limiter also shaves, and what it shaves arrives at the floor
    as descent speed.  One leg in its own dynamics, 140 ms, Kp_z 180, the
    runner's delay and 40 Hz velocity filter: 20 mm needs ~99 N*m/s by this
    number and lands at -0.17 m/s on 150, 0.00 on 300; 40 mm needs ~190 and
    lands at -0.85 on 150, -0.28 on 300, -0.02 on 600.  About THREE TIMES the
    rise number, which is what the banner prints as "to land softly" -- an
    empirical factor from those two points, 2026-09-25, the day 40 mm on 150
    bounced the robot.
    """
    s = np.linspace(0.0, 1.0, n + 1)
    q = np.array([joint_swing_reference(leg, rest_b, float(x), duration,
                                        q_seed, height=height)[0] for x in s])
    dt = duration / n
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    tau = np.abs(qdd) * JOINT_INERTIA_EST
    return dict(qd=np.abs(qd).max(axis=0), tau=tau.max(axis=0),
                slew=float(tau.max() / (duration / 4.0)))


def swing_feedforward(leg: int, q, v_ref, a_ref, jac=None,
                      inertia=None) -> np.ndarray:
    """(3,) tau_ff = M0 qdd_ref: the inertial torque the arc asks for.

    qdd_ref is the arc's TRUNK-frame foot acceleration `a_ref`, LESS the
    Jdot qdot the chain produces on its own at the arc's joint rates, through
    the pitch and knee columns of the leg's Jacobian at the measured `q` --
    the least-squares `joint_swing_reference` uses for qd.  Abd gets none:
    the arc has no y in it.

    JDOT QDOT IS IN, AND IT IS WHY THE FOOT STAYS ON THE ARC'S LINE.  Without
    it qdd_ref = J^+ a_ref makes the foot accelerate along a_ref only at the
    instant q stops changing; along the real arc the chain's own curvature
    adds a horizontal term, and the foot leaves the z line SIDEWAYS.  One
    leg in its own dynamics (RNEA, trunk pinned, Cartesian PD, slew 150), a
    20 mm / 140 ms arc: 4.3 mm of x drift at touchdown with it left out,
    0.8 with it in, at the same apex -- and the operator saw the drift on
    the robot the same day, 2026-09-25.  It costs one more Jacobian, at
    q + qd_ref * eps, on the closed form's 8 us.

    `jac` is the leg's Jacobian at `q` if the caller already has it
    (`BodyState.jac`, the same closed form); None computes it.  `inertia` is
    the full 3x3 `JOINT_INERTIA_M0` unless given; C qd is left out at 0.01
    N*m -- the rotor that dominates M has no velocity term.  See the module
    docstring, "THE INERTIAL TERM IS BACK".

    The two least-squares solves are the 2x2 normal equations, J2^T J2 x =
    J2^T b, not `lstsq`: same answer on a Jacobian this far from singular,
    12 us against 17, and the 2x2 is shared.  57 us the call, measured.
    """
    inertia = (JOINT_INERTIA_M0 if inertia is None
               else np.asarray(inertia, dtype=float))
    q = np.asarray(q, dtype=float)
    jac = HK.foot_jacobian(leg, q) if jac is None else np.asarray(jac, float)
    j2 = jac[:, 1:]
    normal = j2.T @ j2
    qd_ref = np.zeros(3)
    qd_ref[1:] = np.linalg.solve(normal, j2.T @ np.asarray(v_ref, dtype=float))
    jdot_qd = (HK.foot_jacobian(leg, q + qd_ref * _JDOT_EPS) - jac) @ qd_ref
    jdot_qd /= _JDOT_EPS
    qdd_ref = np.zeros(3)
    qdd_ref[1:] = np.linalg.solve(
        normal, j2.T @ (np.asarray(a_ref, dtype=float) - jdot_qd))
    return inertia @ qdd_ref


def joint_swing_torque(state, leg: int, q_ref, qd_ref, kp=None,
                       kd=None) -> np.ndarray:
    """(3,) joint PD, `tau = Kp (q_ref - q) + Kd (qd_ref - qd)`.

    Leg gravity is NOT added, for the same reason as `swing_torque`.
    """
    kp = cfg.KP_SWING_JOINT if kp is None else np.asarray(kp, dtype=float)
    kd = cfg.KD_SWING_JOINT if kd is None else np.asarray(kd, dtype=float)
    q = C.unflat(state.q)[leg]
    qd = C.unflat(state.qd)[leg]
    return (kp * (np.asarray(q_ref, dtype=float) - q)
            + kd * (np.asarray(qd_ref, dtype=float) - qd))
