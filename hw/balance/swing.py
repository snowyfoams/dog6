"""The swing leg: straight up and back down on the spot it left.  TRUNK frame.

    rest_feet_b(h, foot_xy)                   -> (4, 3) where each foot rests
    swing_reference(rest_b, s, duration)      -> (p, v) on the arc
    swing_torque(state, leg, p_ref, v_ref)    -> (3,) the Cartesian impedance
    joint_swing_reference(leg, rest_b, s, duration, q_seed) -> (q, qd)
    joint_swing_torque(state, leg, q_ref, qd_ref)           -> (3,) joint PD

    tau_i = J_i^T [ Kp (p_ref - x_i) + Kd (v_ref - J_i qd_i) ]   + leg gravity

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
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402
from sim.cmpc.swing import SwingTrajectory   # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402

__all__ = ["rest_feet_b", "swing_reference", "swing_torque",
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


def swing_reference(rest_b, progress: float, duration: float,
                    height: float = cfg.SWING_HEIGHT, land_b=None):
    """``(p, v)`` for one foot, trunk frame, at swing `progress` in [0, 1].

    `land_b` None lands on `rest_b`, the in-place trot.  Given, the arc
    lands THERE instead -- the hold's foot step (`law.BalanceLaw.begin_step`),
    not placement: a fixed trunk-frame point, no velocity term.
    """
    land_b = rest_b if land_b is None else land_b
    arc = SwingTrajectory(rest_b, land_b, height=height, duration=duration)
    p, v, _ = arc.at(progress)
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


#: kg m^2 about the pitch and knee axes, AN ESTIMATE FOR THE BANNER ONLY:
#: the reflected rotor (`params.ARMATURE`, 0.0085, which is most of it) plus
#: the links -- thigh and shin about the pitch axis, shin about the knee -- at
#: their lift-pose lever arms.  No torque is computed from these; they turn
#: the arc's joint acceleration into the N*m the SWING WOULD NEED so the
#: operator can hold that against the cap and the slew before pressing T.
JOINT_INERTIA_EST = np.array([0.0, P.ARMATURE + 0.013, P.ARMATURE + 0.005])


def swing_demand(leg: int, rest_b, duration: float, height: float,
                 q_seed, n: int = 200) -> dict:
    """What one swing ASKS of the pitch and knee motors, from the arc alone.

    The z-only arc through the IK (`joint_swing_reference`), sampled and
    differentiated: peak joint speed, peak inertial torque on
    `JOINT_INERTIA_EST`, and the torque rate the gate's slew limiter must
    allow for the torque to reach its peak inside a quarter of the swing --
    ``tau_peak / (duration / 4)``.  THAT LAST NUMBER IS THE ONE THAT DECIDES
    WHETHER THE FOOT LEAVES THE FLOOR.  A swing whose demand outruns the slew
    is a PD winding up behind a limiter: MuJoCo, 2026-09-17 (`hw.fold_trot`),
    160, 240 and 260 ms swings at 40 mm all diverged with the knee
    overshooting ~90 deg, 300 ms tracked to 2.9 mm.  Measured on the robot,
    2026-09-24: at 80 ms of swing the foot simply did not lift, and the trunk
    read steady because it was still standing on four feet.

    Returns ``{"qd": (3,), "tau": (3,), "slew": float}`` -- peak |qd| rad/s,
    peak |tau| N*m per joint, and the slew in N*m/s.
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
