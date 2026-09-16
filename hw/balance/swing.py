"""The swing leg: straight up and back down on the spot it left.  TRUNK frame.

    rest_feet_b(h, foot_xy)                   -> (4, 3) where each foot rests
    swing_reference(rest_b, s, duration)      -> (p, v) on the arc
    swing_torque(state, leg, p_ref, v_ref)    -> (3,) the Cartesian impedance

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

from . import config as cfg          # noqa: E402

__all__ = ["rest_feet_b", "swing_reference", "swing_torque"]


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
                    height: float = cfg.SWING_HEIGHT):
    """``(p, v)`` for one foot, trunk frame, at swing `progress` in [0, 1]."""
    arc = SwingTrajectory(rest_b, rest_b, height=height, duration=duration)
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
