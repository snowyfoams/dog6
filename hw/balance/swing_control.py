"""The swing leg's control law while walking.  TRUNK frame.  Two laws:

    swing_osc_torque(state, leg, ref, wn=..., zeta=..., inertia=...)
        -> (tau (3,), tau_ff (3,))           THE WALK'S DEFAULT ("osc")
    tau = M0 J^-1 (a_ref + wn^2 (p_ref - x) + 2 zeta wn (v_ref - J qd)
                   - Jdot qd)

    swing_control_torque(state, leg, ref, kp, kd, ff=..., inertia=...)
        -> (tau (3,), tau_ff (3,))           "impedance"
    tau = J^T [ Kp (p_ref - x) + Kd (v_ref - J qd) ]
        + M0 J^-1 (a_ref - Jdot qd_ref)                         --swing-ff

    (+ the leg's weight in both, which reaches every leg through
       `torque.stance_torque` with an allocated force of zero)

WHY THE TASK-SPACE LAW IS THE WALK'S
    Its gains are ACCELERATIONS, one bandwidth per trunk axis whatever the
    foot's apparent mass -- and that mass is the whole story on this leg:
    Lambda = (J M0^-1 J^T)^-1 is 0.31 / 0.36 / 5.8 kg in x / y / z at the
    fold stance (the reflected rotor through a short z lever).  So x/y
    bandwidth is nearly free and z bandwidth is torque RATE, which the gate
    slews at 120 N*m/s; `config.WN_SWING_OSC` is stiff in x/y and soft in z
    for that reason.  The impedance's force gains put the same Kp on a
    0.3 kg and a 5.8 kg axis.  doc/walk/README.md section 4 has the numbers.

WHAT IS DIFFERENT FROM swing.py
    `swing.swing_torque` is the impedance, and the trot in place keeps it
    unchanged.  Walking asks two things of it the in-place arc never did:

    1. THE FOOT MOVES SIDEWAYS.  The in-place arc is z alone, so its
       feedforward (`swing.swing_feedforward`) solves through the pitch and
       knee columns with abd held.  A walking arc has x AND y in it -- a
       lateral step, a turn -- so this solves through all three columns:
       qd_ref = J^-1 v_ref, qdd_ref = J^-1 (a_ref - Jdot qd_ref).  J is the
       measured one, a closed form already in `BodyState`, and far from
       singular in every stance this package holds (the fold stand's
       condition number is ~30).
    2. THE GAINS ARE THE WALK'S (`config.KP_SWING_WALK`): the in-place
       10 N/m in x/y is too soft to carry a foot across a stride (config
       has the arithmetic), and z keeps the operator's 400 / 40.

    Jdot qd is the one extra Jacobian at q + qd * eps, as in
    `swing.swing_feedforward`, and for the reason given there: without it
    the chain's own curvature pulls the foot off the arc.

THE ARMATURE IS A GAIN in the task-space law and in --swing-ff alike: M0's
rotor is DOG5's, not DOG6's (the bench read DOG6's as ~1/1.7 of it), and a
real rotor smaller by k makes the law 1/k too strong.  `--ff-armature`
replaces it; in the simulator M0 is exact by construction, and doc/walk
flies the bench's reading as well.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/swing_control.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402
from . import swing as SWING         # noqa: E402

__all__ = ["swing_control_torque", "swing_osc_torque", "inertial_feedforward",
           "SWING_LAWS"]

#: `walk.WalkPlan.swing_law`: the Cartesian impedance above, or the
#: task-space computed torque below.
SWING_LAWS = ("impedance", "osc")

_JDOT_EPS = 1.0e-4


def inertial_feedforward(leg: int, q, jac, v_ref, a_ref,
                         inertia=None) -> np.ndarray:
    """(3,) M0 qdd_ref, all three joints, at the measured `q` and `jac`."""
    inertia = (SWING.JOINT_INERTIA_M0 if inertia is None
               else np.asarray(inertia, dtype=float))
    q = np.asarray(q, dtype=float)
    jac = np.asarray(jac, dtype=float)
    qd_ref = np.linalg.solve(jac, np.asarray(v_ref, dtype=float))
    jdot_qd = (HK.foot_jacobian(leg, q + qd_ref * _JDOT_EPS) - jac) @ qd_ref
    jdot_qd /= _JDOT_EPS
    qdd_ref = np.linalg.solve(jac, np.asarray(a_ref, dtype=float) - jdot_qd)
    return inertia @ qdd_ref


def swing_osc_torque(state, leg: int, ref, wn=None, zeta=None, inertia=None):
    """``(tau, tau_ff)``: the TASK-SPACE COMPUTED TORQUE, the swing law the
    walk flies in MuJoCo (doc/walk/README.md says why):

        a_des = a_ref + wn^2 (p_ref - x) + 2 zeta wn (v_ref - J qd)
        tau   = M0 J^-1 (a_des - Jdot qd)

    It is the operational-space law with Lambda = (J M0^-1 J^T)^-1, since
    J^T Lambda = M0 J^-1 for a square J -- cMPC's swing law (sim.cmpc.swing
    eq 1-2) with the constant M0 of the lift pose in place of the 2.2 ms
    RNEA.  The gains are ACCELERATIONS, so the foot's 5.8 kg in z and 0.3 kg
    in x get one bandwidth, and a z error no longer pushes the foot in x
    through M^-1 J^T (swing.py's coupling, which on the fold's knees-back
    front legs drove a stepping foot into the floor).

    THE ARMATURE IS ITS GAIN, here as in `--swing-ff`: M0 carries DOG5's
    rotor, and a real one smaller by a factor k makes every gain and the
    feedforward 1/k too strong.  `--ff-armature` replaces it.  `tau_ff` is
    the share that does not read the error, M0 J^-1 (a_ref - Jdot qd)."""
    wn = cfg.WN_SWING_OSC if wn is None else np.asarray(wn, dtype=float)
    zeta = cfg.ZETA_SWING_OSC if zeta is None else float(zeta)
    inertia = (SWING.JOINT_INERTIA_M0 if inertia is None
               else np.asarray(inertia, dtype=float))
    jac = state.jac[leg]
    q = C.unflat(state.q)[leg]
    qd = C.unflat(state.qd)[leg]
    v = jac @ qd
    jdot_qd = (HK.foot_jacobian(leg, q + qd * _JDOT_EPS) - jac) @ qd
    jdot_qd /= _JDOT_EPS
    a_ff = np.asarray(ref.a, dtype=float) - jdot_qd
    a_fb = (wn * wn * (np.asarray(ref.p, dtype=float) - state.x_b[leg])
            + 2.0 * zeta * wn * (np.asarray(ref.v, dtype=float) - v))
    # One factorisation for both: J^-1 [a_ff  a_fb].
    qdd = np.linalg.solve(jac, np.column_stack((a_ff, a_fb)))
    tau_ff = inertia @ qdd[:, 0]
    tau = tau_ff + inertia @ qdd[:, 1]
    return tau, tau_ff


def swing_control_torque(state, leg: int, ref, kp=None, kd=None,
                         ff: bool = False, inertia=None):
    """``(tau, tau_ff)`` for swinging `leg`; `ref` is a `footstep.SwingRef`
    (anything with p, v, a in the trunk frame).  `tau` includes `tau_ff`.
    `kp` / `kd` are (3,) for every leg or (4, 3) a row per leg
    (`swing.leg_gains`, `--kp-swing-walk-rl` etc., 2026-10-05)."""
    kp = SWING.leg_gains(cfg.KP_SWING_WALK if kp is None else kp, leg)
    kd = SWING.leg_gains(cfg.KD_SWING_WALK if kd is None else kd, leg)
    jac = state.jac[leg]
    qd = C.unflat(state.qd)[leg]
    force = (kp * (np.asarray(ref.p, dtype=float) - state.x_b[leg])
             + kd * (np.asarray(ref.v, dtype=float) - jac @ qd))
    tau = jac.T @ force
    tau_ff = np.zeros(3)
    if ff:
        tau_ff = inertial_feedforward(leg, C.unflat(state.q)[leg], jac,
                                      ref.v, ref.a, inertia)
        tau = tau + tau_ff
    return tau, tau_ff
