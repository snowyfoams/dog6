"""The swing leg the robot flies, under the MPC's trot.  PORTED, not invented.

    swing_torque(foot_b, jac, qd, p_ref, v_ref, kp, kd)  -> (3,)

    tau = J^T [ Kp (p_ref - x) + Kd (v_ref - J qd) ]          TRUNK frame

WHERE IT COMES FROM (2026-11-10, on request: "the swing leg as the balance
controller already runs it on the hardware")
    `cloud` / `swing-analysis`: hw/balance/swing_control.swing_control_torque,
    the impedance branch with `ff=False`, and hw/balance/swing.leg_gains --
    the walk's swing law since 2026-10-20 ("the operator's test on the robot
    that day says it must be", hw/balance/config.WALK_SWING_LAW), as
    `hw.fold_walk` / `hw.fold2_walk` fly it by default:

      the law      the Cartesian impedance above, in the trunk frame, on the
                   closed-form x_b and J (`hw.kinematics`, the same arrays the
                   balance law reads: `hw.cmpc.state` hands them in as
                   `BodyState.feet_body` / `jacobians_body`)
      the gains    KP_SWING_WALK 150 / 150 / 400 N/m, KD_SWING_WALK 6 / 6 / 40
                   N s/m (x, y, z), a leg's own row by `--kp-swing-walk-rl`
                   and the rest, exactly as there
      no feedforward
                   `--swing-ff` is OFF by default on the robot: no M0, no
                   Lambda, no Coriolis, no a_ref
      no leg weight in it
                   the robot gives every leg its own weight through ONE term
                   (`torque.stance_torque` with zero force on a swinging leg);
                   here that term is `run.HardwareCmpc`'s closed-form leg
                   gravity, added to all four legs, so a swinging leg carries
                   its weight once
      no joint damper on the swinging leg
                   `law.BalanceLaw`, 2026-10-01; sim.cmpc's joint floor is on
                   stance legs only already
      the apex     SWING_HEIGHT 20 mm, the operator's best trot, 2026-10-02

WHAT IT REPLACES
    `sim.cmpc.swing.swing_torque`, eq (1)-(3): Kp 500 / Kd 20 N/m isotropic,
    J' Lambda (a - Jdot qd) + C qd + G through `sim.leg_dynamics`' RNEA.  On
    the Pi that is ~2.5 ms a swing leg (fake bus, 2026-11-10: the trot's
    no-solve sweep 5.5 ms inside a 4 ms sweep), and its G on top of the run's
    leg gravity carried a swinging leg's weight twice.

WHAT IS STILL sim.cmpc's
    The swing REFERENCE: the foothold (eq 33 about the hip, re-planned every
    sweep from the estimate's velocity) and the world-frame quintic from the
    liftoff point (`sim.cmpc.swing.SwingTrajectory`, at this apex), turned into
    the trunk frame by `to_body` without the -omega x p term.  The robot's walk
    plans its own (hw/balance/footstep.py: about the stance's sites, z in the
    trunk frame, and at zero command a trunk-frame arc with no velocity term).
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/cmpc/swing.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.cmpc"

__all__ = ["KP_SWING_WALK", "KD_SWING_WALK", "SWING_HEIGHT", "leg_gains",
           "swing_torque", "describe"]

#: hw/balance/config.KP_SWING_WALK / KD_SWING_WALK on `swing-analysis`, the
#: walk's impedance, trunk frame, (x, y, z), N/m and N s/m.  The trot in
#: place's 10 N/m in x/y is too soft to carry a foot across a stride; z is the
#: operator's 400 / 40.  The robot's walk default.
KP_SWING_WALK = np.array([150.0, 150.0, 400.0])
KD_SWING_WALK = np.array([6.0, 6.0, 40.0])

#: hw/balance/config.SWING_HEIGHT: the apex above the higher end of the arc,
#: 20 mm since 2026-10-02, the operator's best fold trot.  sim.cmpc's is 30.
SWING_HEIGHT = 0.020


def leg_gains(gains, leg: int) -> np.ndarray:
    """(3,) `leg`'s x y z gains out of `gains`: (3,) is every leg's, (4, 3) a
    row per leg, FL FR RL RR.  hw/balance/swing.leg_gains."""
    gains = np.asarray(gains, dtype=float)
    return gains[leg] if gains.ndim == 2 else gains


def swing_torque(foot_b, jac, qd, p_ref, v_ref, kp, kd) -> np.ndarray:
    """(3,) joint torques for one swinging leg.  `foot_b` (3,) and `jac` (3, 3)
    are the leg's measured foot and Jacobian in the trunk frame, `qd` (3,) its
    joint rates, `p_ref` / `v_ref` the arc's in the trunk frame, `kp` / `kd`
    (3,).  hw/balance/swing_control.swing_control_torque with ff=False."""
    jac = np.asarray(jac, dtype=float)
    force = (np.asarray(kp, dtype=float)
             * (np.asarray(p_ref, dtype=float) - np.asarray(foot_b, dtype=float))
             + np.asarray(kd, dtype=float)
             * (np.asarray(v_ref, dtype=float) - jac @ np.asarray(qd, dtype=float)))
    return jac.T @ force


def describe() -> str:
    return ("swing leg (the robot's, hw/balance/swing_control impedance): "
            "Kp %s N/m, Kd %s N s/m, no feedforward, apex %.0f mm"
            % (KP_SWING_WALK.tolist(), KD_SWING_WALK.tolist(),
               1e3 * SWING_HEIGHT))


if __name__ == "__main__":
    print(describe())
