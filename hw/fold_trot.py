"""Trot in place, in the fold stance, on the SRB balance controller.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_trot --fake --auto 1 --no-imu    the whole path, no robot
    $V -m hw.fold_trot --log fold_trot.npz         on the robot

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

`hw.fold_stand` plus a gait -- `hw.trot.trot_options`, shared with the
nominal-crouch trot `hw.trot`.  Same crouch (`posture.FOLD`), same fixed IMU
datum, same SRB-only law, same 45 deg tilt stop, same tracking trip OFF, same
roll gains (`fold_stand.ROLL_GAINS`, 290 / 23) -- all imported from it, none
copied.  What is added is T, and the four things T needs:

    THE GAIT       `balance.gait.TrotGait`: DOG5 trot_demo's clock -- duty
                   0.80, contact ramp 0.15, a 0.2 s four-foot settle every 2
                   cycles, alternating lead [DOG5 FLOWN] -- on the fold's OWN
                   period, 2.0 s (`PERIOD_S`), not `hw.trot`'s 0.8: a 400 ms
                   swing.  The joint swing below cannot follow a shorter one
                   through the 60 N*m/s slew.  `--period` overrides.
    THE SWING      `balance.swing`: cMPC's quintic arc, 40 mm straight up in
                   the trunk frame and back down on the same spot.  NO FOOT
                   PLACEMENT -- the operator's decision, 2026-09-16.  THE
                   FOLD'S OWN SWING, NOT `hw.trot`'s (2026-09-17): the arc
                   through the IK, ABD HELD, joint PD at
                   `config.KP_SWING_JOINT` 30 / `KD_SWING_JOINT` 0.8.  The
                   shared Cartesian PD was ~4 N*m/rad about abd in this
                   stance, and the leg swung round it.
                   MUJOCO, 2026-09-17 (one leg, trunk pinned in the air, the
                   4 ms sweep, the 60 N*m/s slew and the 9 N*m cap): the
                   swing's torque demand outruns the slew below ~0.30 s of
                   swing, and the PD then winds up -- 160, 240 and 260 ms all
                   diverge, the knee overshooting by ~90 deg.  At 300 ms the
                   worst z error is 2.9 mm; at 400 ms 1.9 mm, apex 40.6, abd
                   0.1 deg (0.5 deg under a 0.5 N*m, 20 ms abd push), peak
                   1.1 N*m and 5.5 rad/s.  2.0 s keeps the swing 1/3 clear.
    THE CAP        `--tau-cap` defaults to 9.0 N*m, `safety.TAU_HARD_NM`, and
                   the gate's ceiling is raised to match.  The operator's
                   decision, 2026-09-16.  Offline, a diagonal pair in the old
                   rear-tucked fold needed 3.3 N*m on the rear knee against
                   3.0 staged.
    THE SLEW       60 N*m/s, `config.TAU_SLEW_TROT_NM_S`, what DOG5 trotted
                   on.  The stand's 5 would take 0.35 s to follow one handover.
    NO OVERSPEED   `safety.SafetyGate(overspeed_trip=False)`, DOG5's trot
      TRIP         runner exactly (`trot_hw.TorqueGate.overspeed_reason`):
                   speeds are measured and their peaks printed at exit, and
                   nothing stops on them.  The first hardware trot, 2026-09-16,
                   ended on "confirmed overspeed RR.knee: driver +7.5, encoder
                   +7.4 rad/s" -- and in the fold stance the 40 mm arc ITSELF
                   asks the knee for 7.2-7.3 rad/s at mid-swing (the IK along
                   `swing.swing_reference`), against a 7.0 trip written for a
                   stand.  The operator's call: the trip is wrong for a trot.
                   What still bounds a joint: the swing PD's damping, the
                   60 N*m/s slew, the cap, and the tilt stop.


THE FOLD STANCE AND A DIAGONAL PAIR
    Since 2026-09-17 the rear legs fold like the front ones (`posture.FOLD`):
    knee motor toward the CoM, feet at +-215 mm x, +-71 mm y.  The stance is
    symmetric front to back, c^b x = 0.0, and both diagonal support lines
    pass through the CoM -- the old rear-tucked fold put it 17.2 mm behind
    them, ~0.97 N*m no diagonal pair could make, and the robot fell backward
    every swing.

    THE RESIDUAL TRIP still only counts four-foot sweeps while trotting (see
    `law.BalanceLaw.update`): on two feet a residual is geometry, not a
    contact about to go.


KEYS
    ENTER  the stand's phases, as ever.  REFUSED while trotting.
    T      from HOLD, start trotting.  While trotting, latch the exit: the
           switch back to HOLD waits for all four feet at full weight.
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold_trot.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import fold_stand as FS       # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .trot import TAU_CAP, trot_options   # noqa: E402

__all__ = ["main", "TAU_CAP", "PERIOD_S"]

#: The fold trot's gait period.  400 ms of swing at duty 0.80; the joint
#: swing diverges under ~300 (see THE GAIT / THE SWING above).
PERIOD_S = 2.0


def main(argv=None) -> int:
    """`hw.fold_stand.main`, plus a gait, the 9 N*m cap and DOG5's slew."""
    return STAND.main(argv, crouch=POSE.FOLD, dynamic_setpoint=False,
                      only_law="srb", tilt_stop=FS.TILT_STOP_DEG,
                      roll_gains=FS.ROLL_GAINS, track_stop=FS.TRACK_STOP_DEG,
                      swing="joint", **trot_options(PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
