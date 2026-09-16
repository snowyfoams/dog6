"""Trot in place, in the fold stance, on the SRB balance controller.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_trot --fake --auto 1 --no-imu    the whole path, no robot
    $V -m hw.fold_trot --log fold_trot.npz         on the robot

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

`hw.fold_stand` plus a gait.  Same crouch (`posture.FOLD`), same fixed IMU
datum, same SRB-only law, same 45 deg tilt stop, same tracking trip OFF, same
roll gains (`fold_stand.ROLL_GAINS`, 290 / 23) -- all imported from it, none
copied.  What is added is T, and the four things T needs:

    THE GAIT       `balance.gait.TrotGait`: DOG5 trot_demo's clock -- 1.2 s
                   period, duty 0.80, contact ramp 0.15, a 0.2 s four-foot
                   settle every 2 cycles, alternating lead.  [DOG5 FLOWN]
    THE SWING      `balance.swing`: cMPC's quintic arc, 40 mm straight up in
                   the trunk frame and back down on the same spot.  NO FOOT
                   PLACEMENT -- the operator's decision, 2026-09-16.  Cartesian
                   PD at DOG5's gains, no Lambda feedforward (2.3 ms a leg on
                   this Pi).
    THE CAP        `--tau-cap` defaults to 9.0 N*m, `safety.TAU_HARD_NM`, and
                   the gate's ceiling is raised to match.  The operator's
                   decision, 2026-09-16.  Offline, a diagonal pair in the fold
                   stance needs 3.3 N*m on the rear knee against 3.0 staged.
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


WHAT THE FOLD STANCE DOES TO A DIAGONAL PAIR -- READ BEFORE THE FIRST RUN
    The CoM (c^b x = -13.8 mm) sits 17.2 mm BEHIND both diagonal support
    lines: they cross the midline 33 mm ahead of the trunk origin.  Two feet
    on a line make no moment about that line, so for every 240 ms swing the
    allocator is ~0.97 N*m short, and the robot falls backward about the
    support diagonal -- the same direction for BOTH diagonals, so it does not
    cancel out.  Nothing in this entry point moves the feet or the trunk to
    fix it.  What holds it is the DOG5 demo's answer: the four-foot windows
    and the periodic settle re-level the trunk before the next swing.

    Watch pitch on the trot line.  A nose-up that grows cycle by cycle is
    this, not a gain.

    BECAUSE OF THAT, THE RESIDUAL TRIP only counts four-foot sweeps while
    trotting (see `law.BalanceLaw.update`).  On two feet the residual is the
    geometry above, and it would trip every swing.


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
from . import safety as SAFE         # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import posture as POSE  # noqa: E402

__all__ = ["main", "TAU_CAP"]

#: The operator's cap for the trot, 2026-09-16: the motors' own limit.
TAU_CAP = SAFE.TAU_HARD_NM


def main(argv=None) -> int:
    """`hw.fold_stand.main`, plus a gait, the 9 N*m cap and DOG5's slew."""
    return STAND.main(argv, crouch=POSE.FOLD, dynamic_setpoint=False,
                      only_law="srb", tilt_stop=FS.TILT_STOP_DEG,
                      roll_gains=FS.ROLL_GAINS, track_stop=FS.TRACK_STOP_DEG,
                      gait=GAIT.TrotGait(), tau_cap=TAU_CAP,
                      tau_ceiling=TAU_CAP,
                      tau_slew=BCFG.TAU_SLEW_TROT_NM_S,
                      overspeed_trip=False)


if __name__ == "__main__":
    raise SystemExit(main())
