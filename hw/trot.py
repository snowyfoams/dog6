"""Trot in place from the WIDE crouch, on the SRB balance controller.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.trot --fake --auto 1 --no-imu                  the whole path, no robot
    $V -m hw.trot --log trot.npz                            on the robot

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)
                                   hold --W--> step --> hold   (feet under the hips)

`hw.stand` plus a gait.  Everything that is not the trot is `hw.stand`'s, at
`hw.stand`'s defaults: the wide crouch (`posture.WIDE`, the nominal feet
20 mm further out, trunk 40 mm off the floor -- `posture` says why it cannot
stay down), the attitude setpoint LATCHED in limp, and roll
following `--kp-att / --kd-att` (90 / 17) -- except the tilt stop, 45 deg
(`TILT_STOP_DEG`), and the torque-phase tracking trip, OFF
(`TRACK_STOP_DEG`).  Every one is still a flag.

The trot is `hw.fold_trot`'s, from `trot_options` below -- the same swing,
cap, slew and overspeed decision; that file's docstring says why each -- with
a FASTER clock: 0.8 s a cycle (`PERIOD_S`), not 1.2.

    $V -m hw.trot --period 0.6                  faster still
    $V -m hw.trot --settle 0                    no four-foot re-level

WHY THIS STANCE, AFTER THE FOLD ONE
    The fold trot tipped in roll on 2026-09-17.  That fold stance (rear legs
    tucked; `posture.FOLD` has since been refolded rear-as-front) put the CoM
    17.2 mm BEHIND both diagonal support lines, so every swing was ~0.97 N*m of
    moment no diagonal pair can make.  The nominal stance is symmetric:

                            nominal            fold
        feet, trunk x       +237 / -237 mm     +215 / -143 mm
        feet, trunk y       +-65 mm            +-71 / +-69 mm
        CoM off diagonals   0.0 mm             17.2 mm
        I_xx                0.0281             0.0346 kg m^2
        knee, swing peak    7.09 rad/s         7.26 rad/s

    WIDE, 2026-09-17, is the nominal stance at +-85 mm in y: same x, the CoM
    still on both diagonals by symmetry, I_xx 0.0341 kg m^2.

    Both diagonals pass exactly through the pinned CoM, so on two feet the
    allocator is asked for nothing it cannot give.  What is NOT better: I_xx
    is smaller, so the same roll gain is less restoring moment, and the
    diagonal is longer and more nearly along x, so a roll disturbance on two
    feet has the least to push against.



FEET UNDER THE HIPS: W, on request 2026-09-17
    From HOLD, W steps the feet from the crouch's sites to hip frame (0, 0)
    (`STEP_FOOT_XY`) -- one 1.2 s gait cycle (`STEP_PERIOD_S`), each
    diagonal swinging over and
    landing on the new site.  T then trots THERE.  W again steps back, and
    ENTER in HOLD steps back FIRST and then parks: the park is a joint ramp
    to the crouch and would drag feet left anywhere else.  Static, at the
    145 mm hold, four feet:

                            WIDE sites         under the hips
        feet, trunk x/y     +-237 / +-85 mm    +-156 / +-60 mm
        |tau| abd/pitch/knee 0.36/0.75/0.34    0.00/0.41/1.05 N*m
        reach used          0.795              0.707

    The knee carries three times as much, and the support rectangle is 81 mm
    shorter and 25 mm narrower per side.  The SRB model stays WIDE's (c^b z
    7 mm off, x 0 both) -- `law.BalanceLaw.begin_step` says why.


KEYS
    ENTER  the stand's phases, as ever.  REFUSED while trotting or stepping.
    W      from HOLD, step the feet under the hips, or back.
    T      from HOLD, start trotting.  While trotting, latch the exit: the
           switch back to HOLD waits for all four feet at full weight.
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/trot.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import numpy as np                   # noqa: E402

from . import safety as SAFE         # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import posture as POSE  # noqa: E402

__all__ = ["main", "trot_options", "TAU_CAP", "PERIOD_S", "TILT_STOP_DEG",
           "TRACK_STOP_DEG", "STEP_FOOT_XY", "STEP_PERIOD_S"]

#: The operator's cap for the trot, 2026-09-16: the motors' own limit.
TAU_CAP = SAFE.TAU_HARD_NM

#: Faster handovers, on request 2026-09-17: 0.8 s against DOG5's 1.2.
#: Offline, nominal crouch (before WIDE), swing tracked (`--period` overrides):
#:
#:     period   swing   handover ramp   knee peak    handover rate
#:     1.2 s    240 ms  144 ms           7.1 rad/s   13 N*m/s
#:     1.0 s    200 ms  120 ms           8.5 rad/s   19 N*m/s
#:     0.8 s    160 ms   96 ms          10.6 rad/s   29 N*m/s   <- this
#:     0.6 s    120 ms   72 ms          14.1 rad/s   51 N*m/s
#:     0.5 s    100 ms   60 ms          16.9 rad/s   73 N*m/s   past the slew
#:
#: The 60 N*m/s trot slew follows 0.6 s; at 0.5 s it lags every handover.
PERIOD_S = 0.8

#: Raised from `hw.stand`'s 12 deg on request 2026-09-17, to `hw.fold_trot`'s.
#: STILL A TRIP: past it the run stops and the trunk drops.  Run supported.
TILT_STOP_DEG = 45.0

#: The torque-phase tracking trip: OFF, on request 2026-09-17, after it ended
#: a trot on "RL.knee is 26.3 deg from the IK at the commanded height (limit
#: 25)".  It compares the joints against the IK of a LEVEL trunk at the
#: commanded height, and that IK reaches only this trip and the log's `q_ref`
#: -- no torque reads it -- so turning it off changes no N*m.  Same as
#: `hw.fold_trot`.  The position-mode trips (settle 5 deg, crouch and park
#: 15 deg) are `sequence`'s and are untouched.  `--track-stop 25` restores it.
TRACK_STOP_DEG = 0.0

#: Where W steps the feet: straight under each hip, (4, 2) HIP frame.
STEP_FOOT_XY = np.zeros((4, 2))

#: The step's own gait cycle, DOG5's flown 1.2 s -- NOT `PERIOD_S`.  The
#: foot crosses 85 mm in one swing, so the step is the faster leg motion:
#:
#:     period   swing   knee peak (step)   knee peak (in-place, under hip)
#:     0.8 s    160 ms  15.7 rad/s         13.1 rad/s
#:     1.2 s    240 ms  10.5 rad/s         <- this
#:     1.6 s    320 ms   7.9 rad/s
#:
#: `--step-period` overrides.
STEP_PERIOD_S = 1.2


def trot_options(period: float = BCFG.GAIT_PERIOD) -> dict:
    """What turns `hw.stand.main` into a trot, whatever the crouch.

    A fresh gait each call: the clock carries its own start time.
    """
    return dict(gait=GAIT.TrotGait(period=period), tau_cap=TAU_CAP,
                tau_ceiling=TAU_CAP, tau_slew=BCFG.TAU_SLEW_TROT_NM_S,
                overspeed_trip=False)


def main(argv=None) -> int:
    """`hw.stand.main` from `posture.WIDE`, SRB only, plus the trot and W."""
    return STAND.main(argv, crouch=POSE.WIDE, only_law="srb",
                      tilt_stop=TILT_STOP_DEG, track_stop=TRACK_STOP_DEG,
                      step_to=STEP_FOOT_XY, step_period=STEP_PERIOD_S,
                      **trot_options(PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
