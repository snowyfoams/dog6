"""Trot in place, KNEES OUT: `hw.fold_trot` with the front legs the rear legs mirrored.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold2_trot --fake --auto 1 --no-imu   the whole path, no robot
    $V -m hw.fold2_trot --log fold2_trot.npz       on the robot: 0.6 s, 20 mm
                                                   (the log is `hw.stand`'s)
    $V -m hw.fold2_trot --swing knee --period 1.2  the SHIN-ONLY swing, toward
                                                   the CoM (swing.py, "THE KNEE
                                                   SWING"); refused at 0.6 s:
                                                   120 ms outruns the slew

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

THE POSTURE, 2026-10-01, THE OPERATOR'S: `posture.FOLD2`
    The front legs are the rear legs MIRRORED fore-aft, so every knee motor
    is OUTBOARD of its hip: the rear knees behind the rear hips as in
    `hw.fold_trot`, the front knees AHEAD of the front hips instead of under
    the trunk.  Every foot 35 mm outboard of its pitch hinge
    (`posture.FOLD2_FOOT_OUT`: `hw.fold_trot`'s `STAND_FOOT_BACK` on the
    rear legs, mirrored onto the front), trunk x +-219.5 mm, y +-65.  At the
    160 mm hold every leg is `hw.fold_trot`'s rear leg: thigh 38 deg, shin
    15 deg from vertical.

    ONE SET OF FEET, SO NO SLANTED RISE.  `hw.fold_trot` slides all four
    sites back through the rise, one rigid shift that carries the trunk 85
    mm forward over planted feet.  Mirrored, the front sites would have to
    slide the other way -- the stance spreading 171 mm on the floor -- so the
    crouch has the stand's feet already and the rise is straight up, tracked
    as `hw.fold_trot`'s (`--rise-track`; HOLD latches the IK at the stand,
    `law.BalanceLaw.hold_reference`).  At the fold's crouch height, 60.5 mm,
    that lays the thigh flat with the knee at 129 deg (the fold's 125), and
    puts every knee housing 66 mm off the floor, where the fold's sit on it.

    WHAT THE MIRROR BUYS.  The CoM is ON both trot diagonals: c^b x is 0.0,
    where `hw.fold_trot`'s stand has it 17.8 mm back and 5.7 mm off both
    (W d = 0.33 N*m no diagonal pair can make).  And each leg's x push as it
    extends is cancelled by its mirror's, as in the mirrored fold of
    2026-09-17..25: nothing walks the trunk in x through the rise.

    MuJoCo, the sequence as flown (knee-motor contact, 0.1 N*m friction, qd
    filter, mu 1.0), against `hw.fold_trot` as built in the same harness:
    the trunk 0.0 mm in x from the crouch to the hold (the fold: +103, its
    slanted shift), pitch 0.00 deg, the legs at 37.7 deg thigh / 16.6 shin,
    every foot within 3.9 mm of its site; the trot in place, -0.1 mm in 10
    s (fold: +1.2 mm in 3), roll rms 0.08 deg, pitch rms 0.01 (fold: 0.22 /
    0.12); peak torque 1.18 N*m in the rise, 0.92 in the trot; the park
    slides the feet 0.5 mm (fold: 17); knee housings 83 mm off the floor in
    the trot.  The same within 0.2 mm at mu 0.5 and at 0 / 0.3 N*m of
    friction, and at `hw.trot_esti`'s 0.8 s / 40 mm.

THE FRONT LEGS BEND THE OTHER WAY FROM `hw.fold_trot`'S
    Left in the fold, front knees back, the crouch ramp swings each front
    leg through straight on its way: in MuJoCo the trunk lifts to ~100 mm
    and pitches up to 19 deg, and arrives.  Fold the front legs knee-forward
    by hand in limp and the crouch is a small move.

WHAT IS `hw.fold_trot`'S, UNCHANGED
    The hold height (`HEIGHT`, 160 mm), the trot's clock and apex
    (`PERIOD_S` 0.6 s, `SWING_HEIGHT` 20 mm -- the operator's 2026-10-02
    test, on the parallel fold), the fixed IMU datum, the SRB-only law,
    limits off, the roll gains, `hw.trot`'s trot (cap, slew, the Cartesian
    swing and its flags), the joint layer and its tracked rise,
    `hw.trot_esti`'s Kalman filter under every status line.  `--height`,
    `--period` and `--swing-height` move them per run as there.

KEYS
    ENTER  the stand's phases, as ever.  REFUSED while trotting.
    T      from HOLD, start trotting.  While trotting, latch the exit: the
           switch back to HOLD waits for all four feet at full weight.
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold2_trot.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import fold_stand as FS       # noqa: E402
from . import stand as STAND         # noqa: E402
from . import trot as TROT           # noqa: E402
from .balance import posture as POSE  # noqa: E402

__all__ = ["main", "stand_options", "TAU_CAP", "PERIOD_S", "SWING_HEIGHT",
           "HEIGHT"]

#: `hw.trot`'s, as `hw.fold_trot`'s.
TAU_CAP = TROT.TAU_CAP

#: `hw.fold_trot`'s clock and apex, carried: the operator's 2026-10-02 test
#: on the parallel fold.  120 ms of swing.  0.5 s until then.
PERIOD_S = 0.6
SWING_HEIGHT = 0.020                 # m

#: m, floor to trunk bottom at the hold: `hw.fold_trot`'s stand height.
HEIGHT = 0.160


def stand_options() -> dict:
    """Everything this trot hands `hw.stand.main`, the estimator and its x/y
    loop included (`trot.trot_options`):
    `hw.fold_trot.stand_options` with the knees-out crouch and no slanted
    rise -- the stand's feet are the crouch's.  A fresh gait each call."""
    return dict(crouch=POSE.FOLD2, dynamic_setpoint=False,
                only_law="srb",
                roll_gains=FS.ROLL_GAINS, track_stop=FS.TRACK_STOP_DEG,
                velocity=False, limits=FS.LIMITS,
                swing_height=SWING_HEIGHT, rise_track=True,
                height=HEIGHT,
                **TROT.trot_options(PERIOD_S))


def main(argv=None) -> int:
    """`hw.fold_trot`, from `posture.FOLD2`: knees out, straight rise."""
    return STAND.main(argv, **stand_options())


if __name__ == "__main__":
    raise SystemExit(main())
