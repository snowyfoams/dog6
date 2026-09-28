"""Trot in place, in the fold stance: `hw.trot`'s trot from `hw.fold_stand`'s stand.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_trot --fake --auto 1 --no-imu    the whole path, no robot
    $V -m hw.fold_trot --log fold_trot.npz         on the robot: 0.5 s, 20 mm
                                                   (the log is `hw.stand`'s;
                                                   the filter is not in it)
    $V -m hw.fold_trot --swing joint --period 2.0 --swing-height 40
                                                   the fold trot as it was
                                                   before 2026-09-28

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

TWO HALVES, BOTH IMPORTED, NEITHER COPIED
    THE STAND      `hw.fold_stand`'s: the crouch (`posture.FOLD`), the fixed
                   IMU datum, the SRB-only law, LIMITS OFF (`fold_stand.LIMITS`,
                   since 2026-09-25; with `--limits`, the 45 deg tilt stop and
                   the tracking trip OFF), the roll gains
                   (`fold_stand.ROLL_GAINS`, 290 / 23).
    THE TROT       `hw.trot`'s, the one `hw.trot_esti` flies:
                   `trot.trot_options` -- DOG5's gait (duty 0.80, contact ramp
                   0.15, the 0.2 s four-foot settle every 2 cycles), the 9
                   N*m cap and its ceiling, the 60 N*m/s slew, no overspeed
                   trip -- and the swing at `hw.stand`'s default: the
                   CARTESIAN impedance, `--kd-swing`, `--kp-swing`,
                   `--swing-ff` and `--ff-armature` at the same defaults as
                   there.  `hw.trot`'s docstring says why each piece is what
                   it is.
    ITS OWN CLOCK  0.5 s (`PERIOD_S`, 100 ms of swing) and a 20 mm apex
      AND APEX     (`SWING_HEIGHT`), against `hw.trot`'s 0.8 s and 40 mm.
                   The operator's test, 2026-09-28: `--swing-height 20
                   --period 0.5` was the best fold trot.  `--period 0.8
                   --swing-height 40` is `hw.trot_esti`'s clock and apex.

WHY THE TROT IS `hw.trot`'S, 2026-09-28
    Until then this file flew a trot of its own: the JOINT swing (the z-only
    arc through the IK, abd held, joint PD at `config.KP_SWING_JOINT` 30 /
    `KD_SWING_JOINT` 0.8) on a 2.0 s clock chosen so that PD could follow it
    through the slew.  THE FOLD TROT FAILED ON IT.  The operator's report:
    the swing leg rushed, and the run ended on a CAN missed-reply e-stop that
    the swing leg caused -- not the wiring.  No log was kept.  `hw.trot` had
    already left the joint swing on 2026-09-25, after it failed every run on
    the nominal posture; the trot it kept is the one that trots.

    WHAT THE CARTESIAN SWING WAS LEFT FOR, 2026-09-17.  In the fold stance
    J^T Kp J is only ~3.8 N*m/rad about abd, and the fold stand's swing leg
    went round the abduction axis instead of lifting (swing.py has the
    numbers).  That was the joint swing's whole reason.  If a swing leg goes
    round abd here, that is the thing to look at; `--kd-swing` damps it with
    the rest of the foot.

    `--swing joint --period 2.0 --swing-height 40` is the old fold trot,
    still three flags.

    AND THE FOLD TROT WALKED ON IT, 2026-09-28: rpy held, the trunk
    shifting in x, at 2 mm of apex as at the 20 mm default (the operator's
    report).  The swing leg alone was changed: the Cartesian arc is latched
    at the liftoff foot, as the joint swing's was, and in place lands there
    -- `balance.law.BalanceLaw.update`'s swing block and `balance.swing`
    have the reasoning (a resting site the feet do not stand on is a step
    every swing, and the joint layer moves the trunk after it).  What a
    latched arc cannot do is hold the foot still in the world while the
    trunk moves under it, and this stance's CoM sits behind both diagonals
    (below); a walk that survives the latch is the stance's, and the log's
    `x_b` against `p_swing` says how far each foot moved from its liftoff
    point during the swing.


THE FOLD STANCE AND A DIAGONAL PAIR
    Since 2026-09-25 every leg folds the same way (`posture.FOLD`), knee
    behind its hip -- the front knees tucked under the abd motors, the rear
    ones out behind the rear hips -- and since 2026-09-28 the feet sit
    further forward than that fold put them, the operator's calls: the rear
    20 mm (`posture.FOLD_REAR_FORWARD`; the rear posture was not good and
    the foot could be more to the front), the front 20 mm
    (`posture.FOLD_FRONT_FORWARD`; the front knee sat too close to the abd).
    Feet at +235 / -134 mm x, +-71 mm y.  NOT symmetric front to back: c^b
    x is -6.2 mm, and the CoM sits 20.4 mm behind BOTH diagonal support
    lines, W d = 1.18 N*m about each diagonal that no diagonal pair can
    make.  Its roll part, 1.10, changes sign with the diagonal; its pitch
    part, 0.42 nose-up, is the same for both.  Moving the feet forward is
    what raised it: 14.8 mm and 0.85 N*m before.  The rear-tucked capture of
    2026-09-16 had 17.2 mm, ~0.97 N*m, and the robot fell backward every
    swing.

    THE RESIDUAL TRIP still only counts four-foot sweeps while trotting (see
    `law.BalanceLaw.update`): on two feet a residual is geometry, not a
    contact about to go (`selftest` 12).


THE ESTIMATOR IS `hw.trot_esti`'S KALMAN FILTER, 2026-09-28, on request
    `trot_esti.EstimatorTap`, imported: MIT's linear KF (`hw.state_estimator`)
    fed from the same sweep the law is fed from, in `stand.ESTIMATOR_SLOT`.
    ONLY THE OBSERVATION PRINT CHANGES.  Its two lines -- height beside the
    legs' height, v, the leg-odometry xy, the yaw drift, per-leg trust and
    the innovations -- print under every status line where
    `hw.velocity_estimator`'s accelerometer-only v used to, plus a reading on
    entering every phase and the yaw and xy a trot moved on leaving it.  The
    rest of the stream is `hw.stand`'s as before (NOT `trot_esti`'s terse
    one): weight, moment, torque, |tau|, gap and overrun all still print.
    PRINT ONLY: nothing in the law, the trips or the log reads it.
    `hw.trot_esti`'s docstring says what each field means.

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
from . import trot as TROT           # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .trot_esti import EstimatorTap  # noqa: E402

__all__ = ["main", "TAU_CAP", "PERIOD_S", "SWING_HEIGHT"]

#: `hw.trot`'s, since 2026-09-28 -- see the docstring.
TAU_CAP = TROT.TAU_CAP

#: The fold trot's clock and apex, from the operator's test of 2026-09-28:
#: `--swing-height 20 --period 0.5` was the best.  100 ms of swing.
PERIOD_S = 0.5
SWING_HEIGHT = 0.020                 # m


def main(argv=None) -> int:
    """`hw.fold_stand`'s stand, `hw.trot`'s trot at 0.5 s and 20 mm,
    `hw.trot_esti`'s Kalman filter printed in place of the velocity."""
    return STAND.main(argv, crouch=POSE.FOLD, dynamic_setpoint=False,
                      only_law="srb", tilt_stop=FS.TILT_STOP_DEG,
                      roll_gains=FS.ROLL_GAINS, track_stop=FS.TRACK_STOP_DEG,
                      velocity=False, estimator=EstimatorTap,
                      limits=FS.LIMITS,
                      swing_height=SWING_HEIGHT,
                      **TROT.trot_options(PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
