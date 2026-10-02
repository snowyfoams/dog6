"""Trot in place, in the fold stance: `hw.trot`'s trot from `hw.fold_stand`'s stand.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_trot --fake --auto 1 --no-imu    the whole path, no robot
    $V -m hw.fold_trot --log fold_trot.npz         on the robot: 0.6 s, 20 mm
                                                   (the log is `hw.stand`'s;
                                                   the filter is not in it)
    $V -m hw.fold_trot --swing joint --period 2.0 --swing-height 40
                                                   the fold trot as it was
                                                   before 2026-09-28

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

THE STAND IS NOT THE CROUCH'S FEET, 2026-10-01: THE SLANTED RISE
    The hold is at `HEIGHT`, 160 mm, with the feet `STAND_FOOT_BACK`, 35 mm,
    BEHIND each pitch hinge (`STAND_XY`; the crouch's are 48 mm ahead of
    it): thigh 37 deg from vertical, shin 15, knee 90 deg -- the operator's
    ask after the tracked rise, "the big leg is so straight" at the 145 mm
    hold (thigh 17, shin 49), a lower hold refused because the hip comes
    toward the floor.  The crouch keeps its own feet: with the stand's feet
    the front knee motor folds up INTO the trunk at every crouch height
    below ~150 mm (the trunk's underside is 35 mm below the hip axis, the
    knee housing 32 mm in radius), so the feet differ between the two.

    HOW THE FEET GET THERE WITHOUT LEAVING THE FLOOR.  The IK reference the
    tracked layer follows slides its sites from the crouch's to the stand's
    as the commanded height passes `SLIDE_FROM_H`, 110 mm, to `HEIGHT` --
    and the trunk moves 85 mm forward over four planted feet.  Not before
    110 mm: the knee must stay under the trunk's underside the whole way
    (`law.BalanceLaw.stand_xy` has the numbers).  HOLD then latches the
    law's reference at the stand, not the measured pose, so the layer and
    the trot's resting sites are one stance by construction
    (`law.BalanceLaw.hold_reference`).  ENTER parks from there: the
    position ramp to the crouch takes the trunk back over the feet.

    MuJoCo, the sequence as flown, knee-motor contact, 0.1 N*m friction, qd
    filter, bus latency: the stand within 0.4 mm of its sites, pitch 0.0,
    front knee 7 mm clear of the trunk standing and 6 in the trot, the trot
    in place (+1 mm in 3 s against -41 for the 145 mm stand on the crouch's
    feet), the park sliding the feet 18 mm.  A foot STEP at height was the
    first idea and failed in the same sim: a diagonal-pair step tips the
    trunk about the supporting diagonal (this stance's CoM sits 19 mm off
    both) and the rear swing foot never leaves the floor; one foot at a
    time has a -6 mm three-foot margin on its second swing.  The SRB model
    is pinned at the STAND's feet (`posture.CrouchPose.srb_at`).  `--height
    145` with `--no-rise-track` is the stand as it was.

THE RISE IS TRACKED, 2026-10-01
    `--rise-track` is this script's default (`law.BalanceLaw.rise_track`):
    through the rise the joint-space layer -- the one every trot flies from
    HOLD on, Kp 5 / Kd 0.2 -- holds the legs on the IK reference at the
    commanded height, the feet at this posture's sites, until HOLD latches
    the measured pose as before.  `--no-rise-track` is the rise as it was.

    THE OPERATOR'S REPORT, 2026-10-01: the crouch is good, but after the rise
    the trunk sits forward, the front legs are compressed and the rear legs
    are no longer parallel to the front -- the control, with this posture.
    MuJoCo with this law from the fold crouch, no joint friction, no model
    error, agrees: the trunk walks +95 mm forward over the 3 s rise and the
    legs end at +43/-54 deg from vertical against the designed +-16, because
    nothing holds trunk x over the feet (`config.KP_XY`) and in the PARALLEL
    fold all four legs push the same way as they extend.  The mirrored fold
    (rear knees toward the CoM, 2026-09-17..25) and NOMINAL: 0.3 mm under the
    same law.  Tracked, under knee contact, the cap ramp, friction, the qd
    filter and bus latency: +10 mm, legs +17.7/-17.6, at most 1.1 N*m from
    the layer.  The stance xy spring (`--kp-stance-xy 400 --kd-stance-xy
    25`) also held in that sim and SHOOK on the robot the same day; why is
    not known, and it is not this script's default.

    THE KNEE MOTORS WERE ON THE FLOOR IN THIS CROUCH, UNTIL 2026-10-02.
    With the feet on the floor and the trunk level at h 60, the MG5010 knee
    housings (31.7 mm about the knee axis) reached 20 mm BELOW the floor
    plane, so the robot sat on its knee motors with the feet ~21 mm up --
    the operator saw the feet off the floor.  Now every knee opens 15 deg
    (`posture.FOLD_KNEE_OPEN`, the thigh where it was): the shin swings
    toward the floor, the crouch is h 87.8 with the housings 7.2 mm clear.
    MuJoCo, this sequence with knee-motor contact: crouch feet 9.6 / 19.3 N
    front / rear (0 before, on the knees), the handover lunge +4 mm (+13
    before) -- the trunk still dips ~7 mm onto the rear knee motors for
    ~0.2 s while the torque cap ramps up, then rises on its feet -- the
    stand and the trot as before.

TWO HALVES, BOTH IMPORTED, NEITHER COPIED
    THE STAND      `hw.fold_stand`'s: the crouch (`posture.FOLD`), the fixed
                   IMU datum, the SRB-only law, LIMITS OFF (`fold_stand.LIMITS`,
                   since 2026-09-25; with `--limits`, the 45 deg tilt stop and
                   the tracking trip OFF), the roll gains
                   (`fold_stand.ROLL_GAINS`, 290 / 23).
    THE TROT       `hw.trot`'s, the one `hw.trot_esti` flies:
                   `trot.trot_options` -- DOG5's gait (duty 0.80, contact ramp
                   0.15, the 0.2 s four-foot settle every 2 cycles), the 9
                   N*m cap and its ceiling, the 120 N*m/s slew, no overspeed
                   trip -- and the swing at `hw.stand`'s default: the
                   CARTESIAN impedance, `--kd-swing`, `--kp-swing`,
                   `--swing-ff` and `--ff-armature` at the same defaults as
                   there.  `hw.trot`'s docstring says why each piece is what
                   it is.
    THE CLOCK      0.6 s (`PERIOD_S`, 120 ms of swing) and a 20 mm apex
      AND APEX     (`SWING_HEIGHT`), with the swing gains and slew below.
                   The operator's test, 2026-10-02: `--period 0.6
                   --swing-height 20 --tau-slew 120 --kp-swing 10 10 400
                   --kd-swing 5 5 40` was the best fold trot, flown with no
                   joint-layer damper on the swinging leg -- "now the swing
                   leg height is good".  Every trot's default since, on
                   request (`config.KP_SWING`, `trot.PERIOD_S`).  Before:
                   0.5 s / 20 mm here (2026-09-28), DOG5's 140/140/180 and
                   8/8/15 swing gains, 60 N*m/s.

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
    PRINT ONLY until 2026-10-02.  `hw.trot_esti`'s docstring says what each
    field means.

    AND SINCE 2026-10-02 IT CLOSES THE HOLD'S AND THE TROT'S x/y, ON
    REQUEST -- every trot entry point does (`trot.trot_options`, `--est-xy`
    by default): `trot_esti.EstimatorFeed`,
    what `hw.fully_trot` flew first, its docstring has the account.  One more
    status line, `xy loop`.  `--no-est-xy` is this file as it was.

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

from sim import params as P          # noqa: E402

from . import fold_stand as FS       # noqa: E402
from . import stand as STAND         # noqa: E402
from . import trot as TROT           # noqa: E402
from .balance import posture as POSE  # noqa: E402

__all__ = ["main", "stand_options", "TAU_CAP", "PERIOD_S", "SWING_HEIGHT",
           "HEIGHT", "STAND_FOOT_BACK", "STAND_XY", "SLIDE_FROM_H"]

#: `hw.trot`'s, since 2026-09-28 -- see the docstring.
TAU_CAP = TROT.TAU_CAP

#: The fold trot's clock and apex, from the operator's test of 2026-10-02
#: (the docstring has the whole run): 0.6 s, 120 ms of swing, 20 mm.  0.5 s
#: from 2026-09-28 until then.
PERIOD_S = 0.6
SWING_HEIGHT = 0.020                 # m

#: THE STAND, 2026-10-01 -- see the module docstring.  m, floor to trunk
#: bottom (`--height`'s default here; `config.H_LIFT` is 145).
HEIGHT = 0.160
#: m, how far BEHIND its pitch hinge each foot stands at the hold (trunk x).
#: 30-40 mm is the window: less leaves the thigh straight, more brings the
#: front knee motor within a few mm of the trunk's underside (at 50 mm,
#: 2 mm in the trot's swings; at 66 mm, the vertical shin, 2 mm INTO it).
STAND_FOOT_BACK = 0.035
#: m, the commanded height the sites start sliding at.  Below ~110 the
#: front knee would pass through the trunk's underside on the way.
SLIDE_FROM_H = 0.110
#: (4, 2) HIP-frame sites of the stand: the fold's y, x = the pitch hinge
#: minus `STAND_FOOT_BACK`.  Trunk x +149.5 / -219.5 mm.
STAND_XY = POSE.FOLD.foot_xy.copy()
STAND_XY[:, 0] = P.HIP_TO_PITCH[:, 0] - STAND_FOOT_BACK


def stand_options() -> dict:
    """Everything this trot hands `hw.stand.main`: the fold stand, the
    slanted rise and `hw.trot`'s trot -- the estimator and its x/y loop with
    it -- on this file's clock and apex.  `hw.fully_trot` flies exactly
    these.  A fresh gait each call."""
    return dict(crouch=POSE.FOLD, dynamic_setpoint=False,
                only_law="srb", tilt_stop=FS.TILT_STOP_DEG,
                roll_gains=FS.ROLL_GAINS, track_stop=FS.TRACK_STOP_DEG,
                velocity=False, limits=FS.LIMITS,
                swing_height=SWING_HEIGHT, rise_track=True,
                height=HEIGHT, stand_xy=STAND_XY,
                slide_from=SLIDE_FROM_H,
                **TROT.trot_options(PERIOD_S))


def main(argv=None) -> int:
    """`hw.fold_stand`'s stand, `hw.trot`'s trot at 0.6 s and 20 mm,
    `hw.trot_esti`'s Kalman filter in place of the velocity, closing the
    trot's x/y unless `--no-est-xy`."""
    return STAND.main(argv, **stand_options())


if __name__ == "__main__":
    raise SystemExit(main())
