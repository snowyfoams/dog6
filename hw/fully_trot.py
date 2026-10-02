"""Trot in place with the state estimator IN THE LOOP: `hw.fold_trot`, closed.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fully_trot --fake --auto 1 --no-imu    the whole path, no robot
    $V -m hw.fully_trot --log fully_trot.npz        on the robot
    $V -m hw.fully_trot --kp-xy 0 --kd-xy 0         the A/B: the filter fed and
                                                    logged, the x/y rows zero
    $V -m hw.fully_trot --no-est-xy                 the filter printed, not fed

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

SINCE 2026-10-02 EVERY TROT DOES THIS, ON REQUEST.  `trot.trot_options`
brings `EstimatorFeed` (now in `hw.trot_esti`) and `--est-xy`, on by default,
to `hw.trot`, `hw.trot_esti`, `hw.fold_trot`, `hw.fold2_trot` and
`hw.wide_trot`; `--no-est-xy` is each of them as it was.  So this file IS
`hw.fold_trot` now.  It stays for its name and for the account below, which
is the loop's for every one of them -- the MuJoCo numbers are the fold's.

EVERYTHING IS `hw.fold_trot`'S BUT ONE CHANNEL, IN ONE PHASE
    The stand, the slanted rise to 160 mm, the trot's clock and apex (0.6 s,
    20 mm), the swing, the cap, the slew, the joint layer, the keys and the
    flags: `fold_trot.stand_options()`, imported, not copied.  What changes:

    IN THE TROT, THE CoM's WORLD x/y AND THEIR RATE COME FROM THE KALMAN
    FILTER and the SRB wrench's x and y rows close on them -- the rows every
    other entry point leaves at zero because, in `config.KP_XY`'s words,
    nothing measured them.  `hw.state_estimator` does: leg odometry and the
    accelerometer, the filter `hw.trot_esti` and `hw.fold_trot` have been
    printing since 2026-09-24 / 09-28.  The setpoint is the CoM x/y LATCHED on
    the trot's first sweep, the rate's is zero: station keeping.  Gains
    `config.KP_XY_EST` / `KD_XY_EST`, an authority clamp `XY_ACC_MAX`
    (`law.BalanceLaw.est_xy` has the whole account).

    WHAT STAYS WHERE IT WAS.  Height is the legs' (`state.on_stance`) and
    attitude the IMU's: both are measured, and swapping a measured signal for
    a filtered one needs a reason -- DOG5's rule for its EKF, `ekf_feedback`'s
    header.  The rise and the W step never read the filter; they print it.
    THE HOLD DOES SINCE 2026-10-02, ON REQUEST: the same rows, its own latch
    on the first sweep of each HOLD (`law.BalanceLaw.est_xy`).

WHY THESE TWO ROWS AND NOT THE OTHERS -- DOG5'S RECORD
    DOG5 put a filter's velocity in its fast loop once, and `ekf_active.npz`
    records 224 mm/s of divergence -- its Bloesch EKF, on a 100 Hz thread,
    feeding the MPC's whole velocity state, 591 mm/s wrong while flagged
    healthy.  After that DOG5's "active" fed ONE slow channel (the
    placement EMA) and never the wrench.  This keeps that shape: one slow
    channel (0.5 Hz, see `config.KP_XY_EST`) that nothing else measures,
    clamped, falling back sweep by sweep.  It does not keep DOG5's placement:
    no foot placement is the operator's 2026-09-16 decision for every DOG6
    trot, and the x/y rows move the trunk over the feet instead -- the swing
    then lands each foot where the trunk now is (the trunk frame), so the
    stance walks back under a trunk that was pushed home.

ONE SWEEP LATE, AND WHY THAT IS ALLOWED HERE
    The filter costs 290 us of a 333 us slot and slot 0 is already the law's
    (`hw.stand.ESTIMATOR_SLOT`), so it still runs in its own slot on the
    sweep's latched measurement, and the law reads the answer at the NEXT
    slot 0: 4 ms old.  `adapters.run_once` warns that this is a period of
    delay on every path that feeds on it; on THIS path it is 0.7 deg of phase
    at the loop's 0.48 Hz.  The law ages every estimate and refuses one past
    `config.EST_MAX_AGE_S` (20 ms) or made on a 0x40 packet past
    `EST_ACC_MAX_AGE_S` (50 ms), in another world frame, not finite, or on a
    sweep whose IMU attitude is stale (the R its x/y are rotated by) -- that
    sweep's x/y rows are zero, the flown law -- and counts it.  A filter
    that throws disables itself (`hw.trot_esti`'s rule) and the trot goes on
    as `hw.fold_trot`'s.  What slot 0 itself pays for the x/y rows, on the Pi:
    `law.update` in the trot p50 649 us against 629 without them, 20 us.

THE FILTER IS THE TAP'S, MIT'S TRUST WINDOW INCLUDED -- MEASURED
    `hw.trot_esti` left one decision for the day the filter entered the loop:
    MIT's `trust_window` (0.20 of stance) is wider than DOG5's flown contact
    ramp (0.15), so at the four-foot window the filter believes every leg
    0.9375 while the allocator has all four at full load.  Matching the
    window to the ramp was tried in MuJoCo and made the estimate WORSE: 10 s
    of this trot, filter against the true trunk, 5.3 mm off at 0.15 against
    2.0 at 0.20, and 15.3 against 9.2 under a 3 N push.  A foot rolls and
    slides most while it is lightly loaded, at the two ends of its stance,
    and the wider window believes it least there.  So MIT's stays.

WHAT MuJoCo SAYS IT IS WORTH, 2026-10-01 -- READ THIS BEFORE THE RUN
    The sequence as flown (knee-motor contact, 0.1 N*m joint friction, the
    40 Hz qd filter), the filter fed one sweep late from the model's own
    accelerometer and gyro, 10 s of trot, elliptic friction cone (impratio
    10; the default cone creeps and gives the same picture):

                                     trunk moved in 10 s     the filter saw
        as built, loop off            +0.5 mm                 -1.6 mm
        as built, loop on             +0.9                    -1.2
        3 N push on the trunk, off   +20.6                   +11.4
        3 N push, on                 +18.1                    +9.3
        145 mm on the crouch's feet,
          the stance that walked, off -20.5                  +49.7
          on                          -28.6                  +77.1

    The loop does what it is built to do -- it holds the FILTER's x/y, and
    9/6, 25/10, 50/14 and Kd alone were all stable, roll and pitch rms
    unchanged -- and the trunk's true drift differs from the filter's by
    however far the PLANTED FEET SLIDE.  Under the push the rear pair slid
    12-13 mm in 5 s of it and the loop took back 2.5 mm of 20.6.  In the 145
    mm stance the front pair slid 76-88 mm in 6 s of trot, the filter read the
    walk with the WRONG SIGN, and the loop made it 40 % worse.  As built,
    nothing drifts and the loop sits at 0.2 N.  Whether the robot's feet
    slide like the model's is not something the model can say.

WHAT TO READ
    status, under every line   the filter, as in `hw.fold_trot`, plus
                               `xy loop`: ON with the CoM error from the
                               latch and the acceleration asked, or why not
    leaving a trot             `trot yaw ... xy moved`, the filter's xy over
                               the trot -- the number this loop exists to
                               keep small.  Read the yaw first: the filter's
                               xy is in the run's world, a turned trunk
                               carries the stance with it
    exit report                sweeps driven / refused, peak error, peak
                               acceleration
    the log (--log)            `hw.fold_trot`'s columns plus p_est, v_est,
                               est_age, xy_des, xy_on; b_d[0:2] is the x/y
                               force the loop asked for

    THE LEGS ARE ITS ONLY RULER.  The filter's x/y is leg odometry, so a
    foot that SLIPS moves the robot without moving the estimate, and the
    loop holds the robot where the legs say it is.  A trot that holds its
    `xy moved` near zero and still wanders across the floor is slipping;
    a tape on the floor is the check the filter cannot make.

KEYS
    `hw.fold_trot`'s.  ENTER the phases (refused while trotting); T from HOLD
    starts the trot and latches the x/y it will hold, T again latches the
    exit; X E-STOP -- from rise, hold or trot it DROPS the robot.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fully_trot.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import fold_trot as FT        # noqa: E402
from . import stand as STAND         # noqa: E402
#: Here since 2026-10-01; in `hw.trot_esti` since 2026-10-02, where every
#: trot gets it from (`trot.trot_options`).  Re-exported for old callers.
from .trot_esti import EstimatorFeed  # noqa: E402

__all__ = ["EstimatorFeed", "main"]


def main(argv=None) -> int:
    """`hw.fold_trot`, with the filter fed to the trot's x/y rows -- which
    since 2026-10-02 is `hw.fold_trot` itself."""
    return STAND.main(argv, **FT.stand_options())


if __name__ == "__main__":
    raise SystemExit(main())
