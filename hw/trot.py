"""Trot in place from the normal crouch, on the SRB balance controller.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.trot --fake --auto 1 --no-imu                  the whole path, no robot
    $V -m hw.trot --log trot.npz                            on the robot

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)
                                   hold --W--> step --> hold   (feet under the hips)

`hw.stand` plus a gait.  Everything that is not the trot is `hw.stand`'s, at
`hw.stand`'s defaults: the normal crouch (`posture.NOMINAL`, `CROUCH`; back
from WIDE on request 2026-09-21 -- no abduction splay), the attitude
setpoint LATCHED in limp, the HEADING latched at the end of the crouch
(below), and roll
following `--kp-att / --kd-att` (90 / 17) -- except the tilt stop, 45 deg
(`TILT_STOP_DEG`), and the torque-phase tracking trip, OFF
(`TRACK_STOP_DEG`).  Every one is still a flag.

The trot is `hw.fold_trot`'s, from `trot_options` below -- the same swing,
cap, slew and overspeed decision; that file's docstring says why each -- with
a FASTER clock: 0.8 s a cycle (`PERIOD_S`), not 1.2.

    $V -m hw.trot --period 0.6                  faster still
    $V -m hw.trot --settle 0                    no four-foot re-level
    $V -m hw.trot --kp-joint 3 --kd-joint 0.1   the joint layer (these are
                                                the defaults)
    $V -m hw.trot --no-joint-hold               the SRB law alone: body free
                                                to shift, height and rpy
                                                held -- the sway demos
    $V -m hw.trot --half-gait                   the handover by HAND: each
                                                T is half a gait cycle, the
                                                diagonals alternating

WHY THIS STANCE, AFTER THE FOLD ONE
    The fold trot tipped in roll on 2026-09-17.  That fold stance (rear legs
    tucked; `posture.FOLD` was refolded rear-as-front that day, and parallel
    again on 2026-09-25 -- 14.8 mm, see `hw.fold_trot`) put the CoM
    17.2 mm BEHIND both diagonal support lines, so every swing was ~0.97 N*m of
    moment no diagonal pair can make.  The nominal stance is symmetric:

                            nominal            fold
        feet, trunk x       +237 / -237 mm     +215 / -143 mm
        feet, trunk y       +-65 mm            +-71 / +-69 mm
        CoM off diagonals   0.0 mm             17.2 mm
        I_xx                0.0281             0.0346 kg m^2
        knee, swing peak    7.09 rad/s         7.26 rad/s

    WIDE, 2026-09-17, was the nominal stance at +-85 mm in y (abd splayed
    out); the trot went back to NOMINAL on 2026-09-21.  `posture.WIDE` is
    still there for a caller that wants it.

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

                            WIDE sites (then)  under the hips
        feet, trunk x/y     +-237 / +-85 mm    +-156 / +-60 mm
        |tau| abd/pitch/knee 0.36/0.75/0.34    0.00/0.41/1.05 N*m
        reach used          0.795              0.707

    Measured from WIDE, before the trot went back to NOMINAL (+-65 mm in y,
    so the step now crosses less).  The SRB model stays the crouch's --
    `law.BalanceLaw.begin_step` says why.


THE JOINT-SPACE LAYER, 2026-09-21
    DOG5 trot_hw's `JointImpedance`, ported: every leg gets tau += Kp
    (q_hold - q) - Kd qd on top of the SRB stance torque and the swing.
    q_hold is the joint angles MEASURED ON THE SWEEP THE ROBOT REACHES HOLD,
    fixed once and kept through the hold and every trot (every --half-gait
    press too) -- with the feet planted, fixed joints fix the trunk's
    position and rpy.  T does not re-latch it; W releases it and re-latches
    at the new stance.  A swinging leg's target follows the leg -- the
    damper alone, nothing pulling it back down.  DOG5's 3.0 / 0.1
    (`config.KP_JOINT_HOLD`).

    TWO MODES, BOTH KEPT.  `--joint-hold` (the default) is the above.
    `--no-joint-hold` is the SRB law as it was before: it holds height and
    rpy and nothing pins the trunk's xy, so the body can shift over the
    feet -- the sway demos need exactly that.

THE HEADING IS ZEROED AT THE END OF THE CROUCH, 2026-09-24
    On the handover sweep -- the crouch has arrived, torque is about to be
    live -- the magnetometer's yaw is read once, becomes the world frame's x
    axis, and the reported yaw goes to ZERO.  Everything after that is drift
    off it.  `balance.state.rezero_yaw` has the arithmetic; `balance.law.arm`
    and `balance.sequence` have the timing.

    WHAT IT FIXES, AND IT IS A TROT BUG.  The heading used to stay inside R
    while the setpoint carried a copy of it.  `log(R_des R^T)` does not
    separate, so the heading did not come out of the log map on the yaw axis
    alone: it TURNED the roll and pitch error into each other.  At 37 deg of
    heading a 4.0 deg roll error reads as 3.2 deg of roll and 2.4 deg of
    PITCH -- the trunk pushed back on the wrong axis, on two feet, which is
    where the trot has the least to push against.  `config.KP_YAW = 0` could
    not switch it off: it zeroes the yaw ROW of the gain, not the yaw inside
    the log.  With the frame turned instead, the same 4.0 deg reads 4.0 deg
    of roll and 0.0 of pitch.

    DOG5'S CONVENTION, REACHED FROM THE OTHER SIDE.  DOG5 built its
    world-from-body rotation out of roll and pitch alone (`C_from_rp`) and
    let the heading in as a scalar beside it, so "the frame does not start
    rotating with the heading".  DOG6 gets the whole ZYX triple from the
    DETA10 in one matrix, so it turns the frame by the latched heading
    instead.  Same world, same instant, same reason.

AND THE YAW SPRING IS ON BECAUSE OF IT, 2026-09-24
    `config.KP_YAW` was zero because an absolute heading loop holds the robot
    against whatever twelve motors are doing to the magnetometer's field.
    With the heading zeroed at the handover that is no longer what the gain
    multiplies: it multiplies the DRIFT off a datum measured on this run, a
    difference in which the mount offset and the room cancel.  So the spring
    is on, at 25 against roll and pitch's 90 -- 0.80 Hz, critically damped,
    the slowest loop on the robot, asking 0.099 N*m/deg against the 3.5 N*m
    of yaw a trot's diagonal can make out of friction.  `config.KP_YAW` has
    the whole sizing; `--kp-yaw 0` is the trot as it ran before today.

    WHAT IT IS FOR: the trot's own heading drift.  Two feet, 40 mm of swing
    and a touchdown that slips leave degrees of yaw per run, and the drift
    used to be a one-way walk -- nothing pulled it back.  Now it comes back
    with a 0.2 s time constant, five times inside one 1.2 s cycle.  What no
    gain here reaches is the slip itself.

    AND IT IS WHY `hw.trot_esti` PRINTS THE YAW ACROSS EVERY TROT.  That
    read-out is the spring's report card, and it was the reason for turning
    it on: the filter's xy is leg odometry, so a trunk that turns takes the
    whole stance with it and the xy stops being a claim about the filter.

THE SWING FEEDFORWARD, 2026-09-25 (`--swing-ff`, off by default)
    The operator's diagnosis: the swing law fed forward gravity and nothing
    else, so the PD was making the arc's inertial torque -- 90 % of what a
    15 mm / 140 ms arc asks -- out of tracking error.  That is why 15 mm did
    not lift and 40 mm lifted shaking: the apex was a force gain, not a
    height.  `balance.swing.swing_feedforward` adds M0 J^+ (a_ref - Jdot
    qd_ref) per swing leg, open loop, in both swing modes; swing.py has the
    measurement and the arithmetic.  It is no licence on the slew: it IS the
    curve the banner holds against `--tau-slew`.  Its share of tau is logged
    (`tau_ff`).

    FLOWN THE SAME DAY, on the Cartesian swing: the foot lifts.  At 20 mm of
    apex.  At 40 mm it came down hard enough to bounce the robot -- with the
    inertial term in, the apex is real and so is the descent.  `--swing
    joint` failed every run on this posture and stays the fold's.

    AND ON THE BENCH THE FEEDFORWARD OVERSHOT.  Robot hung up, Cartesian
    swing, 15 mm asked: 26 mm reached on all four legs (16-17 swings each),
    |x| 15-21 mm, touchdown -0.3 to -0.65 m/s, tau_ff peak 2.0-2.3 N*m with
    the PD at 0.5 -- and lowering the PD gains changed nothing, because the
    feedforward is open loop.  The leg answered the model's torque with 1.7x
    the model's motion: M0 is too big, and 61-96 % of M0 is the reflected
    rotor, `params.ARMATURE`, DOG5's number, NOT MEASURED ON DOG6.
    `hw.swing_bench --analyse` fits the armature the leg actually has from a
    log's q and tau; `--ff-armature` puts it into the feedforward (both entry
    points).  The x drift goes with it: pitch and knee are over-driven by
    different factors, so the foot's acceleration tilts off the arc's line.

    WHAT THE BENCH SETTLED ON THE SAME DAY: feedforward OFF, `--kd-swing 20
    20 40`, tracked the apex well (the operator's reading).  Kd v_ref is a
    velocity feedforward with no armature in it -- swing.py has the
    arithmetic -- and it is the swing to carry to the trot first.

    THE EXIT AT DUTY 0.72 RODE ON.  `full_support` -- all four at full
    weight -- is true for one 4 ms sweep a cycle at that duty, because the
    contact ramp fills the four-foot window; the T-latched exit mostly missed
    it.  `sequence._four_foot_window` now falls back to the window's middle.

    AND THE FIRST FORM DRIFTED IN x.  The operator watched the swing foot
    leave the z line fore-aft and come back on the PD.  That was the Jdot
    qdot the first cut left out: the chain's own curvature at the arc's
    joint rates, a horizontal term the PD then had to undo.  One leg in its
    own dynamics, 20 mm / 140 ms: 4.3 mm of x at touchdown without it, 0.8
    with it.  It is in now, one Jacobian more.

VELOCITY, THE WHOLE RUN, 2026-09-21
    `hw.velocity_estimator` -- the DETA10's accelerometer integrated, biases
    taken in LIMP -- printed as `v (x, y, z) m/s` under every status line,
    every phase from limp to park, yaw-free world frame.  Print only: the
    law reads none of it.  No IMU, no velocity.  `hw.fold_trot` too.

KEYS
    ENTER  the stand's phases, as ever.  REFUSED while trotting or stepping.
    W      from HOLD, step the feet under the hips, or back.
    T      from HOLD, start trotting.  While trotting, latch the exit: the
           switch back to HOLD waits for all four feet at full weight.
           With --half-gait, T is half a gait cycle, stepped by hand: one
           diagonal lifts and lands, HOLD by itself, and the next T swings
           the OTHER diagonal (FR/RL, FL/RR, FR/RL, ...).  The swing is the
           gait's own; only the handover between diagonals waits for T.
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

__all__ = ["main", "trot_options", "CROUCH", "TAU_CAP", "PERIOD_S", "TILT_STOP_DEG",
           "TRACK_STOP_DEG", "STEP_FOOT_XY", "STEP_PERIOD_S"]

#: The crouch the trot lifts from and parks in: the normal one, abd not
#: splayed -- back from `posture.WIDE` on request 2026-09-21.
CROUCH = POSE.NOMINAL

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
#:
#: THAT TABLE IS THE HANDOVER'S.  THE SWING HAS ITS OWN SLEW BILL, AND IT IS
#: THE LARGER ONE (2026-09-24).  The arc's knee torque has to reach its peak
#: inside a quarter of the swing, so the slew it needs is ~tau_peak / (swing
#: / 4) -- `balance.swing.swing_demand`, printed in the banner against
#: `--tau-slew`.  At 40 mm of apex, on the leg's measured mass matrix
#: (`leg_dynamics.mass_matrix`, 61-96 % of it the reflected rotor): 160 ms of
#: swing (this 0.8 s) needs ~5 N*m and ~130 N*m/s, DOG5's 240 ms ~2.3 N*m and
#: ~40, and 80 ms (a 0.4 s period at duty 0.80) ~20 N*m and ~1000 -- past the
#: 9 N*m cap, which fits ~18 mm in that 40 ms half-swing.  The foot still
#: lifts there (seen on the robot, 2026-09-25); what it cannot do is follow
#: the arc, so what it does instead is the PD's behind the limiter, and the
#: log's `x_b` against `p_swing` says which -- late and short, or wound up.
#: The apex scales all of it linearly (`--swing-height`); the swing duration
#: is (1 - duty) x period, so a short period needs a LOW duty to keep it.
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
    """`hw.stand.main` from `CROUCH`, SRB only, plus the trot and W."""
    return STAND.main(argv, crouch=CROUCH, only_law="srb",
                      tilt_stop=TILT_STOP_DEG, track_stop=TRACK_STOP_DEG,
                      step_to=STEP_FOOT_XY, step_period=STEP_PERIOD_S,
                      velocity=True, **trot_options(PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
