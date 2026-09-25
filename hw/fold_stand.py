"""Can the balance controller lift out of a crouch it was never designed for?

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_stand --fake --auto 1          the whole path, no robot
    $V -m hw.fold_stand --log fold.npz           on the robot: LIMITS OFF, 9 N*m
    $V -m hw.fold_stand --limits --tau-cap 3.0   the limits and the staged cap back

    LIMITS OFF, 2026-09-25, on request, after the parallel fold did not stand
    up: `--no-limits` is this script's default (`hw.stand.main`'s `limits`).
    The soft joint limits (the gate's torque block and its e-stop), the
    position-mode and torque-phase tracking trips, the tilt stop, the
    residual trip and the overspeed trip are all OFF, and the lift cap is the
    motors' own 9 N*m (`safety.TAU_HARD_NM`), as `hw.fold_trot`'s.  Still on:
    `hw.stand.NO_LIMITS_KEPT` -- X, the drivers' health, the torque shaping.
    `TILT_STOP_DEG` and `TRACK_STOP_DEG` below are what `--limits` flies.

    ROLL GAINS.  This script defaults roll to kp 290 / kd 23 (`ROLL_GAINS`);
    pitch stays at --kp-att / --kd-att (90 / 17).  Same units as --kp-att.
    $V -m hw.fold_stand --kp-roll 1000 --kd-roll 100
    $V -m hw.fold_stand --kp-roll 90 --kd-roll 17
                                          roll back to the hw.stand default

THE SRB BALANCE CONTROLLER ONLY.  There is no `--law` here and that is not an
omission.  The per-leg Cartesian compliance law has been run on DOG6 and it
FAILED; the SRB law is what works, so it is the only thing worth asking this
question of.  It could not be offered from this crouch anyway --
`sim.stand.compliance_torque` pins its four springs at the NOMINAL foot xy and
takes no posture, so from the fold it would spend the lift dragging the feet
90 mm back to the nominal stance.  `StandSequence` refuses the combination
rather than flying it.

`hw.stand`'s sequence, `hw.stand`'s runner, `hw.balance`'s law -- all of it
unchanged.  ONE thing is different: the crouch.  Every phase, every gain,
every trip and the whole CAN loop are `hw.stand`'s, and this file is three
lines of `main` plus the reason for them.


THE EXPERIMENT, AND WHAT IT IS ACTUALLY ASKING
    `sim.stand.Q_CROUCH` is a pose the lift was SOLVED FOR: shins vertical,
    feet at the nominal stance, trunk resting on the floor at h = 0.  Lifting
    out of it is the easy case and it is the only case the law has ever been
    asked for.

    `posture.FOLD` is a crouch a person folded the robot into by hand.  It
    agrees with the nominal one about nothing:

        h at the crouch     60 mm, NOT 0 -- the trunk starts in the air
        feet, trunk x       +215 / -154, against +-237 nominal; since
                            2026-09-25 the rear legs are PARALLEL to the
                            front (2026-09-17 to 09-25 mirrored as the
                            front, at -215)
        reach used          0.43 front, 0.35 rear
        knees               behind every hip: the front ones tucked under
                            the abd motors, 111 mm from the trunk origin,
                            the rear ones out behind the rear hips, 258 mm

    The law should not care about any of that, and the question is whether it
    does.  Stage 4 builds the grasp map from the MEASURED foot positions, so
    an odd stance is already ordinary to the allocator; stage 3's PD is
    written in the CoM and the trunk's attitude and never names a leg.  If
    the SRB law is what it claims to be, this lifts.  If it only ever worked
    because the geometry was the one it was tuned on, this is where that shows.


THE TWO THINGS THAT HAD TO CHANGE TO ASK THE QUESTION AT ALL
    THE TRACKING TRIP'S REFERENCE.  `law.ik_reference` pins the feet while
    the trunk rises, and it used to pin them at the NOMINAL stance.  From the
    fold crouch that is a 91.6 deg joint error on the first torque sweep,
    against a 25 deg trip -- the run would end before the ramp started, and it
    would end reporting a tracking failure that is really just two postures
    being different.  The pin is now a property of the posture
    (`CrouchPose.foot_xy`), so the feet are held WHERE THEY ARE and the trip
    goes back to meaning what it says.

    THE IMU DATUM IS FIXED, NOT LATCHED.  `--fixed-setpoint` is this script's
    default: "level" is `config.SETPOINT_ROLL_DEG / SETPOINT_PITCH_DEG` and
    nothing is captured at zero torque.  The reason is that this run exists to
    be COMPARED with a nominal one, and the zero-torque latch defines level as
    whatever the robot was resting at -- which is a DIFFERENT attitude in the
    two postures.  Two runs each held to their own private level cannot be put
    side by side.  `--latch-setpoint` turns it back on.


WHAT THE CAPTURE WAS, AND WHAT WAS DONE TO IT
    The pose was posed by hand and read off the encoders, so it arrived with
    two defects that belong to the operator and not to the posture: the left
    and right feet sat a few mm apart, and the four feet spanned 21.5 mm of
    trunk-frame z -- which on a flat floor is not four foot heights at all,
    it is the TRUNK PITCHED 3.43 deg.  `posture.regularise` removes both:
    mirror symmetric within each axle, one height for all four.  It moved each
    foot 10-12 mm and each joint 4.8-7.5 deg, and it left h where it was.

    IT IS WORTH KNOWING WHAT THAT BOUGHT.  With the feet at one height the
    level-trunk reference IS the pose, so the tracking error starts at exactly
    zero; on the raw capture it starts at 6.33 deg of the 25 deg budget,
    spent before the robot has moved.  `posture.FOLD_CAPTURED_Q` keeps the raw
    numbers and the regularisation is re-run at import, so the two cannot
    drift apart.


WHAT TO WATCH, IN ORDER
    limp     the attitude the folded robot actually rests at.  It is NOT the
             setpoint and it is not supposed to be -- the setpoint is fixed.
             A big number here is the honest reading of "the fold is tilted".
    crouch   the drivers taking the robot to the fold pose.  Watch the
             tracking line; this is a bigger move than the nominal crouch.
    rise     h from 60 mm.  `arm` latches h0 from the MEASUREMENT, so the
             quintic starts where the robot is, not at 0.
    hold     push it.  That is the whole point, and `hold` on the status line
             is what came back.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold_stand.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import safety as SAFE         # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import posture as POSE  # noqa: E402

__all__ = ["main", "TILT_STOP_DEG", "ROLL_GAINS", "TRACK_STOP_DEG", "TAU_CAP",
           "LIMITS"]

#: `hw.stand.main`'s `limits`: OFF, on request 2026-09-25 -- see the module
#: docstring.  `--limits` on the command line puts them back.
LIMITS = False

#: The lift cap: the motors' own, as `hw.fold_trot`'s.  `--tau-cap 3.0` is
#: the staged one this script used to need.
TAU_CAP = SAFE.TAU_HARD_NM

#: The tilt e-stop for this script, in degrees FROM THE SETPOINT.  Raised from
#: `config.TILT_STOP_DEG` (12) because the question this run exists to answer
#: is what the attitude loop does with a STEADY-STATE tilt, and a stop that
#: fires at 12 deg ends the run before the number can be read.
#:
#: IT IS STILL A TRIP AND IT IS NOW A WEAK ONE.  At 45 deg the robot is far
#: past anything a four-foot stand recovers from, and an e-stop there drops
#: the trunk from a worse attitude than one at 12 would have.  The tracking
#: stop is OFF as well -- see `TRACK_STOP_DEG` for what that leaves guarding
#: the run.  Run supported.
TILT_STOP_DEG = 45.0

#: The joint tracking trip: OFF.  It compares the measured joints against the
#: IK at the COMMANDED height, and that IK reaches nothing but the trip and the
#: log -- the SRB torque never reads it -- so switching it off changes no N*m.
#:
#: WHY IT IS OFF HERE.  From the fold crouch it fires at 40 mm of height
#: shortfall, against a rise that only commands 55 mm, so it conflates "the
#: robot is lagging the ramp" with "a leg is wrong".  And it is referenced to a
#: LEVEL trunk, so the steady attitude errors this script exists to observe eat
#: into it as well.
#:
#: WHAT IS LEFT GUARDING THE RUN, with this off and the tilt stop at 45 deg:
#: the residual monitor, the torque cap and its readback, the CAN gap and
#: input-lost trips, and the non-finite check.  NOTHING among those notices a
#: foot sliding or a leg folding slowly while the attitude still reads fine --
#: that was this trip's job.  `--track-stop 25` puts it back.
TRACK_STOP_DEG = 0.0

#: ROLL gains, (kp 1/s^2, kd 1/s) -- the same units as `--kp-att`.
#:
#: Raised from kp_att 90 / kd_att 17 on roll only.  The PD is an acceleration
#: law, so equal gains give roll and pitch equal bandwidth but the restoring
#: MOMENT scales with inertia, and this robot's I_xx is 4.6x smaller than its
#: I_yy in the fold stance: at 90 a 0.14 N*m bias parks roll at 2.6 deg.  The
#: second fold run's roll that did not come back is that.
#:
#: kp 290 puts roll at the moment stiffness DOG5 held on both axes
#: (KP_ORI_TROT 10 N*m/rad).  kd 23 is chosen for LATENCY: in a sampled model
#: of this loop at 250 Hz it stays stable to 40 ms of attitude age, the most
#: of any damping at this kp.  `IMU_MAX_AGE_S` freezes the attitude half at
#: 50 ms, so the hold line prints the IMU age.
ROLL_GAINS = (290.0, 23.0)


def main(argv=None) -> int:
    """`hw.stand.main`, from `posture.FOLD`, SRB only, IMU datum FIXED,
    limits OFF, the 9 N*m cap."""
    return STAND.main(argv, crouch=POSE.FOLD, dynamic_setpoint=False,
                      only_law="srb", tilt_stop=TILT_STOP_DEG,
                      roll_gains=ROLL_GAINS,
                      track_stop=TRACK_STOP_DEG, tau_cap=TAU_CAP,
                      tau_ceiling=TAU_CAP, limits=LIMITS)


if __name__ == "__main__":
    raise SystemExit(main())
