"""DOG6 hardware module.  The code is here; the MEASUREMENTS are not.

    python -m hw.selftest              gate the plumbing, no robot needed
    python -m hw.bringup plan          where the bring-up has got to

DOG5's PROTOCOL has been ported across.  DOG5's TABLES have not, and the
difference is the whole shape of this module.  What was borrowed is the same
drivers speaking the same frames -- that does not change with a chassis.  What
was NOT borrowed is the two facts that describe one particular assembled
robot:

    a CAN id      says which driver is which joint
    a direction   says which way that joint turns

An id is what somebody flashed into a driver; a direction is which way round
somebody bolted a motor.  Neither is derivable and neither survives being
copied, so `hardware_map` is EMPTY -- twelve rows of None.  A wrong id sends a
knee command to an abduction motor; a wrong sign is a joint whose feedback and
command disagree, which under a torque law is a robot driving itself into the
floor at full current.  NEITHER SHOWS UP IN SIMULATION.

Everything that needs joint coordinates therefore RAISES rather than running
on a guess.  Everything in the MOTOR's own frame still works, which is what
lets the bring-up tools discover the map in the first place.

WHAT IS HERE

    kinematics.py     [RUNS]  closed-form FK, Jacobian and IK -- 0.10 ms for
                      all twelve joints, both directions.  CAD geometry, read
                      out of `sim.params` rather than copied, gated by
                      `sim.selftest` against MuJoCo over 500 random poses.  No
                      wiring in it, so no map needed.  It is also the YARDSTICK
                      the direction measurement is read against.

    motor/            [RUNS]  the CAN transport and the LK protocol, four
                      modules BYTE-FOR-BYTE from DOG5.  Same gearmotors, same
                      drivers.  Includes the over-CAN recovery of the
                      input-signal-lost latch, which is what removes the power
                      cycle between runs.  Addresses raw CAN ids, so it needs
                      no map.

    hardware_map.py   [EMPTY]  the two facts above, and nothing else.  Twelve
                      rows of None, waiting for the robot.
    calibration.py    [HALF]  the motor frame -- `EncoderUnwrap`, the gain,
                      `soft_limits` -- runs today.  Anything in joint
                      coordinates raises `MapIncomplete`.
    imu.py            [RUNS, SIGNS UNVERIFIED]  DETA10 -> trunk frame.  Splits
                      the sensor's NED->FLU convention from the MOUNTING
                      rotation, which is `coordinates.R_BODY_IMU` and still a
                      placeholder because the board is not bolted on.
    safety.py         [REFUSES]  torque ramp, cap, limit block, slew and the
                      e-stop trips.  Cannot be constructed on an empty map,
                      with no override -- there are no signs to shape through.
    fake_bus.py       twelve drivers in software, decoding the same protocol,
                      so every line above runs on a laptop.  Needs explicit
                      ids; it will not invent a default twelve.
    bringup.py        scan / spin / check / setzero / imu / plan.
    selftest.py       gates all of it with no robot.  53 checks.

THE TWO VALUES SIM IS ALREADY HOLDING A PLACE FOR
    Both are in `sim` right now marked as placeholders, and both become
    measurements the moment hardware exists.  When they do, they have to move
    in BOTH places or the sim and the robot quietly disagree:

    coordinates.R_BODY_IMU     identity today.  Also baked into
                               model/dog6.xml as the imu site's quat --
                               `sim.selftest` gates that the two agree, so
                               changing one and not the other fails the gate
                               rather than passing silently.

    params.TAU_MAX_SIM         8.0 N*m today, which is the motor's saturation
                               with 20 % held back.  A SIMULATION NUMBER.  On
                               first power-up this drops to a low staging cap
                               and climbs one logged run at a time, the way
                               DOG5's 3.0 N*m ceiling did.

THREE STATES, NOT TWO, AND THEY ARE CHECKED SEPARATELY
    EMPTY        a row has no CAN id or no direction.  `MapIncomplete`, no
                 override anywhere: there is nothing to override.
    WIRED        the row is filled in from `hw.bringup spin`, but nobody has
                 re-driven it in joint coordinates and watched.
    CONFIRMED    `hw.bringup check` passed on it.  `CONFIRMED_ON_DOG6` is the
                 conjunction over all twelve, and `hw.selftest` checks it
                 cannot be True over an incomplete table.

    The middle state is real and matters: filling a row in is a note about
    what you saw, and confirming it is a separate observation that the note
    was right.  DOG5 collapsed the two and the check lived in an operator's
    memory.

WHAT A GREEN `hw.selftest` DOES NOT MEAN
    It runs the protocol path against `fake_bus` under a SYNTHETIC map that it
    installs itself.  `fake_bus` obeys whatever map it is given, so this can
    never tell you a real map is right.  It says the plumbing is sound.  It
    never says it is safe to power the robot.
"""
from __future__ import annotations

#: Flip to True only after every row of `hardware_map` has been measured on
#: the assembled DOG6 AND re-driven in joint coordinates and watched.  The
#: table carries the same state per row; this flag is the conjunction, and
#: `hw.selftest` checks the two agree so a half-finished bring-up cannot
#: leave the flag on.
CONFIRMED_ON_DOG6 = False

#: What was seen, and when.  Empty until something has been.  Append one line
#: per joint: the date, the CAN id, which limb moved and which way.
VERIFICATION_LOG = ""

__all__ = ["CONFIRMED_ON_DOG6", "VERIFICATION_LOG", "require_confirmed"]


def require_confirmed() -> None:
    """Raise unless the wiring has been measured and confirmed on the robot."""
    if CONFIRMED_ON_DOG6:
        return
    from .hardware_map import N_JOINTS, is_complete, unassigned, unconfirmed
    if not is_complete():
        raise RuntimeError(
            "hw/hardware_map.py is EMPTY: %d of %d rows have no CAN id or no "
            "direction (%s).  Nothing has been read off DOG6, and DOG5's "
            "table is deliberately not here.\n"
            "  Start with `python -m hw.bringup scan`."
            % (len(unassigned()), N_JOINTS, ", ".join(unassigned())))
    raise RuntimeError(
        "hw.CONFIRMED_ON_DOG6 is False -- %d of %d rows are filled in but "
        "have not been re-driven in joint coordinates and watched (%s).\n"
        "  `python -m hw.bringup check --joint <label> --go`."
        % (len(unconfirmed()), N_JOINTS, ", ".join(unconfirmed())))
