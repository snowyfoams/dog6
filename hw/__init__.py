"""DOG6 hardware module.  The code came from DOG5; the MEASUREMENTS are DOG6's.

    python -m hw.selftest              gate the plumbing, no robot needed
    python -m hw.bringup plan          where the bring-up has got to

DOG5's PROTOCOL was ported across.  DOG5's TABLES never were, and the
difference is the whole shape of this module.  What was borrowed is the same
drivers speaking the same frames -- that does not change with a chassis.  What
was NOT borrowed is the two facts that describe one particular assembled
robot:

    a CAN id      says which driver is which joint
    a direction   says which way that joint turns

An id is what somebody flashed into a driver; a direction is which way round
somebody bolted a motor.  Neither is derivable and neither survives being
copied.  A wrong id sends a knee command to an abduction motor; a wrong sign
is a joint whose feedback and command disagree, which under a torque law is a
robot driving itself into the floor at full current.  NEITHER SHOWS UP IN
SIMULATION.

WHAT WAS ESTABLISHED ON 2026-09-15, AND IS NOW THE REFERENCE
    zero          the twelve drivers were zeroed at `coordinates.Q_ZERO`,
                  flat on the belly.  That pose is the robot's zero, not just
                  the model's, and every angle in this project is measured
                  from it.
    dir + can_id  all twelve rows of `hardware_map`, found with `bringup
                  spin` and confirmed with `bringup check`.
    kinematics    `hw.kinematics`' prediction for a positive joint angle is
                  what each direction was READ AGAINST, twelve times.  The
                  feet went where it said.
    coordinates   the flat-zero leg fold in `sim.coordinates` is fixed by the
                  built robot; where it and the CAD disagree, the robot wins.

    `hw.stand` then ran its position phases on all of it.  The LIFT phase did
    not hold -- see `hw.balance`, which is the controller written to replace
    the law that failed it.  The four facts above are not in question; the
    lift law is.

The guards stayed.  Everything needing joint coordinates asks `hardware_map`
at CALL time and raises `MapIncomplete` if a row is missing, so a re-flashed
driver stops the robot instead of moving the wrong joint.  Everything in the
MOTOR's own frame needs no map at all -- which is what let the bring-up tools
discover the table, and what they would use to re-measure a row.

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

    hardware_map.py   [MEASURED]  the two facts above, and nothing else.
                      CAN ids FL=1-3, FR=4-6, RR=7-9, RL=10-12 -- note RR
                      before RL on the bus, which is NOT canonical order --
                      and -1 on CAN 3, 6, 7, 9, 10, 12.
    calibration.py    [RUNS]  the motor frame -- `EncoderUnwrap`, the gain,
                      `soft_limits` -- never needed the map.  The joint
                      conversions resolve against the measured column, and
                      still raise `MapIncomplete` if a row goes missing.
    imu.py            [RUNS, SIGNS UNVERIFIED]  DETA10 -> trunk frame.  Splits
                      the sensor's NED->FLU convention from the MOUNTING
                      rotation, which is `coordinates.R_BODY_IMU` and still a
                      placeholder because the board is not bolted on.
                      `orientation()` is the control-facing view: R and
                      omega^b in SI, with the mounting rotation applied on
                      the correct side of each.

    balance/          [RUNS]  the stand's balance controller -- an SRB wrench
                      law with a force allocator, replacing the per-leg
                      Cartesian compliance in the LIFT PHASE ONLY.  Attitude
                      is measured rather than emergent; gravity enters once
                      as m*g rather than as a fixed mg/4 per foot.  78 checks
                      of its own, no robot needed.  See balance/README.md.

    stand.py          the six-phase sequence on the real drivers, and the
                      only file that decides WHEN anything runs.  Carries
                      both lift laws: `--law srb` and `--law per-leg`, the
                      second unchanged and kept as the A/B baseline.
    safety.py         [RUNS]  torque ramp, cap, limit block, slew and the
                      e-stop trips.  Still cannot be constructed over a map
                      with a hole in it, with no override -- eleven signs out
                      of twelve is not a map.
    fake_bus.py       twelve drivers in software, decoding the same protocol,
                      so every line above runs on a laptop.  Needs explicit
                      ids; it will not invent a default twelve.
    bringup.py        scan / spin / check / setzero / imu / plan.
    selftest.py       gates all of it with no robot.  48 checks.

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
CONFIRMED_ON_DOG6 = True

#: What was seen, and when.  Append one line per observation: the date, the
#: CAN id, which limb moved and which way.  `hardware_map` says WHAT was
#: measured; this says that somebody watched it happen.
VERIFICATION_LOG = """
2026-09-15  operator, on the assembled robot: all twelve CAN ids and directions
            confirmed -- CAN 3, 6, 7, 9, 10, 12 are -1, CAN 1, 2, 4, 5, 8, 11
            are +1.  Each was found with `bringup spin`, then re-driven in
            joint coordinates with `bringup check`; the foot went where
            `hw.kinematics` predicted, twelve times out of twelve.
2026-09-15  the twelve drivers zeroed at Q_ZERO (0x19), flat on the belly.
2026-09-15  `hw.stand` ran its POSITION phases on the robot -- limp, settle,
            crouch, park.  The LIFT phase did not hold.  Nothing was logged
            from those runs, so they could not be analysed afterwards; the
            per-sweep npz log in `hw.balance` exists because of that.
"""

__all__ = ["CONFIRMED_ON_DOG6", "VERIFICATION_LOG", "require_confirmed"]


def require_confirmed() -> None:
    """Raise unless the wiring has been measured and confirmed on the robot."""
    if CONFIRMED_ON_DOG6:
        return
    from .hardware_map import N_JOINTS, is_complete, unassigned, unconfirmed
    if not is_complete():
        raise RuntimeError(
            "hw/hardware_map.py has a HOLE: %d of %d rows have no CAN id or "
            "no direction (%s).\n  DOG6's twelve were measured on 2026-09-15, "
            "so this is an edited table or a replaced driver --\n  and DOG5's "
            "table is deliberately not here.\n"
            "  Re-measure that row: `python -m hw.bringup scan`."
            % (len(unassigned()), N_JOINTS, ", ".join(unassigned())))
    raise RuntimeError(
        "hw.CONFIRMED_ON_DOG6 is False -- %d of %d rows are filled in but "
        "have not been re-driven in joint coordinates and watched (%s).\n"
        "  `python -m hw.bringup check --joint <label> --go`."
        % (len(unconfirmed()), N_JOINTS, ", ".join(unconfirmed())))
