"""Which CAN id is which joint, and which way each one turns.

    python -m hw.hardware_map          what is known, what is not

THIS FILE IS A RECORD OF MEASUREMENTS.
    All twelve rows were filled in from DOG6 itself on 2026-09-15, after
    assembly and calibration.  None is `confirmed` yet.  Before that every
    `can_id` and `direction` was None, and that was the correct state of
    knowledge about a robot nobody had driven a motor on.

    DOG5's values are deliberately NOT here.  They were tempting: same motor,
    same driver, same protocol, and a table of twelve numbers that already
    works on a machine down the bench.  But a CAN id is what somebody flashed
    into a driver and a direction is which way round somebody bolted a motor,
    and neither survives being copied.  A borrowed table is a guess wearing
    the clothes of a measurement, and the whole point of leaving these None is
    that the code CANNOT RUN until they are real -- rather than running
    plausibly on somebody else's robot's numbers.

    A wrong id sends a knee command to an abduction motor.  A wrong sign is a
    joint whose feedback and command disagree, which under a torque law is a
    robot driving itself into the floor at full current.  NEITHER SHOWS UP IN
    SIMULATION.

HOW THE TABLE GETS FILLED, AND WHY IT IS DISCOVERY AND NOT VERIFICATION
    You do not check this table against the robot.  You BUILD it from the
    robot, one motor at a time:

      1. `hw.bringup scan`            which CAN ids answer at all?  This asks
                                      the bus, not this file, so it works on
                                      an empty map.
      2. `hw.bringup spin --id 7`     drive THAT id a small step in the
                                      MOTOR's own frame -- direction is +1
                                      because no direction is known yet, which
                                      is exactly what "unknown" should mean.
                                      Watch which limb moves.  That is the
                                      `can_id` for that joint.
      3. `hw.bringup spin --id 7 --joint FL.knee`
                                      the same step, now with the prediction
                                      `hw.kinematics` makes for a POSITIVE
                                      joint angle printed beside it.  Foot
                                      went the predicted way -> direction +1.
                                      Opposite way -> -1.  It prints both
                                      candidate lines; paste the right one.
      4. `hw.bringup check --joint FL.knee`
                                      once the row is filled, drive it again
                                      in JOINT coordinates and confirm the
                                      foot goes where the kinematics says.
                                      This is the pass that sets
                                      `confirmed=True`.
      5. `hw.bringup setzero`         only once all twelve rows are complete.

    Steps 2 and 3 are the same command; the difference is that by step 3 you
    know which joint you are looking at.

WHAT WORKS ON AN INCOMPLETE MAP, AND WHAT REFUSES
    Works     `scan` and `spin`, because they address raw CAN ids and command
              in the motor's own frame.  The protocol layer (`hw.motor`), all
              of `hw.kinematics`, and `hw.imu`.
    Refuses   anything that needs joint coordinates -- `hw.calibration`'s
              conversions, `joint_state`, `set_zero_all`, and
              `hw.safety.SafetyGate`.  They raise `MapIncomplete`, which names
              the rows that are still empty.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

if __package__ in (None, ""):        # allow `python hw/hardware_map.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402

__all__ = ["HardwareJoint", "MapIncomplete", "HARDWARE_JOINTS", "JOINT_LABELS",
           "N_JOINTS", "motor_ids", "motor_directions", "joint_of", "row_of",
           "is_complete", "assigned", "unassigned", "unconfirmed",
           "confirmed_count", "require_complete", "record_template",
           "direction_check_table", "validate", "describe"]


class MapIncomplete(RuntimeError):
    """Raised when joint coordinates are needed and the map is not filled in."""


@dataclass(frozen=True)
class HardwareJoint:
    """One joint's wiring.  `model_name` ties it to model/dog6.xml.

    `can_id` and `direction` are None until they have been READ OFF THE
    ROBOT.  `confirmed` is a second, stronger claim: the filled-in row was
    driven again in joint coordinates and the foot went where
    `hw.kinematics` said it would.
    """

    leg: str
    joint: str
    model_name: str
    can_id: int | None = None
    direction: int | None = None
    confirmed: bool = False
    #: Free text: when it was seen and what was seen.  One line.
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.leg}.{self.joint}"

    @property
    def wired(self) -> bool:
        return self.can_id is not None and self.direction is not None


#: Canonical order: `coordinates.LEGS` x `coordinates.JOINTS`, so row i is
#: entry i of every (12,) vector in the project and needs no permutation.
#:
#: Read off DOG6 after assembly and calibration, 2026-09-15.  CAN ids are
#: wired FL=1-3, FR=4-6, RR=7-9, RL=10-12 -- note RR comes before RL on the
#: bus, which is NOT canonical order.  None re-driven with `hw.bringup check`
#: yet, so nothing is `confirmed`.
HARDWARE_JOINTS: tuple[HardwareJoint, ...] = (
    HardwareJoint('FL', 'abd', 'hip_abd_FL', can_id=1, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('FL', 'pitch', 'hip_pitch_FL', can_id=2, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('FL', 'knee', 'knee_FL', can_id=3, direction=-1, note='2026-09-15 calibrate'),
    HardwareJoint('FR', 'abd', 'hip_abd_FR', can_id=4, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('FR', 'pitch', 'hip_pitch_FR', can_id=5, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('FR', 'knee', 'knee_FR', can_id=6, direction=-1, note='2026-09-15 calibrate'),
    HardwareJoint('RL', 'abd', 'hip_abd_RL', can_id=10, direction=-1, note='2026-09-15 calibrate'),
    HardwareJoint('RL', 'pitch', 'hip_pitch_RL', can_id=11, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('RL', 'knee', 'knee_RL', can_id=12, direction=-1, note='2026-09-15 calibrate'),
    HardwareJoint('RR', 'abd', 'hip_abd_RR', can_id=7, direction=-1, note='2026-09-15 calibrate'),
    HardwareJoint('RR', 'pitch', 'hip_pitch_RR', can_id=8, direction=+1, note='2026-09-15 calibrate'),
    HardwareJoint('RR', 'knee', 'knee_RR', can_id=9, direction=-1, note='2026-09-15 calibrate'),
)

#: Human labels in canonical order, e.g. "FL.knee".
JOINT_LABELS: tuple[str, ...] = tuple(j.label for j in HARDWARE_JOINTS)

N_JOINTS = len(HARDWARE_JOINTS)


# ---------------------------------------------------------------------------
# what is known
# ---------------------------------------------------------------------------
def is_complete() -> bool:
    """True once every row has both a CAN id and a direction."""
    return all(j.wired for j in HARDWARE_JOINTS)


def assigned() -> tuple[HardwareJoint, ...]:
    """The rows that have been filled in."""
    return tuple(j for j in HARDWARE_JOINTS if j.wired)


def unassigned() -> tuple[str, ...]:
    """Labels of the rows still empty."""
    return tuple(j.label for j in HARDWARE_JOINTS if not j.wired)


def unconfirmed() -> tuple[str, ...]:
    """Labels not yet re-driven in joint coordinates and confirmed."""
    return tuple(j.label for j in HARDWARE_JOINTS if not j.confirmed)


def confirmed_count() -> int:
    return sum(1 for j in HARDWARE_JOINTS if j.confirmed)


def require_complete(what: str = "this operation") -> None:
    """Raise `MapIncomplete` unless every row is filled in."""
    if not is_complete():
        missing = unassigned()
        raise MapIncomplete(
            f"{what} needs joint coordinates, and {len(missing)} of "
            f"{N_JOINTS} rows of hw/hardware_map.py are still empty "
            f"({', '.join(missing)}).\n"
            "  Nothing has been read off DOG6 yet and DOG5's table is "
            "deliberately not here.\n"
            "  Build it: `python -m hw.bringup scan`, then "
            "`python -m hw.bringup spin --id <n>` per motor.")


def motor_ids() -> list[int]:
    """CAN ids in canonical joint order.  Raises if the map is incomplete."""
    require_complete("motor_ids()")
    return [j.can_id for j in HARDWARE_JOINTS]


def directions() -> list[int]:
    """The direction column in canonical joint order.  Raises if incomplete.

    A FUNCTION so that it resolves `HARDWARE_JOINTS` when it is CALLED.
    `hw.calibration` uses this rather than importing the tuple by name,
    which matters for two reasons: the table is edited by hand between runs,
    and `hw.selftest` swaps a synthetic one in to exercise the joint-coordinate
    paths that the real, empty table correctly refuses.
    """
    require_complete("directions()")
    return [j.direction for j in HARDWARE_JOINTS]


def motor_directions() -> dict[int, int]:
    """``{can_id: +/-1}`` for ``MotorBus(ids, dirs=...)``.  Raises if incomplete.

    With this passed, `MotorBus`' torque, speed and position commands are all
    in JOINT coordinates.  WITHOUT it every command is in the motor's own
    frame, which is the right thing during `spin` and the wrong thing after.
    """
    require_complete("motor_directions()")
    return {j.can_id: j.direction for j in HARDWARE_JOINTS}


def row_of(label: str) -> tuple[int, HardwareJoint]:
    """``(canonical index, row)`` for "FL.knee" and friends."""
    for index, entry in enumerate(HARDWARE_JOINTS):
        if entry.label == label:
            return index, entry
    raise ValueError(f"unknown joint {label!r}; expected one of "
                     f"{', '.join(JOINT_LABELS)}")


def joint_of(can_id: int) -> HardwareJoint | None:
    """The row claiming this CAN id, or None if nothing does yet."""
    for entry in HARDWARE_JOINTS:
        if entry.can_id == can_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# filling it in
# ---------------------------------------------------------------------------
def record_template(label: str, can_id: int, direction: int,
                    note: str = "", confirmed: bool = False) -> str:
    """The literal line to paste into HARDWARE_JOINTS for one measured joint."""
    _, entry = row_of(label)
    parts = [repr(entry.leg), repr(entry.joint), repr(entry.model_name),
             "can_id=%d" % can_id, "direction=%+d" % direction]
    if confirmed:
        parts.append("confirmed=True")
    if note:
        parts.append("note=%r" % note)
    return "    HardwareJoint(%s)," % ", ".join(parts)


def _explicit_table() -> str:
    """HARDWARE_JOINTS written out row by row, ready to be edited.

    The tuple above is a comprehension while every row is empty, because
    twelve identical empty rows are noise.  The moment a real value exists it
    should be written out longhand -- a measurement belongs in the source as a
    literal, where a reader can see it and a diff can show it changing.
    """
    lines = ["HARDWARE_JOINTS: tuple[HardwareJoint, ...] = ("]
    for entry in HARDWARE_JOINTS:
        parts = [repr(entry.leg), repr(entry.joint), repr(entry.model_name)]
        if entry.can_id is not None:
            parts.append("can_id=%d" % entry.can_id)
        if entry.direction is not None:
            parts.append("direction=%+d" % entry.direction)
        if entry.confirmed:
            parts.append("confirmed=True")
        if entry.note:
            parts.append("note=%r" % entry.note)
        lines.append("    HardwareJoint(%s)," % ", ".join(parts))
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def validate() -> None:
    """Structural checks on whatever is filled in.  Raises on anything wrong.

    THIS CANNOT TELL YOU A FILLED-IN ROW IS CORRECT.  It checks the rows are
    in `coordinates.JOINT_NAMES` order, that no two joints claim the same CAN
    id, that directions are +/-1, and that a row is not marked `confirmed`
    while it is still empty.  Correctness is a robot's answer, not a
    function's.
    """
    if N_JOINTS != C.N_JOINTS:
        raise ValueError(f"expected {C.N_JOINTS} joints, got {N_JOINTS}")
    if tuple(j.model_name for j in HARDWARE_JOINTS) != C.JOINT_NAMES:
        raise ValueError("hardware map is not in coordinates.JOINT_NAMES order")

    ids = [j.can_id for j in HARDWARE_JOINTS if j.can_id is not None]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"two joints claim the same CAN id: {duplicated}")
    if any(not 0 <= i <= 0x7FF for i in ids):
        raise ValueError(f"CAN ids outside the 11-bit range: {ids}")

    for entry in HARDWARE_JOINTS:
        if entry.direction not in (None, -1, +1):
            raise ValueError(f"{entry.label}: direction must be +1, -1 or "
                             f"None, got {entry.direction!r}")
        if entry.confirmed and not entry.wired:
            raise ValueError(f"{entry.label} is marked confirmed but has no "
                             "CAN id or no direction")


def direction_check_table(step_rad: float = 0.1):
    """What a POSITIVE JOINT step does to each foot, at the stance pose.

    Returns ``[(label, dx, dy, dz), ...]`` in canonical order, millimetres in
    the trunk frame.  Needs no CAN id and no direction -- it is a statement
    about the KINEMATICS, and it is the yardstick the measured direction is
    read against.

    COMPUTED FROM `hw.kinematics`, NOT TYPED HERE.  A stale copy would confirm
    a sign against the wrong expectation, which is the exact failure this
    whole file is trying to prevent.
    """
    import numpy as np

    from . import kinematics as K

    rows = []
    for index, entry in enumerate(HARDWARE_JOINTS):
        leg_index = C.LEG_INDEX[entry.leg]
        dq = np.zeros(3)
        dq[index % C.N_JOINTS_PER_LEG] = step_rad
        q0 = C.Q_STAND[leg_index]
        delta = 1000.0 * (K.foot_position(leg_index, q0 + dq)
                          - K.foot_position(leg_index, q0))
        rows.append((entry.label, *delta))
    return rows


def dominant(dx: float, dy: float, dz: float) -> str:
    """"+x" / "-z" -- the axis a foot displacement mostly lies along."""
    index, value = max(enumerate((dx, dy, dz)), key=lambda p: abs(p[1]))
    return ("-" if value < 0 else "+") + "xyz"[index]


def describe(step_rad: float = 0.1) -> str:
    lines = [
        "DOG6 hardware map",
        "  wired      %d / %d rows have a CAN id and a direction"
        % (len(assigned()), N_JOINTS),
        "  confirmed  %d / %d re-driven in joint coordinates and checked"
        % (confirmed_count(), N_JOINTS),
    ]
    if not is_complete():
        lines += ["  EMPTY: %s" % ", ".join(unassigned()),
                  "",
                  "  Nothing has been read off DOG6.  DOG5's table is",
                  "  deliberately not here -- see the module docstring.",
                  "  Build it:  python -m hw.bringup scan",
                  "             python -m hw.bringup spin --id <n>"]
    lines += [
        "",
        "  the yardstick -- a POSITIVE JOINT step of +%.2f rad from Q_STAND,"
        % step_rad,
        "  in millimetres, trunk frame (x fwd, y left, z up):",
        "",
        "    joint      CAN  dir  ok        dx       dy       dz   dominant",
    ]
    for (label, dx, dy, dz), entry in zip(direction_check_table(step_rad),
                                          HARDWARE_JOINTS):
        lines.append("    %-9s %4s %4s %3s   %+8.2f %+8.2f %+8.2f     %s"
                     % (label,
                        "-" if entry.can_id is None else entry.can_id,
                        "-" if entry.direction is None
                        else "%+d" % entry.direction,
                        "Y" if entry.confirmed else "-",
                        dx, dy, dz, dominant(dx, dy, dz)))
    return "\n".join(lines)


validate()


if __name__ == "__main__":
    print(describe())
    if not is_complete():
        print("\n  the table as it stands, longhand -- paste measured rows in:\n")
        print("\n".join("  " + line for line in _explicit_table().splitlines()))
