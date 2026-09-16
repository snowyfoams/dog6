"""The calibrated joint contract: raw 16-bit encoder <-> joint angle in radians.

Ported from DOG5, whose version of this ran on a real robot.  Same drivers,
same 0x9C register, same 10:1 gearbox, so the CONTRACT carries across intact.

WHAT DOES NOT CARRY ACROSS IS THE DIRECTION COLUMN IT MULTIPLIES BY.  DOG5's
was confirmed on DOG5.  DOG6's was read off DOG6 on 2026-09-15, confirmed
joint by joint, and it lives in `hardware_map` -- measured, not borrowed.
Every function here still goes through `joint_directions()`, which still
raises `MapIncomplete` when a row is missing: the measurement satisfied that
guard, it did not remove it.  `joint_directions()` is a function and not a
module-level array for exactly that reason -- an array would bind whatever
the table said at import time, and a re-flashed driver would go unseen.

Independent of the wiring, and so runnable even with no map at all:
`soft_limits`, `EncoderUnwrap`, `new_unwrappers`, and the parts of `validate`
that do not need it.  Those are the motor's own frame and the project's
conventions -- neither depends on which driver is which.

THE CONTRACT, IN TWO LINES, WITH NOTHING HIDDEN IN IT

    motoroutput_deg = raw_encoder * ENCODER_GAIN        ENCODER_GAIN = 360/65535
    joint_rad       = radians(direction * motoroutput_deg)

There is NO software zero offset and NO gearbox division.  Both absences are
deliberate and both bit DOG5 before they were written down:

    NO OFFSET     the zero lives in the DRIVER, written once by the 0x19
                  set-zero command at the calibration pose (see
                  :func:`set_zero_all`).  A software offset captured at
                  start-up would mean the robot's zero changes with whatever
                  pose it happened to boot in.
    NO /10        the 0x9C encoder register on this driver already reports the
                  OUTPUT shaft, so dividing by the 10:1 reduction would be
                  wrong by exactly that factor.  Older HIL code for DOG5
                  divides the same register by 10; the two calibrations are
                  incompatible and must not be mixed.

THE CALIBRATION POSE IS `coordinates.Q_ZERO` -- flat on the belly, all twelve
joints reading zero, legs straight out fore-and-aft.  That is a convention
DOG6 inherits from DOG5 unchanged, and `coordinates` explains why it is the
pose a human can actually put a robot into on a bench.  DOG6's CAD is drawn
STANDING, so the flat pose is a reconstruction -- and on 2026-09-15 DOG6's
twelve drivers were zeroed at it, which makes it the robot's zero and not
just the model's.  `coordinates` records which way the leg plane folds there;
that fold is fixed by the built robot.

The 16-bit register wraps.  :class:`EncoderUnwrap` turns it into a continuous
angle by counting the wraps, centring the FIRST sample into [-180, 180) deg so
a joint sitting just below zero reads as a small negative number rather than
359 deg.  One tracker per joint, fed every sweep -- it cannot see a wrap it was
not shown, so do not skip samples and do not share a tracker between joints.

SOFT LIMITS come from `params.JOINT_LIMITS`, not from a copy here.  DOG6's
abduction limit is 2.2 rad where DOG5's was 1.75, because DOG6 STANDS at
+-pi/2 = 1.571 and a 1.75 stop would leave the standing pose 0.18 rad from a
limit a trot's abduction ripple would reach.  Porting DOG5's number would have
been the quiet kind of wrong.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/calibration.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from .hardware_map import (          # noqa: E402
    JOINT_LABELS, N_JOINTS, directions, is_complete, motor_directions,
    motor_ids,
)
from .motor import ENCODER_GAIN      # noqa: E402

__all__ = ["ENCODER_GAIN", "joint_directions", "LIMIT_ESTOP_MARGIN",
           "soft_limits", "motoroutput_deg_to_joint_rad",
           "joint_rad_to_motoroutput_deg", "EncoderUnwrap", "new_unwrappers",
           "joint_state", "validate", "set_zero_all", "describe"]


def joint_directions() -> np.ndarray:
    """The measured direction column as a (12,) vector of +/-1.

    A FUNCTION, NOT A CONSTANT, AND THAT IS STILL THE WHOLE POINT.  DOG6's
    twelve are measured now, so this returns them -- but binding them at
    import time would freeze the column against a table that can change, and
    a module-level array of twelve plausible values is the artifact that lets
    an UNMEASURED robot run.  Every conversion below calls this, so every one
    of them raises `MapIncomplete` the moment a row goes missing rather than
    carrying on with a stale copy.

    It goes through `hardware_map.directions()` rather than reading the table
    directly, so the lookup happens at call time and a re-measured row takes
    effect without an import dance.
    """
    return np.asarray(directions(), dtype=float)


#: How far outside `params.JOINT_LIMITS` a joint may be MEASURED before the
#: safety gate calls it an e-stop rather than just refusing to push further
#: out.  A joint can end up slightly outside without anything being wrong --
#: gravity pulls an unpowered leg past a software bound -- and tripping on
#: that is a nuisance-stop.
LIMIT_ESTOP_MARGIN = 0.05


def soft_limits() -> tuple[np.ndarray, np.ndarray]:
    """``(low, high)``, each (12,) radians, in canonical joint order.

    `params.JOINT_LIMITS` restated per joint.  SOFTWARE bounds measured
    against the calibrated zero, not mechanical stops: a pose outside them is
    not necessarily one that damages anything, and one inside them is not
    necessarily safe.  model/dog6.xml carries no ranges at all -- `selftest`
    asserts that -- so nothing but a caller enforces these.
    """
    low = np.tile(P.JOINT_LIMITS[:, 0], C.N_LEGS)
    high = np.tile(P.JOINT_LIMITS[:, 1], C.N_LEGS)
    return low, high


# ---------------------------------------------------------------------------
# the conversion, both ways
# ---------------------------------------------------------------------------
def motoroutput_deg_to_joint_rad(motoroutput_deg) -> np.ndarray:
    """Unwrapped motor-output degrees (12,) -> joint angles in radians."""
    motoroutput_deg = np.asarray(motoroutput_deg, dtype=float)
    if motoroutput_deg.shape != (N_JOINTS,):
        raise ValueError(f"motoroutput must have shape ({N_JOINTS},), "
                         f"got {motoroutput_deg.shape}")
    return np.deg2rad(joint_directions() * motoroutput_deg)


def joint_rad_to_motoroutput_deg(joint_rad) -> np.ndarray:
    """The inverse -- what a joint target looks like on the motor's own dial.

    Useful for reading a commanded pose off the robot with a protractor, and
    for building a 0xA4 position payload by hand.
    """
    joint_rad = np.asarray(joint_rad, dtype=float)
    if joint_rad.shape != (N_JOINTS,):
        raise ValueError(f"joint angle must have shape ({N_JOINTS},), "
                         f"got {joint_rad.shape}")
    # direction is +/-1, so it is its own inverse.
    return joint_directions() * np.rad2deg(joint_rad)


class EncoderUnwrap:
    """Continuous motor-output degrees from the wrapping 16-bit register.

    One per joint.  The first sample is centred into [-180, 180) deg; every
    later sample is compared with the previous one and a jump of more than
    half a turn is counted as a wrap.
    """

    _FULL_TURN = 65536

    def __init__(self):
        self._previous_raw = None
        self._turns = 0

    def update(self, raw: int) -> float:
        raw = int(raw)
        if not 0 <= raw < self._FULL_TURN:
            raise ValueError(f"encoder value outside uint16 range: {raw}")
        if self._previous_raw is None:
            self._turns = -1 if raw >= self._FULL_TURN // 2 else 0
        else:
            delta = raw - self._previous_raw
            if delta > self._FULL_TURN // 2:
                self._turns -= 1
            elif delta < -self._FULL_TURN // 2:
                self._turns += 1
        self._previous_raw = raw
        return (raw + self._turns * self._FULL_TURN) * ENCODER_GAIN


def new_unwrappers() -> list[EncoderUnwrap]:
    """One :class:`EncoderUnwrap` per joint, in canonical order."""
    return [EncoderUnwrap() for _ in range(N_JOINTS)]


def joint_state(mb, unwrappers) -> tuple[np.ndarray, np.ndarray]:
    """``(q, qd)`` in radians from a polled `MotorBus`.

    `qd` is the DRIVER's speed field, direction-corrected.  It arrives in the
    same reply as the encoder, so it is free -- but on DOG5 it has been seen
    to LIE: it once reported 8.1 rad/s on a joint whose encoder had moved
    0.31 rad.  Anything that multiplies velocity by a Jacobian and calls the
    result a force should differentiate the encoder instead and keep this as
    the second witness.  `hw.safety.SafetyGate` already treats them that way.

    Raises RuntimeError if any motor has not answered yet -- a missing encoder
    is never silently substituted.
    """
    ids = motor_ids()                 # raises while the map is incomplete
    raw = [mb.rec(mid).encoder for mid in ids]
    missing = [mid for mid, value in zip(ids, raw) if value is None]
    if missing:
        raise RuntimeError(f"no encoder reply from CAN ids: {missing}")
    motoroutput_deg = np.asarray(
        [tracker.update(value) for tracker, value in zip(unwrappers, raw)]
    )
    q = motoroutput_deg_to_joint_rad(motoroutput_deg)
    speeds = mb.speeds_dps()          # MotorBus applies the same directions
    qd = np.deg2rad(np.asarray([speeds[mid] for mid in ids]))
    return q, qd


# ---------------------------------------------------------------------------
# validation and the set-zero write
# ---------------------------------------------------------------------------
def validate() -> None:
    """Check the contract is well-formed.  Raises on any drift.

    Runs on an INCOMPLETE map, because most of the contract does not depend on
    the wiring: the soft limits, the stance margin and the encoder gain are all
    checkable today.  Only the direction round-trip needs measured values, and
    it is skipped -- not faked -- while they are missing.
    """
    if N_JOINTS != C.N_JOINTS:
        raise ValueError(f"expected {C.N_JOINTS} joints, got {N_JOINTS}")
    if len(JOINT_LABELS) != N_JOINTS:
        raise ValueError("joint labels do not match the joint count")
    low, high = soft_limits()
    if low.shape != (N_JOINTS,) or high.shape != (N_JOINTS,):
        raise ValueError("soft limits are not (12,)")
    if not np.all(high > low):
        raise ValueError("soft limits are not ordered")
    # The stance must sit INSIDE the limits, with room -- this is the check
    # that would have caught porting DOG5's 1.75 rad abduction bound.
    stand = C.flat(C.Q_STAND)
    if not np.all((stand > low) & (stand < high)):
        raise ValueError("Q_STAND is outside params.JOINT_LIMITS")
    if not is_complete():
        return
    ids = motor_ids()
    if len(set(ids)) != N_JOINTS:
        raise ValueError(f"CAN ids must be unique: {ids}")
    if any(d not in (-1, +1) for d in motor_directions().values()):
        raise ValueError("every hardware direction must be +1 or -1")
    probe = np.linspace(-1.5, 1.5, N_JOINTS)
    if not np.allclose(
        motoroutput_deg_to_joint_rad(joint_rad_to_motoroutput_deg(probe)), probe
    ):
        raise ValueError("joint <-> motoroutput conversion does not round-trip")


def set_zero_all(mb, confirm: bool = False) -> dict:
    """HARDWARE set-zero (0x19): the CURRENT pose becomes every joint's zero.

    Pose the robot at the calibration pose FIRST -- flat on its belly, all
    four legs straight out fore-and-aft -- because that pose is what every
    angle in this project is measured from.  `coordinates.Q_ZERO` is it, and
    it is model/dog6.xml's `home` keyframe.

    Three things about 0x19 that are not negotiable:

      * the vendor warns that writing the encoder offset affects the driver's
        lifetime.  Do it when the robot is built or re-assembled, not as part
        of a run;
      * it takes effect only after a POWER CYCLE.  Read back about 0 on every
        joint afterwards to confirm;
      * it is not reversible from software.  There is no previous zero to go
        back to.

    Pass ``confirm=True`` to say you have read that.  Returns
    ``{can_id: encoder_offset or None}``; a None is a motor that never acked
    and whose zero was therefore NOT written.
    """
    if not confirm:
        raise RuntimeError(
            "set_zero_all writes the drivers' encoder offsets, takes effect "
            "only after a power cycle, and cannot be undone.  Pose the robot "
            "flat on its belly at coordinates.Q_ZERO, then call with "
            "confirm=True.")
    return mb.set_zero_all()


def describe() -> str:
    low, high = soft_limits()
    stand = C.flat(C.Q_STAND)
    return "\n".join([
        "DOG6 joint calibration contract",
        "  motoroutput_deg = raw * %.9f      (360/65535, OUTPUT shaft, no /10)"
        % ENCODER_GAIN,
        "  joint_rad       = radians(direction * motoroutput_deg)   no offset",
        "  zero pose       = coordinates.Q_ZERO, flat on the belly",
        "  soft limits     abd +-%.2f  pitch +-%.2f  knee +-%.2f rad"
        % (P.ABD_LIM, P.PITCH_LIM, P.KNEE_LIM),
        "  Q_STAND margin  %.3f rad to the nearest limit (DOG5's 1.75 abd "
        "bound would have left %.3f)"
        % (float(np.min(np.minimum(stand - low, high - stand))),
           1.75 - float(np.max(np.abs(stand[::3])))),
        "  estop margin    %.2f rad outside the soft limits" % LIMIT_ESTOP_MARGIN,
        "  direction       %s"
        % ("measured on all 12" if is_complete() else
           "NOT MEASURED -- every conversion here raises MapIncomplete"),
    ])


validate()


if __name__ == "__main__":
    print(describe())
