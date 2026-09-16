"""DETA10 adapter reporting attitude in DOG6's trunk frame.

Ported from DOG5's `src/IMU_sensor/imu_dog.py`, which ran on the robot.  One
structural change, described below, and it is not cosmetic.

TWO ROTATIONS, NOT ONE -- THE CHANGE FROM DOG5
    DOG5 folded everything between the sensor die and the trunk into a single
    hardcoded map plus two scalar offsets.  That worked because DOG5's board
    was bolted flat and aligned, so the only residual was a couple of degrees
    of mounting error that two offsets absorb.  It stops working the moment a
    board is mounted at an angle, and it has no place to put a yaw offset at
    all.

    DOG6 already has a home for the second rotation: `coordinates.R_BODY_IMU`,
    which is ALSO baked into model/dog6.xml as the imu site's quaternion, with
    `sim.selftest` gating that the two agree.  So the two are kept apart here:

        SENSOR_TO_FLU     a property of the DETA10.  It reports NED body axes
                          (x fwd, y RIGHT, z DOWN); this project is FLU
                          (x fwd, y LEFT, z UP).  FLU = NED turned 180 deg
                          about x.  Fixed, never measured, never calibrated.
        R_BODY_IMU        how the BOARD sits in DOG6.  Identity today and that
                          is a PLACEHOLDER -- DOG6's IMU has not been mounted.
                          When it is, this becomes a measurement and the MJCF's
                          imu site quat moves with it or `selftest` fails.

    The scalar roll/pitch offsets survive on top of both, because they absorb
    something neither rotation can: the difference between "flat on the floor"
    and "level", which includes the floor.  They are a trim, not a frame.

SIGN CONVENTIONS IN THE TRUNK (FLU) FRAME
        roll  > 0  -> right side DOWN   (left side rises)
        pitch > 0  -> nose DOWN
        yaw   > 0  -> nose swings LEFT  (CCW seen from above)

    NED -> FLU is a conjugation by Rx(180 deg):

        roll_body  =  roll_sensor
        pitch_body = -pitch_sensor
        yaw_body   = -heading_sensor
        rates      = (wx, -wy, -wz)_sensor

YAW IS EXPOSED AND UNTRUSTED.  It is magnetometer-based and the magnetometer
    sits next to twelve motors and a steel frame.  Fine for display; do NOT
    close a loop on it until it has been watched under power.  `yaw_rate` is
    different -- it is the gyro's wz, inertial, and fine for short-horizon
    relative heading.

NOTHING HERE HAS BEEN CHECKED ON DOG6.  The board is not mounted.  Verify the
    signs with a physical tilt test before trusting them in control, exactly
    as DOG5's `imu_frame_test.py` did: lift one side, watch the sign.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):        # allow `python hw/imu.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402

# The fdilink_imu vendor SDK is not pip-installable and is not vendored here.
# Install it on the robot host.  The import is guarded so this module stays
# importable on a machine with no SDK -- most call sites want only the frame
# maths or DEFAULT_PORT, and a hard import would block every offline test for
# the sake of one constant.  Opening a real device still fails loudly.
try:
    from fdilink_imu import DETA10, DEFAULT_BAUD, AHRSData   # noqa: F401
    _FDILINK_IMPORT_ERROR = None
except ImportError as _exc:          # no SDK on this host
    DETA10 = None
    AHRSData = object
    DEFAULT_BAUD = None
    _FDILINK_IMPORT_ERROR = _exc

DEFAULT_PORT = "/dev/fdilink_imu"
DEFAULT_CALIB_PATH = Path(__file__).resolve().parent / "imu_calib.json"

__all__ = ["ImuDog", "TrunkAttitude", "TrunkOrientation", "SENSOR_TO_FLU",
           "SENSOR_TO_TRUNK", "sensor_to_trunk", "trunk_rotation", "wrap_deg",
           "MAX_AGE_S", "DEFAULT_PORT", "DEFAULT_CALIB_PATH", "describe"]

#: How old the last packet may be before the balance law stops believing it.
#: The DETA10 streams at 200 Hz on its own clock, so one missed packet is
#: 5 ms and this is ten of them.  `hw.balance` freezes its attitude terms
#: rather than running them on a stale error -- see `ImuDog.orientation`.
MAX_AGE_S = 0.05

#: R_{FLU <- NED}: the DETA10's own axis convention against this project's.
#: Rx(180 deg).  A property of the sensor, not of the mounting -- see the
#: module docstring.  Its own inverse.
SENSOR_TO_FLU = np.diag([1.0, -1.0, -1.0])

#: R_{trunk <- sensor}: the two composed.  `coordinates.R_BODY_IMU` is the
#: mounting rotation and is an identity PLACEHOLDER until the board is bolted
#: on and measured.
SENSOR_TO_TRUNK = C.R_BODY_IMU @ SENSOR_TO_FLU


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = math.fmod(a + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


@dataclass(frozen=True)
class TrunkAttitude:
    """Attitude in DOG6's trunk frame (x fwd, y left, z up)."""

    roll_deg: float          # >0 = right side down; mounting trim removed
    pitch_deg: float         # >0 = nose DOWN; mounting trim removed
    yaw_deg: float           # >0 = nose left; magnetometer -- UNTRUSTED near
                             # the motors, no trim applied; display only
    roll_rate_dps: float
    pitch_rate_dps: float
    yaw_rate_dps: float      # gyro wz -- inertial, OK for relative heading
    age_s: float             # host time since the packet arrived
    raw: object              # untouched sensor packet (NED sensor frame)


def sensor_to_trunk(roll_deg, pitch_deg, heading_deg, angular_rates):
    """Frame transform only, no mounting trim.  Returns deg and deg/s.

    Takes the DETA10's own NED-frame numbers and returns trunk-frame ones.
    Split out as a free function so it can be tested without a device, and so
    `capture_offsets` and `sample` cannot drift apart.
    """
    roll = roll_deg
    pitch = -pitch_deg
    yaw = wrap_deg(-heading_deg)
    rates = np.rad2deg(SENSOR_TO_TRUNK @ np.asarray(angular_rates, dtype=float))
    return roll, pitch, yaw, float(rates[0]), float(rates[1]), float(rates[2])


def trunk_rotation(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """R_{world <- trunk} from a TRUNK-frame ZYX triple in RADIANS.

    This is the second half of the chain the module docstring lays out, and
    the mounting rotation enters it on the OPPOSITE SIDE from the gyro's:

        omega^b = R_BODY_IMU @ SENSOR_TO_FLU @ omega^s        (left)
        R       = Rzyx(roll, pitch, yaw) @ R_BODY_IMU^T       (right)

    The asymmetry is not a choice and cannot be folded into one matrix.
    `omega` is a VECTOR expressed in the board's frame, so carrying it into
    the trunk is a rotation on the left.  The triple passed in here already
    describes a frame chain that TERMINATES in the board's frame -- it is the
    ZYX triple of R_{world <- imu} -- so reaching the trunk means appending
    the imu -> trunk leg on the right, which is R_BODY_IMU^T.

    AT R_BODY_IMU = I THE TWO SIDES AGREE EXACTLY, which is why nothing
    downstream can see the difference today.  When the board is bolted on and
    the rotation becomes a measurement, both sides start mattering in the
    same instant; that is the reason it is written out here rather than left
    implicit in `SENSOR_TO_TRUNK`.

    THE TRIM IS APPLIED BEFORE THIS, NOT AFTER.  `TrunkAttitude` subtracts the
    roll/pitch offsets from the imu-frame triple, so the trim is a rotation of
    R_{world <- imu} and this function carries the trimmed value through.  At
    an identity mounting that is the same thing as trimming the trunk; at a
    real one they differ by the mounting rotation, and the trim is small
    enough that it belongs on whichever side the operator captured it on.
    """
    return C.rot_zyx(roll_rad, pitch_rad, yaw_rad) @ C.R_BODY_IMU.T


@dataclass(frozen=True)
class TrunkOrientation:
    """What the balance law consumes: SI, one rotation, one rate vector.

    `TrunkAttitude` is the human-facing view -- degrees, six scalars, a yaw
    that is labelled untrusted.  This is the control-facing one, and the
    difference is deliberate rather than cosmetic: a law that takes an Euler
    triple in degrees will sooner or later subtract two of them and call the
    result an attitude error.  Nothing here can be subtracted.
    """

    R: np.ndarray            # (3, 3)  world <- trunk, `coordinates.rot_zyx`
    omega_b: np.ndarray      # (3,)    rad/s in the TRUNK frame -- the gyro's
    roll: float              # rad, trimmed.  For the tilt trip and the log,
    pitch: float             # rad, trimmed.  NOT for the control law.
    yaw: float               # rad, magnetometer -- UNTRUSTED, display only
    age_s: float             # host seconds since the packet arrived

    @property
    def omega_w(self) -> np.ndarray:
        """The same rate in the WORLD frame, which is what the SRB law uses."""
        return self.R @ self.omega_b

    def is_stale(self, max_age_s: float = MAX_AGE_S) -> bool:
        return self.age_s > max_age_s

    @property
    def tilt_deg(self) -> float:
        """The larger of |roll| and |pitch| in degrees -- what trips."""
        return float(np.degrees(max(abs(self.roll), abs(self.pitch))))

    @staticmethod
    def level() -> "TrunkOrientation":
        """A perfectly level, motionless, infinitely fresh trunk.

        FOR OFFLINE TESTS AND FOR THE NO-IMU PATH ONLY.  `age_s` is zero, so
        `is_stale` says fresh: anything that accepts this is asserting it does
        not need an IMU, and `hw.stand --no-imu` is the only caller that does.
        """
        return TrunkOrientation(np.eye(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0)


class ImuDog:
    """Wraps DETA10; reports roll/pitch in the trunk frame with staleness."""

    def __init__(self, port: str = DEFAULT_PORT, baud=None,
                 calib_path: Path = DEFAULT_CALIB_PATH):
        if DETA10 is None:
            raise ImportError(
                "the fdilink_imu vendor SDK is required to talk to the IMU but "
                "was not found on this host.  It is not pip-installable; "
                "install it on the robot.  Importing hw.imu without it is "
                "supported -- the constants and the frame maths work -- but "
                "opening a device is not."
            ) from _FDILINK_IMPORT_ERROR
        self._imu = DETA10(port, DEFAULT_BAUD if baud is None else baud)
        self._calib_path = Path(calib_path)
        self.roll_offset_deg = 0.0
        self.pitch_offset_deg = 0.0
        self._last_ahrs = None
        self._last_rx_mono: float = 0.0
        self._imu.on_ahrs(self._on_ahrs)
        self.load_calib()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ImuDog":
        self._imu.start()
        return self

    def stop(self) -> None:
        self._imu.stop()

    def __enter__(self) -> "ImuDog":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        return self._imu.wait_for_data(timeout)

    # -- stream health -----------------------------------------------------
    @property
    def rate_hz(self) -> float:
        return self._imu.rate_hz

    @property
    def crc_error_count(self) -> int:
        return self._imu.crc_error_count

    def age_s(self) -> float:
        """Host time since the last AHRS packet (inf before the first one)."""
        if self._last_rx_mono == 0.0:
            return float("inf")
        return time.monotonic() - self._last_rx_mono

    def is_stale(self, max_age_s: float = 0.05) -> bool:
        return self.age_s() > max_age_s

    # -- attitude ----------------------------------------------------------
    def _on_ahrs(self, a) -> None:
        self._last_ahrs = a
        self._last_rx_mono = time.monotonic()

    def sample(self) -> Optional[TrunkAttitude]:
        a = self._last_ahrs
        if a is None:
            return None
        roll, pitch, yaw, wx, wy, wz = sensor_to_trunk(
            a.roll_deg, a.pitch_deg, a.heading_deg, a.angular_rates)
        return TrunkAttitude(
            roll_deg=wrap_deg(roll - self.roll_offset_deg),
            pitch_deg=pitch - self.pitch_offset_deg,
            yaw_deg=yaw,
            roll_rate_dps=wx,
            pitch_rate_dps=wy,
            yaw_rate_dps=wz,
            age_s=self.age_s(),
            raw=a,
        )

    def orientation(self) -> Optional[TrunkOrientation]:
        """`sample()` in SI: R, omega^b in rad/s.  None before the first packet.

        WHAT THE ADAPTER OWED THE CONTROL LAW, AND THIS IS IT.  Nothing new is
        measured here -- the same packet, the same trim, the same frame chain.
        What changes is the type: degrees become radians, three scalar rates
        become one vector, and the triple becomes the rotation matrix the law
        actually multiplies by.

        DOES NOT BLOCK AND DOES NOT WAIT.  The IMU streams on its own clock,
        not on the control sweep, so the caller gets the most recent packet
        whatever its age and `age_s` says what that age is.  A sweep that
        waits for a packet is a sweep that misses its CAN deadline.
        """
        attitude = self.sample()
        if attitude is None:
            return None
        roll = math.radians(attitude.roll_deg)
        pitch = math.radians(attitude.pitch_deg)
        yaw = math.radians(attitude.yaw_deg)
        return TrunkOrientation(
            R=trunk_rotation(roll, pitch, yaw),
            omega_b=np.deg2rad([attitude.roll_rate_dps,
                                attitude.pitch_rate_dps,
                                attitude.yaw_rate_dps]),
            roll=roll, pitch=pitch, yaw=yaw,
            age_s=attitude.age_s,
        )

    # -- the mounting TRIM (not the mounting rotation) ---------------------
    def capture_offsets(self, duration_s: float = 1.0) -> Optional[tuple]:
        """Average trunk-frame roll/pitch for `duration_s` -> the trim.

        Call with the robot flat on a LEVEL floor.  Circular mean on roll, so
        it is safe right at the wrap point.  Returns (roll, pitch) or None if
        no fresh data arrived.

        THIS IS NOT `coordinates.R_BODY_IMU`.  What it absorbs is the residual
        after the mounting rotation -- board skew of a degree or two, plus
        whatever the floor is doing.  A board mounted at a real angle needs
        R_BODY_IMU set (and model/dog6.xml's imu site quat with it); trimming
        it away here would leave the simulator and the robot disagreeing by
        that angle, with `selftest` still passing.
        """
        sin_r = cos_r = 0.0
        pitch_sum = 0.0
        n = 0
        last_ts = None
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            a = self._last_ahrs
            if a is not None and a.timestamp_us != last_ts:
                last_ts = a.timestamp_us
                roll, pitch, *_ = sensor_to_trunk(
                    a.roll_deg, a.pitch_deg, a.heading_deg, a.angular_rates)
                sin_r += math.sin(math.radians(roll))
                cos_r += math.cos(math.radians(roll))
                pitch_sum += pitch
                n += 1
            time.sleep(0.002)
        if n == 0:
            return None
        self.roll_offset_deg = math.degrees(math.atan2(sin_r, cos_r))
        self.pitch_offset_deg = pitch_sum / n
        return self.roll_offset_deg, self.pitch_offset_deg

    def clear_offsets(self) -> None:
        self.roll_offset_deg = 0.0
        self.pitch_offset_deg = 0.0

    def save_calib(self) -> Path:
        data = {
            "roll_offset_deg": self.roll_offset_deg,
            "pitch_offset_deg": self.pitch_offset_deg,
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frame_note": "trim in the DOG6 TRUNK frame (x fwd, y left, z up), "
                          "applied AFTER coordinates.R_BODY_IMU",
            "robot": "DOG6",
        }
        self._calib_path.write_text(json.dumps(data, indent=2) + "\n")
        return self._calib_path

    def load_calib(self) -> bool:
        try:
            data = json.loads(self._calib_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        self.roll_offset_deg = float(data.get("roll_offset_deg", 0.0))
        self.pitch_offset_deg = float(data.get("pitch_offset_deg", 0.0))
        return True


def describe() -> str:
    mounted = not np.allclose(C.R_BODY_IMU, np.eye(3))
    trim = DEFAULT_CALIB_PATH.exists()
    return "\n".join([
        "DOG6 IMU adapter (DETA10 -> trunk frame)",
        "  sensor axes     NED (x fwd, y right, z down)",
        "  trunk axes      FLU (x fwd, y left, z up)",
        "  SENSOR_TO_FLU   diag%s   Rx(180 deg), a sensor property"
        % (tuple(int(v) for v in np.diag(SENSOR_TO_FLU)),),
        "  R_BODY_IMU      %s"
        % ("measured" if mounted else "IDENTITY -- PLACEHOLDER, board not mounted"),
        "  SENSOR_TO_TRUNK %s" % np.array2string(SENSOR_TO_TRUNK,
                                                 precision=3).replace("\n", "\n"
                                                                      + " " * 18),
        "  trim file       %s" % (DEFAULT_CALIB_PATH if trim else
                                  "%s (absent -- no trim captured)"
                                  % DEFAULT_CALIB_PATH.name),
        "  vendor SDK      %s" % ("present" if DETA10 is not None else
                                  "absent (%s) -- frame maths still works"
                                  % type(_FDILINK_IMPORT_ERROR).__name__),
        "",
        "  Signs are UNVERIFIED on DOG6.  Tilt the robot and watch them before",
        "  closing any loop.  Yaw is magnetometer-based: display only.",
    ])


if __name__ == "__main__":
    print(describe())
