"""Trunk velocity and position by integrating the DETA10's accelerometer.

Ported from DOG5's Bloesch EKF (`state_estimator/dog5_state_estimator.py`)
and its IMU bridge (`state_estimator/imu_ekf_feed.py`), both of which ran on
the robot.  What is taken is the PREDICTION HALF of that filter -- the part
that turns specific force into velocity -- and its two conventions:

    static-hold init   the gyro mean is the gyro bias; the accelerometer mean
                       minus gravity is the accelerometer bias.  Collected
                       ONLY while the robot is still (DOG5's `quiet_stages`
                       rule: init samples taken during motion poison the bias)
                       and DOG5's count, 30 samples.
    propagation        a^w = R f^b + g^w,  v <- v + dt a   (DOG5 eqs. 3
                       and 31), with DOG5's dt clamp.
    position           r <- r + dt v + dt^2/2 a            (DOG5 eq. 30),
                       in DOG5's inertial frame: heading fixed at init,
                       carried by the integrated gyro.

WHAT IS NOT TAKEN: THE FILTER'S OWN ATTITUDE
    DOG5's EKF integrated the gyro for its attitude and let the leg updates
    hold roll/pitch.  Without those updates a gyro attitude walks, and a
    walking attitude is the worst error this integrator can have: a tilt
    error delta leaks g*sin(delta) of gravity into the horizontal axes, 0.17
    m/s^2 per degree.  So attitude is the AHRS's, exactly as DOG5's
    `ekf_feedback` kept it ("attitude stays the AHRS's").  The AHRS roll and
    pitch are fused by the SAME chip from the SAME accelerometer, so the
    specific force and the gravity direction it is compensated with are in
    one frame by construction.  That is also why the UNTRIMMED attitude is
    used: the trim in `hw.imu` absorbs the floor, and gravity does not care
    about the floor.

WHY THE STATE IS A BODY-FRAME VELOCITY, AND YAW NEVER ENTERS
    The loop's world is yaw-free (DOG5's feedback built C from roll/pitch
    alone; `hw.imu` labels yaw display-only).  Integrating in a yaw-free frame
    is wrong the moment the robot turns -- that frame turns with it.  So the
    state is v^b, the trunk velocity in the TRUNK frame, whose dynamics need
    only the gravity DIRECTION in the trunk and the gyro:

        vdot^b = (f^b - b_f) + R^T g^w - omega^b x v^b

    R^T g^w depends on roll and pitch alone.  The yaw-free world velocity is
    then read off as R_rp v^b -- the same composition as DOG5's
    `v_loop = C_loop^T (C_ekf v_I)`, with the EKF's unobservable yaw never
    having been there to discard.  The rotation term is integrated exactly
    (one Rodrigues step per sample), with the acceleration applied
    at the midpoint of the turn.

dt IS THE SENSOR'S CLOCK, NOT THE HOST'S
    Every 0x40 packet carries the DETA10's own microsecond timestamp.  DOG5
    measured host-side packet age at 0-28 ms through a 250 Hz loop: arrival
    time is jitter, and an integrator multiplies every sample by its dt.
    DOG5 averaged per worker tick, which hid that; this integrates per sample,
    so it uses the sensor's dt and clamps it with DOG5's bounds.  A gap longer
    than the upper clamp is integrated as the clamp, not as the gap.

POSITION NEEDS A HEADING, AND IT IS THE GYRO'S
    A position cannot live in the yaw-free frame -- that frame turns with the
    robot, so a walked square would not close.  It lives in the ODOMETRY frame
    `o`: z up, x along the trunk's heading at init, origin at the trunk at
    init.  That is DOG5's EKF inertial frame exactly ("heading is wherever the
    robot pointed at initialisation, and drifts with the integrated gyro").
    The heading is NOT the magnetometer's yaw, which `hw.imu` labels display
    only.  Each step the full attitude is carried forward by the same exact
    gyro step as v^b, its yaw read off, and roll/pitch replaced by the AHRS:

        R_o <- Rz(psi) R_rp        psi from  R_o exp([w dt]x)

    The position step is the trapezoid of v^o across the sample, which for a
    constant acceleration over dt is DOG5's eq. 30 to the letter.

PURE INTEGRATION DRIFTS, AND NOTHING HERE CORRECTS IT
    A residual accelerometer bias of b m/s^2 becomes b*t of velocity and
    b*t^2/2 of position.  DOG6's first live run left about 3 mm/s^2 after
    init: 0.15 m of position after 10 s, 5.4 m after a minute.  A gyro bias
    residual turns the heading, which bends every later metre of r.  That is the physics of the method, and
    it is why DOG5 ran this propagation inside an EKF with leg updates.  The
    integrator exposes `zero()` for a caller that KNOWS the trunk is still
    (a zero-velocity update), `set_velocity()` and `set_position()` for one
    that has a better measurement; deciding when either is true belongs in `hw.balance`, not
    here.

THE INIT GATE
    At rest the accelerometer mean must equal R^T(-g^w) up to a bias.  A
    residual larger than `INIT_RESIDUAL_MAX` means the frame chain is broken
    -- a sign flipped between the raw stream and the AHRS -- and `initialise`
    refuses rather than absorbing a gravity-sized error into a "bias".

RUN
    python -m hw.velocity_estimator            keep the robot STILL for the
                                               first ~0.2 s, then move it
"""
from __future__ import annotations

import collections
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):        # allow `python hw/velocity_estimator.py`
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import imu as IMU             # noqa: E402

__all__ = ["ImuSample", "AccelVelocityEstimator", "ImuRawFeed",
           "gravity_in_trunk", "G", "N_INIT_SAMPLES", "DT_CLAMP",
           "INIT_RESIDUAL_MAX"]

#: m/s^2.  DOG5's `GRAVITY` magnitude.
G = 9.81

#: DOG5's `DEFAULT_N_INIT_SAMPLES`.  0.15 s at the DETA10's 200 Hz.
N_INIT_SAMPLES = 30

#: DOG5's `DEFAULT_DT_CLAMP`, seconds.
DT_CLAMP = (1.0e-4, 0.05)

#: m/s^2.  The largest at-rest |f - R^T(-g)| `initialise` will call a bias.
#: A sign error in the frame chain at a few degrees of tilt is a residual of
#: order 2g sin(tilt) or 2g outright; a genuine DETA10 bias is tens of mg.
INIT_RESIDUAL_MAX = 0.5


def gravity_in_trunk(roll_rad: float, pitch_rad: float) -> np.ndarray:
    """g^b = R^T g^w for the yaw-free R.  At level: (0, 0, -G)."""
    R = IMU.trunk_rotation(roll_rad, pitch_rad, 0.0)
    return R.T @ np.array([0.0, 0.0, -G])


@dataclass(frozen=True)
class ImuSample:
    """One raw packet in the TRUNK frame, paired with the AHRS it arrived under.

    f_b      specific force, m/s^2.  At rest and level (0, 0, +G).
    w_b      angular rate, rad/s.
    roll     rad, UNTRIMMED AHRS -- see the module docstring for why.
    pitch    rad, UNTRIMMED AHRS.
    t_s      the SENSOR's clock, seconds since its power-up.
    """

    f_b: np.ndarray
    w_b: np.ndarray
    roll: float
    pitch: float
    t_s: float


def _rodrigues(phi: np.ndarray) -> np.ndarray:
    """exp([phi]x): the rotation by angle |phi| about phi."""
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        return np.eye(3)
    k = phi / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


class AccelVelocityEstimator:
    """v^b by integrating (f - b_f) + g^b under the gyro, and r^o from v.

    No I/O, no clock.

    Not thread-safe by design: one caller drains the feed and steps this.
    """

    def __init__(self, n_init: int = N_INIT_SAMPLES,
                 dt_clamp: tuple = DT_CLAMP,
                 residual_max: float = INIT_RESIDUAL_MAX):
        self.n_init = int(n_init)
        self.dt_clamp = dt_clamp
        self.residual_max = float(residual_max)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Forget everything, biases included.  Re-initialise from rest."""
        self.initialised = False
        self.b_f = np.zeros(3)
        self.b_w = np.zeros(3)
        self.v_b = np.zeros(3)
        self.r_o = np.zeros(3)        # position in the odometry frame, m
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0                # gyro heading in the odometry frame
        self.t_last: Optional[float] = None
        self.elapsed_s = 0.0          # integrated time since init / last zero
        self._init: list[ImuSample] = []

    def collect(self, samples: Sequence[ImuSample]) -> bool:
        """Feed samples taken while the robot is STILL; initialise when enough.

        The caller asserts stillness -- DOG5's `quiet_stages` -- because
        nothing in the IMU alone can tell a still robot from a steadily moving
        one.  Returns `initialised`.
        """
        if self.initialised:
            return True
        self._init.extend(samples)
        if len(self._init) >= self.n_init:
            self.initialise(self._init)
        return self.initialised

    def initialise(self, samples: Sequence[ImuSample]) -> None:
        """Biases from a static hold; v, r, yaw := 0.  Raises on a broken
        frame chain."""
        if len(samples) == 0:
            raise ValueError("initialise needs at least one static sample")
        f = np.mean([s.f_b for s in samples], axis=0)
        w = np.mean([s.w_b for s in samples], axis=0)
        g_b = np.mean([gravity_in_trunk(s.roll, s.pitch) for s in samples],
                      axis=0)
        b_f = f + g_b                  # at rest a = 0, so f = -g^b + b_f
        if float(np.linalg.norm(b_f)) > self.residual_max:
            self._init = []
            raise ValueError(
                "at rest the accelerometer disagrees with the AHRS gravity by "
                "%.2f m/s^2 (limit %.2f): f_b=%s, -g_b=%s.  That is a frame "
                "error, not a bias -- or the robot was not still."
                % (np.linalg.norm(b_f), self.residual_max,
                   np.round(f, 3), np.round(-g_b, 3)))
        self.b_f = b_f
        self.b_w = w
        self.v_b = np.zeros(3)
        self.r_o = np.zeros(3)
        self.yaw = 0.0
        self.roll, self.pitch = samples[-1].roll, samples[-1].pitch
        self.t_last = samples[-1].t_s
        self.elapsed_s = 0.0
        self._init = []
        self.initialised = True

    # -- propagation -------------------------------------------------------
    def step(self, s: ImuSample) -> None:
        """Integrate one sample.  Ignored until initialised."""
        if not self.initialised:
            return
        if self.t_last is None:
            self.t_last = s.t_s
            return
        dt = s.t_s - self.t_last
        self.t_last = s.t_s
        if dt <= 0.0:                  # duplicate or reordered packet
            return
        dt = min(max(dt, self.dt_clamp[0]), self.dt_clamp[1])

        a_b = (s.f_b - self.b_f) + gravity_in_trunk(s.roll, s.pitch)
        w_b = s.w_b - self.b_w
        v_o_old = self.v_o
        # Over dt the trunk turns by exp(w dt); a vector fixed in the world
        # is seen from the new trunk as exp(w dt)^T times the old one.  That
        # is the -omega x v term, integrated exactly.  The acceleration is
        # felt across the whole step, so it is carried by HALF the turn
        # (midpoint rule): adding it before the full turn grows |v| by
        # (1 + dt^2 w^2 / 2) a step, and a circle stops closing.
        half = _rodrigues(w_b * (0.5 * dt))
        step = half @ half
        self.v_b = step.T @ self.v_b + half.T @ (a_b * dt)
        R_o = self.R_o @ step
        self.yaw = math.atan2(R_o[1, 0], R_o[0, 0])
        self.roll, self.pitch = s.roll, s.pitch
        self.r_o = self.r_o + 0.5 * dt * (v_o_old + self.v_o)
        self.elapsed_s += dt

    def update(self, samples: Sequence[ImuSample]) -> None:
        for s in samples:
            self.step(s)

    # -- external knowledge --------------------------------------------------
    def zero(self) -> None:
        """Zero-velocity update: the CALLER knows the trunk is still."""
        self.v_b = np.zeros(3)
        self.elapsed_s = 0.0

    def set_velocity(self, v_w: np.ndarray) -> None:
        """Overwrite with a yaw-free-world velocity measured some other way."""
        self.v_b = self.R_rp.T @ np.asarray(v_w, dtype=float)
        self.elapsed_s = 0.0

    def set_position(self, r_o=(0.0, 0.0, 0.0)) -> None:
        """Move the trunk to `r_o` in the odometry frame; heading untouched."""
        self.r_o = np.asarray(r_o, dtype=float).copy()

    # -- outputs -------------------------------------------------------------
    @property
    def R_rp(self) -> np.ndarray:
        """R_{yaw-free world <- trunk} at the last integrated sample."""
        return IMU.trunk_rotation(self.roll, self.pitch, 0.0)

    @property
    def v_w(self) -> np.ndarray:
        """Trunk velocity in the loop's yaw-free world, m/s."""
        return self.R_rp @ self.v_b

    @property
    def R_o(self) -> np.ndarray:
        """R_{odometry <- trunk}: the AHRS roll/pitch under the gyro heading."""
        return IMU.trunk_rotation(self.roll, self.pitch, self.yaw)

    @property
    def v_o(self) -> np.ndarray:
        """Trunk velocity in the odometry frame, m/s."""
        return self.R_o @ self.v_b


class ImuRawFeed:
    """The raw 0x40 stream of an EXISTING `ImuDog`, buffered as `ImuSample`s.

    DOG5's `ImuEkfFeed`: one serial port, two readers -- the AHRS keeps
    flowing to `ImuDog.orientation()` while this taps the raw packet on the
    same DETA10.  The callback runs on the reader thread and only converts
    and appends; the control side `drain()`s and steps the estimator.
    """

    def __init__(self, imu: "IMU.ImuDog", maxlen: int = 4096):
        self.imu = imu
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._last: Optional[ImuSample] = None
        self.imu._imu.on_imu(self._on_imu)

    def _on_imu(self, d) -> None:
        a = self.imu._last_ahrs        # the AHRS this packet arrived under
        if a is None:
            return
        roll_deg, pitch_deg, *_ = IMU.sensor_to_trunk(
            a.roll_deg, a.pitch_deg, a.heading_deg, a.angular_rates)
        s = ImuSample(
            f_b=IMU.SENSOR_TO_TRUNK @ np.asarray(d.accel, dtype=float),
            w_b=IMU.SENSOR_TO_TRUNK @ np.asarray(d.gyro, dtype=float),
            roll=math.radians(roll_deg),
            pitch=math.radians(pitch_deg),
            t_s=d.timestamp_us * 1e-6,
        )
        with self._lock:
            self._buf.append(s)
            self._last = s

    def drain(self) -> list:
        """Every buffered sample since the last drain, oldest first."""
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
        return out

    def wait_for_raw(self, timeout: float = 3.0) -> bool:
        """Block until a 0x40 packet has arrived -- `ImuDog.wait_for_data`
        returns on ANY packet, usually the AHRS, which proves nothing about
        whether the raw stream is enabled on the sensor."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._last is not None:
                return True
            time.sleep(0.005)
        return self._last is not None


def main() -> int:
    with IMU.ImuDog() as imu:
        feed = ImuRawFeed(imu)
        if not feed.wait_for_raw(3.0):
            print("no raw 0x40 packet in 3 s -- is the IMU stream enabled?")
            return 1
        est = AccelVelocityEstimator()
        print("hold the robot STILL ...")
        while not est.collect(feed.drain()):
            time.sleep(0.01)
        print("initialised: b_f=%s m/s^2  b_w=%s rad/s"
              % (np.round(est.b_f, 4), np.round(est.b_w, 5)))
        print("Ctrl-C to stop.")
        try:
            while True:
                time.sleep(0.1)
                est.update(feed.drain())
                v, r = est.v_w, est.r_o
                print("\rv_w = (%+.3f, %+.3f, %+.3f) m/s   "
                      "r_o = (%+.3f, %+.3f, %+.3f) m   yaw %+6.1f deg   "
                      "t=%5.1f s"
                      % (v[0], v[1], v[2], r[0], r[1], r[2],
                         math.degrees(est.yaw), est.elapsed_s),
                      end="", flush=True)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
