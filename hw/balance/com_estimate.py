"""The built robot's CoM, measured in every HOLD and applied at T.

    com_from_hold(q, tau_meas, roll, pitch) -> (c_xy LEVEL frame, kt)
        one sweep's statics -- what `hw.com_check` reads off a log
    com_sample(body, tau_meas, z_b) -> (c_xy TRUNK frame, kt)
        the same, online, from the sweep's `state.BodyState`
    ComMeasure
        one HOLD's samples: `start` at each HOLD, `add` every sweep, `summary`
        at T

WHY ONLINE, 2026-10-06, on request.  `hw.com_check` read the CoM off a HOLD
log and printed a `--com-offset` to fly the next run with.  The operator
found every HOLD gives its own number ("each hold will bring the new com
shift"), so a number measured on one run is the wrong one for the next.
Here every HOLD measures its own, and T (`sequence.StandSequence`) applies
it: the SRB pin and the stance sites move together by (measured - pinned),
as `--com-offset` does at start-up, and the trunk slides over the four
planted feet onto it before the trot clock starts.  The log carries the
samples (`com_meas`) and the pin flown every sweep (`com_pin`), so the run
is read straight off the npz.

THE STATICS -- `hw.com_check`'s, unchanged.  In HOLD the robot rests on four
feet, so the ground reactions balance the weight about the CoM.  Each foot's
force is the torque the drivers MEASURED (the q-axis current) through the
leg's Jacobian, less the leg's own weight:

    f_i^b = -(J_i^T)^-1 (tau_i - g_i(q_i, R))         trunk frame
    sum_i (R x_i) x (R f_i) = c x F_z z_hat   ->   c_x = -M_y / F_z,  c_y = M_x / F_z

in the LEVEL trunk frame (roll and pitch taken out, yaw left in).  The CoM
along gravity has no moment and is not observable; `com_sample` takes it
from the pin and solves for the trunk-frame x/y the law's c^b is written in.
F_z over the weight is `kt`: what fraction of it the drivers' torque
constant accounts for.  The CoM is a RATIO of moment to force, so a torque
constant wrong by the same factor on every driver does not move it; joint
friction and a single driver's readback do, and show up as spread.
"""
from __future__ import annotations

import math

import numpy as np

from sim import coordinates as C

from .. import kinematics as HK
from . import config as cfg
from . import torque as TRQ

__all__ = ["level_rotation", "com_from_feet", "com_from_hold",
           "level_to_trunk_xy", "com_sample", "ComMeasure"]

#: The sweep rate the sample counts are sized at (`hw.stand.RATE_HZ`).
RATE_HZ = 250.0


def level_rotation(roll: float, pitch: float) -> np.ndarray:
    """(3, 3) trunk -> level frame: the trunk's roll and pitch, no yaw (ZYX)."""
    return C.rot_y(float(pitch)) @ C.rot_x(float(roll))


def com_from_feet(x_b, jac, tau_meas, gravity, R) -> tuple[np.ndarray, float]:
    """``(c_xy, kt)``: the CoM in the frame `R` turns the trunk into, m, and
    the measured vertical force over the weight.  `x_b` (4, 3) feet from the
    trunk origin, `jac` (4, 3, 3), `tau_meas` (12,), `gravity` (4, 3) each
    leg's own weight torque, `R` (3, 3) with z up.  NaN and 0 when the
    measured vertical force is not positive."""
    # ALL FOUR LEGS IN ONE BATCHED SOLVE: per leg in a loop this was 298 us
    # on the Pi, a whole CAN slot (`hw.stand.COM_SLOT`).
    rhs = (np.asarray(tau_meas, dtype=float).reshape(C.N_LEGS, 3)
           - np.asarray(gravity, dtype=float).reshape(C.N_LEGS, 3))
    jt = np.transpose(np.asarray(jac, dtype=float), (0, 2, 1))
    try:
        f_b = -np.linalg.solve(jt, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:     # a leg at its singularity says nothing
        return np.full(2, np.nan), 0.0
    R = np.asarray(R, dtype=float)
    f_w = f_b @ R.T
    r_w = np.asarray(x_b, dtype=float) @ R.T
    # sum_i r_i x f_i, written out: two components are all the CoM needs.
    mx = float(np.dot(r_w[:, 1], f_w[:, 2]) - np.dot(r_w[:, 2], f_w[:, 1]))
    my = float(np.dot(r_w[:, 2], f_w[:, 0]) - np.dot(r_w[:, 0], f_w[:, 2]))
    fz = float(f_w[:, 2].sum())
    if not (fz > 1e-6 and math.isfinite(fz)):
        return np.full(2, np.nan), 0.0
    # sum r x f = c x (F_z z_hat) = F_z (c_y, -c_x, 0)
    return np.array([-my / fz, mx / fz]), fz / cfg.WEIGHT


def com_from_hold(q, tau_meas, roll, pitch) -> tuple[np.ndarray, float]:
    """``(c_xy, kt)`` for ONE sweep: the CoM in the LEVEL trunk frame, m, and
    the measured vertical force over the weight.  `q`, `tau_meas` (12,);
    angles rad.  `hw.com_check`'s function, which imports it from here."""
    R = level_rotation(roll, pitch)
    q = np.asarray(q, dtype=float)
    q4 = C.unflat(q)
    x_b = np.array([HK.foot_position(i, q4[i]) for i in range(C.N_LEGS)])
    jac = np.array([HK.foot_jacobian(i, q4[i]) for i in range(C.N_LEGS)])
    return com_from_feet(x_b, jac, tau_meas,
                         TRQ.all_leg_gravity_torque(q, R), R)


def level_to_trunk_xy(c_level, roll: float, pitch: float,
                      z_b: float) -> np.ndarray:
    """(2,) the trunk-frame CoM x/y whose level-frame x/y is `c_level`, with
    its trunk z taken as `z_b` (the pin's -- not observable from statics).
    Under a degree of tilt the two differ by z_b * tilt: ~0.5 mm."""
    R = level_rotation(roll, pitch)
    bx = float(c_level[0]) - R[0, 2] * float(z_b)
    by = float(c_level[1]) - R[1, 2] * float(z_b)
    # the 2x2 solve, written out (R's leading 2x2 is near the identity)
    a, b, c, d = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    det = a * d - b * c
    return np.array([(d * bx - b * by) / det, (a * by - c * bx) / det])


def com_sample(body, tau_meas, z_b: float,
               gravity=None) -> tuple[np.ndarray, float]:
    """``(c_xy, kt)`` from one sweep's `state.BodyState` and the drivers'
    measured torque: the CoM x/y in the TRUNK frame -- the frame the SRB
    pin's c^b is written in -- with its z taken as `z_b`.  NaN and 0 when
    the sweep says nothing.  `gravity` (4, 3) is the legs' own weight torque
    at this sweep's q and attitude if the caller already has it (the law
    computed it in slot 0), else it is computed here."""
    R = level_rotation(body.roll, body.pitch)
    if gravity is None:
        gravity = TRQ.all_leg_gravity_torque(body.q, R)
    c_level, kt = com_from_feet(body.x_b, body.jac, tau_meas, gravity, R)
    if not np.all(np.isfinite(c_level)):
        return c_level, kt
    return level_to_trunk_xy(c_level, body.roll, body.pitch, z_b), kt


class ComMeasure:
    """One HOLD's CoM: started at each HOLD, a sample a sweep, read at T.

    Samples in the first `settle_s` of the HOLD are dropped; the newest
    `window_s` are kept; `ready` once `min_s` of them are in.  Reading it is
    a median, so one sweep with a driver's reply missing or a foot unloading
    for a moment costs nothing."""

    def __init__(self, settle_s: float = cfg.COM_SETTLE_S,
                 min_s: float = cfg.COM_MIN_S,
                 window_s: float = cfg.COM_WINDOW_S,
                 rate_hz: float = RATE_HZ) -> None:
        self.settle_s = float(settle_s)
        self.min_n = max(1, int(round(float(min_s) * rate_hz)))
        self.max_n = max(self.min_n, int(round(float(window_s) * rate_hz)))
        self._c = np.full((self.max_n, 2), np.nan)
        self._kt = np.full(self.max_n, np.nan)
        self.t0 = float("nan")
        self.n = 0                    # samples taken this HOLD (ring: <= max_n kept)
        self.t_first = float("nan")
        self.t_last = float("nan")
        #: The latest sample, accepted or not -- the log's `com_meas`.
        self.last = np.full(2, np.nan)

    def start(self, now: float) -> None:
        """A HOLD begins: forget the last one's samples."""
        self.t0 = float(now)
        self.n = 0
        self.t_first = self.t_last = float("nan")
        self.last = np.full(2, np.nan)

    @property
    def started(self) -> bool:
        return math.isfinite(self.t0)

    def add(self, now: float, c_xy, kt: float) -> bool:
        """One sweep's sample; False if it was not kept (before the settle,
        or not finite)."""
        c_xy = np.asarray(c_xy, dtype=float)
        self.last = c_xy.copy()
        if not self.started or float(now) - self.t0 < self.settle_s:
            return False
        if not (np.all(np.isfinite(c_xy)) and math.isfinite(float(kt))):
            return False
        k = self.n % self.max_n
        self._c[k] = c_xy
        self._kt[k] = float(kt)
        self.n += 1
        if not math.isfinite(self.t_first):
            self.t_first = float(now)
        self.t_last = float(now)
        return True

    @property
    def kept(self) -> int:
        return min(self.n, self.max_n)

    @property
    def ready(self) -> bool:
        return self.kept >= self.min_n

    def seconds_to_ready(self, now: float) -> float:
        """Roughly how long until `ready`, s (0 once it is)."""
        if self.ready:
            return 0.0
        if not self.started:
            return float("inf")
        settle_left = max(0.0, self.t0 + self.settle_s - float(now))
        return settle_left + (self.min_n - self.kept) / RATE_HZ

    def summary(self) -> dict:
        """The HOLD's reading: median and p10/p90 of x/y (m), median kt, the
        sample count and the seconds they span.  Empty dict before any."""
        k = self.kept
        if k == 0:
            return {}
        c = self._c[:k]
        kt = self._kt[:k]
        # Once the ring has wrapped the oldest kept sample is no longer the
        # first one taken: the span is the window's.
        seconds = (float(self.t_last - self.t_first) if self.n <= self.max_n
                   else self.max_n / RATE_HZ)
        return dict(c=np.median(c, axis=0),
                    p10=np.percentile(c, 10, axis=0),
                    p90=np.percentile(c, 90, axis=0),
                    kt=float(np.median(kt)), n=k, seconds=seconds)
