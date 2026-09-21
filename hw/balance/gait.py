"""The trot clock.  Time in, contact schedule out.  No robot state at all.

    TrotGait.reset(now)               start the clock at the all-four window
    TrotGait.contact_weight(t)        -> (4,) the share of load each foot may carry
    TrotGait.swing_phase(t)           -> (4,) progress through the current swing
    TrotGait.full_support(t)          -> all four down at FULL weight

DOG5's trot_demo clock, in cMPC's arithmetic.  Nothing in here was designed
for DOG6:

    from DOG5 (flown)   the contact-weight ramp inside stance, the four-foot
                        SETTLE every few cycles -- `trot_demo.SettleTrotGait`
                        and `gait.TrotGait`, numbers in `config` tagged
                        [DOG5 FLOWN].  NOT its alternating lead: removed on
                        request 2026-09-21, one fixed trot, the diagonals
                        strictly taking turns
    from sim.cmpc       the phase itself: the offset is added IN SECONDS,
                        before the modulo.  `sim.cmpc.gait.phase` says why --
                        `mod(t/T + 0.5, 1)` rounds 0.49999999999999994 + 0.5
                        up to 1.0 and reports a foot about to lift as one that
                        has just landed

A PURE TIMETABLE, AS BOTH OF THEM WERE
    There is no contact sensor on DOG6.  A foot is in stance because the
    clock says so, and the allocator is told what the clock says.  Early and
    late touchdown are not detected, and nothing here pretends otherwise.

THE WARP, AND WHY EVERYTHING GOES THROUGH phase()
    The settle is a FLAT in the map from wall time to gait time: once every
    `settle_every` cycles, gait time stops for `settle_s` at the point where
    all four feet carry full weight.  `contact`, `contact_weight` and
    `swing_phase` all read `phase`, so the freeze reaches every one of them
    and no caller has to know a settle exists.  Outside the flats the warp is
    1:1, so swing and stance durations are exactly the unwarped ones.

WHERE THE CLOCK STARTS -- THE ONE THING NOT COPIED FROM DOG5
    DOG5's `reset(now)` put the offset-0 diagonal at phase 0: the start of
    stance, contact weight 0.  From HOLD, where every foot carried weight 1,
    that is a one-sweep step of a whole diagonal's load, which DOG5's 60 N*m/s
    slew then absorbed.  `reset` here starts the clock at the freeze point
    instead -- mid the all-four window, every weight already 1 -- so entering
    the trot moves nothing.  The first liftoff follows `entry_s` later.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/gait.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from . import config as cfg          # noqa: E402

__all__ = ["TrotGait", "GaitSample", "smoothstep"]


class GaitSample(NamedTuple):
    """Everything the law reads from the clock at one instant."""

    contact: np.ndarray      # (4,) bool
    weight: np.ndarray       # (4,) in [0, 1]
    swing_s: np.ndarray      # (4,) swing progress, 0 in stance
    full_support: bool       # all four down at full weight


def smoothstep(u):
    """3u^2 - 2u^3 on [0, 1] -- DOG5's contact ramp shape."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class TrotGait:
    """The contact schedule, a pure function of t once `reset` has been called."""

    def __init__(self, period: float = cfg.GAIT_PERIOD, duty: float = cfg.DUTY,
                 offsets=cfg.PHASE_OFFSET, ramp: float = cfg.CONTACT_RAMP,
                 settle_s: float = cfg.SETTLE_S,
                 settle_every: int = cfg.SETTLE_EVERY):
        if not period > 0.0:
            raise ValueError("period must be positive, got %r" % period)
        if not 0.5 < duty < 1.0:
            raise ValueError(
                "duty %r has no all-four window: at <= 0.5 the contact ramp "
                "has nowhere to hand the load across, and the settle would "
                "hold a foot that is meant to swing" % duty)
        if settle_s < 0.0:
            raise ValueError("settle_s must be >= 0, got %r" % settle_s)
        if int(settle_every) != settle_every or settle_every < 1:
            raise ValueError("settle_every must be a whole number of cycles "
                             ">= 1, got %r" % settle_every)
        self.period = float(period)
        self.duty = float(duty)
        self.offsets = np.mod(np.asarray(offsets, dtype=float)
                              .reshape(C.N_LEGS), 1.0)
        self.ramp = float(ramp)
        self.settle_s = float(settle_s)
        self.settle_every = int(settle_every)
        #: The freeze point, gait seconds after a cycle starts: the MIDDLE of
        #: the all-four window that follows the cycle's second touchdown.
        self.freeze_s = (self.duty - 0.5) / 2.0 * self.period
        self._t0 = 0.0
        # MEASURED, not assumed (trot_demo's guard): at the freeze point every
        # leg must be planted at FULL weight, or the settle is a lean and the
        # clock's start is a step.
        w = self._weight_from_phase(self._raw_phase(self.freeze_s))
        if not bool(np.all(w >= 1.0 - 1e-9)):
            raise ValueError(
                "freeze point %.3f s is not full four-foot support (duty %.2f, "
                "ramp %.2f): weights %s"
                % (self.freeze_s, self.duty, self.ramp, np.round(w, 3)))

    # -- the clock ---------------------------------------------------------
    def reset(self, now: float, half: bool = False) -> None:
        """Start the clock so that `now` is the freeze point of cycle 0.

        All four feet are at full weight at that instant -- see the module
        docstring for why this and not DOG5's phase-0 start.  The FR/RL
        diagonal lifts first.  `half` starts half a cycle later instead --
        the same four-foot instant, but FL/RR lifts first -- which is how
        `--half-gait` hands the next swing to the other diagonal.
        """
        self._t0 = (float(now) - self.freeze_s
                    - (0.5 * self.period if half else 0.0))

    @property
    def entry_s(self) -> float:
        """Seconds from `reset` to the first liftoff.

        The later diagonal sits at phase freeze/T + 0.5 at the freeze point
        and lifts at duty, which is (duty - 0.5)/2 of a cycle away -- the same
        distance the freeze point is from the cycle start."""
        return self.freeze_s

    def _raw_phase(self, gait_s: float) -> np.ndarray:
        """(4,) phase at `gait_s` gait seconds.  THE OFFSET IN SECONDS, before
        the modulo -- `sim.cmpc.gait.phase`."""
        shifted = float(gait_s) + self.offsets * self.period
        return np.mod(shifted, self.period) / self.period

    # -- the settle bookkeeping (trot_demo.SettleTrotGait) -----------------
    def settle_of(self, n: int) -> float:
        """Cycle n's settle, seconds.  Cycle 0 never settles."""
        return (self.settle_s
                if n > 0 and n % self.settle_every == 0 else 0.0)

    def cycle_start(self, n: int) -> float:
        """Wall seconds since `reset`'s zero at which cycle n begins."""
        served = (n - 1) // self.settle_every if n > 0 else 0
        return n * self.period + served * self.settle_s

    def _cycle_index(self, elapsed: float) -> int:
        """Largest n with cycle_start(n) <= elapsed."""
        if elapsed <= 0.0:
            return 0
        block = self.settle_every * self.period + self.settle_s
        n = self.settle_every * int(elapsed // block)
        while n > 0 and self.cycle_start(n) > elapsed:
            n -= 1
        while self.cycle_start(n + 1) <= elapsed:
            n += 1
        return n

    def _warp(self, elapsed: float) -> float:
        """Wall seconds -> gait seconds: slope 1, one flat per settling cycle."""
        n = self._cycle_index(elapsed)
        e = elapsed - self.cycle_start(n)
        settle = self.settle_of(n)
        if e < self.freeze_s:
            w = e
        elif e < self.freeze_s + settle:
            w = self.freeze_s
        else:
            w = e - settle
        return n * self.period + w

    # -- what the law reads ------------------------------------------------
    def phase(self, t: float) -> np.ndarray:
        """(4,) in [0, 1).  0 is touchdown, `duty` is liftoff."""
        return self._raw_phase(self._warp(float(t) - self._t0))

    def contact(self, t: float) -> np.ndarray:
        """(4,) bool.  STRICTLY below duty: the liftoff instant is swing."""
        return self.phase(t) < self.duty

    def swing_phase(self, t: float) -> np.ndarray:
        """(4,) progress through swing in [0, 1); exactly 0 for a stance leg."""
        ph = self.phase(t)
        return np.where(ph < self.duty, 0.0,
                        (ph - self.duty) / (1.0 - self.duty))

    def _weight_from_phase(self, ph) -> np.ndarray:
        s = np.clip(ph / self.duty, 0.0, 1.0)
        r = max(self.ramp, 1e-9)
        w = smoothstep(np.minimum(s / r, (1.0 - s) / r))
        return np.where(ph < self.duty, w, 0.0)

    def contact_weight(self, t: float) -> np.ndarray:
        """(4,) in [0, 1]: 1 through mid-stance, smoothstepped over the first
        and last `ramp` of stance, exactly 0 in swing.  The ramp is INSIDE
        stance, so a foot is already unloaded when the clock lifts it."""
        return self._weight_from_phase(self.phase(t))

    def sample(self, t: float) -> GaitSample:
        """The four quantities the law needs, from ONE evaluation of the
        warp.  The individual methods each re-run it; in a 333 us slot that
        is ~100 us of the same arithmetic four times."""
        ph = self.phase(t)
        contact = ph < self.duty
        weight = self._weight_from_phase(ph)
        swing_s = np.where(contact, 0.0, (ph - self.duty) / (1.0 - self.duty))
        return GaitSample(contact=contact, weight=weight, swing_s=swing_s,
                          full_support=bool(np.all(weight >= 1.0 - 1e-9)))

    def full_support(self, t: float) -> bool:
        """All four feet down at full weight -- where the trot may be left."""
        return bool(np.all(self.contact_weight(t) >= 1.0 - 1e-9))

    def settling(self, t: float) -> bool:
        """True while the clock is frozen for a re-level."""
        elapsed = float(t) - self._t0
        n = self._cycle_index(elapsed)
        e = elapsed - self.cycle_start(n)
        return bool(self.freeze_s <= e < self.freeze_s + self.settle_of(n))

    def cycles(self, t: float) -> int:
        """Whole gait cycles completed since `reset`."""
        return self._cycle_index(float(t) - self._t0)

    @property
    def stance_duration(self) -> float:
        return self.period * self.duty

    @property
    def swing_duration(self) -> float:
        return self.period * (1.0 - self.duty)

    def __repr__(self) -> str:
        return ("TrotGait(period %.2f s, duty %.2f -> swing %.0f ms, ramp "
                "%.2f, settle %.2f s every %d cycle%s)"
                % (self.period, self.duty, 1e3 * self.swing_duration,
                   self.ramp, self.settle_s, self.settle_every,
                   "" if self.settle_every == 1 else "s"))


if __name__ == "__main__":
    g = TrotGait()
    g.reset(0.0)
    print(g)
    print("  t (s)   " + "  ".join("%-5s" % leg for leg in C.LEGS))
    for t in np.arange(0.0, 2 * g.period + g.settle_s, 0.1):
        w = g.contact_weight(t)
        print("  %5.2f   %s%s" % (t, "  ".join("%5.2f" % x for x in w),
                                   "   settle" if g.settling(t) else ""))
