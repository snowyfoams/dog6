"""The walk's governor: the operator's SPACE, pressed by the law.

    WalkGovernor.update(now, roll_err, pitch_err, command, walking) -> Command
        the command to walk on this sweep: `command`, or ZERO while PAUSED

WHAT THE OPERATOR DID, 2026-10-06, and why it is here.  Walking
`hw.fold2_walk` for 17 minutes the operator watched the trunk and, whenever
it leaned or a foot slipped, pressed SPACE -- the keys to zero, the trot in
place -- until it was steady, then walked on.  "That's my control; how about
integrate it into the script, the script decides which time to stop the
command to trot in place for stable."  This object does that, every sweep,
on the signal the operator was reading.

THE SIGNAL IS THE LEAN, NOT THE TILT.  In place the trot rocks +-3-7 deg at
the gait frequency and recovers by itself; a tip-over is a lean that STAYS
on one side across a landing or two.  So the roll and pitch errors from the
setpoint go through a first-order low-pass at `lean_hz` (1.5 Hz: a quarter
of the 0.6 s trot's rocking passes, the lean passes whole), and the larger
of the two is `lean`.  On the 2026-10-06 log (fold2_walk_2.npz, 630 s of
trot, 104 s of it commanded): the in-place lean never passed 5 deg in 526
s; the operator's five SPACE moments read 4-12 deg of tilt; the fall read
7.8 deg of lean 0.4 s before the operator pressed SPACE.  A tilt-rate term
was tried on the same log and refused: the rocking's +-60 deg/s of rate
fires it hundreds of times.

THE RULE.  `lean` past `pause_deg` PAUSES: the command this object returns
is zero (the reference slews to rest in ~60 ms and the next swing is the
trot in place's, footstep.py's at-rest arc) and the keys are left as they
are.  Once `lean` has stayed under `calm_deg` for `calm_s` the operator's
command comes back, slewed by the reference as any press is.  A second
pause within `repause_s` of that resume means the command itself is what
tips the robot: the pause then CLEARS the keys (`clear_pending`, which the
hook that owns the keys reads), as SPACE would, and the operator is told to
press again.  On the log, 5 / 3 deg and 1.0 s pause 5 times in 104 s of
walking (3.0 s paused in all) and never in place.

NOT a trigger here: the swing foot's tracking error.  On the same log the
swing lands 15 mm (median) off its arc walking and 28 mm at the 99th
percentile; the fall's 24-29 mm is inside that.  The slip the operator
saw has no number in the log yet.
"""
from __future__ import annotations

import math

if __package__ in (None, ""):        # allow `python hw/balance/governor.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from . import config as cfg          # noqa: E402
from .trajectory import Command      # noqa: E402

__all__ = ["WalkGovernor"]

_ZERO = Command(vx=0.0, vy=0.0, yaw_rate=0.0)


class WalkGovernor:
    """SPACE, pressed by the law when the trunk leans.  One per walk."""

    def __init__(self, *, pause_deg: float = cfg.WALK_PAUSE_LEAN_DEG,
                 calm_deg: float = cfg.WALK_CALM_LEAN_DEG,
                 calm_s: float = cfg.WALK_CALM_S,
                 lean_hz: float = cfg.WALK_LEAN_HZ,
                 repause_s: float = cfg.WALK_REPAUSE_S) -> None:
        if not pause_deg > 0.0:
            raise ValueError("pause_deg must be positive (0 is the flag's "
                             "'off', which builds no governor)")
        if not 0.0 <= calm_deg < pause_deg:
            raise ValueError("calm_deg %.1f must be below pause_deg %.1f"
                             % (calm_deg, pause_deg))
        self.pause_deg = float(pause_deg)
        self.calm_deg = float(calm_deg)
        self.calm_s = float(calm_s)
        self.lean_hz = float(lean_hz)
        self.repause_s = float(repause_s)
        self.reset()

    def reset(self) -> None:
        """A fresh trot: nothing leaning, nothing paused, the counts kept."""
        self._lr = 0.0                 # low-passed roll error, rad
        self._lp = 0.0                 # low-passed pitch error, rad
        self._t = float("nan")
        #: deg, the lean this sweep (the larger of the two filtered errors).
        self.lean = 0.0
        self.paused = False
        self.t_pause = float("nan")
        self._t_calm = float("nan")
        self.t_resume = float("nan")
        #: Set by a pause that found the command itself at fault; the owner
        #: of the keys reads it once (`take_clear`) and zeroes them.
        self.clear_pending = False
        #: One-shot text for the operator; `take_notice`.
        self.notice: str | None = None
        if not hasattr(self, "pauses"):
            self.pauses = 0
            self.clears = 0
            self.paused_s = 0.0
            self.lean_peak = 0.0
            self.pause_lean_max = 0.0

    def _filter(self, now: float, roll_err: float, pitch_err: float) -> float:
        if not math.isfinite(self._t):
            self._lr, self._lp = float(roll_err), float(pitch_err)
        else:
            dt = min(max(float(now) - self._t, 0.0), 0.05)
            a = 1.0 - math.exp(-2.0 * math.pi * self.lean_hz * dt)
            self._lr += a * (float(roll_err) - self._lr)
            self._lp += a * (float(pitch_err) - self._lp)
        self._t = float(now)
        return math.degrees(max(abs(self._lr), abs(self._lp)))

    def update(self, now: float, roll_err: float, pitch_err: float,
               command: Command, walking: bool = True) -> Command:
        """One sweep.  `roll_err` / `pitch_err` are the trunk's roll and
        pitch FROM THE SETPOINT, rad; `command` the operator's; `walking`
        False means the keys are not reaching the reference anyway (the
        trot's exit is latched) -- the lean is still followed, nothing is
        decided."""
        now = float(now)
        prev = self._t
        self.lean = self._filter(now, roll_err, pitch_err)
        self.lean_peak = max(self.lean_peak, self.lean)
        if self.paused and math.isfinite(prev):
            self.paused_s += max(0.0, now - prev)
        if not walking:
            return command
        if not self.paused:
            if self.lean > self.pause_deg:
                self.paused = True
                self.t_pause = now
                self._t_calm = float("nan")
                self.pauses += 1
                self.pause_lean_max = max(self.pause_lean_max, self.lean)
                again = (math.isfinite(self.t_resume)
                         and now - self.t_resume < self.repause_s)
                if again:
                    self.clears += 1
                    self.clear_pending = True
                    self.notice = (
                        "walk PAUSED again %.1f s after it resumed: lean %.1f "
                        "deg (past %.0f).  The command tips it, so the keys "
                        "are CLEARED -- trotting in place; press again when "
                        "it is steady" % (now - self.t_resume, self.lean,
                                          self.pause_deg))
                else:
                    self.notice = (
                        "walk PAUSED: lean %.1f deg from the setpoint (past "
                        "%.0f) -- trotting in place until it has been under "
                        "%.0f deg for %.1f s, then the keys resume"
                        % (self.lean, self.pause_deg, self.calm_deg,
                           self.calm_s))
            return command if not self.paused else _ZERO
        # paused: the command is zero until it has been calm long enough
        if self.lean < self.calm_deg:
            if not math.isfinite(self._t_calm):
                self._t_calm = now
            elif now - self._t_calm >= self.calm_s:
                self.paused = False
                self.t_resume = now
                self.notice = ("walk RESUMED after %.1f s: lean under %.0f "
                               "deg for %.1f s" % (now - self.t_pause,
                                                   self.calm_deg, self.calm_s))
                return command
        else:
            self._t_calm = float("nan")
        return _ZERO

    def take_clear(self) -> bool:
        """True once per clearing pause: the keys' owner zeroes them."""
        pending, self.clear_pending = self.clear_pending, False
        return pending

    def take_notice(self) -> str | None:
        notice, self.notice = self.notice, None
        return notice

    def status(self) -> str:
        if self.paused:
            return "PAUSED by the governor (lean %.1f deg)" % self.lean
        return "lean %.1f" % self.lean

    def describe(self) -> str:
        return ("pauses the keys at %.0f deg of lean (roll/pitch from the "
                "setpoint, low-passed %.1f Hz), resumes after %.1f s under "
                "%.0f; a re-pause within %.0f s clears them"
                % (self.pause_deg, self.lean_hz, self.calm_s, self.calm_deg,
                   self.repause_s))

    def report(self) -> str:
        return ("  governor        %d pause%s%s, %.1f s paused in all, lean "
                "peak %.1f deg%s"
                % (self.pauses, "" if self.pauses == 1 else "s",
                   "" if not self.pauses else
                   " (lean at the pause up to %.1f deg)" % self.pause_lean_max,
                   self.paused_s, self.lean_peak,
                   "" if not self.clears else
                   "; %d cleared the keys" % self.clears))
