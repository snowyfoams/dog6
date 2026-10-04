"""The operator's keys while walking: W/S x, A/D y, Q/E yaw, SPACE stop.

    WalkKeys.press(ch)      -> str | None, what to print; None if not ours
    WalkKeys.command        the accumulated `trajectory.Command`, body axes

`sim.cmpc.run`'s mapping, key for key, so the hands that drove the model
drive the robot:

    W / S     forward / back      +- WALK_V_STEP     (sim: 0.1 m/s)
    A / D     left / right        +- WALK_V_STEP
    Q / E     turn left / right   +- WALK_YAW_STEP   (sim: 20 deg/s)
    SPACE     stop: all three zero

The keys ACCUMULATE a velocity (a key has no magnitude); the hardware box
clips it here, so what the status line says is what the reference gets.
The command reaches the reference only while the robot trots and is SLEWED
there (`trajectory.WalkReference`), so a press is a ramp, never a step.

X (e-stop), T (trot) and ENTER (phases) stay `hw.stand`'s.  W is the foot
step on `hw.trot`; an entry point that walks has no foot step, and this
object OWNS W so the runner does not offer it to the step too.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/keys.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from . import config as cfg          # noqa: E402
from .trajectory import Command, clip_command   # noqa: E402

__all__ = ["WalkKeys", "KEYMAP"]

#: key -> (axis, sign).  Lower case; `press` folds the case.
KEYMAP = {
    "w": ("vx", +1.0), "s": ("vx", -1.0),
    "a": ("vy", +1.0), "d": ("vy", -1.0),
    "q": ("yaw_rate", +1.0), "e": ("yaw_rate", -1.0),
}


class WalkKeys:
    """The accumulated walking command, and the keys that move it."""

    def __init__(self, *, v_step: float = cfg.WALK_V_STEP,
                 yaw_step: float = cfg.WALK_YAW_STEP,
                 vx_max: float = cfg.WALK_VX_MAX,
                 vy_max: float = cfg.WALK_VY_MAX,
                 yaw_rate_max: float = cfg.WALK_YAW_RATE_MAX):
        self.v_step, self.yaw_step = float(v_step), float(yaw_step)
        self.vx_max, self.vy_max = float(vx_max), float(vy_max)
        self.yaw_rate_max = float(yaw_rate_max)
        self.command = Command(vx=0.0, vy=0.0, yaw_rate=0.0)

    def owns(self, ch) -> bool:
        """True for the keys this object answers, whatever the case."""
        return ch is not None and (ch == " " or ch.lower() in KEYMAP)

    def stop(self) -> None:
        self.command = Command(vx=0.0, vy=0.0, yaw_rate=0.0)

    def press(self, ch) -> str | None:
        """Apply one key.  Returns the line to print, or None if not ours."""
        if not self.owns(ch):
            return None
        if ch == " ":
            self.stop()
            return "walk: STOP  (%s)" % self.describe_command()
        axis, sign = KEYMAP[ch.lower()]
        step = self.yaw_step if axis == "yaw_rate" else self.v_step
        c = self.command
        values = dict(vx=c.vx, vy=c.vy, yaw_rate=c.yaw_rate)
        values[axis] += sign * step
        # Snap a near-zero sum to zero: 0.05 + -0.05 in floats is 1e-17, and
        # a status line reading "-0.00 m/s" is a question nobody should ask.
        if abs(values[axis]) < 1e-9:
            values[axis] = 0.0
        self.command = clip_command(Command(**values), self.vx_max,
                                    self.vy_max, self.yaw_rate_max)
        return "walk: %s" % self.describe_command()

    def describe_command(self) -> str:
        c = self.command
        return ("vx %+.2f  vy %+.2f m/s  yaw %+.0f deg/s"
                % (c.vx, c.vy, np.degrees(c.yaw_rate)))

    @staticmethod
    def help_line() -> str:
        return ("W/S x +-%.2f  A/D y +-%.2f m/s  Q/E yaw +-%.0f deg/s  "
                "SPACE stop" % (cfg.WALK_V_STEP, cfg.WALK_V_STEP,
                                np.degrees(cfg.WALK_YAW_STEP)))


if __name__ == "__main__":
    keys = WalkKeys()
    for ch in "wwwqqdx ":
        said = keys.press(ch)
        print("%r -> %s" % (ch, said))
