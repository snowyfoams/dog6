"""The operator's stick, when the stick is a real one.

    Gamepad().poll().command() -> trajectory.Command

WHAT CHANGES WHEN THE KEYBOARD BECOMES A GAMEPAD, AND IT IS NOT THE WIRING
    `run`'s W/S/A/D/Q/E are an INCREMENTAL command: each press adds
    `cfg.V_STEP` to a value that persists until the next press.  That is the
    only thing a key can do -- a key has no magnitude, so the magnitude has to
    live somewhere, and it lives in an accumulator.

    A thumbstick has a magnitude.  So the mapping here is ABSOLUTE: stick
    centred is zero, stick at the rim is `cfg.V_MAX`, and letting go of the
    stick commands a stop rather than leaving the last accumulated value in
    place.  Nothing accumulates and nothing latches.  That is a genuinely
    different operator model, and it is the safer one: the failure mode of the
    keyboard scheme is a robot still walking because nobody pressed SPACE.

    The consequence for the REFERENCE is nil.  `trajectory.ReferenceTrajectory`
    integrates whatever velocity it is handed; it neither knows nor cares
    whether the number came from an accumulator or a potentiometer.

THE BACKEND IS XINPUT, VIA ctypes, WITH NO PACKAGE TO INSTALL
    An Xbox Wireless Controller paired over Bluetooth is presented by Windows
    as an XInput device, and `XInputGetState` is two structs and one call into
    a DLL this interpreter can already load.  pygame/SDL would also work and
    is used if it happens to be installed (it is the only option off Windows),
    but requiring it for four axes would be the larger dependency.

    XInput exposes exactly four slots, 0-3.  A pad that was on slot 0 and is
    switched off comes back on whichever slot is free, so `poll` re-scans --
    but only once a second, because querying an EMPTY slot is a device-layer
    round trip costing on the order of 100 us, and doing four of them inside a
    25 ms MPC period for a pad that is simply not there is real time wasted.

THE DEADZONE IS RADIAL AND IT RESCALES, WHICH IS NOT COSMETIC EITHER
    The pad on this desk reads (1786, -593) with nobody touching it -- 5.5 %
    of full scale, drift, entirely normal for a stick with a potentiometer in
    it.  Fed straight through, that is a standing sideways command and a robot
    that wanders across the room while the operator watches it do nothing.

    Two ways to kill it, and the obvious one is wrong.  Per-AXIS deadzones
    (|x| < dz -> 0, independently of y) carve a square hole out of the stick's
    disc, so a stick pushed straight up and rolled slightly left snaps its vy
    to zero and back -- the diagonals get quantised.  The deadzone here is on
    the stick's RADIUS: the pair is killed together or passed together.

    And what is passed is RESCALED, (r - dz) / (1 - dz), not passed raw.  Raw,
    the first millimetre of stick past the deadzone is a 24 % step change in
    commanded velocity -- the reference jumps, and on a robot tracking it that
    is a visible lurch.  Rescaled, the command leaves zero continuously.

    The thresholds are Microsoft's own recommended constants, which is why
    they are oddly specific: 7849 of 32767 on the left stick, 8689 on the
    right (the right stick's springs are looser and it drifts more).

EXPO, BECAUSE THE INTERESTING SPEEDS ARE THE SLOW ONES
    DOG6 trots to 0.4 m/s and falls near 0.5, so `cfg.V_MAX` of 0.6 is already
    past the envelope and the whole useful range of the stick is its inner
    half.  A linear map spends most of the travel above what the robot can do.
    `_shape` blends linear and cubic -- the output is still exactly +-1 at the
    rim, but the gain near centre is about a third of linear, so walking-pace
    commands get most of the stick.

SIGNS, ONCE, HERE, SO THEY ARE NOT REDISCOVERED DOWNSTREAM
    XInput's sticks are +X right, +Y up.  `trajectory.Command` is body axes:
    +vx forward, +vy LEFT, +yaw_rate counter-clockwise seen from above.  So

        vx       = +LY     stick up      -> forward
        vy       = -LX     stick left    -> left
        yaw_rate = -RX     stick right   -> turn right (clockwise, negative)

    The two negations are the whole reason this module is more than a struct
    read: a right stick pushed right must turn the robot right, and "right" in
    a counter-clockwise-positive convention is a MINUS sign.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field, replace

import numpy as np

from . import config as cfg
from .trajectory import Command

# ===========================================================================
# stick shaping
# ===========================================================================
#: Microsoft's recommended radial deadzones, in raw counts of 32767.
DEADZONE_L = 7849.0 / 32767.0                        # 0.240
DEADZONE_R = 8689.0 / 32767.0                        # 0.265

#: Trigger counts below this read as released (of 255).
TRIGGER_DEADZONE = 30.0 / 255.0

#: How much of `_shape` is cubic.  0 is linear, 1 is pure cube.  0.65 puts the
#: centre gain at 0.35 of linear and leaves the rim at exactly 1.0.
EXPO = 0.65

#: Left trigger, fully pulled, multiplies every command by this.  A precision
#: modifier for placing the robot, not a mode -- it is continuous, so half a
#: trigger is half the cut.
PRECISION_SCALE = 0.35

#: How long a vanished pad is left alone before the empty slots are re-scanned.
RESCAN_PERIOD = 1.0                                  # s

# ===========================================================================
# XInput
# ===========================================================================
_ERROR_SUCCESS = 0
_ERROR_DEVICE_NOT_CONNECTED = 1167

BUTTONS = {
    "DPAD_UP": 0x0001, "DPAD_DOWN": 0x0002,
    "DPAD_LEFT": 0x0004, "DPAD_RIGHT": 0x0008,
    "START": 0x0010, "BACK": 0x0020,
    "LEFT_THUMB": 0x0040, "RIGHT_THUMB": 0x0080,
    "LB": 0x0100, "RB": 0x0200,
    "A": 0x1000, "B": 0x2000, "X": 0x4000, "Y": 0x8000,
}


class _XInputGamepad(ctypes.Structure):
    _fields_ = [("wButtons", wintypes.WORD),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD),
                ("Gamepad", _XInputGamepad)]


def _load_xinput():
    """The newest XInput present.  1_4 ships with Windows 8 and later."""
    if not hasattr(ctypes, "WinDLL"):                  # not Windows
        return None, ""
    for name in ("XInput1_4.dll", "XInput1_3.dll", "XInput9_1_0.dll"):
        try:
            return ctypes.WinDLL(name), name
        except OSError:
            continue
    return None, ""


# ===========================================================================
# raw stick -> shaped stick
# ===========================================================================
def _shape(value: float) -> float:
    """Linear/cubic blend.  Exactly +-1 at +-1, gentler in the middle."""
    return (1.0 - EXPO) * value + EXPO * value ** 3


def _stick(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """Radial deadzone, rescaled, shaped, clamped to the unit disc.

    The stick comes back as a direction times a magnitude in [0, 1] -- the
    MAGNITUDE is shaped, the direction is not, so a diagonal push stays
    diagonal at every deflection.
    """
    r = float(np.hypot(x, y))
    if r < deadzone:
        return 0.0, 0.0
    # Clip the rescaled radius BEFORE shaping: a stick held into the corner of
    # its gate reads r slightly over 1 (the gate is square, the travel is
    # round) and would otherwise leave here above full scale.
    magnitude = _shape(min((r - deadzone) / (1.0 - deadzone), 1.0))
    return x * magnitude / r, y * magnitude / r


def _axis(raw: int) -> float:
    """Raw short -> [-1, 1].  -32768 over 32767 is clamped, not wrapped."""
    return max(-1.0, min(1.0, raw / 32767.0))


def _trigger(value: float) -> float:
    value = max(0.0, min(1.0, value))
    if value < TRIGGER_DEADZONE:
        return 0.0
    return (value - TRIGGER_DEADZONE) / (1.0 - TRIGGER_DEADZONE)


# ===========================================================================
# one frame of the pad
# ===========================================================================
@dataclass(frozen=True)
class PadState:
    """One poll of the pad, after shaping.  Every stick axis is in [-1, 1]."""

    connected: bool = False
    slot: int = -1
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    buttons: frozenset = frozenset()
    pressed: frozenset = frozenset()      # edges: went down since the last poll
    released: frozenset = frozenset()

    def command(self, z: float = cfg.Z_REF) -> Command:
        """The shaped sticks as a body-axis velocity command.

        ABSOLUTE, not incremental -- see the module docstring.  `clipped` is
        belt and braces: the scaling already cannot exceed the ceilings.
        """
        scale = 1.0 - (1.0 - PRECISION_SCALE) * self.lt
        return Command(
            vx=cfg.V_MAX * scale * self.ly,
            # The + 0.0 is not padding: negating a stick that reads exactly
            # zero gives -0.0, which prints as "-0.00" on a stopped robot and
            # reads like a command nobody gave.  -0.0 + 0.0 is +0.0.
            vy=cfg.V_MAX * scale * -self.lx + 0.0,
            yaw_rate=cfg.YAW_RATE_MAX * scale * -self.rx + 0.0,
            z=z,
        ).clipped()

    def __str__(self) -> str:
        if not self.connected:
            return "no pad"
        return ("slot %d  L(%+.2f %+.2f)  R(%+.2f %+.2f)  LT %.2f RT %.2f  %s"
                % (self.slot, self.lx, self.ly, self.rx, self.ry,
                   self.lt, self.rt, " ".join(sorted(self.buttons)) or "-"))


# ===========================================================================
# the pad itself
# ===========================================================================
@dataclass
class Gamepad:
    """An Xbox pad, polled on demand.  Absent hardware is not an error.

    `poll()` is safe to call at any rate; it never waits on anything.  With no
    pad connected it returns a zeroed `PadState` whose `.connected` is False,
    so a caller can offer a fallback (`run` keeps the keyboard, `teleop`
    offers mouse-dragged virtual sticks) without testing for hardware first.
    """

    slot: int = -1
    backend: str = ""
    _lib: object = field(default=None, repr=False)
    _pygame_joy: object = field(default=None, repr=False)
    _state: object = field(default=None, repr=False)
    _buttons: frozenset = field(default=frozenset(), repr=False)
    _next_scan: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._lib, self.backend = _load_xinput()
        if self._lib is not None:
            self._state = _XInputState()
            self.slot = self._scan()
            if self.slot >= 0:
                return
        joy = _open_pygame()
        if joy is not None:
            self._pygame_joy, self.backend = joy, "pygame"
            self.slot = 0

    # -- XInput slot bookkeeping -------------------------------------------
    def _scan(self) -> int:
        """The lowest connected slot, or -1.  Rate-limited -- see the docstring."""
        now = time.monotonic()
        if now < self._next_scan:
            return -1
        self._next_scan = now + RESCAN_PERIOD
        for i in range(4):
            if self._lib.XInputGetState(i, ctypes.byref(self._state)) \
                    == _ERROR_SUCCESS:
                return i
        return -1

    # -- polling -----------------------------------------------------------
    def poll(self) -> PadState:
        if self._pygame_joy is not None:
            state = self._poll_pygame()
        elif self._lib is not None:
            state = self._poll_xinput()
        else:
            state = PadState()
        pressed = state.buttons - self._buttons
        released = self._buttons - state.buttons
        self._buttons = state.buttons
        return replace(state, pressed=frozenset(pressed),
                       released=frozenset(released))

    def _poll_xinput(self) -> PadState:
        if self.slot < 0:
            self.slot = self._scan()
            if self.slot < 0:
                return PadState()
        rc = self._lib.XInputGetState(self.slot, ctypes.byref(self._state))
        if rc == _ERROR_DEVICE_NOT_CONNECTED:
            # Do not hunt for it here: the pad has just gone away and the next
            # scan is a second out.  One zeroed frame stops the robot, which
            # is the correct response to the operator's pad dying.
            self.slot = -1
            return PadState()
        if rc != _ERROR_SUCCESS:
            return PadState()

        pad = self._state.Gamepad
        lx, ly = _stick(_axis(pad.sThumbLX), _axis(pad.sThumbLY), DEADZONE_L)
        rx, ry = _stick(_axis(pad.sThumbRX), _axis(pad.sThumbRY), DEADZONE_R)
        return PadState(
            connected=True, slot=self.slot,
            lx=lx, ly=ly, rx=rx, ry=ry,
            lt=_trigger(pad.bLeftTrigger / 255.0),
            rt=_trigger(pad.bRightTrigger / 255.0),
            buttons=frozenset(n for n, m in BUTTONS.items() if pad.wButtons & m),
        )

    def _poll_pygame(self) -> PadState:
        import pygame
        pygame.event.pump()
        joy = self._pygame_joy
        # SDL's axis 1 and 4 point DOWN where XInput's point up; the rest of
        # this module is written in XInput's convention, so flip here and
        # nowhere else.
        lx, ly = _stick(joy.get_axis(0), -joy.get_axis(1), DEADZONE_L)
        rx, ry = _stick(joy.get_axis(3), -joy.get_axis(4), DEADZONE_R)
        names = ("A", "B", "X", "Y", "LB", "RB", "BACK", "START",
                 "LEFT_THUMB", "RIGHT_THUMB")
        held = {names[i] for i in range(min(joy.get_numbuttons(), len(names)))
                if joy.get_button(i)}
        return PadState(
            connected=True, slot=0, lx=lx, ly=ly, rx=rx, ry=ry,
            lt=_trigger((joy.get_axis(2) + 1.0) / 2.0),
            rt=_trigger((joy.get_axis(5) + 1.0) / 2.0),
            buttons=frozenset(held),
        )


def _open_pygame():
    """A joystick through pygame, if pygame is installed and one is present."""
    try:
        import pygame
    except ImportError:
        return None
    try:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return None
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    except Exception:                     # pragma: no cover - driver-dependent
        return None


# ===========================================================================
# looking at it
# ===========================================================================
BINDINGS = """\
  left stick    vx / vy      up = forward, left = left      +-%.2f m/s
  right stick   yaw rate     right = turn right (CW)        +-%.0f deg/s
  left trigger  precision    fully pulled = x%.2f
  A             stop         sticks are absolute; this is the extra insurance
  B             reset        pose back to the stand keyframe, reference re-anchored
  START         quit"""


def bindings() -> str:
    return BINDINGS % (cfg.V_MAX, np.rad2deg(cfg.YAW_RATE_MAX), PRECISION_SCALE)


def describe() -> str:
    pad = Gamepad()
    state = pad.poll()
    return "\n".join([
        "DOG6 gamepad",
        "  backend         %s" % (pad.backend or "none"),
        "  connected       %s" % ("slot %d" % state.slot
                                  if state.connected else "no"),
        "  deadzone        L %.3f  R %.3f  (radial, rescaled)"
        % (DEADZONE_L, DEADZONE_R),
        "  expo            %.2f  -> centre gain %.2f of linear"
        % (EXPO, 1.0 - EXPO),
        "",
        bindings(),
        "",
        "  stick -> command:",
        "    L up      full   %s" % PadState(ly=1.0).command(),
        "    L up      half   %s" % PadState(ly=0.5).command(),
        "    L left    full   %s" % PadState(lx=-1.0).command(),
        "    R right   full   %s" % PadState(rx=1.0).command(),
        "    L up full, LT in  %s" % PadState(ly=1.0, lt=1.0).command(),
    ])


def monitor(hz: float = 30.0) -> int:
    """Print the live pad on one line.  Ctrl-C, or START, to stop.

    Run this FIRST when a pad will not drive anything -- it separates "the
    Bluetooth pairing is asleep" from "the mapping is wrong", which otherwise
    look identical from inside the animation.
    """
    pad = Gamepad()
    print(describe())
    print("\n  live -- move the sticks.  Ctrl-C or START to stop.\n")
    try:
        while True:
            state = pad.poll()
            print("\r  %-62s | %s " % (state, state.command()),
                  end="", flush=True)
            if "START" in state.pressed:
                break
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    print()
    return 0


if __name__ == "__main__":
    import sys

    if "--describe" in sys.argv:
        print(describe())
        raise SystemExit(0)
    raise SystemExit(monitor())
