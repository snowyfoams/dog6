"""The CAN motor library, VERBATIM from DOG5.  Not rewritten for DOG6.

DOG6 uses the same twelve MG5010-i10 gearmotors and the same LK/K-TECH CAN
protocol as DOG5, so this is the one part of the hardware layer that is not a
guess when it is inherited: it is a PROTOCOL, and the protocol did not change
with the chassis.  The four modules beside this one are byte-for-byte copies
of DOG5's ``src/motor/`` -- the code that has driven twelve drivers at 250 Hz
on a real machine, including the over-CAN recovery of the input-signal-lost
latch that removes the power cycle between runs.  Nothing is cleaned up.

    from hw.motor import MotorBus, open_bus

WHAT IS INHERITED HERE AND WHAT IS NOT
    Inherited safely      the frame layout, the command bytes, the unit gains
                          in `motor_gains`, the bus pacing, the fault decode
                          and the recovery sequence.  All properties of the
                          driver, identical on both robots.
    NOT inherited here    which CAN id is which joint, and which way each one
                          turns.  Those are `hw.hardware_map`, they are
                          physical facts about an assembled robot, and DOG6
                          has not been assembled.  See that file.

WHY THE LOADING IS NOT A PLAIN IMPORT
    Being verbatim, the modules import each other by FLAT name --
    ``import motor_gains as config``, ``from motor_library import open_bus`` --
    which is what they do in DOG5's research tree, where `dog5_paths` puts
    every source directory on `sys.path`.  This package must not do that:
    adding directories to `sys.path` is how a stray `config.py` shadows
    another one and kills the CAN layer with an unrelated `AttributeError`
    several imports later.  So each module is loaded from its own file and
    registered under both names -- the flat one its siblings expect, and the
    dotted `hw.motor.*` one.  `sys.path` is never touched.

    THE SUPPORTED SPELLING IS ATTRIBUTE ACCESS, and DOG5's copy of this note
    is not quite right about that:

        from hw.motor import motorbus        # works cold
        from hw.motor import MotorBus        # works cold
        import hw.motor.motorbus             # ModuleNotFoundError, cold

    The last one asks Python's own import machinery for a submodule, which
    never reaches `__getattr__` below and so never registers `motor_library`
    under the flat name `motorbus` then imports from.  It succeeds only after
    something has already touched the attribute.  Adding a meta-path finder to
    close that gap would be more machinery than the gap is worth; use the
    first two spellings.

    A module already present under its flat name is REUSED, not shadowed --
    which is what makes it safe to have DOG5's SDK imported in the same
    process.  The two copies are identical, so sharing one is correct; if
    they ever stop being identical, `sim.selftest` is where that shows up.

WHY MOST OF IT IS LAZY
    `motorbus` and `motor_library` import `python-can` at module level.  A
    laptop running the simulator has no CAN adapter and may have no
    python-can, and must not need either to ``import hw``.  Only
    `motor_gains` -- pure numbers, no imports -- is loaded eagerly, because
    `hw.calibration` needs ENCODER_GAIN and nothing else.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Load order is the dependency order: motorbus needs motor_library, which
#: needs mac_can on macOS.
_ORDER = ("mac_can", "motor_gains", "motor_library", "motorbus")

#: Loaded on first use.  motor_gains is the exception, loaded below.
_LAZY_MODULES = ("mac_can", "motor_library", "motorbus")

#: {attribute: (module, name in it)} for what is worth exposing at package
#: level without naming the module it lives in.
_LAZY_ATTRS = {
    "MotorBus": ("motorbus", "MotorBus"),
    "RoundRobinBus": ("motorbus", "RoundRobinBus"),
    "MotorRecord": ("motorbus", "MotorRecord"),
    "arm_motors": ("motorbus", "arm_motors"),
    "decode_errors": ("motorbus", "decode_errors"),
    "stop_all": ("motorbus", "stop_all"),
    "LKMotor": ("motor_library", "LKMotor"),
    "open_bus": ("motor_library", "open_bus"),
    "prime_watchdog": ("motor_library", "prime_watchdog"),
}


def _load(name: str):
    """Import ``<name>.py`` from this directory under its flat module name."""
    existing = sys.modules.get(name)
    if existing is not None:
        sys.modules.setdefault(f"{__name__}.{name}", existing)
        return existing

    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec so a sibling's `import motor_library` during exec
    # finds it, and so a failed load leaves no half-built module behind.
    sys.modules[name] = module
    sys.modules[f"{__name__}.{name}"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        sys.modules.pop(f"{__name__}.{name}", None)
        raise
    return module


def _load_chain(name: str):
    """Load `name` and everything before it in the dependency order."""
    for candidate in _ORDER:
        module = _load(candidate)
        globals()[candidate] = module
        if candidate == name:
            return module
    raise AttributeError(name)


# motor_gains has no imports at all, so it costs nothing, and the joint
# calibration needs its encoder_gain.  mac_can loads with it only to keep the
# order honest: it imports ctypes and os, nothing that needs a CAN adapter.
motor_gains = _load_chain("motor_gains")

#: Unit conversions -- the single source for every one of them.
ENCODER_GAIN = motor_gains.encoder_gain      # raw encoder * this = motor-output deg
TORQUE_GAIN = motor_gains.torque_gain        # N*m * this = iq LSB
VEL_GAIN = motor_gains.vel_gain              # commanded dps * this = speed LSB
VEL_STATE_GAIN = motor_gains.vel_state_gain  # raw speed / this = output dps
POS_GAIN = motor_gains.pos_gain              # output deg * this = angle LSB
MAX_SPEED_POS = motor_gains.max_speed_pos


def __getattr__(name):
    if name in _LAZY_MODULES:
        return _load_chain(name)
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        value = getattr(_load_chain(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))


__all__ = [
    "mac_can", "motor_gains", "motor_library", "motorbus",
    "MotorBus", "RoundRobinBus", "MotorRecord", "LKMotor", "open_bus",
    "arm_motors", "decode_errors", "stop_all", "prime_watchdog",
    "ENCODER_GAIN", "TORQUE_GAIN", "VEL_GAIN", "VEL_STATE_GAIN", "POS_GAIN",
    "MAX_SPEED_POS",
]
