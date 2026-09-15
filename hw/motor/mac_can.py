"""macOS helpers for opening a USB-CAN adapter without sudo.

Two adapter families are supported on macOS, auto-detected by USB id:

* PEAK PCAN-USB -> python-can's ``pcan`` backend, which loads the MacCAN
  ``libPCBUSB.dylib`` userspace driver (no sudo, no kext). Install the dylib
  once (see :func:`ensure_pcbusb_loadable`); we point python-can at
  ``~/.local/lib`` at runtime, so no shell env var and no write to
  ``/usr/local/lib`` is required.
* CANable / candleLight -> python-can's ``gs_usb`` backend via libusb.
  ``gs_usb.start()`` calls ``detach_kernel_driver(0)``, which is root-only on
  macOS and fails with EACCES on every run after the first. libusb can *seize*
  the interface without detaching, so we make ``is_kernel_driver_active``
  report False and skip the detach entirely -- no sudo required.

Everything here is a no-op on non-macOS platforms.
"""
import os
import platform
from ctypes.util import find_library

# (idVendor, idProduct) of the adapters we know how to open.
PEAK_USB_IDS = [(0x0C72, 0x000C), (0x0C72, 0x0012)]  # PCAN-USB, PCAN-USB FD
CANDLELIGHT_USB_IDS = [(0x1D50, 0x606F), (0x1209, 0x2323), (0x1D50, 0x60C4)]

# Where we look for libPCBUSB.dylib. ``~/.local/lib`` is our no-sudo install
# dir; ``/usr/local/lib`` covers the case where the official install.sh was run.
_PCBUSB_DIRS = (os.path.expanduser("~/.local/lib"), "/usr/local/lib")


def detect_adapter():
    """Return ``'pcan'``, ``'gs_usb'`` or ``None`` for the plugged-in adapter.

    Returns ``None`` off macOS or when nothing recognised is connected.

    The two adapter families need *different* detection mechanisms:

    * PEAK PCAN-USB is detected through the PCBUSB driver's own channel-query
      API. We deliberately do NOT probe it with libusb/pyusb: PCBUSB also
      drives the device via libusb, and once it has touched the device the
      generic pyusb enumeration can no longer see it (it reports zero devices).
    * CANable/candleLight is detected by libusb/pyusb USB-id match, which is
      reliable for that adapter.
    """
    if platform.system().lower() != "darwin":
        return None

    # PCAN-USB: ask the PCBUSB driver what PCAN channels are attached.
    try:
        ensure_pcbusb_loadable()
        from can.interfaces.pcan import PcanBus

        if PcanBus._detect_available_configs():
            return "pcan"
    except Exception:
        pass  # PCBUSB driver missing or no PCAN channel -> try the CANable

    # CANable/candleLight: libusb enumeration is reliable here.
    try:
        import usb.core

        if any(usb.core.find(idVendor=v, idProduct=p) for v, p in CANDLELIGHT_USB_IDS):
            return "gs_usb"
    except Exception:
        pass

    return None


def ensure_pcbusb_loadable():
    """Make python-can's ``pcan`` backend able to find ``libPCBUSB.dylib``.

    python-can resolves the driver with ``find_library("PCBUSB")``, which only
    searches the dyld paths. Our no-sudo install lives in ``~/.local/lib`` --
    not a default search dir on recent macOS -- so prepend the candidate dirs
    to ``DYLD_LIBRARY_PATH`` at runtime (ctypes reads ``os.environ`` when it
    resolves the name). Raises ``OSError`` with install hints if still missing.
    """
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    existing = current.split(":") if current else []
    extra = [d for d in _PCBUSB_DIRS if os.path.isdir(d) and d not in existing]
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(extra + existing)

    if not find_library("PCBUSB"):
        raise OSError(
            "libPCBUSB.dylib not found (searched: {}).\n"
            "  The PCAN-USB needs the MacCAN PCBUSB userspace driver. Install it:\n"
            "    1. download a release from\n"
            "       https://github.com/mac-can/PCBUSB-Library/releases (v0.13+),\n"
            "    2. copy libPCBUSB.<ver>.dylib into ~/.local/lib/ and symlink\n"
            "       libPCBUSB.dylib -> it  (or run the bundled `sudo ./install.sh`)."
            .format(", ".join(_PCBUSB_DIRS))
        )


_PATCHED = False


def apply_mac_gs_usb_patches() -> None:
    global _PATCHED
    if _PATCHED or platform.system().lower() != "darwin":
        return
    import usb.core

    # Skip the root-only detach_kernel_driver; libusb seizes the interface.
    usb.core.Device.is_kernel_driver_active = lambda self, intf: False
    _PATCHED = True
