"""can0 is the PROGRAM's job now: brought up, cleaned and re-plugged with no hand on it.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.can_link              the link as the kernel has it, and the USB port under it
    $V -m hw.can_link reset        `ip link set can0 down` / `up`, done for you
    $V -m hw.can_link replug       the PCAN-USB unplugged and plugged back IN SOFTWARE

Every entry point that opens the real bus -- `hw.stand` and every stand, trot
and walk built on it, `hw.swing_bench`, `hw.record`, `hw.bringup` -- opens it
through `open_motor_bus`, which does all of this before the socket opens and
cleans up again after the motors are stopped.  `--fake` never touches can0.


WHY IT WAS DONE BY HAND, 2026-10-05
    The PCAN-USB drops off USB.  `dmesg` that day: 36 disconnects in 1.5 h of
    uptime -- some of them hands on the cable, but some `usb1-port2: disabled
    by hub (EMI?)` and some after a burst of `Rx urb aborted (-71)`, which no
    hand does: the cable, the motors' noise or the supply, nothing this code
    does.  Each time the kernel deletes can0 and makes a NEW one, and the new
    one is DOWN with no bitrate: every send fails, `arm()` times out, and the
    operator types `ip link set can0 up type can bitrate 1000000`.

    `ip link set can0 down` was also the only thing that cleared what a run
    leaves behind: frames still queued in the kernel's 10-frame TX queue and in
    the adapter (the next run's first frames on the wire would be the LAST
    run's), and a controller left in BUS-OFF, which with `restart-ms 0` -- what
    the link had -- it never leaves by itself.


WHAT `reset` DOES: EVERY RUN, BEFORE THE SOCKET OPENS, AND AGAIN AFTER THE STOP
    ip link set can0 down              the queued frames dropped, controller stopped
    ip link set can0 type can bitrate 1000000 restart-ms 100
    ip link set can0 up                ERROR-ACTIVE, error counters zero
    After a run it waits `EXIT_DRAIN_S` first, so `MotorBus.close()`'s stop
    frames are on the wire (48 of them, ~6 ms) before the queue is dropped;
    the next program -- this one or `can_motor_control`'s -- starts clean.
    `restart-ms 100` lets a bus-off recover by itself.  It does not save a
    run (the drivers' window is 50 ms); it saves the next one.


WHAT `replug` DOES, AND WHEN IT RUNS BY ITSELF
    It writes 1 then 0 to the `disable` file of the hub port the PCAN-USB hangs
    off (`/sys/bus/usb/devices/1-0:1.0/usb1-port2/disable` on this Pi): the
    kernel disconnects the device and switches the port's power off for
    `REPLUG_OFF_S` -- this root hub has per-port power switching
    (wHubCharacteristic 0x0009), so the adapter really loses power -- and on
    again, and it enumerates from scratch: what pulling the cable did.  On
    2026-10-05 that took 1.2 s, can0 back UP at the end; the IMU's CP2102, on
    port 1, kept its device number.  The port is remembered in `PORT_CACHE`
    every time can0 is seen, so an adapter that fell off and did not come
    back can still be found.

    It runs by itself, at most once per start, when can0 does not come back
    within `APPEAR_S`, when `reset` fails, and when `probe`'s frame never
    leaves the USB.  And once on the way out of a run whose `arm()` timed out
    with NOT ONE frame back from the bus: motors off, or a dead receive side,
    and only the second is something a replug fixes -- it costs a second.


NOT WHILE ANOTHER PROCESS HAS IT
    A reset in the middle of someone else's run cuts every driver off within
    50 ms, which from RISE or HOLD drops the robot.  So nothing here touches a
    link that has a socket open on it (`/proc/net/can/rcvlist_*`): a second
    run, `candump`, `calibrate_leg.py`.  The start REFUSES instead, before the
    bus opens -- which also ends two loops streaming to the same twelve
    drivers, something that used to be allowed.


ROOT
    `ip link set` and the port's `disable` need root.  This runs `sudo -n`,
    which this Pi's robot01 has without a password; without it the start stops
    before the bus opens and prints the commands to type.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

IFACE = "can0"
BITRATE = 1_000_000

#: The kernel restarts a BUS-OFF controller by itself after this long.  0 --
#: what the link had -- means never, until somebody downs and ups it.
RESTART_MS = 100

#: The adapters `replug` will look for when can0 is gone: PEAK PCAN-USB and
#: PCAN-USB FD, (idVendor, idProduct) as sysfs spells them.
PEAK_USB_IDS = (("0c72", "000c"), ("0c72", "0012"))

#: Where the PCAN-USB's hub port is remembered, for a replug with no can0.
PORT_CACHE = os.path.expanduser("~/.cache/dog6/pcan_usb_port")

#: How long a missing can0 is waited for before `replug`: a re-enumeration
#: takes 0.25-0.6 s on this Pi (dmesg, 2026-10-02 and -05).
APPEAR_S = 3.0

#: How long a replugged adapter gets to come back as can0.
REPLUG_APPEAR_S = 8.0

#: The port is held off this long, so the adapter really loses power and resets.
REPLUG_OFF_S = 0.5

#: `probe`'s wait for a motor's reply to its status read (~1 ms when one comes).
PROBE_S = 0.10

#: After the stop frames, before the exit reset drops whatever is queued.
EXIT_DRAIN_S = 0.02

#: `ip` and `tee` are never allowed to hang a start.
CMD_TIMEOUT_S = 5.0


class LinkError(RuntimeError):
    """can0 cannot be made usable from here; the message says what to do."""


# ---------------------------------------------------------------------------
# reading the link
# ---------------------------------------------------------------------------
def link_state(iface: str = IFACE) -> dict | None:
    """The link as `ip -details -statistics -json` has it, flattened.  None
    when there is no such interface (the adapter is off USB, or between a
    disconnect and the new can0)."""
    try:
        out = subprocess.run(["ip", "-j", "-d", "-s", "link", "show", "dev", iface],
                             capture_output=True, text=True, timeout=CMD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise LinkError("`ip link show %s` failed: %s" % (iface, failure)) from failure
    if out.returncode != 0:
        if "does not exist" in out.stderr:
            return None
        raise LinkError("`ip link show %s`: %s" % (iface, out.stderr.strip()))
    raw = json.loads(out.stdout)[0]
    info = raw.get("linkinfo", {}).get("info_data", {})
    xstats = raw.get("linkinfo", {}).get("info_xstats", {})
    stats = raw.get("stats64", raw.get("stats", {}))
    tx, rx = stats.get("tx", {}), stats.get("rx", {})
    berr = info.get("berr_counter", {})
    return {
        "iface": iface,
        "ifindex": raw.get("ifindex"),
        "up": "UP" in raw.get("flags", ()),
        "state": info.get("state", "?"),
        "bitrate": info.get("bittiming", {}).get("bitrate"),
        "restart_ms": info.get("restart_ms"),
        "berr_tx": berr.get("tx", 0), "berr_rx": berr.get("rx", 0),
        "txqlen": raw.get("txqlen"),
        "tx_packets": tx.get("packets", 0), "tx_errors": tx.get("errors", 0),
        "tx_dropped": tx.get("dropped", 0),
        "rx_packets": rx.get("packets", 0), "rx_errors": rx.get("errors", 0),
        "rx_dropped": rx.get("dropped", 0),
        "bus_off": xstats.get("bus_off", 0),
        "error_passive": xstats.get("error_passive", 0),
        "bus_error": xstats.get("bus_error", 0),
        "restarts": xstats.get("restarts", 0),
        "usb": usb_device(iface),
    }


def describe(st: dict | None) -> str:
    """One line: what an operator needs to read off the link."""
    if st is None:
        return "%s: NOT PRESENT (the adapter is off USB)" % IFACE
    return ("%s: %s, %s, %s bit/s, restart-ms %s, error counters tx %d rx %d, "
            "frames tx %d rx %d (tx errors %d, dropped %d), bus-off %d, USB %s"
            % (st["iface"], "UP" if st["up"] else "DOWN", st["state"],
               st["bitrate"], st["restart_ms"], st["berr_tx"], st["berr_rx"],
               st["tx_packets"], st["rx_packets"], st["tx_errors"],
               st["tx_dropped"], st["bus_off"], st["usb"] or "?"))


def receivers(iface: str = IFACE) -> int:
    """Sockets listening on `iface`, from the kernel's receive lists: another
    process with the bus open.  Our own socket counts too, so ask BEFORE
    opening it.  A python-can socket sits in two lists (`rx_all` for frames,
    `rx_err` for error frames) under one `userdata`, so sockets are counted by
    that column.  Sockets bound to every interface ("any") are not counted --
    a reset does not cut them off anything."""
    texts = []
    for path in glob.glob("/proc/net/can/rcvlist_*"):
        try:
            with open(path) as f:
                texts.append(f.read())
        except OSError:
            continue
    return len(sockets_in(texts, iface))


def sockets_in(texts, iface: str = IFACE) -> set:
    """The `userdata` column of every receive-list row on `iface`, across the
    `/proc/net/can/rcvlist_*` texts given -- one per socket."""
    sockets = set()
    for text in texts:
        for line in text.splitlines():
            cols = line.split()
            if len(cols) >= 5 and cols[0] == iface:
                sockets.add(cols[4])
    return sockets


def usb_device(iface: str = IFACE) -> str | None:
    """The USB device can0 lives on, as sysfs names it ("1-2"), or None."""
    try:
        return os.path.basename(os.path.dirname(
            os.path.realpath("/sys/class/net/%s/device" % iface)))
    except OSError:
        return None


def _usb_port(iface: str) -> str | None:
    """The hub port directory the adapter hangs off, found three ways: under
    can0, under any PEAK device on USB, or as last remembered."""
    device = "/sys/class/net/%s/device/.." % iface
    if os.path.exists(device + "/port"):
        port = os.path.realpath(device + "/port")
        _remember(port)
        return port
    for vendor in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        here = os.path.dirname(vendor)
        try:
            ids = (open(vendor).read().strip(),
                   open(here + "/idProduct").read().strip())
        except OSError:
            continue
        if ids in PEAK_USB_IDS and os.path.exists(here + "/port"):
            port = os.path.realpath(here + "/port")
            _remember(port)
            return port
    try:
        with open(PORT_CACHE) as f:
            port = f.read().strip()
        return port if os.path.isdir(port) else None
    except OSError:
        return None


def _remember(port: str) -> None:
    try:
        os.makedirs(os.path.dirname(PORT_CACHE), exist_ok=True)
        with open(PORT_CACHE, "w") as f:
            f.write(port + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# changing it -- root
# ---------------------------------------------------------------------------
def _root(argv: list, stdin: str | None = None) -> None:
    """Run `argv` as root: directly when we are, `sudo -n` otherwise."""
    cmd = list(argv) if os.geteuid() == 0 else ["sudo", "-n"] + list(argv)
    try:
        out = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                             timeout=CMD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise LinkError("`%s` failed: %s" % (" ".join(cmd), failure)) from failure
    if out.returncode != 0:
        err = out.stderr.strip()
        if "password" in err:
            raise LinkError(
                "`%s` needs root, and sudo asks for a password here.  Either "
                "let this user run `ip` without one, or bring the link up by "
                "hand before the run:\n    sudo ip link set %s down\n    sudo "
                "ip link set %s up type can bitrate %d restart-ms %d"
                % (" ".join(argv), IFACE, IFACE, BITRATE, RESTART_MS))
        raise LinkError("`%s`: %s" % (" ".join(cmd), err or "exit %d" % out.returncode))


def _refuse_if_busy(iface: str) -> None:
    busy = receivers(iface)
    if busy:
        raise LinkError(
            "%s is open in another process (%d socket%s listening: a run, "
            "candump, calibrate_leg.py?).  Resetting it would cut that "
            "process's motors off mid-run, and two loops must not stream to "
            "the same drivers anyway -- stop it first"
            % (iface, busy, "" if busy == 1 else "s"))


def reset(iface: str = IFACE, bitrate: int = BITRATE,
          restart_ms: int = RESTART_MS) -> dict:
    """down -> bitrate and restart-ms -> up.  What was queued is dropped and
    the controller restarts ERROR-ACTIVE.  Returns the new state; raises
    `LinkError` if the link is in use, absent, or does not come up."""
    _refuse_if_busy(iface)
    if link_state(iface) is None:
        raise LinkError("%s is not there to reset" % iface)
    _root(["ip", "link", "set", iface, "down"])
    _root(["ip", "link", "set", iface, "type", "can", "bitrate", str(bitrate),
           "restart-ms", str(restart_ms)])
    _root(["ip", "link", "set", iface, "up"])
    st = link_state(iface)
    if st is None or not st["up"] or st["state"] != "ERROR-ACTIVE":
        raise LinkError("%s did not come up clean: %s" % (iface, describe(st)))
    return st


def replug(iface: str = IFACE, say=print) -> dict | None:
    """The adapter's hub port off for `REPLUG_OFF_S`, then on: a fresh
    enumeration, as if the cable had been pulled.  Returns the link once can0
    is back (not yet reset -- the new one is DOWN), or None if it never came."""
    if link_state(iface) is not None:
        _refuse_if_busy(iface)
    port = _usb_port(iface)
    if port is None:
        raise LinkError("no PCAN-USB on USB and no port remembered for it (%s): "
                        "unplug it and plug it back by hand" % PORT_CACHE)
    switch = os.path.join(port, "disable")
    if not os.path.exists(switch):
        raise LinkError("%s has no `disable` switch: replug by hand" % port)
    say("[can] replugging the PCAN-USB in software: %s off %.1f s, then on"
        % (os.path.basename(port), REPLUG_OFF_S))
    _root(["tee", switch], stdin="1\n")
    time.sleep(REPLUG_OFF_S)
    _root(["tee", switch], stdin="0\n")
    return wait_present(iface, REPLUG_APPEAR_S)


def wait_present(iface: str = IFACE, timeout_s: float = APPEAR_S) -> dict | None:
    """Poll for `iface` to exist, up to `timeout_s`."""
    end = time.monotonic() + timeout_s
    while True:
        st = link_state(iface)
        if st is not None or time.monotonic() >= end:
            return st
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# the start and the end of a run
# ---------------------------------------------------------------------------
def prepare(iface: str = IFACE, bitrate: int = BITRATE, say=print) -> dict:
    """Everything a run used to need a hand for, in order: wait for can0 if
    it is being re-enumerated, replug it if it does not come, reset it.  At
    most one replug.  Returns the link, reset; raises `LinkError`."""
    replugged = False
    st = link_state(iface)
    if st is None:
        say("[can] %s is not there -- waiting up to %.0f s for the PCAN-USB to "
            "come back on USB" % (iface, APPEAR_S))
        st = wait_present(iface, APPEAR_S)
    if st is None:
        st = replug(iface, say)
        replugged = True
        if st is None:
            raise LinkError("%s did not come back after a software replug: "
                            "unplug the PCAN-USB and plug it back by hand, and "
                            "look at `dmesg | tail`" % iface)
    try:
        st = reset(iface, bitrate)
    except LinkError as failure:
        if replugged or receivers(iface):
            raise
        say("[can] reset failed (%s) -- replugging" % failure)
        if replug(iface, say) is None:
            raise LinkError("%s did not come back after a software replug: "
                            "replug by hand" % iface) from failure
        st = reset(iface, bitrate)
    return st


def probe(bus, motor_id: int, iface: str = IFACE) -> str:
    """One 0x9A status read to `motor_id` on the freshly reset link, and what
    became of it within `PROBE_S`:

        "answer"  the motor replied: the adapter sends AND receives
        "silent"  the adapter took the frame and nothing came back: the
                  motors are off, and `arm()` waits for them as it always did
                  -- or, with them on, the adapter's receive side is dead
        "stuck"   the adapter never took the frame off the USB
        "refused" the kernel would not take the frame at all

    THE KERNEL'S TX COUNT IS NOT AN ACK ON THIS ADAPTER.  The old PCAN-USB's
    driver counts a frame transmitted when the USB transfer to the adapter
    completes: on 2026-10-05 it counted 64 with nothing answering and both
    error counters at zero.  So "stuck" is read off it -- the USB side -- and
    only a reply says the CAN side works."""
    import can
    before = _tx_packets(iface)
    try:
        bus.send(can.Message(arbitration_id=0x140 + motor_id,
                             data=[0x9A] + [0] * 7, is_extended_id=False))
    except (can.CanError, OSError):
        return "refused"
    end = time.monotonic() + PROBE_S
    while time.monotonic() < end:
        try:
            msg = bus.recv(timeout=0.005)
        except (can.CanError, OSError):
            return "refused"
        if (msg is not None and not msg.is_error_frame
                and msg.arbitration_id == 0x140 + motor_id
                and len(msg.data) and msg.data[0] == 0x9A):
            return "answer"
    return "silent" if _tx_packets(iface) > before else "stuck"


def _tx_packets(iface: str) -> int:
    try:
        with open("/sys/class/net/%s/statistics/tx_packets" % iface) as f:
            return int(f.read())
    except (OSError, ValueError):
        return -1


def _open_socket(iface: str, bitrate: int):
    import can
    return can.Bus(interface="socketcan", channel=iface, bitrate=bitrate)


def open_motor_bus(ids, *, bitrate: int = BITRATE, dirs=None, say=print,
                   iface: str = IFACE):
    """`motorbus.MotorBus(ids, bitrate=..., dirs=...)` on a link this module
    has just made clean -- `prepare`, then `probe` (a stuck adapter is
    replugged once), then the socket.  The bus it returns resets the link
    again after `close()` has stopped the motors.  Raises `LinkError`."""
    st = prepare(iface, bitrate, say)
    sock = _open_socket(iface, bitrate)
    verdict = probe(sock, list(ids)[0], iface)
    if verdict in ("stuck", "refused"):
        sock.shutdown()
        say("[can] the PCAN-USB took a frame and %s -- replugging"
            % ("never sent it" if verdict == "stuck" else "refused it"))
        if replug(iface, say) is None:
            raise LinkError("%s did not come back after a software replug: "
                            "replug by hand" % iface)
        st = reset(iface, bitrate)
        sock = _open_socket(iface, bitrate)
        verdict = probe(sock, list(ids)[0], iface)
    say("[can] %s reset (down, %d bit/s, restart-ms %s, up): the queues and "
        "the controller are clear; %s" % (
            iface, st["bitrate"], st["restart_ms"], {
                "answer": "CAN %d answered a status read" % list(ids)[0],
                "silent": "NO MOTOR ANSWERED a status read yet -- power them "
                          "on, arm() waits for them.  If they ARE on and arm() "
                          "times out, the adapter's receive side is dead: "
                          "`python -m hw.can_link replug`",
                "stuck": "WARNING: the adapter still does not take frames "
                         "off the USB after a software replug: unplug it by "
                         "hand if arm() times out",
                "refused": "WARNING: the kernel still refuses frames after a "
                           "software replug; arm() will say more"}[verdict]))
    mb = _linked_class()(ids, sock, dirs=dirs, iface=iface, bitrate=bitrate,
                         say=say)
    mb.at_open = link_state(iface)
    return mb


_LINKED = None


def _linked_class():
    global _LINKED
    if _LINKED is not None:
        return _LINKED
    from .motor import motorbus

    class LinkedMotorBus(motorbus.MotorBus):
        """`MotorBus` over a socket `open_motor_bus` opened: owns it, and
        resets the link after the motors are stopped.  The DOG5 class is
        subclassed, not edited -- `hw.motor` stays verbatim."""

        def __init__(self, ids, sock, *, dirs=None, iface=IFACE,
                     bitrate=BITRATE, say=print):
            super().__init__(ids, bus=sock, dirs=dirs)
            self._owns_bus = True            # close() shuts the socket down
            self.iface = iface
            self.bitrate = bitrate
            self._say = say
            #: The last exception a send raised.  `RoundRobinBus.send` counts
            #: a refused frame and drops the reason; `realtime.SendGuard`
            #: wants it for the stop line (ENOBUFS is a full queue, ENETDOWN
            #: a link that went down, ENXIO/ENODEV an adapter that left USB).
            self.last_send_error = None
            #: `link_state()` as the stop left it, before the exit reset.
            self.at_stop = None
            #: ... and as `open_motor_bus` left it, for `after_failed_arm`.
            self.at_open = None
            #: Set by `after_failed_arm`: replug instead of reset on the way out.
            self.replug_on_close = False
            send = sock.send

            def recorded_send(msg, timeout=None):
                try:
                    return send(msg, timeout)
                except Exception as failure:     # noqa: BLE001 -- re-raised
                    self.last_send_error = failure
                    raise

            sock.send = recorded_send

        def close(self):
            try:
                super().close()              # 0x81 + iq=0 to every motor, socket shut
            finally:
                time.sleep(EXIT_DRAIN_S)
                try:
                    self.at_stop = link_state(self.iface)
                except LinkError:
                    self.at_stop = None
                try:
                    if self.replug_on_close and replug(self.iface, self._say) is None:
                        raise LinkError("it did not come back")
                    reset(self.iface, self.bitrate)
                    self._say("[can] %s %s after the stop: nothing left "
                              "queued for the next run"
                              % (self.iface, "replugged and reset"
                                 if self.replug_on_close else "reset"))
                except LinkError as failure:
                    self._say("[can] %s not reset after the stop (%s); the "
                              "next run's start will" % (self.iface, failure))

    _LINKED = LinkedMotorBus
    return _LINKED


def __getattr__(name):
    # LinkedMotorBus subclasses motorbus.MotorBus, which needs python-can; a
    # laptop importing hw without it must still be able to import this file.
    if name == "LinkedMotorBus":
        return _linked_class()
    raise AttributeError(name)


def bus_errors() -> tuple:
    """The exceptions a dead link raises out of `MotorBus.poll()`: python-can's
    `CanError` (ENETDOWN when the link goes down, ENODEV when the adapter
    leaves USB) and a bare `OSError`."""
    import can
    return (can.CanError, OSError)


def after_failed_arm(mb) -> str:
    """After `arm()` timed out on the real bus: the link, which side of it to
    look at -- and, when NOT ONE frame came back while arming, a replug of
    the adapter on the way out (`LinkedMotorBus.close`), so that the next run
    starts on a fresh one whichever of the two it was."""
    try:
        st = link_state(getattr(mb, "iface", IFACE))
    except LinkError as failure:
        return str(failure)
    if st is None:
        return "%s is GONE -- the adapter left USB while arming; see dmesg" % IFACE
    line = describe(st)
    if not st["up"]:
        return (line + "\n      the link went DOWN while arming -- a new can0 "
                "after a USB dropout comes up DOWN; see dmesg")
    before = getattr(mb, "at_open", None)
    if before is not None and st["rx_packets"] == before["rx_packets"]:
        mb.replug_on_close = True
        return (line + "\n      NOT ONE FRAME came back while arming: the "
                "motors are off -- or, if they are on, the adapter's receive "
                "side is dead.  It is replugged in software on the way out; "
                "power the motors and run again")
    if st["state"] in ("ERROR-PASSIVE", "BUS-OFF") or st["berr_tx"] >= 128:
        return (line + "\n      the bus refuses our frames: the CAN cable or "
                "its termination -- not the adapter")
    return (line + "\n      frames came back and the link is clean: look at "
            "the [arm] lines above for which motor never answered or stayed "
            "latched")


_KERNEL_KEYS = ("USB disconnect", "EMI", "urb", "peak_usb", "can0", "-71",
                "-110")


def uptime() -> float:
    """Seconds since boot -- the clock `dmesg` stamps its lines with."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def dmesg_since(t0: float, lines: int = 8) -> list:
    """The kernel's lines about USB and the PCAN since uptime `t0` (a run's
    start), the last `lines` of them."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True,
                             timeout=CMD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return []
    hits = []
    for line in out.stdout.splitlines():
        if not line.startswith("[") or "setting BTR" in line:
            continue
        try:
            stamp = float(line[1:line.index("]")])
        except ValueError:
            continue
        if stamp >= t0 and any(k in line for k in _KERNEL_KEYS):
            hits.append(line)
    return hits[-lines:]


def dmesg_tail(lines: int = 6) -> list:
    """The kernel's last words about USB and the PCAN, whenever they were."""
    return dmesg_since(0.0, lines)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m hw.can_link",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", default="status",
                    choices=("status", "reset", "replug"))
    ap.add_argument("--iface", default=IFACE)
    ap.add_argument("--bitrate", type=int, default=BITRATE)
    args = ap.parse_args(argv)
    try:
        if args.action == "reset":
            st = prepare(args.iface, args.bitrate)
        elif args.action == "replug":
            if replug(args.iface) is None:
                raise LinkError("%s did not come back" % args.iface)
            st = reset(args.iface, args.bitrate)
        else:
            st = link_state(args.iface)
    except LinkError as failure:
        print("[can] %s" % failure, file=sys.stderr)
        return 1
    print(describe(st))
    port = _usb_port(args.iface)
    print("  USB port %s; sockets open on it: %d" % (port or "?", receivers(args.iface)))
    for line in dmesg_tail():
        print("  dmesg: " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
