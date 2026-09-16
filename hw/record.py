"""Read the twelve joint angles off the robot and keep them.  NO TORQUE.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.record --fake --seconds 3        the whole path, no robot
    $V -m hw.record                           watch the twelve, Ctrl-C to stop
    $V -m hw.record --out pose.csv            ... and keep every sweep
    $V -m hw.record --once                    one table, then exit

WHAT THIS IS FOR
    `hw.stand`'s limp phase prints the twelve angles for two seconds and then
    moves on, and its log only starts at the torque phase.  This is the same
    reading with nothing after it: the robot stays BACK-DRIVABLE for as long
    as you leave it running, so a pose put in by hand -- flat on the belly, a
    leg held at a stop, the crouch pushed into place -- can be read off and
    written down.  That is how `Q_ZERO` was checked and how the next pose
    worth naming will be.

EVERY FRAME IT SENDS IS 0xA1 WITH iq=0
    A keep-alive holds the drivers' 50 ms input watchdog open without
    commanding current, which is what keeps the legs free to move by hand.
    There is no torque path in this file at all -- no `safety.SafetyGate`,
    because there is nothing here for it to gate.  Do not add a command to
    this loop; add it to `hw.stand`, which has the gate.

THE FRAME IS JOINT COORDINATES, THROUGH THE MEASURED TABLE
    The bus is opened with `hardware_map.motor_directions()` and read back
    through `calibration.joint_state`, so a column here is the same number
    `hw.stand` commands and `sim` predicts -- degrees rather than radians
    only because that is what a protractor and an operator both read.
    `--raw` adds the MOTOR's own frame beside it, which is the number on the
    driver's dial before the direction sign is applied.

    Asking for the table is what raises `MapIncomplete` on a re-flashed
    driver, which is the guard, not a nuisance: an unmapped row would be
    recorded here as a plausible angle for the wrong joint.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/record.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import kinematics as HK       # noqa: E402
from .stand import KeyPoller         # noqa: E402

__all__ = ["JointRecorder", "snapshot", "run", "main"]

#: Per-motor frame rate, the same 250 Hz `hw.stand` runs: well inside the
#: drivers' 50 ms window even if a sweep is late.
RATE_HZ = 250.0

#: A 0x9A replaces one keep-alive every this many sweeps, rotating -- the
#: only way a latched driver (0x80) shows up in a loop that is otherwise
#: write-only.
STATUS_EVERY_SWEEPS = 2

#: How often the live line is printed, and how often the CSV is flushed, so
#: that a Ctrl-C keeps everything up to the last second.
PRINT_HZ = 5.0


class JointRecorder:
    """Streaming CSV writer: one row per sweep, header fixed by the map.

    STREAMING rather than collected-then-saved, because the run this is for
    ends with Ctrl-C or with the operator's hands still on the robot.  A
    recorder that holds everything in a list and writes on a clean exit
    records nothing on the exit that actually happens.
    """

    def __init__(self, path: str, raw: bool = False):
        self.path = path
        self.raw = raw
        self.rows = 0
        self._file = open(path, "w", newline="")
        self._csv = csv.writer(self._file)
        header = ["t_s", "mark"]
        header += ["%s_deg" % label for label in HM.JOINT_LABELS]
        if raw:
            header += ["%s_motor_deg" % label for label in HM.JOINT_LABELS]
        self._csv.writerow(header)

    def add(self, t: float, mark: int, q_deg, motor_deg=None) -> None:
        row = ["%.4f" % t, mark] + ["%.4f" % v for v in q_deg]
        if self.raw:
            row += ["%.4f" % v for v in motor_deg]
        self._csv.writerow(row)
        self.rows += 1

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> str:
        self._file.close()
        return "%d sweeps -> %s" % (self.rows, self.path)


def _live_line(t: float, q_deg, marks: int) -> str:
    """One line, twelve numbers, four groups -- readable while moving a leg."""
    legs = " | ".join(
        "%s %s" % (name, " ".join("%+6.1f" % v
                                  for v in q_deg[3 * i:3 * i + 3]))
        for i, name in enumerate(C.LEGS))
    return "  t=%6.1f  %s  deg%s" % (
        t, legs, "" if not marks else "   marks %d" % marks)


def snapshot(q, motor_deg=None) -> str:
    """The full table for one pose: joint degrees, the motor's own dial, feet.

    This is what an operator copies into a note or into a constant.  The foot
    positions come from `hw.kinematics` -- the same FK the twelve directions
    were read against on 2026-09-15, so a pose that reads wrong here reads
    wrong against the reference and not against a second opinion.
    """
    q = np.asarray(q, dtype=float)
    q_deg = np.rad2deg(q)
    feet = 1e3 * HK.all_foot_positions(q)
    out = ["    %-4s %8s %8s %8s   %s" % (
        "leg", *C.JOINTS, "foot x,y,z (mm, trunk frame)")]
    for i, name in enumerate(C.LEGS):
        cells = " ".join("%8.2f" % v for v in q_deg[3 * i:3 * i + 3])
        out.append("    %-4s %s   %8.1f %7.1f %7.1f"
                   % (name, cells, *feet[i]))
    if motor_deg is not None:
        out.append("    motor frame (the driver's own dial, no direction "
                   "applied):")
        for i, name in enumerate(C.LEGS):
            out.append("    %-4s %s" % (name, " ".join(
                "%8.2f" % v for v in motor_deg[3 * i:3 * i + 3])))
    out.append("    q_rad = np.array([%s])"
               % ", ".join("%+.4f" % v for v in q))
    return "\n".join(out)


# ===========================================================================
def run(mb, *, rate_hz: float = RATE_HZ, seconds: float | None = None,
        key=None, recorder: "JointRecorder | None" = None,
        clock=time.perf_counter, quiet: bool = False):
    """Keep-alive round robin on an ARMED `mb`, reading q every sweep.

    Returns ``(stop_reason, q)`` -- the reason is None if `seconds` simply ran
    out, and `q` is the LAST pose read, which is what `--once` prints rather
    than re-reading through a second set of unwrappers.  Does not stop the
    motors -- the caller's `with MotorBus(...)` does, on every path.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    slot = mb.slot(rate_hz)

    t0 = clock()
    sweep = marks = 0
    last_print = t0
    latched_warned = False
    q = np.zeros(n)
    motor_deg = np.zeros(n)

    deadline = clock() + slot
    k = 0
    while True:
        mb.poll()
        if k == 0:
            now = clock()
            try:
                q, _ = CAL.joint_state(mb, unwrappers)
            except RuntimeError as missing:
                # Only reachable in the first sweeps after arming, before
                # every driver has answered once.  Never silently substituted.
                if now - t0 > 1.0:
                    return str(missing), q
            else:
                q_deg = np.rad2deg(q)
                motor_deg = CAL.joint_rad_to_motoroutput_deg(q)
                if recorder is not None:
                    recorder.add(now - t0, marks, q_deg, motor_deg)

                pressed = key.get() if key is not None else None
                if pressed in ("x", "X", "q", "Q"):
                    return "operator stopped", q
                if pressed in ("\r", "\n"):
                    marks += 1
                    print("\n  -- mark %d at t=%.2f s\n%s\n"
                          % (marks, now - t0,
                             snapshot(q, motor_deg)), flush=True)

                errors = mb.errors()
                latched = [mid for mid, err in errors.items()
                           if err and err & 0x80]
                if latched and not latched_warned:
                    latched_warned = True
                    print("\n   input-lost latch (0x80) on CAN %s -- those "
                          "drivers stopped listening.  Nothing was being\n"
                          "   commanded, so nothing moved; the encoders are "
                          "still being read." % latched, flush=True)

                if not quiet and now - last_print >= 1.0 / PRINT_HZ:
                    last_print = now
                    if recorder is not None:
                        recorder.flush()
                    print(_live_line(now - t0, q_deg, marks), flush=True)
                sweep += 1
                if seconds is not None and now - t0 >= seconds:
                    return None, q

        # -- one frame, to one motor: a keep-alive, or a status request ----
        mid = ids[k]
        if (sweep % STATUS_EVERY_SWEEPS == 0
                and k == (sweep // STATUS_EVERY_SWEEPS) % n):
            mb.status1_req(mid)
        else:
            mb.keepalive(mid)

        k = (k + 1) % n
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun > 3 * slot:
            deadline = clock() + slot        # re-anchor, never catch up
    # unreachable


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hw.record",
        description="Read and record the twelve joint angles.  Zero torque: "
                    "the legs stay back-drivable throughout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default=None, metavar="FILE.csv",
                    help="write one row per sweep here (streamed, so Ctrl-C "
                         "keeps what was read)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long; the default runs until "
                         "ENTER-less Ctrl-C, X or Q")
    ap.add_argument("--once", action="store_true",
                    help="print one pose table and exit; writes no file")
    ap.add_argument("--raw", action="store_true",
                    help="also record the MOTOR frame (the driver's own dial)")
    ap.add_argument("--rate", type=float, default=RATE_HZ,
                    help="per-motor frame rate, Hz")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--quiet", action="store_true",
                    help="record without the live line")
    ap.add_argument("--fake", action="store_true",
                    help="run the whole CAN path against `hw.fake_bus`")
    args = ap.parse_args(argv)

    ids = HM.motor_ids()             # raises MapIncomplete on a missing row
    print("[record] %d joints, CAN %s, directions from the table measured on "
          "2026-09-15" % (len(ids), ids))
    if not CONFIRMED_ON_DOG6:
        print("[record] NOTE: not every row is confirmed -- the angles below "
              "are only as right as the table.")
    print("[record] ZERO TORQUE: 0xA1 iq=0 keep-alives only, so every leg "
          "stays back-drivable.")
    print("[record] ENTER prints a pose table and marks the row.  X or Q "
          "stops.")

    from .motor import motorbus
    if args.fake:
        from .fake_bus import FakeDriverBus
        mb = motorbus.MotorBus(ids, bus=FakeDriverBus(ids=ids),
                               dirs=HM.motor_directions())
    else:
        mb = motorbus.MotorBus(ids, bitrate=args.bitrate,
                               dirs=HM.motor_directions())

    recorder = None
    if args.out and not args.once:
        recorder = JointRecorder(args.out, raw=args.raw)
    key = KeyPoller()
    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[record] not every motor armed", file=sys.stderr)
                return 1
            stop, q = run(mb, rate_hz=args.rate,
                          seconds=0.5 if args.once else args.seconds,
                          key=None if args.once else key, recorder=recorder,
                          quiet=args.quiet or args.once)
            if args.once:
                print()
                print(snapshot(q, CAL.joint_rad_to_motoroutput_deg(q)))
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()
        if recorder is not None:
            print("\n[record] %s" % recorder.close())

    print("[record] stopped: %s" % (stop or "time was up"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
