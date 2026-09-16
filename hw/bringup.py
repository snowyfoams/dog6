"""Building the hardware map from the robot, in the order it has to happen.

    python -m hw.bringup plan                    where this has got to
    python -m hw.bringup scan                    which CAN ids answer?
    python -m hw.bringup spin --id 7 --go        drive that motor, watch it
    python -m hw.bringup spin --id 7 --joint FL.knee --go
    python -m hw.bringup check --joint FL.knee --go
    python -m hw.bringup setzero --yes
    python -m hw.bringup imu --go

    ...any of them with --fake to run the whole path against software drivers.

DISCOVERY, NOT VERIFICATION, AND THAT IS THE DIFFERENCE FROM DOG5
    DOG5's tools checked a table that already existed.  These BUILD one, and
    on 2026-09-15 they built DOG6's: twelve rows, each spun, then re-driven
    in joint coordinates and confirmed.  They remain the tools -- a replaced
    or re-flashed driver sends its row back to empty and it is re-measured the
    same way, not guessed from the eleven others.

    The two halves are found separately because they are found by different
    observations:

        WHICH JOINT   `spin --id 7` turns motor 7 in the MOTOR's own frame.
                      You watch which limb moves.  That is the can_id.
        WHICH WAY     the same command, with `--joint FL.knee` so the tool can
                      print what `hw.kinematics` says a POSITIVE JOINT angle
                      does to that foot.  Moved that way -> direction +1.
                      Moved the other way -> -1.

    `spin` commands with direction +1 whatever the table says, which is what
    "unknown" has to mean: it drives the motor's own positive direction and
    reports the motor's own encoder.  No joint coordinates are involved
    anywhere in it, so it works on a completely empty map -- and that is why
    it is still the right tool for a row being re-measured: it cannot be
    biased by the value already sitting in that row.

    `check` is the second pass.  It needs a filled-in row and drives in JOINT
    coordinates through the measured direction, so it exercises the same path
    a controller will.  Passing it is what earns `confirmed=True`.

NOTHING IN STEPS 1-4 COMMANDS TORQUE.  `scan` and the arming ladder stream
    iq=0 keep-alives, which hold the drivers' 50 ms input watchdog open
    without producing motion -- every motor stays back-drivable.  `spin` and
    `check` use the drivers' own 0xA4 position loop with a low speed cap.  The
    torque path is `hw.safety`, and it still refuses a map with a hole in it,
    with no override at all -- the measurement satisfied that gate rather than
    removing it.
"""
from __future__ import annotations

import argparse
import sys
import time

if __package__ in (None, ""):        # allow `python hw/bringup.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import CONFIRMED_ON_DOG6, VERIFICATION_LOG          # noqa: E402
from . import calibration as CAL                           # noqa: E402
from . import hardware_map as HM                           # noqa: E402

#: Position-step defaults.  Small enough that the wrong limb moving is
#: startling rather than damaging, slow enough to watch happen.
STEP_RAD = 0.10
MOTOR_DPS = 60.0
HOLD_S = 1.5

#: `scan` probes this range.  The LK drivers answer at 0x140 + id; ids outside
#: 1..32 mean somebody flashed something unusual and the operator should know.
SCAN_RANGE = range(1, 33)

DEG_PER_RAD = 57.29577951308232


def _bus(ids, bitrate: int, fake: bool = False, joint_frame: bool = False):
    """Open a MotorBus over raw CAN `ids`.

    `joint_frame=False` -- the default, and all `scan` and `spin` ever use --
    passes NO direction table, so `MotorBus` defaults every motor to +1 and
    every command and reading is in the MOTOR's own frame.  That is the
    correct frame before a direction has been measured, and using it is what
    lets these tools run on an empty map.

    `joint_frame=True` asks `hardware_map` for the measured directions and
    raises `MapIncomplete` if they are not all there.
    """
    from .motor import motorbus

    dirs = HM.motor_directions() if joint_frame else None
    if fake:
        from .fake_bus import FakeDriverBus
        return motorbus.MotorBus(ids, bus=FakeDriverBus(ids=ids), dirs=dirs)
    return motorbus.MotorBus(ids, bitrate=bitrate, dirs=dirs)


def _hold(mb, can_id, target_deg, seconds, dps, rate_hz, unwrap):
    """0xA4 re-sent every slot until `seconds` is up; returns the angle reached.

    RE-SENT is the point: the driver's input watchdog is 50 ms, so one command
    followed by silence latches error 0x80 and the joint goes limp mid-move.
    The angle returned is in whatever frame the bus was opened in.
    """
    slot = mb.slot(rate_hz)
    deadline = time.perf_counter() + slot
    until = time.perf_counter() + seconds
    while time.perf_counter() < until:
        mb.poll()
        mb.position(can_id, target_deg, max_dps=dps)
        mb.pace(deadline)
        deadline += slot
    return unwrap.update(mb.rec(can_id).encoder)


# ===========================================================================
def cmd_scan(args) -> int:
    """Which CAN ids answer at all?  Needs no map -- it asks the bus."""
    from .motor import motorbus

    ids = ([int(p) for p in args.ids.split(",")] if args.ids
           else list(SCAN_RANGE))
    print(f"[scan] probing ids {ids[0]}..{ids[-1]} at "
          f"{args.bitrate / 1e6:.1f} Mbit/s, motor frame, zero torque")

    with _bus(ids, args.bitrate, args.fake) as mb:
        if not args.no_arm:
            print("[scan] arming (zero-torque stream + 0x9B -> 0x88 ladder); "
                  "switch 24 V on if it is off.  Ids that do not exist simply "
                  "never answer, so give --timeout a few seconds.")
            mb.arm(rate_hz=args.rate, timeout_s=args.timeout, verbose=True)
        status = mb.status1()
        mb.status2()
        mb.poll()
        temps, volts = mb.temps(), mb.voltages()
        encoders = mb.encoders_deg()

        answered = [mid for mid in ids
                    if status.get(mid, (None, None))[0] is not None]
        print()
        print(f"  {'CAN':>3}  {'error':<28} {'T':>4} {'V':>6} "
              f"{'motor deg':>10}   claimed by")
        for mid in answered:
            error, _ = status.get(mid, (None, None))
            owner = HM.joint_of(mid)
            print(f"  {mid:>3}  {motorbus.decode_errors(error):<28} "
                  f"{(temps.get(mid) or 0):>3}C {(volts.get(mid) or 0.0):>5.1f}V "
                  f"{encoders.get(mid, 0):>9.2f}   "
                  f"{owner.label if owner else '-- nothing yet --'}")

        print()
        print(f"[scan] {len(answered)} of {len(ids)} ids answered: {answered}")
        if len(answered) != HM.N_JOINTS:
            print(f"[scan] expected {HM.N_JOINTS}.  Check the termination, the "
                  "daisy chain, and that every driver has 24 V.",
                  file=sys.stderr)
        unclaimed = [m for m in answered if HM.joint_of(m) is None]
        if unclaimed:
            print(f"[scan] not yet in the map: {unclaimed}")
            print("[scan] next: python -m hw.bringup spin --id %d --go"
                  % unclaimed[0])
        return 0 if len(answered) == HM.N_JOINTS else 1


# ===========================================================================
def cmd_spin(args) -> int:
    """Turn ONE motor a small step in its OWN frame.  Works on an empty map.

    With `--joint`, also prints what `hw.kinematics` says a POSITIVE JOINT
    angle does to that foot, which is how the direction is read off.
    """
    can_id = args.id
    owner = HM.joint_of(can_id)
    print("[spin] CAN %d: %s" % (
        can_id, ("currently mapped to " + owner.label) if owner
        else "not in the map"))
    print(f"[spin] commanding {args.step:+.3f} rad "
          f"({args.step * DEG_PER_RAD:+.2f} deg) in the MOTOR's frame -- "
          "no direction is applied.")

    if args.joint is None:
        print("[spin] WATCH WHICH LIMB MOVES.  That tells you the can_id.")
        print("[spin] then re-run with --joint <that joint> to read the sign:")
        print(f"[spin]   python -m hw.bringup spin --id {can_id} "
              "--joint FL.knee --go")
    else:
        index, entry = HM.row_of(args.joint)
        label, dx, dy, dz = HM.direction_check_table(args.step)[index]
        way = HM.dominant(dx, dy, dz)
        print()
        print(f"[spin] IF CAN {can_id} is {label}, a POSITIVE JOINT angle "
              "moves that foot")
        print(f"[spin]   dx {dx:+.2f}  dy {dy:+.2f}  dz {dz:+.2f} mm   "
              f"(dominant {way}), trunk frame")
        print("[spin] so after the move below:")
        print(f"[spin]   the {entry.leg} foot went {way}       -> direction = +1")
        print(f"[spin]   the {entry.leg} foot went the other way -> direction = -1")
        print(f"[spin]   a DIFFERENT limb moved      -> CAN {can_id} is not "
              f"{label}")

    if not args.go:
        print()
        print("[spin] dry run.  Add --go to actually command it.")
        return 0

    with _bus([can_id], args.bitrate, args.fake) as mb:
        if not mb.arm(rate_hz=args.rate, timeout_s=args.timeout):
            print(f"[spin] CAN {can_id} did not arm", file=sys.stderr)
            return 1
        mb.flush_rx()
        mb.poll()
        # Motor frame throughout: `encoders_deg` is the raw output-shaft angle
        # and `position` applies the +1 default, so what goes out and what
        # comes back are the same coordinate.  Unwrapped, because the 16-bit
        # register wraps and a step across zero would read as -360 deg.
        unwrap = CAL.EncoderUnwrap()
        start = unwrap.update(mb.rec(can_id).encoder)
        target = start + args.step * DEG_PER_RAD
        print()
        print(f"[spin] motor at {start:+.2f} deg -> {target:+.2f} deg "
              f"at {args.dps:.0f} motor-dps")
        moved = _hold(mb, can_id, target, args.hold, args.dps, args.rate, unwrap)
        print(f"[spin] motor moved {moved - start:+.2f} deg "
              f"(commanded {args.step * DEG_PER_RAD:+.2f})")
        print("[spin] returning")
        _hold(mb, can_id, start, args.hold, args.dps, args.rate, unwrap)
        mb.stop(can_id)

    if args.joint is not None:
        print()
        print("[spin] write what you saw into hw/hardware_map.py -- ONE of:")
        print(HM.record_template(args.joint, can_id, +1, args.note or ""))
        print(HM.record_template(args.joint, can_id, -1, args.note or ""))
        print("[spin] and append a line to hw.VERIFICATION_LOG.")
    return 0


# ===========================================================================
def cmd_check(args) -> int:
    """Re-drive a FILLED-IN joint in JOINT coordinates and confirm it.

    The second pass.  Unlike `spin`, this goes through the measured direction,
    so it exercises the path a controller will and a sign entered backwards
    shows up as a foot going the wrong way.
    """
    index, entry = HM.row_of(args.joint)
    if not entry.wired:
        print(f"[check] {entry.label} has no CAN id or no direction yet.  "
              "Measure it first:", file=sys.stderr)
        print("[check]   python -m hw.bringup scan", file=sys.stderr)
        print("[check]   python -m hw.bringup spin --id <n> --joint "
              f"{entry.label} --go", file=sys.stderr)
        return 2

    label, dx, dy, dz = HM.direction_check_table(args.step)[index]
    print(f"[check] {entry.label}: CAN {entry.can_id}, direction "
          f"{entry.direction:+d}, "
          f"{'CONFIRMED' if entry.confirmed else 'not yet confirmed'}")
    print(f"[check] commanding {args.step:+.3f} rad in JOINT coordinates, "
          "through the measured direction.")
    print(f"[check] EXPECT: the {entry.leg} foot moves "
          f"dx {dx:+.2f}  dy {dy:+.2f}  dz {dz:+.2f} mm  "
          f"(dominant {HM.dominant(dx, dy, dz)})")
    print("[check]   another limb moving     -> the can_id is wrong")
    print("[check]   this foot the other way -> the direction is wrong")
    if not args.go:
        print("[check] dry run.  Add --go to actually command it.")
        return 0

    can_id, direction = entry.can_id, entry.direction
    with _bus([can_id], args.bitrate, args.fake, joint_frame=True) as mb:
        if not mb.arm(rate_hz=args.rate, timeout_s=args.timeout):
            print("[check] the motor did not arm", file=sys.stderr)
            return 1
        mb.flush_rx()
        mb.poll()
        # `position` applies the measured direction on the way out;
        # `encoders_deg` does NOT apply it on the way back, so the direction
        # has to go on the reading by hand.  Mixing those two up is a joint
        # that reads the mirror of where it is.
        unwrap = CAL.EncoderUnwrap()
        start = direction * unwrap.update(mb.rec(can_id).encoder)
        target = start + args.step * DEG_PER_RAD
        print(f"[check] joint at {start:+.2f} deg -> {target:+.2f} deg")
        reached = direction * _hold(mb, can_id, target, args.hold, args.dps,
                                    args.rate, unwrap)
        print(f"[check] joint moved {reached - start:+.2f} deg "
              f"(commanded {args.step * DEG_PER_RAD:+.2f})")
        print("[check] returning")
        _hold(mb, can_id, start, args.hold, args.dps, args.rate, unwrap)
        mb.stop(can_id)

    print()
    print("[check] if the limb and the direction were both right, set "
          "confirmed=True on that row:")
    print(HM.record_template(entry.label, can_id, direction, entry.note,
                             confirmed=True))
    print("[check] and append a line to hw.VERIFICATION_LOG.")
    return 0


# ===========================================================================
def cmd_setzero(args) -> int:
    """Write the drivers' encoder offsets at the calibration pose (0x19)."""
    if not HM.is_complete() and not args.force:
        print("[setzero] REFUSED: %d rows of the map are still empty (%s)."
              % (len(HM.unassigned()), ", ".join(HM.unassigned())))
        print("[setzero] A zero is written per CAN id, so a zero written "
              "through a map that does not know which id is which joint is a "
              "zero on the wrong joint -- and 0x19 cannot be undone.")
        print("[setzero] Build the map first: `python -m hw.bringup scan`.")
        return 2
    if HM.confirmed_count() < HM.N_JOINTS and not args.force:
        print("[setzero] REFUSED: %d joints are filled in but not confirmed "
              "(%s)." % (len(HM.unconfirmed()), ", ".join(HM.unconfirmed())))
        print("[setzero] Run `check` on each first, or pass --force if you "
              "know why.")
        return 2
    print("[setzero] The robot must be FLAT ON ITS BELLY, all four legs")
    print("[setzero] straight out fore-and-aft, knees straight -- that is")
    print("[setzero] coordinates.Q_ZERO, and every angle in this project is")
    print("[setzero] measured from it.")
    print("[setzero]")
    print("[setzero] 0x19 affects driver lifetime, takes effect only after a")
    print("[setzero] POWER CYCLE, and cannot be undone from software.")
    if not args.yes:
        print("[setzero] dry run.  Add --yes once the robot is posed.")
        return 0

    ids = HM.motor_ids()
    with _bus(ids, args.bitrate, args.fake, joint_frame=True) as mb:
        if not mb.arm(rate_hz=args.rate, timeout_s=args.timeout):
            print("[setzero] not every motor armed -- refusing to write zeros",
                  file=sys.stderr)
            return 1
        result = CAL.set_zero_all(mb, confirm=True)

    print()
    failed = [mid for mid, offset in result.items() if offset is None]
    for entry in HM.HARDWARE_JOINTS:
        offset = result.get(entry.can_id)
        print("  CAN %2d  %-10s  %s"
              % (entry.can_id, entry.label,
                 "offset %d" % offset if offset is not None
                 else "NO ACK -- NOT WRITTEN"))
    if failed:
        print(f"[setzero] {len(failed)} motors did not ack: {failed}",
              file=sys.stderr)
        return 1
    print("[setzero] written.  POWER CYCLE, then `python -m hw.bringup scan` "
          "and check every encoder reads about 0.")
    return 0


# ===========================================================================
def cmd_imu(args) -> int:
    """Is the IMU streaming, and do its numbers look like a level robot?"""
    from . import imu as IMU

    print(IMU.describe())
    if not args.go:
        print()
        print("[imu] dry run.  Add --go to open the device.")
        return 0
    with IMU.ImuDog(port=args.port) as sensor:
        if not sensor.wait_for_data(timeout=2.0):
            print("[imu] no packets in 2 s -- check the port and the cable",
                  file=sys.stderr)
            return 1
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            s = sensor.sample()
            print("  roll %+7.2f  pitch %+7.2f  yaw %+7.2f deg   "
                  "rates %+7.2f %+7.2f %+7.2f dps   %5.1f Hz  age %.3f s"
                  % (s.roll_deg, s.pitch_deg, s.yaw_deg, s.roll_rate_dps,
                     s.pitch_rate_dps, s.yaw_rate_dps, sensor.rate_hz, s.age_s))
            time.sleep(0.2)
        if args.capture_trim:
            trim = sensor.capture_offsets(duration_s=1.0)
            print(f"[imu] trim captured: roll {trim[0]:+.3f} pitch "
                  f"{trim[1]:+.3f} deg -> {sensor.save_calib()}")
    return 0


# ===========================================================================
def cmd_plan(args) -> int:
    """Where the bring-up has got to."""
    wired, total = len(HM.assigned()), HM.N_JOINTS
    steps = [
        ("1  scan      the CAN ids that answer",
         None, "python -m hw.bringup scan"),
        ("2  spin      which id is which joint, and which way   (%d/%d wired)"
         % (wired, total),
         HM.is_complete(), "python -m hw.bringup spin --id <n> --go"),
        ("3  check     each row re-driven in joint coordinates  (%d/%d)"
         % (HM.confirmed_count(), total),
         HM.confirmed_count() == total,
         "python -m hw.bringup check --joint <label> --go"),
        ("4  setzero   encoders zeroed flat on the belly",
         None, "python -m hw.bringup setzero --yes"),
        ("5  imu       mounting rotation measured, not assumed",
         _imu_measured(), "python -m hw.bringup imu --go"),
        ("6  torque    first run at %s N*m, robot supported" % _tau_start(),
         CONFIRMED_ON_DOG6, None),
    ]
    print("DOG6 hardware bring-up")
    for title, done, how in steps:
        print("  %s %s" % ({True: "[x]", False: "[ ]", None: "[?]"}[done], title))
        if how and done is not True:
            print("          %s" % how)
    print()
    print("  hardware_map         %d/%d wired, %d/%d confirmed"
          % (wired, total, HM.confirmed_count(), total))
    print("  hw.CONFIRMED_ON_DOG6 %s" % CONFIRMED_ON_DOG6)
    print("  VERIFICATION_LOG     %s"
          % (VERIFICATION_LOG.strip() or "(empty -- nothing has been seen yet)"))
    print()
    print("  [?] = this script cannot tell; the step leaves no artifact to check.")
    return 0


def _imu_measured():
    """Has anyone established how the board sits, or is identity a guess?

    THE VALUE CANNOT ANSWER THIS.  Identity is both the obvious placeholder
    and, on DOG6, the right answer -- the board is mounted aligned with the
    trunk -- so `allclose(R_BODY_IMU, I)` reads the same either way and used
    to report a measured mounting as an outstanding bring-up step.  The flag
    is the only thing that distinguishes them.
    """
    from sim import coordinates as C
    return bool(C.R_BODY_IMU_MEASURED)


def _tau_start():
    from .safety import TAU_START_MAX
    return TAU_START_MAX


# ===========================================================================
def main(argv=None) -> int:
    # The bus flags live on a PARENT parser so they work on either side of the
    # subcommand.  `bringup scan --fake` is what an operator types; argparse
    # only accepts it before the subcommand unless the subparsers inherit it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bitrate", type=int, default=1_000_000)
    common.add_argument("--rate", type=float, default=250.0,
                        help="keep-alive rate per motor, Hz")
    common.add_argument("--timeout", type=float, default=None,
                        help="give up arming after this many seconds")
    common.add_argument("--fake", action="store_true",
                        help="drive hw.fake_bus instead of a real adapter -- "
                             "exercises the whole protocol path with no robot")

    step = argparse.ArgumentParser(add_help=False)
    step.add_argument("--step", type=float, default=STEP_RAD, help="radians")
    step.add_argument("--dps", type=float, default=MOTOR_DPS,
                      help="motor-side speed cap")
    step.add_argument("--hold", type=float, default=HOLD_S)
    step.add_argument("--go", action="store_true", help="actually command it")

    parser = argparse.ArgumentParser(
        prog="hw.bringup", description=__doc__.splitlines()[0],
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", parents=[common],
                       help=cmd_scan.__doc__.splitlines()[0])
    p.add_argument("--ids", default=None,
                   help="comma-separated ids (default: %d..%d)"
                        % (SCAN_RANGE[0], SCAN_RANGE[-1]))
    p.add_argument("--no-arm", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("spin", parents=[common, step],
                       help=cmd_spin.__doc__.splitlines()[0])
    p.add_argument("--id", type=int, required=True, help="raw CAN id")
    p.add_argument("--joint", default=None,
                   help="the joint you believe this is -- prints the sign "
                        "prediction.  One of %s" % ", ".join(HM.JOINT_LABELS))
    p.add_argument("--note", default=None, help="goes into the pasted row")
    p.set_defaults(func=cmd_spin)

    p = sub.add_parser("check", parents=[common, step],
                       help=cmd_check.__doc__.splitlines()[0])
    p.add_argument("--joint", required=True,
                   help="one of %s" % ", ".join(HM.JOINT_LABELS))
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("setzero", parents=[common],
                       help=cmd_setzero.__doc__.splitlines()[0])
    p.add_argument("--yes", action="store_true", help="the robot is posed flat")
    p.add_argument("--force", action="store_true",
                   help="zero an incomplete or unconfirmed map anyway")
    p.set_defaults(func=cmd_setzero)

    p = sub.add_parser("imu", parents=[common],
                       help=cmd_imu.__doc__.splitlines()[0])
    p.add_argument("--port", default=None)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--capture-trim", action="store_true",
                   help="robot flat on a LEVEL floor; writes imu_calib.json")
    p.add_argument("--go", action="store_true", help="open the device")
    p.set_defaults(func=cmd_imu)

    p = sub.add_parser("plan", parents=[common],
                       help=cmd_plan.__doc__.splitlines()[0])
    p.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    if getattr(args, "port", "sentinel") is None:
        from . import imu as IMU
        args.port = IMU.DEFAULT_PORT
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
