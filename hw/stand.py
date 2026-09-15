"""DOG6 stand ON THE ROBOT: `sim.stand`'s sequence, through the real drivers.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.stand --fake --auto 1          the whole path against hw.fake_bus
    $V -m hw.stand --unconfirmed "first stand, robot supported"
    $V -m hw.stand --unconfirmed "..." --tau-cap 2.5

Six phases.  ENTER steps them; X is an E-STOP at any point:

    limp     0xA1 iq=0 keep-alives.  NO TORQUE.  Joint angles and the FK
             height are printed, so a wrong sign or a wrong zero is READ here,
             before anything moves
    settle   driver position mode (0xA4), holding the pose latched at ENTER
    crouch   0xA4, smoothstep to `sim.stand.Q_CROUCH` -- trunk on the floor
    lift     0xA1 TORQUE, `sim.stand.compliance_torque` through
             `hw.safety.SafetyGate`: xy pinned, z lifts the trunk
    park     0xA4, back to Q_CROUCH from wherever the lift left it
    done     0xA4, holding Q_CROUCH.  ENTER exits and stops the motors

`sim.stand` explains why the sequence is shaped this way.  This file only
explains what is different about running it on twelve MG5010 drivers.


WHAT CARRIES OVER FROM SIM, AND WHAT DOES NOT
    Carries   Q_CROUCH, FOOT_XY, CROUCH_HEIGHT, LIFT_HEIGHT, the ramp
              durations, the smoothstep, and the compliance law with its
              KP_CART / KD_CART and weight feedforward -- all imported from
              `sim.stand`, none copied.  A trajectory and a force law at the
              feet are properties of the robot, not of the simulator.
    Does not  KP_JOINT / KD_JOINT.  Position mode here IS the driver's own
              0xA4 loop, closed in the driver with the driver's gains.  The
              python PD in `sim.stand` was only ever emulating it.

    Frames: every command is in JOINT coordinates.  `MotorBus` is opened with
    `hardware_map.motor_directions()`, so 0xA1/0xA4 apply the measured sign on
    the way out, and `calibration.joint_state` applies it on the way back.
    The law uses `hw.kinematics` (0.09 ms for four legs) rather than
    `sim.kinematics` (1.9 ms for the whole law on the Pi); `sim.selftest`
    gates the two against each other at 1e-12.  The leg-gravity term has no
    closed form, so it is refreshed ONE LEG PER SWEEP instead of four.


THE 50 MS INPUT-LOST PROTECTION IS WHAT SHAPES THE LOOP
    Every driver latches error 0x80 and goes limp if it hears nothing for
    `safety.INPUT_LOST_S`.  So this is ONE single-threaded paced loop in which
    every slot sends exactly one frame to exactly one motor, round-robin,
    from the moment `arm()` returns to the moment the motors are stopped:

      * nothing after `arm()` blocks -- no `status1()`, no `flush_rx()`, no
        `input()`.  Keys are polled with a zero-timeout select;
      * the control law runs once per sweep, at slot 0.  If it overruns, the
        schedule RE-ANCHORS instead of catching up, because catching up is a
        burst of back-to-back frames into a 10-frame SocketCAN TX queue;
      * the driver fault byte only arrives in a 0x9A reply, so one 0x9A
        replaces one control frame every other sweep, rotating over the
        twelve.  That motor's command gap becomes two sweeps, 8 ms at 250 Hz;
      * before every send the gap since that motor's previous frame is
        measured.  Past `GAP_ESTOP_S` -- half the window -- the run stops
        while the drivers are still listening, rather than finding out from a
        0x80 that one of them already went limp.  A 0x80 seen anyway is its
        own e-stop.


AN E-STOP IS A LIMP ROBOT, AND DURING THE LIFT THAT IS A DROP
    X, Ctrl-C, a trip and a crash all end in `MotorBus.close()`: 0x81 stop
    then iq=0 to every motor.  From limp, settle, crouch, park and done that
    is harmless -- the crouch holds at zero torque (sim measured 0.1 mm of
    trunk drop in 3 s released).  From LIFT it drops the trunk onto its belly
    from up to 157 mm.  Run the first lifts with the robot supported.

    `--tau-cap` defaults to `safety.TAU_START_MAX` = 1.0 N*m, which CANNOT
    lift the robot: the lift phase will push, saturate, and leave the trunk on
    the floor.  That is the intended first run.  Standing needs about 2.2;
    `TAU_STAGED_MAX` = 3.0 is the ceiling `SafetyGate` allows.

    Position mode has NO torque cap -- 0xA4 carries a speed limit and nothing
    else, so the crouch and park transients are limited by the driver's own
    configuration, not by `--tau-cap`.  What bounds them here is the
    per-joint speed cap and the tracking-error trip.


THE MAP
    Confirmed on the robot by the operator, 2026-09-15 (CAN 3, 6, 7, 9, 10, 12
    are -1), so `hw.CONFIRMED_ON_DOG6` is True and `--unconfirmed` is not
    needed.  If the map is ever re-opened, `SafetyGate` refuses again until
    `--unconfirmed "why"` is given.  The limp phase stays the place to look:
    move a foot by hand and watch the printed angle go the way
    `hw.hardware_map` predicts.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/stand.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import safety as SAFE         # noqa: E402
from .motor import MAX_SPEED_POS     # noqa: E402

__all__ = ["PHASES", "GAP_ESTOP_S", "HardwareStand", "run", "main"]

PHASES = ("limp", "settle", "crouch", "lift", "park", "done")
BLURB = {
    "limp":   "NO TORQUE -- check the angles, move a foot by hand",
    "settle": "driver position mode, holding the pose it was in",
    "crouch": "driver position mode -> crouch (trunk on the floor)",
    "lift":   "TORQUE mode, Cartesian compliance, xy pinned, z lifts",
    "park":   "driver position mode -> crouch",
    "done":   "driver position mode, holding crouch.  ENTER exits",
}

#: Per-motor command rate.  DOG5's 250 Hz: 3000 frames/s out and 3000 back is
#: about 72 % of a 1 Mbit/s bus, and 4 ms is far inside the 50 ms window.
RATE_HZ = 250.0

#: Stop while the drivers are still listening: half their input-lost window.
GAP_ESTOP_S = 0.5 * SAFE.INPUT_LOST_S

#: A 0x9A replaces one control frame every this many sweeps, rotating.
STATUS_EVERY_SWEEPS = 2

#: Low-pass on the finite-differenced encoder, the velocity the law damps
#: with.  The driver's own speed field is kept as the safety gate's witness
#: only -- see `calibration.joint_state` for why it is not trusted alone.
QD_FILTER_HZ = 40.0

#: 0xA4 speed caps, MOTOR side (1 dps/LSB, 10:1 to the output).  A ramp's cap
#: is its smoothstep peak rate times SPEED_MARGIN, so the driver can follow
#: the reference but a frame error cannot run away at full speed.
SETTLE_MOTOR_DPS = 60.0
MIN_MOTOR_DPS = 60.0
SPEED_MARGIN = 1.5

#: Tracking-error trips in position mode.  Settle's is tight on purpose: it
#: holds the pose it just READ, so any real error there means the driver's
#: position frame and the encoder disagree -- found at 6 deg/s, not at speed.
SETTLE_TRACK_ESTOP = np.deg2rad(5.0)
TRACK_ESTOP = np.deg2rad(15.0)

STATUS_PERIOD_S = 0.5


def ramp_motor_dps(q_from, q_to, seconds: float) -> np.ndarray:
    """(12,) motor-side 0xA4 caps for a smoothstep from `q_from` to `q_to`.

    The smoothstep's peak slope is pi/2, so its peak joint rate is
    ``pi/2 * |dq| / T``.
    """
    peak_rad_s = 0.5 * np.pi * np.abs(C.flat(q_to) - C.flat(q_from)) / seconds
    motor_dps = np.rad2deg(peak_rad_s) * P.GEAR_RATIO * SPEED_MARGIN
    return np.clip(motor_dps, MIN_MOTOR_DPS, MAX_SPEED_POS)


class HardwareStand:
    """The phase machine.  Pure decisions: no bus, no clock of its own.

    `update` is called once per sweep with the measured state and returns
    what every motor should be sent this sweep.  Keeping it free of I/O is what
    lets `--fake` exercise exactly the object the robot runs.
    """

    def __init__(self, gate: SAFE.SafetyGate):
        self.gate = gate
        self.phase = 0
        self.t_phase = 0.0
        self.q_ref0 = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        self.q_des = np.zeros(C.N_JOINTS)
        self.h_cmd = ST.CROUCH_HEIGHT
        self.gravity = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        self.tau = np.zeros(C.N_JOINTS)
        self.tau_request = np.zeros(C.N_JOINTS)
        self.tau_peak = 0.0
        self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        self._sweep = 0

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase]

    @property
    def finished(self) -> bool:
        return self.phase_name == "done"

    def ramp_remaining(self, now: float) -> float:
        """Seconds until the current position ramp has arrived; 0 if none."""
        seconds = {"crouch": ST.RAMP_POSITION, "park": ST.RAMP_POSITION,
                   "lift": ST.RAMP_LIFT}.get(self.phase_name, 0.0)
        return max(0.0, seconds - (now - self.t_phase))

    def advance(self, now: float, q) -> str | None:
        """Enter the next phase.  Returns why it refused, or None."""
        if self.finished:
            return None
        if self.phase_name in ("crouch", "park") and self.ramp_remaining(now) > 0:
            return ("%s ramp still running, %.1f s left"
                    % (self.phase_name, self.ramp_remaining(now)))
        self.phase += 1
        self.t_phase = now
        # From WHERE THE ROBOT IS, as in sim.stand -- after the lift that is
        # wherever the compliance law settled, not Q_CROUCH.
        self.q_ref0 = C.unflat(q).copy()
        name = self.phase_name
        if name == "settle":
            self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        elif name in ("crouch", "park"):
            self.max_dps = ramp_motor_dps(self.q_ref0, ST.Q_CROUCH,
                                          ST.RAMP_POSITION)
        elif name == "lift":
            self.gate.start(now, q=q)
            q4 = C.unflat(q)
            self.gravity = np.stack([SK.leg_gravity_torque(i, q4[i])
                                     for i in range(C.N_LEGS)])
        elif name == "done":
            self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        return None

    def update(self, now: float, q, qd):
        """One sweep.  Returns ``(mode, values, trip)``.

        `mode` is "keepalive" (values None), "position" (values: (12,) joint
        rad) or "torque" (values: (12,) N*m).  `trip` is a reason to stop, or
        None.
        """
        self._sweep += 1
        name = self.phase_name
        elapsed = now - self.t_phase
        q4 = C.unflat(q)

        if name == "limp":
            self.h_cmd = float("nan")
            return "keepalive", None, None

        if name == "lift":
            alpha = ST.smoothstep(elapsed / ST.RAMP_LIFT)
            self.h_cmd = ST.CROUCH_HEIGHT + alpha * (ST.LIFT_HEIGHT
                                                     - ST.CROUCH_HEIGHT)
            leg = self._sweep % C.N_LEGS
            self.gravity[leg] = SK.leg_gravity_torque(leg, q4[leg])
            request = ST.compliance_torque(q4, C.unflat(qd), self.h_cmd,
                                           kin=HK, gravity=self.gravity)
            self.tau_request = C.flat(request)
            self.tau = self.gate.apply(self.tau_request, q, now)
            self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))
            return "torque", self.tau, None

        self.tau = np.zeros(C.N_JOINTS)
        if name == "settle":
            target = self.q_ref0
            limit = SETTLE_TRACK_ESTOP
        elif name in ("crouch", "park"):
            alpha = ST.smoothstep(elapsed / ST.RAMP_POSITION)
            target = self.q_ref0 + alpha * (ST.Q_CROUCH - self.q_ref0)
            limit = TRACK_ESTOP
        else:                                            # done
            target = ST.Q_CROUCH
            limit = TRACK_ESTOP
        self.h_cmd = ST.CROUCH_HEIGHT if name != "settle" else float("nan")
        self.q_des = C.flat(target)

        error = np.abs(q - self.q_des)
        if np.any(error > limit):
            index = int(np.argmax(error))
            trip = ("%s: %s is %.1f deg from its position target (limit %.0f)"
                    % (name, HM.JOINT_LABELS[index], np.rad2deg(error[index]),
                       np.rad2deg(limit)))
            if name == "settle":
                trip += (" -- the driver's position frame disagrees with the "
                         "encoder, or something is pushing the leg")
            return "position", self.q_des, trip
        return "position", self.q_des, None


# ===========================================================================
# the loop
# ===========================================================================
class KeyPoller:
    """Non-blocking single keys from a POSIX tty; `ok` is False without one.

    cbreak, not raw, so Ctrl-C still raises KeyboardInterrupt and the
    `finally` still stops the motors.  Always `restore()`.
    """

    def __init__(self):
        self.ok = False
        self._saved = None
        try:
            import termios
            import tty
            if not sys.stdin.isatty():
                return
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.ok = True
        except Exception:                # noqa: BLE001
            self.restore()

    def get(self):
        if not self.ok:
            return None
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        if self._saved is not None:
            import termios
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:            # noqa: BLE001
                pass
        self._saved = None
        self.ok = False


def _print_phase(stand: HardwareStand) -> None:
    print("\n>> phase %d %-6s : %s" % (stand.phase, stand.phase_name,
                                       BLURB[stand.phase_name]), flush=True)


def run(mb, stand: HardwareStand, *, rate_hz: float = RATE_HZ, key=None,
        auto_s: float | None = None, clock=time.perf_counter) -> str | None:
    """Drive `stand` on an ARMED `mb` until done, X, or a trip.

    Returns the stop reason, or None for a clean exit from "done".  Does not
    stop the motors -- the caller's `with MotorBus(...)` does, on every path.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)

    sweep = 0
    mode, values = "keepalive", None
    q_prev = t_prev = None
    qd_ctrl = np.zeros(n)
    q = np.zeros(n)
    worst_gap = np.zeros(n)
    overruns = 0
    last_status = stand.t_phase = clock()
    _print_phase(stand)

    deadline = clock() + slot
    k = 0
    while True:
        mb.poll()
        if k == 0:
            now = clock()
            q, qd_driver = CAL.joint_state(mb, unwrappers)
            if q_prev is not None and now > t_prev:
                qd_raw = (q - q_prev) / (now - t_prev)
                qd_ctrl += alpha_qd * (qd_raw - qd_ctrl)
            q_prev, t_prev = q, now

            # -- operator ---------------------------------------------------
            pressed = key.get() if key is not None else None
            if pressed in ("x", "X"):
                return "operator X"
            wants_next = pressed in ("\r", "\n")
            # --auto: `auto_s` after the phase's ramp has ARRIVED.  Asking
            # ramp_remaining about `auto_s` ago is exactly that test.
            if (auto_s is not None and now - stand.t_phase >= auto_s
                    and stand.ramp_remaining(now - auto_s) <= 0.0):
                wants_next = True
            if wants_next:
                if stand.finished:
                    return None
                refused = stand.advance(now, q)
                if refused:
                    print("\n   ENTER ignored: " + refused, flush=True)
                else:
                    _print_phase(stand)

            # -- the law ----------------------------------------------------
            mode, values, trip = stand.update(now, q, qd_ctrl)
            if trip:
                return trip

            # -- the trips --------------------------------------------------
            errors = mb.errors()
            latched = [mid for mid, err in errors.items() if err & 0x80]
            if latched:
                return ("input-lost latch (0x80) on CAN %s: a driver heard "
                        "nothing for %.0f ms and went limp"
                        % (latched, 1e3 * SAFE.INPUT_LOST_S))
            temps = np.asarray([mb.rec(mid).temp or 0 for mid in ids])
            reason = stand.gate.estop_reason(
                q, qd_driver, now, temps=temps,
                miss_streaks=misses.update(mb), errors=errors)
            if reason:
                return reason

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                deg = np.rad2deg(C.unflat(q))
                print("   %-6s t=%5.1f  h_cmd=%.4f h_fk=%.4f  |tau|=%.2f/%.2f"
                      "  gap=%.1f ms  overrun=%d  %s"
                      % (stand.phase_name, now - stand.t_phase, stand.h_cmd,
                         ST.height_from_fk(C.unflat(q)),
                         float(np.abs(stand.tau).max()), stand.tau_peak,
                         1e3 * worst_gap.max(), overruns,
                         " ".join("%s(%+.0f,%+.0f,%+.0f)" % (leg, *deg[i])
                                  for i, leg in enumerate(C.LEGS))),
                      flush=True)
            sweep += 1

        # -- one frame, to one motor --------------------------------------
        mid = ids[k]
        sent_at = clock()
        previous = mb.rec(mid).last_cmd_t
        if previous is not None:
            gap = sent_at - previous
            worst_gap[k] = max(worst_gap[k], gap)
            if gap > GAP_ESTOP_S:
                return ("CAN %d went %.1f ms without a frame, past the %.0f ms "
                        "stop line (the drivers' window is %.0f ms)"
                        % (mid, 1e3 * gap, 1e3 * GAP_ESTOP_S,
                           1e3 * SAFE.INPUT_LOST_S))
        if (sweep % STATUS_EVERY_SWEEPS == 0
                and k == (sweep // STATUS_EVERY_SWEEPS) % n):
            mb.status1_req(mid)
        elif mode == "position":
            mb.position(mid, float(np.rad2deg(values[k])),
                        max_dps=float(stand.max_dps[k]))
        elif mode == "torque":
            mb.torque(mid, float(values[k]))
        else:
            mb.keepalive(mid)

        k = (k + 1) % n
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun > 3 * slot:
            # RE-ANCHOR, do not catch up: catching up is a burst of frames.
            overruns += 1
            deadline = clock() + slot
    # unreachable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tau-cap", type=float, default=SAFE.TAU_START_MAX,
                    help="lift-phase torque cap, N*m (SafetyGate allows up to "
                         "%.1f)" % SAFE.TAU_STAGED_MAX)
    ap.add_argument("--rate", type=float, default=RATE_HZ,
                    help="per-motor command rate, Hz")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--unconfirmed", default=None, metavar="WHY",
                    help="run although hw.CONFIRMED_ON_DOG6 is False")
    ap.add_argument("--fake", action="store_true",
                    help="hw.fake_bus instead of can0: protocol only, no "
                         "dynamics")
    ap.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                    help="advance phases by themselves, SECONDS after each "
                         "ramp arrives (--fake only)")
    args = ap.parse_args(argv)

    if args.auto is not None and not args.fake:
        ap.error("--auto is only allowed with --fake: on the robot a person "
                 "steps the phases")
    if args.rate * 2 * STATUS_EVERY_SWEEPS < 1.0 / GAP_ESTOP_S:
        ap.error("--rate %.0f Hz cannot keep every motor inside the %.0f ms "
                 "stop line" % (args.rate, 1e3 * GAP_ESTOP_S))
    reason = args.unconfirmed or ("hw.fake_bus" if args.fake else None)

    # Everything that can refuse, refuses BEFORE the bus opens.
    ids = HM.motor_ids()
    try:
        gate = SAFE.SafetyGate(args.tau_cap, unconfirmed_reason=reason)
    except (RuntimeError, ValueError) as refusal:
        print("[stand] REFUSED before opening the bus:\n  %s" % refusal,
              file=sys.stderr)
        if not CONFIRMED_ON_DOG6 and reason is None:
            print('[stand] to run on the unconfirmed map anyway: '
                  '--unconfirmed "why"', file=sys.stderr)
        return 2
    stand = HardwareStand(gate)
    key = KeyPoller()
    if not key.ok and not args.fake:
        print("[stand] stdin is not a terminal, so neither ENTER nor the X "
              "e-stop can reach this process.  Run it from a shell.",
              file=sys.stderr)
        return 2

    print("DOG6 stand on %s" % ("hw.fake_bus" if args.fake else "can0"))
    print("  CAN ids %s  directions %s" % (ids, HM.directions()))
    print("  map confirmed: %s%s" % (CONFIRMED_ON_DOG6,
                                     "" if CONFIRMED_ON_DOG6
                                     else "  (running on: %r)" % reason))
    print("  lift tau cap %.2f N*m (standing needs ~2.2), crouch %.3f m -> "
          "lift %.3f m" % (args.tau_cap, ST.CROUCH_HEIGHT, ST.LIFT_HEIGHT))
    print("  %.0f Hz per motor, %.1f ms sweep; stop line %.0f ms of the "
          "drivers' %.0f ms input-lost window"
          % (args.rate, 1e3 / args.rate, 1e3 * GAP_ESTOP_S,
             1e3 * SAFE.INPUT_LOST_S))
    print("  ENTER steps the phase.  X is an E-STOP -- during lift it DROPS "
          "the robot.")

    from .motor import motorbus
    if args.fake:
        from .fake_bus import FakeDriverBus
        bus = FakeDriverBus(ids=ids)
        mb = motorbus.MotorBus(ids, bus=bus, dirs=HM.motor_directions())
    else:
        mb = motorbus.MotorBus(ids, bitrate=args.bitrate,
                               dirs=HM.motor_directions())

    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[stand] not every motor armed", file=sys.stderr)
                return 1
            # Straight into the loop: arm() streamed until this instant, and
            # nothing may sit between it and the first slot.
            stop = run(mb, stand, rate_hz=args.rate, key=key,
                       auto_s=args.auto)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()

    print()
    if stop is None:
        print("[stand] done; motors stopped in the crouch.  Peak lift torque "
              "%.2f N*m." % stand.tau_peak)
        return 0
    print("[stand] E-STOP in phase %s: %s" % (stand.phase_name, stop))
    print("[stand] motors stopped.")
    return 0 if stop == "operator X" else 1


if __name__ == "__main__":
    raise SystemExit(main())
