#!/usr/bin/env python3
"""
motorbus.py -- the clean, reusable CAN motor-control library for the LKMotor /
MG5010E-i10 rig (SocketCAN can0 @ 1 Mbit/s, up to 12 motors on one bus).

Two layers:

  RoundRobinBus (+ MotorRecord)   low-level non-blocking send/match. One shared
                                  bus, N motors, at most one outstanding request
                                  per motor; replies matched by arbitration id.
                                  This is the ONLY correct primitive for a
                                  multi-motor high-rate loop: LKMotor's blocking
                                  request-reply discards frames from other ids,
                                  which destroys interleaved replies.

  MotorBus                        high-level facade over RoundRobinBus: bus
                                  lifecycle (context manager), arming, unit-aware
                                  control (torque N*m / speed dps / position deg),
                                  over-CAN fault recovery, batch status reads and
                                  telemetry accessors.

Arming and recovery use the t7-proven method: the minimal ladder 0x9B (clear)
-> 0x88 (run) interleaved with a zero-torque keep-alive (0xA1, iq=0), FOLLOWED
by a paced keep-alive settle stream and a status re-read. t7_recover_no_
powercycle.py measured on hardware that this -- ladder + settle + verify, not
the one-shot ladder alone -- clears the input-signal-lost latch (status-1 error
bit 7 = 0x80) over CAN, with NO 0x80 shutdown step and NO power cycle. So there
is no `prime_watchdog` and no "power-cycle between runs" here: just open the
bus, arm(), and stream commands (MotorBus.recover() runs the full method).

Strictly single-threaded: the caller owns the control loop
(`mid = ids[i % n]; mb.poll(); mb.torque(mid, tau); mb.pace(deadline)`). A reader
thread under the GIL adds ms-scale jitter on the Pi; a poll loop is deterministic
and measurable.

Motor unit constants default to config.py (the existing single source -- no new
copy); every MotorBus method takes per-instance overrides for a config-less
consumer.
"""
import struct
import time
from typing import Optional

import can

import motor_gains as config
from motor_library import open_bus                      # keep the bus opener there

# ---------------------------------------------------------------------------
# Bus / protocol constants
# ---------------------------------------------------------------------------
MOTOR_IDS = list(range(1, 13))
REPLY_BASE = 0x140
CAN_IFACE = "can0"

# LK single-motor command bytes (protocol V2.36); replies echo the command byte.
CMD_STATUS1 = 0x9A   # temp / voltage / current / state / error flags
CMD_CLEAR = 0x9B     # clear error flags
CMD_STATUS2 = 0x9C   # temp / iq / speed / encoder
CMD_SHUTDOWN = 0x80
CMD_STOP = 0x81
CMD_RUN = 0x88
CMD_TORQUE = 0xA1    # iq control (iq=0 == watchdog keep-alive, no motion)
CMD_SPEED = 0xA2     # speed loop: [0x00, iq_lo, iq_hi, spd int32]
CMD_SET_ZERO = 0x19  # set current encoder position as zero (writes encoder
                     # offset; vendor warns it affects driver lifetime -- use
                     # rarely; takes effect only after a power-cycle)

# errorState bit meanings (status-1 reply), same table as read_state.py.
ERROR_BITS = [
    "low voltage", "high voltage", "driver over-temp", "motor over-temp",
    "over-current", "short circuit", "stall", "input signal lost timeout",
]

# Rough wire cost of one standard 11-bit 8-byte data frame incl. average bit
# stuffing (~111 nominal + stuff bits). Used only for the bus-load estimate.
FRAME_BITS = 120
BITRATE = 1_000_000

# ---------------------------------------------------------------------------
# Motor unit gains -- default to config.py (single source; no new copy).
# Overridable per MotorBus instance for a config-less consumer.
# ---------------------------------------------------------------------------
ENCODER_GAIN = config.encoder_gain      # output-deg = raw_encoder * ENCODER_GAIN
                                        # (config.py: gain -> OUTPUT-shaft angle)
TORQUE_GAIN = config.torque_gain        # iq_LSB = tau_Nm * TORQUE_GAIN
VEL_GAIN = config.vel_gain              # speed_LSB = dps * VEL_GAIN (command)
VEL_STATE_GAIN = config.vel_state_gain  # output_dps = raw_speed / VEL_STATE_GAIN
POS_GAIN = config.pos_gain              # angle_LSB = output_deg * POS_GAIN
MAX_SPEED_POS = config.max_speed_pos    # default position max speed (motor dps)

IQ_HARD = 2048                          # protocol max |iq| for 0xA1/0xA2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def decode_errors(err: int) -> str:
    flags = [name for i, name in enumerate(ERROR_BITS) if err & (1 << i)]
    return "none" if not flags else ", ".join(flags)


def i16(value: float) -> list:
    return list(struct.pack("<h", int(value)))


def i32(value: float) -> list:
    return list(struct.pack("<i", int(value)))


def speed_cmd_data(iq_limit: int, speed_lsb: int) -> list:
    """0xA2 payload (7 data bytes): [0x00, iq int16, speed int32]."""
    return [0x00] + i16(iq_limit) + i32(speed_lsb)


def pace(deadline: float) -> float:
    """Sleep-then-busy-wait until perf_counter() == deadline.

    Returns the overrun in seconds (0.0 if the deadline was met). Overruns are
    Pi/Python scheduler jitter, not bus problems -- scripts count them
    separately so the two bottlenecks stay distinguishable. Tip: running under
    `chrt -f 50` usually cuts the overrun count; measure the difference.
    """
    now = time.perf_counter()
    remaining = deadline - now
    if remaining <= 0:
        return -remaining
    if remaining > 0.0015:
        time.sleep(remaining - 0.001)
    while time.perf_counter() < deadline:
        pass
    return 0.0


def flush_rx(bus: "can.BusABC", settle_s: float = 0.05) -> int:
    """Drop every frame already sitting in the RX queue, then keep draining for
    `settle_s` so in-flight stragglers (a reply to the previous process's last
    command is still on the wire when this process opens the socket) are caught
    too. Returns the number of frames dropped. Never raises.

    Run this before the first poll() of a run: a stale reply from a previous
    run would otherwise be matched against THIS run's first request (the
    command-byte check catches a mismatched command, but a stale reply to the
    same command byte is indistinguishable) and its telemetry would seed the
    loop with dead data.
    """
    dropped = 0
    deadline = time.perf_counter() + settle_s
    while True:
        try:
            msg = bus.recv(timeout=0.0)
        except Exception:
            break
        if msg is not None:
            dropped += 1
            continue
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.002)
    return dropped


# ---------------------------------------------------------------------------
# RoundRobinBus -- non-blocking send/match layer
# ---------------------------------------------------------------------------
class MotorRecord:
    """Per-motor counters + last-known telemetry for one test step."""

    __slots__ = ("sent", "rcvd", "missed", "send_fail", "unexpected",
                 "latencies", "pending_t", "pending_cmd", "last_cmd_t",
                 "max_gap", "temp", "voltage", "current", "state", "error",
                 "error_ever", "iq", "speed", "encoder", "encoder_offset",
                 "last_reply_t")

    def __init__(self):
        self.reset()
        # telemetry survives reset() only within a run; init all here
        self.temp = None
        self.voltage = None
        self.current = None
        self.state = None
        self.error = None
        self.error_ever = 0
        self.iq = None
        self.speed = None
        self.encoder = None
        self.encoder_offset = None     # last 0x19 reply (set-zero) encoder offset
        self.last_reply_t = None

    def reset(self):
        self.sent = 0
        self.rcvd = 0
        self.missed = 0
        self.send_fail = 0
        self.unexpected = 0
        self.latencies = []
        self.pending_t = None
        self.pending_cmd = None
        self.last_cmd_t = None
        self.max_gap = 0.0


class RoundRobinBus:
    """One shared bus, N motors, at most one outstanding request per motor.

    send() is fire-and-forget (a full TX queue is counted, not raised);
    poll() drains the RX queue and matches replies by arbitration id.
    A new send() to a motor whose previous request is still unanswered counts
    a MISS -- so the effective reply deadline is the motor's own command
    period, which is exactly the number that matters against the input
    protection window.
    """

    def __init__(self, bus: "can.BusABC", motor_ids=MOTOR_IDS):
        self.bus = bus
        self.motor_ids = list(motor_ids)
        self.rec = {mid: MotorRecord() for mid in self.motor_ids}
        self.foreign = 0        # RX frames from ids outside motor_ids
        self.error_frames = 0
        self.t_start = time.perf_counter()

    def reset_stats(self):
        for rec in self.rec.values():
            rec.reset()
        self.foreign = 0
        self.error_frames = 0
        self.t_start = time.perf_counter()

    # -- TX ------------------------------------------------------------

    def send(self, mid: int, cmd: int, data7: Optional[list] = None) -> bool:
        rec = self.rec[mid]
        now = time.perf_counter()
        if rec.pending_t is not None:          # previous request unanswered
            rec.missed += 1
            rec.pending_t = None
        if rec.last_cmd_t is not None:
            gap = now - rec.last_cmd_t
            if gap > rec.max_gap:
                rec.max_gap = gap
        rec.last_cmd_t = now

        msg = can.Message(arbitration_id=REPLY_BASE + mid,
                          data=[cmd] + (data7 if data7 else [0x00] * 7),
                          is_extended_id=False)
        try:
            self.bus.send(msg)
        except (can.CanOperationError, OSError):
            # ENOBUFS at overload: the TX queue is full. A measurement, not a
            # crash -- it means the bus (or queue) cannot sustain this rate.
            rec.send_fail += 1
            return False
        rec.sent += 1
        rec.pending_t = now
        rec.pending_cmd = cmd
        return True

    # -- RX ------------------------------------------------------------

    def poll(self) -> int:
        """Drain the RX queue completely; returns matched-reply count."""
        n = 0
        while True:
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                return n
            if msg.is_error_frame:
                self.error_frames += 1
                continue
            # No is_rx filter: own frames are never delivered back
            # (receive_own_messages defaults to False) and vcan marks ALL
            # frames locally-generated, which would discard every reply.
            mid = msg.arbitration_id - REPLY_BASE
            rec = self.rec.get(mid)
            now = time.perf_counter()
            if rec is None:
                self.foreign += 1
                continue
            if rec.pending_t is None:
                # late reply after its slot was re-armed, or a SECOND reply to
                # one request -> duplicate-id smoking gun. Reported per motor.
                rec.unexpected += 1
                self._telemetry(rec, msg.data)
                continue
            if (len(msg.data) > 0 and rec.pending_cmd is not None
                    and msg.data[0] != rec.pending_cmd):
                # Replies echo the command byte. A late reply to an older
                # request must not satisfy the current request (especially a
                # one-shot calibration write such as 0x19).
                rec.unexpected += 1
                self._telemetry(rec, msg.data)
                continue
            rec.latencies.append(now - rec.pending_t)
            rec.pending_t = None
            rec.rcvd += 1
            rec.last_reply_t = now
            self._telemetry(rec, msg.data)
            n += 1

    def wait_reply(self, mid: int, timeout: float) -> bool:
        """Poll until the pending request of `mid` is answered (sequential /
        low-rate use, e.g. the ping scan). True on reply within timeout."""
        rec = self.rec[mid]
        deadline = time.perf_counter() + timeout
        while rec.pending_t is not None:
            self.poll()
            if time.perf_counter() > deadline:
                return False
            time.sleep(0.0002)
        return True

    def finalize(self, timeout: float = 0.05):
        """Drain stragglers at the end of a step; still-unanswered requests
        become misses so the last command of a run is not silently dropped."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.poll()
            time.sleep(0.001)
        for rec in self.rec.values():
            if rec.pending_t is not None:
                rec.missed += 1
                rec.pending_t = None

    # -- telemetry -------------------------------------------------------

    @staticmethod
    def _telemetry(rec: MotorRecord, data) -> None:
        if len(data) < 8:
            return
        cmd = data[0]
        if cmd in (CMD_STATUS1, CMD_CLEAR):
            rec.temp = struct.unpack_from("<b", data, 1)[0]
            rec.voltage = struct.unpack_from("<h", data, 2)[0] * 0.01
            rec.current = struct.unpack_from("<h", data, 4)[0] * 0.01
            rec.state = data[6]
            rec.error = data[7]
            rec.error_ever |= data[7]
        elif cmd in (CMD_STATUS2, CMD_TORQUE, CMD_SPEED, 0xA0, 0xA3, 0xA4,
                     0xA5, 0xA6, 0xA7, 0xA8):
            rec.temp = struct.unpack_from("<b", data, 1)[0]
            rec.iq = struct.unpack_from("<h", data, 2)[0]
            rec.speed = struct.unpack_from("<h", data, 4)[0]
            rec.encoder = struct.unpack_from("<H", data, 6)[0]
        elif cmd == CMD_SET_ZERO:
            # 0x19 reply echoes the new encoder offset (a raw single-turn count,
            # 0..65535) in bytes 6-7. Unsigned, consistent with the 0x9C encoder
            # decode above. Confirms the write landed.
            rec.encoder_offset = struct.unpack_from("<H", data, 6)[0]

    # -- derived metrics ---------------------------------------------------

    def bus_load(self) -> float:
        """Estimated fraction of wire time used by this test's own traffic."""
        elapsed = time.perf_counter() - self.t_start
        frames = sum(r.sent + r.rcvd for r in self.rec.values())
        return frames * FRAME_BITS / BITRATE / max(elapsed, 1e-9)


# ---------------------------------------------------------------------------
# Arming -- multi-motor watchdog-safe startup (minimal t7 recovery ladder)
# ---------------------------------------------------------------------------
def arm_motors(rr: "RoundRobinBus", rate_hz: float = 250.0,
               timeout_s: Optional[float] = 60.0, ladder_retries: int = 4,
               status_period_s: float = 0.1, verbose: bool = True) -> bool:
    """Bring every motor to READY (replying, input-timeout latch cleared)
    without ever starving any motor's input protection.

    There is no power-on `prime_watchdog` step and no power cycle: a motor that
    boots with the input-signal-lost latch set (status-1 error bit 7, 0x80) is
    recovered over CAN with the minimal t7-proven ladder 0x9B -> 0x88 (clear +
    run), streamed in that motor's own slots.

    This routine:
      - streams zero-torque keep-alives (0xA1, iq=0 -- no motion) round-robin
        at rate_hz per motor, non-blocking, so every motor sees a command gap
        of ~2/rate_hz worst-case (8 ms at 250 Hz: inside a 10 ms window);
      - substitutes an occasional 0x9A status read per motor to watch the
        error flags;
      - on a latched motor, runs the minimal ladder 0x9B -> 0x88 in that
        motor's own slots, then re-verifies;
      - after `ladder_retries` failed ladders flags the motor as needing a
        power cycle -- the stream keeps running, so the moment it is
        power-cycled it boots INTO the stream and comes up clean. (A different
        latch -- over-current / stall -- may not clear over CAN; that is when
        needs_cycle fires.)

    A motor also re-latches within its window the moment any script exits, so
    call this at the start of every run, in the same process as the control
    loop that follows.

    Returns True when every motor replies with bit 0x80 clear; False on
    timeout (the per-motor verdicts are printed).
    """
    ids = rr.motor_ids
    n = len(ids)
    slot = 1.0 / (rate_hz * n)
    visits_per_status = max(2, int(status_period_s * rate_hz))
    # Recovery ladder with a keep-alive interleaved: the two config frames
    # (clear, run) never open a gap over 2 command periods (8 ms at 250 Hz), so
    # the very latch being cleared is not re-tripped mid-recovery.
    LADDER = (CMD_CLEAR, CMD_TORQUE, CMD_RUN)

    # Every motor gets one ladder pass even if it boots clean, so all motors
    # leave this function in the RUN state (callers must not follow up with
    # blocking per-motor enables -- that starves the stream).
    st = {mid: {"ladder_i": None, "ran_ladder": False, "attempts": 0,
                "await_status": False,
                "probe_in": 1 + (i % visits_per_status)}   # stagger probes
          for i, mid in enumerate(ids)}
    ready = {mid: False for mid in ids}
    needs_cycle = set()

    if verbose:
        print(f"[arm] zero-torque stream to {ids} at {rate_hz:.0f} Hz/motor "
              "-- power motors on now if they are off (Ctrl-C to abort)")

    t_start = time.perf_counter()
    deadline = t_start + slot
    last_show = 0.0
    idx = 0
    while not all(ready.values()):
        mid = ids[idx % n]
        idx += 1
        rec = rr.rec[mid]
        s = st[mid]

        rr.poll()
        if not ready[mid] and s["await_status"]:
            s["await_status"] = False
            if rec.pending_t is None and rec.error is not None:
                # the status request was answered; rec.error is current
                if not (rec.error & 0x80) and s["ran_ladder"]:
                    ready[mid] = True
                    needs_cycle.discard(mid)
                elif s["attempts"] < ladder_retries:
                    s["attempts"] += 1
                    s["ran_ladder"] = True
                    s["ladder_i"] = 0
                else:
                    needs_cycle.add(mid)
            # no reply: motor is off / unplugged -- keep streaming

        if ready[mid]:
            rr.send(mid, CMD_TORQUE)               # keep feeding it
        elif s["ladder_i"] is not None:
            rr.send(mid, LADDER[s["ladder_i"]])
            s["ladder_i"] += 1
            if s["ladder_i"] >= len(LADDER):
                s["ladder_i"] = None
                s["probe_in"] = 2                  # verify soon
        elif s["probe_in"] <= 0:
            rr.send(mid, CMD_STATUS1)
            s["await_status"] = True
            s["probe_in"] = visits_per_status
        else:
            rr.send(mid, CMD_TORQUE)
            s["probe_in"] -= 1

        now = time.perf_counter()
        if verbose and now - last_show > 0.5:
            pend = [m for m in ids if not ready[m]]
            cyc = sorted(needs_cycle)
            msg = f"  waiting: {pend}"
            if cyc:
                msg += f"  POWER-CYCLE these (stream will catch them): {cyc}"
            print(msg + "   ", end="\r")
            last_show = now
        if timeout_s is not None and now - t_start > timeout_s:
            if verbose:
                pend = [m for m in ids if not ready[m]]
                print(f"\n[arm] timed out; not ready: {pend}"
                      + (f", power-cycle candidates: {sorted(needs_cycle)}"
                         if needs_cycle else ""))
            return False

        pace(deadline)
        deadline += slot

    rr.poll()
    if verbose:
        print(f"\n[arm] all {n} motor(s) ready -- no input-timeout latch.")
    return True


# ---------------------------------------------------------------------------
# E-stop / cleanup
# ---------------------------------------------------------------------------
def stop_all(bus: "can.BusABC", motor_ids=MOTOR_IDS) -> None:
    """Best-effort halt of every motor: 0x81 stop then iq=0, sent twice.
    Never raises -- callable from any finally/except path."""
    for _ in range(2):
        for cmd in (CMD_STOP, CMD_TORQUE):
            for mid in motor_ids:
                try:
                    bus.send(can.Message(arbitration_id=REPLY_BASE + mid,
                                         data=[cmd] + [0x00] * 7,
                                         is_extended_id=False))
                except Exception:
                    pass
            time.sleep(0.005)
    # drop the stop replies so a later reader starts on live data
    try:
        while bus.recv(timeout=0.0) is not None:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MotorBus -- high-level facade
# ---------------------------------------------------------------------------
class MotorBus:
    """High-level motor control over one shared CAN bus.

    Lifecycle (context manager)::

        with MotorBus([1, 2]) as mb:
            mb.arm(rate_hz=250)                 # no prime, no power cycle
            slot = mb.slot(CONTROL_HZ); deadline = time.perf_counter() + slot
            i = 0; ids = mb.ids; n = len(ids)
            while running:
                mid = ids[i % n]; i += 1
                mb.poll()                       # refresh telemetry
                mb.torque(mid, tau_nm)          # or speed()/position()
                mb.pace(deadline); deadline += slot
        # __exit__ ran stop_all() + shutdown() (only if MotorBus opened the bus)

    The caller owns the loop; MotorBus never spawns a thread. Control helpers
    are fire-and-forget; feedback comes from the previous poll() (one slot old).
    """

    def __init__(self, ids, bus=None, *, bitrate: int = 1_000_000, dirs=None,
                 encoder_gain: float = ENCODER_GAIN,
                 torque_gain: float = TORQUE_GAIN,
                 vel_gain: float = VEL_GAIN,
                 vel_state_gain: float = VEL_STATE_GAIN,
                 pos_gain: float = POS_GAIN):
        self.ids = list(ids)
        if bus is None:
            self._bus = open_bus(bitrate=bitrate)
            self._owns_bus = True
        else:
            self._bus = bus
            self._owns_bus = False               # injected bus: caller closes it
        # A previous run's leftover replies (crash paths never reach the
        # stop_all() drain) must not be matched against this run's requests.
        n_stale = flush_rx(self._bus)
        if n_stale:
            print(f"[motorbus] dropped {n_stale} stale RX frame(s) from a "
                  "previous run")
        self._rr = RoundRobinBus(self._bus, self.ids)
        self.dirs = {mid: 1 for mid in self.ids}
        if dirs:
            self.dirs.update(dirs)
        self.encoder_gain = encoder_gain
        self.torque_gain = torque_gain
        self.vel_gain = vel_gain
        self.vel_state_gain = vel_state_gain
        self.pos_gain = pos_gain

    # -- accessors -------------------------------------------------------
    @property
    def bus(self):
        return self._bus

    @property
    def rr(self) -> RoundRobinBus:
        """The underlying RoundRobinBus -- for metric code (rec[mid].missed,
        .max_gap, .latencies, .error_ever, reset_stats(), finalize())."""
        return self._rr

    def rec(self, mid) -> MotorRecord:
        return self._rr.rec[mid]

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "MotorBus":
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Best-effort stop, then shut the bus down IFF this object opened it."""
        try:
            self.stop_all()
        finally:
            if self._owns_bus:
                self._bus.shutdown()

    # -- arming ----------------------------------------------------------
    def arm(self, rate_hz: float = 250.0, timeout_s: Optional[float] = None,
            verbose: bool = True) -> bool:
        """Stream zero-torque keep-alives + run the minimal recovery ladder
        (0x9B -> 0x88) per motor until every motor replies with bit 0x80 clear.
        Power the motors on when prompted; a latched motor is recovered over
        CAN. No prime_watchdog, no power cycle. Returns True on success."""
        return arm_motors(self._rr, rate_hz=rate_hz, timeout_s=timeout_s,
                          verbose=verbose)

    # -- control (fire-and-forget; unit + direction conversion inside) ----
    def torque(self, mid: int, tau_nm: float, iq_max: int = IQ_HARD) -> bool:
        """0xA1 torque loop. tau_nm (output N*m) -> iq LSB, signed by dir and
        clamped to +/-iq_max."""
        iq = int(round(tau_nm * self.torque_gain * self.dirs[mid]))
        iq = max(-iq_max, min(iq_max, iq))
        return self._rr.send(mid, CMD_TORQUE,
                             [0x00, 0x00, 0x00, iq & 0xFF, (iq >> 8) & 0xFF,
                              0x00, 0x00])

    def speed(self, mid: int, dps: float, iq_limit: int = 100) -> bool:
        """0xA2 speed loop. dps (output deg/s) -> speed LSB, signed by dir."""
        lsb = int(round(dps * self.vel_gain * self.dirs[mid]))
        return self._rr.send(mid, CMD_SPEED, speed_cmd_data(iq_limit, lsb))

    def position(self, mid: int, deg: float,
                 max_dps: Optional[float] = None) -> bool:
        """0xA3/0xA4 multi-turn position. deg (output) -> angle LSB, signed by
        dir. max_dps, if given, is the motor-side speed cap (1 dps/LSB)."""
        lsb = int(round(deg * self.pos_gain * self.dirs[mid]))
        if max_dps is None:
            return self._rr.send(mid, 0xA3, [0x00, 0x00, 0x00] + i32(lsb))
        return self._rr.send(mid, 0xA4, [0x00] + i16(int(max_dps)) + i32(lsb))

    def keepalive(self, mid: int) -> bool:
        """0xA1 with iq=0: feed the input-timeout watchdog, no motion."""
        return self._rr.send(mid, CMD_TORQUE)

    def stop(self, mid: int) -> bool:
        """0x81 motor stop (state not cleared)."""
        return self._rr.send(mid, CMD_STOP)

    # -- loop plumbing ---------------------------------------------------
    def poll(self) -> int:
        """Drain RX, refresh telemetry. Call once per slot."""
        return self._rr.poll()

    def flush_rx(self, settle_s: float = 0.05) -> int:
        """Drop everything in the RX queue WITHOUT matching or telemetry
        (see module-level flush_rx). __init__ already ran one; call again
        only to discard replies deliberately, e.g. after an abort."""
        return flush_rx(self._bus, settle_s)

    def slot(self, rate_hz: float) -> float:
        """Per-motor slot period: 1/(rate_hz * n_motors)."""
        return 1.0 / (rate_hz * len(self.ids))

    @staticmethod
    def pace(deadline: float) -> float:
        return pace(deadline)

    def hold(self, seconds: float, rate_hz: float = 250.0) -> None:
        """Round-robin zero-torque keep-alive to every motor for `seconds`
        (feeds the watchdog while nothing else is commanding)."""
        n = len(self.ids)
        slot = 1.0 / (rate_hz * n)
        t_end = time.perf_counter() + seconds
        deadline = time.perf_counter() + slot
        i = 0
        while time.perf_counter() < t_end:
            self._rr.poll()
            self._rr.send(self.ids[i % n], CMD_TORQUE)
            i += 1
            pace(deadline)
            deadline += slot
        self._rr.poll()

    # -- recovery (the t7-proven method) ---------------------------------
    def recover(self, targets=None, *, settle_s: float = 0.5,
                rate_hz: float = 250.0, verify: bool = True) -> bool:
        """Recover the input-signal-lost latch (status-1 error bit 7, 0x80) over
        CAN with the EXACT method proven on hardware by
        t7_recover_no_powercycle.py -- NO 0x80 shutdown, NO power cycle:

          1. minimal ladder -- 0x9B (clear) then 0x88 (run) on each target, with
             one zero-torque keep-alive round to EVERY motor interleaved so
             nothing opens a gap over ~1/rate between the clear and the run;
          2. a paced zero-torque keep-alive SETTLE stream to every motor for
             settle_s (t7's post-ladder stream_keepalive): it lets the 0x88
             "take" and keeps non-target motors fed while the target re-enables;
          3. VERIFY -- re-read status-1 (t7's batch_status1) and report whether
             the 0x80 latch actually cleared.

        Steps 2-3 are what the bare ladder was missing: t7 recovers with
        ladder + settle + re-read, not the one-shot ladder alone. `targets` is an
        id, an iterable of ids, or None = all.

        settle_s BLOCKS (it streams for that long); the default 0.5 s matches t7
        and is right for a bring-up or a dedicated recovery. Inside a hard
        real-time loop that already streams keep-alives every slot (its own
        stream IS the settle and it re-reads the flags each round), pass
        settle_s=0.0, verify=False so recover() only lays down the ladder and
        never stalls the loop.

        Returns True when no target still reports the 0x80 latch (only meaningful
        when verify=True; always True when verify=False)."""
        if targets is None:
            targets = list(self.ids)
        elif isinstance(targets, int):
            targets = [targets]
        else:
            targets = list(targets)
        if not targets:
            return True
        # 1) minimal ladder (t7 minimal_ladder): clear -> keep-alive all -> run
        for mid in targets:
            self._rr.send(mid, CMD_CLEAR)          # 0x9B
        for mid in self.ids:
            self._rr.send(mid, CMD_TORQUE)         # 0xA1 iq=0: keep everyone fed
        for mid in targets:
            self._rr.send(mid, CMD_RUN)            # 0x88
        self._rr.poll()
        # 2) paced keep-alive settle stream (t7 stream_keepalive after the ladder)
        if settle_s > 0.0:
            self.hold(settle_s, rate_hz=rate_hz)
        # 3) verify the 0x80 latch actually cleared (t7 batch_status1)
        if not verify:
            return True
        errs = self.status1()
        return not any(errs[mid][0] is None or (errs[mid][0] & 0x80)
                       for mid in targets)

    # -- batch status reads ---------------------------------------------
    def status1(self, timeout: float = 0.1) -> dict:
        """Status-1 (0x9A) read of EVERY motor at once: send to all
        back-to-back, then collect. {mid: (error_byte, motor_state)};
        (None, None) for a motor that never replied within `timeout`."""
        self._rr.poll()
        for mid in self.ids:
            self._rr.rec[mid].error = None
            self._rr.send(mid, CMD_STATUS1)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self._rr.poll()
            if all(self._rr.rec[m].pending_t is None for m in self.ids):
                break
            time.sleep(0.0002)
        return {m: (self._rr.rec[m].error, self._rr.rec[m].state)
                for m in self.ids}

    def status2(self, timeout: float = 0.1) -> dict:
        """Status-2 (0x9C) read of every motor at once; {mid: speed_raw}."""
        self._rr.poll()
        for mid in self.ids:
            self._rr.rec[mid].speed = None
            self._rr.send(mid, CMD_STATUS2)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self._rr.poll()
            if all(self._rr.rec[m].pending_t is None for m in self.ids):
                break
            time.sleep(0.0002)
        return {m: self._rr.rec[m].speed for m in self.ids}

    def status1_req(self, mid: int) -> bool:
        """Fire a single-slot 0x9A (in-loop substitution for a control frame;
        the reply is dispatched by the next poll())."""
        return self._rr.send(mid, CMD_STATUS1)

    # -- set-zero (hardware 0x19 encoder-offset write) -------------------
    def set_zero_all(self, timeout: float = 1.0, rate_hz: float = 250.0,
                     max_rounds: int = 4, targets=None) -> dict:
        """Set-zero (0x19) the motors in `targets` (default every id) at the current
        position -- each motor's current encoder reading becomes its zero. Returns
        {mid: encoder_offset or None} for EVERY id (None = not written or never
        acked; ids not in `targets` are always None).

        PACED, not bursted: an unpaced N-frame burst overflows the short SocketCAN
        TX queue (qlen ~10) and the last id(s) fail to transmit with ENOBUFS -- so
        0x19 goes out one per bus slot (1/(rate_hz*n)), letting the wire drain
        between frames. A motor that does not ack (dropped send or lost reply) is
        re-sent next round, up to max_rounds or `timeout`.

        WARNING: 0x19 writes the driver's encoder offset; the vendor warns it
        affects the driver lifetime. In the common case each motor is written
        exactly once (a re-send happens only for a motor whose ack was lost). The
        new zero takes effect only after a POWER-CYCLE. Sends no 0x88, so a limp
        bring-up stays limp."""
        targets = list(self.ids if targets is None else targets)
        for mid in targets:
            self._rr.rec[mid].encoder_offset = None
        n = max(1, len(targets))
        slot = 1.0 / (rate_hz * n)
        end = time.perf_counter() + timeout
        for _ in range(max_rounds):
            todo = [m for m in targets if self._rr.rec[m].encoder_offset is None]
            if not todo:
                break
            next_t = time.perf_counter() + slot
            for mid in todo:
                self._rr.poll()                 # drain acks as we go
                self._rr.send(mid, CMD_SET_ZERO)
                pace(next_t)                    # one per slot -> queue never fills
                next_t += slot
            grace = time.perf_counter() + 0.05
            while time.perf_counter() < grace:
                self._rr.poll()
                if all(self._rr.rec[m].encoder_offset is not None for m in targets):
                    break
                time.sleep(0.0005)
            if time.perf_counter() > end:
                break
        self._rr.poll()
        return {m: self._rr.rec[m].encoder_offset for m in self.ids}

    def clear(self, mid: int) -> bool:
        """0x9B clear error flags (e.g. an input-lost latch) WITHOUT enabling the
        motor -- unlike recover(), sends no 0x88, so the motor stays limp."""
        return self._rr.send(mid, CMD_CLEAR)

    # -- telemetry accessors (last-known from poll(), unit-converted) -----
    def temps(self) -> dict:
        return {m: self._rr.rec[m].temp for m in self.ids}

    def errors(self) -> dict:
        return {m: (self._rr.rec[m].error or 0) for m in self.ids}

    def voltages(self) -> dict:
        return {m: self._rr.rec[m].voltage for m in self.ids}

    def speeds_dps(self) -> dict:
        return {m: self.dirs[m] * (self._rr.rec[m].speed or 0)
                / self.vel_state_gain for m in self.ids}

    def torques_nm(self) -> dict:
        return {m: self.dirs[m] * (self._rr.rec[m].iq or 0) / self.torque_gain
                for m in self.ids}

    def encoders_deg(self) -> dict:
        return {m: (self._rr.rec[m].encoder or 0) * self.encoder_gain
                for m in self.ids}

    def offsets(self) -> dict:
        """Last-known encoder offset per motor (from the most recent set_zero_all
        reply); None for any motor that has not answered a 0x19."""
        return {m: self._rr.rec[m].encoder_offset for m in self.ids}

    # -- e-stop ----------------------------------------------------------
    def stop_all(self) -> None:
        stop_all(self._bus, self.ids)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test: pure helpers + gain wiring, no hardware needed
    assert decode_errors(0) == "none"
    assert decode_errors(0x80) == "input signal lost timeout"
    assert decode_errors(0x81) == "low voltage, input signal lost timeout"

    assert i16(-1) == [0xFF, 0xFF] and i32(1) == [1, 0, 0, 0]
    sd = speed_cmd_data(100, 5000)
    assert len(sd) == 7 and sd[0] == 0x00

    # torque byte-layout math (iq at data bytes 4-5, little-endian, signed)
    iq = int(round(1.0 * TORQUE_GAIN))          # 1 N*m -> iq LSB
    lo, hi = iq & 0xFF, (iq >> 8) & 0xFF
    assert struct.unpack("<h", bytes([lo, hi]))[0] == iq

    # minimal recovery ladder no longer contains the 0x80 shutdown step
    src = arm_motors.__doc__
    assert "0x9B -> 0x88" in src and CMD_SHUTDOWN == 0x80

    # 0x19 set-zero reply decode: encoder offset (unsigned) from bytes 6-7
    assert CMD_SET_ZERO == 0x19
    _r = MotorRecord()
    RoundRobinBus._telemetry(_r, bytes([CMD_SET_ZERO, 0, 0, 0, 0, 0, 0xFF, 0xFF]))
    assert _r.encoder_offset == 65535

    # flush_rx drains a pre-loaded queue and returns the count
    class _FakeBus:
        def __init__(self, frames):
            self._frames = list(frames)
        def recv(self, timeout=0.0):
            return self._frames.pop(0) if self._frames else None
    assert flush_rx(_FakeBus(["stale"] * 3), settle_s=0.0) == 3
    assert flush_rx(_FakeBus([]), settle_s=0.0) == 0

    # gains resolved from config.py (single source, no new copy)
    assert TORQUE_GAIN == config.torque_gain
    assert VEL_STATE_GAIN == config.vel_state_gain
    print("motorbus self-test OK  (torque_gain=%.2f, ladder=0x9B->0x88, "
          "no prime/power-cycle)" % TORQUE_GAIN)
