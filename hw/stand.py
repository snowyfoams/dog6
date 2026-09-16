"""DOG6 stand ON THE ROBOT: `sim.stand`'s sequence, through the real drivers.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.stand --fake --auto 1 --no-imu   the whole path, no robot
    $V -m hw.stand --tau-cap 1.0 --log run.npz
    $V -m hw.stand --law per-leg --tau-cap 1.0 --log baseline.npz

Six phases.  ENTER steps them; X is an E-STOP at any point:

    limp     0xA1 iq=0 keep-alives.  NO TORQUE.  Joint angles, the FK height
             and the trunk attitude are printed, so a wrong sign or a wrong
             zero is READ here, before anything moves
    settle   driver position mode (0xA4), holding the pose latched at ENTER
    crouch   0xA4, smoothstep to `sim.stand.Q_CROUCH` -- trunk on the floor
    lift     0xA1 TORQUE, one of two laws (below), through
             `hw.safety.SafetyGate`
    park     0xA4, back to Q_CROUCH from wherever the lift left it
    done     0xA4, holding Q_CROUCH.  ENTER exits and stops the motors

`sim.stand` explains why the sequence is shaped this way.  This file only
explains what is different about running it on twelve MG5010 drivers.


TWO LIFT LAWS, AND KEEPING BOTH IS THE POINT
    --law srb        `hw.balance`, the default.  The trunk's attitude is
                     MEASURED, the error is an SO(3) log map, gravity enters
                     once as m*g, and an allocator splits the resulting wrench
                     across four feet by the geometry the robot actually has.
                     Roll and pitch have gains that are roll and pitch gains.
    --law per-leg    `sim.stand.compliance_torque`, UNCHANGED.  Four
                     independent Cartesian springs in their own hip frames, a
                     fixed mg/4 feedforward, no sensor that reads trunk
                     attitude and no variable that names it.

    The second one is kept because without it the first question after a bad
    run -- "is this worse than what we had?" -- has no answer.  It is also the
    A/B that says whether a fault is above or below the model: if the per-leg
    law misbehaves too, no controller swap will fix it and the place to look
    is the torque gain, the current loop or the CAN timing.

    `--ablate-attitude` is the third point on that line: the SRB law with both
    attitude gains zeroed, which is the height loop and gravity split properly
    and nothing else.  Between "the old law" and "the new law" it is the step
    that isolates what the attitude loop itself did.


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
    Both laws use `hw.kinematics` (0.09 ms for four legs) rather than
    `sim.kinematics`; `sim.selftest` gates the two at 1e-12.

    HEIGHTS.  Everything this file PRINTS is floor to trunk BOTTOM -- the
    number a ruler reads.  `sim.stand`'s constants are trunk-ORIGIN heights,
    35.01 mm higher, and `hw.balance.state` is the only conversion.  DOG5
    printed 191 mm where a ruler read 160 for want of that distinction.


THE TIMING, MEASURED RATHER THAN ASSUMED
    The SRB law runs entirely at slot 0, once per 4 ms sweep, and it delays
    the same motor every sweep -- so it has to fit in a 333 us slot.  It does:
    210 us in isolation and p50 250-310 / p95 375 us inside the fake-bus loop
    on a laptop, with all four leg-gravity terms refreshed every sweep.

    That last part is only affordable because `hw.balance.torque` has a
    closed form for it: the chain walk `sim.kinematics` uses costs 465 us for
    four legs and would be two thirds of the whole law.  `--gravity-legs 1`
    sub-rates it the way the per-leg law does, at the cost of the 12 ms skew
    across the legs.  The exit report prints the law's real p50/p95/max every
    run; if the Pi disagrees with the numbers above, believe the Pi.


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
from . import imu as IMU             # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import safety as SAFE         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import controller as BCTRL   # noqa: E402
from .balance import law as BLAW     # noqa: E402
from .balance import state as BSTATE  # noqa: E402
from .motor import MAX_SPEED_POS     # noqa: E402

__all__ = ["PHASES", "GAP_ESTOP_S", "LAWS", "HardwareStand", "StandLog",
           "run", "main"]

#: The two lift laws, and the whole point of there being two.
#:
#:   srb       the SRB balance controller of `hw.balance`: attitude is
#:             MEASURED, gravity enters once as m*g, and the allocator splits
#:             it across four feet by the geometry the robot actually has.
#:   per-leg   `sim.stand.compliance_torque`: four independent Cartesian
#:             springs, a fixed mg/4 feedforward, and no variable anywhere in
#:             it that names the trunk's orientation.
#:
#: THE SECOND ONE IS KEPT DELIBERATELY.  Without it the first question after a
#: bad run -- "is this worse than what we had?" -- has no answer.  It is also
#: the A/B that says whether a fault is above or below the model: if the
#: PER-LEG law misbehaves too, nothing a controller swap can do will fix it.
LAWS = ("srb", "per-leg")

PHASES = ("limp", "settle", "crouch", "lift", "park", "done")
BLURB = {
    "limp":   "NO TORQUE -- check the angles, move a foot by hand",
    "settle": "driver position mode, holding the pose it was in",
    "crouch": "driver position mode -> crouch (trunk on the floor)",
    "lift":   "TORQUE mode -- LIFT_BLURB says which law",
    "park":   "driver position mode -> crouch",
    "done":   "driver position mode, holding crouch.  ENTER exits",
}

#: What the lift phase is actually doing, which depends on `--law`.  The
#: phase banner is the one place an operator reads it, so it says which.
LIFT_BLURB = {
    "srb": "TORQUE, SRB balance controller -- attitude MEASURED, one wrench "
           "allocated across four feet",
    "per-leg": "TORQUE, per-leg Cartesian compliance (THE BASELINE) -- xy "
               "pinned, z lifts, mg/4 each",
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

    def __init__(self, gate: SAFE.SafetyGate, *, law: str = "srb",
                 balance: BLAW.BalanceLaw | None = None):
        if law not in LAWS:
            raise ValueError("law must be one of %s, got %r" % (LAWS, law))
        self.gate = gate
        self.law = law
        self.balance = balance if balance is not None else BLAW.BalanceLaw()
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
        #: The last sweep`s measurement and law output, for the status line
        #: and the log.  None until the lift arms.
        self.body = None
        self.out = None
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
            if self.law == "srb":
                # h0 and the heading are latched HERE, from what is measured
                # at the handover -- see `balance.law.BalanceLaw.arm`.
                self.balance.arm(now, self.body)
        elif name == "done":
            self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        return None

    def update(self, now: float, body):
        """One sweep.  Returns ``(mode, values, trip)``.

        `body` is a `balance.state.BodyState` -- the sweep's measurement,
        built by the caller so that `advance` and this method see the same
        one.  `mode` is "keepalive" (values None), "position" (values: (12,)
        joint rad) or "torque" (values: (12,) N*m).  `trip` is a reason to
        stop, or None.
        """
        self._sweep += 1
        self.body = body
        q, qd = body.q, body.qd
        name = self.phase_name
        elapsed = now - self.t_phase
        q4 = C.unflat(q)

        if name == "limp":
            self.h_cmd = float("nan")
            return "keepalive", None, None

        if name == "lift":
            if self.law == "srb":
                return self._lift_srb(now, body)
            return self._lift_per_leg(now, elapsed, q, q4, qd)

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

    # -- the two lift laws -------------------------------------------------
    def _lift_srb(self, now: float, body):
        """The SRB balance controller.  Five stages, once, at slot 0.

        The law returns a trip reason; the GATE still shapes the torque
        afterwards, so there is exactly one limiter per quantity.  A trip
        during the lift is a drop, which is why the law reports and this
        method -- and ultimately `run` -- decides.
        """
        self.out = self.balance.update(now, body, clock=time.perf_counter)
        self.h_cmd = BSTATE.height_to_origin(self.out.command.h)
        self.q_des = self.out.q_ref
        self.gravity = self.balance.gravity
        self.tau_request = self.out.tau
        self.tau = self.gate.apply(self.tau_request, body.q, now)
        self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))
        return "torque", self.tau, self.out.trip

    def _lift_per_leg(self, now: float, elapsed: float, q, q4, qd):
        """`sim.stand.compliance_torque`: THE A/B BASELINE, unchanged.

        Four Cartesian springs in their own hip frames, a fixed mg/4
        feedforward, a raised-cosine height ramp, and the leg gravity term
        refreshed one leg per sweep -- so at any instant the four legs carry
        terms computed 0, 4, 8 and 12 ms ago.  Every one of those is a thing
        the SRB law changes, and leaving them exactly as they were is what
        makes the comparison mean something.

        It reads no IMU, so it trips on nothing an IMU could see.
        """
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


class StandLog:
    """Per-sweep recorder.  The 2026-09-15 runs could not be analysed because
    nothing was kept -- a 2 Hz status line is not a log.

    Columns are chosen so that every term of the law is RECOVERABLE: the
    commanded height beside the measured one, the wrench asked for beside the
    forces allocated, the torque commanded beside the torque the drivers
    report producing.  A log of a bad run that holds the result and neither
    the cause nor the command is what the last set of runs had.
    """

    #: field -> per-sweep shape.  DECLARED rather than inferred, so the
    #: SRB-only columns come out the same shape under `--law per-leg` (full of
    #: NaN) as under `--law srb`.  An A/B is two files read by one script, and
    #: a script that has to branch on which law wrote the file is a script
    #: that will eventually compare the wrong columns.
    FIELDS = {
        "t": (), "h_cmd": (), "h": (), "p_cz": (), "p_cz_dot": (),
        "roll": (), "pitch": (), "yaw": (), "imu_age": (),
        "omega_b": (3,), "b_d": (6,), "e_R": (3,), "fz": (C.N_LEGS,),
        "f_w": (C.N_LEGS, 3), "residual": (6,),
        "q": (C.N_JOINTS,), "qd": (C.N_JOINTS,), "q_ref": (C.N_JOINTS,),
        "tau_cmd": (C.N_JOINTS,), "tau_meas": (C.N_JOINTS,),
        "tau_req": (C.N_JOINTS,),
    }

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def save(self, path: str) -> str:
        if not self.rows:
            return "nothing to save -- the run never reached a torque phase"
        columns = {"phase": np.array([row.get("phase", "") for row in self.rows],
                                     dtype="U8")}
        for field, shape in self.FIELDS.items():
            blank = np.full(shape, np.nan)
            columns[field] = np.array(
                [blank if row.get(field) is None else row[field]
                 for row in self.rows], dtype=float)
        np.savez_compressed(path, **columns)
        return "%d sweeps -> %s" % (len(self.rows), path)


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
    blurb = (LIFT_BLURB[stand.law] if stand.phase_name == "lift"
             else BLURB[stand.phase_name])
    print("\n>> phase %d %-6s : %s" % (stand.phase, stand.phase_name, blurb),
          flush=True)


def run(mb, stand: HardwareStand, *, rate_hz: float = RATE_HZ, key=None,
        auto_s: float | None = None, clock=time.perf_counter,
        imu=None, log: "StandLog | None" = None) -> str | None:
    """Drive `stand` on an ARMED `mb` until done, X, or a trip.

    Returns the stop reason, or None for a clean exit from "done".  Does not
    stop the motors -- the caller's `with MotorBus(...)` does, on every path.

    `imu` is an `hw.imu.ImuDog` or None.  NONE IS AN ABLATION, NOT A DEFAULT:
    with no IMU the trunk is assumed perfectly level and infinitely fresh, the
    attitude error is identically zero, and the SRB law degenerates to a
    height loop plus gravity split by the geometry.  That is a legitimate
    first run -- it is strictly more than the per-leg law had -- but it is not
    the controller this exists to be.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    readback = SAFE.TorqueReadback(n)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)
    level = IMU.TrunkOrientation.level()

    sweep = 0
    mode, values = "keepalive", None
    q_prev = t_prev = None
    qd_ctrl = np.zeros(n)
    q = np.zeros(n)
    worst_gap = np.zeros(n)
    overruns = 0
    imu_warned = False
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

            # -- stages 1 and 2, every phase ------------------------------
            # DO NOT WAIT FOR A PACKET.  The IMU streams on its own clock;
            # this takes the most recent one whatever its age and the law
            # decides what to do about the age.  A sweep that blocks on the
            # IMU is a sweep that misses its CAN deadline.
            orientation = level if imu is None else (imu.orientation() or level)
            body = BSTATE.read(q, qd_ctrl, orientation)
            stand.body = body
            if body.imu_stale and not imu_warned and imu is not None:
                imu_warned = True
                print("\n   IMU stale (%.0f ms): the attitude half of the law "
                      "is HELD at zero while it lasts"
                      % (1e3 * body.imu_age_s), flush=True)

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
            mode, values, trip = stand.update(now, body)
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

            # THE DRIVERS REPORT q-AXIS CURRENT IN EVERY REPLY, so commanded
            # torque can be compared against produced torque for free.  On
            # DOG5 a blackout produced ZERO measured torque with the command
            # pinned at the cap and CAN still answering -- every trip DOG6 has
            # would have stayed silent through it.  This is the one witness
            # that sees it.
            tau_meas = np.asarray([mb.torques_nm()[mid] for mid in ids])
            reason = readback.reason(stand.tau, tau_meas,
                                     live=(mode == "torque"))
            if reason:
                return reason

            if log is not None and mode == "torque":
                out = stand.out
                log.add(t=now - stand.t_phase, phase=stand.phase_name,
                        h_cmd=BSTATE.origin_to_height(stand.h_cmd), h=body.h,
                        p_cz=body.p_cz, p_cz_dot=body.p_cz_dot,
                        roll=body.roll, pitch=body.pitch, yaw=body.yaw,
                        omega_b=body.omega_b, imu_age=body.imu_age_s,
                        b_d=None if out is None else out.wrench.b_d,
                        e_R=None if out is None else out.wrench.e_R,
                        fz=None if out is None else out.allocation.fz,
                        f_w=None if out is None else out.allocation.f_w,
                        residual=None if out is None else out.allocation.residual,
                        q=body.q, qd=body.qd, q_ref=stand.q_des,
                        tau_cmd=stand.tau, tau_meas=tau_meas,
                        tau_req=stand.tau_request)

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                print("   %-6s t=%5.1f  h_cmd=%6.1f  %s  |tau|=%.2f/%.2f"
                      "  gap=%.1f ms  overrun=%d%s"
                      % (stand.phase_name, now - stand.t_phase,
                         1e3 * BSTATE.origin_to_height(stand.h_cmd),
                         body.status(),
                         float(np.abs(stand.tau).max()), stand.tau_peak,
                         1e3 * worst_gap.max(), overruns,
                         "" if stand.out is None else
                         "  res %.1fN/%.2fNm" % (
                             stand.out.allocation.residual_force,
                             stand.out.allocation.residual_moment)),
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

    law = ap.add_argument_group(
        "the lift law",
        "`srb` is the balance controller of hw.balance; `per-leg` is the "
        "Cartesian compliance law it replaces, kept as the A/B baseline.")
    law.add_argument("--law", choices=LAWS, default="srb",
                     help="which law drives the lift phase")
    law.add_argument("--rise", type=float, default=BCFG.T_RISE,
                     metavar="SECONDS",
                     help="S-curve duration; the only knob the reference has")
    law.add_argument("--height", type=float, default=1e3 * BCFG.H_LIFT,
                     metavar="MM",
                     help="lift target, mm FLOOR TO TRUNK BOTTOM (the ruler's "
                          "number, not the code's trunk-origin frame)")
    law.add_argument("--kp-z", type=float, default=BCFG.KP_Z)
    law.add_argument("--kd-z", type=float, default=BCFG.KD_Z)
    law.add_argument("--kp-att", type=float, default=BCFG.KP_ATT)
    law.add_argument("--kd-att", type=float, default=BCFG.KD_ATT)
    law.add_argument("--mu", type=float, default=BCFG.MU,
                     help="friction coefficient the allocator PROJECTS onto; "
                          "it bounds what is asked for, not what the floor gives")
    law.add_argument("--ablate-attitude", action="store_true",
                     help="zero both attitude gains -- the other half of the "
                          "A/B.  Height loop and gravity only; the trunk will "
                          "NOT push back")
    law.add_argument("--gravity-legs", type=int, default=4, choices=(1, 2, 4),
                     metavar="N", help="leg-gravity terms refreshed per sweep; "
                                       "4 removes the 12 ms cross-leg skew")

    sensing = ap.add_argument_group("sensing and recording")
    sensing.add_argument("--imu-port", default=IMU.DEFAULT_PORT,
                         help="DETA10 serial port")
    sensing.add_argument("--no-imu", action="store_true",
                         help="assume a perfectly level trunk.  AN ABLATION: "
                              "the attitude error is then identically zero and "
                              "the SRB law degenerates to height + gravity")
    sensing.add_argument("--log", default=None, metavar="FILE.npz",
                         help="record every torque-mode sweep")
    args = ap.parse_args(argv)

    if args.auto is not None and not args.fake:
        ap.error("--auto is only allowed with --fake: on the robot a person "
                 "steps the phases")
    if args.law == "per-leg" and not args.no_imu:
        # Not an error: the baseline reads no IMU by construction, and saying
        # so beats letting an operator think the A/B differs in two ways.
        args.no_imu = True
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
    gains = BCTRL.BalanceGains()
    gains.kp_pos[2], gains.kd_pos[2] = args.kp_z, args.kd_z
    gains.kp_att[0] = gains.kp_att[1] = args.kp_att
    gains.kd_att[0] = gains.kd_att[1] = args.kd_att
    if args.ablate_attitude:
        gains.ablate_attitude()
    balance = BLAW.BalanceLaw(gains=gains, rise_s=args.rise,
                              h_lift=1e-3 * args.height,
                              gravity_legs_per_sweep=args.gravity_legs,
                              mu=args.mu)
    stand = HardwareStand(gate, law=args.law, balance=balance)
    log = StandLog() if args.log else None
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
    print("  lift tau cap %.2f N*m (standing needs ~2.2)" % args.tau_cap)
    print("  HEIGHTS BELOW ARE FLOOR TO TRUNK BOTTOM -- a ruler reaches them.")
    print("  crouch %.0f mm -> lift %.0f mm over %.1f s; the code's own frame "
          "is %.1f mm higher" % (1e3 * BCFG.H_CROUCH, args.height, args.rise,
                                 1e3 * BCFG.TRUNK_BOTTOM_OFFSET))
    if args.law == "srb":
        print("  LAW: SRB balance controller.  %s" % gains)
        print("       mu %.2f, leg gravity %d leg%s/sweep, CoM PINNED at "
              "(%+.1f, %+.1f, %+.1f) mm"
              % (args.mu, args.gravity_legs,
                 "" if args.gravity_legs == 1 else "s",
                 *(1e3 * BCFG.COM_BODY)))
        if args.ablate_attitude:
            print("       ATTITUDE GAINS ZEROED -- the ablation half of the "
                  "A/B; the trunk will NOT push back")
        if args.no_imu:
            print("       NO IMU: the trunk is ASSUMED level.  The attitude "
                  "error is identically zero,")
            print("       so this is the height loop plus gravity split by "
                  "the geometry -- more than the")
            print("       per-leg law had, and less than this law is.")
    else:
        print("  LAW: per-leg Cartesian compliance -- THE A/B BASELINE, "
              "unchanged.")
        print("       Four independent springs, a fixed mg/4, no variable "
              "anywhere in it that names")
        print("       the trunk's orientation.  If THIS misbehaves too, the "
              "fault is below the model.")
    print("  %.0f Hz per motor, %.1f ms sweep; stop line %.0f ms of the "
          "drivers' %.0f ms input-lost window"
          % (args.rate, 1e3 / args.rate, 1e3 * GAP_ESTOP_S,
             1e3 * SAFE.INPUT_LOST_S))
    print("  trips: tilt %.0f deg, tracking %.0f deg, residual %.1f N / "
          "%.2f N*m sustained"
          % (BCFG.TILT_STOP_DEG, np.rad2deg(BCFG.TRACK_STOP_RAD),
             BCFG.RESIDUAL_FORCE_N, BCFG.RESIDUAL_MOMENT_NM))
    print("  ENTER steps the phase.  X is an E-STOP -- during lift it DROPS "
          "the robot.")
    if args.fake and args.law == "srb":
        print("  NOTE: hw.fake_bus has NO DYNAMICS.  The legs do not move "
              "under torque, so the")
        print("  measured height cannot follow the ramp and the TRACKING TRIP "
              "fires partway through")
        print("  the lift.  That is the trip working, not the law failing -- "
              "the per-leg law runs")
        print("  the whole sequence there only because it has no such trip.")

    from .motor import motorbus
    if args.fake:
        from .fake_bus import FakeDriverBus
        bus = FakeDriverBus(ids=ids)
        mb = motorbus.MotorBus(ids, bus=bus, dirs=HM.motor_directions())
    else:
        mb = motorbus.MotorBus(ids, bitrate=args.bitrate,
                               dirs=HM.motor_directions())

    # The IMU is opened BEFORE the bus and started before arming: its stream
    # takes a moment to come up, and the one place that must not wait for a
    # packet is the control loop.
    imu = None
    if not args.no_imu:
        try:
            imu = IMU.ImuDog(port=args.imu_port).start()
            if not imu.wait_for_data(3.0):
                raise RuntimeError("no AHRS packet in 3 s on %s" % args.imu_port)
        except Exception as failure:                     # noqa: BLE001
            if imu is not None:
                imu.stop()
            print("[stand] no IMU: %s\n[stand] pass --no-imu to run the "
                  "level-trunk ablation deliberately." % failure,
                  file=sys.stderr)
            return 2

    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[stand] not every motor armed", file=sys.stderr)
                return 1
            # Straight into the loop: arm() streamed until this instant, and
            # nothing may sit between it and the first slot.
            stop = run(mb, stand, rate_hz=args.rate, key=key,
                       auto_s=args.auto, imu=imu, log=log)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()
        if imu is not None:
            imu.stop()

    print()
    if args.law == "srb" and stand.balance.armed:
        print("[stand] the lift, as the law saw it:")
        print(stand.balance.report())
    if log is not None:
        print("[stand] log: %s" % log.save(args.log))
    if stop is None:
        print("[stand] done; motors stopped in the crouch.  Peak lift torque "
              "%.2f N*m." % stand.tau_peak)
        return 0
    print("[stand] E-STOP in phase %s: %s" % (stand.phase_name, stop))
    print("[stand] motors stopped.")
    return 0 if stop == "operator X" else 1


if __name__ == "__main__":
    raise SystemExit(main())
