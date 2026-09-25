"""DOG6 stand ON THE ROBOT: `sim.stand`'s sequence, through the real drivers.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.stand --fake --auto 1 --no-imu   the whole path, no robot
    $V -m hw.stand --tau-cap 3.0 --log run.npz
    $V -m hw.stand --law per-leg --tau-cap 1.0 --log baseline.npz

    ATTITUDE GAINS, all in the PD's own units (kp 1/s^2, kd 1/s):
    $V -m hw.stand --kp-att 90 --kd-att 17           roll AND pitch together
    $V -m hw.stand --kp-roll 290 --kd-roll 23        roll only; pitch keeps
                                                     --kp-att / --kd-att
    $V -m hw.stand --kp-att 120 --kp-roll 290        pitch 120, roll 290; roll
                                                     kd follows --kd-att (17)
    $V -m hw.stand --kp-yaw 25 --kd-yaw 10          the HEADING spring, on
                                                     since 2026-09-24
    $V -m hw.stand --kp-yaw 0                        the damper-only yaw this
                                                     ran with before that
    --kp-roll and --kd-roll are independent: pass one and the other follows
    --kp-att / --kd-att.  The banner prints what each axis actually got.

    THE YAW SPRING HOLDS A DRIFT, NOT A NORTH.  It acts on the heading error
    off the yaw latched at the crouch -> rise handover, which is what makes a
    magnetometer beside twelve motors a fair thing to close on at all --
    `balance.config.KP_YAW` has the sizing and `hw.trot_esti` prints what a
    trot does to the heading.

Seven phases.  ENTER steps them -- except RISE, which ends on its own clock.
X is an E-STOP at any point:

    limp     0xA1 iq=0 keep-alives.  NO TORQUE.  Joint angles, the FK height
             and the trunk attitude are printed, so a wrong sign or a wrong
             zero is READ here, before anything moves
    settle   driver position mode (0xA4), holding the pose latched at ENTER
    crouch   0xA4, smoothstep to `sim.stand.Q_CROUCH` -- trunk on the floor
    rise     0xA1 TORQUE, one of two laws (below), through
             `hw.safety.SafetyGate`.  Ends WITHOUT a keypress, the sweep the
             S-curve arrives; ENTER is refused until then
    hold     the same law, same gains, at height.  THIS IS THE EXPERIMENT:
             push the trunk and watch whether it comes back.  The status line
             carries the worst tilt from the setpoint, the moment the law
             asked for, and how long recovery took
    park     0xA4, back to Q_CROUCH from wherever the hold left it
    done     0xA4, holding Q_CROUCH.  ENTER exits and stops the motors

`sim.stand` explains why the sequence is shaped this way.  This file only
explains what is different about running it on twelve MG5010 drivers.


WHAT IS IN THIS FILE, AND WHAT IS IN `hw.balance`
    This file is the RUNNER: the CAN loop and its slots, the key poller, the
    per-sweep log, the exit report and the command line.  Every one of those
    is I/O, and I/O is the dividing line.

    THE SEQUENCE ITSELF IS `hw.balance.sequence.StandSequence` -- the six
    phases, what each one sends, the 0xA4 speed caps and the tracking trips
    that decide to stop.  It sits next to the law it hands the lift to,
    because every seam between a position phase and the lift is a statement
    about that law: the lift arms from the pose the CROUCH left, `arm` latches
    h0 and the heading from what is MEASURED there, and the PARK starts from
    wherever the lift settled rather than from Q_CROUCH.

    Nothing in that file opens a bus, reads a clock of its own or prints, so
    the object `--fake` exercises is exactly the object the robot runs, and it
    can be stepped in a test with a hand-built state and no hardware at all.


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

    ANYTHING ELSE THAT COSTS REAL TIME GETS ITS OWN SLOT.  The eleven slots
    that are not slot 0 send one frame and then wait, so there is 300-odd us of
    each going spare.  `ESTIMATOR_SLOT` spends one of them on an `estimator`
    tap -- 290 us of Kalman filter that slot 0 has no room for.


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


AN E-STOP IS A LIMP ROBOT, AND DURING RISE OR HOLD THAT IS A DROP
    X, Ctrl-C, a trip and a crash all end in `MotorBus.close()`: 0x81 stop
    then iq=0 to every motor.  From limp, settle, crouch, park and done that
    is harmless -- the crouch holds at zero torque (sim measured 0.1 mm of
    trunk drop in 3 s released).  From RISE or HOLD it drops the trunk onto
    its belly from up to 157 mm.  Run the first lifts with the robot
    supported -- and note that HOLD is the phase an operator deliberately
    PUSHES, so it is the phase in which a trip is most likely and a drop
    least expected.

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
from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import imu as IMU             # noqa: E402
from . import safety as SAFE         # noqa: E402
from . import velocity_estimator as VEL  # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import controller as BCTRL   # noqa: E402
from .balance import law as BLAW     # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .balance import state as BSTATE  # noqa: E402
from .balance import swing as BSWING  # noqa: E402
from .balance.sequence import (      # noqa: E402
    LAWS, PHASES, StandSequence, phase_blurb)

#: Re-exported from `balance.sequence`, which owns the sequence itself.
#: This module is the RUNNER: the bus, the slots, the keys, the log, the CLI.
__all__ = ["PHASES", "LAWS", "StandSequence", "GAP_ESTOP_S", "StandLog",
           "run", "main"]

#: Per-motor command rate.  DOG5's 250 Hz: 3000 frames/s out and 3000 back is
#: about 72 % of a 1 Mbit/s bus, and 4 ms is far inside the 50 ms window.
RATE_HZ = 250.0

#: Stop while the drivers are still listening: half their input-lost window.
GAP_ESTOP_S = 0.5 * SAFE.INPUT_LOST_S

#: What `--no-limits` leaves ON.  None of it limits where the robot goes:
#: the operator's key, the drivers' health, and the gate's torque shaping.
NO_LIMITS_KEPT = ("X, CAN input-lost / missed replies / sweep gap, "
                  "over-temperature, motor faults, torque readback, non-finite "
                  "state; the torque cap, the 9 N*m clip, the slew, the soft "
                  "start")

#: A 0x9A replaces one control frame every this many sweeps, rotating.
STATUS_EVERY_SWEEPS = 2

#: Which slot an `estimator` tap runs in.  NOT SLOT 0, AND THAT IS MEASURED.
#: Slot 0 belongs to the control law, and on the Pi (2026-09-24) the law alone
#: is p50 470 us in `balance.selftest` and p50 560 us inside this loop -- the
#: slot is 333 us, so it is over budget before anything is added to it.
#: `hw.state_estimator`'s filter is another 290 us, and run at slot 0 it pushed
#: EVERY sweep past the 1 ms re-anchor line (2000 overruns in a run against 59)
#: and widened the worst CAN gap from 6.2 to 6.5 ms.  Every other slot sends one
#: frame and then waits, so one of them has the time to spare.
#:
#: THE TAP STILL SEES THE SWEEP'S OWN MEASUREMENT: `body` and `orientation` are
#: latched at slot 0 and handed over here unchanged, so only the arithmetic
#: happens 2 ms later, not the reading.  That is a luxury a tap can have and a
#: controller cannot -- the law needs the estimate before it acts, which is
#: exactly why `state_estimator.adapters.run_once` documents the order it does.
ESTIMATOR_SLOT = 6

#: Low-pass on the finite-differenced encoder, the velocity the law damps
#: with.  The driver's own speed field is kept as the safety gate's witness
#: only -- see `calibration.joint_state` for why it is not trusted alone.
QD_FILTER_HZ = 40.0

STATUS_PERIOD_S = 0.5


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
        # the trot: NaN outside it.  x_b beside p_swing is the swing error.
        "contact_w": (C.N_LEGS,), "swing_s": (C.N_LEGS,),
        "p_swing": (C.N_LEGS, 3), "x_b": (C.N_LEGS, 3),
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


def _print_phase(stand: StandSequence) -> None:
    print("\n>> phase %d %-6s : %s"
          % (stand.phase, stand.phase_name, phase_blurb(stand)), flush=True)


def _torque_lines(tau_cmd, tau_meas) -> str:
    """All twelve joints, commanded beside measured, in N*m, JOINT frame.

    Printed in RISE and HOLD, the two torque phases.  `tau_cmd` is what went
    out AFTER the safety gate; `tau_meas` is the q-axis current each driver
    reported back, with the direction already applied, so the two rows are
    directly comparable joint by joint.  A column where cmd moves and meas
    does not is a driver that is not following; a column where both move and
    the robot does not is torque being absorbed downstream of the motor.
    """
    rows = []
    for tag, tau in (("cmd ", tau_cmd), ("meas", tau_meas)):
        t4 = C.unflat(np.asarray(tau, dtype=float))
        rows.append("          tau %s %s" % (tag, "  ".join(
            "%s %+5.2f %+5.2f %+5.2f" % (leg, *t4[i])
            for i, leg in enumerate(("FL", "FR", "RL", "RR")))))
    return "\n".join(rows)


class VelocityTap:
    """`hw.velocity_estimator` on the stand's own IMU, for the status lines.

    The biases are taken in LIMP -- the one phase with no torque anywhere,
    so the robot is still (DOG5's `quiet_stages`) -- and the integration
    runs every sweep after that on the sensor's own clock.  Nothing the law
    does reads it.  No zero-velocity update: it drifts as the integrator
    does, and the status line shows the seconds since its zero.
    """

    def __init__(self, imu):
        self.feed = VEL.ImuRawFeed(imu)
        self.est = VEL.AccelVelocityEstimator()
        self.failed: str | None = None

    def update(self, still: bool) -> str | None:
        """Drain the raw stream; returns a line to print, once, or None."""
        if self.failed:
            return None
        samples = self.feed.drain()
        if self.est.initialised:
            self.est.update(samples)
            return None
        if not still:
            return None
        try:
            if self.est.collect(samples):
                return ("velocity estimator initialised in limp: b_f %s "
                        "m/s^2" % np.round(self.est.b_f, 3))
        except ValueError as refusal:
            self.failed = str(refusal)
            return "velocity estimator OFF: %s" % refusal
        return None

    def status(self) -> str:
        if self.failed:
            return "v OFF"
        if not self.est.initialised:
            return "v -- (not initialised: needs limp)"
        v = self.est.v_w
        return ("v (%+.3f, %+.3f, %+.3f) m/s  %.0f s since zero"
                % (v[0], v[1], v[2], self.est.elapsed_s))


def run(mb, stand: StandSequence, *, rate_hz: float = RATE_HZ, key=None,
        auto_s: float | None = None, clock=time.perf_counter,
        imu=None, log: "StandLog | None" = None,
        velocity: "VelocityTap | None" = None, estimator=None,
        hook=None, terse: bool = False) -> str | None:
    """Drive `stand` on an ARMED `mb` until done, X, or a trip.

    Returns the stop reason, or None for a clean exit from "done".  Does not
    stop the motors -- the caller's `with MotorBus(...)` does, on every path.

    `imu` is an `hw.imu.ImuDog` or None.  NONE IS AN ABLATION, NOT A DEFAULT:
    with no IMU the trunk is assumed perfectly level and infinitely fresh, the
    attitude error is identically zero, and the SRB law degenerates to a
    height loop plus gravity split by the geometry.  That is a legitimate
    first run -- it is strictly more than the per-leg law had -- but it is not
    the controller this exists to be.

    `velocity` is a `VelocityTap` or None: print-only, nothing reads it.
    `estimator` is another print-only tap, duck-typed the same way --
    `update(now, stand, body, orientation) -> str | None` every sweep, and
    `status() -> str`, which may hand back several lines, under every status
    line.  `hw.trot_esti.EstimatorTap` is one, over `hw.state_estimator`'s
    filter; it costs as much as the law does, so it is stepped in
    `ESTIMATOR_SLOT` on the sweep's own latched measurement.  NOTHING HERE
    READS WHAT EITHER TAP RETURNS: the law is fed by `stand.update(now, body)`
    alone, and the day a tap does enter the loop it stops being a tap and this
    argument is the wrong door.
    `hook` is `main`'s: keys, an ENTER veto, and a call every sweep.

    `terse` cuts the 2 Hz stream down to the phase's OWN line -- rpy, its
    error and the height -- plus whatever a tap prints.  The weight, moment,
    torque, |tau|, gap and overrun lines are what an operator watching the
    LAW needs; a run whose subject is a tap's numbers wants those numbers
    unbroken instead, and `hw.trot_esti` asks for this.  It changes printing
    and nothing else: every trip still runs, and the exit reports are whole.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    readback = SAFE.TorqueReadback(n)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)
    level = IMU.TrunkOrientation.level()
    est_slot = ESTIMATOR_SLOT % n
    est_sweep = None                  # (now, stand, body, orientation), latched

    sweep = 0
    mode, values = "keepalive", None
    q_prev = t_prev = None
    qd_ctrl = np.zeros(n)
    q = np.zeros(n)
    worst_gap = np.zeros(n)
    overruns = 0
    imu_warned = False
    # The phase NAME as well as the index: the trot is a sub-state of hold
    # and changes the name without changing the index.
    last_phase = (stand.phase, stand.phase_name)
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
            # The SRB model is the POSTURE'S, not the module constant's -- the
            # measurement must use the same c^b the reference does.
            body = BSTATE.read(q, qd_ctrl, orientation, srb=stand.crouch.srb)
            stand.body = body
            if velocity is not None:
                said = velocity.update(stand.phase_name == "limp")
                if said:
                    print("\n   " + said, flush=True)
            if estimator is not None:
                # LATCHED HERE, RUN AT `ESTIMATOR_SLOT`.  `body` is rebound
                # further down, to the state the law acted on, so the tap's
                # copy is taken before anything can move it.  `stand` is the
                # live object, so by the time the tap reads its phase and its
                # yaw offset, THIS sweep's advance and rezero are already in
                # it -- which is what puts the read-out in the right phase and
                # the right world frame on the sweep the handover happens.
                est_sweep = (now, stand, body, orientation)
            if body.imu_stale and not imu_warned and imu is not None:
                imu_warned = True
                print("\n   IMU stale (%.0f ms): the attitude half of the law "
                      "is HELD at zero while it lasts"
                      % (1e3 * body.imu_age_s), flush=True)

            # -- operator ---------------------------------------------------
            pressed = key.get() if key is not None else None
            if pressed in ("x", "X"):
                return "operator X"
            if pressed in ("t", "T"):
                print("\n   " + stand.toggle_trot(now), flush=True)
            if pressed in ("w", "W"):
                print("\n   " + stand.toggle_step(now), flush=True)
            if hook is not None and pressed is not None:
                said = hook.key(pressed, now, stand)
                if said:
                    print("\n   " + said, flush=True)
            wants_next = pressed in ("\r", "\n")
            # --auto: `auto_s` after the phase's ramp has ARRIVED.  Asking
            # ramp_remaining about `auto_s` ago is exactly that test.  With a
            # gait, the hold presses T once and the trot latches its own exit
            # after two settles' worth of cycles.
            auto_trot = (auto_s is not None and stand.gait is not None
                         and (stand.trotting or (stand.phase_name == "hold"
                                                 and stand.trot_runs == 0)))
            # With a step: the first hold presses W, the trot runs from the
            # stepped stance, and the final ENTER steps back before parking.
            auto_step = (auto_s is not None and stand.step_to is not None
                         and stand.phase_name == "hold"
                         and stand.step_runs == 0)
            if auto_step:
                if now - stand.t_phase >= auto_s:
                    print("\n   " + stand.toggle_step(now), flush=True)
            elif auto_trot:
                trot_s = stand.gait.cycle_start(2 * stand.gait.settle_every)
                if (not stand.trotting and now - stand.t_phase >= auto_s) or (
                        stand.trotting and not stand.trot_exit
                        and now - stand.t_phase >= trot_s):
                    print("\n   " + stand.toggle_trot(now), flush=True)
            elif (auto_s is not None and now - stand.t_phase >= auto_s
                    and stand.phase_name != "step"
                    and stand.ramp_remaining(now - auto_s) <= 0.0):
                wants_next = True
            if wants_next and hook is not None:
                veto = hook.blocks_advance(stand)
                if veto:
                    print("\n   ENTER ignored: " + veto, flush=True)
                    wants_next = False
            if wants_next:
                if stand.finished:
                    return None
                refused = stand.advance(now, q)
                if refused:
                    print("\n   ENTER ignored: " + refused, flush=True)

            # -- the law ----------------------------------------------------
            if hook is not None:
                hook.sweep(now, stand)
            mode, values, trip = stand.update(now, body)
            # THE BANNER FOLLOWS THE PHASE, NOT THE KEYSTROKE.  RISE ends on
            # its own clock, so a banner printed only where ENTER is handled
            # would silently skip the one phase change nobody pressed a key
            # for -- and that is the phase the operator is waiting for.
            if (stand.phase, stand.phase_name) != last_phase:
                last_phase = (stand.phase, stand.phase_name)
                _print_phase(stand)
            # The sequence does no I/O, so anything it needs an operator to
            # read comes back as text and is printed here.  Generic on
            # purpose: this is a channel, not a feature.
            notice = stand.take_notice()
            if notice:
                print("\n   " + notice, flush=True)
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
                if out is not None and out.state is not None:
                    body = out.state          # the height the law acted on
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
                        tau_req=stand.tau_request,
                        tau_ff=None if out is None else out.tau_ff,
                        contact_w=None if out is None else out.contact,
                        swing_s=None if out is None else out.swing_s,
                        p_swing=None if out is None else out.p_swing,
                        x_b=body.x_b)

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                tail = "" if terse else (
                        "  |tau|=%.2f/%.2f  gap=%.1f ms  overrun=%d%s"
                        % (float(np.abs(stand.tau).max()), stand.tau_peak,
                           1e3 * worst_gap.max(), overruns,
                           "" if stand.out is None else
                           "  res %.1fN/%.2fNm"
                           % (stand.out.allocation.residual_force,
                              stand.out.allocation.residual_moment)))
                # THE HOLD PRINTS SOMETHING ELSE ENTIRELY, and REPLACES the
                # usual line rather than adding to it.  Every other phase is
                # asking "is it getting there"; the hold is asking "is the
                # attitude loop correcting", and the three numbers that answer
                # that -- rpy, rpy MINUS setpoint, and the moment being asked
                # for -- do not fit beside h_cmd and the gyro.
                watch = {"hold": stand.hold,
                         "trot": stand.trot}.get(stand.phase_name)
                if watch is not None and watch.live:
                    print("   %-6s t=%5.1f  %s"
                          % (stand.phase_name, now - stand.t_phase,
                             watch.status()), flush=True)
                    # IMU AGE IS BACK ON THE HOLD LINE.  A stiffer roll loop
                    # has less latency margin than the freeze threshold, so
                    # the age is now the number that says whether a roll
                    # oscillation is the gain or the stream.
                    if stand.trotting and not terse:
                        print("          cycle %d  weight %s%s%s"
                              % (stand.gait.cycles(now),
                                 np.array2string(stand.gait.contact_weight(now),
                                                 precision=2),
                                 "  SETTLE" if stand.gait.settling(now) else "",
                                 "  exit latched" if stand.trot_exit else ""),
                              flush=True)
                    if not terse:
                        print("          %s  imu %3.0f ms%s%s"
                              % (watch.moment_status(),
                                 1e3 * body.imu_age_s,
                                 " STALE" if body.imu_stale else "", tail),
                              flush=True)
                else:
                    print("   %-6s t=%5.1f  h_cmd=%6.1f  %s%s"
                          % (stand.phase_name, now - stand.t_phase,
                             1e3 * BSTATE.origin_to_height(stand.h_cmd),
                             body.status(), tail), flush=True)
                if velocity is not None and not terse:
                    print("          " + velocity.status(), flush=True)
                if estimator is not None:
                    for line in estimator.status().splitlines():
                        print("          " + line, flush=True)
                if (hook is not None and stand.phase_name == "hold"
                        and not terse):
                    print("          " + hook.status(now, stand), flush=True)
                if (stand.phase_name in ("rise", "hold", "trot", "step")
                        and not terse):
                    print(_torque_lines(stand.tau, tau_meas), flush=True)
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

        # A tap that costs real time runs HERE, in a slot of its own and
        # AFTER that slot's frame is already out -- see `ESTIMATOR_SLOT`.
        if k == est_slot and est_sweep is not None:
            said = estimator.update(*est_sweep)
            est_sweep = None
            if said:
                print("\n   " + said, flush=True)

        k = (k + 1) % n
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun > 3 * slot:
            # RE-ANCHOR, do not catch up: catching up is a burst of frames.
            overruns += 1
            deadline = clock() + slot
    # unreachable


def main(argv=None, crouch: POSE.CrouchPose = POSE.NOMINAL,
         dynamic_setpoint: bool = None, only_law: str = None,
         tilt_stop: float = None, roll_gains: tuple = None,
         track_stop: float = None, gait=None, tau_cap: float = None,
         tau_ceiling: float = None, tau_slew: float = None,
         overspeed_trip: bool = True, step_to=None,
         step_period: float = None, swing: str = "cartesian",
         velocity: bool = False, estimator=None, hook=None,
         terse: bool = False, limits: bool = True) -> int:
    """The runner.  `crouch` is the posture the lift starts from and PARK
    returns to; `dynamic_setpoint` None defers to `config.SETPOINT_DYNAMIC`;
    `only_law` pins the lift law and drops `--law` from the parser.

    `limits` False is `--no-limits` by default: every limit on WHERE the
    robot goes is off -- the soft joint limits (the gate's torque block and
    its e-stop), the position-mode tracking trips, the torque-phase tracking
    trip, the tilt stop, the residual trip and the overspeed trip.  What stays
    on is `NO_LIMITS_KEPT`.

    `gait` (a `balance.gait.TrotGait`) enables T; `tau_cap`, `tau_ceiling`
    and `tau_slew` are the gate's default cap, the ceiling it will accept and
    its slew -- None keeps the stand's.  `overspeed_trip` False keeps the
    joint-speed peaks and stops nothing on them (`safety.SafetyGate`).
    `step_to` ((4, 2) hip-frame foot xy, needs `gait`) enables W: the hold
    steps the feet there, and steps them back before the park, on its own
    gait clock of `step_period` seconds (None: the trot's).
    `swing` is `law.BalanceLaw.swing`: "joint" is the fold's z-only joint PD.
    `hook` is an extra operator layer (`hw.sway.Sway`): its keys, a veto on
    ENTER, a call every sweep before the law, and lines for the banner and
    the status.  It turns the joint-space layer's flags on without a gait.
    `velocity` True prints `hw.velocity_estimator`'s v under every status
    line, every phase (needs the IMU; `--no-imu` has nothing to integrate).
    `estimator` is a CALLABLE taking the live `ImuDog` (or None on the
    `--no-imu` path) and returning a tap (see `run`), or None.  A factory
    rather than an object because the IMU is opened in here, after the caller
    is done: `hw.trot_esti` passes a class.
    `terse` is `run`'s: the 2 Hz stream carries the phase's rpy line and the
    taps, and nothing else.

    They are arguments rather than flags-only so that `hw.fold_stand` is three
    lines instead of a copy of this parser.
    """
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ceiling = SAFE.TAU_STAGED_MAX if tau_ceiling is None else float(tau_ceiling)
    slew = SAFE.DEFAULT_TAU_SLEW_NM_S if tau_slew is None else float(tau_slew)
    ap.add_argument("--tau-cap", type=float,
                    default=SAFE.TAU_START_MAX if tau_cap is None else tau_cap,
                    help="lift-phase torque cap, N*m (SafetyGate allows up to "
                         "%.1f)" % ceiling)
    ap.add_argument("--tau-slew", type=float, default=slew, metavar="NM_PER_S",
                    help="SafetyGate's |dtau/dt| limit on every joint.  The "
                         "stand's 5 and the trot's 60 are DOG5's; it bounds "
                         "how fast a swing torque can rise, so the trot "
                         "banner prints what the swing needs against it")
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
        "Cartesian compliance law it replaces, kept as the A/B baseline."
        if only_law is None else
        "pinned to `%s` by this entry point." % only_law)
    if only_law is None:
        law.add_argument("--law", choices=LAWS, default="srb",
                         help="which law drives the lift phase")
    else:
        ap.set_defaults(law=only_law)
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
    law.add_argument("--kp-yaw", type=float, default=BCFG.KP_YAW,
                     help="HEADING spring, same units as --kp-att.  It holds "
                          "the drift off the heading latched at the handover; "
                          "0 is the damper-only yaw of before 2026-09-24")
    law.add_argument("--kd-yaw", type=float, default=BCFG.KD_YAW,
                     help="heading damper, same units as --kd-att -- the "
                          "gyro's omega_z, inertial, on even at --kp-yaw 0")
    law.add_argument("--kp-roll", type=float,
                     default=None if roll_gains is None else roll_gains[0],
                     help="roll only, same units as --kp-att; default follows "
                          "--kp-att")
    law.add_argument("--kd-roll", type=float,
                     default=None if roll_gains is None else roll_gains[1],
                     help="roll only, same units as --kd-att; default follows "
                          "--kd-att")
    law.add_argument("--mu", type=float, default=BCFG.MU,
                     help="friction coefficient the allocator PROJECTS onto; "
                          "it bounds what is asked for, not what the floor gives")
    law.add_argument("--ablate-attitude", action="store_true",
                     help="zero both attitude gains -- the other half of the "
                          "A/B.  Height loop and gravity only; the trunk will "
                          "NOT push back")
    law.add_argument("--tilt-stop", type=float,
                     default=BCFG.TILT_STOP_DEG if tilt_stop is None
                     else tilt_stop, metavar="DEG",
                     help="e-stop past this many degrees FROM THE SETPOINT.  "
                          "IT IS A TRIP: raising it lets the robot reach an "
                          "attitude it cannot come back from, and an e-stop "
                          "there is a harder fall.  Raise it to WATCH a "
                          "steady-state tilt the default stops you seeing")
    law.add_argument("--track-stop", type=float,
                     default=(np.rad2deg(BCFG.TRACK_STOP_RAD) if track_stop is None
                              else track_stop), metavar="DEG",
                     help="e-stop when any joint is this far from the IK at "
                          "the commanded height; 0 turns it OFF.  A MONITOR "
                          "ONLY -- the SRB torque never reads that IK, so "
                          "this changes no N*m the drivers get.  It is what "
                          "notices a leg that has gone wrong while the "
                          "attitude still reads fine")
    lim = ap.add_argument_group(
        "limits", "every limit on where the robot goes, as one switch")
    lim.add_argument("--limits", dest="limits", action="store_true",
                     default=bool(limits),
                     help="ON: the soft joint limits (the gate's torque block "
                          "and e-stop), the position-mode and torque-phase "
                          "tracking trips, the tilt stop, the residual trip "
                          "and the overspeed trip -- at the flags above")
    lim.add_argument("--no-limits", dest="limits", action="store_false",
                     default=argparse.SUPPRESS,
                     help="all of those OFF, whatever the flags above say.  "
                          "Still on: %s" % NO_LIMITS_KEPT)
    if gait is not None:
        trot = ap.add_argument_group(
            "the trot", "the gait clock T starts; the entry point's defaults")
        trot.add_argument("--period", type=float, default=gait.period,
                          metavar="SECONDS",
                          help="one full gait cycle.  Shorter = faster "
                               "handovers, AND a faster swing: the knee's peak "
                               "speed and the handover torque rate scale as "
                               "1/period")
        trot.add_argument("--duty", type=float, default=gait.duty,
                          help="stance fraction, in (0.5, 1)")
        trot.add_argument("--contact-ramp", type=float, default=gait.ramp,
                          metavar="FRACTION",
                          help="fraction of STANCE a foot's share of the load "
                               "smoothsteps over at touchdown and at liftoff.  "
                               "It has to fit inside the four-foot window, so "
                               "the ceiling is (duty - 0.5) / (2 duty) = 0.19 "
                               "at duty 0.80; past it the clock refuses and "
                               "says so")
        trot.add_argument("--settle", type=float, default=gait.settle_s,
                          metavar="SECONDS",
                          help="the four-foot re-level; 0 turns it off")
        trot.add_argument("--settle-every", type=int,
                          default=gait.settle_every, metavar="CYCLES")
        trot.add_argument("--swing", choices=BSWING.SWING_MODES, default=swing,
                          help="the swing leg's controller: 'cartesian' is "
                               "the foot impedance the nominal trot flies -- "
                               "soft (~1 Hz at the foot), so pair it with "
                               "--swing-ff; 'joint' is the FOLD's z-only arc "
                               "through the IK with a joint PD, abd held -- "
                               "on the nominal trot it failed every run "
                               "(2026-09-25)")
        trot.add_argument("--swing-height", type=float,
                          default=1e3 * BCFG.SWING_HEIGHT, metavar="MM",
                          help="the swing apex above the resting foot.  Speed, "
                               "torque and slew demand all scale with it, so "
                               "it is the first thing to lower for a fast "
                               "gait; the banner prints the demand.  With "
                               "--swing-ff the apex is real: 20 mm flew on "
                               "the nominal trot, 40 mm landed hard enough to "
                               "bounce the robot (2026-09-25)")
        trot.add_argument("--swing-ff", action="store_true",
                          help="add the arc's own inertial torque, M0 J^+ "
                               "(a_ref - Jdot qd_ref), to each swing leg open "
                               "loop (both swing modes).  It is ~90 %% of the "
                               "torque a 15 mm / 140 ms arc asks for, which "
                               "the PD was making out of tracking error; see "
                               "swing.py.  Logged as tau_ff")
        trot.add_argument("--ff-armature", type=float, default=None, metavar="KG_M2",
                          help="the reflected rotor inertia the feedforward's M0 "
                               "is built on, joint side.  Default params.ARMATURE "
                               "%.4f, DOG5's, not measured on DOG6; "
                               "`hw.swing_bench --analyse` fits it from a log"
                               % P.ARMATURE)
        trot.add_argument("--kp-swing", type=float, nargs=3, metavar="N_PER_M",
                          default=list(BCFG.KP_SWING),
                          help="the Cartesian swing PD's x y z stiffness.  With "
                               "--swing-ff, Kp is no longer what lifts the foot "
                               "and stiffening z drives the foot into x (no "
                               "Lambda in this PD): leave it")
        trot.add_argument("--kd-swing", type=float, nargs=3,
                          metavar="NS_PER_M", default=list(BCFG.KD_SWING),
                          help="the Cartesian swing PD's x y z damping.  z at "
                               "15 is zeta 0.28 on the foot's 3.9 kg; 30 is "
                               "0.57 and tracked the arc a little better in the "
                               "leg's own dynamics (2026-09-25)")
        trot.add_argument("--half-gait", action="store_true",
                          help="T steps the gait by HAND, half a cycle a "
                               "press: one diagonal lifts and lands, HOLD by "
                               "itself, and the next T swings the other one")
        if step_to is not None:
            trot.add_argument("--step-period", type=float,
                              default=(gait.period if step_period is None
                                       else step_period), metavar="SECONDS",
                              help="the gait cycle W steps the feet on.  The "
                                   "foot crosses the whole step in one swing, "
                                   "so this sets the joint speeds of the step")
    if gait is not None or hook is not None:
        joint = ap.add_argument_group(
            "the joint-space layer", "DOG5's JointImpedance, from HOLD on")
        joint.add_argument("--joint-hold", dest="joint_hold",
                           action="store_true", default=True,
                           help="MODE: the joint angles on reaching HOLD are "
                                "the joint-space target -- trunk position and "
                                "rpy pinned by the legs")
        joint.add_argument("--no-joint-hold", dest="joint_hold",
                           action="store_false", default=argparse.SUPPRESS,
                           help="MODE: the SRB law alone, no joint target -- "
                                "height and rpy held, the body free to shift "
                                "under a push")
        joint.add_argument("--kp-joint", type=float,
                           default=BCFG.KP_JOINT_HOLD, metavar="NM_PER_RAD",
                           help="the joint-space layer: every leg held at the "
                                "joint angles latched on reaching HOLD, through "
                                "the hold and the trot (a swinging leg gets the "
                                "damper only).  0 turns the spring off")
        joint.add_argument("--kd-joint", type=float,
                           default=BCFG.KD_JOINT_HOLD, metavar="NMS_PER_RAD")
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
    datum = sensing.add_mutually_exclusive_group()
    datum.add_argument("--latch-setpoint", dest="latch", action="store_true",
                       default=None,
                       help="take the attitude setpoint at zero torque, in "
                            "LIMP -- 'level' becomes the attitude the robot "
                            "was resting at")
    datum.add_argument("--fixed-setpoint", dest="latch", action="store_false",
                       help="do NOT latch: fly config.SETPOINT_ROLL/PITCH_DEG. "
                            "A run comparing two postures wants this, or the "
                            "two are measured against two different 'level's")
    if hook is not None:
        hook.add_arguments(ap)
    args = ap.parse_args(argv)
    if hook is not None:
        hook.configure(args)
    if args.latch is None:
        args.latch = (bool(BCFG.SETPOINT_DYNAMIC) if dynamic_setpoint is None
                      else bool(dynamic_setpoint))

    if gait is not None:
        from .balance.gait import TrotGait
        try:
            gait = TrotGait(period=args.period, duty=args.duty,
                            offsets=gait.offsets, ramp=args.contact_ramp,
                            settle_s=args.settle,
                            settle_every=args.settle_every)
            step_gait = (None if step_to is None else TrotGait(
                period=args.step_period, duty=args.duty, offsets=gait.offsets,
                ramp=gait.ramp, settle_s=args.settle,
                settle_every=args.settle_every))
        except ValueError as refusal:
            ap.error("the gait: %s" % refusal)
    else:
        step_gait = None

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
    slew = float(args.tau_slew)
    if gait is not None:
        swing = args.swing
    if not args.limits:
        # --no-limits, `main`'s docstring.  Set before anything is built or
        # printed, so the banner reports what actually flies.
        args.tilt_stop, args.track_stop = float("inf"), 0.0
        overspeed_trip = False
    try:
        gate = SAFE.SafetyGate(args.tau_cap, unconfirmed_reason=reason,
                               ceiling=ceiling, tau_slew=slew,
                               overspeed_trip=overspeed_trip,
                               limits=(None if args.limits else
                                       (np.full(C.N_JOINTS, -np.inf),
                                        np.full(C.N_JOINTS, np.inf))))
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
    if args.kp_roll is not None:
        gains.kp_att[0] = args.kp_roll
    if args.kd_roll is not None:
        gains.kd_att[0] = args.kd_roll
    gains.kp_att[2], gains.kd_att[2] = args.kp_yaw, args.kd_yaw
    if args.ablate_attitude:
        gains.ablate_attitude()
    balance = BLAW.BalanceLaw(gains=gains, rise_s=args.rise,
                              h_lift=1e-3 * args.height,
                              gravity_legs_per_sweep=args.gravity_legs,
                              mu=args.mu, foot_xy=crouch.foot_xy,
                              dynamic_setpoint=args.latch,
                              tilt_stop_deg=args.tilt_stop,
                              track_stop_deg=args.track_stop,
                              residual_trip=args.limits, srb=crouch.srb,
                              swing=swing,
                              swing_height=(1e-3 * args.swing_height
                                            if gait is not None
                                            else BCFG.SWING_HEIGHT),
                              swing_ff=bool(gait is not None and args.swing_ff),
                              ff_inertia=(None if gait is None or args.ff_armature is None
                                          else BSWING.feedforward_inertia(args.ff_armature)),
                              **({} if gait is None else dict(
                                  kp_swing=np.asarray(args.kp_swing, float),
                                  kd_swing=np.asarray(args.kd_swing, float))),
                              **({} if (gait is None and hook is None)
                                 or not args.joint_hold
                                 else dict(kp_joint=args.kp_joint,
                                           kd_joint=args.kd_joint)))
    stand = StandSequence(gate, law=args.law, balance=balance, crouch=crouch,
                          gait=gait, step_to=step_to, step_gait=step_gait,
                          trot_swings=(1 if gait is not None and args.half_gait
                                       else 0),
                          position_trips=args.limits)
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
    print("  lift tau cap %.2f N*m (standing needs ~2.2), slew %.0f N*m/s"
          % (args.tau_cap, slew))
    if gait is not None:
        print("  TROT on T from HOLD: %s%s"
              % (gait, "\n       HALF GAIT: each T is one diagonal's swing, alternating "
                 "press by press -- no clock hands over"
                 if args.half_gait else ""))
        if swing == "joint":
            print("       swing apex %.0f mm straight up, NO placement; JOINT "
                  "PD, abd held: Kp %s N*m/rad Kd %s N*m*s/rad"
                  % (args.swing_height, BCFG.KP_SWING_JOINT,
                     BCFG.KD_SWING_JOINT))
        else:
            print("       swing apex %.0f mm straight up, NO placement; Kp %s "
                  "N/m Kd %s N s/m" % (args.swing_height,
                                       np.asarray(args.kp_swing),
                                       np.asarray(args.kd_swing)))
        # WHAT THE SWING ASKS, AGAINST WHAT THE GATE ALLOWS.  The one line
        # that says in advance whether the foot can FOLLOW the arc.  It lifts
        # either way -- it did at 80 ms of swing on the robot, 2026-09-25 --
        # but past this a swing is the PD behind a limiter, late or wound up,
        # and the log's `x_b` against `p_swing` says which.
        demand = BSWING.swing_demand(
            0, BSWING.rest_feet_b(1e-3 * args.height, crouch.foot_xy)[0],
            gait.swing_duration, 1e-3 * args.swing_height,
            C.unflat(crouch.q)[0])
        print("       swing %.0f ms: knee ~%.1f rad/s, ~%.1f N*m (M at the lift "
              "pose); slew: ~%.0f N*m/s to rise, ~%.0f to LAND SOFTLY, "
              "against %.0f%s"
              % (1e3 * gait.swing_duration, demand["qd"][2], demand["tau"].max(),
                 demand["slew"], 3.0 * demand["slew"], slew,
                 "" if demand["slew"] <= slew else
                 "\n       WARNING: THE SWING OUTRUNS THE SLEW.  The PD winds "
                 "up behind the limiter (fold_trot's MuJoCo run: 160/240/260 "
                 "ms all diverged, knee ~90 deg over).  Lower --swing-height, "
                 "lower --duty or lengthen --period until this fits, or raise "
                 "--tau-slew knowing what it is."))
        if demand["tau"].max() > args.tau_cap:
            print("       WARNING: the swing's ~%.1f N*m is over the %.1f N*m "
                  "cap -- the arc cannot be followed at any gain"
                  % (demand["tau"].max(), args.tau_cap))
        print("       swing feedforward %s"
              % ("ON: tau += M0 J^+ (a_ref - Jdot qd_ref) per swing leg, open "
                 "loop, M0 diag %s kg m^2 (armature %s); logged as tau_ff"
                 % (np.array2string(np.diag(BSWING.feedforward_inertia(args.ff_armature)),
                                    precision=4),
                    "%.5f measured" % args.ff_armature if args.ff_armature is not None
                    else "%.5f INHERITED from DOG5" % P.ARMATURE)
                 if args.swing_ff else
                 "OFF (--swing-ff turns it on): the PD alone makes the "
                 "swing torque, out of tracking error"))
        if args.swing_ff and args.swing_height > 20.0:
            print("       NOTE: with the feedforward the apex is REAL.  On the "
                  "nominal trot 20 mm is what flew (2026-09-25); at 40 mm the "
                  "foot came down hard enough to bounce the robot.")
        if args.joint_hold:
            print("       MODE joint-hold: every leg held at the angles "
                  "latched on reaching HOLD, hold and trot, Kp %.1f N*m/rad "
                  "Kd %.2f N*m*s/rad; swing legs damper only"
                  % (args.kp_joint, args.kd_joint))
        else:
            print("       MODE no-joint-hold: SRB alone -- height and rpy "
                  "held, the body free to shift")
        print("       residual trip counts four-foot sweeps only while "
              "trotting")
    if not args.limits:
        print("  LIMITS OFF (--no-limits): soft joint limits (torque block and "
              "e-stop), position-mode and torque-phase tracking, tilt stop, "
              "residual trip, overspeed trip.  --limits puts them back.")
        print("       still on: %s" % NO_LIMITS_KEPT)
    if not overspeed_trip:
        print("  OVERSPEED TRIP OFF: joint speed is recorded, never stopped "
              "on (the peaks print at exit)")
    print("  HEIGHTS BELOW ARE FLOOR TO TRUNK BOTTOM -- a ruler reaches them.")
    print("  crouch %.0f mm -> lift %.0f mm over %.1f s; the code's own frame "
          "is %.1f mm higher" % (1e3 * crouch.h, args.height, args.rise,
                                 1e3 * BCFG.TRUNK_BOTTOM_OFFSET))
    print("  CROUCH POSTURE: %s" % crouch.name.upper())
    for line in crouch.describe().splitlines()[1:]:
        print("   %s" % line)
    print("  attitude setpoint: %s"
          % ("LATCHED at zero torque, in limp" if args.latch else
             "FIXED at the config statics %+.2f / %+.2f deg"
             % (BCFG.SETPOINT_ROLL_DEG, BCFG.SETPOINT_PITCH_DEG)))
    if args.law == "srb":
        print("  LAW: SRB balance controller.  %s" % gains)
        print("       mu %.2f, leg gravity %d leg%s/sweep"
              % (args.mu, args.gravity_legs,
                 "" if args.gravity_legs == 1 else "s"))
        print("       SRB PINNED at the %s stance: %s"
              % (crouch.name, crouch.srb.describe()))
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
    if not args.limits:
        print("  trips: tilt OFF, tracking OFF, residual OFF, position-mode "
              "tracking OFF, joint limits OFF -- nothing stops on where the "
              "robot goes")
    else:
        print("  trips: tilt %.0f deg%s, tracking %s, residual %.1f N / "
              "%.2f N*m sustained"
              % (args.tilt_stop,
                 "" if args.tilt_stop == BCFG.TILT_STOP_DEG else
                 "  <-- RAISED from %.0f, the robot can reach an attitude it "
                 "cannot recover from" % BCFG.TILT_STOP_DEG,
                 ("%.0f deg" % args.track_stop) if args.track_stop > 0
                 else "OFF",
                 BCFG.RESIDUAL_FORCE_N, BCFG.RESIDUAL_MOMENT_NM))
    print("  ENTER steps the phase (RISE ends on its own).  X is an E-STOP "
          "-- from rise or hold it DROPS the robot.")
    if hook is not None:
        for line in hook.banner(args):
            print("  " + line)
    if step_gait is not None:
        print("  W in HOLD steps the feet to hip-frame xy %s mm (W again, or "
              "ENTER, steps them back to the crouch's first), on %s"
              % (np.array2string(1e3 * np.asarray(step_to)[0], precision=0),
                 step_gait))
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

    tap = None
    if velocity and imu is not None:
        tap = VelocityTap(imu)
        if not tap.feed.wait_for_raw(3.0):
            tap = None
            print("[stand] no raw 0x40 IMU packet in 3 s: no velocity "
                  "this run", file=sys.stderr)
    elif velocity:
        print("  velocity: none -- no IMU to integrate")

    est_tap = None
    if estimator is not None:
        # BUILT EVEN WITH NO IMU.  The tap reads `orientation`, and on the
        # `--no-imu` path that is `TrunkOrientation.level()`, which carries a
        # level trunk's specific force.  What comes out is the level-trunk
        # ablation, not a measurement -- but the whole path runs, which is
        # what `--fake --no-imu` is for.
        est_tap = estimator(imu)
        if imu is None:
            print("  estimator: on TrunkOrientation.level() -- the ablation, "
                  "not a sensor")

    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[stand] not every motor armed", file=sys.stderr)
                return 1
            # Straight into the loop: arm() streamed until this instant, and
            # nothing may sit between it and the first slot.
            stop = run(mb, stand, rate_hz=args.rate, key=key,
                       auto_s=args.auto, imu=imu, log=log, velocity=tap,
                       estimator=est_tap, hook=hook, terse=terse)
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
    if stand.hold.live:
        print("[stand] the hold -- whether the controller CORRECTS:")
        print(stand.hold.report())
    if stand.trot.live:
        print("[stand] the trot (%d run%s, the last one below):"
              % (stand.trot_runs, "" if stand.trot_runs == 1 else "s"))
        print(stand.trot.report().replace("  hold  ", "  trot  "))
    if gate.started_at is not None:
        print("[stand] peak joint speed, driver / encoder (rad/s; trip %s at "
              "%.1f):" % ("ON" if gate.overspeed_trip else "OFF",
                          gate.qd_estop))
        for leg in range(C.N_LEGS):
            row = slice(3 * leg, 3 * leg + 3)
            print("  %s  %s" % (C.LEGS[leg], "   ".join(
                "%4.1f / %4.1f" % pair for pair in zip(
                    gate.qd_peak[row], gate.encoder_qd_peak[row]))))
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
