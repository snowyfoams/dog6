"""`hw.trot`, with `hw.state_estimator` READ OUT beside it.  Print only.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.trot_esti --fake --auto 1 --no-imu      the whole path, no robot
    $V -m hw.trot_esti                               on the robot
    $V -m hw.trot_esti --log trot_esti.npz           (the log is `hw.trot`'s;
                                                      the filter is not in it)

THE TROT IS `hw.trot`'S, UNCHANGED, AND IT IS IMPORTED RATHER THAN COPIED.
Every constant, gain, trip, crouch and key comes from that module -- see its
docstring for all of it.  The only thing this file adds is `EstimatorTap`:
MIT's linear KF (`hw.state_estimator`) fed from the same sweep's sensors the
balance law is fed from, and printed.

NOTHING READS WHAT IT PRINTS, AND THAT IS THE EXPERIMENT
    The estimator is OUTSIDE the control loop.  `stand.run` LATCHES the sweep's
    `BodyState` and `TrunkOrientation` where the law is given them, hands the
    tap that pair in a CAN slot of its own (below), and throws the result away
    -- the law is fed by `stand.update(now, body)` alone, exactly as under
    `hw.trot`.  So this run produces the same torques as that one and can be
    compared against it sweep for sweep.

    The first question about a new estimator is not "does it stabilise the
    robot", it is "does it agree with what is already measured".  Height is
    where it can be checked: `hw.balance.state` reports the trunk height from
    the legs and the IMU, the filter reports it from the accelerometer and the
    legs, and the two are printed side by side in every phase.  A frame error
    shows up as a standing gap, a sign error as a gap that grows with tilt,
    and a bad datum as a constant 15 mm.

    A tap must not be able to end a run.  Any exception inside it disables it,
    prints why, and the trot goes on.

WHAT IT PRINTS, AND WHAT IT SILENCES
    THE STREAM IS THE FILTER'S AND THE TRUNK'S ATTITUDE, NOTHING ELSE.  This
    run is read to decide whether an estimate is true, so it asks `stand.run`
    for `terse`: the 2 Hz stream keeps the phase's own rpy line and these two,
    and drops the weight, moment, torque, |tau|, gap and overrun lines an
    operator watching the LAW wants.  The accelerometer-only `VelocityTap` is
    off with them -- the filter's v is the same quantity, measured better.
    Every trip, every phase banner and every exit report is untouched.

        trot   t= 12.3  rpy  +0.41/ -0.88/  -7.20   err ...   h  145.1 mm
        est h  145.3 mm (fk  145.1, d  +0.2)  v (+0.012, -0.003, +0.001) m/s
            xy (+0.021, -0.004) m  yaw   -7.2 deg  trust 1.00/...  innov ...

    h      the filter's trunk height, FLOOR TO TRUNK BOTTOM -- `hw.stand`'s
           convention, so it is the same number as the `h` on the line above.
           `fk` is that line's, from the legs; `d` is filter minus legs.
    v      trunk velocity in the run's world frame, m/s.  `vz fk` beside it is
           `BodyState.zdot_origin`, the height rate the law damps with, which
           is the only component of v anything else in `hw` measures.
    xy     the filter's world x and y: LEG ODOMETRY.  Unobservable, drifts,
           and printed so the drift is visible rather than surprising.
    yaw    the trunk's heading DRIFT off the run's world x axis, degrees --
           the same number as the third field of the `rpy` line above.  It is
           here because xy cannot be read without it; see below.
    trust  tau per leg, FL FR RL RR: 1 planted and believed, 0 in the air.
    innov  y - C x_pre, the largest row of each block -- position, velocity,
           foot height.  This is the number that says the filter and the legs
           disagree, and it is where a frame error appears first.
    acc    the 0x40 packet's own age.  It is a DIFFERENT packet from the
           attitude, on a different clock, so it is aged separately.

    And on entering every phase, one `estimator, <phase>:` line with the same
    fields -- the per-phase reading this file exists for.  Leaving a trot adds
    one more, the yaw and the xy the trot itself moved:

        trot yaw  -0.3 -> -7.2 deg (d -6.9) off the latched heading;
            xy moved (+0.031, -0.055) m, |d| 0.063 over 18.4 s

WHAT ZEROES YAW, AND WHY THE TROT'S IS PRINTED
    ONE THING ZEROES IT, ONCE.  `law.BalanceLaw.arm` latches `yaw_offset` from
    the heading measured at the crouch -> rise handover and `state.rezero_yaw`
    puts every reading after it in that frame, so the yaw printed here is
    drift off that one heading.  NOTHING ELSE TOUCHES IT: starting a trot does
    not, latching its exit does not, and this tap does not -- `yaw_offset` is
    also the law's heading SETPOINT (`arm` builds `R_des` from it), so
    re-zeroing it mid-run would move where the attitude loop is pulling the
    trunk.  That is a control change, and a read-out does not get to make one.

    WHAT DOES ACT ON IT IS `config.KP_YAW`, ON SINCE 2026-09-24 BECAUSE OF
    THAT LATCH.  The heading spring holds the drift printed here at 25 /s^2,
    0.80 Hz, critically damped, and these lines are its report card: the
    `trot yaw` line below is the number to read before and after `--kp-yaw 0`,
    which is the trot as it ran with the damper alone.  The spring does not
    make the xy trustworthy -- it makes d yaw small enough that the xy can be
    read at all.

    WHICH IS WHY THE TROT'S YAW IS THE FIRST NUMBER TO READ BEFORE THE xy.
    The filter's x and y are leg odometry: world foot positions differentiated
    through R.  Turn the trunk by psi and the whole stance turns with it, so a
    trot that ends 7 deg off its heading has moved every foot some 20 mm
    sideways at a 160 mm stance radius WITHOUT THE ROBOT GOING ANYWHERE.  An
    xy read across a trot is a claim about the filter only to the extent that
    d yaw is small; when it is not, the yaw line above says so and the xy is
    that rotation before it is drift, bias or a frame error.

WHICH WORLD, AND WHY THE FILTER IS RESET ONCE
    The tap puts every measurement in the run's own world frame first
    (`state.rezero_yaw` at `stand.yaw_offset`), so the filter's v and xy are in
    the frame the law controls in: world x is where the trunk pointed at the
    handover, not where the magnetometer calls north.

    BOTH SOURCES, NOT ONE.  `rezero_yaw` turns `body.R`, and the legs read
    off `body`; the IMU source is built from the sweep's `TrunkOrientation`,
    whose R still carries the magnetometer's heading, so it is handed
    `body.R` in place of its own (`_update`).  FOUND ON THE ROBOT 2026-09-24:
    with the raw R the filter integrated in the magnetometer's world while
    `reset` had laid the feet out in the run's, and at that day's 170 deg
    heading the trunk walking forward printed as -x and left as -y, with the
    status line's yaw reading 0.3 deg the whole time because that yaw is the
    law's, already zeroed.  `tests/test_dog6_sources.py` pins a forward walk
    at that heading to +x.

    That frame is fixed ONCE, at the end of the crouch, and the instant it
    turns is the one instant the filter's own state -- p, v and four world foot
    positions -- would be left behind in the old frame.  So the tap resets the
    filter on the sweep the offset changes, which is the handover: feet
    planted, the crouch arrived, torque just live.  `reset` starts it at the
    legs' own geometry, which is the only thing it can honestly start from.
    Before the handover the offset is 0.0 and the world is the magnetometer's;
    that is the frame the limp, settle and crouch readings are in.

THE CONTACT PHASES ARE THE GAIT'S, NOT FOUR FEET DOWN
    Four feet down (`run_once`'s default) says every foot is planted on the
    floor.  In a trot two of them are up to 40 mm above it twice a cycle, so
    while `trot` or `step` is running the tap hands the filter the SCHEDULE's
    phases -- `adapters.stance_phase(gait.phase(now), gait.duty)` -- and each
    leg's trust ramps to 0 through its swing.  Same gait object the law is
    using that sweep, same lookup `sequence.update` does.

    The schedule is not a contact sensor.  `hw.balance.state.on_stance`
    averages the height over the same intended set, for the same reason.

    AND THE TWO RAMPS DO NOT LINE UP, WHICH THE PRINT SHOWS.  The gait's load
    ramp is `config.CONTACT_RAMP` = 0.15 of stance; the filter's trust ramp is
    MIT's `trust_window` = 0.20 of it.  So at the settle's four-foot window the
    status line reads `weight [1. 1. 1. 1.]` beside `trust 0.94` on ALL FOUR
    legs: the gait has the load fully handed across and the filter is still
    fading every leg in, at s = 1 + 100 (1 - tau) = 7.25 instead of 1.  Neither
    number is tuned to the other here -- one is MIT's and one is DOG5's flown
    one -- and
    `tests/test_dog6_sources.py` pins the gap so it is a decision when the
    filter enters the loop rather than a surprise.

IT COSTS AS MUCH AS THE LAW DOES, AND THAT IS WHY IT HAS ITS OWN SLOT
    MEASURED on the Pi, 2026-09-24: one `run_once` is 290 us (p95 301), and the
    tap around it is 350 us of a 333 us CAN slot.  The 28x28 solve in
    `lkf.update` is most of it.  The balance law is p50 560 us of the SAME
    budget inside this loop (470 us in `balance.selftest`), so slot 0 is over
    its 333 us before the tap is added and there is nothing there to share: run
    at slot 0, the tap put every sweep past the 1 ms re-anchor line -- 2000
    overruns in a run against 59 without it -- and widened the worst CAN gap
    from 6.2 to 6.5 ms.  `stand.ESTIMATOR_SLOT` gives it a slot of its own,
    after that slot's frame is out, on the sweep's own latched measurement.
    With it there the run is 85 overruns against 59, and the worst gap is the
    same 6.2 ms.

    THAT IS A TAP'S LUXURY AND NOT A CONTROLLER'S.  A law that fed on this
    would need the estimate BEFORE it acts, which means slot 0, which does not
    fit today.  The status line prints `cost mean/max us` every half second so
    the number stays in front of whoever has to solve that.

WHAT IT IS NOT
    Not in the log: `--log` writes `hw.trot`'s columns.  Not in the law: no
    torque, no trip and no setpoint reads it.  Not a yaw estimate: yaw is the
    magnetometer's, `hw.imu` calls it untrusted, and it reaches the filter
    through R -- so x and y are only as good as it is, while z and v_z do not
    depend on it at all.  Not a yaw CORRECTION either: the yaw it prints is
    measured and reported, never zeroed -- see above.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/trot_esti.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import time                          # noqa: E402
from dataclasses import replace      # noqa: E402

import numpy as np                   # noqa: E402

from sim import params as P          # noqa: E402

from . import stand as STAND         # noqa: E402
from . import trot as TROT           # noqa: E402
from . import velocity_estimator as VEL  # noqa: E402
from .balance import state as BSTATE  # noqa: E402
from .state_estimator import adapters as ADAPT  # noqa: E402
from .state_estimator.estimator import (         # noqa: E402
    LinearKFPosVelEstimator, LKFParams)

__all__ = ["EstimatorTap", "main", "ACC_WAIT_S", "DT_CLAMP"]

#: Seconds to wait, BEFORE arming, for the DETA10's first 0x40 packet.  The
#: accelerometer is NaN until one arrives and the filter cannot run on that;
#: waiting here costs nothing, waiting in the loop would cost a CAN deadline.
ACC_WAIT_S = 3.0

#: The sweep period clamp, `hw.velocity_estimator`'s -- DOG5's own bounds.  One
#: stalled sweep integrated at face value is a step; past the upper bound the
#: gap is integrated AS the bound instead.
DT_CLAMP = VEL.DT_CLAMP


class EstimatorTap:
    """`hw.state_estimator`'s filter, stepped once per sweep and printed.

    Built by `hw.stand.main` as `estimator(imu)` before the bus is armed, so
    the wait for the first accelerometer packet happens here, not in the loop.
    `imu` is None on the `--no-imu` path and then every sweep is
    `TrunkOrientation.level()`: the level-trunk ablation, which exercises the
    whole path and measures nothing.
    """

    def __init__(self, imu, params: LKFParams = None,
                 dt_clamp: tuple = DT_CLAMP) -> None:
        self.imu = imu
        #: `config/lkf.yaml` holds the same ten numbers for a caller that
        #: wants to tune without editing code; it is not read here, because
        #: PyYAML is not installed on the robot and a print-only tap is not
        #: the place to add a dependency the control path would then have.
        self.params = LKFParams() if params is None else params
        self.est = LinearKFPosVelEstimator(self.params)
        self.dt_clamp = dt_clamp
        self.out = None               # the last LKFOutput
        self.body = None              # the sweep it was made from, rezeroed
        self.dt = float("nan")
        #: The 0x40 packet's own age at the last sweep.  Aged apart from the
        #: attitude on purpose: one stream arriving is no evidence about the
        #: other, and a dead accelerometer behind a live AHRS is exactly the
        #: failure a shared age would hide.
        self.acc_age_s = float("inf")
        self.t_prev = None
        self.world = None             # the yaw_offset the filter was reset in
        self.refusal = None           # why this sweep was skipped, or None
        self.said_refusal = False
        self.broken = None            # an unexpected failure: the tap is off
        self.phase_said = None
        #: The trunk's heading drift off the run's world x axis, degrees, this
        #: sweep -- `body.yaw` after the rezero.
        self.yaw_deg = float("nan")
        #: (yaw_deg, x, y, t) at the sweep a trot began, or None.  Kept so the
        #: turn the trot put in can be printed beside the xy it moved, which
        #: is the one comparison that says whether the xy means anything.
        self.trot_entry = None
        self.sweeps = 0
        self.cost_s = 0.0             # the tap's own time, summed
        self.cost_n = 0
        self.cost_max_s = 0.0
        if imu is not None and not self._wait_for_acc(imu):
            print("  estimator: no 0x40 packet in %.0f s -- the accelerometer "
                  "stream looks off, and the filter cannot start without it"
                  % ACC_WAIT_S)

    @staticmethod
    def _wait_for_acc(imu, timeout: float = ACC_WAIT_S) -> bool:
        """Block until the DETA10 has delivered one 0x40 packet.

        `ImuDog.wait_for_data` returns on ANY packet, usually the attitude
        one, which proves nothing about the accelerometer -- the same trap
        `velocity_estimator.ImuRawFeed.wait_for_raw` exists for.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if np.isfinite(imu.acc_age_s()):
                return True
            time.sleep(0.005)
        return bool(np.isfinite(imu.acc_age_s()))

    # -- the sweep ---------------------------------------------------------
    def update(self, now: float, stand, body, orientation) -> str | None:
        """One sweep.  Returns a line to print, once, or None.

        Called by `stand.run` in `stand.ESTIMATOR_SLOT`, with the `BodyState`
        and `TrunkOrientation` LATCHED at slot 0 -- so the reading is the
        sweep's own and only the arithmetic is 2 ms later.  `stand` is the live
        object, so this sweep's phase and yaw offset are already in it.
        Nothing this returns or stores reaches the law.
        """
        if self.broken:
            return None
        started = time.perf_counter()
        try:
            return self._update(now, stand, body, orientation)
        except Exception as failure:                     # noqa: BLE001
            self.broken = "%s: %s" % (type(failure).__name__, failure)
            return ("estimator OFF -- %s.  A read-out must not be able to end "
                    "a run, so the tap is disabled and the trot goes on."
                    % self.broken)
        finally:
            # ITS OWN SLOT IS 333 us AND THE FILTER IS 290 OF THEM.  The cost
            # is printed every status line because that is the number which
            # decides whether this can ever move into slot 0 beside the law.
            spent = time.perf_counter() - started
            self.cost_s += spent
            self.cost_n += 1
            self.cost_max_s = max(self.cost_max_s, spent)

    def _update(self, now: float, stand, body, orientation) -> str | None:
        self.sweeps += 1
        # THE RUN'S WORLD, NOT THE MAGNETOMETER'S.  A no-op before the
        # handover and a no-op on a body already in it.
        self.body = body = BSTATE.rezero_yaw(body, stand.yaw_offset)
        self.yaw_deg = float(np.degrees(body.yaw))
        # THE IMU SOURCE TAKES THE REZEROED R TOO.  `orientation.R` still has
        # the magnetometer's heading in it; `body.R` is Rz(-yaw_offset) times
        # that.  Hand the filter the raw one and it integrates every leg
        # velocity in the magnetometer's world while `reset` above laid the
        # feet out in the run's: at a 170 deg heading, which is where DOG6
        # stood on 2026-09-24, the trunk walking forward read as -x and left
        # read as -y.  omega_b and acc_b are trunk-frame and a yaw of the
        # world does not touch them, so R is the only field that changes.
        src = ADAPT.OrientationImu(replace(orientation, R=body.R))
        legs = ADAPT.BodyLegs(body)
        try:
            if stand.yaw_offset != self.world:
                # The world frame turned (or this is the first sweep): the
                # filter's p, v and feet are in the old one.  Read the IMU
                # FIRST -- a reset into a frame the tap cannot then integrate
                # in is a reset that hides why nothing is printing.
                src.read()
                self.est.reset(body.R, body.x_b)
                self.world = float(stand.yaw_offset)
                self.t_prev = now
                self.refusal = None
                return self._reset_line()
            dt = float(np.clip(now - self.t_prev, *self.dt_clamp))
            self.out = ADAPT.run_once(src, legs, self.est, dt, P.FOOT_RADIUS,
                                      self._contact_phase(now, stand))
            self.dt = dt
            self.t_prev = now
            self.acc_age_s = float(orientation.acc_age_s)
            self.refusal = None
        except ValueError as refusal:
            # The IMU source refused the sweep -- the accelerometer is not
            # finite.  Expected before the first 0x40 packet, and the state is
            # untouched: `run_once` reads before it updates.
            self.refusal = str(refusal)
            if self.said_refusal:
                return None
            self.said_refusal = True
            return "estimator waiting -- " + self.refusal
        if self.out is not None and stand.phase_name != self.phase_said:
            was, self.phase_said = self.phase_said, stand.phase_name
            return self._phase_lines(now, stand.phase_name, was)
        return None

    def _phase_lines(self, now: float, phase: str, was: str | None) -> str:
        """The per-phase reading, and across a trot the turn it put in."""
        lines = []
        if phase == "trot":
            self.trot_entry = (self.yaw_deg, float(self.out.p_w[0]),
                               float(self.out.p_w[1]), float(now))
        elif was == "trot" and self.trot_entry is not None:
            lines.append(self._trot_yaw_line(now))
            self.trot_entry = None
        lines.append(self.reading(phase))
        # `stand.run` indents the first line; the rest are indented here.
        return "\n   ".join(lines)

    def _trot_yaw_line(self, now: float) -> str:
        """Yaw and xy, trot entry against trot exit.  Two lines."""
        yaw0, x0, y0, t0 = self.trot_entry
        dx = float(self.out.p_w[0]) - x0
        dy = float(self.out.p_w[1]) - y0
        return ("trot yaw %+.1f -> %+.1f deg (d %+.1f) off the latched "
                "heading;\n       xy moved (%+.3f, %+.3f) m, |d| %.3f over "
                "%.1f s"
                % (yaw0, self.yaw_deg, self.yaw_deg - yaw0, dx, dy,
                   float(np.hypot(dx, dy)), now - t0))

    def _contact_phase(self, now: float, stand):
        """The filter's contact phase, or None for four feet down.

        The same `{"trot": gait, "step": step_gait}` lookup `sequence.update`
        makes, so the phases the filter is told about are the phases the swing
        is being commanded from.
        """
        gait = {"trot": stand.gait, "step": stand.step_gait}.get(
            stand.phase_name)
        if gait is None:
            return None
        return ADAPT.stance_phase(gait.phase(now), gait.duty)

    # -- what the operator reads -------------------------------------------
    def _reset_line(self) -> str:
        # x[2] IS THE BALL-CENTRE DATUM -- `to_floor` is on the way out of
        # `run_once` and the filter's own state never sees it.  Print it
        # without the radius and this line reads 15 mm below every other
        # height in `hw`, which is the whole reason `foot_radius` is a
        # required argument one level down.
        return ("estimator reset at the legs: h %.1f mm, world x at heading "
                "%+.1f deg%s"
                % (1e3 * BSTATE.origin_to_height(self.est.x[2] + P.FOOT_RADIUS),
                   np.degrees(self.world),
                   "" if self.world else " (the magnetometer's own)"))

    def reading(self, phase: str) -> str:
        """The per-phase read-out: the filter beside the legs, one line."""
        out, body = self.out, self.body
        h = BSTATE.origin_to_height(out.p_w[2])
        return ("estimator, %-6s h %6.1f mm (fk %6.1f, d %+5.1f)  "
                "v (%+.3f, %+.3f, %+.3f) m/s  xy (%+.3f, %+.3f) m  "
                "yaw %+6.1f deg  %s"
                % (phase + ":", 1e3 * h, 1e3 * body.h, 1e3 * (h - body.h),
                   out.v_w[0], out.v_w[1], out.v_w[2],
                   out.p_w[0], out.p_w[1], self.yaw_deg, self._innov()))

    def status(self) -> str:
        """Two lines under every status line.  `stand.run` indents them."""
        if self.broken:
            return "est OFF -- " + self.broken
        if self.out is None:
            return "est -- (%s)" % (self.refusal or "no sweep yet")
        out, body = self.out, self.body
        h = BSTATE.origin_to_height(out.p_w[2])
        return "\n".join((
            "est h %6.1f mm (fk %6.1f, d %+5.1f)  v (%+.3f, %+.3f, %+.3f) m/s"
            "  vz fk %+.3f"
            % (1e3 * h, 1e3 * body.h, 1e3 * (h - body.h),
               out.v_w[0], out.v_w[1], out.v_w[2], body.zdot_origin),
            "    xy (%+.3f, %+.3f) m  yaw %+6.1f deg  trust %s  %s"
            "  acc %3.0f ms  dt %.1f ms  cost %3.0f/%3.0f us"
            % (out.p_w[0], out.p_w[1], self.yaw_deg,
               "/".join("%.2f" % t for t in out.trust), self._innov(),
               1e3 * self.acc_age_s, 1e3 * self.dt,
               1e6 * self.cost_s / max(self.cost_n, 1), 1e6 * self.cost_max_s)))

    def _innov(self) -> str:
        """The largest row of each innovation block: where a frame error shows."""
        e = np.abs(self.out.innov)
        return ("innov p %.1f mm / v %3.0f mm/s / h %.1f mm"
                % (1e3 * e[0:12].max(), 1e3 * e[12:24].max(),
                   1e3 * e[24:28].max()))


def main(argv=None) -> int:
    """`hw.trot.main`'s run, with `EstimatorTap` printing beside it."""
    return STAND.main(argv, crouch=TROT.CROUCH, only_law="srb",
                      tilt_stop=TROT.TILT_STOP_DEG,
                      track_stop=TROT.TRACK_STOP_DEG,
                      step_to=TROT.STEP_FOOT_XY,
                      step_period=TROT.STEP_PERIOD_S,
                      velocity=False, estimator=EstimatorTap, terse=True,
                      **TROT.trot_options(TROT.PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
