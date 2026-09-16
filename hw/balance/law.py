"""The five stages chained: one call per sweep, from sensors to twelve torques.

    BalanceLaw.arm(now, state)      latch h0 and the heading, plan the ramp
    BalanceLaw.update(now, state)   -> LawOutput

`hw.stand` owns WHEN this runs -- the phase machine, the CAN slots, the keys.
This owns WHAT it computes.  Keeping the two apart is what lets the whole law
be exercised offline, against a `state.BodyState` built by hand, with no bus
and no IMU in the process.

WHAT IT DOES NOT DO
    It does not shape torque.  Ramp, cap, limit block and slew are
    `safety.SafetyGate`'s, applied by the runner AFTER this returns, so there
    is exactly one limiter per quantity and the runaway argument stays
    countable.  It does not stop the run either: it RETURNS trip reasons and
    the runner decides, because a trip during the lift is a drop and that
    decision belongs with the thing that can also choose to limp.

ONE RATE, AND THAT IS A CHOICE WORTH KEEPING
    The cMPC stack runs its blocks at different rates because it has a
    prediction horizon and has to hold a solution between solves.  DOG6 is the
    degenerate case: there is no horizon, every block is closed form, and
    everything here runs at the sweep rate.  That removes the held-solution
    staleness cMPC has to reason about.

    The ONE exception is the leg gravity term, which has no closed form and is
    the expensive one -- `gravity_legs_per_sweep` sub-rates it.  It defaults to
    all four, which removes the 12 ms skew the per-leg law carries.  If the
    slot will not hold the law, this is the first thing to turn down and the
    attitude loop is the last.

THE FOUR TRIPS IT CAN RAISE, AND THE ONE IT DELIBERATELY DOES NOT
    tilt         |roll| or |pitch| past cfg.TILT_STOP_DEG.  The stand had no
                 way to trip on the failure it actually exhibits.
    tracking     the IK at the pinned foot xy and the commanded height gives
                 the joint-space target the stand otherwise does not have, and
                 catches a badly wrong leg long before the tilt stop does.
    residual     sustained, via `allocation.ResidualMonitor`.
    non-finite   a NaN reaching the drivers is twelve motors at the cap.

    IMU STALENESS IS NOT A TRIP.  Past cfg.IMU_MAX_AGE_S the attitude half of
    the PD is frozen to zero and the height loop and gravity keep running --
    which is the behaviour a robot with no attitude sensor has anyway, and is
    what the per-leg law did for its entire life.  `LawOutput.imu_held` says
    it happened and the runner prints it once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/law.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import stand as ST          # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import allocation as ALLOC    # noqa: E402
from . import config as cfg          # noqa: E402
from . import controller as CTRL     # noqa: E402
from . import reference as REF       # noqa: E402
from . import torque as TRQ          # noqa: E402

__all__ = ["BalanceLaw", "LawOutput", "Timing", "ik_reference"]


def ik_reference(h_cmd: float, q_seed) -> np.ndarray:
    """(4, 3) joint angles putting the four feet at FOOT_XY and height `h_cmd`.

    THE TRACKING TRIP'S TARGET, and nothing else -- it is never commanded.
    `h_cmd` is in h (floor to trunk bottom); `sim.stand.foot_targets` is in
    the trunk-origin frame, so the conversion happens here and in one place.

    `hw.kinematics`' CLOSED-FORM inverse, seeded with the measured pose:
    20 us a leg against `sim.kinematics.leg_ik`'s 463 us, a fixed operation
    count, and no tolerance to stop on.  All three matter in a 333 us slot.
    The seed also stops the solver returning a mirrored elbow or a 2 pi step,
    which as a trip threshold would read as a leg that is wildly wrong.
    """
    targets = ST.foot_targets(h_cmd + cfg.TRUNK_BOTTOM_OFFSET)
    return HK.all_leg_ik(targets, q_seed=q_seed)


class Timing:
    """Per-sweep cost of the law, as a running p50/p95/max.

    ADDED AT THE SAME TIME AS THE LAW, NOT AFTER.  The whole block runs BEFORE
    slot 0's frame goes out, so it delays the same motor on every sweep; the
    kinematics alone already cost 0.09 ms of a 333 us slot.  A timing number
    that arrives after the first bad run is a number nobody has.

    Keeps the samples rather than a running moment, because p95 is the useful
    statistic and a mean hides exactly the occasional long sweep that matters.
    Capped so a long run cannot grow without bound.
    """

    LIMIT = 20000

    def __init__(self) -> None:
        self.samples: list[float] = []
        self.dropped = 0

    def add(self, seconds: float) -> None:
        if len(self.samples) < self.LIMIT:
            self.samples.append(float(seconds))
        else:
            self.dropped += 1

    def summary(self) -> str:
        if not self.samples:
            return "no samples"
        a = np.asarray(self.samples)
        return ("n %d  p50 %.0f us  p95 %.0f us  max %.0f us%s"
                % (a.size, 1e6 * np.percentile(a, 50),
                   1e6 * np.percentile(a, 95), 1e6 * a.max(),
                   "  (+%d dropped)" % self.dropped if self.dropped else ""))


@dataclass
class LawOutput:
    """One sweep's result.  Everything the log needs, and nothing derived."""

    tau: np.ndarray                  # (12,) N*m, BEFORE the safety gate
    command: REF.HeightCommand       # the reference in h
    com_cmd: REF.ComCommand
    wrench: CTRL.Wrench
    allocation: ALLOC.Allocation
    q_ref: np.ndarray                # (12,) the tracking trip's target
    imu_held: bool                   # the attitude half was frozen this sweep
    trip: str | None


@dataclass
class BalanceLaw:
    """The controller, armed once and then called once per sweep."""

    gains: CTRL.BalanceGains = field(default_factory=CTRL.BalanceGains)
    rise_s: float = cfg.T_RISE
    h_lift: float = cfg.H_LIFT
    gravity_legs_per_sweep: int = C.N_LEGS
    mu: float = cfg.MU

    def __post_init__(self) -> None:
        self.ramp: REF.Quintic | None = None
        self.t0 = 0.0
        self.R_des = np.eye(3)
        self.h0 = float("nan")
        self.residual = ALLOC.ResidualMonitor()
        self.timing = Timing()
        self.gravity = np.zeros((C.N_LEGS, 3))
        self.tau_peak = 0.0
        self.imu_held_sweeps = 0
        self._sweep = 0

    # -- arming ----------------------------------------------------------
    def arm(self, now: float, state) -> None:
        """Latch h0 and the heading, and plan the ramp.  Call once, at handover.

        h0 IS THE MEASURED HEIGHT, NOT `CROUCH_HEIGHT`.  Starting a ramp from
        a height the robot is not at is a step input -- into k_d,z, which
        differentiates it.  The same argument is why the heading is latched
        here rather than tracked: R_des must not move under the loop.
        """
        self.t0 = float(now)
        self.h0 = float(state.h)
        self.ramp = REF.Quintic.ramp(self.h0, self.h_lift, self.rise_s)
        self.R_des = CTRL.level_attitude(state.yaw)
        self.gravity = TRQ.all_leg_gravity_torque(state.q, state.R)

    def replan(self, now: float, state, h_target: float, seconds: float) -> None:
        """Re-plan from the CURRENT (h, hdot, hddot) instead of restarting s.

        Pause, retime and abort are all this call.  It keeps the reference C2
        across the re-plan, where restarting the S-curve would put a velocity
        step into the middle of a lift.
        """
        if self.ramp is None:
            raise RuntimeError("BalanceLaw.arm() has not been called")
        current = self.ramp.at(float(now) - self.t0)
        self.ramp = REF.Quintic.between(current.h, current.hdot, current.hddot,
                                        float(h_target), 0.0, 0.0, seconds)
        self.t0 = float(now)
        self.h_lift = float(h_target)

    @property
    def armed(self) -> bool:
        return self.ramp is not None

    def remaining(self, now: float) -> float:
        """Seconds of ramp left; 0 once it has arrived."""
        if self.ramp is None:
            return 0.0
        return max(0.0, self.ramp.T - (float(now) - self.t0))

    # -- the sweep -------------------------------------------------------
    def update(self, now: float, state, clock=None) -> LawOutput:
        """Stages 1-5 for one sweep.  `state` is a `state.BodyState`."""
        if self.ramp is None:
            raise RuntimeError("BalanceLaw.arm() has not been called")
        started = None if clock is None else clock()
        self._sweep += 1

        # -- the reference -------------------------------------------------
        command = self.ramp.at(float(now) - self.t0)
        com_cmd = REF.com_command(command, state.R)

        # -- stage 3 -------------------------------------------------------
        held = bool(state.imu_stale)
        if held:
            self.imu_held_sweeps += 1
        wrench = CTRL.balance_wrench(state, com_cmd, self.R_des, self.gains,
                                     hold_attitude=held)

        # -- stage 4 -------------------------------------------------------
        allocation = ALLOC.allocate(state.r_w, wrench.b_d, mu=self.mu)

        # -- stage 5 -------------------------------------------------------
        # Sub-rated when asked: `gravity_legs_per_sweep` legs are refreshed,
        # the rest keep the previous sweep's value.  At the default of four
        # nothing is held and the 12 ms cross-leg skew is gone.
        if self.gravity_legs_per_sweep >= C.N_LEGS:
            self.gravity = TRQ.all_leg_gravity_torque(state.q, state.R)
        else:
            q4 = C.unflat(state.q)
            for k in range(max(1, self.gravity_legs_per_sweep)):
                leg = (self._sweep * self.gravity_legs_per_sweep + k) % C.N_LEGS
                self.gravity[leg] = TRQ.leg_gravity_torque(leg, q4[leg], state.R)
        tau = TRQ.stance_torque(state, allocation.f_w, self.gravity)

        # -- the trips -----------------------------------------------------
        q_ref = ik_reference(command.h, C.unflat(state.q))
        trip = self._trip(state, allocation, C.flat(q_ref))
        if trip is None:
            self.tau_peak = max(self.tau_peak, float(np.abs(tau).max()))

        if started is not None:
            self.timing.add(clock() - started)
        return LawOutput(tau=tau, command=command, com_cmd=com_cmd,
                         wrench=wrench, allocation=allocation,
                         q_ref=C.flat(q_ref), imu_held=held, trip=trip)

    def _trip(self, state, allocation, q_ref) -> str | None:
        if not np.all(np.isfinite(allocation.f_w)):
            return ("the allocator returned a non-finite force -- the grasp "
                    "map is singular or the state is NaN")
        if state.tilt_deg > cfg.TILT_STOP_DEG:
            return ("tilt %.1f deg past the %.0f deg stop (roll %+.1f, "
                    "pitch %+.1f)" % (state.tilt_deg, cfg.TILT_STOP_DEG,
                                      np.degrees(state.roll),
                                      np.degrees(state.pitch)))
        error = np.abs(state.q - q_ref)
        if np.any(error > cfg.TRACK_STOP_RAD):
            from ..hardware_map import JOINT_LABELS
            index = int(np.argmax(error))
            return ("tracking: %s is %.1f deg from the IK at the commanded "
                    "height (limit %.0f)"
                    % (JOINT_LABELS[index], np.rad2deg(error[index]),
                       np.rad2deg(cfg.TRACK_STOP_RAD)))
        return self.residual.reason(allocation)

    # -- the exit report -------------------------------------------------
    def report(self) -> str:
        return "\n".join([
            "  law timing      %s" % self.timing.summary(),
            "  peak |tau|      %.2f N*m (before the gate)" % self.tau_peak,
            "  peak residual   %.2f N / %.3f N*m"
            % (self.residual.peak_force, self.residual.peak_moment),
            "  imu held        %d sweeps" % self.imu_held_sweeps,
        ])
