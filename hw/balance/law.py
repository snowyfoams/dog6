"""The five stages chained: one call per sweep, from sensors to twelve torques.

    BalanceLaw.arm(now, state)      latch h0 and the heading, plan the ramp
    BalanceLaw.update(now, state)   -> LawOutput

`sequence` owns WHEN this runs -- it is the phase machine, and the lift phase
is the one place it calls this.  `hw.stand` owns the CAN slots and the keys.
This owns WHAT it computes.  Keeping the three apart is what lets the whole law
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
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import allocation as ALLOC    # noqa: E402
from . import config as cfg          # noqa: E402
from . import controller as CTRL     # noqa: E402
from . import reference as REF       # noqa: E402
from . import state as STATE         # noqa: E402
from . import swing as SWING         # noqa: E402
from . import torque as TRQ          # noqa: E402

__all__ = ["BalanceLaw", "LawOutput", "Timing", "ik_reference"]


def _wrap_pi(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.remainder(a + np.pi, 2.0 * np.pi) - np.pi)


def ik_reference(h_cmd: float, q_seed, foot_xy=None) -> np.ndarray:
    """(4, 3) joint angles putting the four feet at `foot_xy` and height `h_cmd`.

    THE TRACKING TRIP'S TARGET, and nothing else -- it is never commanded.
    `h_cmd` is in h (floor to trunk bottom); `sim.stand.foot_targets` is in
    the trunk-origin frame, so the conversion happens here and in one place.

    `foot_xy` IS WHERE THE FEET ARE PINNED WHILE THE TRUNK RISES, in each
    leg's own HIP frame, and None means `sim.stand.FOOT_XY` -- the nominal
    crouch's.  It is an argument because it is a property of the POSTURE THE
    LIFT STARTED FROM, not of the robot: a crouch folded by hand puts the feet
    somewhere else entirely, and measuring the tracking error against the
    nominal stance would then report the difference between two postures as a
    tracking failure.  From `posture.FOLD` that reads 91.6 deg on the first
    sweep, against a 25 deg trip.

    Only z carries the height, for all four, which is to say THE REFERENCE IS
    A LEVEL TRUNK -- the attitude the law is driving toward, whatever the
    crouch was resting at.

    `hw.kinematics`' CLOSED-FORM inverse, seeded with the measured pose:
    20 us a leg against `sim.kinematics.leg_ik`'s 463 us, a fixed operation
    count, and no tolerance to stop on.  All three matter in a 333 us slot.
    The seed also stops the solver returning a mirrored elbow or a 2 pi step,
    which as a trip threshold would read as a leg that is wildly wrong.
    """
    height = float(h_cmd) + cfg.TRUNK_BOTTOM_OFFSET
    if foot_xy is None:
        targets = ST.foot_targets(height)
    else:
        targets = np.zeros((C.N_LEGS, 3))
        targets[:, :2] = np.asarray(foot_xy, dtype=float)
        targets[:, 2] = -(height - P.FOOT_RADIUS)
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
    #: The state the law ACTED ON.  In a stand this is the caller's; in a
    #: trot its height is re-taken over the stance feet (`state.on_stance`).
    state: object = None
    #: The trot, when one is running; all None in a stand.
    contact: np.ndarray | None = None    # (4,) the weight the allocator got
    swing_s: np.ndarray | None = None    # (4,) swing progress, 0 in stance
    p_swing: np.ndarray | None = None    # (4, 3) swing target, TRUNK; NaN in stance


@dataclass
class BalanceLaw:
    """The controller, armed once and then called once per sweep."""

    gains: CTRL.BalanceGains = field(default_factory=CTRL.BalanceGains)
    rise_s: float = cfg.T_RISE
    h_lift: float = cfg.H_LIFT
    gravity_legs_per_sweep: int = C.N_LEGS
    mu: float = cfg.MU
    #: (4, 2) HIP-frame foot xy the tracking reference pins; None = nominal.
    #: `posture.CrouchPose.foot_xy` is what fills it.
    foot_xy: np.ndarray | None = None
    #: Whether the attitude setpoint is LATCHED at zero torque or taken from
    #: the config statics.  None defers to `config.SETPOINT_DYNAMIC`; False is
    #: a FIXED OFFSET, which is what a run that wants two postures compared
    #: needs -- a per-run datum would move "level" between the two and the
    #: comparison would be against two different references.
    dynamic_setpoint: bool | None = None
    #: Where the tilt e-stop fires, in degrees from the SETPOINT.  None takes
    #: `config.TILT_STOP_DEG`.  IT IS A TRIP, NOT A TUNING -- raising it is a
    #: deliberate act by whoever is standing next to the robot, which is why
    #: it is an argument and why the banner prints it on every run.
    tilt_stop_deg: float | None = None
    #: The joint TRACKING trip, in degrees.  None takes `config.TRACK_STOP_RAD`;
    #: 0 (or less) turns it OFF.  It is a MONITOR and nothing else --
    #: `ik_reference` feeds only `_trip` and the log, never the torque -- so
    #: switching it off changes no N*m the drivers receive.  What it gives up
    #: is the one check that notices a LEG far from where the model thinks it
    #: is while the attitude still reads fine.  `q_ref` is still computed and
    #: still logged either way.
    track_stop_deg: float | None = None
    #: The pinned c^b and I^b, `posture.CrouchPose.srb`.  None is the nominal
    #: `config.SRB`.  It reaches BOTH the reference and the measurement from
    #: here, which is what keeps the CoM offset cancelling in the z error.
    srb: "cfg.SrbModel | None" = None
    #: The swing leg's controller, `swing.SWING_MODES`.  "cartesian" is the
    #: impedance the nominal and wide trots flew; "joint" is the fold's: the
    #: same z-only arc through the IK, abd held, joint PD.  See swing.py.
    swing: str = "cartesian"
    #: m, the swing apex above the resting foot.  `config.SWING_HEIGHT` (40 mm,
    #: DOG5's) unless the entry point says otherwise -- `--swing-height`.  It
    #: is the one number that scales the whole swing demand (speed, torque,
    #: slew) linearly, so it is the first thing to lower for a short swing.
    swing_height: float = cfg.SWING_HEIGHT
    #: The joint-space layer, `config.KP_JOINT_HOLD`: live only while
    #: `hold_joints` has latched a pose.  OFF (0 / 0) unless the entry point
    #: passes gains -- the trot entry points do, the plain stand does not.
    kp_joint: float = 0.0
    kd_joint: float = 0.0

    def __post_init__(self) -> None:
        if self.swing not in SWING.SWING_MODES:
            raise ValueError("swing %r: one of %s"
                             % (self.swing, ", ".join(SWING.SWING_MODES)))
        # OWN COPY: the foot step moves it leg by leg, and it usually arrives
        # as a posture's array.
        if self.foot_xy is not None:
            self.foot_xy = np.array(self.foot_xy, dtype=float)
        #: The foot step's destination, (4, 2) HIP frame, or None.  See
        #: `begin_step`.
        self.foot_xy_to: np.ndarray | None = None
        self._swinging = np.zeros(C.N_LEGS, dtype=bool)
        #: The joint swing's LIFTOFF LATCH, (4, 3) each, NaN while the leg is
        #: down: the measured trunk-frame foot and joints on the first
        #: swinging sweep.  See the swing block in `update`.
        self._lift_x = np.full((C.N_LEGS, 3), np.nan)
        self._lift_q = np.full((C.N_LEGS, 3), np.nan)
        #: (12,) the joint layer's stance target, or None while it is off.
        self.q_hold: np.ndarray | None = None
        if self.srb is None:
            self.srb = cfg.SRB
        if self.dynamic_setpoint is None:
            self.dynamic_setpoint = bool(cfg.SETPOINT_DYNAMIC)
        if self.tilt_stop_deg is None:
            self.tilt_stop_deg = float(cfg.TILT_STOP_DEG)
        if self.track_stop_deg is None:
            self.track_stop_deg = float(np.rad2deg(cfg.TRACK_STOP_RAD))
        self.ramp: REF.Quintic | None = None
        self.t0 = 0.0
        self.R_des = np.eye(3)
        #: The attitude setpoint, RADIANS, the pair that defines "level" for
        #: this run.  `latch_setpoint` moves them; until it does they are the
        #: config statics, which is exactly what DOG5 flew before the dynamic
        #: latch existed and is what `SETPOINT_DYNAMIC = False` still flies.
        self.sp_roll = float(np.radians(cfg.SETPOINT_ROLL_DEG))
        self.sp_pitch = float(np.radians(cfg.SETPOINT_PITCH_DEG))
        #: The heading half, latched by `arm`.  NaN until then -- there is no
        #: yaw reference before torque is live and saying so beats reporting
        #: an error against zero.  `arm` sets it to EXACTLY ZERO, because by
        #: then the world frame has been turned onto the measured heading:
        #: see `yaw_offset`.
        self.sp_yaw = float("nan")
        #: rad, the magnetometer heading the world frame is pinned to -- what
        #: `state.rezero_yaw` is called with for the rest of the run.  `arm`
        #: latches it from the reading at the handover; 0.0 until then means
        #: "the magnetometer's own world", which is what LIMP through CROUCH
        #: report.  This is DOG5's yaw lock: the number is the same one
        #: `yaw_ref` held, and zeroing the frame by it is DOG5's "world :=
        #: body here".
        self.yaw_offset = 0.0
        #: (3,) rad, roll/pitch/yaw ADDED to the setpoint -- `offset_attitude`.
        #: Zero except while `hw.sway` nods or shakes the trunk.
        self.att_offset = np.zeros(3)
        self.setpoint_latched = False
        self.h0 = float("nan")
        self.residual = ALLOC.ResidualMonitor()
        self.timing = Timing()
        self.gravity = np.zeros((C.N_LEGS, 3))
        self.tau_peak = 0.0
        self.imu_held_sweeps = 0
        self._sweep = 0

    # -- the setpoint, latched at ZERO TORQUE ----------------------------
    def latch_setpoint(self, state) -> tuple | None:
        """setpoint := the roll/pitch being measured RIGHT NOW.  -> (r, p) deg.

        DOG5's SETPOINT_DYNAMIC latch, ported.  `sequence` calls this during
        the LIMP phase -- the one phase with no torque anywhere and the trunk
        resting wherever the operator put it -- and from then on "level" for
        this run means that attitude.  Returns the pair in DEGREES for the
        banner, or None if it refused.

        IT LATCHES ONCE.  A second call is ignored, which is what makes "once
        per run, in limp only" a property of this object rather than a rule
        the caller has to keep.  The asymmetry with the yaw lock is deliberate
        and `config.SETPOINT_DYNAMIC` explains it: heading has no truth to
        return to, level does.

        IT REFUSES A STALE IMU.  A setpoint latched from a packet that arrived
        before the operator put the robot down is a constant the attitude loop
        then fights for the whole run, and unlike the yaw lock there is no
        later re-latch to correct it.  Better to fly the config statics and
        say so.
        """
        if self.setpoint_latched or not self.dynamic_setpoint:
            return None
        if state is None or state.imu_stale:
            return None
        self.sp_roll = float(state.roll)
        self.sp_pitch = float(state.pitch)
        self.setpoint_latched = True
        return (float(np.degrees(self.sp_roll)),
                float(np.degrees(self.sp_pitch)))

    def attitude_error_deg(self, state) -> np.ndarray:
        """(3,) measured rpy MINUS the setpoint, in degrees.  The operator's
        view of the error, not the law's.

        THE LAW DOES NOT SUBTRACT EULER ANGLES -- it forms `e_R` from the
        SO(3) log map of R_des and R, and this is not that.  It is the same
        quantity to within the small-angle agreement of the two, and it is
        what a person means by "how far off is it".  Read `wrench.e_R` for
        what actually multiplies the gain.

        Yaw is against the heading latched at `arm`; before that it is NaN,
        because there is no reference to be wrong about yet.
        """
        dr, dp, dy = self.att_offset
        yaw_err = (float("nan") if not np.isfinite(self.sp_yaw)
                   else np.degrees(_wrap_pi(state.yaw - self.sp_yaw - dy)))
        return np.array([np.degrees(state.roll - self.sp_roll - dr),
                         np.degrees(state.pitch - self.sp_pitch - dp),
                         yaw_err])

    def setpoint_drift_deg(self) -> float:
        """How far the latched pair sits from the config statics, in degrees.

        The number `config.SETPOINT_WARN_DEG` is compared against.  Big means
        the robot was not resting flat when the datum was taken -- propped
        against something, or held.
        """
        return max(abs(np.degrees(self.sp_roll) - cfg.SETPOINT_ROLL_DEG),
                   abs(np.degrees(self.sp_pitch) - cfg.SETPOINT_PITCH_DEG))

    def tilt_from_setpoint_deg(self, state) -> float:
        """max(|roll|, |pitch|) measured FROM THE SETPOINT, in degrees.

        What the tilt stop reads.  `state.tilt_deg` measures from true level
        and is the right number for the log and the operator's status line;
        this is the right one for the trip, because the trip has to mean
        "the robot has left the attitude the law is holding it at".  On a
        floor with any slope in it those two differ by the slope, and DOG5
        made the same choice for the same reason.
        """
        dr, dp, _ = self.att_offset
        return float(np.degrees(max(abs(state.roll - self.sp_roll - dr),
                                    abs(state.pitch - self.sp_pitch - dp))))

    # -- arming ----------------------------------------------------------
    def arm(self, now: float, state) -> None:
        """Latch h0 and the heading, and plan the ramp.  Call once, at handover.

        h0 IS THE MEASURED HEIGHT, NOT `CROUCH_HEIGHT`.  Starting a ramp from
        a height the robot is not at is a step input -- into k_d,z, which
        differentiates it.  The same argument is why the heading is latched
        here rather than tracked: R_des must not move under the loop.

        THE HEADING IS LATCHED HERE AND THE SETPOINT IS NOT.  Yaw is taken at
        the handover, which is the first sweep torque is live, so by
        construction the yaw error is exactly zero on the sweep it is taken
        and arming can never step the wrench -- DOG5's reason, unchanged.
        Roll and pitch were latched earlier, at LIMP, because by the time this
        runs the crouch has already put the trunk on the floor and its lean is
        not what "level" should mean.

        AND THE LATCH TURNS THE WORLD, IT DOES NOT JUST RECORD A NUMBER.  The
        heading read here becomes `yaw_offset`, `state.rezero_yaw` turns the
        world frame onto it for every sweep after this one, and `sp_yaw` is
        then zero rather than a magnetometer reading.  Carrying it as a
        nonzero setpoint instead -- which is what this did before -- left the
        measured heading inside R, and `state.rezero_yaw` sets out what the
        SO(3) log map then does with it: the roll correction is attenuated
        and part of it comes out as pitch.  Latching the same number at the
        same instant and turning the frame by it costs one 3x3 product a
        sweep and leaves nothing for the log map to mix.

        `state` is read in whatever world frame it arrived in, so the raw
        heading is its yaw plus its own offset.  At the handover that offset
        is zero; the sum is written out so a re-arm cannot double-count.
        """
        self.t0 = float(now)
        self.h0 = float(state.h)
        self.ramp = REF.Quintic.ramp(self.h0, self.h_lift, self.rise_s)
        self.yaw_offset = float(state.yaw) + float(state.yaw_offset)
        self.sp_yaw = 0.0
        self.R_des = CTRL.latched_attitude(self.sp_roll, self.sp_pitch,
                                           self.sp_yaw)
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

    # -- the foot step --------------------------------------------------
    def begin_step(self, foot_xy_to) -> None:
        """Move the feet to `foot_xy_to` (4, 2, HIP frame) with the gait.

        Drive `update` with a gait clock from here.  Every leg that swings
        lands on its `foot_xy_to` site instead of where it left, and
        `foot_xy` -- what the tracking reference and the next arc start from
        -- takes the new site AT TOUCHDOWN, leg by leg.  One gait cycle moves
        all four; `step_landed` says when.

        THE SRB MODEL STAYS PINNED.  `srb` and `state.read`'s must be the
        same object or the CoM offset stops cancelling, and the runner reads
        the posture's.  For WIDE -> under the hips that is c^b z 7 mm off and
        x exactly 0 either way (a symmetric stance), so no pitch moment.

        REFUSED with the joint swing: that one lifts in z and nothing else.
        """
        if self.swing == "joint":
            raise ValueError("the joint swing only lifts in z; a foot step "
                             "needs swing='cartesian'")
        if self.foot_xy is None:
            self.foot_xy = np.array(ST.FOOT_XY, dtype=float)
        self.foot_xy_to = np.array(foot_xy_to, dtype=float).reshape(
            C.N_LEGS, 2)
        self._swinging[:] = False

    @property
    def step_landed(self) -> bool:
        """All four feet are down on the step's destination."""
        return (self.foot_xy_to is not None and not self._swinging.any()
                and bool(np.allclose(self.foot_xy, self.foot_xy_to,
                                     rtol=0.0, atol=1e-12)))

    def end_step(self) -> None:
        self.foot_xy_to = None
        self._swinging[:] = False

    # -- a moving attitude target (hw.sway) --------------------------------
    def offset_attitude(self, droll: float, dpitch: float,
                        dyaw: float) -> None:
        """R_des := the latched setpoint PLUS (droll, dpitch, dyaw), rad.

        The tilt stop and the attitude error read the offset too: the trip
        means "left the attitude the law is holding it at", and while the
        target moves that is the moved one.  (0, 0, 0) is the latched hold.
        """
        if self.ramp is None:
            raise RuntimeError("BalanceLaw.arm() has not been called")
        self.att_offset = np.array([droll, dpitch, dyaw], dtype=float)
        self.R_des = CTRL.latched_attitude(self.sp_roll + droll,
                                           self.sp_pitch + dpitch,
                                           self.sp_yaw + dyaw)

    # -- the trot's joint-space layer ------------------------------------
    def hold_joints(self, q) -> None:
        """Latch `q` (12,) as the posture the joint layer holds.  `sequence`
        calls it with the measured angles on the sweep the robot REACHES
        HOLD, and keeps it through the hold and the trot."""
        self.q_hold = np.array(q, dtype=float).reshape(C.N_JOINTS)

    def release_joints(self) -> None:
        self.q_hold = None

    # -- the sweep -------------------------------------------------------
    def update(self, now: float, state, clock=None, gait=None) -> LawOutput:
        """Stages 1-5 for one sweep.  `state` is a `state.BodyState`.

        `gait` is a `gait.TrotGait` while the robot trots, None while it
        stands.  It changes three things and nothing else: the allocator is
        given the clock's contact weights, a swinging leg gets the swing
        impedance on top of its own weight, and the residual trip only counts
        sweeps with all four feet at full weight.  The wrench, the gains and
        the reference are the stand's, unchanged.
        """
        if self.ramp is None:
            raise RuntimeError("BalanceLaw.arm() has not been called")
        started = None if clock is None else clock()
        self._sweep += 1
        clock_now = None if gait is None else gait.sample(now)
        if clock_now is not None:
            # THE HEIGHT FROM THE FEET THAT ARE DOWN -- see `state.on_stance`.
            state = STATE.on_stance(state, clock_now.contact, self.srb)

        # -- the reference -------------------------------------------------
        command = self.ramp.at(float(now) - self.t0)
        com_cmd = REF.com_command(command, state.R, self.srb)

        # -- stage 3 -------------------------------------------------------
        held = bool(state.imu_stale)
        if held:
            self.imu_held_sweeps += 1
        wrench = CTRL.balance_wrench(state, com_cmd, self.R_des, self.gains,
                                     hold_attitude=held, srb=self.srb)

        # -- stage 4 -------------------------------------------------------
        weight = None if clock_now is None else clock_now.weight
        allocation = ALLOC.allocate(state.r_w, wrench.b_d, mu=self.mu,
                                    contact=weight)

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

        # -- the swing legs ------------------------------------------------
        q_ref = ik_reference(command.h, C.unflat(state.q), self.foot_xy)
        swing_s = p_swing = None
        if clock_now is not None:
            swinging = ~clock_now.contact
            if self.foot_xy_to is not None:
                # TOUCHDOWN: the leg now stands on the step's destination.
                landed = self._swinging & ~swinging
                self.foot_xy[landed] = self.foot_xy_to[landed]
                self._swinging = swinging.copy()
            swing_s = clock_now.swing_s
            p_swing = np.full((C.N_LEGS, 3), np.nan)
            if swinging.any():
                rest = SWING.rest_feet_b(command.h, self.foot_xy)
                land = (rest if self.foot_xy_to is None
                        else SWING.rest_feet_b(command.h, self.foot_xy_to))
                q4 = C.unflat(state.q)
                for i in np.flatnonzero(swinging):
                    p, v = SWING.swing_reference(rest[i], float(swing_s[i]),
                                                 gait.swing_duration,
                                                 height=self.swing_height,
                                                 land_b=land[i])
                    p_swing[i] = p
                    if self.swing == "joint":
                        # z only, abd held: no step destination to go to.
                        # THE ARC STARTS WHERE THE FOOT IS, LATCHED AT LIFTOFF
                        # -- not at `foot_xy` and a height.  The first hardware
                        # run, 2026-09-17, kicked hard: at 30 N*m/rad 10 mm
                        # of foot offset is 5 N*m on the knee, and a foot that
                        # slid, a pitched trunk (2 deg x 215 mm = 7.5 mm) or
                        # a trunk off the command all make that offset.  In
                        # MuJoCo with the runner's one-sweep delay and 40 Hz
                        # velocity filter, 8 mm of z offset demanded 94 N*m;
                        # latched, 1.0 N*m.  The foot lands where it lifted.
                        if not np.isfinite(self._lift_x[i, 0]):
                            self._lift_x[i] = state.x_b[i]
                            self._lift_q[i] = q4[i]
                        qj, qdj = SWING.joint_swing_reference(
                            i, self._lift_x[i], float(swing_s[i]),
                            gait.swing_duration, self._lift_q[i],
                            height=self.swing_height)
                        p_swing[i] = SWING.swing_reference(
                            self._lift_x[i], float(swing_s[i]),
                            gait.swing_duration,
                            height=self.swing_height)[0]
                        tau[3 * i:3 * i + 3] += SWING.joint_swing_torque(
                            state, i, qj, qdj)
                    else:
                        tau[3 * i:3 * i + 3] += SWING.swing_torque(
                            state, i, p, v)
                    # THE TRACKING REFERENCE FOLLOWS A SWINGING LEG, as DOG5's
                    # q_ref did: pinned at the stance IK, the trip would read
                    # a 40 mm apex as a leg gone wrong.
                    q_ref[i] = q4[i]
            self._lift_x[~swinging] = np.nan
            self._lift_q[~swinging] = np.nan
        else:
            self._lift_x[:] = np.nan
            self._lift_q[:] = np.nan

        # -- the joint-space layer (DOG5's JointImpedance) ------------------
        # The joint angles latched on reaching HOLD are the target for every
        # leg, standing or trotting: with the feet planted, fixed joints are
        # a fixed trunk pose -- position AND rpy.  A swinging leg's target is
        # where it IS, so it gets the damper and no spring pulling it down.
        if self.q_hold is not None:
            target = C.unflat(self.q_hold).copy()
            if clock_now is not None:
                off = ~clock_now.contact
                target[off] = C.unflat(state.q)[off]
            tau += (self.kp_joint * (C.flat(target) - state.q)
                    - self.kd_joint * np.asarray(state.qd))

        # -- the trips -----------------------------------------------------
        # THE RESIDUAL TRIP COUNTS ONLY FOUR-FOOT SWEEPS IN A TROT.  A pair of
        # diagonal feet has no moment about its own support line, so on two
        # feet a residual is geometry, not a contact about to go -- in the
        # old rear-tucked fold it was ~0.97 N*m, over the 0.94 limit, every
        # swing.  The trip keeps its meaning where it has one.
        monitor = clock_now is None or clock_now.full_support
        trip = self._trip(state, allocation, C.flat(q_ref),
                          monitor_residual=monitor)
        if trip is None:
            self.tau_peak = max(self.tau_peak, float(np.abs(tau).max()))

        if started is not None:
            self.timing.add(clock() - started)
        return LawOutput(tau=tau, command=command, com_cmd=com_cmd,
                         wrench=wrench, allocation=allocation,
                         q_ref=C.flat(q_ref), imu_held=held, trip=trip,
                         state=state,
                         contact=weight, swing_s=swing_s, p_swing=p_swing)

    def _trip(self, state, allocation, q_ref,
              monitor_residual: bool = True) -> str | None:
        if not np.all(np.isfinite(allocation.f_w)):
            return ("the allocator returned a non-finite force -- the grasp "
                    "map is singular or the state is NaN")
        tilt = self.tilt_from_setpoint_deg(state)
        if tilt > self.tilt_stop_deg:
            return ("tilt %.1f deg from the setpoint, past the %.0f deg stop "
                    "(roll %+.1f, pitch %+.1f; setpoint %+.1f / %+.1f)"
                    % (tilt, self.tilt_stop_deg,
                       np.degrees(state.roll), np.degrees(state.pitch),
                       np.degrees(self.sp_roll + self.att_offset[0]),
                       np.degrees(self.sp_pitch + self.att_offset[1])))
        error = np.abs(state.q - q_ref)
        if (self.track_stop_deg > 0.0
                and np.any(error > np.deg2rad(self.track_stop_deg))):
            from ..hardware_map import JOINT_LABELS
            index = int(np.argmax(error))
            return ("tracking: %s is %.1f deg from the IK at the commanded "
                    "height (limit %.0f)"
                    % (JOINT_LABELS[index], np.rad2deg(error[index]),
                       self.track_stop_deg))
        if not monitor_residual:
            self.residual.streak = 0
            return None
        return self.residual.reason(allocation)

    # -- the exit report -------------------------------------------------
    def report(self) -> str:
        return "\n".join([
            "  law timing      %s" % self.timing.summary(),
            "  peak |tau|      %.2f N*m (before the gate)" % self.tau_peak,
            "  peak residual   %.2f N / %.3f N*m"
            % (self.residual.peak_force, self.residual.peak_moment),
            "  imu held        %d sweeps" % self.imu_held_sweeps,
            "  att setpoint    roll %+.2f  pitch %+.2f deg  (%s)"
            % (np.degrees(self.sp_roll), np.degrees(self.sp_pitch),
               "latched at limp" if self.setpoint_latched
               else "config statics -- NOT latched"),
            "  heading         %s"
            % ("%+.1f deg at the handover, then world x -- yaw since is "
               "drift off it" % np.degrees(self.yaw_offset) if self.armed
               else "never latched -- the run did not reach the rise"),
        ])
