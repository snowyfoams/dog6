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

    A STALE OR MISSING ESTIMATE IS NOT ONE EITHER.  With `est_xy` (every
    trot's `--est-xy`, on by default since 2026-10-02; `hw.fully_trot` flew it
    first) the hold's and the trot's x/y rows read the state estimator; any
    sweep
    whose estimate cannot be used -- none fed, too old, another world frame,
    non-finite, a stale accelerometer behind it, a stale attitude this sweep
    -- gets the x/y rows at zero,
    which is exactly the law every other trot flies, and the refusal is
    counted for the report.  DOG5's per-tick fallback to the flown source.
"""
from __future__ import annotations

import math
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
from . import qp as QP               # noqa: E402
from . import reference as REF       # noqa: E402
from . import state as STATE         # noqa: E402
from . import swing as SWING         # noqa: E402
from . import swing_control as SC    # noqa: E402
from . import torque as TRQ          # noqa: E402

__all__ = ["BalanceLaw", "LawOutput", "Timing", "TrunkEstimate",
           "ik_reference"]


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


@dataclass(frozen=True)
class TrunkEstimate:
    """The state estimator's answer as the law reads it.  WORLD frame.

    Built by whatever runs the filter (`hw.trot_esti.EstimatorFeed`) and
    handed in through `BalanceLaw.feed_estimate`; this package does not
    import the filter and does not care which one it is.  Every field the law
    needs to decide whether to BELIEVE it is here beside the numbers: when it
    was measured, in which world, and how old its accelerometer was.
    """

    #: s, the `now` of the sweep whose measurement the estimate was made
    #: from -- NOT when the arithmetic ran.  The law ages it against its own.
    t: float
    p_w: np.ndarray          # (3,) m, trunk ORIGIN, the run's world
    v_w: np.ndarray          # (3,) m/s, the same point's velocity
    #: rad, the heading the world frame was pinned to (`yaw_offset`).  An
    #: estimate made in another world is refused, never rotated.
    yaw_offset: float
    #: s, the age of the 0x40 packet the filter integrated on that sweep.
    acc_age_s: float = 0.0


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
    #: (12,) `swing.swing_feedforward`'s share of `tau`: zeros unless the law
    #: has `swing_ff` and the leg is swinging.  Logged so a run can say what
    #: the feedforward did, apart from what the PD did.
    tau_ff: np.ndarray | None = None
    #: The state estimator, when one is fed (`BalanceLaw.est_xy`): the
    #: estimate this sweep READ, believed or not, and its age; the CoM x/y
    #: setpoint, None until a hold or a trot latches one; and whether the
    #: x/y rows were driven by it this sweep -- False is the flown law.
    estimate: TrunkEstimate | None = None
    est_age_s: float = float("nan")
    xy_des: np.ndarray | None = None     # (2,) m, CoM, WORLD
    xy_on: bool = False
    #: The walk's reference this sweep (`trajectory.RefSample`), None when
    #: not walking; and each swinging foot's planned touchdown, WORLD x/y,
    #: NaN for a foot that is down.
    ref: object = None
    land_w: np.ndarray | None = None     # (4, 2)


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
    #: The sustained-residual trip, `allocation.ResidualMonitor`.  False turns
    #: it OFF -- `hw.stand --no-limits` -- and keeps the peaks for the report.
    residual_trip: bool = True
    #: The pinned c^b and I^b, `posture.CrouchPose.srb`.  None is the nominal
    #: `config.SRB`.  It reaches BOTH the reference and the measurement from
    #: here, which is what keeps the CoM offset cancelling in the z error.
    srb: "cfg.SrbModel | None" = None
    #: The swing leg's controller, `swing.SWING_MODES`.  "cartesian" is the
    #: impedance every trot flies, the fold's too since 2026-09-28; "joint"
    #: is the same z-only arc through the IK, abd held, joint PD, which the
    #: fold trot flew until then.  See swing.py.
    swing: str = "cartesian"
    #: m, the swing apex above the resting foot.  `config.SWING_HEIGHT` (20 mm
    #: since 2026-10-02, DOG5's 40 before) unless the entry point says
    #: otherwise -- `--swing-height`.  It
    #: is the one number that scales the whole swing demand (speed, torque,
    #: slew) linearly, so it is the first thing to lower for a short swing.
    swing_height: float = cfg.SWING_HEIGHT
    #: The swing's INERTIAL feedforward, `swing.swing_feedforward`: tau +=
    #: M0 J^+ (a_ref - Jdot qd_ref) on each swing leg, open loop, both modes.
    #: OFF unless the entry point says so (`--swing-ff`).  swing.py, "THE
    #: INERTIAL TERM IS BACK", has what it is worth: ~90 % of the torque a
    #: 15 mm / 140 ms arc asks for, which the PD was making out of error.
    swing_ff: bool = False
    #: The feedforward's 3x3 M0; None is `swing.JOINT_INERTIA_M0`, the
    #: inherited armature.  `swing.feedforward_inertia(measured)` builds one
    #: from a DOG6-measured armature (`--ff-armature`); swing.py says why.
    ff_inertia: np.ndarray | None = None
    #: The Cartesian swing PD's (x, y, z) gains, N/m and N s/m; None is
    #: `config.KP_SWING` / `KD_SWING`: the operator's 10/10/400 and 5/5/40
    #: since 2026-10-02, DOG5's 140/140/180 and 8/8/15 before.  WITH
    #: THE FEEDFORWARD ON, Kp IS NOT THE LIFT ANY MORE AND STIFFENING IT
    #: HURTS: a Cartesian PD without Lambda couples a z force into x through
    #: M^-1 J^T -- one leg in its own dynamics, Kp_z 180 -> 800 took the x
    #: drift from 2 to 10 mm.  Kd_z 15 -> 30 (zeta 0.28 -> 0.57) is the one
    #: worth trying: z rms 3.5 -> 2.8 mm for 1 mm of x.  `--kp-swing`,
    #: `--kd-swing`.
    kp_swing: np.ndarray | None = None
    kd_swing: np.ndarray | None = None
    #: The KNEE swing's joint PD (`swing == "knee"`), (abd, pitch, knee),
    #: N*m/rad and N*m*s/rad; None is `config.KP_SWING_KNEE` /
    #: `KD_SWING_KNEE`.  `--kp-swing-knee`, `--kd-swing-knee`.
    kp_swing_knee: np.ndarray | None = None
    kd_swing_knee: np.ndarray | None = None
    #: THE STANCE XY SPRING, 2026-09-25, on request: each PLANTED foot pulled
    #: toward its resting site in the TRUNK frame, x and y only, N/m and
    #: N s/m per leg, on top of the SRB torque and the joint layer.  With the
    #: feet on the floor that is trunk xy stiffness relative to the feet --
    #: what the joint layer gives at 113 N/m per leg per 3 N*m/rad, but here
    #: without the 1250 N/m of z that comes with it and fights the attitude
    #: loop.  z is untouched: height is the SRB law's.  OFF at 0 (the trot
    #: and the sway demos as they were); `--kp-stance-xy`, `--kd-stance-xy`.
    #: Sizing: four feet at 400 N/m are 1600 N/m on 5.9 kg, 2.6 Hz; zeta 0.5
    #: wants ~25 N s/m per leg.
    kp_stance_xy: float = 0.0
    kd_stance_xy: float = 0.0
    #: The joint-space layer, `config.KP_JOINT_HOLD`: live only while
    #: `hold_joints` has latched a pose -- or, with `rise_track`, from the
    #: first torque sweep.  OFF (0 / 0) unless the entry point passes gains
    #: -- the trot entry points do, the plain stand does not.
    kp_joint: float = 0.0
    kd_joint: float = 0.0
    #: The ABDUCTION joints' own gains in that layer once a pose is LATCHED
    #: (HOLD and the trot), 2026-10-02, on request: lock abd, pitch and knee
    #: keep `kp_joint` / `kd_joint`.  None is `kp_joint` / `kd_joint`, the
    #: layer as it was.  The tracked rise (`rise_track`) is not touched: it
    #: keeps the scalar gains on all three joints.  `config.KP_JOINT_HOLD_ABD`.
    kp_joint_abd: float | None = None
    kd_joint_abd: float | None = None
    #: THE SWING ARC LEVELED IN ROLL, 2026-10-02, on request ("把roll融入到
    #: foot trajectory").  Off, the arc and its resting site live in the
    #: trunk frame and tilt with the trunk: rolled 1 deg, the fold stand's
    #: (160 mm) swing foot lands 1.1 mm above the floor on the high side and
    #: aims 1.1 mm into it on the low side.  On, they are taken about the
    #: TRUNK ORIGIN in the trunk frame with the roll removed -- roll from
    #: this run's level, `sp_roll` -- and turned back into the trunk frame
    #: every sweep (`swing.roll_level`), so the foot lands on the floor
    #: whatever the roll.  Pitch untouched.  Cartesian swing only.
    #:
    #: THE STANCE LATCH DOES NOT FOLLOW IT.  `q_hold` stays HOLD's, level, so
    #: a leg landing on a rolled trunk is pulled toward the level stance --
    #: that pull is the joint layer's roll stiffness.  Built first with a
    #: re-latch on every touchdown (the leg's `q_hold` := the IK of the
    #: leveled site): in MuJoCo, `hw.fold_trot`, the layer then held the roll
    #: each stance started at, calm roll rms 0.6 -> 1.5-1.6 deg, and 1 s
    #: after an 8 N side push 2-3 deg against 0.2-0.3.  Removed the same day
    #: on the operator's call.  The arc alone: roll as without it (0.63-0.69).
    swing_roll: bool = False
    #: THE RISE TRACKED, 2026-10-01.  While no pose is latched (the rise, and
    #: a foot step) the joint layer's target is `q_ref` -- the IK at the
    #: commanded height with the feet pinned at `foot_xy`, the posture's own
    #: path -- so the legs are held on it until HOLD latches the measured
    #: pose as before.  OFF by default; `hw.fold_trot` turns it on.
    #:
    #: WHY.  The SRB law commands height and attitude and nothing holds the
    #: trunk's x over the feet (`config.KP_XY`).  In a fore-aft symmetric
    #: stance that costs nothing: every x effect of one leg is cancelled by
    #: its mirror.  In the PARALLEL fold (`posture.FOLD`, every knee behind
    #: its hip) all four legs push the same way as they extend, and the trunk
    #: walks forward through the rise.  The operator saw it on 2026-10-01:
    #: trunk forward, front legs compressed, rear not parallel.  In MuJoCo
    #: with this law, from the fold crouch, no joint friction, no model
    #: error: +95 mm over the 3 s rise, legs at +43/-54 deg from vertical
    #: against the designed +-16; the same for a 6 or 12 s rise; the same
    #: with the true CoM every sweep; the mirrored fold and NOMINAL: 0.3 mm.
    #: With this on, under knee contact, the 1 s cap ramp, 0.1 N*m friction,
    #: a 40 Hz qd filter and two sweeps of latency: +10 mm, legs +17.7/-17.6,
    #: pitch 0.0, at most 1.1 N*m from the layer.  The Cartesian stance xy
    #: spring held as well in that sim and SHOOK on the robot the same day
    #: (unexplained); this is the joint-space PD every trot already flies.
    #:
    #: The reference is a LEVEL trunk, so with the dynamic setpoint latched
    #: off level the layer and the attitude loop disagree by that tilt; the
    #: fold flies the fixed setpoint.  The tracking trip's `track_stop_deg`
    #: still reads the same `q_ref`.
    rise_track: bool = False
    #: THE SLANTED RISE, 2026-10-01: (4, 2) HIP-frame foot sites the STAND
    #: holds, when they are not the crouch's.  None (every entry point but
    #: `hw.fold_trot`) is a straight rise on the crouch's feet.  Given, the
    #: IK reference's sites slide from `foot_xy` (the crouch's) to these as
    #: the commanded height passes from `slide_from_h` to `h_lift` -- a
    #: smoothstep in HEIGHT, not in time -- and the tracked joint layer
    #: carries the trunk OVER the planted feet: no foot leaves the floor.
    #: Needs `rise_track`.
    #:
    #: WHY A SLIDE AND NOT A STEP.  The fold's stand wants the thigh swept
    #: and the shin upright (the operator, 2026-10-01: "the big leg is so
    #: straight"), which puts the feet BEHIND the pitch hinges; the crouch
    #: cannot have them there -- the front knee motor folds up into the
    #: trunk at every crouch height below ~150 mm -- so the feet differ
    #: between the two.  Moving them at height was tried first, in MuJoCo:
    #: a diagonal-pair step tips the trunk about the supporting diagonal
    #: (this stance's CoM is 19 mm off both), the rear swing foot never
    #: leaves the floor (5-17 N on it mid-swing) and lands 70-86 mm short;
    #: a one-foot-at-a-time step fares no better (three-foot margin -6 mm
    #: on the second swing for every order) and would need a crawl with
    #: body shifts.  Sliding the sites through the rise moves the trunk 85
    #: mm forward over four planted feet: pitch within 0.2 deg, the stand
    #: within 0.4 mm of its sites, the trot there stays in place (+1 mm in
    #: 3 s against -41 for the old stance), the park slides the feet 18 mm.
    #:
    #: WHY FROM A HEIGHT AND NOT FROM THE START.  The front knee has to stay
    #: BELOW the trunk's underside the whole way: for the fold that bounds
    #: the sites from below at each height (a knee axis 67 mm under the hip
    #: axis), and sliding from 110 mm keeps it 3 mm clear in the sim, where
    #: a slide over the whole rise puts the knee 5 mm into the trunk at
    #: 100 mm.  `posture.py` has the clearance arithmetic.
    stand_xy: np.ndarray | None = None
    #: m, the commanded height the slide starts at; None is the crouch's h0
    #: (the whole rise).  `hw.fold_trot` passes 0.110.
    slide_from_h: float | None = None
    #: THE STATE ESTIMATOR IN THE LOOP, 2026-10-01 -- `hw.fully_trot`; the
    #: HOLD too since 2026-10-02, on request.  While the sequence says HOLD or
    #: TROT (`update(..., xy_hold=True)`), the x and y rows of the PD read the
    #: estimate fed by `feed_estimate` instead of the zeros every other law
    #: has: the CoM's world x/y against the CoM x/y LATCHED on the first sweep
    #: of that hold or trot that has a usable estimate, and its rate against
    #: zero, on `gains.kp_pos[:2]` / `kd_pos[:2]` (`config.KP_XY_EST`).  That
    #: is station keeping: the stand and the trot IN PLACE, by the filter's
    #: account.  Each HOLD and each trot latches its own (`release_xy`): a
    #: hold does not pull back the trot before it, nor the W step's walk.
    #:
    #: WHAT IT DOES NOT TOUCH.  Height stays the legs' (`state.on_stance`) and
    #: attitude the IMU's: both are measured, and replacing a measured signal
    #: with a filtered one needs a reason (DOG5's EkfFeedback rule).  x/y have
    #: no other source.  The rise and the W step never read it.
    #:
    #: False (every other entry point): an estimate fed in is kept for the log
    #: and moves nothing.
    est_xy: bool = False
    #: s, the oldest estimate (`TrunkEstimate.t` against `now`) the x/y rows
    #: may use, and the oldest accelerometer packet behind it.
    est_max_age_s: float = cfg.EST_MAX_AGE_S
    est_acc_max_age_s: float = cfg.EST_ACC_MAX_AGE_S
    #: m/s^2, the authority clamp on the x/y rows (`controller.balance_wrench`).
    xy_acc_max: float = cfg.XY_ACC_MAX
    #: THE FORCE ALLOCATOR, 2026-10-04.  "wls" is `allocation.allocate`, the
    #: least squares every entry point flew until now; "qp" is
    #: `qp.QpAllocator`, the cone and the contact ramp's box inside the
    #: problem instead of clipped on afterwards (`--alloc`; `hw.fold_walk`'s
    #: default).  qp.py says where the two differ and where they cannot.
    alloc: str = "wls"
    #: WALKING, 2026-10-04: a `walk.WalkPlan`, or None -- every entry point
    #: but `hw.fold_walk`, for which nothing below changes.  Attached, it
    #: engages on the first trot sweep and from then on owns the x/y rows'
    #: reference, the heading, the joint layer's targets and the swing
    #: (walk.py: one reference, every target taken from it).
    walk: object | None = None

    def __post_init__(self) -> None:
        if self.swing not in SWING.SWING_MODES:
            raise ValueError("swing %r: one of %s"
                             % (self.swing, ", ".join(SWING.SWING_MODES)))
        if self.alloc not in ("wls", "qp"):
            raise ValueError("alloc %r: 'wls' or 'qp'" % (self.alloc,))
        if self.walk is not None and self.swing != "cartesian":
            raise ValueError("walking places the foot in x/y; the %s swing "
                             "lifts in z alone" % self.swing)
        #: The QP allocator (qp.py), when `alloc` asks for it.
        self.qp = QP.QpAllocator(mu=self.mu) if self.alloc == "qp" else None
        if self.swing_roll and self.swing != "cartesian":
            raise ValueError("swing_roll levels the Cartesian arc; the joint "
                             "swing latches its own at liftoff")
        # OWN COPY: the foot step moves it leg by leg, and it usually arrives
        # as a posture's array.
        if self.foot_xy is not None:
            self.foot_xy = np.array(self.foot_xy, dtype=float)
        if self.stand_xy is not None:
            if self.foot_xy is None:
                self.foot_xy = np.array(ST.FOOT_XY, dtype=float)
            if not self.rise_track:
                raise ValueError("stand_xy needs rise_track: only the tracked "
                                 "joint layer carries the trunk over the feet")
            self.stand_xy = np.array(self.stand_xy, dtype=float).reshape(
                C.N_LEGS, 2)
        #: The crouch's sites, kept for the slide; `foot_xy` is what moves.
        self._crouch_xy = (None if self.foot_xy is None
                           else self.foot_xy.copy())
        #: The foot step's destination, (4, 2) HIP frame, or None.  See
        #: `begin_step`.
        self.foot_xy_to: np.ndarray | None = None
        self._swinging = np.zeros(C.N_LEGS, dtype=bool)
        #: The joint swing's LIFTOFF LATCH, (4, 3) each, NaN while the leg is
        #: down: the measured trunk-frame foot and joints on the first
        #: swinging sweep.  See the swing block in `update`.
        self._lift_x = np.full((C.N_LEGS, 3), np.nan)
        self._lift_q = np.full((C.N_LEGS, 3), np.nan)
        #: The knee swing's signed amplitude per leg, rad, planned at liftoff
        #: (`swing.knee_swing_amplitude`), NaN while the leg is down; and the
        #: swings it refused (the leg then holds where it lifted).
        self._knee_amp = np.full(C.N_LEGS, np.nan)
        self.knee_refused = 0
        self.knee_refusal = ""
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
        self.residual = (ALLOC.ResidualMonitor() if self.residual_trip else
                         ALLOC.ResidualMonitor(force_n=np.inf,
                                               moment_nm=np.inf))
        self.timing = Timing()
        self.gravity = np.zeros((C.N_LEGS, 3))
        self.tau_peak = 0.0
        self.imu_held_sweeps = 0
        self._sweep = 0
        #: The last `TrunkEstimate` fed in, or None.  See `est_xy`.
        self.estimate: TrunkEstimate | None = None
        #: (2,) m, the CoM x/y the trot holds, WORLD; None outside a trot and
        #: until its first usable estimate.  `release_xy` clears it.
        self.xy_des: np.ndarray | None = None
        #: The x/y loop's own account, for the exit report: trot sweeps it
        #: drove, trot sweeps it refused and the last reason, and its peaks.
        self.xy_sweeps = 0
        self.xy_refused = 0
        self.xy_refusal: str | None = None
        self.xy_err_peak = 0.0
        self.xy_acc_peak = 0.0

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

        REFUSED with the joint swing: that one lifts in z and nothing else;
        and with the knee swing, which lands where it lifted.
        """
        if self.swing in ("joint", "knee"):
            raise ValueError("the %s swing lands where it lifted; a foot "
                             "step needs swing='cartesian'" % self.swing)
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

    # -- the slanted rise -------------------------------------------------
    @property
    def home_xy(self) -> np.ndarray | None:
        """The stand's sites -- where the feet are 'home' for the park:
        `stand_xy` when the rise slides, else the crouch's `foot_xy`."""
        return self._crouch_xy if self.stand_xy is None else self.stand_xy

    def _slide_feet(self, h_cmd: float) -> None:
        """`foot_xy` := the crouch's sites carried toward `stand_xy` by a
        smoothstep in the commanded height, EXACTLY `stand_xy` once the ramp
        has arrived.  Called only while no pose is latched and no step is
        running -- the rise."""
        h_from = self.h0 if self.slide_from_h is None else float(self.slide_from_h)
        span = float(self.h_lift) - h_from
        # ARRIVED is a test on the HEIGHT, not on the fraction: the fraction
        # rounds to 0.999... at the top and the sites would then miss the
        # stand's by an ulp, which `feet_home`'s exact compare would read as
        # "away" and ENTER would step the feet to where they already are.
        if float(h_cmd) >= float(self.h_lift) - 1e-9 or span <= 1e-9:
            self.foot_xy[:] = self.stand_xy
        else:
            s = float(np.clip((float(h_cmd) - h_from) / span, 0.0, 1.0))
            self.foot_xy[:] = (self._crouch_xy
                               + ST.smoothstep(s) * (self.stand_xy - self._crouch_xy))

    def hold_reference(self, q) -> np.ndarray:
        """(12,) the IK at `h_lift` with the feet on the stand's sites --
        what `hold_joints` should latch when the rise was TRACKED, so the
        layer's target and the swing's resting sites are one stance by
        construction.  Latching the MEASURED pose instead froze whatever lag
        the trunk had when the ramp arrived (12-22 mm in the sim) and the
        trot then walked on the disagreement.  `q` seeds the IK branch."""
        if self.stand_xy is not None:
            self.foot_xy[:] = self.stand_xy
        return C.flat(ik_reference(self.h_lift, C.unflat(q), self.foot_xy))

    # -- the trot's joint-space layer ------------------------------------
    def hold_joints(self, q) -> None:
        """Latch `q` (12,) as the posture the joint layer holds.  `sequence`
        calls it with the measured angles on the sweep the robot REACHES
        HOLD, and keeps it through the hold and the trot."""
        self.q_hold = np.array(q, dtype=float).reshape(C.N_JOINTS)

    def release_joints(self) -> None:
        self.q_hold = None

    def _roll_from_level(self, state) -> tuple[float, float]:
        """(roll from this run's level, its rate), rad and rad/s, from R and
        the gyro: ZYX roll, the Euler rate from the body rates."""
        R = state.R
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        w = state.omega_b
        rate = w[0] + math.tan(pitch) * (math.sin(roll) * w[1]
                                         + math.cos(roll) * w[2])
        return roll - self.sp_roll, rate

    # -- the state estimator (est_xy, hw.fully_trot) ----------------------
    def feed_estimate(self, estimate: TrunkEstimate | None) -> None:
        """The filter's latest answer, every sweep it has one.  Read by the
        NEXT `update` -- the filter runs after slot 0 has acted, so what the
        law reads is one sweep old, and `est_max_age_s` is what bounds it."""
        if estimate is not None:
            self.estimate = estimate

    def release_xy(self) -> None:
        """Forget the x/y setpoint; the next hold or trot sweep with a usable
        estimate latches a fresh one.  `sequence` calls it as a trot starts,
        as it ends and as a W step starts, so neither a trot nor a hold ever
        inherits the position of the phase before it."""
        self.xy_des = None

    def _estimate_ok(self, now: float, state):
        """(the fed `TrunkEstimate`, None) if it may be used this sweep, else
        (None, why not).  The x/y rows and the walk ask the same question."""
        est = self.estimate
        if est is None:
            return None, "no estimate fed yet"
        age = float(now) - float(est.t)
        if not 0.0 <= age <= self.est_max_age_s:
            return None, ("estimate %.0f ms old (limit %.0f)"
                          % (1e3 * age, 1e3 * self.est_max_age_s))
        if float(est.yaw_offset) != float(self.yaw_offset):
            return None, ("estimate in the world at heading %+.1f deg, the "
                          "law's is %+.1f" % (np.degrees(est.yaw_offset),
                                             np.degrees(self.yaw_offset)))
        if not float(est.acc_age_s) <= self.est_acc_max_age_s:
            return None, ("its accelerometer packet was %.0f ms old (limit "
                          "%.0f)" % (1e3 * est.acc_age_s,
                                     1e3 * self.est_acc_max_age_s))
        if state.imu_stale:
            # The attitude half is frozen this sweep (`imu_held`), and the
            # filter's x/y are legs rotated by that same stale R.
            return None, ("the IMU attitude is %.0f ms old -- the R the "
                          "filter's x/y are rotated by" % (1e3 * state.imu_age_s))
        if not (math.isfinite(float(est.p_w[0])) and math.isfinite(float(est.p_w[1]))
                and math.isfinite(float(est.v_w[0]))
                and math.isfinite(float(est.v_w[1]))):
            return None, "the estimate is not finite"
        return est, None

    def _xy_feedback(self, now: float, state, ref=None, est=None,
                     refusal=None):
        """(`controller.XyFeedback` or None, why not or None) for this sweep.

        THE CoM, NOT THE TRUNK ORIGIN.  The filter tracks the origin; the PD
        regulates the CoM, as the z row does, and the two differ by R c^b --
        whose rate is omega x R c^b, the trunk's own sway, which would
        otherwise reach the x/y damper as a velocity the CoM does not have.
        Converted with THIS sweep's R and omega: the estimate is 4 ms older
        than both, which at a trot's few deg/s is a hundredth of a mm.

        `ref` is the walk's reference (`trajectory.RefSample`) or None.  Given,
        the setpoint is the REFERENCE's CoM x/y and its rate the reference's,
        every sweep -- no latch: the walk owns the setpoint (walk.py).  `est`
        and `refusal` are `_estimate_ok`'s answer if the caller already has
        it; None asks again.
        """
        if est is None and refusal is None:
            est, refusal = self._estimate_ok(now, state)
        if est is None:
            return None, refusal
        # SCALARS, AND THE CROSS PRODUCT WRITTEN OUT -- `state.on_stance`'s
        # reason: this is slot-0 arithmetic, and `np.cross` plus a numpy call
        # per element cost the Pi 83 us a sweep where this costs a few.
        px, py = float(est.p_w[0]), float(est.p_w[1])
        vx, vy = float(est.v_w[0]), float(est.v_w[1])
        c = state.R @ self.srb.com_body                  # R c^b, WORLD
        w = state.omega_w
        p_c = np.array([px + c[0], py + c[1]])
        v_c = np.array([vx + w[1] * c[2] - w[2] * c[1],  # (omega x R c^b)_x
                        vy + w[2] * c[0] - w[0] * c[2]])  # (omega x R c^b)_y
        if ref is not None:
            self.xy_des = np.asarray(ref.p, dtype=float).copy()
            return CTRL.XyFeedback(p_error=self.xy_des - p_c, v=v_c,
                                   v_des=np.asarray(ref.v, dtype=float)), None
        if self.xy_des is None:
            # LATCHED, ON THE FIRST USABLE SWEEP OF THE HOLD OR THE TROT: zero
            # error on the sweep the loop closes, so closing it cannot step
            # the wrench.
            self.xy_des = p_c
        return CTRL.XyFeedback(p_error=self.xy_des - p_c, v=v_c), None

    # -- the sweep -------------------------------------------------------
    def update(self, now: float, state, clock=None, gait=None,
               xy_hold: bool = False) -> LawOutput:
        """Stages 1-5 for one sweep.  `state` is a `state.BodyState`.

        `gait` is a `gait.TrotGait` while the robot trots, None while it
        stands.  It changes three things and nothing else: the allocator is
        given the clock's contact weights, a swinging leg gets the swing
        impedance on top of its own weight, and the residual trip only counts
        sweeps with all four feet at full weight.  The wrench, the gains and
        the reference are the stand's, unchanged.

        `xy_hold` is the sequence saying THIS IS THE HOLD OR THE TROT.  With
        `est_xy` it puts the state estimator on the x/y rows (`est_xy` says
        how); without `est_xy` it changes nothing.
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
        if (self.stand_xy is not None and self.q_hold is None
                and self.foot_xy_to is None):
            self._slide_feet(command.h)          # the slanted rise
        com_cmd = REF.com_command(command, state.R, self.srb)

        # -- the walk's reference (walk.py) ---------------------------------
        # None until the first trot sweep of a law that walks: the stand, the
        # rise and the first HOLD are the flown ones.  From then on the
        # heading, the x/y rows, the joint layer and the swing all take their
        # target from this one sample.
        ref = est = est_refusal = None
        omega_des = None
        if self.walk is not None or self.est_xy:
            est, est_refusal = self._estimate_ok(now, state)
        if self.walk is not None:
            ref = self.walk.step(now, self, state, gait, clock_now, est)
        if ref is not None:
            self.att_offset[2] = float(ref.yaw)
            self.R_des = CTRL.latched_attitude(
                self.sp_roll + self.att_offset[0],
                self.sp_pitch + self.att_offset[1],
                self.sp_yaw + float(ref.yaw))
            omega_des = np.array([0.0, 0.0, float(ref.yaw_rate)])

        # -- stage 3 -------------------------------------------------------
        held = bool(state.imu_stale)
        if held:
            self.imu_held_sweeps += 1
        # THE ESTIMATOR'S x/y, IN THE HOLD AND THE TROT ONLY -- `est_xy`.  A
        # sweep whose estimate is unusable gets None: the x/y rows are zero,
        # the flown law.
        xy = None
        if self.est_xy and xy_hold:
            xy, refusal = self._xy_feedback(now, state, ref, est, est_refusal)
            if xy is None:
                self.xy_refused += 1
                self.xy_refusal = refusal
            else:
                self.xy_sweeps += 1
                self.xy_err_peak = max(self.xy_err_peak, math.hypot(
                    float(xy.p_error[0]), float(xy.p_error[1])))
        # The reference's own acceleration rides on the x/y rows only while
        # they are closed: with the estimate refused they are zero, the flown
        # law, and an open-loop push would be neither.
        wrench = CTRL.balance_wrench(state, com_cmd, self.R_des, self.gains,
                                     omega_des=omega_des,
                                     hold_attitude=held, srb=self.srb,
                                     xy=xy, xy_acc_max=self.xy_acc_max,
                                     acc_ff=(None if ref is None or xy is None
                                             else ref.a))
        if xy is not None:
            self.xy_acc_peak = max(self.xy_acc_peak, math.hypot(
                float(wrench.acc_lin[0]), float(wrench.acc_lin[1])))

        # -- stage 4 -------------------------------------------------------
        weight = None if clock_now is None else clock_now.weight
        if self.qp is not None:
            allocation = self.qp.allocate(state.r_w, wrench.b_d,
                                          contact=weight)
        else:
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
        swing_s = p_swing = land_w = None
        tau_ff = np.zeros(C.N_JOINTS)
        if ref is not None:
            # TOUCHDOWNS: a landed foot is anchored where it landed, and every
            # foot that is down forgets its swing plan.
            self.walk.contacts(self, state, ref,
                               np.ones(C.N_LEGS, dtype=bool) if clock_now is None
                               else clock_now.contact, est)
        if clock_now is not None and ref is not None:
            # WALKING: the foothold and arc of footstep.py, the swing law of
            # swing_control.py (walk.swing_law).  The tracking reference
            # follows a swinging leg, as below.
            swinging = ~clock_now.contact
            swing_s = clock_now.swing_s
            p_swing = np.full((C.N_LEGS, 3), np.nan)
            land_w = np.full((C.N_LEGS, 2), np.nan)
            q4 = C.unflat(state.q)
            for i in np.flatnonzero(swinging):
                sref = self.walk.swing_ref(int(i), float(swing_s[i]), self,
                                           state, ref, est)
                p_swing[i] = sref.p
                land_w[i] = sref.land_w
                if self.walk.swing_law == "osc":
                    t_sw, t_ff = SC.swing_osc_torque(
                        state, int(i), sref, wn=self.walk.wn_swing,
                        zeta=self.walk.zeta_swing,
                        inertia=self.walk.ff_inertia)
                else:
                    t_sw, t_ff = SC.swing_control_torque(
                        state, int(i), sref, kp=self.walk.kp_swing,
                        kd=self.walk.kd_swing, ff=self.walk.swing_ff,
                        inertia=self.walk.ff_inertia)
                tau[3 * i:3 * i + 3] += t_sw
                tau_ff[3 * i:3 * i + 3] = t_ff
                q_ref[i] = q4[i]
        elif clock_now is not None:
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
                level = (self._roll_from_level(state) if self.swing_roll
                         else None)
                for i in np.flatnonzero(swinging):
                    p, v, a = SWING.swing_reference_pva(
                        rest[i], float(swing_s[i]), gait.swing_duration,
                        height=self.swing_height, land_b=land[i])
                    if level is not None:            # `swing_roll`
                        p, v, a = SWING.roll_level(p, v, a, *level)
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
                        p_swing[i], v, a = SWING.swing_reference_pva(
                            self._lift_x[i], float(swing_s[i]),
                            gait.swing_duration, height=self.swing_height)
                        tau[3 * i:3 * i + 3] += SWING.joint_swing_torque(
                            state, i, qj, qdj)
                    elif self.swing == "knee":
                        # THE SHIN ALONE, IN THE KNEE FRAME -- swing.py, "THE
                        # KNEE SWING".  Latched at liftoff as the joint swing
                        # is: abd and pitch held there, the knee on its bump,
                        # signed toward the CoM, the foot back where it lifted.
                        if not np.isfinite(self._lift_x[i, 0]):
                            self._lift_x[i] = state.x_b[i]
                            self._lift_q[i] = q4[i]
                            try:
                                self._knee_amp[i] = SWING.knee_swing_amplitude(
                                    i, q4[i], self.swing_height)
                            except ValueError as refusal:
                                self._knee_amp[i] = 0.0  # no lift: held
                                self.knee_refused += 1
                                self.knee_refusal = str(refusal)
                        qj, qdj, qddj = SWING.knee_swing_reference(
                            self._lift_q[i], float(self._knee_amp[i]),
                            float(swing_s[i]), gait.swing_duration)
                        p_swing[i] = HK.foot_position(i, qj)
                        tau[3 * i:3 * i + 3] += SWING.joint_swing_torque(
                            state, i, qj, qdj,
                            kp=(cfg.KP_SWING_KNEE if self.kp_swing_knee is None
                                else self.kp_swing_knee),
                            kd=(cfg.KD_SWING_KNEE if self.kd_swing_knee is None
                                else self.kd_swing_knee))
                    else:
                        tau[3 * i:3 * i + 3] += SWING.swing_torque(
                            state, i, p, v, kp=self.kp_swing, kd=self.kd_swing)
                    if self.swing_ff and self.swing == "knee":
                        # M0 qdd_ref: the bump's own acceleration, knee only.
                        ff = (SWING.JOINT_INERTIA_M0 if self.ff_inertia is None
                              else self.ff_inertia) @ qddj
                        tau[3 * i:3 * i + 3] += ff
                        tau_ff[3 * i:3 * i + 3] = ff
                    elif self.swing_ff:
                        # THE ARC'S OWN INERTIAL TORQUE, OPEN LOOP -- swing.py,
                        # "THE INERTIAL TERM IS BACK".  At the MEASURED q: the
                        # reference's is a 191 us IK away in the Cartesian
                        # mode, and the two differ by the tracking error, which
                        # is what the PD beside it is for.  `v` and `a` are the
                        # arc this leg is actually on -- the latched one in the
                        # joint mode -- and Jdot qd_ref inside is what keeps
                        # the foot on the arc's LINE, not just its height.
                        ff = SWING.swing_feedforward(i, q4[i], v, a,
                                                     jac=state.jac[i],
                                                     inertia=self.ff_inertia)
                        tau[3 * i:3 * i + 3] += ff
                        tau_ff[3 * i:3 * i + 3] = ff
                    # THE TRACKING REFERENCE FOLLOWS A SWINGING LEG, as DOG5's
                    # q_ref did: pinned at the stance IK, the trip would read
                    # a 40 mm apex as a leg gone wrong.
                    q_ref[i] = q4[i]
            self._lift_x[~swinging] = np.nan
            self._lift_q[~swinging] = np.nan
            self._knee_amp[~swinging] = np.nan
        else:
            self._lift_x[:] = np.nan
            self._lift_q[:] = np.nan
            self._knee_amp[:] = np.nan

        # -- the stance xy spring ------------------------------------------
        if self.kp_stance_xy > 0.0 or self.kd_stance_xy > 0.0:
            rest = SWING.rest_feet_b(command.h, self.foot_xy)
            planted = (np.ones(C.N_LEGS, dtype=bool) if clock_now is None
                       else clock_now.contact)
            qd4 = C.unflat(state.qd)
            for i in np.flatnonzero(planted):
                jac = state.jac[i]
                e = rest[i] - state.x_b[i]
                v = jac @ qd4[i]
                f = np.array([self.kp_stance_xy * e[0] - self.kd_stance_xy * v[0],
                              self.kp_stance_xy * e[1] - self.kd_stance_xy * v[1],
                              0.0])
                tau[3 * i:3 * i + 3] += jac.T @ f

        # -- the joint-space layer (DOG5's JointImpedance) ------------------
        # The joint angles latched on reaching HOLD are the target for every
        # PLANTED leg, standing or trotting: with the feet planted, fixed
        # joints are a fixed trunk pose -- position AND rpy.
        #   A SWINGING LEG GETS NONE OF IT, 2026-10-01, on request.  Until
        # then its spring target followed the leg and the damper stayed:
        # -Kd qd against an arc that runs the knee at 7-13 rad/s, 83 N s/m at
        # the foot in z at 0.2 -- twice the swing PD's own Kd_z, all of it
        # opposing the lift, and it halved a 20 mm apex in the one-leg model
        # (doc/dog6_swing_leg.tex).  cMPC damps stance legs only, for the
        # same reason (`sim.cmpc.controller`); the swing law is complete
        # without it.  The bench never had it, so a swing tuned hung up is
        # now the swing the trot flies.
        if self.q_hold is not None:
            kp = np.full(C.N_JOINTS, self.kp_joint)
            kd = np.full(C.N_JOINTS, self.kd_joint)
            if self.kp_joint_abd is not None:        # abd, every leg
                kp[0::3] = self.kp_joint_abd
            if self.kd_joint_abd is not None:
                kd[0::3] = self.kd_joint_abd
            if ref is not None:
                # WALKING: the target is where the reference trunk sees each
                # planted foot's anchor, and its rate the anchor's motion in
                # that trunk -- so the damper drags only the error, not the
                # walk (walk.py).  Equal to q_hold on the engaging sweep.
                q_t, qd_t = self.walk.hold_targets(
                    self, state, ref,
                    None if clock_now is None else clock_now.contact)
                hold = C.unflat(kp * (C.flat(q_t) - state.q)
                                + kd * (C.flat(qd_t) - np.asarray(state.qd)))
            else:
                hold = C.unflat(kp * (self.q_hold - state.q)
                                - kd * np.asarray(state.qd))
            if clock_now is not None:
                hold[~clock_now.contact] = 0.0
            tau += C.flat(hold)
        elif self.rise_track and (self.kp_joint > 0.0 or self.kd_joint > 0.0):
            # NO POSE LATCHED YET -- the rise, or a foot step: the target is
            # the IK reference, the posture's own path at the commanded
            # height (`rise_track` says why).  A stepping leg gets none of
            # it, as above.
            hold = C.unflat(self.kp_joint * (C.flat(q_ref) - state.q)
                            - self.kd_joint * np.asarray(state.qd))
            if clock_now is not None:
                hold[~clock_now.contact] = 0.0
            tau += C.flat(hold)

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
        est = self.estimate
        return LawOutput(tau=tau, command=command, com_cmd=com_cmd,
                         wrench=wrench, allocation=allocation,
                         q_ref=C.flat(q_ref), imu_held=held, trip=trip,
                         state=state,
                         contact=weight, swing_s=swing_s, p_swing=p_swing,
                         tau_ff=tau_ff, estimate=est,
                         est_age_s=(float("nan") if est is None
                                    else float(now) - float(est.t)),
                         xy_des=(None if self.xy_des is None
                                 else self.xy_des.copy()),
                         xy_on=xy is not None, ref=ref, land_w=land_w)

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
        ] + ([] if not self.est_xy else [
            "  x/y on the estimator  %d hold/trot sweeps driven, %d refused (flown "
            "law those sweeps)%s" % (self.xy_sweeps, self.xy_refused,
                                     "" if self.xy_refusal is None
                                     else "; last refusal: " + self.xy_refusal),
            "                  peak |CoM xy error| %.1f mm, peak |a_xy| %.2f "
            "m/s^2 (%.1f N; clamp %.1f m/s^2)"
            % (1e3 * self.xy_err_peak, self.xy_acc_peak,
               cfg.MASS * self.xy_acc_peak, self.xy_acc_max),
        ]) + ([] if self.qp is None else ["  " + self.qp.report()])
            + ([] if self.walk is None else [self.walk.report()]))
