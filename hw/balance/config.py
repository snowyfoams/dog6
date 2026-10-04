"""Every number the balance controller has.  Nothing else in `hw.balance`
holds one.

    python -m hw.balance.config          print the lot, with its provenance

Four kinds of number live here and they are NOT interchangeable, so each one
is tagged:

    [CAD]        read out of `sim.params`, which `sim.selftest` gates against
                 model/dog6.xml.  True before the robot is powered.
    [DERIVED]    computed here from [CAD] numbers at import.  Not a choice.
    [UNTUNED]    a control gain with no hardware provenance at all.  Every
                 gain below is one of these: the 2026-09-15 runs logged
                 nothing, so there is no number to inherit.
    [INHERITED]  DOG5's, or the cMPC stack's, carried across with a reason.

WHY THE CoM IS A CONSTANT HERE, AND WHAT THAT COSTS
    `sim.kinematics.body_inertia(q)` gives the whole-robot CoM at a POSE, and
    on DOG6 it moves: the legs are 60 % of the mass and they unfold downward
    as the trunk rises, so c^b_z runs from +21.9 mm at the crouch to -14.8 mm
    at the lift height.  37 mm over a 115 mm ramp.

    THIS FILE PINS IT AT THE LIFT POSE AND USES THAT ONE VALUE EVERYWHERE.
    Three reasons, and the third is the one that matters:

      1. cost.  `body_inertia` walks four chains and accumulates fifteen
         parallel-axis terms.  It is the most expensive thing the law could
         put in a 333 us slot, and it is the one term that would then have to
         be sub-rated -- which means moment arms and an inertia at a different
         age from the Jacobians that produced them.
      2. the hold is where the robot lives.  The ramp is 3 s; the hold is
         indefinite, and it is where a balance law earns its keep.  Pinning
         the model at the pose it holds makes it exact where it matters and
         approximate only where it is transient.
      3. IT CANCELS.  The CoM offset enters twice -- once in the measured
         p_c,z and once in the commanded p_c,z,d, both as the same
         (R c^b)_z -- so with ONE constant c^b the PD's z error is EXACTLY
         the trunk-height error, and the chain-rule coefficient that a
         pose-dependent c^b forces into the velocity reference (0.677,
         measured over this ramp) is identically 1.  A pose-dependent c^b used
         in only one of the two places is a 48 % velocity-reference error; a
         constant one used in both is no error at all, just a different
         definition of the regulated point.

    WHAT IT COSTS is the moment arms.  r_i^w = R (x_i^b - c^b) is then taken
    about a point up to 37 mm from the true CoM early in the ramp: a bias in
    the moment row that looks exactly like a small constant pitch offset.
    Against the 237 mm fore-aft lever that is 15 % at the crouch, falling to
    zero by the top.  IF A LOG SHOWS A RESIDUAL TILT THAT SHRINKS AS THE ROBOT
    RISES, THIS IS THE FIRST THING TO UNPIN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/config.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

# ===========================================================================
# the three heights, and which one a number is in
# ===========================================================================
#: Floor to trunk BOTTOM, the number a ruler reads against the bench.  EVERY
#: height this package calls `h` is in this frame.  [DERIVED]
#:
#: It is NOT the frame `sim.stand` carries.  `CROUCH_HEIGHT`, `LIFT_HEIGHT`,
#: `height_from_fk` and `pose_for_height` are all heights of the TRUNK ORIGIN
#: -- the abduction-axis plane, the mid-height of the trunk box, a plane
#: nothing physical rests on.  The two differ by a constant:
#:
#:      h = z_origin - TRUNK_BOTTOM_OFFSET
#:
#: DOG5 paid for this one: its runner printed 191 mm where a ruler read ~160,
#: and the 38 mm was this frame and nothing else.  The operator and the code
#: disagree by a known constant ON PURPOSE, and this is where that is written
#: down.  `params.HOME_HEIGHT` and `TRUNK_BOX_HALF_z` are the same number
#: because the crouch is DEFINED by the box being down.
TRUNK_BOTTOM_OFFSET = float(P.TRUNK_BOX_HALF[2])        # 0.03501 m   [CAD]

#: The crouch and the lift, in h.  The crouch is 0 by construction.
H_CROUCH = float(ST.CROUCH_HEIGHT - TRUNK_BOTTOM_OFFSET)        # 0.000 m
H_LIFT = float(ST.LIFT_HEIGHT - TRUNK_BOTTOM_OFFSET)            # 0.115 m
#: Where the CAD is drawn, for scale.  The lift stops short of it on purpose.
H_CAD_STAND = float(P.STAND_HEIGHT - TRUNK_BOTTOM_OFFSET)       # 0.1575 m

#: The ramp duration.  `sim.stand`'s, so the two sequences are the same
#: trajectory and only the LAW differs -- which is the whole point of keeping
#: the old one as an A/B.  T is the only knob the reference has, and it sets
#: both the peak velocity and the peak acceleration.  [INHERITED sim.stand]
T_RISE = float(ST.RAMP_LIFT)                                    # 3.0 s

#: Foot xy, pinned in each leg's own HIP frame for the whole phase.
#: `sim.stand`'s, read off Q_CROUCH: a vertical shin puts the foot 80.9 mm out
#: from the hip, which in the TRUNK frame is +-237 mm fore and aft and +-65 mm
#: left and right.  Constant through the stand, so the grasp map's
#: conditioning is constant too.  [CAD, via sim.stand]
FOOT_XY = np.array(ST.FOOT_XY, dtype=float)

#: m, THE STANCE'S YAW LEVER: |foot xy| from the trunk origin, all four equal
#: at 246 mm.  A yaw moment has no normal-force component to be made out of --
#: fz is straight down and its moment about z is zero -- so it is tangential
#: friction across this arm and nothing else, which is what bounds `KP_YAW`.
#: [DERIVED from FOOT_XY + HIP_OFFSET]
FOOT_RADIUS_XY = float(np.linalg.norm(
    np.asarray(P.HIP_OFFSET)[:, :2] + FOOT_XY, axis=1).mean())


# ===========================================================================
# the single-rigid-body model -- FIXED, see the module docstring
# ===========================================================================
#: The pose the model is pinned at: feet at FOOT_XY, trunk origin at
#: LIFT_HEIGHT.  SOLVED, not tabulated, so it cannot drift from the trajectory
#: that arrives there.  [DERIVED]
NOMINAL_POSE = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)

_com, _inertia = SK.body_inertia(NOMINAL_POSE)

#: Whole-robot CoM relative to the TRUNK ORIGIN, body axes.  (0, +0.03, -14.8)
#: mm at the lift pose.  [DERIVED, PINNED]
COM_BODY = np.array(_com, dtype=float)
COM_BODY.flags.writeable = False

#: Whole-robot composite inertia ABOUT THAT CoM, body axes.  (0.028, 0.224,
#: 0.241) kg m^2, against the trunk's OWN Ixx of 0.009 -- the legs are 60 % of
#: the mass and hang below the trunk, so the parallel-axis terms dominate.  A
#: template model that wants one rigid-body inertia wants THIS one, at the
#: stance it will actually hold.  [DERIVED, PINNED]
INERTIA_BODY = np.array(_inertia, dtype=float)
INERTIA_BODY.flags.writeable = False


@dataclass(frozen=True)
class SrbModel:
    """The two pinned numbers, KEPT TOGETHER because they must not drift apart.

    c^b and I^b are both DERIVED FROM ONE POSE, and three places consume them:
    `state.read` (the moment arms and the measured CoM height),
    `reference.com_command` (the commanded CoM height) and
    `controller.balance_wrench` (I_G).  Passing them separately is how the
    reference ends up converting with one c^b while the measurement uses
    another -- and `com_command`'s docstring is explicit that the whole offset
    only cancels out of the z error because BOTH SIDES USE THE SAME CONSTANT.
    One object, passed whole, is what makes that structural instead of a rule.

    STILL PINNED, STILL CONSTANT.  This is not a step toward evaluating the
    CoM per sweep -- that would break exactly the cancellation above and make
    the commanded CoM rate 48 % too fast over the ramp.  What it fixes is the
    POSE THE CONSTANT IS DERIVED AT: the model wants "the stance it will
    actually hold", and holding a folded crouch is a different stance from
    holding the nominal one.
    """

    name: str
    com_body: np.ndarray             # (3,) m, whole-robot CoM from the origin
    inertia_body: np.ndarray         # (3, 3) kg m^2, about that CoM

    @classmethod
    def from_pose(cls, name: str, q) -> "SrbModel":
        """Derive both from a (4, 3) joint pose -- the stance that will be held."""
        com, inertia = SK.body_inertia(np.asarray(q, dtype=float))
        com = np.array(com, dtype=float)
        inertia = np.array(inertia, dtype=float)
        com.flags.writeable = False
        inertia.flags.writeable = False
        return cls(name=name, com_body=com, inertia_body=inertia)

    def describe(self) -> str:
        return ("c^b (%+.1f, %+.1f, %+.1f) mm   I^b diag (%.4f, %.4f, %.4f) "
                "kg m^2   [%s]"
                % (*(1e3 * self.com_body), *np.diag(self.inertia_body),
                   self.name))


#: The nominal model: what `hw.stand` has always flown.  Identical to the two
#: constants above, which stay because `describe`, `selftest` and the banner
#: all name them.
SRB = SrbModel(name="nominal", com_body=COM_BODY, inertia_body=INERTIA_BODY)

MASS = float(P.MASS)                                    # 5.8828 kg   [CAD]
WEIGHT = float(P.WEIGHT)                                # 57.71 N     [CAD]

#: World-frame gravity, z UP.  `b_d` adds +9.81 z_hat, not -9.81: the feet
#: must supply m*a MINUS m*g_vector, and g_vector is this.
GRAVITY_W = np.array([0.0, 0.0, -P.GRAVITY])
GRAVITY_W.flags.writeable = False


# ===========================================================================
# the balance PD  (controller.py)
# ===========================================================================
# ALL SIX GAINS BELOW ARE ACCELERATIONS, NOT FORCES.  The Cheetah-3 balance
# law's PD output is a pair of desired ACCELERATIONS and the robot's own mass
# and inertia turn that into the wrench, so kp is in 1/s^2 and kd in 1/s, and
# one number means the SAME closed-loop bandwidth on every axis whatever the
# pose does to the inertia tensor.
#
# THAT IS THE ENTIRE REASON FOR WRITING THE LAW THIS WAY.  DOG5 had the moment
# as kp*(rpy_ref - rpy) with kp in N*m/rad, which folds the inertia INTO the
# gain -- and this robot's is not isotropic: Iyy and Izz are 8.0x and 8.6x Ixx
# here, so one shared kp gave each axis a different closed loop.  Dividing by
# I_world puts all three at sqrt(kp)/2pi with the same zeta.
#
#   omega_n = sqrt(kp)          zeta = kd / (2 sqrt(kp))

#: Height loop.  2.0 Hz, critically damped.  [UNTUNED -- START HERE]
KP_Z = 160.0            # 1/s^2   -> 2.01 Hz
KD_Z = 25.0             # 1/s     -> zeta 0.99

#: Roll and pitch.  1.5 Hz, slightly under-damped.  THESE ARE THE TWO GAINS
#: THE WHOLE CHANGE EXISTS TO MAKE TUNABLE, so they are the ones to sweep
#: first, in sim, against the recorded failures.  Under the per-leg law there
#: was no such gain to turn.  [UNTUNED -- START HERE]
KP_ATT = 90.0           # 1/s^2   -> 1.51 Hz
KD_ATT = 17.0           # 1/s     -> zeta 0.90

#: Yaw.  THE SPRING IS ON AS OF 2026-09-24, AND WHAT TURNED IT ON IS THE
#: HEADING REZERO, NOT A NEW MAGNETOMETER.  It was off because an absolute
#: heading loop holds the robot against whatever twelve motors and a steel
#: frame are doing to the field.  `law.BalanceLaw.arm` now latches the heading
#: at the crouch -> rise handover and `state.rezero_yaw` turns the world frame
#: onto it, so what multiplies this gain is DRIFT OFF A DATUM MEASURED ON THIS
#: RUN, minutes old -- a difference, in which the mount's own offset and the
#: room's field cancel.  That is the quantity a magnetometer beside motors is
#: least bad at, and it is the quantity the trot moves: `hw.trot_esti` prints
#: the yaw a trot puts in, and it is degrees, not tenths.
#:
#: IT CANNOT ACT BEFORE THE LATCH.  `R_des` is built in `arm` and no moment
#: is asked for until then, so there is no sweep on which this gain is applied
#: to a raw magnetometer reading.
#:
#: SIZED AGAINST THE YAW CAPACITY A DIAGONAL HAS, NOT THE ONE FOUR FEET HAVE.
#: Yaw moment is tangential friction across `FOOT_RADIUS_XY`: mu * W * r =
#: 7.1 N*m on four feet, 3.5 on the two of a trot.  Izz * kp = 5.70 N*m/rad =
#: 0.099 N*m/deg, so the drift a trot actually leaves -- degrees, not tens of
#: them -- asks 0.7 N*m at 7 deg, a fifth of the diagonal's budget, and the
#: cone clamps rather than the gain winning only past about 36 deg.  KP_ATT
#: (90) would ask 2.5 N*m at the same 7 deg and spend 71 % of that budget
#: competing with the roll and pitch moments for it, on two feet, which is
#: where the trot has the least to push against.
#:
#: 0.80 Hz AGAINST ROLL/PITCH'S 1.51 IS DELIBERATE: the heading is the
#: SLOWEST loop on the robot, because it is the one attitude axis whose
#: measurement can lie.  zeta 1.00 goes with that -- a critically damped yaw
#: loop cannot ring on a magnetometer glitch, and the accumulated drift still
#: comes back with a 0.2 s time constant, five times inside one trot cycle.
#: What is left over is slip at touchdown, which no gain here reaches.
#: [UNTUNED -- `--kp-yaw` sweeps it, `--kp-yaw 0` is the old behaviour]
KP_YAW = 25.0           # 1/s^2   -> 0.80 Hz
KD_YAW = 10.0           # 1/s     -> zeta 1.00, the gyro's omega_z

#: Horizontal CoM.  BOTH OFF, AND THIS IS NOT A TUNING EITHER.  p_c,x and
#: p_c,y are WORLD coordinates and nothing in `state.BodyState` measures them
#: -- pinning foot xy in the BODY frame says where the feet are relative to
#: the trunk, not where the trunk is in the world.  A non-zero gain here is a
#: spring anchored to a point that does not exist.
#:
#: Whatever tangential force the feet end up applying comes from the MOMENT
#: rows of the allocation, not from a commanded CoM translation.  Trunk xy
#: stays unregulated exactly as it was under the per-leg law; the difference
#: is that it is now an explicit zero rather than a quantity the law has no
#: way to name.
#:
#: THE ONE EXCEPTION IS `hw.fully_trot`'S TROT, 2026-10-01: there the state
#: estimator (`hw.state_estimator`, MIT's KF) is fed to the law and DOES
#: measure p_c,x/y and their rate, so that trot closes x/y on it with the
#: gains below -- and its hold too since 2026-10-02.  These two stay zero for
#: every phase and entry point that has no estimate fed to it.
KP_XY = 0.0
KD_XY = 0.0

#: THE x/y LOOP ON THE STATE ESTIMATOR -- `hw.fully_trot`, the trot phase;
#: the hold too since 2026-10-02, on request (`law.BalanceLaw.est_xy`).  The
#: filter's world x/y and their rate, converted to the CoM, against the CoM
#: x/y latched on the first sweep of each hold and each trot: station
#: keeping, the stand and the trot IN place.  [UNTUNED; `--kp-xy` /
#: `--kd-xy`; the MuJoCo numbers below are the trot's]
#:
#: A SLOW LOOP ON PURPOSE.  The estimate reaches the law one sweep late (the
#: filter runs in `hw.stand.ESTIMATOR_SLOT`, after slot 0 has acted), 4 ms:
#: 1.4 deg of phase at 1 Hz, 0.7 at this loop's 0.48 Hz, and a reason never
#: to put the attitude loop on it.
#:
#: ITS x/y ARE LEG ODOMETRY, SO IT HOLDS THE ROBOT WHERE THE LEGS SAY IT IS.
#: A planted foot that slides moves the trunk without moving the estimate.
#: In MuJoCo (`hw.fully_trot` has the table) 9/6, 25/10, 50/14 and Kd alone
#: were all stable and all held the FILTER's x/y; none held the true trunk:
#: under a 3 N push the feet slid and the best took back 2 of 20 mm, and in
#: a stance whose front feet slide the filter read the walk with the wrong
#: sign and the loop made it worse.  9/6 is the gentlest that still closes.
KP_XY_EST = 9.0         # 1/s^2   -> 0.48 Hz
KD_XY_EST = 6.0         # 1/s     -> zeta 1.00
#: m/s^2, the most horizontal acceleration that loop may ask for, as a norm:
#: 5.9 N, a tenth of the weight and a fifth of what mu 0.5 lets four feet
#: carry.  An AUTHORITY clamp, DOG5's yaw-error clamp in spirit: a wrong
#: estimate can push the trunk this hard and no harder.
XY_ACC_MAX = 1.0
#: s, how old an estimate may be when the law reads it.  The normal age is
#: one sweep, 4 ms; past this the x/y rows are zero for that sweep -- the
#: flown law, `hw.fold_trot`'s -- and the refusal is counted.  DOG5's rule
#: (`EKF_STALE_S`): a stale estimate driving real force is a fault.
EST_MAX_AGE_S = 0.020
#: s, the same for the accelerometer packet the estimate was made from.  The
#: 0x40 stream is ~100 Hz on its own clock; the filter integrates whatever
#: sample it was last handed, so a stalled stream is a velocity that walks.
EST_ACC_MAX_AGE_S = 0.050


# ===========================================================================
# the attitude SETPOINT  (law.py, latched by sequence.py)
# ===========================================================================
#: WHAT "LEVEL" MEANS FOR A RUN.  Ported from DOG5's config, where it flew as
#: SETPOINT_DYNAMIC from 2026-08-28 -- the IMU mount is the same board in the
#: same orientation on both robots, so the convention ports with it.
#:
#: With this True, `sequence.StandSequence` reads the roll/pitch the IMU
#: reports during the LIMP phase -- zero torque, the robot resting wherever
#: the operator put it -- and latches the attitude setpoint to THAT reading.
#: `law.BalanceLaw.arm` then builds R_des from the latched pair at the
#: arm-time heading, so body <-> world is EXACTLY (0, 0, 0) at the initial
#: stand on all three axes.  It is the yaw lock's "world := body here"
#: convention, completed: yaw already had it, roll and pitch did not.
#:
#: WHAT IT ABSORBS, PER RUN, WITH NOTHING TO MEASURE OR TRANSCRIBE: the IMU
#: mount tilt, the floor's slope, and the resting pose's lean, all three at
#: once.  That is why DOG5 stopped typing a setpoint into this file.
#:
#: WHAT IT COSTS, AND IT IS THE SAME COST DOG5 ACCEPTED: "level" for the run
#: is the LIMP attitude.  Start the robot on a slope and it will hold that
#: slope, and `TILT_STOP_DEG` measures from it too.  For a stand on one patch
#: of floor that is the right trade -- the legs push against the floor that is
#: there.  It is the wrong trade the day the robot has to stay upright across
#: a slope it did not start on.
#:
#: LATCHED ONCE PER RUN, IN LIMP ONLY.  Deliberately NOT re-latched when a
#: later phase passes back through zero torque, and the asymmetry with the yaw
#: lock is the point: heading has no truth to return to, but LEVEL does, and
#: re-latching roll/pitch mid-run would redefine level as whatever tilt the
#: robot was limping at and drag the tilt stop's reference along with it.
SETPOINT_DYNAMIC = True

#: The PRE-LATCH pair, and the sanity reference the latch is warned against.
#: These are DOG5's measured values for this board on a level floor; with
#: SETPOINT_DYNAMIC False they are the whole story, exactly as DOG5 ran before
#: the dynamic latch existed.  UNVERIFIED ON DOG6 -- the board is not mounted,
#: and the moment it is these become the number to re-measure.
SETPOINT_ROLL_DEG = -0.29
SETPOINT_PITCH_DEG = 0.12

#: How far the latched pair may sit from the statics above before the runner
#: says so.  Not a trip: a robot legitimately limps at a few degrees on an
#: uneven floor.  It is the warning DOG5 printed, and it is what catches the
#: case the convention cannot -- the robot propped against something at WAIT.
SETPOINT_WARN_DEG = 2.0


# ===========================================================================
# the allocator  (allocation.py)
# ===========================================================================
#: Friction coefficient the allocator projects onto.  BELOW the cMPC stack's
#: 0.6, and the reason is the 2026-09-15 runs: the rear feet slid forward,
#: late in the lift, repeatedly.
#:
#: mu HERE BOUNDS WHAT THE ALLOCATOR ASKS THE FLOOR FOR.  It cannot bound what
#: the floor gives.  Lowering it is the one thing this law can do about a
#: slide and it is not a fix -- nothing in the loop measures whether a foot
#: stayed put.  [UNTUNED]
MU = 0.5

#: Normal-force box, per foot.  FZ_MIN > 0 is the UNILATERAL CONTACT
#: constraint the per-leg law does not have: nothing in a Cartesian spring
#: stops it commanding a foot to pull upward on the floor.  [INHERITED cmpc]
FZ_MIN = 1.0                        # N
FZ_MAX = 2.0 * WEIGHT               # 115.4 N

#: The least-squares weight, per foot, as (tangential, tangential, normal).
#: The objective is min f^T W f, so a LARGER weight means a SMALLER component:
#: putting the tangential weight an order of magnitude above the normal one
#: biases every solution towards vertical force, which is a soft friction cone
#: at zero cost.  It is what keeps the nominal solution well INSIDE the hard
#: cone instead of on its boundary.  [UNTUNED]
W_TANGENTIAL = 10.0
W_NORMAL = 1.0

#: Per-foot scaling on top of that, in [0, 1].  All ones for a four-foot
#: stand.  It exists because fading a foot out of the solution continuously is
#: how this allocator will later carry a trot, and a term that first appears
#: when it is first needed is a term nothing has ever exercised.
CONTACT_WEIGHT = np.ones(4)

#: Tikhonov damping on the 6x6 normal matrix.  The moment rows' conditioning
#: depends on the xy spread of the feet, and DOG6's is CONSTANT through the
#: stand (+-237 mm, +-65 mm), so this is insurance rather than a load-bearing
#: term: at 1e-6 against a moment-row scale of ~2e-2 it moves the solution by
#: parts in ten thousand.  [UNTUNED]
LAMBDA = 1.0e-6


# ===========================================================================
# the trips  (state.py, allocation.py, and hw.stand)
# ===========================================================================
#: Absolute tilt run-stop, on |roll| or |pitch|.  The stand phase had NO way
#: to trip on the failure it actually exhibits.  DOG5's levelling script
#: carried this and DOG6's stand did not.
#:
#: IT IS NOT A SAFETY NET.  DOG5 logged a run where the tilt trip did not fire
#: and the robot ended up on its belly at 2.4x body weight, reading perfectly
#: LEVEL while doing so -- because a robot lying flat is level.  [UNTUNED]
TILT_STOP_DEG = 12.0

#: Joint tracking trip for the lift.  The stand has no joint-space target, so
#: it has no tracking trip at all; the IK at the pinned foot xy and the
#: commanded height supplies one, and it catches a leg that is badly wrong
#: long before the tilt stop does.  [UNTUNED]
TRACK_STOP_RAD = float(np.deg2rad(25.0))

#: IMU staleness.  A lost IMU mid-stand means the attitude terms are running
#: on an old error.  Past this the law FREEZES the attitude half and holds --
#: it does not trip, because a trip during the lift is a drop.  `hw.imu`
#: already reports packet age and carries `is_stale`; this is a decision the
#: law makes, not a facility it has to build.
IMU_MAX_AGE_S = 0.05

#: Residual trip: how much of the requested wrench the allocator may fail to
#: deliver inside the cone, and for how long.  A large residual means the
#: stance cannot supply what the law is asking for, which is the early warning
#: for a contact that is about to go.
#:
#: SPLIT IN TWO because the six rows are not commensurable -- newtons against
#: newton-metres, and a single norm over both is a number with no units.  The
#: moment threshold is taken against the SHORT lever (+-65 mm, roll), which is
#: the axis that saturates first.  SUSTAINED, not instantaneous: a sweep or
#: two on the cone boundary during a transient is normal.  [UNTUNED]
RESIDUAL_FORCE_N = 0.25 * WEIGHT                 # 14.4 N
RESIDUAL_MOMENT_NM = 0.25 * WEIGHT * 0.065       # 0.94 N*m
RESIDUAL_STREAK = 25                             # sweeps -> 0.10 s at 250 Hz


# ===========================================================================
# the trot in place  (gait.py, swing.py, and the trot half of law/sequence)
# ===========================================================================
# EVERY NUMBER IN THIS SECTION IS ONE DOG5 FLEW in
# dog5/src/dog5_trot_quasi_static_model/config.py (trot_demo.py's run), and
# is tagged [DOG5 FLOWN].  The STRUCTURE is cMPC's -- `sim.cmpc.gait`'s phase
# arithmetic and `sim.cmpc.swing`'s arc -- and the operator's two decisions of
# 2026-09-16 sit on top: no foot placement (the foot rises and lands on the
# spot it left, in the trunk frame), and a torque cap at the motors' own 9 N*m.

#: One full cycle, the duty, and the diagonal pairing over (FL, FR, RL, RR).
#: DUTY > 0.5 IS NOT A PREFERENCE: the contact ramp below needs a window with
#: both diagonals down to hand the load across in, and at 0.5 there is none.
GAIT_PERIOD = 1.2                   # s       [DOG5 FLOWN]
DUTY = 0.80                         #         [DOG5 FLOWN]
PHASE_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])    # FL+RR, FR+RL   [DOG5 FLOWN]

#: Fraction of stance over which a foot's share of the load smoothsteps up
#: after touchdown and down before liftoff.  DOG5 measured a 2.2 N*m torque
#: step per handover without it, and collapsed eight runs in a row on
#: 2026-08-28 the one time it was removed.  [DOG5 FLOWN]
CONTACT_RAMP = 0.15

#: trot_demo's re-level: every SETTLE_EVERY full cycles the gait clock freezes
#: for SETTLE_S with all four feet at full weight.  DOG5 never solved its
#: trot roll; this is how its demo held.  Its alternating lead diagonal is
#: NOT kept: removed on request 2026-09-21, one fixed trot.
SETTLE_S = 0.2                      # s       [DOG5 FLOWN]
SETTLE_EVERY = 2                    # cycles  [DOG5 FLOWN]

#: THE JOINT-SPACE LAYER, 2026-09-21: DOG5 trot_hw's `JointImpedance`,
#: tau += KP (q_hold - q) - KD qd under EVERY leg, on top of the SRB stance
#: torque and the swing.  q_hold is the measured joint angles on the sweep
#: the robot REACHES HOLD, fixed from then on through the hold and every
#: trot: feet planted, fixed joints are a fixed trunk -- position and rpy.
#: A W step releases it and re-latches at the new stance.  A SWINGING leg
#: gets NONE of the layer since 2026-10-01: before then its target followed
#: the leg, as DOG5's q_ref did ("left at the stance pose the joint floor
#: fights the swing"), and the damper alone stayed -- a drag on the arc that
#: halved the apex (`law.BalanceLaw.update` says why it went).  DOG5's reason for it: a pure force law is velocity-level, and a
#: foot off the ground coasts; kp gives the joint a fixed point.  DOG5 put
#: the value back to 3.0 on 2026-08-28 -- 8.0 was at the delay-phase gate
#: and 15.0 shook at 9-12 Hz.  `--kp-joint` / `--kd-joint`.
#:
#: 5.0 / 0.2 SINCE 2026-09-25, ON REQUEST: a stiffer trunk.  Through the
#: lift pose's Jacobian one leg's 3 N*m/rad was 113 / 111 / 1253 N/m in
#: x / y / z and 5 is 189 / 186 / 2088 -- 755 N/m of trunk xy on four feet
#: against 453, 377 on a diagonal against 226.  Between DOG5's flown 3.0 and
#: its 8.0 gate; zeta on the knee's rotor goes 0.31 -> 0.47.  The z share
#: (eleven times the xy) is the part that leans on the attitude loop, which
#: is why the step past this is `--kp-stance-xy`, not more of this.
#: [DOG5 FLOWN at 3.0 / 0.1; 5.0 / 0.2 is DOG6's]
KP_JOINT_HOLD = 5.0                 # N*m/rad
KD_JOINT_HOLD = 0.2                 # N*m*s/rad

#: The ABDUCTION joints' own gains in that layer once a pose is latched
#: (HOLD and the trot), 2026-10-02, on request: lock abd in joint space,
#: pitch and knee keep the two above, and the tracked rise keeps those on
#: all three joints.  `--kp-joint-abd` / `--kd-joint-abd`.  Defaults are the
#: layer as it was; no stiffer value has been checked yet.
KP_JOINT_HOLD_ABD = KP_JOINT_HOLD   # N*m/rad
KD_JOINT_HOLD_ABD = KD_JOINT_HOLD   # N*m*s/rad

#: The swing apex above the resting foot, in the trunk frame.  20 mm since
#: 2026-10-02, the operator's best fold trot (below); DOG5's was 40.
SWING_HEIGHT = 0.020                # m

#: Swing-foot Cartesian impedance, TRUNK frame, (x, y, z).  N/m and N s/m --
#: FORCE gains on the foot, not the balance PD's acceleration gains.
#: THE OPERATOR'S BEST, 2026-10-02, on the robot, every trot's default on
#: request: `hw.fold_trot --period 0.6 --swing-height 20 --tau-slew 120
#: --kp-swing 10 10 400 --kd-swing 5 5 40`, flown with no joint-layer damper
#: on the swinging leg (`law.BalanceLaw.update`) -- "now the swing leg height
#: is good".  Soft in x/y, stiff in z.  DOG5's flown 140/140/180 and 8/8/15
#: until then.
KP_SWING = np.array([10.0, 10.0, 400.0])
KD_SWING = np.array([5.0, 5.0, 40.0])

#: The JOINT swing (`--swing joint`), (abd, pitch, knee).  N*m/rad and
#: N*m*s/rad.  The fold trot's until 2026-09-28, when it failed on it.
#: [DERIVED -- NOT FLOWN; DOG5 had no joint swing to copy]
#: From the reflected rotor alone, `params.ARMATURE` 0.0085 kg m^2, at the
#: 250 Hz sweep: omega_n = sqrt(30 / 0.0085) = 59 rad/s, zeta = 0.79, and the
#: explicit-PD bound Kd dt / I = 0.38 against 2.  Against the Cartesian
#: swing's 3.8 N*m/rad about abd in the fold stance, abd is 8x stiffer.
KP_SWING_JOINT = np.array([30.0, 30.0, 30.0])
KD_SWING_JOINT = np.array([0.8, 0.8, 0.8])

#: The KNEE swing (`--swing knee`, 2026-10-02, `swing.py`), (abd, pitch,
#: knee), N*m/rad and N*m*s/rad: abd and pitch held at their liftoff angles,
#: the knee on its bump.  `--kp-swing-knee` / `--kd-swing-knee`.
#: [DERIVED -- NOT FLOWN] Started from the joint swing's rotor numbers above.
KP_SWING_KNEE = np.array([30.0, 30.0, 30.0])
KD_SWING_KNEE = np.array([0.8, 0.8, 0.8])

#: `safety.SafetyGate`'s slew for a trot.  The gate's default 5 N*m/s is the
#: stand's and would take 0.35 s to follow one handover -- longer than the
#: ramp it is following.  60 is what every DOG5 trot run used.  120 since
#: 2026-10-02, the operator's best fold trot (`KP_SWING` above).
TAU_SLEW_TROT_NM_S = 120.0


# ===========================================================================
# walking  (trajectory.py, keys.py, qp.py, footstep.py, swing_control.py,
#           walk.py -- the `hw.fold_walk` entry point, 2026-10-04)
# ===========================================================================
# THE STRUCTURE IS `sim.cmpc`'S: the operator's body-axis velocity is
# integrated into an x, y, yaw reference (`sim.cmpc.trajectory`, imported),
# the swing foot lands on eq (33) plus a velocity-feedback term, and the arc
# is cMPC's own quintic in the WORLD so the foot meets the ground at ground
# speed.  The law under it is still the SRB PD with no horizon, so every
# number below is [SIM-TUNED] in doc/walk/walksim.py's MuJoCo harness (the
# repo's own sequence, law, gait and gate) and NOTHING here has flown.

#: One W/S/A/D press, m/s; one Q/E press, rad/s.  Half the simulator's
#: (`sim.cmpc.config.V_STEP`, `YAW_RATE_STEP`): the hardware law has no
#: horizon to plan a velocity change into, so it gets smaller ones.
WALK_V_STEP = 0.05
WALK_YAW_STEP = float(np.radians(10.0))

#: What the keys may accumulate to: inside what MuJoCo walked, not what the
#: robot might.  [SIM-TUNED] doc/walk/README.md has the sweep: with the
#: shipped walk, six gait phases a case, 0.10 m/s forward or back, 0.05 and
#: 0.08 m/s sideways, 20 and 40 deg/s turns and 0.10 m/s with a 20 deg/s turn
#: all stayed under 10 deg of tilt; 0.15 m/s forward tipped 3 of 6 (2 to the
#: tilt stop).  The box is the first flights', below the 0.08 / 40 that also
#: passed; --v-max and --yaw-rate-max open it.
WALK_VX_MAX = 0.10                  # m/s
WALK_VY_MAX = 0.05                  # m/s
WALK_YAW_RATE_MAX = float(np.radians(20.0))   # rad/s

#: The command is SLEWED toward what the keys ask, never stepped: a step in
#: v_ref is a step in the x/y rows' velocity error and in the footholds.
WALK_ACC_MAX = 0.5                  # m/s^2
WALK_YAW_ACC_MAX = float(np.radians(90.0))    # rad/s^2

#: THE LEASH -- MIT's `max_pos_error` (ConvexMPCLocomotion.cpp, 0.1 m on
#: Cheetah 3).  The integrated reference is kept within this of where the
#: filter has the CoM, so a robot that falls behind is never chased by a
#: setpoint running away from it; it bounds what the x/y rows AND the
#: world-anchored stance targets can ask.  DOG6 is a third of Cheetah's size.
WALK_LEASH = 0.03                   # m
WALK_YAW_LEASH = float(np.radians(10.0))      # rad

#: The foothold, eq (33) with Raibert's feedback term:
#:   p_land = site(t_td) + v_hat T_st / 2 + STEP_KV (v_hat - v_ref)
#: STEP_KV is MIT's 0.03 s by default (`pfx_rel`); the capture point would
#: be sqrt(h/g) = 0.14 s.  STEP_MAX bounds the landing's distance from the
#: stance's own neutral site, trunk x / y.  [SIM-TUNED]
STEP_KV = 0.03                      # s
STEP_MAX_XY = (0.06, 0.04)          # m

#: The swing foot's Cartesian impedance while WALKING, trunk frame.  The trot
#: in place's 10 N/m in x/y (`KP_SWING`) is too soft to carry a foot across
#: a stride: at 0.1 m/s it must move 48 mm in 120 ms, ~6 N peak on the 0.32 kg
#: the foot weighs in x, against the 0.1 N 10 N/m makes of 10 mm.  z keeps
#: the operator's 400 / 40.  [SIM-TUNED]
KP_SWING_WALK = np.array([150.0, 150.0, 400.0])
KD_SWING_WALK = np.array([6.0, 6.0, 40.0])

#: The walk's TASK-SPACE COMPUTED TORQUE swing (swing_control.swing_osc_torque):
#: acceleration gains, so a bandwidth per axis whatever the foot's apparent
#: mass.  wn rad/s per axis (x, y, z), zeta.  [SIM-TUNED]
#: Z IS THE EXPENSIVE AXIS.  The foot is 0.31 / 0.36 kg in x / y and 5.8 kg
#: in z (the reflected rotor through a short z lever), so z bandwidth is
#: torque RATE, which the gate slews at 120 N*m/s: at 25/25/30 the swing's
#: request peaked at 12-15 N*m and the gate passed a triangle of it, the foot
#: rising 80 mm for a 20 mm apex (doc/walk/README.md, fig_swing_slew).
#: 25/25/20 against 25/25/30, six gait phases a case in MuJoCo: a 40 deg/s
#: turn fell 1 of 6 against 5, 0.08 m/s sideways 0 of 5 against 2, 0.10 m/s
#: forward tipped past 15 deg 1 of 6 (no tilt stop) against 2 of 6 to the
#: stop.  Softer z (15) or stiffer x/y (35) were both worse.
WN_SWING_OSC = np.array([25.0, 25.0, 20.0])
ZETA_SWING_OSC = 0.7

#: THE WALK'S SWING LAW AND CLOCK, `hw.fold_walk`'s defaults.  [SIM-TUNED]
#: doc/walk/README.md has the runs; in short: at the trot's duty 0.80 (120 ms
#: of swing) no swing law walked 0.1 m/s in MuJoCo, and the reason is the
#: z axis -- the reflected rotor makes the foot 5.8 kg in z (0.3 in x), so
#: the 20 mm arc's own feedforward asks ~300 N*m/s of a knee the gate slews
#: at 120 (TAU_SLEW_TROT_NM_S).  0.70 gives the swing 180 ms; the contact
#: ramp has to fit the four-foot window, (duty - 0.5) / (2 duty) of stance,
#: and takes 90 % of it.
WALK_SWING_LAW = "osc"
WALK_DUTY = 0.70
WALK_CONTACT_RAMP = 0.9 * (WALK_DUTY - 0.5) / (2.0 * WALK_DUTY)
#: NO SETTLE WHILE WALKING.  DOG5's re-level freezes the gait clock 0.2 s
#: every 2 cycles with the four feet down -- and the reference does not
#: freeze: at 0.1 m/s the trunk moves 20 mm over planted feet, the stance
#: outlasts the T_st the footholds were planned for, and the next swing
#: starts 20 mm behind.  Six gait phases a case in MuJoCo, the settle off
#: against on: 0.10 m/s forward never past 6.6 deg (against 17.9), 0.10 m/s
#: with a 20 deg/s turn 0 of 6 tipped (against 4), a 40 deg/s turn 0 of 6
#: (against 1 to the tilt stop), peak torque 3.8-6.2 N*m (against 4.2-8.7).
WALK_SETTLE_S = 0.0

#: The walking stance targets are WORLD-anchored (walk.py): each planted
#: foot's target is where the reference trunk sees it.  Per leg, the xy part
#: of (target - measured) is clamped to this -- K_c,xx * 20 mm is 2.8 N a leg
#: at the fold stand's Kp 5 -- so an estimate gone wrong pulls no harder.
HOLD_XY_ERR_MAX = 0.020             # m

#: THE QP ALLOCATOR (qp.py) -- the cost it trades:
#:   1/2 |A f - b_d|^2_S + 1/2 alpha f' W f + 1/2 beta |f - f_prev|^2
#: S = I weighs a newton like a newton-metre, which is the metric the least
#: squares (`allocation.py`) resolves an unreachable wrench in -- so on a
#: diagonal pair the QP gives up the same part of the moment the flown law
#: did, and differs from it in WHERE the cone is enforced: inside the
#: optimisation, not by clipping and rescaling afterwards.  alpha W is the
#: soft cone (W is allocation's (10, 10, 1)); small, so the wrench is
#: tracked to ~0.1 N.  beta 0: the gate's slew already smooths.  [UNTUNED]
QP_S = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
QP_ALPHA = 1.0e-4
QP_BETA = 0.0
#: Active-set iterations a sweep may take; the cap returns the last FEASIBLE
#: iterate.  Most sweeps take none (no face touched: the unconstrained
#: optimum is the answer); the slowest of a MuJoCo walk took 13, and 400
#: random wrenches in `selftest` 23.
QP_MAX_ITER = 30


def describe() -> str:
    return "\n".join([
        "DOG6 balance controller configuration",
        "  HEIGHTS ARE FLOOR TO TRUNK BOTTOM.  The code's own frame is the",
        "  trunk ORIGIN, %.2f mm higher (params.TRUNK_BOX_HALF_z)."
        % (1e3 * TRUNK_BOTTOM_OFFSET),
        "    crouch      h = %6.1f mm   (origin %6.1f mm)"
        % (1e3 * H_CROUCH, 1e3 * ST.CROUCH_HEIGHT),
        "    lift        h = %6.1f mm   (origin %6.1f mm) over %.1f s"
        % (1e3 * H_LIFT, 1e3 * ST.LIFT_HEIGHT, T_RISE),
        "    CAD stand   h = %6.1f mm   -- the lift stops short of it"
        % (1e3 * H_CAD_STAND),
        "    stance      x %+.0f/%+.0f mm, y %+.0f/%+.0f mm in the trunk frame"
        % (1e3 * (FOOT_XY[0, 0] + P.HIP_OFFSET[0, 0]),
           1e3 * (FOOT_XY[2, 0] + P.HIP_OFFSET[2, 0]),
           1e3 * (FOOT_XY[0, 1] + P.HIP_OFFSET[0, 1]),
           1e3 * (FOOT_XY[1, 1] + P.HIP_OFFSET[1, 1])),
        "  SRB model, PINNED at the lift pose (see the module docstring):",
        "    c^b         (%+.2f, %+.2f, %+.2f) mm from the trunk origin"
        % tuple(1e3 * COM_BODY),
        "    I^b diag    (%.4f, %.4f, %.4f) kg m^2   trunk alone Ixx %.4f"
        % (*np.diag(INERTIA_BODY), P.TRUNK_INERTIA[0][0]),
        "    m           %.4f kg = %.2f N" % (MASS, WEIGHT),
        "  balance PD -- ACCELERATION gains (kp 1/s^2, kd 1/s)  [ALL UNTUNED]",
        "    height      kp %6.1f  kd %5.1f   -> %.2f Hz, zeta %.2f"
        % (KP_Z, KD_Z, np.sqrt(KP_Z) / (2 * np.pi), KD_Z / (2 * np.sqrt(KP_Z))),
        "    roll/pitch  kp %6.1f  kd %5.1f   -> %.2f Hz, zeta %.2f"
        % (KP_ATT, KD_ATT, np.sqrt(KP_ATT) / (2 * np.pi),
           KD_ATT / (2 * np.sqrt(KP_ATT))),
        "    yaw         kp %6.1f  kd %5.1f   -> %s"
        % (KP_YAW, KD_YAW,
           "spring OFF: the damper only" if not KP_YAW else
           "%.2f Hz, zeta %.2f -- DRIFT off the latched heading"
           % (np.sqrt(KP_YAW) / (2 * np.pi), KD_YAW / (2 * np.sqrt(KP_YAW)))),
        "                asks %.3f N*m/deg against %.1f N*m of yaw capacity "
        "on four feet, %.1f on a trot's diagonal"
        % (np.radians(1.0) * INERTIA_BODY[2, 2] * KP_YAW,
           MU * WEIGHT * FOOT_RADIUS_XY, 0.5 * MU * WEIGHT * FOOT_RADIUS_XY),
        "    CoM x,y     kp %6.1f  kd %5.1f   -- OFF: nothing measures them"
        % (KP_XY, KD_XY),
        "    CoM x,y     kp %6.1f  kd %5.1f   -> %.2f Hz, zeta %.2f -- "
        "every trot's hold and trot (--est-xy), ON THE ESTIMATOR"
        % (KP_XY_EST, KD_XY_EST, np.sqrt(KP_XY_EST) / (2 * np.pi),
           KD_XY_EST / (2 * np.sqrt(KP_XY_EST))),
        "                |a_xy| <= %.1f m/s^2 (%.1f N); estimate <= %.0f ms "
        "old, its 0x40 <= %.0f ms, or the rows are zero"
        % (XY_ACC_MAX, MASS * XY_ACC_MAX, 1e3 * EST_MAX_AGE_S,
           1e3 * EST_ACC_MAX_AGE_S),
        "  saturation: kp_att on this inertia is %.2f N*m/rad in roll and"
        % (INERTIA_BODY[0, 0] * KP_ATT),
        "    %.2f in pitch, against a stance capacity of %.1f and %.1f N*m"
        % (INERTIA_BODY[1, 1] * KP_ATT, WEIGHT * 0.065, WEIGHT * 0.237),
        "  allocator",
        "    mu %.2f   fz [%.1f, %.1f] N   W (t,t,n) = (%.0f, %.0f, %.0f)"
        "   lambda %.1e" % (MU, FZ_MIN, FZ_MAX, W_TANGENTIAL, W_TANGENTIAL,
                            W_NORMAL, LAMBDA),
        "  attitude setpoint",
        "    %s" % ("DYNAMIC -- latched from the LIMP reading, once per run"
                    if SETPOINT_DYNAMIC else
                    "STATIC -- the config pair below, as DOG5 ran pre-2026-08-28"),
        "    statics     roll %+.2f  pitch %+.2f deg   (%s; warn past %.1f)"
        % (SETPOINT_ROLL_DEG, SETPOINT_PITCH_DEG,
           "pre-latch + sanity reference" if SETPOINT_DYNAMIC else "IN USE",
           SETPOINT_WARN_DEG),
        "  trips",
        "    tilt %.0f deg   tracking %.0f deg   imu age %.0f ms (freeze, "
        "not trip)" % (TILT_STOP_DEG, np.rad2deg(TRACK_STOP_RAD),
                       1e3 * IMU_MAX_AGE_S),
        "    residual %.1f N / %.2f N*m sustained %d sweeps"
        % (RESIDUAL_FORCE_N, RESIDUAL_MOMENT_NM, RESIDUAL_STREAK),
        "  trot in place  [DOG5 FLOWN]",
        "    period %.2f s  duty %.2f  ramp %.2f  settle %.2f s every %d "
        "cycles"
        % (GAIT_PERIOD, DUTY, CONTACT_RAMP, SETTLE_S, SETTLE_EVERY),
        "    swing apex %.0f mm, no placement   Kp %s N/m  Kd %s N s/m   "
        "slew %.0f N*m/s"
        % (1e3 * SWING_HEIGHT, KP_SWING, KD_SWING, TAU_SLEW_TROT_NM_S),
        "    joint swing          Kp %s N*m/rad  Kd %s N*m*s/rad   abd held"
        % (KP_SWING_JOINT, KD_SWING_JOINT),
        "    joint hold layer     Kp %.1f N*m/rad  Kd %.2f N*m*s/rad   every "
        "leg, q latched on reaching HOLD" % (KP_JOINT_HOLD, KD_JOINT_HOLD),
        "  walking (hw.fold_walk)  [SIM-TUNED, NOT FLOWN]",
        "    keys        +-%.2f m/s, +-%.0f deg/s a press; box vx %.2f vy %.2f "
        "m/s, yaw %.0f deg/s"
        % (WALK_V_STEP, np.degrees(WALK_YAW_STEP), WALK_VX_MAX, WALK_VY_MAX,
           np.degrees(WALK_YAW_RATE_MAX)),
        "    reference   slew %.2f m/s^2, %.0f deg/s^2; leash %.0f mm, %.0f deg"
        % (WALK_ACC_MAX, np.degrees(WALK_YAW_ACC_MAX), 1e3 * WALK_LEASH,
           np.degrees(WALK_YAW_LEASH)),
        "    foothold    eq (33) + %.2f s (v_hat - v_ref), within %s mm of "
        "the site" % (STEP_KV, tuple(int(1e3 * x) for x in STEP_MAX_XY)),
        "    swing       %s: wn %s rad/s zeta %.2f   (impedance: Kp %s Kd %s)"
        % (WALK_SWING_LAW, WN_SWING_OSC, ZETA_SWING_OSC, KP_SWING_WALK,
           KD_SWING_WALK),
        "    clock       duty %.2f, ramp %.3f, settle %.2f s   stance "
        "targets world-anchored, |xy err| <= %.0f mm"
        % (WALK_DUTY, WALK_CONTACT_RAMP, WALK_SETTLE_S,
           1e3 * HOLD_XY_ERR_MAX),
        "    QP          S %s  alpha %.0e  beta %.0e  cap %d iterations"
        % (QP_S, QP_ALPHA, QP_BETA, QP_MAX_ITER),
    ])


if __name__ == "__main__":
    print(describe())
