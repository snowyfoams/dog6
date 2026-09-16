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

#: Yaw.  THE SPRING IS OFF AND THAT IS A DECISION, NOT A PLACEHOLDER.
#: Absolute yaw comes from the magnetometer, which sits beside twelve motors
#: and a steel frame; `hw.imu` labels it untrusted, and closing a loop on it
#: would hold the robot against whatever the frame is doing to the field.  The
#: yaw RATE is the gyro's omega_z -- inertial, fine -- so the DAMPER stays on.
KP_YAW = 0.0            # 1/s^2
KD_YAW = 10.0           # 1/s

#: Horizontal CoM.  BOTH OFF, AND THIS IS NOT A TUNING EITHER.  p_c,x and
#: p_c,y are WORLD coordinates and nothing on DOG6 measures them -- there is
#: no state estimator, and pinning foot xy in the BODY frame says where the
#: feet are relative to the trunk, not where the trunk is in the world.  A
#: non-zero gain here is a spring anchored to a point that does not exist.
#:
#: Whatever tangential force the feet end up applying comes from the MOMENT
#: rows of the allocation, not from a commanded CoM translation.  Trunk xy
#: stays unregulated exactly as it was under the per-leg law; the difference
#: is that it is now an explicit zero rather than a quantity the law has no
#: way to name.
KP_XY = 0.0
KD_XY = 0.0


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
        "    yaw         kp %6.1f  kd %5.1f   -- spring OFF: magnetometer"
        % (KP_YAW, KD_YAW),
        "    CoM x,y     kp %6.1f  kd %5.1f   -- OFF: nothing measures them"
        % (KP_XY, KD_XY),
        "  saturation: kp_att on this inertia is %.2f N*m/rad in roll and"
        % (INERTIA_BODY[0, 0] * KP_ATT),
        "    %.2f in pitch, against a stance capacity of %.1f and %.1f N*m"
        % (INERTIA_BODY[1, 1] * KP_ATT, WEIGHT * 0.065, WEIGHT * 0.237),
        "  allocator",
        "    mu %.2f   fz [%.1f, %.1f] N   W (t,t,n) = (%.0f, %.0f, %.0f)"
        "   lambda %.1e" % (MU, FZ_MIN, FZ_MAX, W_TANGENTIAL, W_TANGENTIAL,
                            W_NORMAL, LAMBDA),
        "  trips",
        "    tilt %.0f deg   tracking %.0f deg   imu age %.0f ms (freeze, "
        "not trip)" % (TILT_STOP_DEG, np.rad2deg(TRACK_STOP_RAD),
                       1e3 * IMU_MAX_AGE_S),
        "    residual %.1f N / %.2f N*m sustained %d sweeps"
        % (RESIDUAL_FORCE_N, RESIDUAL_MOMENT_NM, RESIDUAL_STREAK),
    ])


if __name__ == "__main__":
    print(describe())
