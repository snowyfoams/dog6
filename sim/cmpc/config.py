"""Every constant the convex MPC reads.  No logic, no sibling imports.

A reproduction of Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3
Through Convex Model-Predictive Control" (IROS 2018), against DOG6.

WHAT THIS REPRODUCTION ASSUMES, DELIBERATELY
    Two things are handed to the controller rather than estimated, and both
    are simplifications the paper does not make.  They are here because the
    question being asked is "does the convex MPC formulation work on DOG6",
    not "does a state estimator work":

    1.  THE TWELVE-STATE BODY POSE IS FULLY OBSERVED.  Position, linear
        velocity, orientation and angular velocity are read from the
        simulator.  No IMU, no leg odometry, no filter.  The model carries an
        IMU site and three sensors, and this controller ignores all of them.

    2.  THE GAIT IS A PURE TIMETABLE.  Contact state is a function of the
        clock alone -- `contact(t)`, no contact sensing, no early-touchdown
        detection, no schedule adaptation.  A foot is in stance because the
        table says so, and the MPC plans forces for it on that basis.

    Both are what makes the result interpretable: anything that goes wrong is
    the MPC or the swing law, not a filter or a contact estimator.

WHERE THE NUMBERS COME FROM
    [PAPER]   the paper's own value, or its open-source implementation's
              default.  Carried unchanged so the reproduction is a
              reproduction.
    [DOG6]    a property of this robot, read from `sim.params`.
    [TUNED]   chosen here, with the reason written down.
"""
from __future__ import annotations

import numpy as np

from .. import params as P

# ===========================================================================
# the state vector
# ===========================================================================
# THIRTEEN, NOT TWELVE, AND THE THIRTEENTH IS A CONSTANT.  The twelve states
# that are observed are the ones the paper lists; gravity is appended as a
# fake state so that the affine term m*g in the translational dynamics becomes
# a linear one.  Without it the dynamics are x_dot = A x + B u + c and the
# condensed QP would need the constant carried separately through every step
# of the horizon; with it, x_dot = A x + B u exactly, and A's last column does
# the work.  x[12] is initialised to -9.81 and never changes.        [PAPER]
#
#   0:3   Theta  = (roll, pitch, yaw)     ZYX Euler, world
#   3:6   p      = (x, y, z)              world
#   6:9   omega  = (wx, wy, wz)           world
#   9:12  v      = (vx, vy, vz)           world
#   12    g      = -9.81                  constant
STATE_DIM = 13
INPUT_DIM = 12                   # four feet x (fx, fy, fz), world frame

RPY = slice(0, 3)
POS = slice(3, 6)
OMEGA = slice(6, 9)
VEL = slice(9, 12)
GRAV = 12

STATE_NAMES = ("roll", "pitch", "yaw", "x", "y", "z",
               "wx", "wy", "wz", "vx", "vy", "vz", "g")

# ===========================================================================
# horizon and rate
# ===========================================================================
#: The MPC re-solves at 40 Hz.  Each solve plans HORIZON steps of MPC_DT.
#:
#: 40, NOT 20, AND THE REASON IS AN UNSTABLE MODE THIS ROBOT HAS.  While a
#: trot stands on one diagonal, the body is an inverted pendulum about that
#: support line: CoM 0.163 m up, 0.211 kg m^2 about the line, so
#:
#:     omega = sqrt(m g h / I) = 6.69 rad/s = 1.06 Hz,  time constant 150 ms
#:
#: At 20 Hz the MPC gets THREE samples per time constant of that mode, which
#: is not enough to stabilise it.  Measured at 0.25 m/s, the roll became a
#: period-2 subharmonic -- the sign alternating every stride, -6.4, +7.3,
#: -6.5, +8.1 deg -- oscillating at 1.00 Hz against a mode at 1.06 Hz, and
#: growing: 3.5 deg rms over the first seconds, 19 deg by twelve, then a fall.
#:
#:     MPC      samples per time constant     roll rms at 0.25 m/s
#:      10 Hz              1.5                19.50 deg, fell at 3.9 s
#:      20 Hz              3.0                14.64 deg, fell at 4.9 s
#:      40 Hz              6.0                 1.58 deg
#:      50 Hz              7.5                 1.24 deg
#:
#: NO LEG-LOOP RATE FIXES THIS, and that is worth knowing before reaching for
#: one.  Sweeping the control rate 100 -> 1000 Hz moves the roll from 14.0 to
#: 17.0 deg and every run still falls; sweeping the IMU 50 -> 500 Hz does
#: nothing either.  The stance force is HELD between solves, so a faster leg
#: loop re-maps the same force through a fresher Jacobian -- it never produces
#: a new one.  Only the MPC changes the wrench, so only the MPC can fight a
#: roll error.
MPC_HZ = 40.0                                        # [TUNED, see above]
MPC_DT = 1.0 / MPC_HZ                                # 0.025 s

#: TEN STEPS -- 0.25 s, which is ONE STANCE PHASE, not one gait period.
#:
#: THE HORIZON'S LENGTH TURNS OUT NOT TO MATTER MUCH HERE, WHICH IS NOT WHAT
#: THIS COMMENT USED TO SAY.  It argued that the horizon had to cover a full
#: gait period so every solve could see both diagonals touch down.  Measured,
#: that is not what decides stability:
#:
#:     20 Hz x 10 = 0.50 s lookahead    roll 14.64 deg, fell
#:     20 Hz x 20 = 1.00 s lookahead    roll 16.47 deg, fell
#:     40 Hz x 10 = 0.25 s lookahead    roll  1.58 deg      6.5 ms/solve
#:     40 Hz x 20 = 0.50 s lookahead    roll  1.70 deg     17.0 ms/solve
#:
#: Doubling the lookahead at 20 Hz makes it slightly worse; halving it at
#: 40 Hz costs nothing.  The rate is what matters.  Ten steps is kept because
#: it is the cheapest option that works -- the QP is dense, so twenty steps is
#: 240 variables and 68 % of the 25 ms budget against 18 % for ten.
HORIZON = 10                                         # [PAPER: 10]

#: The leg-level loop.  EVERYTHING below the MPC runs at this one rate: the
#: state is read once per sweep (R, q, qd), and both the world-to-body force
#: rotation fb = R' f and the torque map tau = J' fb happen in that same
#: sweep, inside `swing.stance_torque`.  There is no intermediate rate.
#:
#: The MPC's force is HELD across sweeps -- that zero-order hold is the method,
#: not a shortcut, and it is what `dynamics.discretize` assumes.
CONTROL_HZ = P.CONTROL_HZ                            # 250 Hz   [DOG6]
CONTROL_DT = 1.0 / CONTROL_HZ

#: The IMU's own output rate, and therefore the rate at which the body
#: orientation R exists at all.
#:
#: THE TWO SENSORS DO NOT AGREE ON A RATE, AND THE CONTROLLER MUST NOT PRETEND
#: THEY DO.  R comes from the IMU at 200 Hz; the joint angles q come back from
#: the encoders with the CAN sweep at 250 Hz.  So the world-to-body force
#: rotation fb = R' f can only be refreshed 200 times a second, while the
#: torque map tau = J' fb is refreshed 250 times a second on fresh encoder
#: data.  Between IMU samples the body-frame force is HELD.
#:
#: Simulating that split is the point.  A sim that reads one perfectly
#: synchronised state at 250 Hz is optimistic about a robot whose attitude is
#: up to 8 ms old by the time a torque is written -- and it is optimistic in
#: exactly the axis (orientation) that decides whether a stance force pushes
#: the robot up or sideways.
IMU_HZ = 200.0                                       # [DOG6 hardware]
IMU_DT = 1.0 / IMU_HZ                                # 0.005 s

#: Sweeps per solve, NOMINALLY.  20 Hz does not divide 250 Hz -- the ratio is
#: 12.5 -- so the real schedule alternates 12 and 13 sweeps and averages
#: exactly MPC_DT.  See `controller.update` for why the deadline is counted
#: from a fixed origin rather than from the last solve: re-anchoring on the
#: actual solve turns that 12.5 into a steady 13 and the MPC into a 19.23 Hz
#: loop that still calls itself 20.
SWEEPS_PER_SOLVE = CONTROL_HZ / MPC_HZ               # 12.5

# ===========================================================================
# the gait, as a timetable
# ===========================================================================
#: One full cycle.  DOG5/DOG6's trot work settled on 0.50 s, and the reason is
#: the gearbox: a swing has to lift the foot and put it back in (1-duty)*T, so
#: the acceleration it demands goes as 1/T_swing^2, and what has to be
#: accelerated is the foot's APPARENT mass -- 4.54 kg vertically with the
#: armature, not the 0.45 kg the links weigh.  At 0.40 s the arc asks for
#: roughly the motor's entire saturation torque for the swing alone.  [DOG6]
GAIT_PERIOD = 0.50                                   # s

#: Fraction of the cycle each foot spends in stance.
#:
#: 0.5 IS A TRUE TROT AND HAS ZERO DOUBLE SUPPORT.  That is fine HERE and was
#: not fine for the quasi-static controller this project's predecessor ran: a
#: force law reacting to today's error has nowhere to hand the load over at
#: the crossover instant, so it needs duty > 0.5 and a contact ramp.  An MPC
#: plans the handover -- at the switch it already knows the next diagonal is
#: coming down, and it loads it in advance.  Not needing the ramp is one of
#: the things this reproduction is meant to show.             [PAPER]
DUTY = 0.5

#: Trot pairing: FL with RR, FR with RL.  Offsets are in CYCLES, indexed in
#: `coordinates.LEGS` order (FL, FR, RL, RR).
PHASE_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])

STANCE_DURATION = DUTY * GAIT_PERIOD                 # 0.25 s
SWING_DURATION = (1.0 - DUTY) * GAIT_PERIOD          # 0.25 s

# ===========================================================================
# MPC cost weights
# ===========================================================================
#: Diagonal of Q, one weight per state, in STATE_NAMES order.
#:
#: THESE ARE THE CHEETAH IMPLEMENTATION'S DEFAULTS, CARRIED UNCHANGED.  They
#: are not tuned for DOG6 and are not claimed to be optimal for it -- carrying
#: them is what makes this a reproduction rather than a new controller, and it
#: means any DOG6 retune shows up as a readable diff against the paper.
#:
#: The shape is worth reading, though.  Yaw is weighted 40x roll and pitch,
#: because roll and pitch are gravity-stabilised and yaw is not.  Height is
#: 25x horizontal position, because the horizontal reference is an INTEGRAL of
#: a commanded velocity and therefore has no physical meaning to track
#: tightly, while z is a real setpoint.  Roll and pitch RATE are weighted zero
#: -- the angles are what matter, and weighting both fights the same error
#: twice.
Q_DIAG = np.array([
    0.25, 0.25, 10.0,        # roll, pitch, yaw
    2.0, 2.0, 50.0,          # x, y, z
    0.0, 0.0, 0.30,          # wx, wy, wz
    0.20, 0.20, 0.10,        # vx, vy, vz
    0.0,                     # g -- never tracked, it is a constant
])                                                   # [PAPER]

#: R, the force penalty.  One scalar for all twelve inputs.
#:
#: TINY ON PURPOSE.  Its job is not to discourage force -- the robot needs
#: 57.7 N of it just to stand -- but to make the Hessian positive DEFINITE.
#: The task cost alone is rank-deficient: with four feet planted there are 12
#: force variables and 6 wrench equations, so a 6-dimensional subspace of
#: internal forces costs nothing, and the QP would be free to return any
#: member of it.  1e-6 picks the minimum-norm one.
ALPHA_FORCE = 1.0e-6                                 # [PAPER Table I]
R_DIAG = np.full(INPUT_DIM, ALPHA_FORCE)

# ===========================================================================
# friction cone and force bounds
# ===========================================================================
#: Tangential friction limit, as a LINEARISED PYRAMID rather than the true
#: cone -- four half-space rows per foot.  That linearisation is the whole
#: reason this problem is a QP: the true second-order cone constraint would
#: make it an SOCP, which no 20 Hz loop is going to solve.  The pyramid is
#: inscribed, so it is conservative: it never permits a force the real cone
#: forbids.                                                      [PAPER]
MU = 0.6                                             # [PAPER Table I]

#: A planted foot is never allowed to unload completely.  Zero would let the
#: QP silently drop a foot the gait says is in stance, and the swing law would
#: then be asked to catch a robot it is not holding.
FZ_MIN = 1.0                                         # N   [TUNED]

#: Ceiling on a single foot's normal force: about 2x the whole robot's weight,
#: so one diagonal can carry everything with margin for a push.
#:
#: IT IS NOT A TORQUE LIMIT, AND THE DIFFERENCE MATTERS.  The MPC has no model
#: of the legs -- its input is a force at a contact point, and it does not know
#: that producing 120 N horizontally at the 0.18 m hip-to-foot lever would ask
#: for 21 N*m from a motor that saturates at 9.94.  The torque limit is applied
#: downstream, by the controller, AFTER the QP.  A force the QP returns and the
#: joints then clip is a force the plan assumed and did not get; this is the
#: known weak seam in the method, not a bug in this implementation.
FZ_MAX = 2.0 * P.WEIGHT                              # 115.4 N   [TUNED]

# ===========================================================================
# reference trajectory
# ===========================================================================
#: One keypress of the operator's stick.  W/S drive x-velocity, A/D drive
#: y-velocity, Q/E drive yaw rate.
V_STEP = 0.1                                         # m/s per press
YAW_RATE_STEP = np.deg2rad(20.0)                     # rad/s per press

#: Ceilings on what the operator can ask for.
V_MAX = 0.6                                          # m/s
YAW_RATE_MAX = np.deg2rad(90.0)                      # rad/s

#: Commanded trunk height.  The CAD's own drawn stance -- see params.
Z_REF = P.STAND_HEIGHT                               # 0.192535 m

# ===========================================================================
# swing leg
# ===========================================================================
#: Use the HIP's velocity in the foot-placement law instead of the CoM's.
#:
#: FALSE, WHICH IS EQUATION (33) LITERALLY, BECAUSE THE CORRECTION WAS
#: MEASURED AND BOUGHT NOTHING.  Setting it True is the refinement the Cheetah
#: implementation makes: under a yaw rate the hip sits 0.17 m off the turn
#: axis and moves at w x r_hip even when the CoM is still, so "put the foot
#: where the hip will be at mid-stance" wants the hip's velocity, not the
#: CoM's.  At 90 deg/s that shifts the target by 33 mm, which is not a small
#: correction.
#:
#: It makes no measurable difference on DOG6.  Over 10 s at 20/40/60/90 deg/s
#: the achieved yaw is 95-96 % of commanded either way, and roll stays under
#: 0.3 deg rms either way.  The reason is worth stating, because it is the
#: paper's own argument showing up in the data: the MPC has yaw-moment
#: authority over a whole horizon, so a 33 mm placement error is something it
#: simply absorbs into the wrench it was already planning.  A reactive
#: controller would not have that slack.
#:
#: This flag was added while chasing a turn that diverged above 20 deg/s.  The
#: divergence was NOT the placement law -- it was the yaw branch cut in
#: `controller.solve` -- and the correction is kept only because it is a real
#: refinement and the measurement above is worth being able to repeat.
YAW_PLACEMENT_CORRECTION = False                     # [PAPER: eq (33) as written]

#: Apex of the swing arc above the foot's liftoff height.
SWING_HEIGHT = 0.030                                 # m   [DOG6]

#: Cartesian impedance at the foot, in BODY coordinates -- the frame the swing
#: reference is generated in, so no rotation sits between error and gain.
#:
#: ISOTROPIC, AND THE PREDECESSOR'S WAS NOT.  DOG6's earlier quasi-static trot
#: needed diag(400, 400, 5000) -- a factor of 12 between horizontal and
#: vertical -- because the foot's apparent mass IS anisotropic by roughly that
#: factor (0.46 kg horizontally, 4.54 kg vertically) and a fixed-gain
#: impedance has no way to know.  Here the Lambda feedforward supplies exactly
#: that mass, so the gain no longer has to encode it, and one number works in
#: all three axes.  That cancellation is a claim this reproduction can check,
#: and `selftest` does.
KP_SWING = np.diag([500.0, 500.0, 500.0])            # N/m   [TUNED]
KD_SWING = np.diag([20.0, 20.0, 20.0])               # N s/m [TUNED]

#: A joint-space PD floor under the stance legs.  A pure force law is
#: velocity-level: it says nothing about where the joint should BE, so a leg
#: that loses its contact integrates away without bound.  Small enough not to
#: fight the MPC's forces.
KP_JOINT = 5.0                                       # N*m/rad   [TUNED]
KD_JOINT = 0.2                                       # N*m*s/rad [TUNED]

# ===========================================================================
# limits applied after the QP
# ===========================================================================
#: Per-joint torque clamp.  `params.TAU_MAX_SIM` is 8.0 N*m, the motor's 9.94
#: saturation with 20 % held back -- A SIMULATION NUMBER that comes down to a
#: staging cap before DOG6 is ever powered.
TAU_MAX = P.TAU_MAX_SIM                              # 8.0 N*m   [DOG6]


def describe() -> str:
    return "\n".join([
        "DOG6 convex MPC configuration",
        "  state           %d (12 observed + gravity)" % STATE_DIM,
        "  input           %d (4 feet x 3 force components, world)" % INPUT_DIM,
        "  MPC             %.0f Hz, horizon %d x %.3f s = %.2f s"
        % (MPC_HZ, HORIZON, MPC_DT, HORIZON * MPC_DT),
        "  IMU             %.0f Hz -- R arrives, fb = R' f refreshed and held"
        % IMU_HZ,
        "  encoder         %.0f Hz -- q arrives, tau = J' fb written"
        % CONTROL_HZ,
        "  sweeps/solve    %.2f MPC, %.2f IMU -- neither divides, both alternate"
        % (SWEEPS_PER_SOLVE, CONTROL_HZ / IMU_HZ),
        "  worst R age     %.0f ms when a torque is written"
        % (1000 * 2 * CONTROL_DT),
        "  gait            trot, T %.2f s, duty %.2f (stance %.3f / swing %.3f s)"
        % (GAIT_PERIOD, DUTY, STANCE_DURATION, SWING_DURATION),
        "  horizon covers  %.1f gait periods" % (HORIZON * MPC_DT / GAIT_PERIOD),
        "  friction        mu %.2f, fz in [%.1f, %.1f] N" % (MU, FZ_MIN, FZ_MAX),
        "  force weight    %.0e" % ALPHA_FORCE,
        "  operator        %.2f m/s and %.0f deg/s per press"
        % (V_STEP, np.rad2deg(YAW_RATE_STEP)),
        "  swing           %.0f mm arc, Kp %.0f N/m, Kd %.0f N s/m"
        % (1000 * SWING_HEIGHT, KP_SWING[0, 0], KD_SWING[0, 0]),
        "  torque clamp    %.1f N*m" % TAU_MAX,
    ])


if __name__ == "__main__":
    print(describe())
