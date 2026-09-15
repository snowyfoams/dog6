"""DOG6 mechanical parameters.  Numbers only -- no frames, no maps, no logic.

WHERE THESE COME FROM, AND WHICH ONES YOU CAN TRUST
    Three different kinds of number live in this file and they do NOT deserve
    the same confidence.  Each block says which it is, and every one that is
    not a measurement of DOG6 says so in capitals.

    [CAD]        measured off the Fusion 360 design `quadruped_robot` and
                 carried into model/dog6.xml by the build pipeline.  These are
                 exact to the CAD and `selftest` gates every one of them
                 against the MJCF, so they cannot drift.

    [SIM]        a property of the simulation rather than of the robot --
                 timestep, integrator, contact classes.  Read off
                 model/dog6.xml; changing one here changes nothing unless the
                 XML changes with it, which is why `selftest` checks them.

    [INHERITED]  DOG5's value, adopted because DOG6 has the same motors and
                 nobody has powered DOG6.  NOT A MEASUREMENT OF THIS ROBOT.
                 Every one of these has to be re-established on hardware
                 before DOG6 moves under its own torque.

    model/dog6.xml is a generated artifact.  The generator (`build_dog6_mjcf.py`
    plus `dog6_raw.json`) deliberately stays in D:\\mujoco\\dog6_description --
    when the CAD moves, rebuild there and re-copy the XML, meshes and
    robot_export.json into model/, then run `python -m sim.selftest`, which is
    what will tell you if a number in this file went stale.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):        # allow `python sim/params.py` too
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402

# ===========================================================================
# where the model lives
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "model")
XML_PATH = os.path.join(MODEL_DIR, "dog6.xml")
MESH_DIR = os.path.join(MODEL_DIR, "meshes")
EXPORT_JSON = os.path.join(MODEL_DIR, "robot_export.json")


def load_export() -> dict:
    """model/robot_export.json -- the build pipeline's machine-readable summary.

    Not used by anything in this package (every number is written out below so
    it can be read and commented), but it is the artifact to diff against
    after a CAD rebuild.
    """
    with open(EXPORT_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# leg geometry  [CAD]
# ===========================================================================
@dataclass(frozen=True)
class LegGeometry:
    """The four fixed translations of one three-hinge leg.

    Each vector is expressed in the frame of the body it starts from, which is
    exactly how MuJoCo stores a body `pos`.  So these ARE the `pos` attributes
    in model/dog6.xml, and `selftest` reads them back out to prove it.
    """

    hip: tuple[float, float, float]             # trunk  -> hip hinge
    hip_to_pitch: tuple[float, float, float]    # hip    -> pitch hinge
    pitch_to_knee: tuple[float, float, float]   # thigh  -> knee hinge
    knee_to_foot: tuple[float, float, float]    # shin   -> foot site


#: THE ABDUCTION AND PITCH AXES INTERSECT, AND THAT IS THE REAL DIFFERENCE
#: FROM DOG5.  `hip_to_pitch` is a pure axial offset -- 28.19 mm along the
#: leg's own x with no z term at all.  DOG5's carried a -52.9 mm z offset, so
#: its hip was a dogleg and DOG6's is a plain elbow.  Every "generic quadruped"
#: formula that assumes a planar hip/knee chain perpendicular to the abduction
#: axis is closer to true on DOG6 than it was on DOG5 -- but not exactly true,
#: because `knee_to_foot` still carries a 5 mm out-of-plane z.  Use the full
#: vectors; the scalar lengths below are for sizing arguments only.
LEG_GEOMETRY = {
    "FL": LegGeometry(
        hip=(0.1563145029, 0.0600000000, 0.0000000000),
        hip_to_pitch=(0.0281854971, 0.0000000000, 0.0000000000),
        pitch_to_knee=(0.1000000000, 0.0000000000, 0.0000000000),
        knee_to_foot=(0.1050000000, 0.0000000000, +0.0050000000),
    ),
    "FR": LegGeometry(
        hip=(0.1563145029, -0.0600000000, 0.0000000000),
        hip_to_pitch=(0.0281854971, 0.0000000000, 0.0000000000),
        pitch_to_knee=(0.1000000000, 0.0000000000, 0.0000000000),
        knee_to_foot=(0.1050000000, 0.0000000000, +0.0050000000),
    ),
    "RL": LegGeometry(
        hip=(-0.1563145029, 0.0600000000, 0.0000000000),
        hip_to_pitch=(-0.0281854971, 0.0000000000, 0.0000000000),
        pitch_to_knee=(-0.1000000000, 0.0000000000, 0.0000000000),
        knee_to_foot=(-0.1050000000, 0.0000000000, +0.0050000000),
    ),
    "RR": LegGeometry(
        hip=(-0.1563145029, -0.0600000000, 0.0000000000),
        hip_to_pitch=(-0.0281854971, 0.0000000000, 0.0000000000),
        pitch_to_knee=(-0.1000000000, 0.0000000000, 0.0000000000),
        knee_to_foot=(-0.1050000000, 0.0000000000, +0.0050000000),
    ),
}

# The same table as (4, 3) arrays in C.LEGS order, which is what vectorised
# code wants.  One source, two shapes -- never edit these.
HIP_OFFSET = np.array([LEG_GEOMETRY[leg].hip for leg in C.LEGS])
HIP_TO_PITCH = np.array([LEG_GEOMETRY[leg].hip_to_pitch for leg in C.LEGS])
PITCH_TO_KNEE = np.array([LEG_GEOMETRY[leg].pitch_to_knee for leg in C.LEGS])
KNEE_TO_FOOT = np.array([LEG_GEOMETRY[leg].knee_to_foot for leg in C.LEGS])

#: Scalar link lengths, for sizing arguments.  L3 is the NORM, so it is
#: slightly longer than the 105 mm shin: the foot ball sits 5 mm off the leg
#: plane.  sqrt(105^2 + 5^2) = 105.119 mm.
L1 = float(np.linalg.norm(HIP_TO_PITCH[0]))     # 0.0281855 m, hip elbow
L2 = float(np.linalg.norm(PITCH_TO_KNEE[0]))    # 0.1000000 m, thigh
L3 = float(np.linalg.norm(KNEE_TO_FOOT[0]))     # 0.1051190 m, shin + foot
LEG_REACH = L1 + L2 + L3                        # 0.2333045 m, hip to foot

#: Hip rectangle: how far apart the four abduction hinges are.
HIP_SPACING_X = 2.0 * abs(HIP_OFFSET[0, 0])     # 0.3126290 m, front to rear
HIP_SPACING_Y = 2.0 * abs(HIP_OFFSET[0, 1])     # 0.1200000 m, left to right


# ===========================================================================
# mass properties  [CAD]
# ===========================================================================
@dataclass(frozen=True)
class LinkInertial:
    """One link's mass, CoM and full inertia tensor, in its OWN body frame.

    `inertia` is (ixx, iyy, izz, ixy, ixz, iyz) about the link's own CoM --
    MuJoCo's `fullinertia` ordering, so it round-trips through the XML.

    THE OFF-DIAGONAL TERMS ARE CARRIED, NOT DROPPED, AND DOG5 DROPPED THEM.
    DOG5's config discarded the twelve link tensors on the grounds that they
    were 0.46 % of its whole-robot Ixx.  That argument does not survive the
    move: on DOG6 the same tensors are 7.2 % of Ixx, because DOG6 is compact
    enough that the parallel-axis terms they compete with are three times
    smaller, and because twelve 330 g gearmotors live inside them.
    """

    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float] = (0.0,) * 6

    def tensor(self) -> np.ndarray:
        """The 3x3 symmetric tensor about this link's own CoM."""
        xx, yy, zz, xy, xz, yz = self.inertia
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


# Controlled copy of model/dog6.xml's hip/thigh/shin <inertial> elements.
# Order within each tuple is (hip, thigh, shin) -- outwards along the chain.
LINK_INERTIALS = {
    "FL": (
        LinkInertial(0.399904629786, (+0.025739067229, -0.000044016023, -0.002309584619),
                     (+1.590352985944e-04, +1.902866963400e-04, +2.261190839071e-04,
                      -2.049390574023e-07, -3.441477371965e-06, -3.004433070487e-07)),
        LinkInertial(0.411841894399, (+0.089984265439, -0.000001451576, +0.008400952548),
                     (+1.708286050670e-04, +4.833303907493e-04, +5.084736358936e-04,
                      -5.698614586491e-09, +7.775926612845e-05, -8.615475209372e-09)),
        LinkInertial(0.071702202369, (+0.064049266667, +0.000000000000, -0.007204167227),
                     (+1.658087231558e-05, +1.150125288444e-04, +1.091813237385e-04,
                      +0.000000000000e+00, -2.449740711161e-05, +0.000000000000e+00)),
    ),
    "FR": (
        LinkInertial(0.399904668289, (+0.025739064536, +0.000044015998, -0.002309583773),
                     (+1.590353085243e-04, +1.902867328770e-04, +2.261191179403e-04,
                      +2.049385425013e-07, -3.441467235851e-06, +3.004441801760e-07)),
        LinkInertial(0.411841959654, (+0.089984251011, +0.000001451913, +0.008400954958),
                     (+1.708286356661e-04, +4.833309563707e-04, +5.084742010822e-04,
                      +5.710045361250e-09, +7.775935668294e-05, +8.612996056097e-09)),
        LinkInertial(0.071702202369, (+0.064049266667, +0.000000000000, -0.007204167227),
                     (+1.658087231557e-05, +1.150125288444e-04, +1.091813237385e-04,
                      +0.000000000000e+00, -2.449740711161e-05, +0.000000000000e+00)),
    ),
    "RL": (
        LinkInertial(0.399904668289, (-0.025739064536, -0.000044015998, -0.002309583773),
                     (+1.590353085243e-04, +1.902867328770e-04, +2.261191179403e-04,
                      +2.049385425013e-07, +3.441467235851e-06, -3.004441801760e-07)),
        LinkInertial(0.411841959654, (-0.089984251011, -0.000001451913, +0.008400954958),
                     (+1.708286356661e-04, +4.833309563707e-04, +5.084742010822e-04,
                      +5.710045362639e-09, -7.775935668294e-05, -8.612996055928e-09)),
        LinkInertial(0.071702202369, (-0.064049266667, +0.000000000000, -0.007204167227),
                     (+1.658087231557e-05, +1.150125288444e-04, +1.091813237385e-04,
                      +0.000000000000e+00, +2.449740711161e-05, +0.000000000000e+00)),
    ),
    "RR": (
        LinkInertial(0.399904629786, (-0.025739067229, +0.000044016023, -0.002309584619),
                     (+1.590352985944e-04, +1.902866963400e-04, +2.261190839071e-04,
                      -2.049390574023e-07, +3.441477371965e-06, +3.004433070487e-07)),
        LinkInertial(0.411841894399, (-0.089984265439, +0.000001451576, +0.008400952548),
                     (+1.708286050670e-04, +4.833303907493e-04, +5.084736358936e-04,
                      -5.698614586491e-09, -7.775926612845e-05, +8.615475209372e-09)),
        LinkInertial(0.071702202369, (-0.064049266667, +0.000000000000, -0.007204167227),
                     (+1.658087231558e-05, +1.150125288444e-04, +1.091813237385e-04,
                      +0.000000000000e+00, +2.449740711161e-05, +0.000000000000e+00)),
    ),
}

#: (4, 3) views of the same table.
LINK_MASS = np.array([[li.mass for li in LINK_INERTIALS[leg]] for leg in C.LEGS])
LINK_COM = np.array([[li.com for li in LINK_INERTIALS[leg]] for leg in C.LEGS])
LINK_INERTIA = np.array([[li.tensor() for li in LINK_INERTIALS[leg]] for leg in C.LEGS])

TRUNK_MASS = 2.349038942902
TRUNK_COM = np.array([+0.000000000000, +0.000071184208, -0.001423902677])
TRUNK_INERTIA = np.array([
    [+8.989714647121e-03, +0.000000000000e+00, +0.000000000000e+00],
    [+0.000000000000e+00, +2.759694070216e-02, -5.672224097576e-06],
    [+0.000000000000e+00, -5.672224097576e-06, +3.468717042051e-02],
])

#: Whole-robot mass.  THE TRUNK IS ONLY 40 % OF IT -- DOG5's was 45 % -- so
#: the four legs are 60 % of DOG6, hanging ~0.10 m below the trunk CoM.  That
#: lever is why the assembled robot's Ixx (0.0362) is four times the trunk's
#: own (0.0090), and why `kinematics.body_inertia` has to be evaluated at a
#: pose rather than read off the trunk.
MASS = 5.882834056637
LEG_MASS = float(LINK_MASS[0].sum())            # 0.8834 kg, one leg
TRUNK_MASS_FRACTION = TRUNK_MASS / MASS         # 0.399

GRAVITY = 9.81
WEIGHT = MASS * GRAVITY                         # 57.71 N


# ===========================================================================
# the ground, and the heights that are not each other  [CAD]
# ===========================================================================
#: Foot ball radius.  model/dog6.xml puts a sphere of this radius centred on
#: the foot SITE, so a planted foot has its site exactly one radius up.
#: DOG5's was 20 mm.
FOOT_RADIUS = 0.015

#: Trunk-origin height at Q_STAND with the balls on the floor.  NOT A CHOICE
#: ON DOG6 -- it is the height the CAD is drawn at.  DOG5 had to pick one.
STAND_HEIGHT = 0.192535208

#: Trunk-origin height at the flat zero, robot on its belly.  Half the trunk
#: box: the collision box is 70 mm tall, so its bottom is on the floor.
HOME_HEIGHT = 0.03501

#: LEG EXTENSION AT THE DRAWN STANCE, AND THE NUMBER THIS ROBOT EXISTS FOR.
#: DOG6's foot sits 25.7 mm ahead of and 5 mm outboard of its hip, 177.5 mm
#: below it -- 179.4 mm from the hip, which is 76.9 % of LEG_REACH.  DOG5 at
#: its stand height used 92.3 %, because its flat-zero leg had to abduct 90 deg
#: and then reach 120 mm FORWARD to touch the floor.  A leg at 92 % extension
#: has almost no room left for a step; a leg at 77 % has 107 mm.  Everything
#: DOG6 can do that DOG5 could not comes from this line.
#:
#: Computed in `kinematics`, not here, because it is a function of Q_STAND and
#: the chain rather than a measurement.  Named here so it is findable.
#:     kinematics.reach_used()   -> 0.769
#:     kinematics.reach_room()   -> 0.1069 m along +x


# ===========================================================================
# actuators  [INHERITED -- NOT MEASURED ON DOG6]
# ===========================================================================
# Twelve MG5010-i10 gearmotors, one per joint, same part DOG5 uses.  DOG5's
# numbers came from runners that had actually driven them; DOG6 has never been
# powered, so everything in this block is inherited on the strength of it being
# the same motor, not on any DOG6 observation.
MOTOR_MODEL = "MG5010-i10"
MOTOR_MASS = 0.330                  # kg each; 12 of them are 3.96 kg, 67 % of
                                    # the robot.  They are inside the link
                                    # inertials above, not added on top.
GEAR_RATIO = 10.0                   # 10:1 planetary

#: Rotor inertia on the MOTOR side, before the gearbox.
ROTOR_INERTIA_MOTOR_SIDE = 8.5e-5   # kg*m^2

#: ...and reflected to the JOINT side, which is rotor inertia * ratio^2.  This
#: is model/dog6.xml's `<default><joint armature=...>`.
#:
#: IT DOMINATES THE LEG, WHICH IS NOT AN INTUITIVE PLACE TO END UP.  The shin
#: weighs 72 g and has 9.3e-5 kg*m^2 about the knee, so the reflected rotor is
#: 91x the link it drives.  Measured off MuJoCo's own mass matrix at Q_STAND,
#: the foot's apparent VERTICAL mass is 4.54 kg with the armature and 0.449 kg
#: without -- a factor of ten, because at that pose only the knee has vertical
#: authority (J_z = 52.5 mm/rad) and 0.0085/0.0525^2 = 3.1 kg appears through
#: that lever.  A swing-foot impedance gain sized for the links is wrong by
#: that factor.  Without the armature the knee is also numerically unstable at
#: every sampled damping.
ARMATURE = ROTOR_INERTIA_MOTOR_SIDE * GEAR_RATIO ** 2    # 0.0085 kg*m^2

#: Joint viscous damping in model/dog6.xml's joint default.  [SIM]
JOINT_DAMPING = 0.1                 # N*m*s/rad

#: Where the current loop saturates: iq counts 2048 / 206.04 counts-per-Nm.
#: A hard ceiling of the hardware, not a policy.
TAU_SATURATION = 9.94               # N*m

#: What simulation is allowed to command -- the saturation with 20 % held back.
#:
#: IT IS A SIMULATION NUMBER AND IT MUST COME BACK DOWN BEFORE HARDWARE.
#: DOG5 ran a deliberately low 3.0 N*m staging cap while it was being
#: commissioned, and raised it one logged run at a time.  DOG6 gets the same
#: treatment: when the robot is assembled this drops to a staging value and
#: climbs from there.  3.0 would not even be sufficient here -- one diagonal
#: carrying 58 N at the 0.180 m hip-to-foot lever needs about 5.2 N*m at the
#: pitch joint at mid-stance, so a 3.0 clamp would saturate through every
#: handover and a planner's solution would never reach the joints.
TAU_MAX_SIM = 8.0                   # N*m per joint

#: Overspeed gate.  A joint faster than this is a fault, not a fast move.
QD_ESTOP = 7.0                      # rad/s


# ===========================================================================
# software joint limits  [INHERITED -- NOT MEASURED ON DOG6]
# ===========================================================================
# MODEL/DOG6.XML CARRIES NO JOINT RANGES AT ALL.  Every hinge in the MJCF is
# unlimited, which `selftest` asserts rather than assumes -- so these limits
# are NOT enforced by the simulator.  They are a software policy that a
# controller has to apply itself, and a sim run will happily drive straight
# through them if nothing checks.
#
# ABD_LIM IS 2.2, NOT DOG5's 1.75.  DOG6 stands at hip_abd = +-pi/2 = 1.571,
# so a 1.75 rad stop leaves the standing pose 0.18 rad from a limit that a
# trot's abduction ripple would reach.  2.2 puts the stance in the middle of
# the range instead of against its edge.
ABD_LIM = 2.2                       # rad, symmetric about 0
PITCH_LIM = 2.6
KNEE_LIM = 2.6

#: (3, 2) of (lower, upper) for one leg's three joints, in chain order.
JOINT_LIMITS = np.array([
    [-ABD_LIM, +ABD_LIM],
    [-PITCH_LIM, +PITCH_LIM],
    [-KNEE_LIM, +KNEE_LIM],
])


def clamp_q(q) -> np.ndarray:
    """Clip a (3,) or (4, 3) joint pose into JOINT_LIMITS."""
    q = np.asarray(q, dtype=float)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    return np.clip(q, lo, hi)


def within_limits(q) -> bool:
    """True if every joint of a (3,) or (4, 3) pose is inside JOINT_LIMITS."""
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1]))


# ===========================================================================
# collision and simulation setup  [SIM]
# ===========================================================================
# Carried from DOG5 for the reasons DOG5 found them.
#
# LINK MESHES ARE VISUAL ONLY.  MuJoCo collides convex HULLS, and the hulls of
# these links produce phantom self-collisions as the legs fold -- a thigh hull
# and a shin hull overlap long before the real parts do.  Collision is a trunk
# box plus four foot spheres and nothing else.
#
# The contact classes are what keep that honest: robot collision geoms are
# contype=2 conaffinity=1 and the floor is 1/1, so a robot geom meets the floor
# (2 & 1 nonzero) and never another robot geom (2 & 2 shares no bit).  Self
# collision is structurally impossible rather than merely absent.
FLOOR_CONTYPE, FLOOR_CONAFFINITY = 1, 1
ROBOT_CONTYPE, ROBOT_CONAFFINITY = 2, 1

#: Trunk collision box half-extents.  2 * 0.03501 = 70.02 mm tall, which is
#: what puts the trunk origin at HOME_HEIGHT with the box resting on the floor.
TRUNK_BOX_HALF = np.array([0.11001, 0.09501, 0.03501])

TIMESTEP = 0.002                    # s, model/dog6.xml <option timestep=...>
INTEGRATOR = "implicitfast"

#: What a controller sweep costs.  DOG5 ran 300 Hz against its single CAN bus;
#: 250 is what DOG6's demos use so both control backends see the same plant.
CONTROL_HZ = 250.0
CONTROL_DT = 1.0 / CONTROL_HZ


def describe() -> str:
    return "\n".join([
        "DOG6 mechanical parameters",
        "  mass            %.4f kg   trunk %.4f (%.0f%%), leg %.4f x4"
        % (MASS, TRUNK_MASS, 100 * TRUNK_MASS_FRACTION, LEG_MASS),
        "  links           hip %.4f  thigh %.4f  shin %.4f kg"
        % tuple(LINK_MASS[0]),
        "  hip rectangle   %.4f x %.4f m" % (HIP_SPACING_X, HIP_SPACING_Y),
        "  link lengths    elbow %.4f  thigh %.4f  shin %.4f m" % (L1, L2, L3),
        "  leg reach       %.4f m" % LEG_REACH,
        "  foot radius     %.3f m" % FOOT_RADIUS,
        "  heights         stand %.6f   home %.5f m" % (STAND_HEIGHT, HOME_HEIGHT),
        "  actuator        12 x %s, %.0f:1, armature %.4f kg m^2"
        % (MOTOR_MODEL, GEAR_RATIO, ARMATURE),
        "  torque          sat %.2f, sim cap %.1f N*m   [INHERITED]"
        % (TAU_SATURATION, TAU_MAX_SIM),
        "  joint limits    abd %.1f  pitch %.1f  knee %.1f rad   [INHERITED,"
        " not in the XML]" % (ABD_LIM, PITCH_LIM, KNEE_LIM),
        "  sim             dt %.4f s, %s, control %.0f Hz"
        % (TIMESTEP, INTEGRATOR, CONTROL_HZ),
    ])


if __name__ == "__main__":
    print(describe())
