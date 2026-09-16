"""DOG6 frame, sign and pose conventions.  No mechanical numbers live here.

This module owns the ANSWERS TO "IN WHAT FRAME?", and nothing else.  Link
lengths, masses and limits are in `params`; the maps that use both are in
`kinematics`.  The split is deliberate: a frame convention is the thing that
gets silently violated, and it is much easier to gate when it is not mixed in
with a table of numbers.

Everything below is a convention carried over from DOG5 unchanged, except
where a comment says otherwise.  Carrying them means a DOG5 controller reads
DOG6 correctly, and -- more to the point -- that DOG5's calibration procedure,
which has actually been performed on a real robot, still describes DOG6.


THE TRUNK FRAME
    x forward, y left, z up.
    Origin at the centre of the abduction-axis plane -- the point equidistant
    from all four hip hinges, in the plane those four axes lie in.

    So the trunk origin is NOT the trunk's centre of mass and NOT the bottom
    of the shell.  Three heights get confused here and they are 35 mm apart:

        floor                     0
        foot SITE, planted        FOOT_RADIUS          0.015
        trunk bottom              STAND_HEIGHT-0.035   0.1575   <- a ruler
        trunk ORIGIN              STAND_HEIGHT         0.19254  <- z_ref, FK
        whole-robot CoM           see kinematics.body_inertia

    The trunk origin is the one the leg kinematics work in, so it is the one
    a height reference means.


THE FUSION FRAME, AND THE REMAP OUT OF IT
    The CAD (`quadruped_robot`) is drawn with x lateral, y fore/aft, z up, in
    centimetres, with the abduction axes lying in the plane z = 4 cm.  Stage 1
    of the model pipeline reads that frame verbatim into `dog6_raw.json`;
    stage 2 (`build_dog6_mjcf.py`, which stays in D:\\mujoco\\dog6_description)
    applies FUSION_TO_TRUNK below and writes the MJCF.

    Both stages live outside this project -- `model/dog6.xml` arrives here as a
    finished artifact.  The remap is recorded anyway, because it is the one
    transform that cannot be recovered by reading the MJCF, and because
    `fusion_to_trunk()` is what any future re-read of the CAD has to agree
    with.

    LEG NAMES ARE DERIVED FROM GEOMETRY, NOT FROM CAD OCCURRENCE NAMES.  A leg
    is FL because its hip lands at +x, +y AFTER the remap.  The Fusion
    occurrence that produced it is called `quadruped_leg_MIR:1`, and there are
    two occurrences with `_MIR` in the name on opposite sides.  Trusting those
    names would let a re-mirrored CAD swap two legs without a single number
    changing.


THE ZERO
    EVERY JOINT READS 0 WITH THE ROBOT FLAT ON ITS BELLY: all four legs fully
    extended fore/aft, horizontal, knees straight.

    This is a calibration convention, not a modelling one.  It is chosen
    because it is the only pose a human can actually put a robot into on a
    bench, one leg at a time, well enough to zero an encoder against -- the
    legs lie out straight and either touch the bench or do not.

    The price is that the zero is very nearly SINGULAR: at q = 0 a leg lies
    along its own abduction axis, so rotating the abduction joint moves the
    foot hardly at all and the Jacobian loses rank.  Nothing plans through it.
    Controllers start from Q_STAND, and `kinematics.leg_ik` seeds from Q_STAND
    rather than from zero for exactly this reason.

    DOG6's CAD is drawn STANDING, not flat, so the flat pose is a
    reconstruction -- `build_dog6_mjcf.py` rotates each link back onto it.
    Which way the leg plane folds down is otherwise a free 180 deg choice.

    IT IS FIXED BY THE BUILT ROBOT, AND THAT IS CONFIRMED.  At hip_abd = 0
    the thigh plate sits ON TOP OF its pitch motor, so the foot ball points
    UP and a leg laid out flat rests on its own face.  DOG6's twelve drivers
    are CALIBRATED to this zero and `hw.kinematics` is written in this frame
    -- both on the operator's word, checked against the bench, 2026-09-15.
    This is the one thing in this file that is not inherited from DOG5 and
    not merely read off the CAD: it is a statement about the built robot, and
    where it disagrees with the CAD pipeline, the robot wins.

    THIS IS A CORRECTION.  The choice used to be pinned to DOG5 instead, by
    DEFINING the standing pose to be hip_abd = +90 deg on the left legs
    (DOG5's `Q_ROLL` sign).  That folds every leg the other way and puts the
    thigh UNDER its motor at zero, 180 deg off the robot on the bench.

    THE FOLD IS THE ONLY THING THAT MOVES.  Rolling a leg 180 deg about trunk
    +x carries the whole chain: the pitch and knee axis LINES stay where they
    were, the chain's offsets all lie ON the roll axis and so are its fixed
    points, and at q = 0 every body frame still coincides with the trunk
    frame.  What the roll does change is which way "positive" points -- and
    DOG6 IS CALIBRATED SO THAT EVERY JOINT OBEYS THE RIGHT-HAND RULE about
    the axis as written in "THE JOINT AXES" below, reading +x, +z, +z in the
    TRUNK frame at the zero, not just in its own body frame.  Holding that
    fixed while the leg rolls negates all three columns at once:

        Q_STAND_new = -Q_STAND_old

    and the feet still land at (+-0.182, +-0.065, FOOT_RADIUS) with no
    residual.  NOTHING ELSE IN THIS FILE MOVES -- not JOINT_AXES, not
    `link_rotations`, not `hw.kinematics`'s own closed form -- because the
    roll lives in the MJCF's GEOMETRY (each leg body's geoms, sites and
    inertial) and not in its body frames.  That is what keeps the aligned
    zero, and with it `link_rotations` as a plain product with no offsets.

    THE REBUILD LANDED WITH THE CORRECTION, IN THE SAME COMMIT.  What it did,
    written out because `model/dog6.xml` is a generated artifact whose
    generator is not in this repository and this is the only record of it:
    each leg body's geoms, sites and inertial rolled 180 deg about x -- pos
    (x, y, z) -> (x, -y, -z), quat pre-multiplied by Rx(180 deg), the
    inertia's ixy and ixz negated -- and the `stand` keyframe's twelve joint
    entries negated.  Every body `pos`, every absent body `quat` and all
    twelve joint `axis` attributes stayed exactly as they were.  The same
    roll went through `params`, which copies the MJCF: every leg link's `com`
    y and z negated, every leg link inertia's ixy and ixz negated, and
    `knee_to_foot`'s z flipped from -0.005 to +0.005.  Masses, link lengths
    and the trunk's own inertial were untouched, and so was L3, which is a
    norm.

    THE GATE THAT SAYS SO IS `Q_STAND == the stand keyframe`, in `selftest`,
    and it is GREEN.  It is the one that has to be read, because the
    feet-on-the-floor gate beside it would pass either way: that one checks
    only z, and the old fold lands the foot ball 10 mm to the other side of
    the shin at the same height.  So a green `sim.selftest` is what says the
    MJCF, `params`, `kinematics` and `hw.kinematics` all agree on THIS zero
    -- and it is the only thing that says it.


THE JOINT AXES
    hip_abd     about trunk +x            for every leg, left and right alike
    hip_pitch   about the hip body's  +z  (i.e. +z after the abduction roll)
    knee        about the thigh body's +z

    All three are +z-or-+x in their PARENT frame, with no per-leg sign.  The
    left/right asymmetry lives entirely in the geometry (the hip offsets
    mirror) and in Q_STAND, never in an axis.  SIDE_SIGN below is the only
    place a left/right sign is written down, and it is used for reasoning
    about outward direction, not for building the chain.

    Because at the flat zero all four of a leg's frames are aligned with the
    trunk frame, the body rotations are a plain product of elementary
    rotations with no fixed offsets -- see `link_rotations`.  The corrected
    fold preserves that; see "THE ZERO" above for why.
"""
from __future__ import annotations

import numpy as np

# ===========================================================================
# naming, ordering, indexing
# ===========================================================================
#: Canonical leg order.  This is the MuJoCo joint order in model/dog6.xml, the
#: row order of every (4, 3) array in this package, and the order the hardware
#: CAN table will be written in, so `tau[i]` needs no permutation anywhere.
LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
N_LEGS = 4

#: The three joints of a leg, in chain order from the trunk outwards.
JOINTS: tuple[str, ...] = ("abd", "pitch", "knee")
N_JOINTS_PER_LEG = 3
N_JOINTS = N_LEGS * N_JOINTS_PER_LEG        # 12

#: leg name -> row index.
LEG_INDEX = {leg: i for i, leg in enumerate(LEGS)}

#: MuJoCo joint names, flattened in canonical order.  These are the literal
#: `<joint name=...>` strings in model/dog6.xml; `selftest` gates that.
JOINT_NAMES: tuple[str, ...] = tuple(
    f"{stem}_{leg}" for leg in LEGS
    for stem in ("hip_abd", "hip_pitch", "knee")
)

#: Foot site names in model/dog6.xml, canonical order.
FOOT_SITE_NAMES: tuple[str, ...] = tuple(f"foot_{leg}" for leg in LEGS)

#: MuJoCo body names, (4, 3) to match the joint layout.
BODY_NAMES: tuple[tuple[str, str, str], ...] = tuple(
    (f"hip_{leg}", f"thigh_{leg}", f"shin_{leg}") for leg in LEGS
)

TRUNK_BODY = "trunk"
IMU_SITE = "imu"
IMU_SENSORS: tuple[str, ...] = ("imu_quat", "imu_gyro", "imu_acc")

#: JOINT_INDEX[leg] are the three entries of a flat 12-vector belonging to that
#: leg.  It is 3*leg + (0, 1, 2) everywhere, written out rather than recomputed
#: so it can be checked against a CAN id table at a glance.
JOINT_INDEX = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=int)

#: +1 for a left leg, -1 for a right one.  The abduction hinge is about trunk
#: +x for EVERY leg, so this is not an axis sign -- it is the sign of the
#: OUTWARD direction, used when reasoning about a foot crossing under the body.
SIDE_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])

#: +1 for a front leg, -1 for a rear one.
END_SIGN = np.array([+1.0, +1.0, -1.0, -1.0])


# ===========================================================================
# the MuJoCo state layout
# ===========================================================================
# model/dog6.xml is a floating-base model: one freejoint then twelve hinges.
# qpos is 19 and qvel is 18, and the two do NOT line up -- the free joint takes
# 7 of qpos (3 position + 4 quaternion) but 6 of qvel (3 linear + 3 angular).
# Slicing qvel at 7 is the classic version of this mistake, and it silently
# reads each joint velocity one slot late.
NQ = 19
NV = 18

QPOS_ROOT_POS = slice(0, 3)
QPOS_ROOT_QUAT = slice(3, 7)        # MuJoCo order: w, x, y, z
QPOS_JOINTS = slice(7, 19)

QVEL_ROOT_LIN = slice(0, 3)
QVEL_ROOT_ANG = slice(3, 6)
QVEL_JOINTS = slice(6, 18)


def q_from_qpos(qpos) -> np.ndarray:
    """The twelve joint angles out of a MuJoCo qpos, as (4, 3) in LEGS order."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape[-1] != NQ:
        raise ValueError(f"qpos must have {NQ} entries, got {qpos.shape[-1]}")
    return qpos[QPOS_JOINTS].reshape(N_LEGS, N_JOINTS_PER_LEG).copy()


def qpos_from_q(q, root_pos=(0.0, 0.0, 0.0), root_quat=(1.0, 0.0, 0.0, 0.0)):
    """Build a full 19-vector qpos from a (4, 3) joint pose and a base pose."""
    q = np.asarray(q, dtype=float).reshape(N_LEGS, N_JOINTS_PER_LEG)
    qpos = np.zeros(NQ)
    qpos[QPOS_ROOT_POS] = np.asarray(root_pos, dtype=float)
    qpos[QPOS_ROOT_QUAT] = np.asarray(root_quat, dtype=float)
    qpos[QPOS_JOINTS] = q.reshape(-1)
    return qpos


def flat(q) -> np.ndarray:
    """(4, 3) -> (12,) in canonical order."""
    return np.asarray(q, dtype=float).reshape(N_LEGS, N_JOINTS_PER_LEG).reshape(-1)


def unflat(q) -> np.ndarray:
    """(12,) -> (4, 3) in canonical order."""
    return np.asarray(q, dtype=float).reshape(N_LEGS, N_JOINTS_PER_LEG)


# ===========================================================================
# elementary rotations and the leg frame chain
# ===========================================================================
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

#: The hinge axis of each joint, expressed in its OWN body frame.  Identical
#: for all four legs -- see the module docstring.
JOINT_AXES = (X_AXIS, Z_AXIS, Z_AXIS)


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R_{world <- trunk} from the ZYX (yaw-pitch-roll) triple, RADIANS.

    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, so ``v_world = R @ v_trunk``.  On
    FLU axes the right-hand rule fixes the signs, and they are worth writing
    down because they are what a physical tilt test has to reproduce:

        roll  > 0   right side DOWN, left side rises
        pitch > 0   nose DOWN     <- the FLU consequence, and the OPPOSITE of
        yaw   > 0   nose LEFT        the aeronautical NED convention the
                                     DETA10 reports in.  `hw.imu` flips it.

    This is the parameterisation `hw.imu` hands the balance controller.  The
    control law carries R and never the triple; the triple exists for the tilt
    trip, the status line and the log.
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def zyx_from_rot(R) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in radians from R.  The inverse of `rot_zyx`.

    Degenerate at pitch = +-90 deg, where roll and yaw stop being separable.
    A standing quadruped is nowhere near it -- the tilt trip that reads this
    fires at a tenth of the way there -- so no branch is taken for it.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    pitch = float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def log_so3(R) -> np.ndarray:
    """The rotation vector of R: ``log(R)^vee``, axis times angle.  Shape (3,).

    THE ATTITUDE ERROR IS FORMED WITH THIS, NOT BY SUBTRACTING EULER ANGLES.
    An Euler difference is not a vector: it does not compose, it is not
    frame-covariant, and near a gimbal it is not even continuous.  The log map
    is the exact axis-angle of the residual rotation, which is the thing a
    moment can be produced about.

    Three branches, and the two outer ones are not optimisations.  At
    theta -> 0 the ``1 / (2 sin theta)`` factor is 0/0 and the antisymmetric
    part alone IS the answer to second order.  At theta -> pi, sin theta
    collapses again and the axis has to be recovered from R + I instead, where
    the antisymmetric part has gone to zero and carries no axis at all.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    theta = float(np.arccos(min(1.0, max(-1.0, 0.5 * (float(np.trace(R)) - 1.0)))))
    antisym = np.array([R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]])
    if theta < 1.0e-6:
        return 0.5 * antisym
    if theta > np.pi - 1.0e-6:
        k = int(np.argmax(np.diag(R)))
        column = R[:, k] + np.eye(3)[:, k]
        return theta * (column / np.linalg.norm(column))
    return (theta / (2.0 * np.sin(theta))) * antisym


def link_rotations(q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R_{trunk <- hip}, R_{trunk <- thigh}, R_{trunk <- shin} for one leg.

    `q` is that leg's (abd, pitch, knee) in radians.  These are the rotations
    that carry a vector in a link's own body frame into the trunk frame -- so
    a link inertia tensor is carried by ``R @ I @ R.T``, and a link CoM offset
    by ``origin + R @ com``.

    No fixed offsets appear: at the flat zero every body frame is aligned with
    the trunk frame, which is the whole point of that zero.  The same fact is
    what lets model/dog6.xml give every body a `pos` and no `quat`.
    """
    q_abd, q_pitch, q_knee = (float(v) for v in np.asarray(q, dtype=float).reshape(3))
    r_hip = rot_x(q_abd)
    r_thigh = r_hip @ rot_z(q_pitch)
    r_shin = r_thigh @ rot_z(q_knee)
    return r_hip, r_thigh, r_shin


def joint_axes_trunk(q) -> np.ndarray:
    """The three hinge axes of one leg, in the TRUNK frame, stacked (3, 3).

    Row i is joint i's axis.  The abduction axis is constant (trunk +x); the
    other two rotate with everything above them.  This is what a Jacobian's
    cross products need.
    """
    r_hip, r_thigh, _ = link_rotations(q)
    return np.stack((X_AXIS, r_hip @ Z_AXIS, r_thigh @ Z_AXIS), axis=0)


# ===========================================================================
# the two poses that have names
# ===========================================================================
#: The calibration zero -- flat on the belly, every joint reading 0.  Present
#: as a named constant so nothing has to spell `np.zeros((4, 3))` and mean
#: something by it.  Also `model/dog6.xml`'s `home` keyframe.
Q_ZERO = np.zeros((N_LEGS, N_JOINTS_PER_LEG))

#: The pose the CAD is drawn in, and the stance every controller starts from.
#:
#: EXACT, NOT AN IK SOLVE.  It is the OLD DOG5-folded Q_STAND negated, every
#: column: abduction at -+90 deg (LEFT NEGATIVE) and pitch/knee at -+60 deg.
#: The sign is the opposite of DOG5's `Q_ROLL` throughout, because DOG6's
#: zero is fixed by DOG6's bench and its right-hand-rule calibration, not by
#: DOG5's convention -- see "THE ZERO" above.  Driving the chain to these exact values puts all
#: four foot balls precisely on the floor: the foot sites land at
#: (+-0.182, +-0.065, FOOT_RADIUS) with the trunk origin at STAND_HEIGHT,
#: with no residual.  `selftest` gates that against MuJoCo.
#:
#: The sign pattern still pairs FL with RR and FR with RL.  model/dog6.xml's
#: `stand` keyframe carries the OLD abduction signs and disagrees with this
#: array until the model is rebuilt -- again, see "THE ZERO".
Q_STAND = np.array([
    [-np.pi / 2, +np.pi / 3, +np.pi / 3],      # FL
    [+np.pi / 2, -np.pi / 3, -np.pi / 3],      # FR
    [-np.pi / 2, -np.pi / 3, -np.pi / 3],      # RL
    [+np.pi / 2, +np.pi / 3, +np.pi / 3],      # RR
])

#: The keyframes model/dog6.xml ships, name -> MJCF keyframe id.
KEYFRAMES = {"home": 0, "stand": 1}


# ===========================================================================
# the IMU
# ===========================================================================
#: R_{trunk <- imu}: carries a vector in the IMU's own axes into the trunk
#: frame.  The SAME rotation is baked into model/dog6.xml as the `imu` site's
#: orientation, so the simulated `imu_gyro` / `imu_acc` report in the IMU frame
#: and every consumer has to undo it -- which is the point.  A sim that reports
#: sensors already in body axes is a sim that will not catch a mounting error.
#:
#: IT IS THE IDENTITY, AND THAT IS A PLACEHOLDER, NOT A MEASUREMENT.  DOG6's
#: IMU has not been mounted.  When it is, this becomes the measured mounting
#: rotation and model/dog6.xml's imu site quat has to move with it: they are
#: one value with two homes, and `selftest` gates that they agree.
R_BODY_IMU = np.eye(3)


# ===========================================================================
# the Fusion 360 frame, and the way out of it
# ===========================================================================
#: R_{trunk <- fusion}.  x_trunk = -y_fusion, y_trunk = +x_fusion, z unchanged.
FUSION_TO_TRUNK = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

#: The trunk origin, in Fusion's frame and Fusion's units (cm): the centre of
#: the abduction-axis plane sits 4 cm up from the CAD origin.
FUSION_ORIGIN_CM = np.array([0.0, 0.0, 4.0])

#: Fusion 360's API works in centimetres.  Everything in this package is SI.
CM_TO_M = 0.01


def fusion_to_trunk(p_cm) -> np.ndarray:
    """A point in the Fusion design frame (cm) -> the trunk frame (m).

    This is the transform `build_dog6_mjcf.py` applies to every CAD number on
    its way into `model/dog6.xml`.  It is reproduced here so that a future
    re-read of the CAD has something to agree with, and so the convention is
    checkable without opening a file in another directory.
    """
    p_cm = np.asarray(p_cm, dtype=float)
    return (FUSION_TO_TRUNK @ (p_cm - FUSION_ORIGIN_CM).T).T * CM_TO_M


def fusion_tensor_to_trunk(inertia) -> np.ndarray:
    """A 3x3 inertia tensor in Fusion's axes -> the trunk axes.

    A rotation, not the translation: use the parallel-axis theorem separately
    if the reference point moves.  FUSION_TO_TRUNK is orthonormal with
    det = +1, so this is a similarity transform and the eigenvalues are
    unchanged -- which is the cheap check that a remap has not been botched.
    """
    inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
    return FUSION_TO_TRUNK @ inertia @ FUSION_TO_TRUNK.T


def describe() -> str:
    return "\n".join([
        "DOG6 coordinate conventions",
        "  trunk frame     x forward, y left, z up",
        "  origin          centre of the abduction-axis plane",
        "  zero            all 12 joints 0, robot flat on its belly",
        "  legs            %s" % (", ".join(LEGS),),
        "  joints/leg      %s" % (", ".join(JOINTS),),
        "  axes            abd about trunk +x, pitch/knee about parent +z",
        "  qpos/qvel       %d / %d  (freejoint is 7 of qpos but 6 of qvel)" % (NQ, NV),
        "  Q_STAND         -(DOG5 fold): abd -+90 (LEFT NEG), pitch/knee -+60",
        "  R_body_imu      %s" % ("identity (PLACEHOLDER -- not measured)"
                                  if np.allclose(R_BODY_IMU, np.eye(3))
                                  else "measured"),
    ])


if __name__ == "__main__":
    print(describe())
