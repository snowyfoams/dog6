"""DOG6 forward kinematics, Jacobians, inverse kinematics and leg statics.

This is the only module here that computes anything.  It reads its frames from
`coordinates` and its numbers from `params`, and owns neither -- so there is
exactly one copy of the geometry in this package and it is the one `selftest`
gates against model/dog6.xml.  A second forward kinematics written somewhere
else, agreeing by inspection, is the failure mode this layout exists to make
impossible.

    q = (hip_abduction, hip_pitch, knee)     radians, chain order

THE TWO FRAMES A FOOT LIVES IN, AND WHY BOTH
    trunk    origin at the abduction-axis plane centre.  What the balance and
             swing layers want, because it is the frame the four legs can be
             compared against each other in.
    hip      the same point minus that leg's HIP_OFFSET.  What IK solves in,
             because a leg's reachable set is fixed in it and moves in the
             other.

    The offset is a constant, so the JACOBIAN IS THE SAME IN BOTH and there is
    only one of them in this file.

LEGS MAY BE NAMED OR NUMBERED
    Every function takes `leg` as either "FL".."RR" or 0..3.  `coordinates`
    fixes the correspondence.  One translation, in one place -- the trot stack
    addresses legs as integers to index (4, 3) arrays while the description
    layer names them, and having both call the same function beats a wrapper
    module whose only job is `LEGS[i]`.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python sim/kinematics.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402
from . import params as P            # noqa: E402

_GRAVITY = P.GRAVITY


# ===========================================================================
# argument handling
# ===========================================================================
def leg_name(leg) -> str:
    """Accept "FL".."RR" or 0..3, return the canonical name."""
    if isinstance(leg, str):
        if leg not in C.LEG_INDEX:
            raise ValueError(f"unknown leg {leg!r}; expected one of {C.LEGS}")
        return leg
    index = int(leg)
    if not 0 <= index < C.N_LEGS:
        raise ValueError(f"leg index must be 0..{C.N_LEGS - 1}, got {leg}")
    return C.LEGS[index]


def leg_index(leg) -> int:
    """Accept "FL".."RR" or 0..3, return the canonical row index."""
    return C.LEG_INDEX[leg_name(leg)]


def _checked_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f"q must have shape (3,), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain only finite values")
    return q


# ===========================================================================
# the chain, walked once
# ===========================================================================
def leg_frames(leg, q):
    """Everything one walk down a leg produces.

    Returns ``(foot, anchors, axes, rotations, link_coms)``:

        foot        (3,)     foot site, trunk frame
        anchors     (3, 3)   the hip, pitch and knee hinge POINTS, trunk frame
        axes        (3, 3)   those hinge AXES, trunk frame
        rotations   (3, 3, 3) R_{trunk <- hip/thigh/shin}
        link_coms   (3, 3)   each link's centre of mass, trunk frame

    Every other function here is a projection of this one, so a pose costs one
    walk rather than three.  `foot_position` and `foot_jacobian` share it,
    which matters in an IK loop: the damped-least-squares step needs both and
    computing them separately doubles the cost of the inner iteration.
    """
    name = leg_name(leg)
    q_abd, q_pitch, q_knee = _checked_q(q)
    geom = P.LEG_GEOMETRY[name]

    r_hip, r_thigh, r_shin = C.link_rotations((q_abd, q_pitch, q_knee))

    # Each body carries one hinge at its own origin, so a child's translation
    # is rotated by the accumulated orientation of its PARENT.
    hip = np.asarray(geom.hip)
    pitch = hip + r_hip @ np.asarray(geom.hip_to_pitch)
    knee = pitch + r_thigh @ np.asarray(geom.pitch_to_knee)
    foot = knee + r_shin @ np.asarray(geom.knee_to_foot)

    anchors = np.stack((hip, pitch, knee), axis=0)
    axes = np.stack((C.X_AXIS, r_hip @ C.Z_AXIS, r_thigh @ C.Z_AXIS), axis=0)
    rotations = np.stack((r_hip, r_thigh, r_shin), axis=0)

    inertials = P.LINK_INERTIALS[name]
    link_coms = np.stack([
        origin + rotation @ np.asarray(inertial.com)
        for origin, rotation, inertial
        in zip(anchors, rotations, inertials)
    ], axis=0)

    return foot, anchors, axes, rotations, link_coms


def leg_state(leg, q) -> tuple[np.ndarray, np.ndarray]:
    """``(foot_in_trunk_frame, jacobian)`` from ONE pass down the chain."""
    foot, anchors, axes, _, _ = leg_frames(leg, q)
    jac = np.column_stack([
        np.cross(axis, foot - anchor) for anchor, axis in zip(anchors, axes)
    ])
    return foot, jac


# ===========================================================================
# forward kinematics
# ===========================================================================
def foot_position(leg, q) -> np.ndarray:
    """The foot SITE in the trunk frame, metres.

    The site is the centre of the foot ball, so a planted foot has this
    FOOT_RADIUS above the floor, not on it.
    """
    return leg_frames(leg, q)[0]


def foot_position_hip(leg, q) -> np.ndarray:
    """The foot site relative to that leg's own hip hinge."""
    return foot_position(leg, q) - P.HIP_OFFSET[leg_index(leg)]


def all_foot_positions(q) -> np.ndarray:
    """(4, 3) foot sites in the trunk frame, for a (4, 3) whole-robot pose."""
    q = C.unflat(q)
    return np.stack([foot_position(i, q[i]) for i in range(C.N_LEGS)], axis=0)


def foot_jacobian(leg, q) -> np.ndarray:
    """``d(foot_position)/dq`` as a 3x3 geometric Jacobian, trunk frame.

    The same matrix in the hip frame: the two differ by a constant, so their
    derivatives are equal.  Transposed, it maps a foot force to joint torques,
    which is what every force-level controller here actually uses it for.
    """
    return leg_state(leg, q)[1]


def all_foot_velocities(q, qd) -> np.ndarray:
    """(4, 3) foot velocities in the trunk frame, trunk held still."""
    q, qd = C.unflat(q), C.unflat(qd)
    return np.stack([foot_jacobian(i, q[i]) @ qd[i] for i in range(C.N_LEGS)], axis=0)


# ===========================================================================
# inverse kinematics
# ===========================================================================
def leg_ik(leg, p_hip, q_seed=None, iters: int = 40, tol: float = 1.0e-6):
    """Joint angles putting this leg's foot at `p_hip`, in the HIP frame.

    Levenberg-Marquardt damped least squares.  DAMPED rather than a plain
    pseudo-inverse because the reachable boundary IS reached in normal use --
    a foot placement at speed asks for a point the leg cannot quite meet, and
    an undamped step there returns an enormous joint command instead of the
    nearest pose.  Damped, the answer degrades to "as close as the leg gets",
    which is what a controller can survive.

    THE DAMPING SCALES WITH THE RESIDUAL, AND A FIXED ONE DOES NOT WORK.  A
    constant lambda shortens every step by the same factor, so convergence is
    linear and stalls: a fixed 1e-6 sat at 2.5e-7 m after 40 iterations and was
    still at 7e-8 after 120.  Scaling it gives robustness far from the target
    or near a singular pose, and hands back the true Newton step once the
    residual is small -- which is quadratic.  Same code, 3-5 iterations.

    `q_seed` is the warm start.  Pass the previous sweep's answer when there is
    one: it is faster, and more importantly it picks the SAME elbow branch
    every step.  Without a seed this leg's own row of Q_STAND is used, which
    chooses the knee-down stance branch.  A seed of zeros would be the flat
    calibration pose, which is very nearly singular -- see `coordinates`.

    `tol` IS A DISTANCE IN METRES, not a squared one.  1 um is already 20x
    finer than the ~20 um a 0.01 deg encoder count moves the foot, so tighter
    buys nothing real and costs the whole iteration budget.  Compared squared
    internally so the loop still costs no sqrt.

    Returns the best pose found, and does NOT raise on an unreachable target:
    the caller can see the residual through `foot_position_hip`.  It also does
    not apply `params.JOINT_LIMITS` -- clamp afterwards if that matters.
    """
    index = leg_index(leg)
    target = np.asarray(p_hip, dtype=float).reshape(3) + P.HIP_OFFSET[index]
    q = (C.Q_STAND[index].copy() if q_seed is None
         else np.asarray(q_seed, dtype=float).reshape(3).copy())

    for _ in range(int(iters)):
        foot, jac = leg_state(index, q)          # one chain walk, not two
        err = target - foot
        err2 = float(err @ err)
        if err2 < tol * tol:
            break
        lam2 = min(1.0e-4, max(1.0e-14, 1.0e-2 * err2))
        # (J^T J + lam^2 I)^-1 J^T err, solved in the 3x3 JOINT space rather
        # than task space so the damping bounds the joint increment, which is
        # the thing that needs bounding.
        dq = np.linalg.solve(jac.T @ jac + lam2 * np.eye(3), jac.T @ err)
        step = float(np.max(np.abs(dq)))
        if step > 0.3:                           # cap one step at ~17 deg
            dq *= 0.3 / step
        q = q + dq
    return q


def all_leg_ik(p_hip, q_seed=None, **kwargs) -> np.ndarray:
    """(4, 3) IK for four feet given as (4, 3) hip-frame positions."""
    p_hip = np.asarray(p_hip, dtype=float).reshape(C.N_LEGS, 3)
    seeds = (None,) * C.N_LEGS if q_seed is None else C.unflat(q_seed)
    return np.stack([
        leg_ik(i, p_hip[i], None if q_seed is None else seeds[i], **kwargs)
        for i in range(C.N_LEGS)
    ], axis=0)


# ===========================================================================
# leg statics
# ===========================================================================
def leg_gravity_torque(leg, q) -> np.ndarray:
    """Torque the three motors hold against this leg's OWN link weights.

    Statics with no foot contact, so measured joint torque equals this value
    and subtracting it from torque feedback isolates the external (ground)
    load.  That is the whole reason it exists.

    ASSUMES THE TRUNK IS HORIZONTAL -- gravity along trunk -z.  For a tilted
    trunk, rotate the gravity vector into the trunk frame first; this function
    has no way to know the base orientation.
    """
    _, anchors, axes, _, link_coms = leg_frames(leg, q)
    inertials = P.LINK_INERTIALS[leg_name(leg)]

    tau = np.zeros(3)
    for link, inertial in enumerate(inertials):
        weight = inertial.mass * _GRAVITY
        # A joint only carries the links at or below it in the chain.
        for joint in range(link + 1):
            dcom = np.cross(axes[joint], link_coms[link] - anchors[joint])
            tau[joint] += weight * dcom[2]
    return tau


def foot_force_to_torque(leg, q, force) -> np.ndarray:
    """Joint torques that produce a given force AT the foot, trunk frame.

    ``tau = J^T f``.  Sign: `force` is what the LEG applies to the world, so a
    stance leg holding the robot up takes a force with positive z.
    """
    return foot_jacobian(leg, q).T @ np.asarray(force, dtype=float).reshape(3)


# ===========================================================================
# whole-robot composite
# ===========================================================================
def body_inertia(q=None):
    """``(com, inertia)`` of the WHOLE robot at a pose, in the trunk frame.

    The inertia is about that CoM.  `q` defaults to Q_STAND.

    NOT THE TRUNK'S OWN TENSOR, AND THE DIFFERENCE IS A FACTOR OF FOUR.  The
    legs are 60 % of DOG6's mass and hang ~0.10 m below the trunk CoM, so the
    parallel-axis terms dominate: the trunk alone has Ixx 0.0090 and the
    assembled robot 0.0362.  Any template model that wants a single rigid-body
    inertia wants THIS one, evaluated at the stance it will actually hold.

    Computed rather than stored as a literal, because it is a function of the
    pose and a literal here goes stale the moment Q_STAND or the CAD moves.
    `selftest` gates the result against MuJoCo's own composite inertia.
    """
    q = C.Q_STAND if q is None else C.unflat(q)

    points = [np.asarray(P.TRUNK_COM, dtype=float)]
    masses = [P.TRUNK_MASS]
    tensors = [np.asarray(P.TRUNK_INERTIA, dtype=float)]

    for i, leg in enumerate(C.LEGS):
        _, _, _, rotations, link_coms = leg_frames(i, q[i])
        for k, inertial in enumerate(P.LINK_INERTIALS[leg]):
            rot = rotations[k]
            points.append(link_coms[k])
            masses.append(inertial.mass)
            # A tensor is carried between frames by a similarity transform.
            tensors.append(rot @ inertial.tensor() @ rot.T)

    points = np.asarray(points, dtype=float)
    masses = np.asarray(masses, dtype=float)
    com = (masses[:, None] * points).sum(axis=0) / masses.sum()

    inertia = np.zeros((3, 3))
    for point, mass, own in zip(points, masses, tensors):
        d = point - com
        inertia += own + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return com, inertia


def com_height(q=None, trunk_z: float | None = None) -> float:
    """Whole-robot CoM height above the floor, trunk origin at `trunk_z`."""
    trunk_z = P.STAND_HEIGHT if trunk_z is None else float(trunk_z)
    return trunk_z + float(body_inertia(q)[0][2])


# ===========================================================================
# derived stance geometry
# ===========================================================================
def foot_stance(q=None) -> np.ndarray:
    """(4, 3) foot sites in the trunk frame at Q_STAND (or a given pose).

    At Q_STAND this is (+-0.182, +-0.065, -0.177535), exactly square -- DOG6's
    CAD mirrors cleanly, so unlike DOG5 there is no small front/rear asymmetry
    to allow for.
    """
    return all_foot_positions(C.Q_STAND if q is None else q)


def hip_to_foot_stance(q=None) -> np.ndarray:
    """(4, 3) foot sites relative to their own hips at the stance pose."""
    return foot_stance(q) - P.HIP_OFFSET


def reach_used(q=None) -> float:
    """Fraction of LEG_REACH the leg is extended to at the stance pose.

    0.769 at Q_STAND.  DOG5's was 0.923, and that gap is the whole point of
    this robot: a leg at 92 % extension has almost nothing left for a step.
    """
    return float(np.linalg.norm(hip_to_foot_stance(q)[0])) / P.LEG_REACH


def reach_room(direction=(1.0, 0.0, 0.0), leg=0, margin: float = 0.95,
               q=None) -> float:
    """How far the foot may travel from its stance along `direction` before it
    hits a ``|hip_to_foot| <= margin * LEG_REACH`` clamp.

    SOLVES THE QUADRATIC THE CLAMP ACTUALLY POSES rather than subtracting two
    norms, and on DOG6 those disagree by a factor of 2.5.  Subtracting norms
    prices a step as if it were RADIAL -- straight out along the leg.  On DOG5
    it nearly was: the foot stood 120 mm ahead of the hip and 190 mm below, so
    the leg already pointed forward and stepping forward stretched it.  DOG6's
    foot hangs 25.7 mm ahead and 177.5 mm below, so a forward step is very
    nearly PERPENDICULAR to the leg and costs almost nothing in reach.

    The linear estimate says 42.2 mm; this says 106.9 mm.  So reach is NOT
    what limits DOG6's step -- torque, swing time and joint range are, and
    those are found by running it.
    """
    r = hip_to_foot_stance(q)[leg_index(leg)]
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    radius = margin * P.LEG_REACH
    b = float(r @ u)
    c = float(r @ r) - radius * radius
    disc = b * b - c
    if disc <= 0.0:
        return 0.0
    return float(-b + np.sqrt(disc))


def describe() -> str:
    com, inertia = body_inertia()
    stance = foot_stance()
    return "\n".join([
        "DOG6 kinematics, at Q_STAND",
        "  foot FL         %s" % np.round(stance[0], 6),
        "  stance          %.3f x %.3f m, %.1f mm below the trunk origin"
        % (stance[0, 0] - stance[2, 0], stance[0, 1] - stance[1, 1],
           -1000.0 * stance[0, 2]),
        "  CoM in trunk    %s" % np.round(com, 6),
        "  CoM above floor %.6f m" % com_height(),
        "  inertia @ CoM   diag %s" % np.round(np.diag(inertia), 6),
        "  leg reach       %.4f m, used %.1f%%" % (P.LEG_REACH, 100 * reach_used()),
        "  step room +x    %.1f mm  (radial estimate %.1f mm)"
        % (1000 * reach_room(),
           1000 * (0.95 * P.LEG_REACH - np.linalg.norm(hip_to_foot_stance()[0]))),
    ])


if __name__ == "__main__":
    print(describe())
