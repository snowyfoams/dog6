"""Rigid-body dynamics of ONE leg, in the trunk frame.  M, C qd + G, Lambda.

`kinematics` answers where the foot is; this answers what it takes to move it.
It exists because the swing control law needs three things that no amount of
forward kinematics provides:

    tau = J' [Kp (p_ref - p) + Kd (v_ref - v)] + tau_ff
    tau_ff = J' Lambda (a_ref - Jdot qdot) + C qdot + G
             ~~~~~~~~   ~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~

THE TRUNK IS TREATED AS AN INERTIAL FRAME, WHICH IS AN APPROXIMATION AND THE
RIGHT ONE HERE.  A swing leg is 0.88 kg on a 5.88 kg robot, and the swing law
is written in body coordinates (Bp, Bv, Ba) precisely so that it does not have
to model the base motion it is reacting to.  The stance legs and the MPC own
the base; this module owns the leg hanging off it.  Nothing here is valid for
a stance leg, and nothing here is used for one.

WHY RECURSIVE NEWTON-EULER AND NOT A CLOSED FORM
    A 3-DOF chain has a closed-form mass matrix, and writing it out is the
    traditional thing to do.  It is also nine trigonometric expressions that
    no reviewer can check and that silently stop matching the CAD the moment a
    link inertia changes.  RNEA is one loop that reads `params.LINK_INERTIALS`
    directly, so it cannot go stale, and `cmpc.selftest` gates every output of
    it against MuJoCo's own `mj_fullM` and `qfrc_bias`.

    One algorithm gives all three quantities:

        bias(q, qd)   = RNEA(q, qd, qdd=0, gravity on)   -> C qdot + G
        mass_matrix   = RNEA(q,  0, qdd=e_j, gravity off) column by column
        jdot_qdot     = the forward pass alone, read at the foot

THE ARMATURE IS PART OF THE MASS MATRIX AND IT DOMINATES IT
    `params.ARMATURE` is 0.0085 kg m^2 at every joint -- the 10:1 gearbox's
    rotor reflected to the joint side.  The shin has 9.3e-5 kg m^2 about the
    knee, so the rotor is 91x the link it drives.  Leave it out of M and the
    operational-space inertia Lambda is wrong by a factor of ten in the
    vertical direction (0.449 kg against 4.54 kg), which is the difference
    between a swing that clears the ground and one that does not.

    It is added to the DIAGONAL of M and contributes nothing to the bias: a
    constant inertia has no velocity-dependent term.  MuJoCo does exactly the
    same thing, which is why the gates match.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python sim/leg_dynamics.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402
from . import kinematics as K        # noqa: E402
from . import params as P            # noqa: E402

#: Gravity as a vector in the trunk frame, trunk assumed level.
GRAVITY_VEC = np.array([0.0, 0.0, -P.GRAVITY])

_N = C.N_JOINTS_PER_LEG              # 3


def _link_tensors(leg, rotations) -> list[np.ndarray]:
    """Each link's inertia about its own CoM, rotated into the trunk frame."""
    return [rot @ inertial.tensor() @ rot.T
            for rot, inertial in zip(rotations, P.LINK_INERTIALS[K.leg_name(leg)])]


def _forward(anchors, axes, link_coms, qd, qdd, base_accel):
    """RNEA forward pass: per-link angular velocity, angular and CoM acceleration.

    `base_accel` is the linear acceleration of the TRUNK ORIGIN.  Passing -g
    there is the standard trick that makes the backward pass produce gravity
    torques without gravity appearing anywhere else: an accelerating frame and
    a gravity field are indistinguishable, so a base "accelerating upward at g"
    loads the links exactly as gravity does.
    """
    omega = np.zeros((_N, 3))
    omega_dot = np.zeros((_N, 3))
    com_accel = np.zeros((_N, 3))

    w = np.zeros(3)                  # the base does not rotate
    wd = np.zeros(3)
    a_prev = np.asarray(base_accel, dtype=float)
    p_prev = np.zeros(3)             # the trunk origin

    for k in range(_N):
        anchor, axis = anchors[k], axes[k]
        # Carry the acceleration of the parent's frame out to this joint's
        # anchor.  w / wd are still the PARENT's at this point.
        d = anchor - p_prev
        a_anchor = a_prev + np.cross(wd, d) + np.cross(w, np.cross(w, d))

        # Now add this joint's own motion.  The cross term uses the PARENT's
        # omega against this joint's contribution, so it is formed before w is
        # reassigned.
        spin = axis * qd[k]
        wd = wd + axis * qdd[k] + np.cross(w, spin)
        w = w + spin

        r = link_coms[k] - anchor
        com_accel[k] = (a_anchor + np.cross(wd, r)
                        + np.cross(w, np.cross(w, r)))
        omega[k], omega_dot[k] = w, wd
        a_prev, p_prev = a_anchor, anchor

    return omega, omega_dot, com_accel, a_prev, p_prev


def _backward(anchors, axes, link_coms, masses, tensors,
              omega, omega_dot, com_accel):
    """RNEA backward pass: joint torques from per-link inertial wrenches."""
    force = np.zeros(3)              # f_{k+1}, the wrench the child exerts
    moment = np.zeros(3)             # n_{k+1}
    child_anchor = None
    tau = np.zeros(_N)

    for k in range(_N - 1, -1, -1):
        anchor = anchors[k]
        f_link = masses[k] * com_accel[k]
        n_link = (tensors[k] @ omega_dot[k]
                  + np.cross(omega[k], tensors[k] @ omega[k]))

        moment = moment + np.cross(link_coms[k] - anchor, f_link) + n_link
        if child_anchor is not None:
            moment = moment + np.cross(child_anchor - anchor, force)
        force = force + f_link

        tau[k] = float(moment @ axes[k])
        child_anchor = anchor
    return tau


def _rnea(leg, q, qd, qdd, gravity: bool) -> np.ndarray:
    """Joint torques for a commanded acceleration.  The whole algorithm."""
    name = K.leg_name(leg)
    _, anchors, axes, rotations, link_coms = K.leg_frames(name, q)
    inertials = P.LINK_INERTIALS[name]
    tensors = _link_tensors(name, rotations)
    masses = [inertial.mass for inertial in inertials]

    base = -GRAVITY_VEC if gravity else np.zeros(3)
    omega, omega_dot, com_accel, _, _ = _forward(
        anchors, axes, link_coms, np.asarray(qd, float), np.asarray(qdd, float), base)
    return _backward(anchors, axes, link_coms, masses, tensors,
                     omega, omega_dot, com_accel)


# ===========================================================================
# the three quantities the swing law needs
# ===========================================================================
def mass_matrix(leg, q, armature: bool = True) -> np.ndarray:
    """The 3x3 joint-space mass matrix M(q), armature included by default.

    Assembled from each link's own Jacobians rather than by running the
    recursion three times:

        M = sum_k ( Jv_k' m_k Jv_k + Jw_k' I_k Jw_k ) + armature I

    where Jv_k and Jw_k map joint rates to link k's CoM velocity and angular
    velocity.  Both are trivial for a serial chain -- column j of Jw_k is
    joint j's axis, and column j of Jv_k is that axis crossed into the lever
    from joint j to link k's CoM -- and joints past k contribute nothing.

    ONE CHAIN WALK INSTEAD OF THREE.  `mass_matrix_rnea` below is the column-
    by-column definition, which needs a separate RNEA pass per column and so
    three `leg_frames` calls; this needs one.  At 250 Hz with four legs that
    was the single largest cost in the control loop.  The two are independent
    derivations of the same matrix and `cmpc.selftest` checks they agree, so
    the fast one is not taken on trust.
    """
    name = K.leg_name(leg)
    _, anchors, axes, rotations, link_coms = K.leg_frames(name, q)
    tensors = _link_tensors(name, rotations)
    inertials = P.LINK_INERTIALS[name]

    mass = np.zeros((_N, _N))
    for k in range(_N):
        jv = np.zeros((3, _N))
        jw = np.zeros((3, _N))
        for j in range(k + 1):           # joints past k do not move link k
            jv[:, j] = np.cross(axes[j], link_coms[k] - anchors[j])
            jw[:, j] = axes[j]
        mass += inertials[k].mass * (jv.T @ jv) + jw.T @ tensors[k] @ jw

    if armature:
        mass[np.diag_indices(_N)] += P.ARMATURE
    return mass


def mass_matrix_rnea(leg, q, armature: bool = True) -> np.ndarray:
    """M(q) from the definition: column j is the torque to accelerate joint j.

    The slow, obviously-correct route.  Kept because it is what `mass_matrix`
    is gated against -- two derivations that share no code.
    """
    q = np.asarray(q, dtype=float)
    columns = [_rnea(leg, q, np.zeros(_N), np.eye(_N)[j], gravity=False)
               for j in range(_N)]
    mass = np.column_stack(columns)
    if armature:
        mass = mass + P.ARMATURE * np.eye(_N)
    return mass


def bias(leg, q, qd) -> np.ndarray:
    """``C(q, qd) qd + G(q)`` -- the torque to hold a pose while moving through it.

    At qd = 0 this is pure gravity and equals `kinematics.leg_gravity_torque`;
    the self-test gates that the two agree, since they are independent routes
    to the same number.
    """
    return _rnea(leg, q, qd, np.zeros(_N), gravity=True)


def coriolis(leg, q, qd) -> np.ndarray:
    """``C(q, qd) qd`` alone, gravity removed."""
    return _rnea(leg, q, qd, np.zeros(_N), gravity=False)


def gravity_torque(leg, q) -> np.ndarray:
    """``G(q)`` alone.  The same number `kinematics.leg_gravity_torque` returns."""
    return _rnea(leg, q, np.zeros(_N), np.zeros(_N), gravity=True)


def jdot_qdot(leg, q, qd) -> np.ndarray:
    """``Jdot(q, qd) qdot`` -- the foot's acceleration at zero joint acceleration.

    That identity IS the definition: a_foot = J qddot + Jdot qdot, so setting
    qddot = 0 and gravity off leaves exactly the term wanted.  No numerical
    differentiation of J is involved, which matters -- differencing a Jacobian
    at control rate is noisy in precisely the fast part of a swing where this
    term is largest.
    """
    name = K.leg_name(leg)
    foot, anchors, axes, _, link_coms = K.leg_frames(name, q)
    omega, omega_dot, _, a_anchor, p_last = _forward(
        anchors, axes, link_coms, np.asarray(qd, float), np.zeros(_N), np.zeros(3))

    # Carry the shin's frame acceleration out to the foot point.  `a_anchor`
    # and `p_last` are the knee's, and omega[-1] / omega_dot[-1] are the
    # shin's, which is the body the foot is rigidly attached to.
    d = foot - p_last
    return (a_anchor + np.cross(omega_dot[-1], d)
            + np.cross(omega[-1], np.cross(omega[-1], d)))


def operational_inertia(leg, q, armature: bool = True, damping: float = 1e-9):
    """``Lambda = (J M^-1 J')^-1`` -- the apparent inertia seen AT the foot.

    This is what turns a desired foot acceleration into the force that
    produces it, and it is the term that makes a swing possible on this robot:
    with the armature included, the foot's vertical apparent mass at the stand
    pose is 4.54 kg against the 0.449 kg its links weigh.

    `damping` regularises the inverse.  J loses rank at a singular pose -- and
    DOG6's calibration zero is very nearly one, with the leg lying along its
    own abduction axis -- so J M^-1 J' becomes singular there and an exact
    inverse returns garbage instead of a large number.  1e-9 is far below any
    physical inertia here and only bites where the exact answer is undefined.
    """
    jac = K.foot_jacobian(leg, q)
    mass = mass_matrix(leg, q, armature=armature)
    lambda_inv = jac @ np.linalg.solve(mass, jac.T)
    return np.linalg.inv(lambda_inv + damping * np.eye(3))


def foot_apparent_mass(leg, q, direction=(0.0, 0.0, 1.0), **kwargs) -> float:
    """The scalar inertia the foot presents along one direction, in kg."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    return float(u @ operational_inertia(leg, q, **kwargs) @ u)


def describe() -> str:
    q = C.Q_STAND[0]
    with_arm = foot_apparent_mass("FL", q)
    without = foot_apparent_mass("FL", q, armature=False)
    jac = K.foot_jacobian("FL", q)
    return "\n".join([
        "DOG6 leg dynamics, FL at Q_STAND",
        "  M diag          %s kg m^2" % np.round(np.diag(mass_matrix("FL", q)), 6),
        "  M without arm   %s" % np.round(
            np.diag(mass_matrix("FL", q, armature=False)), 6),
        "  G               %s N m" % np.round(gravity_torque("FL", q), 4),
        "  Lambda diag     %s kg" % np.round(
            np.diag(operational_inertia("FL", q)), 4),
        "  foot mass, z    %.3f kg with the armature, %.3f without"
        % (with_arm, without),
        "  knee J_z        %.1f mm/rad -- the lever the rotor appears through"
        % (1000 * jac[2, 2]),
    ])


if __name__ == "__main__":
    print(describe())
