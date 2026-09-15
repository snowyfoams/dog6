"""The linearised, discretised body dynamics the QP is written against.

    m p_ddot   = sum_i f_i + m g
    d/dt(I w)  = sum_i r_i x f_i
    R_dot      = [w]x R                R: body -> world, ZYX Euler

    ->  x_dot = A(psi) x + B(psi, r) u          x in R^13, u in R^12

THE THREE APPROXIMATIONS, AND WHAT EACH ONE BUYS
    This is the whole content of the paper's modelling section, so it is worth
    naming them separately rather than calling the result "the linearised
    dynamics" and moving on.  Each is independently checkable, and
    `selftest` checks each against the unapproximated form.

    1.  ROLL AND PITCH ARE SMALL.  The exact map from angular velocity to
        Euler rates is Theta_dot = T(Theta)^-1 w with

            T = [[cos(psi)cos(th), -sin(psi), 0],
                 [sin(psi)cos(th),  cos(psi), 0],
                 [       -sin(th),         0, 1]]

        At th = 0 this is exactly Rz(psi), so Theta_dot = Rz(psi)' w.  A
        trotting robot holds roll and pitch near zero by construction -- they
        are weighted in Q and referenced to zero -- so the error is second
        order in quantities the controller is actively driving to zero.  This
        is what makes A depend on the state ONLY through yaw.

    2.  THE INERTIA IS EVALUATED AT THE YAW, NOT THE FULL ORIENTATION.
        I_world = R I_body R' becomes Rz(psi) I_body Rz(psi)'.  Same argument.

    3.  THE GYROSCOPIC TERM IS DROPPED.  The exact rotational dynamics are
        I w_dot = sum r x f - w x (I w).  The paper drops w x (I w) on the
        grounds that it is small for the angular rates a legged robot reaches.
        On DOG6 at 1 rad/s it is ~0.03 N*m against contact moments of several
        N*m, so the argument holds here too -- `selftest` measures it rather
        than assuming it.

    What all three buy is that A and B depend on the state only through yaw
    and on nothing that the horizon cannot know in advance.  The dynamics are
    therefore LINEAR TIME-VARYING with a schedule that is known at solve time,
    which is exactly the structure a convex QP needs.  Give up any one of them
    and the problem is a nonlinear program.

THE THIRTEENTH STATE
    m p_ddot = sum f + m g is AFFINE, not linear: the gravity term is a
    constant that no choice of A and B can produce from x and u.  Appending
    g = -9.81 as a state whose derivative is zero turns the constant into a
    linear term -- A[vz, g] = 1 -- at the cost of one row and column.  The
    alternative is carrying an affine offset through every step of the
    condensation, which is the same arithmetic written less tidily.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .. import params as P
from . import config as cfg


def rot_z(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v) -> np.ndarray:
    """[v]x, the matrix with [v]x a == v cross a."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def world_inertia(yaw: float, inertia_body=None) -> np.ndarray:
    """``Rz(psi) I_body Rz(psi)'`` -- approximation 2 above.

    `inertia_body` defaults to DOG6's whole-robot composite at the stance
    pose, NOT the trunk's own tensor.  The legs are 60 % of the mass hanging
    0.10 m below the trunk CoM, so the two differ by a factor of four in Ixx.
    The MPC's single rigid body is the whole robot, so it is the composite
    that belongs here.
    """
    inertia_body = INERTIA_BODY if inertia_body is None else inertia_body
    rot = rot_z(yaw)
    return rot @ np.asarray(inertia_body, dtype=float) @ rot.T


def euler_rate_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """T(Theta), the EXACT map from ZYX Euler rates to world angular velocity.

    Only the self-test uses this -- the MPC uses its small-angle form,
    Rz(psi).  It is here so that "small" is a measured claim.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[cy * cp, -sy, 0.0],
                     [sy * cp, cy, 0.0],
                     [-sp, 0.0, 1.0]])


# ===========================================================================
# the continuous-time model
# ===========================================================================
def continuous(yaw: float, r_feet, inertia_body=None, mass: float | None = None):
    """``(A, B)`` of ``x_dot = A x + B u`` at a given yaw and foot geometry.

    `r_feet` is (4, 3): the vector from the body CoM to each foot, in WORLD
    axes.  It is the only way the leg configuration enters the body model --
    the MPC has no kinematics, it has four points it may push from.

    A is (13, 13) and depends on yaw alone.  B is (13, 12) and depends on yaw
    and on where the feet are.  Both are rebuilt at every step of the horizon,
    because the feet move as the robot walks.
    """
    mass = P.MASS if mass is None else float(mass)
    r_feet = np.asarray(r_feet, dtype=float).reshape(4, 3)
    inv_inertia = np.linalg.inv(world_inertia(yaw, inertia_body))

    a = np.zeros((cfg.STATE_DIM, cfg.STATE_DIM))
    a[cfg.RPY, cfg.OMEGA] = rot_z(yaw).T      # Theta_dot = Rz' w
    a[cfg.POS, cfg.VEL] = np.eye(3)           # p_dot = v
    a[cfg.VEL.start + 2, cfg.GRAV] = 1.0      # v_dot gets the gravity state

    b = np.zeros((cfg.STATE_DIM, cfg.INPUT_DIM))
    for i in range(4):
        cols = slice(3 * i, 3 * i + 3)
        b[cfg.OMEGA, cols] = inv_inertia @ skew(r_feet[i])
        b[cfg.VEL, cols] = np.eye(3) / mass
    return a, b


def discretize(a, b, dt: float | None = None):
    """Exact zero-order-hold discretisation of (A, B) over `dt`.

    The textbook route is the block-matrix exponential identity

        expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]

    exact for a constant A and B over the interval, which is precisely what a
    zero-order hold on u makes them.  `discretize_expm` below does that, and
    `selftest` gates this function against it.

    THIS ONE USES A CLOSED FORM INSTEAD, BECAUSE THIS PARTICULAR A IS
    NILPOTENT.  A has exactly three nonzero blocks -- Theta <- w, p <- v, and
    v_z <- g -- and none of them feeds back, so the only surviving product is
    p_z <- g through v_z:

        A^2 has one nonzero entry.      A^3 = 0 exactly.

    The series for the block matrix therefore terminates after four terms:

        Ad = I + A dt + A^2 dt^2/2
        Bd = B dt + A B dt^2/2 + A^2 B dt^3/6

    This is not a truncation and not an approximation -- it is the same number
    `expm` computes, reached by noticing the series is finite.  It is ~40x
    faster, and the horizon needs ten of them per solve at 20 Hz.

    EXACT RATHER THAN EULER, AND AT THIS TIMESTEP THAT IS NOT PEDANTRY.
    MPC_DT is 0.05 s, which is long.  First-order Euler drops the A B dt^2/2
    term coupling force to POSITION, leaving the model believing a force
    changes velocity within a step but not position -- over ten steps that
    compounds into a plan that under-predicts how far the body travels.
    `selftest` measures Euler's error against this, for exactly that reason.
    """
    dt = cfg.MPC_DT if dt is None else float(dt)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a2 = a @ a
    ad = np.eye(a.shape[0]) + a * dt + a2 * (0.5 * dt * dt)
    bd = b * dt + (a @ b) * (0.5 * dt * dt) + (a2 @ b) * (dt ** 3 / 6.0)
    return ad, bd


def discretize_expm(a, b, dt: float | None = None):
    """The same thing via `scipy.linalg.expm`.  The reference `discretize` is
    checked against -- slower, and makes no assumption about A's structure."""
    dt = cfg.MPC_DT if dt is None else float(dt)
    n, m = cfg.STATE_DIM, cfg.INPUT_DIM
    block = np.zeros((n + m, n + m))
    block[:n, :n] = a
    block[:n, n:] = b
    exponent = expm(block * dt)
    return exponent[:n, :n].copy(), exponent[:n, n:].copy()


def discrete_sequence(yaws, r_feet_seq, dt=None, inertia_body=None, mass=None):
    """``(Ad, Bd)`` stacked over a horizon: ``(k, 13, 13)`` and ``(k, 13, 12)``.

    `yaws` is (k,) and `r_feet_seq` is (k, 4, 3) -- the yaw and foot geometry
    PREDICTED at each step, not today's repeated k times.  Rebuilding per step
    is what makes this an LTV model rather than an LTI one; it is also most of
    the solve's linear-algebra cost, which is why the horizon is ten and not
    a hundred.
    """
    yaws = np.atleast_1d(np.asarray(yaws, dtype=float))
    r_feet_seq = np.asarray(r_feet_seq, dtype=float).reshape(len(yaws), 4, 3)
    mats = [discretize(*continuous(yaw, r, inertia_body, mass), dt)
            for yaw, r in zip(yaws, r_feet_seq)]
    return (np.stack([m[0] for m in mats]), np.stack([m[1] for m in mats]))


# ===========================================================================
# the unapproximated model -- used only to measure the approximations
# ===========================================================================
def true_derivative(x, u, r_feet, inertia_body=None, mass: float | None = None):
    """``x_dot`` from the FULL rigid-body dynamics, no approximations.

    Keeps the exact Euler-rate map, the full orientation in the inertia, and
    the gyroscopic term.  Nothing in the controller calls this; `selftest`
    uses it to put a number on each approximation instead of quoting the
    paper's word for it.
    """
    mass = P.MASS if mass is None else float(mass)
    inertia_body = INERTIA_BODY if inertia_body is None else inertia_body
    x = np.asarray(x, dtype=float)
    forces = np.asarray(u, dtype=float).reshape(4, 3)
    r_feet = np.asarray(r_feet, dtype=float).reshape(4, 3)

    roll, pitch, yaw = x[cfg.RPY]
    omega = x[cfg.OMEGA]

    # Full R = Rz Ry Rx, and the inertia carried by it.
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rot = rot_z(yaw) @ rot_y @ rot_x
    inertia = rot @ np.asarray(inertia_body, dtype=float) @ rot.T

    torque = sum(np.cross(r_feet[i], forces[i]) for i in range(4))
    gyroscopic = np.cross(omega, inertia @ omega)

    dx = np.zeros(cfg.STATE_DIM)
    dx[cfg.RPY] = np.linalg.solve(euler_rate_matrix(roll, pitch, yaw), omega)
    dx[cfg.POS] = x[cfg.VEL]
    dx[cfg.OMEGA] = np.linalg.solve(inertia, torque - gyroscopic)
    dx[cfg.VEL] = forces.sum(axis=0) / mass + np.array([0.0, 0.0, x[cfg.GRAV]])
    return dx


def approx_derivative(x, u, r_feet, inertia_body=None, mass=None):
    """``A x + B u`` -- the model the MPC actually uses, evaluated directly."""
    a, b = continuous(float(np.asarray(x)[cfg.RPY][2]), r_feet, inertia_body, mass)
    return a @ np.asarray(x, dtype=float) + b @ np.asarray(u, dtype=float)


# ===========================================================================
# DOG6's composite inertia, computed once
# ===========================================================================
def _composite() -> tuple[np.ndarray, np.ndarray]:
    # Imported inside the function: `kinematics` does not depend on anything
    # in `cmpc`, and keeping the import here makes that direction obvious and
    # keeps `cmpc.config` importable without pulling in the chain.
    from .. import kinematics as K
    return K.body_inertia()


#: Where the CoM sits relative to the TRUNK ORIGIN, in body axes.
#:
#: THE MPC'S STATE IS THE CoM AND THE KINEMATICS' FRAME IS THE TRUNK ORIGIN,
#: and they are 29 mm apart in z.  Every foot vector handed to `continuous` as
#: `r_feet` has to be measured from the CoM, not from the trunk origin, or
#: every moment arm in B is wrong by this offset -- which a QP will happily
#: absorb into a small steady pitch rather than report as an error.
COM_OFFSET_BODY, INERTIA_BODY = _composite()

#: The whole robot's inertia about its own CoM at the stance pose, in body
#: axes.  Computed once at import: it is a function of the stance, and the
#: stance is fixed.  A trot changes it by only a few percent -- the legs stay
#: near the stance pose -- and the paper makes the same constant-inertia
#: assumption.  NOT the trunk's own tensor, which is 4x smaller in Ixx.
INERTIA_BODY = np.asarray(INERTIA_BODY, dtype=float)


def describe() -> str:
    from .. import kinematics as K
    feet = K.foot_stance() - COM_OFFSET_BODY
    a, b = continuous(0.0, feet)
    ad, bd = discretize(a, b)
    return "\n".join([
        "DOG6 convex-MPC body model",
        "  mass            %.4f kg" % P.MASS,
        "  CoM offset      %s m from the trunk origin" % np.round(COM_OFFSET_BODY, 6),
        "  inertia (body)  diag %s kg m^2" % np.round(np.diag(INERTIA_BODY), 6),
        "  A nonzeros      %d of %d" % (np.count_nonzero(a), a.size),
        "  B nonzeros      %d of %d" % (np.count_nonzero(b), b.size),
        "  Ad-I max        %.4g   (dt = %.3f s)"
        % (np.abs(ad - np.eye(cfg.STATE_DIM)).max(), cfg.MPC_DT),
        "  Bd max          %.4g" % np.abs(bd).max(),
    ])


if __name__ == "__main__":
    print(describe())
