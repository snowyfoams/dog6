"""Discretise, condense, and solve the convex MPC as one dense QP.

    min_U  sum_{i=0}^{k-1}  ||x_{i+1} - x_{i+1,ref}||_Qi + ||u_i||_Ri     (18)
    s.t.   x_{i+1} = Ai xi + Bi ui                                       (19)
           c_i <= Ci ui <= c_i_bar                                       (20)
           Di ui = 0                                                     (21)

CONDENSED, NOT SPARSE, AND THE CHOICE IS THE PAPER'S
    There are two ways to hand (18)-(21) to a solver.  Keep the states as
    decision variables and the dynamics (19) as equality constraints -- giving
    a big, very sparse QP in 13k + 12k variables -- or SUBSTITUTE (19) to
    express every future state in terms of the forces alone, giving a small
    dense QP in 12k variables.  This is the second.

        X = A_qp x0 + B_qp U

    where X stacks x_1..x_k.  At k = 10 that is 120 variables and a dense
    120x120 Hessian: about 30 microseconds of linear algebra.  The sparse form
    is 250 variables and needs a sparse factorisation to beat it.  The
    condensed form also makes the dynamics EXACTLY satisfied by construction
    rather than to solver tolerance, which matters at a 20 Hz outer loop --
    there is no chance to correct a constraint violation before the next
    solve.

    The cost is that A_qp and B_qp must be rebuilt each solve, and B_qp is
    block lower triangular with k(k+1)/2 blocks of matrix products.  That is
    the dominant cost here and it is why the horizon is ten.

HOW (21) IS ENFORCED, AND WHY NOT AS AN EQUALITY
    D_i u_i = 0 zeroes the forces of feet not in contact.  Written literally
    it is an equality block whose rank changes every step of the horizon as
    feet lift and land -- a solver-unfriendly shape that also has to be
    rebuilt with a different size each solve.

    Instead the swing feet keep their five friction rows and get a normal
    force box of [0, 0].  With f_z pinned to zero the four pyramid rows read
    |f_x| <= 0 and |f_y| <= 0, so the tangential components are forced to zero
    too.  The feasible set is exactly {u_i : D_i u_i = 0} intersected with the
    cone -- identical mathematics, constant sparsity pattern, no equality
    block at all.  `selftest` checks the swing forces come back at machine
    zero rather than trusting the argument.
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from . import config as cfg
from . import dynamics as dyn

#: Rows of the friction pyramid for one foot, acting on (fx, fy, fz).
#:
#: The first four are the LINEARISED cone: |f_t| <= mu * f_z, inscribed in the
#: true circular cone so it never permits a force the real one forbids.  The
#: fifth selects f_z so the normal force can be boxed -- which is where both
#: FZ_MIN/FZ_MAX and the swing-foot zeroing are applied.
def cone_rows(mu: float | None = None) -> np.ndarray:
    mu = cfg.MU if mu is None else float(mu)
    return np.array([
        [1.0, 0.0, -mu],      # fx - mu fz <= 0
        [-1.0, 0.0, -mu],     # -fx - mu fz <= 0
        [0.0, 1.0, -mu],      # fy - mu fz <= 0
        [0.0, -1.0, -mu],     # -fy - mu fz <= 0
        [0.0, 0.0, 1.0],      # fz, boxed below
    ])


ROWS_PER_FOOT = 5


# ===========================================================================
# condensation
# ===========================================================================
def condense(a_seq, b_seq):
    """``(A_qp, B_qp)`` such that ``X = A_qp x0 + B_qp U``.

    `a_seq` is (k, n, n) and `b_seq` is (k, n, m) -- the DISCRETE matrices,
    one pair per horizon step, already time-varying.

    A_qp is (kn, n) with block row i equal to ``A_{i} A_{i-1} ... A_0``.
    B_qp is (kn, km), block lower triangular, with

        B_qp[i, j] = A_i A_{i-1} ... A_{j+1} B_j      for j <= i
                   = 0                                otherwise

    Built by forward accumulation rather than by recomputing each product:
    block row i is block row i-1 premultiplied by A_i, with B_i appended on
    the diagonal.  That is O(k^2) matrix multiplies instead of O(k^3).
    """
    a_seq = np.asarray(a_seq, dtype=float)
    b_seq = np.asarray(b_seq, dtype=float)
    k, n, m = a_seq.shape[0], a_seq.shape[1], b_seq.shape[2]

    a_qp = np.zeros((k * n, n))
    b_qp = np.zeros((k * n, k * m))

    for i in range(k):
        rows = slice(i * n, (i + 1) * n)
        if i == 0:
            a_qp[rows] = a_seq[0]
        else:
            prev = slice((i - 1) * n, i * n)
            a_qp[rows] = a_seq[i] @ a_qp[prev]
            # Everything already accumulated in row i-1 propagates through A_i.
            b_qp[rows, : i * m] = a_seq[i] @ b_qp[prev, : i * m]
        b_qp[rows, i * m: (i + 1) * m] = b_seq[i]

    return a_qp, b_qp


def rollout(x0, forces, a_seq, b_seq) -> np.ndarray:
    """Integrate the discrete dynamics directly -- the thing `condense` replaces.

    Only the self-test calls this.  It exists so that "X = A_qp x0 + B_qp U"
    is a checked identity rather than a derivation nobody re-derived.
    """
    a_seq, b_seq = np.asarray(a_seq, float), np.asarray(b_seq, float)
    forces = np.asarray(forces, float).reshape(a_seq.shape[0], -1)
    x = np.asarray(x0, dtype=float)
    out = []
    for i in range(a_seq.shape[0]):
        x = a_seq[i] @ x + b_seq[i] @ forces[i]
        out.append(x.copy())
    return np.concatenate(out)


# ===========================================================================
# the cost
# ===========================================================================
def cost(a_qp, b_qp, x0, x_ref, q_diag=None, r_diag=None):
    """``(H, g)`` of ``1/2 U' H U + g' U``, dropping terms constant in U.

    Substituting X = A_qp x0 + B_qp U into (18) and writing e = A_qp x0 - X_ref
    for the part of the state error the forces cannot change:

        (B_qp U + e)' Q (B_qp U + e) + U' R U
      = U' (B_qp' Q B_qp + R) U + 2 e' Q B_qp U + const

    so H = 2(B_qp' Q B_qp + R) and g = 2 B_qp' Q e.  The factor of two is the
    1/2 in the solver's own objective convention; getting it wrong scales H
    and g together and changes nothing, which is exactly why it is easy to
    leave wrong and worth stating.

    H IS SYMMETRIC POSITIVE DEFINITE AS LONG AS R > 0.  B_qp' Q B_qp alone is
    only semidefinite -- with four feet down there are twelve forces and six
    wrench equations per step, so internal forces cost nothing.  R = 1e-6 I is
    what makes the factorisation well posed and picks the minimum-norm member
    of that null space.
    """
    q_diag = cfg.Q_DIAG if q_diag is None else np.asarray(q_diag, float)
    r_diag = cfg.R_DIAG if r_diag is None else np.asarray(r_diag, float)
    horizon = b_qp.shape[1] // cfg.INPUT_DIM

    q_big = np.tile(q_diag, horizon)
    r_big = np.tile(r_diag, horizon)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    error = a_qp @ np.asarray(x0, dtype=float) - x_ref
    weighted = b_qp * q_big[:, None]                  # Q @ B_qp, Q diagonal

    # `b_qp.T` is a transposed view, and handing a non-contiguous operand to
    # the matmul costs an internal copy of a 130x120 array every solve.  One
    # explicit ascontiguousarray is cheaper and makes the copy visible.
    b_t = np.ascontiguousarray(b_qp.T)
    hessian = b_t @ weighted
    hessian *= 2.0
    # R is diagonal; adding it in place beats allocating a 120x120 np.diag.
    hessian[np.diag_indices_from(hessian)] += 2.0 * r_big

    gradient = 2.0 * (weighted.T @ error)
    return hessian, gradient


# ===========================================================================
# the constraints
# ===========================================================================
def constraints(contacts, mu=None, fz_min=None, fz_max=None):
    """``(C, lo, hi)`` for ``lo <= C U <= hi`` over the whole horizon.

    `contacts` is (k, 4) bool -- the gait schedule, already sampled at the
    horizon steps.  A foot in contact gets ``fz in [fz_min, fz_max]``; a foot
    in swing gets ``fz in [0, 0]``, which with the pyramid rows forces all
    three components to zero.  That is constraint (21).

    C is block diagonal with one 5x3 block per foot per step, so it is 96 % +
    zeros at k = 10 and is built sparse.
    """
    mu = cfg.MU if mu is None else float(mu)
    fz_min = cfg.FZ_MIN if fz_min is None else float(fz_min)
    fz_max = cfg.FZ_MAX if fz_max is None else float(fz_max)

    contacts = np.asarray(contacts, dtype=bool).reshape(-1, 4)
    horizon = contacts.shape[0]
    block = cone_rows(mu)

    matrix = sp.block_diag([block] * (4 * horizon), format="csc")
    lo = np.empty(ROWS_PER_FOOT * 4 * horizon)
    hi = np.empty_like(lo)

    for step in range(horizon):
        for foot in range(4):
            rows = slice(ROWS_PER_FOOT * (4 * step + foot),
                         ROWS_PER_FOOT * (4 * step + foot + 1))
            planted = bool(contacts[step, foot])
            lo[rows] = (-np.inf, -np.inf, -np.inf, -np.inf,
                        fz_min if planted else 0.0)
            hi[rows] = (0.0, 0.0, 0.0, 0.0, fz_max if planted else 0.0)
    return matrix, lo, hi


# ===========================================================================
# the solver
# ===========================================================================
class Solver:
    """An OSQP instance reused across solves.

    SET UP ONCE, UPDATED THEREAFTER.  The sparsity patterns of H and C are
    fixed by the horizon -- only their VALUES change as the robot moves and
    the gait advances -- so `osqp.setup` is called on the first solve and
    every later one calls `update`.  Re-running setup each time would re-do
    the symbolic factorisation at 20 Hz for no reason; measured, it is about
    an order of magnitude of the solve time.

    The warm start is OSQP's own: it keeps its primal and dual iterates
    between solves, and consecutive MPC problems differ only slightly, so a
    typical re-solve converges in a handful of iterations.
    """

    def __init__(self, horizon: int | None = None, **settings):
        self.horizon = cfg.HORIZON if horizon is None else int(horizon)
        self.n_vars = self.horizon * cfg.INPUT_DIM
        self._problem: osqp.OSQP | None = None
        self._settings = dict(
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=4000,
            polishing=True,
            warm_starting=True,
        )
        self._settings.update(settings)
        self.last_status = "not solved"
        self.last_iterations = 0

    def solve(self, hessian, gradient, matrix, lo, hi) -> np.ndarray:
        """Return the stacked force sequence U, shape ``(horizon * 12,)``."""
        # OSQP wants only the upper triangle of P, and wants it symmetric.
        p_mat = sp.csc_matrix(np.triu((hessian + hessian.T) * 0.5))
        a_mat = sp.csc_matrix(matrix)

        if self._problem is None:
            self._problem = osqp.OSQP()
            self._problem.setup(P=p_mat, q=gradient, A=a_mat, l=lo, u=hi,
                                **self._settings)
        else:
            self._problem.update(Px=p_mat.data, Ax=a_mat.data,
                                 q=gradient, l=lo, u=hi)

        result = self._problem.solve()
        self.last_status = str(result.info.status)
        self.last_iterations = int(result.info.iter)

        solution = result.x
        if solution is None or not np.all(np.isfinite(solution)):
            # A FAILED SOLVE MUST NOT PROPAGATE NaN INTO THE JOINTS.  OSQP
            # returns None on primal infeasibility, which here would mean the
            # gait is asking for something the friction cone cannot deliver.
            # Zero force is wrong but bounded; the caller sees `last_status`.
            return np.zeros(self.n_vars)
        return np.asarray(solution, dtype=float)


def solve_once(x0, x_ref, a_seq, b_seq, contacts, solver: Solver | None = None,
               q_diag=None, r_diag=None, **bounds):
    """Build and solve one MPC problem.  Returns ``(u0, U)``.

    `u0` is the first 12-vector -- the forces actually applied -- and `U` is
    the whole horizon, which is useful for plotting what the controller
    believes is about to happen.
    """
    a_qp, b_qp = condense(a_seq, b_seq)
    hessian, gradient = cost(a_qp, b_qp, x0, x_ref, q_diag, r_diag)
    matrix, lo, hi = constraints(contacts, **bounds)
    solver = Solver(horizon=len(a_seq)) if solver is None else solver
    forces = solver.solve(hessian, gradient, matrix, lo, hi)
    return forces[:cfg.INPUT_DIM].reshape(4, 3), forces


def describe() -> str:
    from .. import kinematics as K
    from . import gait

    feet = K.foot_stance() - dyn.COM_OFFSET_BODY
    yaws = np.zeros(cfg.HORIZON)
    a_seq, b_seq = dyn.discrete_sequence(yaws, np.tile(feet, (cfg.HORIZON, 1, 1)))
    a_qp, b_qp = condense(a_seq, b_seq)
    contacts = gait.horizon_contacts(0.0)

    x0 = np.zeros(cfg.STATE_DIM)
    x0[cfg.POS] = (0.0, 0.0, cfg.Z_REF)
    x0[cfg.GRAV] = -9.81
    x_ref = np.tile(x0, (cfg.HORIZON, 1))

    solver = Solver()
    u0, _ = solve_once(x0, x_ref, a_seq, b_seq, contacts, solver)
    from .. import params as P
    return "\n".join([
        "DOG6 convex-MPC QP",
        "  variables       %d  (%d steps x 12 forces)"
        % (cfg.HORIZON * cfg.INPUT_DIM, cfg.HORIZON),
        "  constraint rows %d  (4 feet x 5 rows x %d steps)"
        % (ROWS_PER_FOOT * 4 * cfg.HORIZON, cfg.HORIZON),
        "  A_qp            %s" % (a_qp.shape,),
        "  B_qp            %s  (%.0f%% nonzero, block lower triangular)"
        % (b_qp.shape, 100.0 * np.count_nonzero(b_qp) / b_qp.size),
        "  status          %s in %d iterations"
        % (solver.last_status, solver.last_iterations),
        "  contacts at t=0 %s" % (contacts[0].astype(int),),
        "  fz held         %s N   (sum %.2f, weight %.2f)"
        % (np.round(u0[:, 2], 2), u0[:, 2].sum(), P.WEIGHT),
    ])


if __name__ == "__main__":
    print(describe())
