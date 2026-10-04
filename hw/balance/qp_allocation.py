"""Stage 4, the QP alternative to `allocation`: eq (4), the review's form.

    F* = argmin  (A F - b_d)^T S (A F - b_d) + alpha ||F||^2 + beta ||F - F*_prev||^2
         s.t.    C F <= d

`allocation.allocate` is the weighted least squares the law runs: it solves
A f = b_d EXACTLY, with the cone nowhere in the problem, and projects onto
the cone afterwards, foot by foot.  Whatever the projection removes is
lost -- it becomes `residual` and nothing re-assigns it.  This file is the
other formulation: the equality is a weighted PENALTY, the cone is a HARD
constraint inside the solve, so when one foot reaches its cone the solver
moves the load to a foot that still has room.  Same inputs, same
`Allocation` out, so `law.BalanceLaw(allocator=QPAllocator())` swaps one for
the other without touching a gain.

THE THREE TERMS, AND WHAT EACH ONE OWNS
    (AF - b_d)^T S (AF - b_d)   wrench tracking.  S is 6x6 diagonal: its
                                force rows are in N^2, its moment rows in
                                (N*m)^2, so S is also where N and N*m are put
                                on one scale.  `config.QP_S_MOMENT` is 1/l^2
                                with l the roll lever, which prices 1 N*m of
                                moment error at the force it would take at
                                that lever to make it up.
    alpha ||F||^2               picks the smallest, most even point in the
                                six-dimensional internal-force null space,
                                and makes the Hessian positive definite.  It
                                is UNWEIGHTED, as eq (4) writes it: the soft
                                friction cone the least-squares W carried is
                                not here, the hard one in C F <= d is.
    beta ||F - F*_prev||^2      a low-pass on the solution.  Unconstrained,
                                the internal-force component decays by
                                beta / (alpha + beta) per sweep (0.91 at the
                                defaults, ~10 sweeps, 40 ms at 250 Hz) while
                                the wrench-tracking component sees
                                beta / (sigma^2 + alpha + beta) ~ 0.003 --
                                so beta smooths how the feet SHARE the
                                load, not how fast the body gets its wrench.

THE CONSTRAINT SET IS NEVER EMPTY
    C F <= d is the linearised friction pyramid |f_x|, |f_y| <= mu f_z and
    the box c_i fz_min <= f_z <= c_i fz_max, per foot, with c_i the contact
    weight.  f = (0, 0, c_i fz_min) satisfies every row of every foot, so
    the QP is feasible by construction and OSQP cannot report primal
    infeasibility here.  What it CAN do is hit `QP_MAX_ITER`; the iterate is
    then still finite and is used after the same hard projection the
    least-squares path applies, and `last_status` says so.  Only a
    non-finite result falls back to `allocation.allocate`, counted in
    `fallbacks`.

SET UP ONCE, UPDATED THEREAFTER
    P = 2 (A^T S A + (alpha + beta) I) is dense 12x12 and its upper triangle
    is stored with every entry explicit, so the sparsity pattern OSQP
    factorised at setup never changes and each sweep is one `update` of
    (Px, q, u) -- a numeric refactorisation of a 36x36 KKT system -- plus a
    warm-started solve.  C depends only on mu, so A_x is updated only when
    mu changes.  `describe()` prints the measured cost; the number that
    matters is the warm p95 against the 333 us CAN slot, and it is the
    reason this file is OPT-IN while `allocation` stays the default.
"""
from __future__ import annotations

import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/qp_allocation.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from . import allocation as ALLOC    # noqa: E402
from . import config as cfg          # noqa: E402

try:                                 # the robot's Pi has neither; see requirements.txt
    import osqp
    import scipy.sparse as sp
except ImportError:                  # pragma: no cover - platform dependent
    osqp = None
    sp = None

__all__ = ["available", "cone_rows", "constraints", "cost", "QPAllocator",
           "ROWS_PER_FOOT", "N_VARS"]

N_VARS = 3 * C.N_LEGS
ROWS_PER_FOOT = 6


def available() -> bool:
    """Whether osqp and scipy are importable on this interpreter."""
    return osqp is not None


def cone_rows(mu: float | None = None) -> np.ndarray:
    """The (6, 3) block of C for one foot, acting on (fx, fy, fz).

    Four pyramid rows (the inscribed, linearised cone) and the two sides of
    the normal-force box.  Everything is `row . f <= d`, matching eq (4)
    literally; `constraints` supplies the d.
    """
    mu = cfg.MU if mu is None else float(mu)
    return np.array([
        [1.0, 0.0, -mu],      #  fx - mu fz <= 0
        [-1.0, 0.0, -mu],     # -fx - mu fz <= 0
        [0.0, 1.0, -mu],      #  fy - mu fz <= 0
        [0.0, -1.0, -mu],     # -fy - mu fz <= 0
        [0.0, 0.0, 1.0],      #  fz <= c fz_max
        [0.0, 0.0, -1.0],     # -fz <= -c fz_min
    ])


def constraints(contact=None, mu=None, fz_min=None, fz_max=None):
    """``(C, d)`` with C (24, 12) block diagonal and d (24,): C F <= d.

    `contact` is the (4,) weight in [0, 1]; None is all ones.  A foot at 0
    gets the box [0, 0], which with the pyramid rows pins all three of its
    components to zero -- it is out of the problem without changing the
    problem's shape.
    """
    mu = cfg.MU if mu is None else float(mu)
    fz_min = cfg.FZ_MIN if fz_min is None else float(fz_min)
    fz_max = cfg.FZ_MAX if fz_max is None else float(fz_max)
    weight = (np.ones(C.N_LEGS) if contact is None
              else np.clip(np.asarray(contact, dtype=float).reshape(C.N_LEGS),
                           0.0, 1.0))
    block = cone_rows(mu)
    c_mat = np.zeros((ROWS_PER_FOOT * C.N_LEGS, N_VARS))
    d_vec = np.zeros(ROWS_PER_FOOT * C.N_LEGS)
    for i in range(C.N_LEGS):
        rows = slice(ROWS_PER_FOOT * i, ROWS_PER_FOOT * (i + 1))
        c_mat[rows, 3 * i:3 * i + 3] = block
        d_vec[ROWS_PER_FOOT * i + 4] = weight[i] * fz_max
        d_vec[ROWS_PER_FOOT * i + 5] = -weight[i] * fz_min
    return c_mat, d_vec


def cost(A, b_d, f_prev, s_diag, alpha: float, beta: float):
    """``(P, q)`` so that eq (4)'s objective is ``1/2 F^T P F + q^T F`` + const.

        P = 2 (A^T S A + (alpha + beta) I)
        q = -2 (A^T S b_d + beta F_prev)
    """
    a_s = A.T * s_diag                                   # A^T S, S diagonal
    P = 2.0 * (a_s @ A)
    P[np.diag_indices(N_VARS)] += 2.0 * (alpha + beta)
    q = -2.0 * (a_s @ b_d + beta * f_prev)
    return P, q


class QPAllocator:
    """Eq (4) solved by OSQP, one instance per law, holding F*_prev.

    `allocate(r_w, b_d, mu=, contact=)` has `allocation.allocate`'s
    signature and returns its `Allocation`, so it drops into stage 4 of
    `law.BalanceLaw` as `allocator=`.  `Allocation.clipped` here means
    ACTIVE: a foot with a cone or box row at its bound, which is the QP's
    counterpart of "the projection moved it".
    """

    def __init__(self, *, s_force=None, s_moment=None, alpha=None, beta=None,
                 mu=None, fz_min=None, fz_max=None, max_iter=None, eps=None,
                 polish: bool = False):
        if osqp is None:
            raise ImportError("QPAllocator needs osqp and scipy "
                              "(`pip install osqp scipy`); see requirements.txt")
        self.s_force = cfg.QP_S_FORCE if s_force is None else float(s_force)
        self.s_moment = cfg.QP_S_MOMENT if s_moment is None else float(s_moment)
        self.alpha = cfg.QP_ALPHA if alpha is None else float(alpha)
        self.beta = cfg.QP_BETA if beta is None else float(beta)
        self.mu = cfg.MU if mu is None else float(mu)
        self.fz_min = cfg.FZ_MIN if fz_min is None else float(fz_min)
        self.fz_max = cfg.FZ_MAX if fz_max is None else float(fz_max)
        self.max_iter = cfg.QP_MAX_ITER if max_iter is None else int(max_iter)
        self.eps = cfg.QP_EPS if eps is None else float(eps)
        self.polish = bool(polish)
        self.s_diag = np.concatenate([np.full(3, self.s_force),
                                      np.full(3, self.s_moment)])

        # the upper triangle of P, column-major, EVERY entry explicit
        cols = np.concatenate([np.full(j + 1, j) for j in range(N_VARS)])
        rows = np.concatenate([np.arange(j + 1) for j in range(N_VARS)])
        self._p_rows, self._p_cols = rows, cols
        self._p_indptr = np.concatenate([[0], np.cumsum(np.arange(1, N_VARS + 1))])

        self.f_prev = np.zeros(N_VARS)
        self._problem = None
        self._c_mu = None
        self._c_mat = None
        self._d_box = None
        self.last_status = "not solved"
        self.last_iterations = 0
        self.last_solve_s = 0.0
        self.last_setup_s = 0.0
        self.fallbacks = 0
        self.max_iter_hits = 0

    # -- state -------------------------------------------------------------
    def reset(self, f_prev=None) -> None:
        """Forget F*_prev (zero, or the (4, 3) / (12,) given)."""
        self.f_prev = (np.zeros(N_VARS) if f_prev is None
                       else np.asarray(f_prev, dtype=float).reshape(N_VARS).copy())

    # -- the solve -----------------------------------------------------------
    def _p_data(self, P) -> np.ndarray:
        return P[self._p_rows, self._p_cols]

    def _constraints(self, weight):
        """``(C, d)`` with C cached -- it depends on mu alone -- and d the
        box rows scaled by the contact weight.  13 us a sweep otherwise."""
        if self._c_mat is None or self._c_mu != self.mu:
            self._c_mat, d_unit = constraints(None, self.mu, self.fz_min, self.fz_max)
            self._d_box = d_unit.reshape(C.N_LEGS, ROWS_PER_FOOT)
        return self._c_mat, (self._d_box * weight[:, None]).reshape(-1)

    def _setup(self, P, q, c_mat, d_vec) -> None:
        t0 = time.perf_counter()
        p_csc = sp.csc_matrix((self._p_data(P), self._p_rows, self._p_indptr),
                              shape=(N_VARS, N_VARS))
        self._c_csc = sp.csc_matrix(c_mat)
        self._problem = osqp.OSQP()
        self._problem.setup(P=p_csc, q=q, A=self._c_csc,
                            l=np.full(d_vec.shape, -np.inf), u=d_vec,
                            verbose=False, eps_abs=self.eps, eps_rel=self.eps,
                            max_iter=self.max_iter, polishing=self.polish,
                            warm_starting=True)
        self._c_mu = self.mu
        self.last_setup_s = time.perf_counter() - t0

    def allocate(self, r_w, b_d, *, mu=None, contact=None) -> ALLOC.Allocation:
        """Eq (4) for this sweep.  `mu` other than the instance's rebuilds C."""
        if mu is not None and float(mu) != self.mu:
            self.mu = float(mu)
        A = ALLOC.grasp_map(r_w)
        b_d = np.asarray(b_d, dtype=float).reshape(6)
        weight = (np.ones(C.N_LEGS) if contact is None
                  else np.clip(np.asarray(contact, dtype=float).reshape(C.N_LEGS),
                               0.0, 1.0))
        P, q = cost(A, b_d, self.f_prev, self.s_diag, self.alpha, self.beta)
        rebuild = self._problem is not None and self._c_mu != self.mu
        c_mat, d_vec = self._constraints(weight)

        t0 = time.perf_counter()
        if self._problem is None:
            self._setup(P, q, c_mat, d_vec)
        else:
            kwargs = dict(Px=self._p_data(P), q=q, u=d_vec)
            if rebuild:
                self._c_csc = sp.csc_matrix(c_mat)
                kwargs["Ax"] = self._c_csc.data
                self._c_mu = self.mu
            self._problem.update(**kwargs)
        result = self._problem.solve()
        self.last_solve_s = time.perf_counter() - t0
        self.last_status = str(result.info.status)
        self.last_iterations = int(result.info.iter)
        if "maximum iterations" in self.last_status:
            self.max_iter_hits += 1

        x = result.x
        if x is None or not np.all(np.isfinite(x)):
            # Cannot happen for an empty-proof constraint set, but a solver
            # that returns None must never put NaN on the CAN bus.
            self.fallbacks += 1
            fallback = ALLOC.allocate(r_w, b_d, mu=self.mu, fz_min=self.fz_min,
                                      fz_max=self.fz_max, contact=contact)
            self.f_prev = fallback.f_w.reshape(-1).copy()
            return fallback
        f_qp = np.asarray(x, dtype=float).reshape(C.N_LEGS, 3)

        # The same hard projection the least-squares path applies, so the
        # output is inside the cone to machine precision rather than to the
        # solver's eps -- and the one thing that still bites a max-iter
        # iterate.  For a converged solve it moves nothing material.
        planted = weight > 0.0
        f_w = f_qp.copy()
        f_w[:, 2] = np.clip(f_w[:, 2], weight * self.fz_min, weight * self.fz_max)
        tangent = np.linalg.norm(f_w[:, :2], axis=1)
        limit = self.mu * f_w[:, 2]
        over = planted & (tangent > limit)
        if np.any(over):
            scale = np.where(over, limit / np.maximum(tangent, 1.0e-12), 1.0)
            f_w[:, :2] *= scale[:, None]
        f_w[~planted] = 0.0

        slack = d_vec - c_mat @ f_w.reshape(-1)            # >= 0 on every row
        # a row is ACTIVE when its slack is within 1e-3 N of zero: OSQP's
        # eps is on the scaled residuals, and a 115 N bound comes back with
        # ~2e-4 N of slack at eps 1e-6
        active = (slack.reshape(C.N_LEGS, ROWS_PER_FOOT) <= 1.0e-3).any(axis=1)
        residual = A @ f_w.reshape(-1) - b_d
        self.f_prev = f_w.reshape(-1).copy()
        return ALLOC.Allocation(f_w=f_w, f_unclipped=f_qp, residual=residual,
                                clipped=planted & active)


# ===========================================================================
# the report: feasibility, what it buys over least squares, and the cost
# ===========================================================================
def _lift_stance():
    from . import state as ST_STATE
    from sim import stand as ST
    from .. import imu as IMU
    q = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)
    body = ST_STATE.read(C.flat(q), np.zeros(C.N_JOINTS),
                         IMU.TrunkOrientation.level())
    return body.r_w


def benchmark(r_w, sweeps: int = 2000, seed: int = 0) -> dict:
    """Warm-started QP against least squares over `sweeps` sweeps of a
    stand that is being pushed about: b_d wanders (±8 N in x/y, ±1 N*m in
    roll/pitch at a few Hz) and r_w tilts ±2 deg with it.  Returns the
    per-call wall times and the worst constraint violation seen."""
    rng = np.random.default_rng(seed)
    qp_alloc = QPAllocator()
    t = np.arange(sweeps) / 250.0
    c_mat, d_vec = constraints(None, qp_alloc.mu, qp_alloc.fz_min, qp_alloc.fz_max)
    qp_t, ls_t, iters, viol, qp_res, ls_res = [], [], [], 0.0, [], []
    for k in range(sweeps):
        b_d = np.array([8.0 * np.sin(2 * np.pi * 0.7 * t[k]),
                        8.0 * np.sin(2 * np.pi * 1.3 * t[k] + 1.0),
                        cfg.WEIGHT + 4.0 * np.sin(2 * np.pi * 2.0 * t[k]),
                        1.0 * np.sin(2 * np.pi * 0.9 * t[k]),
                        1.0 * np.sin(2 * np.pi * 1.1 * t[k] + 2.0),
                        0.0]) + rng.normal(0.0, 0.2, 6)
        tilt = np.deg2rad(2.0 * np.sin(2 * np.pi * 0.5 * t[k]))
        r_k = r_w @ C.rot_x(tilt).T
        t0 = time.perf_counter(); ls = ALLOC.allocate(r_k, b_d); ls_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); qp = qp_alloc.allocate(r_k, b_d); qp_t.append(time.perf_counter() - t0)
        iters.append(qp_alloc.last_iterations)
        viol = max(viol, float((c_mat @ qp.f_w.reshape(-1) - d_vec).max()))
        qp_res.append(qp.residual_force); ls_res.append(ls.residual_force)
    # the first call carries `setup`; it is reported on its own
    return dict(qp_us=1e6 * np.asarray(qp_t[1:]), ls_us=1e6 * np.asarray(ls_t[1:]),
                iters=np.asarray(iters), violation=viol,
                qp_res=np.asarray(qp_res), ls_res=np.asarray(ls_res),
                first_us=1e6 * qp_t[0], setup_us=1e6 * qp_alloc.last_setup_s,
                max_iter_hits=qp_alloc.max_iter_hits, fallbacks=qp_alloc.fallbacks,
                status=qp_alloc.last_status)


def describe() -> str:
    if osqp is None:
        return "DOG6 QP allocator (eq 4): osqp/scipy not installed, nothing to run"
    r_w = _lift_stance()
    lines = ["DOG6 stance force allocator, eq (4): QP in OSQP, warm-started",
             "  S = diag(%.0f x3, %.0f x3)   alpha %.0e   beta %.0e   "
             "mu %.2f   fz [%.1f, %.1f] N   max_iter %d   eps %.0e"
             % (cfg.QP_S_FORCE, cfg.QP_S_MOMENT, cfg.QP_ALPHA, cfg.QP_BETA,
                cfg.MU, cfg.FZ_MIN, cfg.FZ_MAX, cfg.QP_MAX_ITER, cfg.QP_EPS)]

    cases = [("hold mg only", [0, 0, cfg.WEIGHT, 0, 0, 0]),
             ("Fy 25 N, Mx 2 N*m (reachable)", [0, 25, cfg.WEIGHT, 2, 0, 0]),
             ("Fy 40 N, Mx 5 N*m (past capacity)", [0, 40, cfg.WEIGHT, 5, 0, 0])]
    c_mat, d_vec = constraints()
    lines.append("  against least squares, cold, beta term off (one-shot):")
    for name, b in cases:
        b = np.asarray(b, dtype=float)
        ls = ALLOC.allocate(r_w, b)
        qp = QPAllocator(beta=0.0).allocate(r_w, b)
        lines.append("    %-36s LS resid %5.2f N %5.2f N*m | QP resid %5.2f N "
                     "%5.2f N*m  worst C F - d %+.1e  |f|max LS %.1f QP %.1f N"
                     % (name, ls.residual_force, ls.residual_moment,
                        qp.residual_force, qp.residual_moment,
                        (c_mat @ qp.f_w.reshape(-1) - d_vec).max(),
                        np.linalg.norm(ls.f_w, axis=1).max(),
                        np.linalg.norm(qp.f_w, axis=1).max()))

    # beta: the step response of the load SHARE
    hold = np.array([0.0, 0.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
    step = np.array([0.0, 25.0, cfg.WEIGHT, 2.0, 0.0, 0.0])
    for beta in (0.0, cfg.QP_BETA):
        a = QPAllocator(beta=beta)
        a.allocate(r_w, hold)
        final = QPAllocator(beta=0.0).allocate(r_w, step).f_w
        gap = [np.abs(a.allocate(r_w, step).f_w - final).max() for _ in range(60)]
        settled = next((k + 1 for k, g in enumerate(gap) if g < 0.1), None)
        lines.append("  beta %.0e: a 25 N / 2 N*m step settles to 0.1 N in %s sweeps"
                     % (beta, settled if settled else ">60"))

    bench = benchmark(r_w)
    pct = lambda a: tuple(np.percentile(a, (50, 95, 100)))
    lines += ["  2000 sweeps, stand pushed about, this machine:",
              "    QP warm   p50 %6.0f  p95 %6.0f  max %6.0f us   (first call, with "
              "setup: %.0f us)" % (*pct(bench["qp_us"]), bench["first_us"]),
              "    LS        p50 %6.0f  p95 %6.0f  max %6.0f us" % pct(bench["ls_us"]),
              "    iterations p50 %d  p95 %d  max %d   max-iter hits %d   fallbacks %d"
              % (*pct(bench["iters"]), bench["max_iter_hits"], bench["fallbacks"]),
              "    worst constraint violation C F - d  %+.1e   (feasible iff <= 0)"
              % bench["violation"],
              "    force residual p95  QP %.2f N   LS %.2f N"
              % (np.percentile(bench["qp_res"], 95), np.percentile(bench["ls_res"], 95)),
              "    CAN slot 333 us: QP p95 %s" % ("FITS" if np.percentile(bench["qp_us"], 95) < 333
                                                  else "DOES NOT FIT")]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
