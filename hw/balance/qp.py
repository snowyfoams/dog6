"""Stage 4 as a QP: split the wrench across the planted feet INSIDE the cone.

    QpAllocator.allocate(r_w, b_d, contact=w)  -> allocation.Allocation

    min_f  1/2 |A f - b_d|^2_S  +  1/2 alpha f' W f  +  1/2 beta |f - f_prev|^2
    s.t.   |fx_i| <= mu fz_i,  |fy_i| <= mu fz_i        the pyramid, per foot
           w_i fz_min <= fz_i <= w_i fz_max             the contact ramp's box
           f_i = 0 for a swinging foot                   (not a variable at all)

WHAT IT CHANGES, AND WHAT IT DOES NOT
    `allocation.allocate` solves A f = b_d exactly by weighted least squares
    and THEN clips each foot into the cone and rescales the total -- so the
    wrench it delivers is whatever the clip left, and the clip knows nothing
    about the wrench.  This puts the cone and the box inside the problem: the
    delivered wrench is the closest one, in the S-metric, that the cone can
    make.  With every constraint slack the two agree (alpha -> 0, S = I is
    the least squares' own metric; `selftest` checks it), so a four-foot
    stand well inside the cone is the same law.  Where they differ is
    exactly where the old one was giving up wrench by accident: a foot
    ramping out, a diagonal pair asked for a moment about its own line, a
    push that puts a foot on the cone.

    THE DEFAULT SINCE 2026-10-04 (`law.BalanceLaw.alloc`, `hw.stand --alloc`),
    on request, for the robot: the cone is a constraint here, where the least
    squares meets the wrench first and clips after.  `--alloc wls` keeps it.

THE COST OF ENFORCING THE CONE HARD, MEASURED ON DOG5 (allocation.py's
    docstring): the QP's torque peaks were p95 2.73 against 1.61 N*m for
    the min-norm allocator in the same trot stance -- a QP that must meet
    the wrench pushes forces onto the cone's faces.  The S weights are the
    knob for that: S = I gives up the unreachable moment the way the least
    squares did (config.QP_S says why), so the comparison is like for like.

THE SOLVER: A DENSE PRIMAL ACTIVE SET, NOT OSQP
    `sim.cmpc.qp` uses OSQP; it is not on the robot's Pi and an ADMM's
    iteration count is the wrong shape for a 333 us slot anyway.  This
    problem is tiny -- at most 12 variables, 6 rows a foot -- and on most
    sweeps no face is touched at all: the unconstrained optimum, one 12x12
    solve, IS the answer, and that is checked first (faster than the least
    squares, ~80 against ~120 us on the same machine).  When a face is
    touched, a primal active set starts from the unconstrained optimum
    PROJECTED into the cone, with the clipped rows as its working set, and
    takes a few 24x24 solves, ~70 us an iteration -- numpy's per-call cost,
    not the solve: a Schur-complement step (an m x m solve, H inverted once)
    was tried, gave the same iterates to 1e-9 N and saved nothing.  Every
    iterate stays feasible and none is worse than the one before, so the cap
    (`config.QP_MAX_ITER`, 20: ~1.5 ms) returns a force the cone allows even
    if it stops early -- suboptimal, never infeasible.

    Nocedal & Wright, Numerical Optimization, 2nd ed., Alg. 16.3.

A SWINGING FOOT IS NOT IN THE PROBLEM
    As in the least squares' trot path: weight 0 means no variables, so its
    force is exactly zero, not a one-in-a-million share and not fz_min.  A
    planted foot's weight w scales its box to [w fz_min, w fz_max] -- DOG5's
    rule, `allocation.allocate` has the measurement -- and divides its
    regularisation, so a foot ramping out is both bounded and expensive.

THE PYRAMID IS mu ON EACH AXIS, sim.cmpc.qp's ROWS.  Its corners reach
    sqrt(2) mu: outside the round cone `allocation` projects onto.  mu here is
    `config.MU` = 0.5, already below the cMPC stack's 0.6 for the
    2026-09-15 slides; forces at a trot's stance are within a few degrees of
    vertical, far from either.
"""
from __future__ import annotations

import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/qp.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from . import config as cfg          # noqa: E402
from .allocation import Allocation   # noqa: E402

__all__ = ["QpAllocator", "pyramid_rows", "project_feasible"]

#: Rows per foot, acting on (fx, fy, fz): the four pyramid faces, then the
#: box on fz as two one-sided rows.  C f <= d.
ROWS_PER_FOOT = 6

#: A foot below this contact weight is out of the problem, as a swinging one
#: is.  Its box would be [w fz_min, w fz_max] = at most 0.12 N wide, and at
#: the ramp's far end (smoothstep weights reach 1e-22) the box collapses onto
#: fz = 0, where OPPOSITE pyramid faces both hold and the active set has no
#: independent set to stand on -- the cycle the walk's first MuJoCo runs hit
#: (2026-10-04).  0.12 N of a 57.7 N robot is nothing to give up for it.
W_PLANTED_MIN = 1.0e-3


def pyramid_rows(mu: float) -> np.ndarray:
    """(6, 3) C for one foot: four faces, fz <= hi, -fz <= -lo."""
    mu = float(mu)
    return np.array([[1.0, 0.0, -mu],
                     [-1.0, 0.0, -mu],
                     [0.0, 1.0, -mu],
                     [0.0, -1.0, -mu],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, -1.0]])


def project_feasible(f, lo, hi, mu: float) -> np.ndarray:
    """(n, 3) each foot pushed into its box and pyramid: fz clipped first,
    then fx and fy clipped to +-mu fz.  Feasible by construction -- which is
    all the active set asks of its starting point."""
    f = np.array(f, dtype=float).reshape(-1, 3)
    f[:, 2] = np.clip(f[:, 2], lo, hi)
    lim = mu * f[:, 2]
    f[:, 0] = np.clip(f[:, 0], -lim, lim)
    f[:, 1] = np.clip(f[:, 1], -lim, lim)
    return f


class QpAllocator:
    """The QP and its counters.  One per law."""

    def __init__(self, *, mu: float = cfg.MU, fz_min: float = cfg.FZ_MIN,
                 fz_max: float = cfg.FZ_MAX, s_diag=None,
                 alpha: float = cfg.QP_ALPHA, beta: float = cfg.QP_BETA,
                 max_iter: int = cfg.QP_MAX_ITER):
        self.mu = float(mu)
        self.fz_min, self.fz_max = float(fz_min), float(fz_max)
        self.s_diag = (np.asarray(cfg.QP_S, dtype=float) if s_diag is None
                       else np.asarray(s_diag, dtype=float).reshape(6))
        self.alpha, self.beta = float(alpha), float(beta)
        self.max_iter = int(max_iter)
        self.per_foot_w = np.array([cfg.W_TANGENTIAL, cfg.W_TANGENTIAL,
                                    cfg.W_NORMAL])
        # Built once per number of planted feet: the force rows of A, the
        # constraint matrix (block diagonal), the diagonal index.
        rows = pyramid_rows(self.mu)
        self._eye_tiles = {n: np.tile(np.eye(3), n) for n in range(1, 5)}
        self._c_blocks = {n: np.kron(np.eye(n), rows) for n in range(1, 5)}
        self._diag = {n: np.diag_indices(3 * n) for n in range(1, 5)}
        self._ones = np.ones(C.N_LEGS)
        #: (4, 3) the last solution, all four feet (swing feet zero).
        self.f_prev = np.zeros((C.N_LEGS, 3))
        #: The last working set as (leg, row) pairs: which faces held.
        self.active: set[tuple[int, int]] = set()
        #: Diagnostics: the last solve's iterations and whether it converged,
        #: and running counts for the exit report.
        self.last_iter = 0
        self.last_converged = True
        self.solves = 0
        self.capped = 0
        self.iter_max_seen = 0

    def reset(self) -> None:
        self.f_prev[:] = 0.0
        self.active = set()

    # -- the problem --------------------------------------------------------
    def _build(self, r_w, b_d, legs, weight):
        """H, g, A, C, d, lo, hi for the planted `legs` (an index array)."""
        n = legs.size
        r = r_w[legs]
        A = np.zeros((6, 3 * n))
        A[0:3] = self._eye_tiles[n]
        # Column by column, r x f for f along x, y, z -- `hat(r)` per foot,
        # written into the strided columns at once.
        A[3, 1::3], A[3, 2::3] = -r[:, 2], r[:, 1]
        A[4, 0::3], A[4, 2::3] = r[:, 2], -r[:, 0]
        A[5, 0::3], A[5, 1::3] = -r[:, 1], r[:, 0]
        w = weight[legs]
        w_reg = (np.maximum(w, 1e-3)[:, None] ** -1 * self.per_foot_w).reshape(-1)
        AtS = A.T * self.s_diag                       # A' S, S diagonal
        H = AtS @ A
        H[self._diag[n]] += self.alpha * w_reg + self.beta
        g = -(AtS @ b_d)
        if self.beta:
            g -= self.beta * self.f_prev[legs].reshape(-1)
        lo, hi = w * self.fz_min, w * self.fz_max
        d = np.zeros(ROWS_PER_FOOT * n)
        d[4::ROWS_PER_FOOT] = hi
        d[5::ROWS_PER_FOOT] = -lo
        return H, g, A, self._c_blocks[n], d, lo, hi

    # -- the solve ----------------------------------------------------------
    def allocate(self, r_w, b_d, *, contact=None) -> Allocation:
        """The QP for this sweep.  `contact` is the clock's (4,) weights, None
        for four feet at weight 1 (a stand).  Same return type, frame and
        sign as `allocation.allocate`: ground reaction ON THE ROBOT, WORLD."""
        r_w = np.asarray(r_w, dtype=float).reshape(C.N_LEGS, 3)
        b_d = np.asarray(b_d, dtype=float).reshape(6)
        weight = (self._ones if contact is None else
                  np.clip(np.asarray(contact, dtype=float).reshape(C.N_LEGS),
                          0.0, 1.0))
        legs = np.flatnonzero(weight >= W_PLANTED_MIN)
        f_w = np.zeros((C.N_LEGS, 3))
        f_free = np.zeros((C.N_LEGS, 3))
        clipped = np.zeros(C.N_LEGS, dtype=bool)
        if legs.size == 0:
            self.reset()
            return Allocation(f_w=f_w, f_unclipped=f_free, residual=-b_d,
                              clipped=clipped)
        H, g, A, Cm, d, lo, hi = self._build(r_w, b_d, legs, weight)
        nv = H.shape[0]
        x_free = np.linalg.solve(H, -g)
        f_free[legs] = x_free.reshape(-1, 3)

        # -- THE COMMON CASE: every face and bound slack.  Then the
        # unconstrained optimum is the QP's, and there is nothing to iterate.
        if bool(np.all(Cm @ x_free <= d + 1e-12)):
            x, work, it, converged = x_free, [], 0, True
        else:
            x, work, it, converged = self._active_set(H, g, Cm, d, lo, hi,
                                                      legs, x_free, nv)
            # Exactly feasible on the way out: the regularised KKT can leave
            # a working row a hair over its bound, and the drivers get this.
            x = project_feasible(x.reshape(-1, 3), lo, hi, self.mu).reshape(-1)
        f_w[legs] = x.reshape(-1, 3)
        self.f_prev = f_w
        self.active = {(int(legs[j // ROWS_PER_FOOT]), int(j % ROWS_PER_FOOT))
                       for j in work}
        for j in work:
            clipped[legs[j // ROWS_PER_FOOT]] = True
        self.last_iter = it
        self.last_converged = converged
        self.solves += 1
        self.capped += 0 if converged else 1
        self.iter_max_seen = max(self.iter_max_seen, it)
        residual = A @ x - b_d
        return Allocation(f_w=f_w, f_unclipped=f_free, residual=residual,
                          clipped=clipped)

    def _active_set(self, H, g, Cm, d, lo, hi, legs, x_free, nv):
        """Nocedal & Wright Alg. 16.3 from the PROJECTED unconstrained optimum.

        THE START IS THIS SWEEP'S OWN, NOT LAST SWEEP'S.  The unconstrained
        optimum pushed into each foot's box and pyramid is feasible, and the
        rows the projection clipped are the working set.  Warm starting from
        the previous sweep's forces and faces instead was measured worse,
        because the hard sweeps are exactly the ones where the problem moved:
        on the 77 slowest solves of a MuJoCo walk (doc/walk) it took 9.8
        iterations on average and 38 at worst, this 3.2 and 13."""
        x = project_feasible(x_free.reshape(-1, 3), lo, hi,
                             self.mu).reshape(-1)
        slack = d - Cm @ x
        tol_act = 1e-9 * (1.0 + np.abs(d))
        work = self._independent([int(j) for j in
                                  np.flatnonzero(slack <= tol_act)])
        converged = False
        it = 0
        #: Rows that blocked a ZERO step but are dependent on the working set
        #: -- numerically blocking, structurally implied.  Excluded after the
        #: first time, so a degenerate vertex cannot stall the loop.
        implied = np.zeros(Cm.shape[0], dtype=bool)
        for it in range(1, self.max_iter + 1):
            grad = H @ x + g
            m = len(work)
            if m == 0:
                p = -np.linalg.solve(H, grad)
                lam = None
            else:
                Cw = Cm[work]
                K = np.zeros((nv + m, nv + m))
                K[:nv, :nv] = H
                K[:nv, nv:] = Cw.T
                K[nv:, :nv] = Cw
                # A whisker on the multiplier block keeps K solvable if two
                # working rows ever become nearly dependent.
                K[nv:, nv:] = -1e-12 * np.eye(m)
                sol = np.linalg.solve(K, np.concatenate([-grad, np.zeros(m)]))
                p, lam = sol[:nv], sol[nv:]
            if float(np.abs(p).max()) <= 1e-10 * (1.0 + float(np.abs(x).max())):
                if lam is None or float(lam.min()) >= -1e-10:
                    converged = True
                    break
                work.pop(int(np.argmin(lam)))         # leave that face
                continue
            Cp = Cm @ p
            slack = d - Cm @ x
            step, block = 1.0, None
            # RELATIVE to the step: every row of C is O(1), so a C p a billionth
            # of |p| is the arithmetic's, not the direction's.
            open_ = Cp > 1e-9 * (1.0 + float(np.abs(p).max()))
            open_[work] = False
            open_ &= ~implied
            cand = np.flatnonzero(open_)
            if cand.size:
                ratios = np.maximum(slack[cand], 0.0) / Cp[cand]
                k = int(np.argmin(ratios))
                if ratios[k] < 1.0:
                    step, block = float(ratios[k]), int(cand[k])
            x = x + step * p
            if block is not None:
                grown = self._independent(work + [block])
                if len(grown) == len(work):
                    implied[block] = True
                work = grown
        return x, work, it, converged

    @staticmethod
    def _independent(work) -> list[int]:
        """`work` with dependent rows dropped, by STRUCTURE, no factorisation.

        A foot's rows pair up: x faces (0, 1), y faces (2, 3), the fz box
        (4, 5).  Two of a pair cannot both hold -- opposite faces meet only at
        fz = 0, below the box's floor, and the box's two sides only if lo =
        hi.  And one row from each pair is ALWAYS independent: the 3x3 of
        (+-1, 0, -mu), (0, +-1, -mu), (0, 0, +-1) is triangular with a unit
        diagonal.  So keeping the first of each pair per foot is the whole
        test."""
        keep, seen = [], set()
        for j in work:
            key = (j // ROWS_PER_FOOT, (j % ROWS_PER_FOOT) // 2)
            if key not in seen:
                seen.add(key)
                keep.append(j)
        return keep

    def report(self) -> str:
        return ("qp allocator    %d solves, iterations max %d, %d stopped at "
                "the cap (last feasible iterate used)"
                % (self.solves, self.iter_max_seen, self.capped))


def describe() -> str:
    """The QP beside the least squares, in a stand and on a diagonal pair."""
    from . import allocation as ALLOC
    from . import state as ST_STATE
    from .. import imu as IMU
    from sim import stand as ST
    q = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)
    body = ST_STATE.read(C.flat(q), np.zeros(C.N_JOINTS),
                         IMU.TrunkOrientation.level())
    qp = QpAllocator()
    rows = ["DOG6 QP force allocator (dense primal active set, projected start)",
            "  S %s  alpha %.0e  beta %.0e  mu %.2f  cap %d iterations"
            % (np.array2string(qp.s_diag, precision=0), qp.alpha, qp.beta,
               qp.mu, qp.max_iter)]
    cases = (("stand, weight only", np.array([0, 0, cfg.WEIGHT, 0, 0, 0.0]),
              None),
             ("stand, 1 N*m roll + 1 N*m pitch",
              np.array([0, 0, cfg.WEIGHT, 1.0, 1.0, 0.0]), None),
             ("diagonal FL+RR, 0.5 N*m roll",
              np.array([0, 0, cfg.WEIGHT, 0.5, 0.0, 0.0]),
              np.array([1.0, 0.0, 0.0, 1.0])),
             ("stand, 20 N sideways (cone-bound)",
              np.array([0, 20.0, cfg.WEIGHT, 0, 0, 0.0]), None))
    for label, b_d, contact in cases:
        qp.reset()
        t0 = time.perf_counter()
        a_qp = qp.allocate(body.r_w, b_d, contact=contact)
        t_qp = time.perf_counter() - t0
        it_cold = qp.last_iter
        t0 = time.perf_counter()
        a_qp2 = qp.allocate(body.r_w, b_d, contact=contact)   # caches warm
        t_warm = time.perf_counter() - t0
        a_ls = ALLOC.allocate(body.r_w, b_d, contact=contact)
        rows.append("  %-36s qp |res| %.3f N / %.3f N*m (%d it cold %.0f us, "
                    "%d it warm %.0f us)   wls %.3f N / %.3f N*m"
                    % (label, a_qp.residual_force, a_qp.residual_moment,
                       it_cold, 1e6 * t_qp, qp.last_iter,
                       1e6 * t_warm, a_ls.residual_force,
                       a_ls.residual_moment))
        rows.append("  %36s fz qp %s  wls %s" % (
            "", np.array2string(a_qp2.fz, precision=1),
            np.array2string(a_ls.fz, precision=1)))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
