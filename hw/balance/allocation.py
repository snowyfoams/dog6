"""Stage 4: split one wrench across four feet.  Pure transmission, no gains.

    b_d (6,)  --grasp map + weighted least squares + cone-->  f_i^w (4, 3)

`controller` decides how much wrench the body needs.  This decides HOW that
wrench is produced: which foot pushes how hard, and in which direction.  There
is no trunk feedback anywhere in this file and no state except the previous
solution's residual streak.

SIX EQUATIONS, TWELVE UNKNOWNS
        A f^w = b_d,    A = [ I    I    I    I  ]
                            [ r1^  r2^  r3^  r4^]                  (6 x 12)

    The system is under-determined and the six-dimensional solution space is
    the INTERNAL-FORCE null space: the forces the four feet can apply against
    each other that produce no net wrench on the body at all.  Four independent
    leg controllers pick a point in it with no criterion whatsoever -- that is
    what the per-leg law was doing, and it is the part of its behaviour nobody
    could name.  The allocator's job is to pick that point DELIBERATELY.

THE MOMENT ARMS ARE ABOUT THE CoM, IN THE WORLD
        r_i^w = R (x_i^b - c^b)

    Building the grasp map about the trunk geometric centre instead puts a
    constant bias in the moment row that looks exactly like a calibration
    error -- and on DOG6 c^b is 15 to 22 mm from the origin, against a 65 mm
    roll lever.  `state` computes r_w; this file only consumes it.

WEIGHTED LEAST SQUARES, NOT A QP, AND THE REASON IS MEASURED
        min 1/2 f^T W f  s.t.  A f = b_d
        f* = W^-1 A^T (A W^-1 A^T + lambda I)^-1 b_d

    The only inverse is 6x6 with W block-diagonal: closed form, fixed cost, no
    iteration.  That is what makes it safe inside a 333 us slot, and it is the
    whole argument for it over the QP the Balance Controller paper writes.

    ON DOG5, IN THE SAME TWO-LEG TROT STANCE, per-sweep torque peaks were
    p95 2.73 / max 3.90 N*m for the QP against 1.61 / 1.74 N*m for the
    weighted min-norm allocator -- same robot, same battery, same hour.  A QP
    that enforces the cone HARD pushes forces onto the cone boundary to meet
    b_d; least squares with a tangential weight stays inside it and gives up
    some wrench accuracy instead.  For a stand the nominal solution sits well
    inside the cone, so the two agree closely and the QP's advantage is small
    while its iteration count in a 333 us slot is unbounded.

    MOVE TO THE QP WHEN THE LOGGED RESIDUAL SAYS LEAST SQUARES IS GIVING UP
    REAL WRENCH, AND NOT BEFORE.  That is what `Allocation.residual` is for.

W CARRIES THREE JOBS
    Tangential weight above normal weight biases the solution towards vertical
    force, which is a SOFT FRICTION CONE at zero cost.  Per-leg scaling adjusts
    load sharing.  And per-leg scaling by a contact weight in [0, 1] is how a
    foot is faded out continuously -- which matters for a trot and not for a
    four-foot stand, where every weight is 1.

THE PROJECTION BREAKS THE EQUALITY, AND NOTHING COMPENSATES
    After the cone projection A f_proj != b_d in general.  The residual is
    absorbed by the PD on the NEXT sweep -- that is the dashed return in the
    doc's flow chart, and it is not feedback: this allocator has no state.
    Log the residual every sweep.  It is the number that says whether least
    squares is sufficient, and it is the early warning for a stance the ground
    cannot actually supply.

    Note that fz_min >= 0 in the clip is the UNILATERAL CONTACT constraint the
    per-leg law does not have: nothing in a Cartesian spring stops it
    commanding a foot to pull upward on the floor.

    Because the whole problem is in the WORLD frame the cone is axis-aligned
    and the projection is two lines with no rotations in it.  In the body
    frame it is a tilted pyramid whose face normals have to be rebuilt every
    sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/allocation.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from ..kinematics import hat         # noqa: E402
from . import config as cfg          # noqa: E402

__all__ = ["grasp_map", "weight_vector", "Allocation", "allocate",
           "ResidualMonitor", "moment_capacity"]


def grasp_map(r_w) -> np.ndarray:
    """A (6, 12) from the four world-frame moment arms.  `r_w` is (4, 3)."""
    r_w = np.asarray(r_w, dtype=float).reshape(C.N_LEGS, 3)
    A = np.empty((6, 3 * C.N_LEGS))
    for i in range(C.N_LEGS):
        A[0:3, 3 * i:3 * i + 3] = np.eye(3)
        A[3:6, 3 * i:3 * i + 3] = hat(r_w[i])
    return A


def weight_vector(contact=None) -> np.ndarray:
    """The (12,) diagonal of W: (wt, wt, wn) per foot, scaled by contact.

    A contact weight of 0 would divide by zero in W^-1, so it is floored --
    fading a foot out means making it very expensive to use, not making it
    undefined.  At the floor the foot's share of any solution is 1e-6 of what
    an equal foot would take, which is off by any measure that matters.
    """
    contact = cfg.CONTACT_WEIGHT if contact is None else np.asarray(
        contact, dtype=float).reshape(C.N_LEGS)
    scale = 1.0 / np.maximum(contact, 1.0e-6)
    per_foot = np.array([cfg.W_TANGENTIAL, cfg.W_TANGENTIAL, cfg.W_NORMAL])
    return np.concatenate([scale[i] * per_foot for i in range(C.N_LEGS)])


@dataclass(frozen=True)
class Allocation:
    """What the allocator produced, and how far short it fell."""

    f_w: np.ndarray          # (4, 3) ground reaction ON THE ROBOT, WORLD frame
    f_unclipped: np.ndarray  # (4, 3) before the cone -- the least-squares point
    residual: np.ndarray     # (6,) A f_proj - b_d.  Zero iff nothing clipped
    clipped: np.ndarray      # (4,) bool, which feet the cone actually moved

    @property
    def residual_force(self) -> float:
        """|force rows| in newtons."""
        return float(np.linalg.norm(self.residual[:3]))

    @property
    def residual_moment(self) -> float:
        """|moment rows| in newton-metres.  Kept SEPARATE from the force rows:
        a single norm over both is a number with no units."""
        return float(np.linalg.norm(self.residual[3:]))

    @property
    def fz(self) -> np.ndarray:
        """(4,) the normal force under each foot -- the load split."""
        return self.f_w[:, 2]


def allocate(r_w, b_d, *, mu=None, fz_min=None, fz_max=None,
             contact=None, lam=None) -> Allocation:
    """Solve A f = b_d by weighted least squares, then project onto the cone.

    Returns the ground reaction ON THE ROBOT, in the WORLD frame -- which is
    the sign convention `torque` expects and the opposite of the one
    `sim.kinematics.foot_force_to_torque` takes.  A stance foot holding the
    robot up gets a POSITIVE z here.

    MEASURED: 34 us for the whole function on this laptop -- the grasp map
    fill, the 6x6 normal matrix, the solve and four projections.  Per-call
    numpy overhead dominates the arithmetic at this size.
    """
    mu = cfg.MU if mu is None else float(mu)
    fz_min = cfg.FZ_MIN if fz_min is None else float(fz_min)
    fz_max = cfg.FZ_MAX if fz_max is None else float(fz_max)
    lam = cfg.LAMBDA if lam is None else float(lam)

    A = grasp_map(r_w)
    b_d = np.asarray(b_d, dtype=float).reshape(6)
    w_inv = 1.0 / weight_vector(contact)

    # (A W^-1 A^T + lambda I) y = b_d, then f = W^-1 A^T y.
    #
    # A 6x6 SOLVE, AND IT IS AN LU RATHER THAN THE CHOLESKY THE DERIVATION
    # NAMES.  The matrix is symmetric positive definite for lambda > 0, so a
    # Cholesky is the textbook factorisation -- but numpy exposes no
    # triangular solve, so taking it would mean `cholesky` plus two general
    # solves, which is slower than one LU at n = 6.  `selftest` checks the
    # matrix is actually positive definite; the run does not pay to re-check
    # it every sweep.
    normal = (A * w_inv) @ A.T
    normal[np.diag_indices(6)] += lam
    f_flat = w_inv * (A.T @ np.linalg.solve(normal, b_d))
    f_unclipped = f_flat.reshape(C.N_LEGS, 3)

    # -- the cone, axis-aligned because the frame is world --------------
    f_w = f_unclipped.copy()
    f_w[:, 2] = np.clip(f_w[:, 2], fz_min, fz_max)
    tangent = np.linalg.norm(f_w[:, :2], axis=1)
    limit = mu * f_w[:, 2]
    over = tangent > limit
    if np.any(over):
        scale = np.where(over, limit / np.maximum(tangent, 1.0e-12), 1.0)
        f_w[:, :2] *= scale[:, None]

    residual = A @ f_w.reshape(-1) - b_d
    clipped = over | (f_unclipped[:, 2] != f_w[:, 2])
    return Allocation(f_w=f_w, f_unclipped=f_unclipped, residual=residual,
                      clipped=clipped)


class ResidualMonitor:
    """Sustained-residual trip.  One call per control decision; it counts calls.

    INSTANTANEOUS IS THE WRONG TEST.  A sweep or two on the cone boundary
    during a transient is normal and says nothing; a residual that persists
    says the stance cannot supply what the law is asking for, which is the
    early warning for a contact that is about to go.  So it counts CONSECUTIVE
    sweeps over threshold and resets on the first one under, the same shape as
    `safety.SafetyGate`'s two-witness overspeed streak and for the same
    reason.
    """

    def __init__(self, *, force_n=None, moment_nm=None, streak=None):
        self.force_n = cfg.RESIDUAL_FORCE_N if force_n is None else float(force_n)
        self.moment_nm = (cfg.RESIDUAL_MOMENT_NM if moment_nm is None
                          else float(moment_nm))
        self.streak_limit = cfg.RESIDUAL_STREAK if streak is None else int(streak)
        self.streak = 0
        self.peak_force = 0.0
        self.peak_moment = 0.0

    def reason(self, allocation: Allocation) -> str | None:
        """A trip reason, or None.  Also accumulates the peaks for the report."""
        force = allocation.residual_force
        moment = allocation.residual_moment
        self.peak_force = max(self.peak_force, force)
        self.peak_moment = max(self.peak_moment, moment)
        if force > self.force_n or moment > self.moment_nm:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak < self.streak_limit:
            return None
        return ("allocator residual %.1f N / %.2f N*m for %d consecutive "
                "sweeps (limits %.1f / %.2f): the stance cannot supply the "
                "wrench the law is asking for"
                % (force, moment, self.streak, self.force_n, self.moment_nm))


def moment_capacity(r_w, weight: float = None) -> tuple[float, float]:
    """(Mx, My) the stance could produce at best, given it must carry `weight`.

    The bound a flat-ground stance has: shifting the whole load onto the feet
    furthest out in that axis.  Not used by the law -- it is printed in the
    banner, because knowing the roll axis saturates at a quarter of the pitch
    axis is the difference between reading a gain and understanding it.
    """
    weight = cfg.WEIGHT if weight is None else float(weight)
    r_w = np.asarray(r_w, dtype=float).reshape(C.N_LEGS, 3)
    return (weight * float(np.abs(r_w[:, 1]).max()),
            weight * float(np.abs(r_w[:, 0]).max()))


def describe() -> str:
    from . import state as ST_STATE
    from sim import stand as ST
    from .. import imu as IMU
    q = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)
    body = ST_STATE.read(C.flat(q), np.zeros(C.N_JOINTS),
                         IMU.TrunkOrientation.level())
    b_d = np.array([0.0, 0.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
    result = allocate(body.r_w, b_d)
    mx, my = moment_capacity(body.r_w)
    return "\n".join([
        "DOG6 stance force allocator: weighted least squares + cone",
        "  at the lift pose, holding the robot up and nothing else:",
        "    fz per foot  %s N" % np.array2string(result.fz, precision=2),
        "    residual     %.2e N / %.2e N*m  (nothing clipped: %s)"
        % (result.residual_force, result.residual_moment,
           not result.clipped.any()),
        "  moment capacity of this stance at %.1f N: roll %.1f, pitch %.1f N*m"
        % (cfg.WEIGHT, mx, my),
        "  W (t, t, n) = (%.0f, %.0f, %.0f) -- a soft cone at zero cost"
        % (cfg.W_TANGENTIAL, cfg.W_TANGENTIAL, cfg.W_NORMAL),
        "  hard cone mu %.2f, fz in [%.1f, %.1f] N" % (cfg.MU, cfg.FZ_MIN,
                                                       cfg.FZ_MAX),
    ])


if __name__ == "__main__":
    print(describe())
