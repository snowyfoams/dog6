"""Gate `hw.balance.mpc` and `hw.fold2_cmpc` with no robot, no bus, no MuJoCo.

    python -m hw.balance.mpc_selftest

A file of its own beside `selftest.py` (whose fixtures it borrows): it needs
scipy and osqp, which nothing else in `hw.balance` does.

WHAT THIS CAN AND CANNOT TELL YOU
    CAN     that the MPC drives the foot forces on a walking clock's sweeps
            and on no other; that on a fixed body its forces carry the
            weight on every sweep, handovers included, with a swinging leg at
            exactly zero; that everything else the law makes on those sweeps
            -- the swing leg's torque, the joint layer, the reference -- is
            what `hw.fold2_walk`'s law makes; that a failed QP keeps the last
            good forces; and that the entry point is fold2_walk's options
            with cmpc's clock.
    CANNOT   whether the robot trots under it.  The body here does not move.
"""
from __future__ import annotations

import sys

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/mpc_selftest.py`
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim.cmpc import config as SIMCFG    # noqa: E402

from .. import fold2_cmpc as F2C     # noqa: E402
from .. import fold2_trot as F2T     # noqa: E402
from .. import fold2_walk as F2W     # noqa: E402
from .. import fold_walk as FW       # noqa: E402
from . import config as cfg          # noqa: E402
from . import gait as GAIT           # noqa: E402
from . import law as LAW             # noqa: E402
from . import mpc as MPC             # noqa: E402
from . import posture as POSE        # noqa: E402
from . import reference as REF       # noqa: E402
from . import selftest as ST         # noqa: E402
from . import walk as WWALK          # noqa: E402

check, close = ST.check, ST.close

H = F2T.HEIGHT
F2 = POSE.FOLD2


def body(R=None):
    return ST.state_at(H, R=R, foot_xy=F2.foot_xy, q_seed=F2.q, srb=F2.srb)


def estimate(t, at, p=None, v=None):
    return LAW.TrunkEstimate(
        t=t, p_w=(np.array([0.0, 0.0, at.z_origin]) if p is None
                  else np.asarray(p, dtype=float)),
        v_w=np.zeros(3) if v is None else np.asarray(v, dtype=float),
        yaw_offset=0.0)


def walking_law(at, mpc):
    """fold2_walk's law at FOLD2's hold: damped planted legs, the walk plan."""
    plan = WWALK.WalkPlan()
    law = LAW.BalanceLaw(foot_xy=F2.foot_xy.copy(), dynamic_setpoint=False,
                         srb=F2.srb, track_stop_deg=0.0, est_xy=True,
                         kp_joint=cfg.WALK_KP_JOINT, kd_joint=cfg.KD_JOINT_HOLD,
                         h_lift=H, qp_s=cfg.QP_S_FOLD2_WALK, walk=plan, mpc=mpc)
    law.arm(0.0, at)
    law.ramp = REF.Quintic.ramp(H, H, 1.0)
    law.hold_joints(at.q)
    return law


def cmpc_gait():
    return GAIT.TrotGait(period=MPC.GAIT_PERIOD, duty=MPC.GAIT_DUTY, ramp=0.0,
                         settle_s=0.0)


def main() -> int:
    print("DOG6 MPC-in-the-trot self-test  (no robot, no bus, no MuJoCo)\n")
    at = body()

    print("1. where the MPC runs")
    try:
        LAW.BalanceLaw(foot_xy=F2.foot_xy, srb=F2.srb, mpc=MPC.MpcForces())
        refused = False
    except ValueError:
        refused = True
    check("an MPC with no walk plan is refused (it tracks the walk's "
          "reference)", refused)
    m = MPC.MpcForces()
    m.warm_up()
    law = walking_law(at, m)
    plain = walking_law(at, None)
    for L in (law, plain):
        L.feed_estimate(estimate(0.996, at))
    o_h, p_h = law.update(1.0, at, xy_hold=True), plain.update(1.0, at,
                                                               xy_hold=True)
    check("HOLD: the MPC is not called, the forces and torques are "
          "fold2_walk's to the bit",
          m.solves == 0 and m.sweeps == 0 and np.isnan(o_h.mpc_ms)
          and np.array_equal(o_h.allocation.f_w, p_h.allocation.f_w)
          and np.array_equal(o_h.tau, p_h.tau))

    print("\n2. the trot, on cmpc's clock, a fixed body")
    g, gp = cmpc_gait(), cmpc_gait()
    check("the clock is sim.cmpc's: %.2f s at duty %.2f, no ramp, no settle"
          % (SIMCFG.GAIT_PERIOD, SIMCFG.DUTY),
          g.period == SIMCFG.GAIT_PERIOD and g.duty == SIMCFG.DUTY
          and g.ramp == 0.0 and g.settle_s == 0.0)
    g.reset(1.004)
    gp.reset(1.004)
    t = 1.004
    fz_sum, swing_f, trips, same_swing, same_ref = [], 0.0, set(), True, True
    solved_first = None
    swung = set()
    while t < 1.004 + 3.0 * g.period:
        for L in (law, plain):
            L.feed_estimate(estimate(t - 0.004, at))
        o = law.update(t, at, gait=g, xy_hold=True)
        p = plain.update(t, at, gait=gp, xy_hold=True)
        if solved_first is None:
            solved_first = bool(np.isfinite(o.mpc_ms))
        if o.trip:
            trips.add(o.trip)
        down = g.contact(t)
        swung |= set(np.flatnonzero(~down).tolist())
        fz_sum.append(float(o.allocation.fz.sum()))
        swing_f = max(swing_f, float(np.abs(o.allocation.f_w[~down]).max()))
        for i in np.flatnonzero(~down):
            rows = slice(3 * i, 3 * i + 3)
            same_swing &= bool(np.array_equal(o.tau[rows], p.tau[rows]))
        same_ref &= bool(np.array_equal(o.ref.p, p.ref.p)
                         and np.array_equal(o.p_swing, p.p_swing,
                                            equal_nan=True)
                         and np.array_equal(o.q_ref, p.q_ref))
        t += 0.004
    fz_sum = np.asarray(fz_sum)
    check("the trot's first sweep solves", bool(solved_first))
    check("every leg swung, nothing tripped", swung == {0, 1, 2, 3}
          and not trips, "trips %s" % (trips or "none"))
    close("the MPC's forces carry the weight on EVERY sweep, handovers "
          "included", fz_sum, np.full(fz_sum.size, cfg.WEIGHT),
          0.01 * cfg.WEIGHT, " N")
    check("a swinging leg's force is exactly zero", swing_f == 0.0)
    check("a handover is a solve of its own", m.handover_solves >= 5,
          "%d handovers solved, %d solves in all" % (m.handover_solves,
                                                     m.solves))
    check("the swing leg's torque is fold2_walk's to the bit: the MPC does "
          "not touch it", same_swing)
    check("...and so are the reference, the swing targets and the tracking "
          "reference", same_ref)
    check("no solve failed", m.failures == 0 and m.status in MPC.QP_OK_STATUSES)
    print("  " + m.report().strip())

    print("\n3. what it does about an error")
    tilted = body(R=C.rot_zyx(np.radians(3.0), 0.0, 0.0))
    law.feed_estimate(estimate(t - 0.004, tilted))
    m.t_next = None
    o = law.update(t, tilted, gait=g, xy_hold=True)
    moment = np.zeros(3)
    for i in range(C.N_LEGS):
        moment += np.cross(tilted.r_w[i], o.allocation.f_w[i])
    check("3 deg of roll: a restoring roll moment, the SRB wrench's sign",
          moment[0] * o.wrench.b_d[3] > 0.0 and moment[0] < 0.0,
          "MPC %+.2f N*m, the SRB law would ask %+.2f" % (moment[0],
                                                           o.wrench.b_d[3]))
    t += 0.004
    law.feed_estimate(None)
    m.t_next = None
    o = law.update(t, at, gait=g, xy_hold=True)
    check("no estimate: the x/y error reads zero, and the forces still carry "
          "the weight", abs(float(o.allocation.fz.sum()) - cfg.WEIGHT)
          < 0.01 * cfg.WEIGHT
          and float(np.abs(o.allocation.f_w[:, :2]).max()) < 1.0,
          "|f_xy| max %.2f N" % np.abs(o.allocation.f_w[:, :2]).max())
    t += 0.004
    good = m.forces.copy()
    real = m.solver.solve
    def failing(*args, **kwargs):
        real(*args, **kwargs)
        m.solver.last_status = "primal infeasible"
        return np.zeros(m.solver.n_vars)
    m.solver.solve = failing
    m.t_next = None
    law.feed_estimate(estimate(t - 0.004, at))
    o = law.update(t, at, gait=g, xy_hold=True)
    m.solver.solve = real
    check("a QP that does not solve keeps the last good forces, counted, "
          "no trip", m.failures == 1 and np.array_equal(m.forces, good)
          and o.trip is None)

    print("\n4. leaving the trot")
    t += 0.004
    law.feed_estimate(estimate(t - 0.004, at))
    plain.feed_estimate(estimate(t - 0.004, at))
    o, p = law.update(t, at, xy_hold=True), plain.update(t, at, xy_hold=True)
    check("back in HOLD the allocator is fold2_walk's again and the MPC "
          "forgets its forces", np.array_equal(o.allocation.f_w,
                                               p.allocation.f_w)
          and m.t_next is None and not m.forces.any()
          and np.isnan(o.mpc_ms))

    print("\n5. hw.fold2_cmpc: fold2_walk's options, cmpc's clock")
    co, wo = F2C.cmpc_options(), F2W.walk_options()
    same = [k for k in wo if k not in ("gait", "hook")
            and (co[k] is wo[k] or np.array_equal(co[k], wo[k]))]
    check("every option but the clock and the hook is hw.fold2_walk's",
          set(co) == set(wo)
          and len(same) == len([k for k in wo if k not in ("gait", "hook")]),
          "%d options carried: crouch %s, height %.0f mm"
          % (len(same), co["crouch"].name, 1e3 * co["height"]))
    check("the clock is cmpc's and the hook is the walk's, plus the MPC",
          co["gait"].period == SIMCFG.GAIT_PERIOD
          and co["gait"].duty == SIMCFG.DUTY and co["gait"].ramp == 0.0
          and co["gait"].settle_s == 0.0
          and isinstance(co["hook"], F2C.CmpcHook)
          and isinstance(co["hook"], FW.WalkHook)
          and np.array_equal(co["gait"].offsets, wo["gait"].offsets))

    failures, passes = ST._FAILURES, ST._PASSES
    print("\n%d checks, %d failed" % (passes + len(failures), len(failures)))
    for name in failures:
        print("  FAILED: %s" % name)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
