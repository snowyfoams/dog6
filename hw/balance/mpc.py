"""The convex MPC as the TROT's force law: `sim.cmpc`'s QP in the place of
stages 3-4 (the SRB wrench and its allocator), and nowhere else.

    MpcForces.allocate(now, law, state, gait, clock, ref, est, com_cmd)
        -> allocation.Allocation          the four foot forces, WORLD

`law.BalanceLaw.mpc` holds one of these (`hw.fold2_cmpc`); None, every other
entry point, leaves the law exactly as it was.  2026-11-10, on request: "cmpc
only for the trot and the walk; the crouch, the stand, the swing control, the
trajectory and the sweep handling are fold2_walk's; the duty and the ramp are
cmpc's own".

WHAT IS THE MPC'S, AND ONLY WHILE THE CLOCK RUNS
    The foot forces of the planted legs.  Every `1 / hz` (40 Hz, sim.cmpc's)
    one QP -- `sim.cmpc.qp`'s condensed problem, its Q and R, its horizon of
    10, `sim.cmpc.dynamics`' model -- is solved inside the law's slot, and its
    first force vector is held, in the world, until the next.  The clock is
    cmpc's too: 0.5 s at duty 0.5, no contact ramp, no settle (the entry
    point's `gait`), so the contacts are 0/1 and the horizon reads them off
    that same clock.

A HANDOVER IS A SOLVE, AND THE HORIZON READS EACH INTERVAL AT ITS START
    `sim.cmpc.gait.horizon_contacts` reads the contact at the END of each
    interval and solves on a fixed tick.  On a fixed body that leaves every
    foot at zero force from the solve before a handover to the handover
    itself -- the lifting pair already planned out, the landing pair still
    in the air: up to one MPC period, 25 ms, of no support at all, twice a
    cycle (measured here: mean total fz 46.2 of 51.1 N).  So on the robot a
    sweep whose contacts differ from the last solve's SOLVES, whatever the
    tick says, and re-anchors the tick there; and row i is the contact at
    the START of interval i, which with that rule is never stale.  A foot
    that lifts is zeroed on its own sweep either way (`allocate`).

WHAT IS STILL fold2_walk's, UNTOUCHED
    limp, settle, crouch, the tracked rise, HOLD and its CoM re-centring, the
    park: the SRB law and the QP allocator, as flown -- the MPC is not
    called on a sweep with no gait clock.  In the trot: the reference
    (`walk.WalkPlan`, slewed, clipped, leashed), the footholds and the arc
    (footstep.py), the swing law (swing_control.py), the leg-weight term and
    J^T f (`torque.stance_torque`), the joint layer on the planted legs, the
    fall hold, the governor, the estimator's feed, the gate, the sweep.

THE STATE, FROM WHAT THE LAW ALREADY HAS -- no second estimator
    roll pitch yaw   `state` (the IMU, the run's world); a stale IMU reads as
                     the reference's attitude and rate, the SRB law's
                     `hold_attitude`
    x y, vx vy       the filter's trunk origin moved to the CoM (`est`,
                     `law.TrunkEstimate`), exactly what the SRB x/y rows read;
                     a sweep whose estimate is refused reads the REFERENCE's,
                     so the x/y error is zero -- the flown law's fallback
    z, vz            `state.p_cz` / `p_cz_dot`, over the feet that are down
    omega            `state.omega_w`

THE REFERENCE OVER THE HORIZON IS THE WALK'S, ROLLED FORWARD
    `ref` (trajectory.RefSample) is where the one walking reference is this
    sweep; `sim.cmpc.trajectory.ReferenceTrajectory.horizon` integrates its
    slewed body command from there.  Roll and pitch are the setpoint the law
    holds (`sp_roll` / `sp_pitch` and the operator's offset), z the commanded
    CoM height (`reference.com_command`), wrapped onto the measured yaw's
    branch as sim.cmpc does.

THE FEET OVER THE HORIZON
    A planted foot is where it is (`state.r_w`), the CoM moving off it at the
    reference's velocity.  A swinging foot is taken to be at its planned
    touchdown -- the planner's `land_w` -- or, where there is none (the trunk
    -frame swing at rest), on its own site under the reference trunk.

THE MODEL'S NUMBERS ARE THE ROBOT'S
    `config.MASS` (weighed, 5.210 kg), the pinned inertia of the stance being
    held (`law.srb`), the law's friction coefficient and fz box.

A QP THAT DOES NOT SOLVE keeps the last good forces (the zero-order hold the
    method assumes anyway) and is counted; the solve's cost is kept for the
    exit report.  THE SOLVE DELAYS EVERY MOTOR BEHIND IT by its whole
    duration, once per MPC period: ~5 ms on the Pi (hw-cmpc, fake bus,
    2026-11-10).  Not flown.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace as dc_replace

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/mpc.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C             # noqa: E402
from sim import params as P                  # noqa: E402
from sim.cmpc import config as SIMCFG        # noqa: E402
from sim.cmpc import dynamics as DYN         # noqa: E402
from sim.cmpc import qp as SIMQP             # noqa: E402
from sim.cmpc import trajectory as SIMTRAJ   # noqa: E402

from . import config as cfg                  # noqa: E402
from .allocation import Allocation           # noqa: E402

__all__ = ["MpcForces", "GAIT_PERIOD", "GAIT_DUTY", "QP_OK_STATUSES"]

#: cmpc's own clock, kept (2026-11-10, on request): `sim.cmpc.config`'s
#: period and duty.  No contact ramp and no settle go with a duty of 0.5.
GAIT_PERIOD = float(SIMCFG.GAIT_PERIOD)      # 0.5 s
GAIT_DUTY = float(SIMCFG.DUTY)               # 0.5

#: OSQP statuses whose forces are used; any other keeps the last good ones.
QP_OK_STATUSES = ("solved", "solved inaccurate")

_SAMPLES = 20000


class MpcForces:
    """The trot's foot forces, from `sim.cmpc`'s QP, held between solves."""

    def __init__(self, *, hz: float = SIMCFG.MPC_HZ,
                 horizon: int = SIMCFG.HORIZON, mu: float = cfg.MU,
                 fz_min: float = cfg.FZ_MIN, fz_max: float = cfg.FZ_MAX,
                 mass: float = cfg.MASS, q_diag=None, r_diag=None,
                 polish: bool = False):
        if not hz > 0.0:
            raise ValueError("hz must be positive, got %r" % (hz,))
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.horizon = int(horizon)
        self.mu, self.fz_min, self.fz_max = float(mu), float(fz_min), float(fz_max)
        self.mass = float(mass)
        self.q_diag = (SIMCFG.Q_DIAG if q_diag is None
                       else np.asarray(q_diag, dtype=float))
        self.r_diag = (SIMCFG.R_DIAG if r_diag is None
                       else np.asarray(r_diag, dtype=float))
        # Polishing OFF: OSQP 1.x prints a line from C on every solve whose
        # optimum has no active constraint (hw-cmpc's finding).
        self.solver = SIMQP.Solver(horizon=self.horizon, polishing=bool(polish))
        self.forces = np.zeros((C.N_LEGS, 3))        # WORLD, the last good solve
        self.t_next: float | None = None
        #: (4,) the contacts the held forces were solved for; a sweep whose
        #: clock says otherwise solves again (module docstring).
        self._contact = None
        self.handover_solves = 0
        self.status = "not solved"
        self.solve_ms = float("nan")                 # the last solve's
        self.solved_now = False                      # a solve ran this sweep
        self.solves = 0
        self.failures = 0
        self.over_25ms = 0
        self._ms: list[float] = []
        self.sweeps = 0

    # -- the clock ----------------------------------------------------------
    def idle(self) -> None:
        """Not trotting this sweep: forget the held forces, so the next trot's
        first sweep solves afresh.  The solver keeps its warm start."""
        self.t_next = None
        self._contact = None
        self.solved_now = False
        self.forces = np.zeros((C.N_LEGS, 3))

    def _due(self, now: float, contact) -> bool:
        contact = np.asarray(contact, dtype=bool)
        if self._contact is not None and not np.array_equal(contact,
                                                            self._contact):
            # A HANDOVER: solve now, and count the period from here.
            self.handover_solves += 1
            self.t_next = now + self.dt
        elif self.t_next is None or now >= self.t_next - 1.0e-9:
            # RE-ANCHOR, do not catch up, when a whole period was missed.
            late = self.t_next is None or now - self.t_next >= self.dt
            self.t_next = now + self.dt if late else self.t_next + self.dt
        else:
            return False
        self._contact = contact.copy()
        return True

    def warm_up(self) -> None:
        """Pay OSQP's setup and every first-call cost BEFORE the loop: two
        solves of a nominal four-foot problem.  The first solve of a fresh
        solver is its symbolic factorisation, ~25 ms inside the sweep on the
        Pi -- past the 25 ms CAN gap line on the trot's first sweep
        (`hw.fold2_cmpc --fake`, 2026-11-10)."""
        r = np.array([[0.2, 0.07, -0.15], [0.2, -0.07, -0.15],
                      [-0.2, 0.07, -0.15], [-0.2, -0.07, -0.15]])
        x0 = SIMTRAJ.initial_state((0.0, 0.0, 0.15), (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        x_ref = SIMTRAJ.ReferenceTrajectory().horizon(
            SIMTRAJ.Command(z=0.15), self.horizon, self.dt)
        a_seq, b_seq = DYN.discrete_sequence(
            np.zeros(self.horizon), np.tile(r, (self.horizon, 1, 1)),
            dt=self.dt, mass=self.mass)
        for pattern in ((1, 1, 1, 1), (1, 0, 0, 1)):
            SIMQP.solve_once(x0, x_ref, a_seq, b_seq,
                             np.tile(np.array(pattern, dtype=bool),
                                     (self.horizon, 1)),
                             solver=self.solver, q_diag=self.q_diag,
                             r_diag=self.r_diag, mu=self.mu,
                             fz_min=self.fz_min, fz_max=self.fz_max)

    # -- the problem --------------------------------------------------------
    def _x0(self, law, state, ref, est):
        """(13,) the MPC's state, and the CoM's world x/y it was built on."""
        com_w = state.R @ np.asarray(law.srb.com_body, dtype=float)
        if est is not None:
            p_xy = np.asarray(est.p_w, dtype=float)[:2] + com_w[:2]
            w = state.omega_w
            v_xy = (np.asarray(est.v_w, dtype=float)[:2]
                    + np.array([w[1] * com_w[2] - w[2] * com_w[1],
                                w[2] * com_w[0] - w[0] * com_w[2]]))
        else:                            # refused: the x/y error is zero
            p_xy = np.asarray(ref.p, dtype=float)
            v_xy = np.asarray(ref.v, dtype=float)
        dr, dp, _ = law.att_offset
        if state.imu_stale:              # the SRB law's hold_attitude
            rpy = (law.sp_roll + dr, law.sp_pitch + dp,
                   math.remainder(float(ref.yaw), 2.0 * math.pi))
            omega = (0.0, 0.0, float(ref.yaw_rate))
        else:
            rpy = (float(state.roll), float(state.pitch), float(state.yaw))
            omega = state.omega_w
        x0 = SIMTRAJ.initial_state((p_xy[0], p_xy[1], float(state.p_cz)), rpy,
                                   (v_xy[0], v_xy[1], float(state.p_cz_dot)),
                                   omega)
        return x0, p_xy

    def _reference(self, law, ref, com_cmd, yaw_meas: float) -> np.ndarray:
        """(horizon, 13): the walk's reference rolled forward."""
        command = dc_replace(ref.command, z=float(com_cmd.p_cz))
        traj = SIMTRAJ.ReferenceTrajectory(x=float(ref.p[0]), y=float(ref.p[1]),
                                           yaw=float(ref.yaw))
        x_ref = traj.horizon(command, self.horizon, self.dt)
        dr, dp, _ = law.att_offset
        x_ref[:, 0] = law.sp_roll + dr
        x_ref[:, 1] = law.sp_pitch + dp
        # ONE offset for the whole block, onto the measured yaw's branch
        # (`sim.cmpc.controller.solve` says why).
        x_ref[:, 2] -= 2.0 * math.pi * round(
            (float(x_ref[0, 2]) - yaw_meas) / (2.0 * math.pi))
        return x_ref

    def _feet(self, law, state, clock, ref, p_xy, x_ref) -> np.ndarray:
        """(horizon, 4, 3) CoM -> foot, WORLD, at each step."""
        planted = np.asarray(clock.contact, dtype=bool)
        r_now = np.asarray(state.r_w, dtype=float)
        z_foot = float(r_now[planted, 2].mean()) if planted.any() else float(
            P.FOOT_RADIUS - state.p_cz)
        planner = law.walk.planner
        land = None if planner is None else planner.land_w
        com_b = np.asarray(law.srb.com_body, dtype=float)[:2]
        out = np.empty((self.horizon, C.N_LEGS, 3))
        v = np.asarray(ref.v, dtype=float)
        for k in range(self.horizon):
            travel = v * (k + 1) * self.dt
            c, s = math.cos(float(x_ref[k, 2])), math.sin(float(x_ref[k, 2]))
            for i in range(C.N_LEGS):
                if planted[i]:
                    out[k, i, :2] = r_now[i, :2] - travel
                    out[k, i, 2] = r_now[i, 2]
                    continue
                if land is not None and np.all(np.isfinite(land[i])):
                    out[k, i, :2] = land[i] - (p_xy + travel)
                else:                    # on its site, under the trunk then
                    sx, sy = law.walk.sites_b[i] - com_b
                    out[k, i, 0] = c * sx - s * sy
                    out[k, i, 1] = s * sx + c * sy
                out[k, i, 2] = z_foot
        return out

    def solve(self, now: float, law, state, gait, clock, ref, est,
              com_cmd) -> None:
        """One QP; its first force vector becomes `forces` if it solved."""
        started = time.perf_counter()
        x0, p_xy = self._x0(law, state, ref, est)
        x_ref = self._reference(law, ref, com_cmd, float(x0[2]))
        # Row i is the contact at the START of interval i (module docstring);
        # row 0 is this sweep's own clock sample.
        contacts = np.stack([np.asarray(clock.contact, dtype=bool)]
                            + [gait.contact(float(now) + self.dt * k)
                               for k in range(1, self.horizon)])
        r_feet = self._feet(law, state, clock, ref, p_xy, x_ref)
        a_seq, b_seq = DYN.discrete_sequence(
            x_ref[:, 2], r_feet, dt=self.dt,
            inertia_body=law.srb.inertia_body, mass=self.mass)
        forces, _ = SIMQP.solve_once(x0, x_ref, a_seq, b_seq, contacts,
                                     solver=self.solver, q_diag=self.q_diag,
                                     r_diag=self.r_diag, mu=self.mu,
                                     fz_min=self.fz_min, fz_max=self.fz_max)
        self.status = self.solver.last_status
        self.solves += 1
        if self.status in QP_OK_STATUSES and np.all(np.isfinite(forces)):
            self.forces = np.array(forces, dtype=float)
        else:
            self.failures += 1           # the last good forces stay
        self.solve_ms = 1e3 * (time.perf_counter() - started)
        if len(self._ms) < _SAMPLES:
            self._ms.append(self.solve_ms)
        if self.solve_ms > 25.0:
            self.over_25ms += 1

    # -- what the law calls -------------------------------------------------
    def allocate(self, now: float, law, state, gait, clock, ref, est,
                 com_cmd) -> Allocation:
        """This sweep's foot forces: a solve if one is due, else the held
        ones; a leg the clock has in swing gets exactly zero."""
        self.sweeps += 1
        self.solved_now = self._due(float(now), clock.contact)
        if self.solved_now:
            self.solve(now, law, state, gait, clock, ref, est, com_cmd)
        f = self.forces.copy()
        f[~np.asarray(clock.contact, dtype=bool)] = 0.0
        return Allocation(f_w=f, f_unclipped=f.copy(), residual=np.zeros(6),
                          clipped=np.zeros(C.N_LEGS, dtype=bool))

    def report(self) -> str:
        if not self._ms:
            return "  MPC             attached, never solved (no trot)"
        ms = np.asarray(self._ms)
        return ("  MPC             %d solves over %d trot sweeps at %.0f Hz "
                "(%d of them at a handover): p50 %.2f ms  p95 %.2f ms  max "
                "%.2f ms (%d over 25 ms); %d not solved (kept the last good "
                "forces), last status '%s'"
                % (self.solves, self.sweeps, self.hz, self.handover_solves,
                   np.percentile(ms, 50), np.percentile(ms, 95), ms.max(),
                   self.over_25ms, self.failures, self.status))

    def describe(self) -> str:
        return ("sim.cmpc's QP at %.0f Hz, horizon %d x %.0f ms, Q %s, mass "
                "%.3f kg, mu %.2f, fz %.0f..%.0f N"
                % (self.hz, self.horizon, 1e3 * self.dt,
                   np.array2string(self.q_diag[:12], precision=2),
                   self.mass, self.mu, self.fz_min, self.fz_max))


if __name__ == "__main__":
    print(MpcForces().describe())
