"""Walking: ONE reference, and every layer that has a target takes it from it.

    WalkPlan.step(now, law, state, gait, clock, est) -> RefSample or None
    WalkPlan.contacts(law, state, ref, planted, est) touchdowns: anchor
    WalkPlan.hold_targets(law, state, ref)           -> (q_t, qd_t)
    WalkPlan.swing_ref(leg, s, law, state, ref, est) -> SwingRef

`law.BalanceLaw.walk` holds one of these; None (every entry point but
`hw.fold_walk`) leaves the law exactly as it was.  Attached but not yet
engaged -- before the first T -- it does nothing either: the stand, the rise
and the first HOLD are the flown ones.

THE RULE THIS FILE EXISTS FOR
    The SRB PD, the joint layer and the swing each regulate the trunk against
    a target of their own.  Standing still those targets agree by
    construction (`law.BalanceLaw.hold_reference`); walking, they agree only
    if every one of them is computed from the SAME reference -- otherwise each
    pushes the trunk toward a different place and the robot carries the
    difference as internal force (doc/walk/README.md, "why one reference").
    So from the first trot sweep on:

        x/y rows     CoM x/y and rate against p_ref, v_ref, a_ref fed forward
        attitude     R_des = Rz(psi_ref) on the latched roll/pitch, w_des = r
        joint layer  each planted foot's target is where the REFERENCE trunk
                     sees it: x_t = Rz(psi_ref)^T (anchor - p_ref,origin),
                     anchor = the foot's world x/y, latched at touchdown
        swing        lands on the foothold planned from the reference
                     (footstep.py), in the same world -- AT REST (v_ref and r
                     zero) on the leg's own site in the trunk frame instead,
                     the filter nowhere in it: hw.fold_trot's in-place target
                     (footstep.py, 2026-10-05; `place_at_rest` undoes it)

WHY THE STANCE TARGET IS WORLD-ANCHORED (the joint layer kept, as asked)
    A joint target latched once at HOLD is a stance the walking feet leave:
    with the feet moving under the trunk at v, a fixed q_hold is a spring
    stretched by v T_st / 2 at every touchdown and liftoff.  Re-latched at
    each touchdown it is a brake, K v T_st^2 / 2 a step.  Anchored in the
    world and seen from the reference trunk it is neither: along the
    reference the target moves exactly as the foot does, and the layer pulls
    only on the trunk's error from p_ref -- the same error, by the same
    measurement, the x/y rows pull on.  The anchor of a foot that has stepped
    is where the filter put it at touchdown: p_hat + R x_b, the full R, so the
    tilt the attitude loop is correcting is not baked into the target.

    At engagement every foot is still on its HOLD site, and its anchor is set
    VIRTUALLY to p_eq + Rz(psi_0) site, with p_eq the least-squares trunk
    origin that puts the four feet on their sites: so on the first trot sweep
    the targets ARE q_hold's feet and the reference IS the layer's own
    equilibrium -- no step in the joint torque, none in the x/y rows.

    Per leg, the xy part of (target - measured) is clamped to
    `config.HOLD_XY_ERR_MAX`, and the reference is leashed to the filter
    (trajectory.py), so an estimate gone wrong pulls on a bounded spring.

THE PLACEMENT'S VELOCITY IS LOW-PASSED (2026-10-05, on request)
    The planner's trunk velocity is the filter's x/y velocity through a
    first-order low-pass, `v_filter_hz` (config.WALK_V_FILTER_HZ), stepped
    once per estimate and started at rest when the walk engages: on the
    robot the raw v_hat is noise enough to move a foothold by centimetres.
    It reaches the foothold (eq 33 and the Raibert term) and the swing's
    world-to-trunk velocity, nothing else; the x/y rows read v_hat raw.

WITHOUT AN ESTIMATE
    Before the filter's first answer, or on any sweep it is refused, the
    trunk's world x/y is taken to be the reference's, the x/y rows are zero
    (the flown law), the foothold uses v_ref, and a landing foot is anchored
    where the reference trunk sees it.  The walk then runs open loop on its
    own plan, as cMPC's reference does between solves.
"""
from __future__ import annotations

import math

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/walk.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402
from . import swing as SWING         # noqa: E402
from .footstep import FootstepPlanner, TrunkXY, rot2   # noqa: E402
from .trajectory import Command, WalkReference          # noqa: E402

__all__ = ["WalkPlan"]

_ZERO = Command(vx=0.0, vy=0.0, yaw_rate=0.0)


class WalkPlan:
    """The walk's reference, its footholds and its stance anchors."""

    def __init__(self, *, reference: WalkReference | None = None,
                 kv: float = cfg.STEP_KV, step_max=cfg.STEP_MAX_XY,
                 kp_swing=None, kd_swing=None, swing_ff: bool = False,
                 ff_inertia=None, hold_xy_max: float = cfg.HOLD_XY_ERR_MAX,
                 swing_law: str = cfg.WALK_SWING_LAW, wn_swing=None,
                 zeta_swing: float | None = None,
                 place_at_rest: bool = cfg.WALK_PLACE_AT_REST,
                 v_filter_hz: float = cfg.WALK_V_FILTER_HZ):
        self.reference = reference if reference is not None else WalkReference()
        #: False (the default): a swing that lifts with the reference at rest
        #: lands on its site, trunk frame, no velocity term (footstep.py).
        self.place_at_rest = bool(place_at_rest)
        #: Hz, the corner of the low-pass on the filter's x/y velocity as the
        #: planner reads it (`trunk_xy`); 0 or less reads it raw.  `v_lp` is
        #: its output, WORLD m/s, and `v_lp_t` the estimate time it last took.
        self.v_filter_hz = float(v_filter_hz)
        self.v_lp = np.zeros(2)
        self.v_lp_t = float("nan")
        self.kv = float(kv)
        self.step_max = np.asarray(step_max, dtype=float).reshape(2)
        #: The impedance's (x, y, z) gains: (3,) every leg's, or (4, 3) a row
        #: per leg, FL FR RL RR (`--kp-swing-walk-rl` etc., 2026-10-05).
        self.kp_swing = (cfg.KP_SWING_WALK if kp_swing is None
                         else np.array(kp_swing, dtype=float))
        self.kd_swing = (cfg.KD_SWING_WALK if kd_swing is None
                         else np.array(kd_swing, dtype=float))
        for name, gains in (("kp_swing", self.kp_swing),
                            ("kd_swing", self.kd_swing)):
            if gains.shape not in ((3,), (C.N_LEGS, 3)):
                raise ValueError("%s: (3,) for every leg or (%d, 3) a row "
                                 "per leg, not %s"
                                 % (name, C.N_LEGS, gains.shape))
        self.swing_ff = bool(swing_ff)
        self.ff_inertia = ff_inertia
        if swing_law not in ("impedance", "osc"):
            raise ValueError("swing_law %r: 'impedance' or 'osc'" % (swing_law,))
        #: swing_control.py's two laws: the walking impedance (+ --swing-ff),
        #: or the task-space computed torque with its acceleration gains.
        self.swing_law = swing_law
        self.wn_swing = (cfg.WN_SWING_OSC if wn_swing is None
                         else np.asarray(wn_swing, dtype=float))
        self.zeta_swing = (cfg.ZETA_SWING_OSC if zeta_swing is None
                           else float(zeta_swing))
        self.hold_xy_max = float(hold_xy_max)
        #: What the keys ask (`keys.WalkKeys.command`); reaches the reference
        #: only while trotting.
        self.command = _ZERO
        self.engaged = False
        self.planner: FootstepPlanner | None = None
        #: (4, 2) each planted foot's WORLD x/y anchor; (4, 2) the hold's
        #: neutral feet, trunk frame -- the sites the planner places about.
        self.anchor_w = np.full((C.N_LEGS, 2), np.nan)
        self.sites_b = np.full((C.N_LEGS, 2), np.nan)
        self.z_rest = np.full(C.N_LEGS, np.nan)
        self._planted = np.ones(C.N_LEGS, dtype=bool)
        self.steps = 0                # touchdowns anchored, for the report
        self.engaged_at = float("nan")

    def set_command(self, command: Command | None) -> None:
        self.command = _ZERO if command is None else command

    # -- the trunk the plan works from --------------------------------------
    @staticmethod
    def _com_xy_b(law) -> np.ndarray:
        return np.asarray(law.srb.com_body, dtype=float)[:2]

    def _origin_ref(self, law, ref):
        """(p, v) of the REFERENCE trunk origin, WORLD x/y: the CoM reference
        less the CoM offset at the reference heading (level, as R_des is)."""
        cb = rot2(ref.yaw) @ self._com_xy_b(law)
        p = ref.p - cb
        v = ref.v - ref.yaw_rate * np.array([-cb[1], cb[0]])
        return p, v

    def trunk_xy(self, law, state, ref, est) -> TrunkXY:
        """The filter's trunk origin when it is usable this sweep, its x/y
        velocity low-passed (`v_filter_hz`), else the reference's -- see the
        module docstring, WITHOUT AN ESTIMATE."""
        if est is not None:
            v = (self.v_lp.copy() if self.v_filter_hz > 0.0
                 else np.asarray(est.v_w, dtype=float)[:2].copy())
            return TrunkXY(p=np.asarray(est.p_w, dtype=float)[:2].copy(),
                           v=v,
                           yaw=float(state.yaw),
                           yaw_rate=float(state.omega_w[2]), measured=True)
        p, v = self._origin_ref(law, ref)
        return TrunkXY(p=p, v=v, yaw=float(ref.yaw),
                       yaw_rate=float(ref.yaw_rate), measured=False)

    # -- engaging -----------------------------------------------------------
    def engage(self, now: float, law, state, gait, est) -> None:
        """The first trot sweep: anchor the reference on the layer's own
        equilibrium and the four feet on their HOLD sites (module docstring)."""
        yaw0 = float(state.yaw)
        if law.q_hold is not None:
            sites = np.array([HK.foot_position(i, C.unflat(law.q_hold)[i])
                              for i in range(C.N_LEGS)])
        else:
            sites = SWING.rest_feet_b(law.h_lift, law.foot_xy)
        self.sites_b = sites[:, :2].copy()
        #: (4,) the commanded foot depth, trunk z: the stance targets' and
        #: the arc's rest height.  Fixed for the trot, so computed once.
        self.z_rest = SWING.rest_feet_b(law.h_lift, law.foot_xy)[:, 2].copy()
        R2 = rot2(yaw0)
        if est is not None:
            feet_w = (np.asarray(est.p_w, dtype=float)[None, :]
                      + state.x_b @ state.R.T)[:, :2]
            p_eq = (feet_w - self.sites_b @ R2.T).mean(axis=0)
        else:
            p_eq = np.zeros(2)
        self.anchor_w = p_eq[None, :] + self.sites_b @ R2.T
        p_com = p_eq + R2 @ self._com_xy_b(law)
        self.reference.reset(p_com, yaw0)
        self.planner = FootstepPlanner(self.sites_b,
                                       t_stance=gait.stance_duration,
                                       t_swing=gait.swing_duration,
                                       height=law.swing_height, kv=self.kv,
                                       step_max=self.step_max,
                                       place_at_rest=self.place_at_rest)
        self._planted = np.ones(C.N_LEGS, dtype=bool)
        # The placement's low-pass starts AT REST: the trot starts from HOLD.
        self.v_lp = np.zeros(2)
        self.v_lp_t = float("nan")
        self.engaged = True
        self.engaged_at = float(now)

    def _low_pass(self, est) -> None:
        """Step the planner's low-passed x/y velocity by the estimate `est`
        (None: hold it).  Once per ESTIMATE, by `est.t`: one the law reads on
        two sweeps counts once.  The first only starts the clock."""
        if est is None or self.v_filter_hz <= 0.0:
            return
        t = float(est.t)
        if not math.isfinite(self.v_lp_t):
            self.v_lp_t = t
            return
        dt = t - self.v_lp_t
        if dt <= 0.0:
            return
        a = 1.0 - math.exp(-2.0 * math.pi * self.v_filter_hz * dt)
        self.v_lp = self.v_lp + a * (np.asarray(est.v_w, dtype=float)[:2]
                                     - self.v_lp)
        self.v_lp_t = t

    # -- the sweep ----------------------------------------------------------
    def step(self, now: float, law, state, gait, clock, est):
        """This sweep's reference, or None while not engaged.  `gait` is the
        trot clock and `clock` its sample this sweep (both None outside a
        trot); `est` the law's usable estimate this sweep, or None."""
        if not self.engaged:
            if clock is None:
                return None
            self.engage(now, law, state, gait, est)
        self._low_pass(est)
        target = self.command if clock is not None else _ZERO
        p_hat = None
        if est is not None:
            c = state.R @ np.asarray(law.srb.com_body, dtype=float)
            p_hat = np.asarray(est.p_w, dtype=float)[:2] + c[:2]
        yaw_hat = None if state.imu_stale else float(state.yaw)
        return self.reference.update(now, target, p_hat, yaw_hat)

    def contacts(self, law, state, ref, planted, est) -> None:
        """Anchor every foot that touched down this sweep; forget the plan of
        every foot that is down."""
        planted = np.asarray(planted, dtype=bool)
        landed = planted & ~self._planted
        if landed.any():
            if est is not None:
                feet_w = (np.asarray(est.p_w, dtype=float)[None, :]
                          + state.x_b @ state.R.T)[:, :2]
            else:
                p_org, _ = self._origin_ref(law, ref)
                feet_w = p_org[None, :] + state.x_b[:, :2] @ rot2(ref.yaw).T
            self.anchor_w[landed] = feet_w[landed]
            self.steps += int(landed.sum())
        if self.planner is not None:
            for i in np.flatnonzero(planted):
                self.planner.release(int(i))
        self._planted = planted.copy()

    def hold_targets(self, law, state, ref, planted=None):
        """``(q_t, qd_t)``, (4, 3) each: the joint layer's target for every
        PLANTED leg as the reference trunk sees its anchor.  `planted` (4,)
        bool is the clock's contact, None for all four; a swinging leg gets
        its measured q and zero rate, which the caller masks out anyway --
        its IK is the one thing here that costs, so it is not done."""
        p_org, v_org = self._origin_ref(law, ref)
        c, sn = math.cos(float(ref.yaw)), math.sin(float(ref.yaw))
        r = float(ref.yaw_rate)
        q4 = C.unflat(state.q)
        q_t = q4.copy()
        qd_t = np.zeros((C.N_LEGS, 3))
        # v_b = Rz^T v_org: the reference origin's velocity, trunk axes.
        vbx = c * float(v_org[0]) + sn * float(v_org[1])
        vby = -sn * float(v_org[0]) + c * float(v_org[1])
        legs = (range(C.N_LEGS) if planted is None
                else np.flatnonzero(np.asarray(planted, dtype=bool)))
        for i in legs:
            ex = float(self.anchor_w[i, 0]) - float(p_org[0])
            ey = float(self.anchor_w[i, 1]) - float(p_org[1])
            tx, ty = c * ex + sn * ey, -sn * ex + c * ey
            mx, my = float(state.x_b[i, 0]), float(state.x_b[i, 1])
            n = math.hypot(tx - mx, ty - my)
            if n > self.hold_xy_max:
                k = self.hold_xy_max / n
                tx, ty = mx + (tx - mx) * k, my + (ty - my) * k
            z = float(self.z_rest[i])
            # d/dt Rz^T (a - p) = -Rz^T v - w x x_t: the foot moves backward
            # under a trunk going forward, round it under a turning one.
            xd_t = np.array([-vbx + r * ty, -vby - r * tx, 0.0])
            hip = P.HIP_OFFSET[i]
            q_t[i] = HK.leg_ik(int(i), np.array([tx - hip[0], ty - hip[1],
                                                 z - hip[2]]),
                               q_seed=q4[i])
            qd_t[i] = np.linalg.solve(state.jac[i], xd_t)
        return q_t, qd_t

    def swing_ref(self, leg: int, s: float, law, state, ref, est):
        """`footstep.SwingRef` for swinging `leg` at progress `s`."""
        trunk = self.trunk_xy(law, state, ref, est)
        return self.planner.plan(leg, s, trunk, ref, state.x_b[leg],
                                 float(self.z_rest[leg]))

    def report(self) -> str:
        if not self.engaged:
            return "  walk            attached, never engaged (no trot)"
        fp = self.planner
        return ("  walk            %d touchdowns anchored, leash acted on %d "
                "sweeps, %d landings clamped to the step bound; swings: %d "
                "at rest (trunk frame, onto the site), %d placed%s"
                % (self.steps, self.reference.leash_sweeps,
                   0 if fp is None else fp.clamped,
                   0 if fp is None else fp.swings_at_rest,
                   0 if fp is None else fp.swings_placed,
                   " (--place-at-rest)" if self.place_at_rest else "")
                + ("; placement velocity low-passed at %.1f Hz"
                   % self.v_filter_hz if self.v_filter_hz > 0.0 else
                   "; placement velocity raw"))
