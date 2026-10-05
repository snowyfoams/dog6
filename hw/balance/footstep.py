"""The swing leg's trajectory while walking: where the foot lands, and the arc.

    FootstepPlanner.plan(leg, s, trunk, ref, x_b, z_rest)  -> SwingRef
    FootstepPlanner.release(leg)                            the leg is down

WHERE THE FOOT LANDS -- eq (33), with Raibert's feedback term
    cMPC's placement, `sim.cmpc.swing.foot_placement`: put the foot where its
    neutral site will be at the MIDDLE of the coming stance, so the stance is
    symmetric about it and the body neither speeds up nor slows down across
    it.  In the run's world, at the predicted touchdown:

        p_land = p(t_td) + Rz(psi(t_td) + r T_st / 2) site       the site, turned
               + v_hat T_st / 2                                   (33)
               + STEP_KV (v_hat - v_ref)                          Raibert's term

    `site` is the STANCE's OWN neutral foot in the trunk frame -- the hold's
    sites (the fold stand's feet are 7 / 63 mm BEHIND their hips), not the
    hip as cMPC's: placed about the hip, the walking stance would be a
    different stance from the one the SRB model is pinned at, and the stance
    targets (walk.py) would be preloaded by the difference on every landing.
    The yaw turn r T_st / 2 is MIT's `pYawCorrected`, the hip-velocity
    correction `sim.cmpc.config.YAW_PLACEMENT_CORRECTION` measured as
    harmless; without a horizon this law cannot absorb a 33 mm error the way
    the MPC did, so it is on here.  p(t_td) is predicted with v_REF, not
    v_hat, so the filter's noise reaches the foothold only through T_st / 2
    and STEP_KV.

    (33) alone sustains the velocity the robot has; the feedback term is what
    corrects it, and with no horizon above it the SRB law's slow x/y rows
    are the only other thing that can.  The landing is clamped to
    `STEP_MAX_XY` of the site, in the trunk's heading frame.

THE ARC: x/y IN THE WORLD, z IN THE TRUNK -- the split is deliberate
    x/y is cMPC's quintic from the foot latched at LIFTOFF to p_land, in the
    world, re-aimed every sweep as p_land moves (cMPC re-plans every sweep
    too).  World-fixed ends mean zero WORLD velocity at liftoff and at
    touchdown: the foot leaves and meets the floor at ground speed, so a
    walking stride does not scuff or slide it.  A trunk-frame arc lands at
    the trunk's speed, which at 0.1 m/s is 0.1 m/s of slip at every
    touchdown.

    z keeps the in-place swing's trunk-frame bump on the commanded height
    (`swing.rest_feet_b`), apex `swing_height` above it: the height loop and
    the joint layer are written against trunk-frame z, the operator tuned
    the trot on this one, and the filter's z datum is the feet's own, so
    putting z in the world would only re-import the filter into the one
    axis the legs measure directly.

    World -> trunk is the HEADING alone, Rz(psi_hat): roll and pitch of a
    degree move a ground point 3 mm in the trunk frame, inside what the
    landing tolerates, and the planted targets that follow it are measured,
    not planned.  The trunk's world x/y and velocity are the filter's
    (`law.TrunkEstimate`); without one they are the reference's, and the
    walk runs open loop on its own plan.

    v_b carries -omega x p_b: a world-fixed point seen from a turning trunk
    moves, and a damper fed the trunk-frame velocity without it brakes every
    turn.  cMPC's `to_body` dropped it and said so; at 40 deg/s on a 0.2 m
    lever it is 0.14 m/s.

AT REST THE ARC IS IN THE TRUNK FRAME -- hw.fold_trot's swing, 2026-10-05
    With the reference AT REST (`at_rest`: v_ref and r both zero) a swing
    that lifts is planned in the TRUNK frame for its whole flight: the same
    x/y quintic from the foot latched at liftoff, landing on the leg's own
    site, and the same z bump -- no eq (33), no Raibert term and no world,
    so the filter's p, v and heading do not enter it (they place only the
    logged `land_w`).  That is hw.fold_trot's in-place target, the arc from
    rest site to rest site (`swing.swing_reference_pva`), started on the foot
    instead of on the site so that a stiff swing law takes no step at
    liftoff.  Why, on request: on the robot, standing, v_hat is rms 0.05-0.08
    m/s with spikes to 0.4 (fold_walk_imp.npz), so the foothold above,
    re-aimed every sweep, wandered 10-60 mm inside one 115 ms swing, and
    Kp 150 chased it and threw the feet.  Decided once per swing, on its
    first sweep: a key pressed mid-swing places the NEXT swing, and a swing
    that lifted while walking lands where it was placed.  `place_at_rest`
    keeps the placement at rest, as flown before.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/footstep.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from . import config as cfg          # noqa: E402

__all__ = ["TrunkXY", "SwingRef", "FootstepPlanner", "rot2"]


def rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _quintic(s: float):
    """`sim.cmpc.swing._smoothstep` for one scalar: 6s^5 - 15s^4 + 10s^3 on
    [0, 1] and its first two derivatives."""
    s = min(max(s, 0.0), 1.0)
    return (s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
            30.0 * s * s * (1.0 + s * (-2.0 + s)),
            60.0 * s * (1.0 + s * (-3.0 + 2.0 * s)))


class TrunkXY(NamedTuple):
    """The trunk ORIGIN's horizontal state the planner works from.  WORLD."""

    p: np.ndarray            # (2,) m
    v: np.ndarray            # (2,) m/s
    yaw: float               # rad, the run's world
    yaw_rate: float          # rad/s
    measured: bool           # True: the filter's; False: the reference's


class SwingRef(NamedTuple):
    """One swing leg's reference this sweep, TRUNK frame, plus the plan."""

    p: np.ndarray            # (3,) m
    v: np.ndarray            # (3,) m/s
    a: np.ndarray            # (3,) m/s^2
    land_w: np.ndarray       # (2,) m, the planned touchdown, WORLD x/y
    lift_w: np.ndarray       # (2,) m, the latched liftoff, WORLD x/y


class FootstepPlanner:
    """Foothold and arc for each swinging leg; latches at liftoff."""

    def __init__(self, sites_b, *, t_stance: float, t_swing: float,
                 height: float = cfg.SWING_HEIGHT, kv: float = cfg.STEP_KV,
                 step_max=cfg.STEP_MAX_XY,
                 place_at_rest: bool = cfg.WALK_PLACE_AT_REST):
        #: (4, 2) the stance's neutral feet, TRUNK frame x/y about the origin.
        self.sites_b = np.array(sites_b, dtype=float).reshape(C.N_LEGS, 2)
        self.t_stance, self.t_swing = float(t_stance), float(t_swing)
        self.height = float(height)
        self.kv = float(kv)
        self.step_max = np.asarray(step_max, dtype=float).reshape(2)
        self.lift_w = np.full((C.N_LEGS, 2), np.nan)
        self.land_w = np.full((C.N_LEGS, 2), np.nan)
        #: (4, 2) the foot at liftoff, TRUNK frame: where an arc at rest
        #: starts.  (4,) True while the leg's swing in the air is one --
        #: decided on its first sweep (`at_rest`), never with `place_at_rest`.
        self.lift_b = np.full((C.N_LEGS, 2), np.nan)
        self.rest_swing = np.zeros(C.N_LEGS, dtype=bool)
        self.place_at_rest = bool(place_at_rest)
        #: Landings the workspace clamp moved, for the report.
        self.clamped = 0
        #: Swings flown at rest (trunk frame, onto the site), and placed.
        self.swings_at_rest = 0
        self.swings_placed = 0

    #: How near zero v_ref (m/s) and r (rad/s) must be for the reference to
    #: be AT REST.  The keys snap a sum to exactly zero and the slew lands on
    #: its target, so this absorbs float dust and nothing else.
    REST_EPS = 1e-9

    @classmethod
    def at_rest(cls, ref) -> bool:
        """True when the walking reference `ref` is AT REST: v_ref and r both
        zero -- before the first key, or once SPACE has slewed it to a stop."""
        return (abs(float(ref.v[0])) < cls.REST_EPS
                and abs(float(ref.v[1])) < cls.REST_EPS
                and abs(float(ref.yaw_rate)) < cls.REST_EPS)

    def release(self, leg: int) -> None:
        """The leg is down: forget its liftoff and plan."""
        self.lift_w[leg] = np.nan
        self.land_w[leg] = np.nan
        self.lift_b[leg] = np.nan
        self.rest_swing[leg] = False

    def foothold(self, leg: int, s: float, trunk: TrunkXY, ref) -> np.ndarray:
        """(2,) WORLD x/y the foot should touch down at -- the module's eq.
        Scalars throughout: it runs for every swinging leg every sweep."""
        t_rem = max(0.0, (1.0 - float(s)) * self.t_swing)
        r = float(ref.yaw_rate)
        vrx, vry = float(ref.v[0]), float(ref.v[1])
        vx, vy = float(trunk.v[0]), float(trunk.v[1])
        yaw_td = float(trunk.yaw) + r * t_rem
        turn = yaw_td + 0.5 * r * self.t_stance
        ca, sa = math.cos(turn), math.sin(turn)
        sx, sy = float(self.sites_b[leg, 0]), float(self.sites_b[leg, 1])
        nx = float(trunk.p[0]) + vrx * t_rem + ca * sx - sa * sy
        ny = float(trunk.p[1]) + vry * t_rem + sa * sx + ca * sy
        stx = 0.5 * self.t_stance * vx + self.kv * (vx - vrx)
        sty = 0.5 * self.t_stance * vy + self.kv * (vy - vry)
        # The workspace bound, in the heading frame at touchdown.
        ct, st = math.cos(yaw_td), math.sin(yaw_td)
        lx, ly = ct * stx + st * sty, -st * stx + ct * sty
        mx, my = float(self.step_max[0]), float(self.step_max[1])
        bx, by = min(max(lx, -mx), mx), min(max(ly, -my), my)
        if bx != lx or by != ly:
            self.clamped += 1
        return np.array([nx + ct * bx - st * by, ny + st * bx + ct * by])

    def plan(self, leg: int, s: float, trunk: TrunkXY, ref, x_b,
             z_rest: float) -> SwingRef:
        """The reference for swinging `leg` at progress `s` in [0, 1).

        `x_b` (3,) is the leg's MEASURED foot, trunk frame -- latched into the
        world on the first swinging sweep, so the arc starts exactly where
        the foot is and the first sweep of a swing is no step.  `z_rest` is
        the commanded foot depth, trunk frame (`swing.rest_feet_b`).

        The two arcs are `sim.cmpc.swing.SwingTrajectory`'s, written out in
        scalars (`selftest` holds them to it): x/y its quintic between two
        world points, z its two half-arcs to one apex with both ends at
        `z_rest`.  Building two of its objects per leg per sweep cost the law
        more than the QP did.

        AT REST (`at_rest(ref)` on the swing's first sweep, held until
        `release`) the x/y arc is the TRUNK-frame one instead, from the
        latched foot to the leg's site (`_plan_at_rest`, module docstring)."""
        c, sn = math.cos(float(trunk.yaw)), math.sin(float(trunk.yaw))
        px, py = float(trunk.p[0]), float(trunk.p[1])
        if not math.isfinite(float(self.lift_w[leg, 0])):
            bx, by = float(x_b[0]), float(x_b[1])
            self.lift_w[leg, 0] = px + c * bx - sn * by
            self.lift_w[leg, 1] = py + sn * bx + c * by
            self.lift_b[leg] = (bx, by)
            self.rest_swing[leg] = (not self.place_at_rest
                                    and self.at_rest(ref))
            if self.rest_swing[leg]:
                self.swings_at_rest += 1
            else:
                self.swings_placed += 1
        if self.rest_swing[leg]:
            return self._plan_at_rest(leg, s, c, sn, px, py, z_rest)
        self.land_w[leg] = self.foothold(leg, s, trunk, ref)
        s = min(max(float(s), 0.0), 1.0)
        rate = 1.0 / self.t_swing

        # x/y in the world: one quintic from liftoff to the landing.
        lx, ly = float(self.lift_w[leg, 0]), float(self.lift_w[leg, 1])
        dx = float(self.land_w[leg, 0]) - lx
        dy = float(self.land_w[leg, 1]) - ly
        h, dh, ddh = _quintic(s)
        pwx, pwy = lx + dx * h, ly + dy * h
        vwx, vwy = dx * dh * rate, dy * dh * rate
        awx, awy = dx * ddh * rate * rate, dy * ddh * rate * rate
        # z in the trunk: the in-place bump on the commanded height.
        pz, vz, az = self._bump(s, z_rest)

        # World -> trunk by the heading: p_b = Rz^T (p_w - p), and its rate
        # carries the turn, v_b = Rz^T (v_w - v) - w x p_b.
        r = float(trunk.yaw_rate)
        ex, ey = pwx - px, pwy - py
        pbx, pby = c * ex + sn * ey, -sn * ex + c * ey
        ex, ey = vwx - float(trunk.v[0]), vwy - float(trunk.v[1])
        vbx = c * ex + sn * ey + r * pby
        vby = -sn * ex + c * ey - r * pbx
        ex, ey = awx - float(ref.a[0]), awy - float(ref.a[1])
        abx = c * ex + sn * ey + 2.0 * r * vby + r * r * pbx
        aby = -sn * ex + c * ey - 2.0 * r * vbx + r * r * pby
        return SwingRef(p=np.array([pbx, pby, pz]),
                        v=np.array([vbx, vby, vz]),
                        a=np.array([abx, aby, az]),
                        land_w=self.land_w[leg].copy(),
                        lift_w=self.lift_w[leg].copy())

    def _bump(self, s: float, z_rest: float):
        """``(pz, vz, az)``, trunk z at clamped progress `s`: the in-place bump
        on the commanded height.  Both arcs fly this one."""
        rate = 1.0 / self.t_swing
        seg, dseg = (2.0 * s, 2.0) if s < 0.5 else (2.0 - 2.0 * s, -2.0)
        hz, dhz, ddhz = _quintic(seg)
        return (z_rest + self.height * hz,
                self.height * dhz * dseg * rate,
                self.height * ddhz * dseg * dseg * rate * rate)

    def _plan_at_rest(self, leg: int, s: float, c: float, sn: float,
                      px: float, py: float, z_rest: float) -> SwingRef:
        """The swing AT REST: TRUNK frame, the x/y quintic from the foot
        latched at liftoff to the leg's site, and the z bump.  The filter's
        heading and position (`c, sn, px, py`) place only the logged `land_w`,
        the site where the filter has it now."""
        s = min(max(float(s), 0.0), 1.0)
        rate = 1.0 / self.t_swing
        lx, ly = float(self.lift_b[leg, 0]), float(self.lift_b[leg, 1])
        sx, sy = float(self.sites_b[leg, 0]), float(self.sites_b[leg, 1])
        dx, dy = sx - lx, sy - ly
        h, dh, ddh = _quintic(s)
        pz, vz, az = self._bump(s, z_rest)
        self.land_w[leg, 0] = px + c * sx - sn * sy
        self.land_w[leg, 1] = py + sn * sx + c * sy
        return SwingRef(p=np.array([lx + dx * h, ly + dy * h, pz]),
                        v=np.array([dx * dh * rate, dy * dh * rate, vz]),
                        a=np.array([dx * ddh * rate * rate,
                                    dy * ddh * rate * rate, az]),
                        land_w=self.land_w[leg].copy(),
                        lift_w=self.lift_w[leg].copy())


def describe() -> str:
    from .. import fold_trot as FT
    from sim import params as P
    from .trajectory import RefSample, Command
    sites = np.asarray(P.HIP_OFFSET)[:, :2] + FT.STAND_XY
    t_st = cfg.WALK_DUTY * FT.PERIOD_S
    t_sw = (1.0 - cfg.WALK_DUTY) * FT.PERIOD_S
    planner = FootstepPlanner(sites, t_stance=t_st, t_swing=t_sw)
    rows = ["DOG6 footstep planner: eq (33) + %.2f s feedback, sites = the "
            "stance's own feet" % cfg.STEP_KV,
            "  stance %.0f ms  swing %.0f ms  step bound %s mm"
            % (1e3 * t_st, 1e3 * t_sw, 1e3 * planner.step_max)]
    for v, r in ((0.1, 0.0), (0.0, np.radians(30.0)), (0.1, np.radians(30.0))):
        ref = RefSample(p=np.zeros(2), v=np.array([v, 0.0]), a=np.zeros(2),
                        yaw=0.0, yaw_rate=float(r),
                        command=Command(vx=v, vy=0.0, yaw_rate=float(r)),
                        leashed=False)
        trunk = TrunkXY(p=np.zeros(2), v=np.array([v, 0.0]), yaw=0.0,
                        yaw_rate=float(r), measured=True)
        land = planner.foothold(0, 0.0, trunk, ref)
        rows.append("  v %.2f m/s, r %+3.0f deg/s: FL lands %s mm (site %s), "
                    "i.e. %s mm from its site"
                    % (v, np.degrees(r), np.round(1e3 * land, 1),
                       np.round(1e3 * sites[0], 1),
                       np.round(1e3 * (land - sites[0]), 1)))
    rows.append("  at rest (v_ref 0, r 0): %s"
                % ("placed as above, v_hat in it (place_at_rest)"
                   if planner.place_at_rest else
                   "the TRUNK-frame arc from the liftoff foot to the site, "
                   "no velocity term -- hw.fold_trot's in-place target"))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
