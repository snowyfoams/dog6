"""analyse -- the tracked quantities of a trot or walk, from an `hw.stand --log` npz.

    python doc/gait_study/analyse.py p1_inplace_1.npz               summary, figures beside the log
    python doc/gait_study/analyse.py p1_lat_1.npz --out figs/       figures into figs/
    python doc/gait_study/analyse.py a.npz b.npz c.npz              one summary block each
    python doc/gait_study/analyse.py p1_lat_1.npz --zoom 60 62.4    the anatomy figure on trot seconds 60..62.4
    python doc/gait_study/analyse.py p1_lat_1.npz --setpoint -0.29 0.12   the attitude setpoint, deg (default: config's statics)

No touchdown TIME is estimated anywhere.  Contact is read as a HEIGHT at the
instants the gait clock defines: each foot's clearance above the plane of
the planted feet (`z_clear`, from the encoders and the IMU), sampled where
the clock lands the foot (`h_td`) and lifts it (the apex, the scuffs), and
the normal force the allocator asked of a foot while it was above 4 mm
(`Phi`).  Everything else is the log's own columns, per sweep, per swing,
per landing, per cycle.

WHAT THE LOG IS, AND TWO THINGS TO KNOW ABOUT IT
    `hw.stand.StandLog` writes one row a sweep.  Its `t` column RESTARTS AT
    ZERO IN EVERY PHASE (hold 0..2, trot 0..120), so an absolute clock is
    rebuilt here from the row order and the sweep intervals.  Its `qd`
    column is one row repeated (commit 9a4e591, "known, not fixed"), so
    every velocity below is a differentiated position.

THE FRAME
    "Levelled" means the trunk frame rotated by the IMU's roll and pitch
    only: z up, x the trunk's heading, y its left.  Foot heights, the stance
    width and the lateral ratchet are read in it.  Roll > 0 is the RIGHT
    side down (sim.coordinates.rot_zyx); a lateral offset > 0 is to the LEFT
    (+y, the A key).

THE QUANTITIES (doc/gait_study/README.md section 2)
    per sweep     roll, pitch from the setpoint; tilt; the governor's lean
                  replayed (1.5 Hz); z_clear per foot; the residual moment
                  on the support diagonal during two-foot sweeps
    per swing     apex reached; scuffed (never above 4 mm); liftoff lag;
                  the swing tracking error; the roll at liftoff and at the
                  clock's landing, and the landing's lateral offset from the
                  liftoff point in the levelled frame (the ratchet)
    per landing   h_td; Phi; the roll step across the 80 ms after it
    per cycle     stance width; each foot's offset from its HOLD site; the
                  cycle mean of e_R and of the commanded moment; fz per
                  leg; the estimator's error and the x/y rows' force
    per run       fell (the fall hold engaged), the time and the side;
                  the estimator's drift; torque per amp by joint
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from sim import coordinates as C             # noqa: E402
from hw.balance import config as BCFG         # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
LEFT = (0, 2)                     # FL, RL: +y
RIGHT = (1, 3)                    # FR, RR: -y
CLEAR_MM = 4.0                    # the gate: below it a foot is "down"
STEP_MS = 80.0                    # the roll step is read this long after a landing
LEAN_HZ = float(BCFG.WALK_LEAN_HZ)


# ===========================================================================
# loading, the clock, the frame
# ===========================================================================
def load(path: str) -> dict:
    """The npz as a dict, with `t_abs` (s, one clock for the whole run) and
    `dt` (s, the sweep interval) added."""
    raw = np.load(path, allow_pickle=False)
    d = {k: raw[k] for k in raw.files}
    t = np.asarray(d["t"], dtype=float)
    dt = np.diff(t)
    good = dt[(dt > 0.0) & (dt < 0.1)]
    med = float(np.median(good)) if good.size else 0.004
    step = np.where((dt > 0.0) & (dt < 0.1), dt, med)
    d["t_abs"] = np.concatenate([[0.0], np.cumsum(step)])
    d["dt"] = med
    d["name"] = os.path.splitext(os.path.basename(path))[0]
    return d


def rotations(roll, pitch):
    """(n, 3, 3) R = Ry(pitch) Rx(roll): the levelled frame, no yaw."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    n = roll.size
    R = np.zeros((n, 3, 3))
    # Ry(p) @ Rx(r)
    R[:, 0, 0] = cp
    R[:, 0, 1] = sp * sr
    R[:, 0, 2] = sp * cr
    R[:, 1, 0] = 0.0
    R[:, 1, 1] = cr
    R[:, 1, 2] = -sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def lean_replay(t, roll_err, pitch_err, hz=LEAN_HZ):
    """`balance.governor`'s filter on the log: the larger of the two
    low-passed errors, deg."""
    lr = lp = 0.0
    out = np.zeros(t.size)
    prev = None
    for i in range(t.size):
        if prev is None:
            lr, lp = float(roll_err[i]), float(pitch_err[i])
        else:
            dt = min(max(float(t[i] - prev), 0.0), 0.05)
            a = 1.0 - math.exp(-2.0 * math.pi * hz * dt)
            lr += a * (float(roll_err[i]) - lr)
            lp += a * (float(pitch_err[i]) - lp)
        prev = float(t[i])
        out[i] = math.degrees(max(abs(lr), abs(lp)))
    return out


# ===========================================================================
# the analysis
# ===========================================================================
class Run:
    """One log, analysed.  Every array is over the TROT rows unless named
    otherwise; `self.sel` is their index into the log."""

    def __init__(self, d: dict, setpoint_deg=(None, None)):
        self.d = d
        self.name = d["name"]
        ph = d["phase"]
        self.phase = ph
        self.fell = bool(np.any(ph == "fall"))
        trot = np.flatnonzero(ph == "trot")
        if not trot.size:
            raise ValueError("%s: no trot rows" % self.name)
        self.sel = trot
        t_abs = d["t_abs"]
        self.t = t_abs[trot] - t_abs[trot[0]]          # s into the trot
        self.dt = d["dt"]
        sp_r = (BCFG.SETPOINT_ROLL_DEG if setpoint_deg[0] is None
                else setpoint_deg[0])
        sp_p = (BCFG.SETPOINT_PITCH_DEG if setpoint_deg[1] is None
                else setpoint_deg[1])
        self.sp = (float(sp_r), float(sp_p))
        self.roll = np.degrees(d["roll"][trot])          # from true level
        self.pitch = np.degrees(d["pitch"][trot])
        self.yaw = np.degrees(np.unwrap(d["yaw"][trot]))
        self.roll_e = self.roll - sp_r                   # from the setpoint
        self.pitch_e = self.pitch - sp_p
        self.tilt = np.maximum(np.abs(self.roll_e), np.abs(self.pitch_e))
        self.lean = lean_replay(self.t, np.radians(self.roll_e),
                                np.radians(self.pitch_e))
        self.h = 1e3 * d["h"][trot]
        self.h_cmd = 1e3 * d["h_cmd"][trot]
        self.cw = d["contact_w"][trot]                   # (n, 4)
        self.contact = self.cw > 0.0
        self.swing_s = d["swing_s"][trot]
        self.x_b = d["x_b"][trot]                        # (n, 4, 3) trunk
        self.p_swing = d["p_swing"][trot]
        self.fz = d["fz"][trot]
        self.f_w = d["f_w"][trot]
        self.b_d = d["b_d"][trot]
        self.e_R = np.degrees(d["e_R"][trot])
        self.residual = d["residual"][trot]
        self.p_est = d["p_est"][trot]
        self.v_est = d["v_est"][trot]
        self.xy_des = d["xy_des"][trot]
        self.xy_on = d["xy_on"][trot]
        self.tau_cmd = d["tau_cmd"][trot]
        self.tau_meas = d["tau_meas"][trot]
        self.tau_req = d["tau_req"][trot]
        self.com_pin = d["com_pin"][trot]
        self.q = d["q"][trot]
        # the fall, if any: the first `fall` row, in trot seconds; and the
        # tip, the first sweep past 15 deg from the setpoint (a run the
        # operator stopped before the fall hold took it has the tip and no
        # fall)
        self.t_fall = None
        self.fall_side = None
        fall = np.flatnonzero(ph == "fall")
        if fall.size:
            self.t_fall = float(t_abs[fall[0]] - t_abs[trot[0]])
            r = float(np.degrees(d["roll"][fall[0]]))
            self.fall_side = "right" if r > 0 else "left"
        self.t_tip = None
        self.tip_side = None
        tip = np.flatnonzero(self.tilt > 15.0)
        if tip.size:
            self.t_tip = float(self.t[tip[0]])
            self.tip_side = ("right" if self.roll_e[tip[0]] > 0 else "left"
                             if abs(self.roll_e[tip[0]]) > abs(self.pitch_e[tip[0]])
                             else "nose" if self.pitch_e[tip[0]] > 0 else "tail")
        # the levelled frame
        R = rotations(d["roll"][trot], d["pitch"][trot])
        self.x_lev = np.einsum("nij,nkj->nki", R, self.x_b)   # (n, 4, 3)
        # the HOLD's sites: the trunk-frame feet over the last second of the
        # first HOLD before the trot
        hold = np.flatnonzero(ph == "hold")
        hold = hold[hold < trot[0]]
        if hold.size:
            last = hold[-max(1, int(round(1.0 / self.dt))):]
            self.sites = np.nanmean(d["x_b"][last], axis=0)
            self.hold_rows = last
        else:
            self.sites = np.nanmean(self.x_b[:50], axis=0)
            self.hold_rows = None
        self._clock()
        self._planarity()
        self._clearance()
        self._swings()
        self._landings()
        self._cycles()
        self._diagonal()
        self._estimator()
        self._torque()

    # -- the clock -----------------------------------------------------------
    def _clock(self):
        c = self.contact
        self.lo = []          # per leg: liftoff indices
        self.td = []          # per leg: touchdown indices
        for i in range(4):
            ci = c[:, i]
            self.lo.append(np.flatnonzero(ci[:-1] & ~ci[1:]) + 1)
            self.td.append(np.flatnonzero(~ci[:-1] & ci[1:]) + 1)
        per = np.diff(self.t[self.lo[0]]) if self.lo[0].size > 1 else np.array([np.nan])
        self.period = float(np.nanmedian(per))
        sw = []
        for i in range(4):
            for a in self.lo[i]:
                b = self.td[i][self.td[i] > a]
                if b.size:
                    sw.append(self.t[b[0]] - self.t[a])
        self.swing_s_median = float(np.median(sw)) if sw else np.nan
        full = self.cw >= 1.0 - 1e-9
        rm = []
        for i in range(4):
            for a in self.td[i]:
                b = np.flatnonzero(full[a:, i])
                if b.size:
                    rm.append(self.t[a + b[0]] - self.t[a])
        self.ramp_s_median = float(np.median(rm)) if rm else np.nan
        self.n_two_foot = int((c.sum(axis=1) == 2).sum())
        self.n_cycles = int(self.lo[0].size)

    # -- the feet's plane against the IMU --------------------------------------
    @staticmethod
    def _feet_attitude(x_b):
        """(roll, pitch) deg of the TRUNK relative to the plane through the
        four feet, from trunk-frame foot positions (k, 4, 3) or (4, 3):
        z = a x + b y + c fitted to the four feet; roll > 0 right side down
        (the left feet then sit LOWER in the trunk frame: dz/dy < 0), pitch >
        0 nose down (the front feet sit higher: dz/dx > 0)."""
        p = np.asarray(x_b, dtype=float).reshape(-1, 4, 3).mean(axis=0)
        A = np.column_stack([p[:, 0], p[:, 1], np.ones(4)])
        a, b, _ = np.linalg.lstsq(A, p[:, 2], rcond=None)[0]
        return math.degrees(math.atan(-b)), math.degrees(math.atan(a))

    def _planarity(self):
        """Each foot's levelled height minus the four-foot mean while all
        four carry full weight: on a flat floor the feet ARE coplanar, so
        what the levelled frame reads as a height difference is the IMU's
        attitude against the plane the feet stand on (its mount, the floor's
        slope, or both).  Subtracted from every clearance below, so that
        `h_td`, the apex and `Phi` read the swing and not the datum."""
        full = (self.cw >= 1.0 - 1e-9).all(axis=1)
        z = self.x_lev[:, :, 2]
        if full.sum() > 20:
            dev = z[full] - z[full].mean(axis=1, keepdims=True)
            self.plane_offset = 1e3 * dev.mean(axis=0)          # mm, (4,)
            self.plane_offset_std = 1e3 * dev.std(axis=0)
            self.plane_from = "%d four-foot full-weight sweeps of the trot" % full.sum()
            feet = self._feet_attitude(self.x_b[full])
            imu = (float(self.roll[full].mean()), float(self.pitch[full].mean()))
        elif self.hold_rows is not None:
            xb = self.d["x_b"][self.hold_rows]
            R = rotations(self.d["roll"][self.hold_rows], self.d["pitch"][self.hold_rows])
            zl = np.einsum("nij,nkj->nki", R, xb)[:, :, 2]
            dev = zl - zl.mean(axis=1, keepdims=True)
            self.plane_offset = 1e3 * dev.mean(axis=0)
            self.plane_offset_std = 1e3 * dev.std(axis=0)
            self.plane_from = "the HOLD before the trot"
            feet = self._feet_attitude(xb)
            imu = (float(np.degrees(self.d["roll"][self.hold_rows].mean())),
                   float(np.degrees(self.d["pitch"][self.hold_rows].mean())))
        else:
            self.plane_offset = np.zeros(4)
            self.plane_offset_std = np.zeros(4)
            self.plane_from = "nothing (no four-foot window, no HOLD): uncorrected"
            feet, imu = (np.nan, np.nan), (np.nan, np.nan)
        self.feet_att = feet
        self.imu_att = imu
        # the same pair in the HOLD, for the record
        if self.hold_rows is not None:
            self.hold_feet_att = self._feet_attitude(self.d["x_b"][self.hold_rows])
            self.hold_imu_att = (float(np.degrees(self.d["roll"][self.hold_rows].mean())),
                                 float(np.degrees(self.d["pitch"][self.hold_rows].mean())))
        else:
            self.hold_feet_att = self.hold_imu_att = (np.nan, np.nan)

    # -- the feet's heights ----------------------------------------------------
    def _clearance(self):
        z = self.x_lev[:, :, 2] - 1e-3 * self.plane_offset[None, :]   # corrected
        n = z.shape[0]
        plane = np.zeros(n)
        for k in range(n):
            planted = np.flatnonzero(self.contact[k])
            if planted.size >= 2:
                zz = np.sort(z[k, planted])
                plane[k] = zz[:2].mean()
            elif planted.size == 1:
                plane[k] = z[k, planted[0]]
            else:
                plane[k] = np.sort(z[k])[:2].mean()
        self.z_clear = 1e3 * (z - plane[:, None])   # mm, (n, 4)
        zr = self.x_lev[:, :, 2]
        self.z_clear_raw = 1e3 * (zr - plane[:, None] - 1e-3 * self.plane_offset.mean())
        # THE CLOCK'S OWN ARC: the swing reference `p_swing` (trunk frame,
        # NaN in stance), levelled and measured against the same plane and
        # the same per-foot offset -- what the law asked the foot to do,
        # beside what it did.
        R = rotations(self.d["roll"][self.sel], self.d["pitch"][self.sel])
        p_lev = np.einsum("nij,nkj->nki", R, np.nan_to_num(self.p_swing, nan=0.0))
        z_ref = p_lev[:, :, 2] - 1e-3 * self.plane_offset[None, :]
        self.z_ref = 1e3 * (z_ref - plane[:, None])
        self.z_ref[~np.isfinite(self.p_swing[:, :, 2])] = np.nan

    # -- per swing -------------------------------------------------------------
    def _swings(self):
        rows = []
        for i in range(4):
            for a in self.lo[i]:
                b = self.td[i][self.td[i] > a]
                if not b.size:
                    continue
                b = int(b[0])
                zc = self.z_clear[a:b, i]
                apex = float(np.nanmax(zc)) if zc.size else np.nan
                up = np.flatnonzero(zc > CLEAR_MM)
                lag = float(self.t[a + up[0]] - self.t[a]) if up.size else np.inf
                err = self.p_swing[a:b, i] - self.x_b[a:b, i]
                ok = np.isfinite(err).all(axis=1)
                rms = (1e3 * float(np.sqrt(np.nanmean(np.sum(err[ok] ** 2, axis=1))))
                       if ok.any() else np.nan)
                e_end = (1e3 * err[ok][-1] if ok.any() else np.full(3, np.nan))
                # the ratchet: where the foot comes down against where it
                # lifted, in the levelled frame, LESS the trunk's own motion
                # over the same interval as the planted feet show it (a
                # planted foot's levelled x/y moves by minus the trunk's).
                # What is left is the foot's displacement over the floor, if
                # the planted feet did not slip; a slip is invisible here,
                # as it is to the leg odometry.
                dxy = 1e3 * (self.x_lev[b, i, :2] - self.x_lev[a, i, :2])
                others = [j for j in range(4) if j != i and self.contact[a, j]
                          and self.contact[b, j]]
                if others:
                    trunk = 1e3 * np.mean([self.x_lev[b, j, :2] - self.x_lev[a, j, :2]
                                           for j in others], axis=0)
                else:
                    trunk = np.zeros(2)
                rows.append(dict(leg=i, t_lo=float(self.t[a]), t_td=float(self.t[b]),
                                 i_lo=int(a), i_td=b, apex=apex, lag=lag,
                                 scuffed=bool(apex < CLEAR_MM), rms=rms,
                                 e_end_y=float(e_end[1]), e_end_x=float(e_end[0]),
                                 roll_lo=float(self.roll_e[a]), roll_td=float(self.roll_e[b]),
                                 pitch_lo=float(self.pitch_e[a]), pitch_td=float(self.pitch_e[b]),
                                 dx=float(dxy[0]), dy=float(dxy[1]),
                                 dx_floor=float(dxy[0] - trunk[0]),
                                 dy_floor=float(dxy[1] - trunk[1])))
        self.swings = rows

    # -- per landing -----------------------------------------------------------
    def _landings(self):
        rows = []
        n = self.t.size
        k80 = max(1, int(round(STEP_MS * 1e-3 / self.dt)))
        k10 = max(1, int(round(0.010 / self.dt)))
        for i in range(4):
            for b in self.td[i]:
                b = int(b)
                h_td = float(self.z_clear[b, i])
                # Phi: fz asked while the foot is still above the gate
                j = b
                phi = 0.0
                air = 0
                while j < n and self.contact[j, i]:
                    if self.z_clear[j, i] > CLEAR_MM:
                        phi += float(self.fz[j, i]) * self.dt
                        air += 1
                    else:
                        break
                    j += 1
                a, c = max(0, b - k10), min(n - 1, b + k80)
                dphi = float(self.roll_e[c] - self.roll_e[a])
                toward = dphi if i in RIGHT else -dphi
                rows.append(dict(leg=i, t=float(self.t[b]), i_td=b, h_td=h_td,
                                 phi=phi, air_ms=1e3 * air * self.dt,
                                 dphi=dphi, dphi_toward=toward))
        self.landings = rows

    # -- per cycle -------------------------------------------------------------
    def _cycles(self):
        rows = []
        lo = self.lo[0]
        for k in range(lo.size - 1):
            a, b = int(lo[k]), int(lo[k + 1])
            sl = slice(a, b)
            four = np.flatnonzero(self.contact[sl].all(axis=1)) + a
            mid = four if four.size else np.arange(a, b)
            y = self.x_lev[mid, :, 1].mean(axis=0)
            x = self.x_lev[mid, :, 0].mean(axis=0)
            off = 1e3 * (self.x_b[mid].mean(axis=0)[:, :2] - self.sites[:, :2])
            cmd = np.abs(self.tau_cmd[sl])
            meas = np.abs(self.tau_meas[sl])
            planted = np.repeat(self.contact[sl], 3, axis=1) & (cmd > 0.2)
            kt = np.array([(meas[:, j][planted[:, j]].sum() / cmd[:, j][planted[:, j]].sum())
                           if planted[:, j].any() else np.nan for j in range(12)])
            rows.append(dict(
                k=k, t=float(self.t[a]), kt=kt,
                width_front=1e3 * float(y[0] - y[1]),
                width_rear=1e3 * float(y[2] - y[3]),
                length_left=1e3 * float(x[0] - x[2]),
                length_right=1e3 * float(x[1] - x[3]),
                off_x=off[:, 0], off_y=off[:, 1],
                e_R=self.e_R[sl].mean(axis=0),
                M=self.b_d[sl, 3:6].mean(axis=0),
                F=self.b_d[sl, 0:2].mean(axis=0),
                fz=np.nanmean(self.fz[sl], axis=0),
                roll=float(self.roll_e[sl].mean()),
                pitch=float(self.pitch_e[sl].mean()),
                tilt_rms=float(np.sqrt(np.mean(self.tilt[sl] ** 2))),
                v_est=np.nanmean(self.v_est[sl, :2], axis=0),
                xy_err=(np.nanmean(self.xy_des[sl] - (self.p_est[sl, :2]), axis=0)
                        if np.isfinite(self.xy_des[sl]).any() else np.full(2, np.nan)),
            ))
        self.cycles = rows

    # -- the support diagonal ---------------------------------------------------
    def _diagonal(self):
        """Two-foot sweeps: the commanded moment and the residual projected
        on the support line; two-foot PHASES: the roll excursion against the
        CoM's signed offset from the line at the phase's first sweep."""
        c = self.contact
        two = c.sum(axis=1) == 2
        n = self.t.size
        proj_cmd = np.full(n, np.nan)
        proj_res = np.full(n, np.nan)
        d_off = np.full(n, np.nan)
        com = np.zeros((n, 3))
        com[:, :2] = np.nan_to_num(self.com_pin)
        R = rotations(self.d["roll"][self.sel], self.d["pitch"][self.sel])
        com_lev = np.einsum("nij,nj->ni", R, com)
        for k in np.flatnonzero(two):
            i, j = np.flatnonzero(c[k])
            p1, p2 = self.x_lev[k, i, :2], self.x_lev[k, j, :2]
            u = p2 - p1
            L = np.linalg.norm(u)
            if L < 1e-6:
                continue
            u = u / L
            proj_cmd[k] = float(self.b_d[k, 3:5] @ u)
            # `residual` is A f - b_d (allocation.py): what was DELIVERED on
            # the line is b_d + residual.  Two feet on a line can deliver,
            # about the CoM, only the total force's moment about that line --
            # the weight times the CoM's offset from it -- never the PD's.
            proj_res[k] = float((self.b_d[k, 3:5] + self.residual[k, 3:5]) @ u)
            # the CoM's signed offset from the line, m: + to the line's left
            v = com_lev[k, :2] - p1
            d_off[k] = float(u[0] * v[1] - u[1] * v[0])
        self.proj_cmd, self.proj_res, self.d_off = proj_cmd, proj_res, d_off
        # phases
        rows = []
        k = 0
        while k < n:
            if two[k]:
                a = k
                while k < n and two[k] and (c[k] == c[a]).all():
                    k += 1
                b = k - 1
                if b > a + 2:
                    pair = "".join(LEGS[i] for i in np.flatnonzero(c[a]))
                    rows.append(dict(t=float(self.t[a]), pair=pair,
                                     d_mm=1e3 * float(d_off[a]),
                                     roll_a=float(self.roll_e[a]),
                                     excursion=float(self.roll_e[b] - self.roll_e[a]),
                                     peak=float(np.max(np.abs(self.roll_e[a:b + 1] - self.roll_e[a]))),
                                     dur_ms=1e3 * float(self.t[b] - self.t[a]),
                                     cmd=float(np.nanmean(proj_cmd[a:b + 1])),
                                     delivered=float(np.nanmean(proj_res[a:b + 1])),
                                     toppling=float(BCFG.WEIGHT * np.nanmean(d_off[a:b + 1]))))
            else:
                k += 1
        self.two_foot = rows

    # -- the estimator and the x/y rows ------------------------------------------
    def _estimator(self):
        p = self.p_est
        ok = np.isfinite(p[:, 0])
        self.est_drift = (1e3 * (p[ok][-1, :2] - p[ok][0, :2]) if ok.any()
                          else np.full(2, np.nan))
        self.v_est_mean = np.nanmean(self.v_est[:, :2], axis=0)
        self.F_xy_mean = np.nanmean(self.b_d[:, :2], axis=0)
        self.M_mean = np.nanmean(self.b_d[:, 3:6], axis=0)
        self.xy_on_frac = float(np.nanmean(self.xy_on)) if np.isfinite(self.xy_on).any() else np.nan
        self.f_w_mean = np.nanmean(self.f_w, axis=0)          # (4, 3)
        self.fz_mean = np.nanmean(self.fz, axis=0)

    # -- torque ------------------------------------------------------------------
    def _torque(self):
        planted = np.repeat(self.contact, 3, axis=1)          # (n, 12)
        cmd = np.abs(self.tau_cmd)
        meas = np.abs(self.tau_meas)
        kt = np.zeros(12)
        for j in range(12):
            m = planted[:, j] & (cmd[:, j] > 0.2)
            kt[j] = (meas[m, j].sum() / cmd[m, j].sum()) if m.any() else np.nan
        self.kt = kt
        self.tau_peak = float(np.nanmax(np.abs(self.tau_cmd)))
        self.slew_deficit = float(np.nansum(np.maximum(
            0.0, np.abs(self.tau_req) - np.abs(self.tau_cmd))) * self.dt)
        self.clip_sweeps = int((np.abs(self.tau_cmd) >= 8.99).any(axis=1).sum())

    # -- the summary --------------------------------------------------------------
    def summary(self) -> str:
        L = []
        w = L.append
        sw = self.swings
        ld = self.landings
        cy = self.cycles
        w("== %s ==" % self.name)
        outcome = ("FELL (the fall hold) at %.1f s to the %s" % (self.t_fall, self.fall_side)
                   if self.fell else "no fall hold")
        if self.t_tip is not None:
            outcome += "; TIPPED past 15 deg at %.1f s, %s side down, %.0f deg at the end" % (
                self.t_tip, self.tip_side, self.tilt[-1])
        w("  trot %.1f s, %d cycles; period %.3f s, swing %.0f ms, ramp %.0f ms; "
          "sweep %.2f ms; %s"
          % (self.t[-1], self.n_cycles, self.period, 1e3 * self.swing_s_median,
             1e3 * self.ramp_s_median, 1e3 * self.dt, outcome))
        w("  attitude from the setpoint (%.2f / %.2f deg): roll mean %+.2f  rms %.2f  "
          "min %+.2f max %+.2f;  pitch mean %+.2f rms %.2f;  tilt max %.2f;  lean max %.2f"
          % (self.sp[0], self.sp[1], self.roll_e.mean(),
             np.sqrt(np.mean(self.roll_e ** 2)), self.roll_e.min(),
             self.roll_e.max(), self.pitch_e.mean(),
             np.sqrt(np.mean(self.pitch_e ** 2)), self.tilt.max(), self.lean.max()))
        w("  tilt rms per 10 s: %s"
          % " ".join("%.2f" % v for v in self.tilt_rms_windows(10.0)))
        w("  height mean %.1f mm (cmd %.1f), min %.1f;  yaw drift %+.2f deg"
          % (self.h.mean(), self.h_cmd.mean(), self.h.min(),
             self.yaw[-1] - self.yaw[0]))
        w("  THE IMU AGAINST THE FEET.  Trunk attitude relative to the plane of the four feet "
          "(roll, pitch, deg): HOLD %+.2f / %+.2f, trot four-foot windows %+.2f / %+.2f;  the IMU "
          "says HOLD %+.2f / %+.2f, trot %+.2f / %+.2f.  IMU minus feet: HOLD %+.2f / %+.2f, "
          "trot %+.2f / %+.2f -- on a flat floor that is the IMU's datum error (mount + statics)"
          % (self.hold_feet_att[0], self.hold_feet_att[1], self.feet_att[0], self.feet_att[1],
             self.hold_imu_att[0], self.hold_imu_att[1], self.imu_att[0], self.imu_att[1],
             self.hold_imu_att[0] - self.hold_feet_att[0], self.hold_imu_att[1] - self.hold_feet_att[1],
             self.imu_att[0] - self.feet_att[0], self.imu_att[1] - self.feet_att[1]))
        w("  levelled foot heights with all four loaded, minus the mean (mm, FL FR RL RR): %s "
          "(std %s), from %s -- subtracted from every clearance below"
          % (" ".join("%+.1f" % v for v in self.plane_offset),
             " ".join("%.1f" % v for v in self.plane_offset_std), self.plane_from))
        for i, leg in enumerate(LEGS):
            s = [r for r in sw if r["leg"] == i]
            l_ = [r for r in ld if r["leg"] == i]
            if not s or not l_:
                continue
            apex = np.array([r["apex"] for r in s])
            lag = np.array([r["lag"] for r in s])
            htd = np.array([r["h_td"] for r in l_])
            phi = np.array([r["phi"] for r in l_])
            dph = np.array([r["dphi_toward"] for r in l_])
            w("  %s  swings %3d: apex med %5.1f mm (p10 %5.1f), scuffed %3.0f %%, "
              "liftoff lag med %3.0f ms, track rms %4.1f mm, end y err med %+5.1f mm | "
              "h_td med %4.1f p90 %4.1f mm, Phi med %.2f max %.2f N s, in air %3.0f ms med, "
              "roll step toward med %+.2f p90 %+.2f deg"
              % (leg, len(s), np.median(apex), np.percentile(apex, 10),
                 100.0 * np.mean([r["scuffed"] for r in s]),
                 1e3 * np.median(lag[np.isfinite(lag)]) if np.isfinite(lag).any() else np.nan,
                 np.nanmedian([r["rms"] for r in s]),
                 np.nanmedian([r["e_end_y"] for r in s]),
                 np.median(htd), np.percentile(htd, 90), np.median(phi), phi.max(),
                 np.median([r["air_ms"] for r in l_]),
                 np.median(dph), np.percentile(dph, 90)))
        # the ratchet
        dy = np.array([r["dy_floor"] for r in sw])
        dx = np.array([r["dx_floor"] for r in sw])
        dr = np.array([r["roll_td"] - r["roll_lo"] for r in sw])
        w("  the ratchet (each foot's landing against its liftoff, over the floor as the planted "
          "feet show it): dy mean %+.2f mm per swing -> %+.0f mm over the run (%+.2f mm per cycle, "
          "averaged over the four feet); dx %+.2f mm per swing -> %+.0f mm.  A real drift that is "
          "NOT here moved the planted feet: slip.  Roll at landing minus at liftoff: mean %+.2f deg"
          % (dy.mean(), dy.sum() / 4.0, dy.sum() / 4.0 / max(1, self.n_cycles),
             dx.mean(), dx.sum() / 4.0, dr.mean()))
        for i, leg in enumerate(LEGS):
            s = [r for r in sw if r["leg"] == i]
            if s:
                w("      %s: dy %+.2f mm/swing (levelled, before the trunk's motion %+.2f), "
                  "roll at lo %+.2f at td %+.2f"
                  % (leg, np.mean([r["dy_floor"] for r in s]), np.mean([r["dy"] for r in s]),
                     np.mean([r["roll_lo"] for r in s]), np.mean([r["roll_td"] for r in s])))
        # cycles
        if len(cy) > 4:
            t = np.array([r["t"] for r in cy])
            wf = np.array([r["width_front"] for r in cy])
            wr = np.array([r["width_rear"] for r in cy])
            offy = np.array([r["off_y"] for r in cy])       # (k, 4)
            offx = np.array([r["off_x"] for r in cy])
            eR = np.array([r["e_R"] for r in cy])
            M = np.array([r["M"] for r in cy])
            w("  stance width front %.1f -> %.1f mm (slope %+.2f mm/min), rear %.1f -> %.1f (%+.2f mm/min)"
              % (wf[:3].mean(), wf[-3:].mean(), 60.0 * np.polyfit(t, wf, 1)[0],
                 wr[:3].mean(), wr[-3:].mean(), 60.0 * np.polyfit(t, wr, 1)[0]))
            w("  foot offset from the HOLD site, y (mm, start -> end): %s;  x: %s"
              % (", ".join("%s %+.1f->%+.1f" % (LEGS[i], offy[:3, i].mean(), offy[-3:, i].mean())
                           for i in range(4)),
                 ", ".join("%s %+.1f->%+.1f" % (LEGS[i], offx[:3, i].mean(), offx[-3:, i].mean())
                           for i in range(4))))
            w("  e_R cycle mean (deg) roll %+.2f -> %+.2f (slope %+.3f deg/min), pitch %+.2f -> %+.2f;  "
              "commanded moment mean x %+.3f y %+.3f N m (slope x %+.4f N m/min)"
              % (eR[:5, 0].mean(), eR[-5:, 0].mean(), 60.0 * np.polyfit(t, eR[:, 0], 1)[0],
                 eR[:5, 1].mean(), eR[-5:, 1].mean(), M[:, 0].mean(), M[:, 1].mean(),
                 60.0 * np.polyfit(t, M[:, 0], 1)[0]))
        # the diagonal
        tf = self.two_foot
        if tf:
            cmd = np.array([r["cmd"] for r in tf])
            dl = np.array([r["delivered"] for r in tf])
            tp = np.array([r["toppling"] for r in tf])
            w("  two-foot phases %d (%.0f ms median).  Moment about the support line, per phase: "
              "commanded |median| %.3f N m, delivered |median| %.3f, and the weight's own toppling "
              "moment W d |median| %.3f -- delivered minus toppling median %+.3f (two feet can make "
              "no other).  Roll excursion over a phase median %+.2f deg, |peak| median %.2f"
              % (len(tf), np.median([r["dur_ms"] for r in tf]), np.median(np.abs(cmd)),
                 np.median(np.abs(dl)), np.median(np.abs(tp)), np.median(dl - tp),
                 np.median([r["excursion"] for r in tf]), np.median([r["peak"] for r in tf])))
        # the slow part of the hardware: torque per amp by leg, first and last fifth
        if len(cy) > 10:
            kt = np.array([r["kt"] for r in cy])
            n5 = max(1, len(cy) // 5)
            w("  torque per amp by joint, first fifth -> last fifth of the trot (abd/pitch/knee): %s"
              % "  ".join("%s %s" % (LEGS[i], "/".join(
                  "%.2f>%.2f" % (np.nanmean(kt[:n5, 3 * i + j]), np.nanmean(kt[-n5:, 3 * i + j]))
                  for j in range(3))) for i in range(4)))
            fz = np.array([r["fz"] for r in cy])
            w("  load left / right, first fifth -> last fifth: %.1f / %.1f -> %.1f / %.1f N"
              % (fz[:n5][:, list(LEFT)].sum(axis=1).mean(), fz[:n5][:, list(RIGHT)].sum(axis=1).mean(),
                 fz[-n5:][:, list(LEFT)].sum(axis=1).mean(), fz[-n5:][:, list(RIGHT)].sum(axis=1).mean()))
        # loads, the estimator, torque
        w("  fz mean per leg (N): %s;  left %.1f right %.1f;  f_w y mean per leg (N): %s"
          % (" ".join("%s %.1f" % (LEGS[i], self.fz_mean[i]) for i in range(4)),
             self.fz_mean[list(LEFT)].sum(), self.fz_mean[list(RIGHT)].sum(),
             " ".join("%+.2f" % v for v in self.f_w_mean[:, 1])))
        w("  estimator: p_est drift x %+.1f y %+.1f mm over the trot; v_est mean x %+.1f y %+.1f mm/s; "
          "x/y rows on %.0f %% of sweeps; their mean force x %+.2f y %+.2f N; commanded moment mean "
          "x %+.3f y %+.3f z %+.3f N m"
          % (self.est_drift[0], self.est_drift[1], 1e3 * self.v_est_mean[0],
             1e3 * self.v_est_mean[1], 100.0 * self.xy_on_frac, self.F_xy_mean[0],
             self.F_xy_mean[1], self.M_mean[0], self.M_mean[1], self.M_mean[2]))
        w("  torque per amp, |meas|/|cmd| on planted legs by joint (abd pitch knee): %s"
          % "  ".join("%s %s" % (LEGS[i], "/".join("%.2f" % v for v in self.kt[3 * i:3 * i + 3]))
                      for i in range(4)))
        w("  peak |tau_cmd| %.2f N m; slew deficit %.3f N m s; sweeps at the 9 N m clip %d"
          % (self.tau_peak, self.slew_deficit, self.clip_sweeps))
        return "\n".join(L)

    def tilt_rms_windows(self, win: float):
        out = []
        for a in np.arange(0.0, self.t[-1] + 1e-9, win):
            s = (self.t >= a) & (self.t < a + win)
            if s.sum() > 20:
                out.append(float(np.sqrt(np.mean(self.tilt[s] ** 2))))
        return out


# ===========================================================================
# figures
# ===========================================================================
def figures(run: Run, out_dir: str, zoom=None, cycles=None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    t = run.t
    name = run.name

    # -- 1. overview -----------------------------------------------------------
    fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(t, run.roll_e, lw=0.6, label="roll")
    ax[0].plot(t, run.pitch_e, lw=0.6, label="pitch")
    ax[0].plot(t, run.lean, lw=1.0, color="k", label="lean (1.5 Hz)")
    ax[0].axhline(0, color="0.7", lw=0.5)
    ax[0].set_ylabel("deg from setpoint")
    ax[0].legend(loc="upper right", ncol=3, fontsize=8)
    cy = run.cycles
    if cy:
        tc = [r["t"] for r in cy]
        ax[1].plot(tc, [r["roll"] for r in cy], "o-", ms=2, lw=0.8, label="roll, cycle mean")
        ax[1].plot(tc, [r["e_R"][0] for r in cy], "s-", ms=2, lw=0.8, label="e_R roll, cycle mean")
        ax[1].plot(tc, [r["M"][0] for r in cy], "^-", ms=2, lw=0.8, label="commanded roll moment, N m")
        ax[1].axhline(0, color="0.7", lw=0.5)
        ax[1].set_ylabel("deg / N m")
        ax[1].legend(loc="upper right", ncol=3, fontsize=8)
    ax[2].plot(t, run.h, lw=0.6, label="height")
    ax[2].plot(t, run.h_cmd, lw=0.6, color="k", label="commanded")
    ax[2].set_ylabel("mm")
    ax[2].legend(loc="upper right", fontsize=8)
    ok = np.isfinite(run.p_est[:, 0])
    ax[3].plot(t[ok], 1e3 * (run.p_est[ok, 0] - run.p_est[ok][0, 0]), lw=0.6, label="p_est x")
    ax[3].plot(t[ok], 1e3 * (run.p_est[ok, 1] - run.p_est[ok][0, 1]), lw=0.6, label="p_est y")
    okd = np.isfinite(run.xy_des[:, 0])
    if okd.any():
        ax[3].plot(t[okd], 1e3 * (run.xy_des[okd, 1] - run.p_est[ok][0, 1]), lw=0.6,
                   color="0.5", label="xy_des y")
    ax[3].set_ylabel("mm (leg odometry)")
    ax[3].legend(loc="upper right", ncol=3, fontsize=8)
    ax[4].plot(t, run.b_d[:, 0], lw=0.5, label="x/y rows' force x")
    ax[4].plot(t, run.b_d[:, 1], lw=0.5, label="force y")
    ax[4].plot(t, 1e3 * run.v_est[:, 1], lw=0.5, alpha=0.6, label="v_est y, mm/s")
    ax[4].axhline(0, color="0.7", lw=0.5)
    ax[4].set_ylabel("N / mm s^-1")
    ax[4].set_xlabel("s into the trot")
    ax[4].legend(loc="upper right", ncol=3, fontsize=8)
    for a in ax:
        if run.t_fall is not None:
            a.axvline(run.t_fall, color="r", lw=1.0)
        if run.t_tip is not None:
            a.axvline(run.t_tip, color="r", lw=0.8, ls="--")
    fig.suptitle("%s: overview" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_1_overview.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 2. the anatomy: a zoom with the clock, the heights and the forces ------
    if zoom is None:
        t_end = run.t_fall if run.t_fall is not None else run.t_tip
        if t_end is not None:
            zoom = (max(0.0, t_end - 2.4), min(t[-1], t_end + 0.4))
        else:
            mid = 0.5 * t[-1]
            zoom = (mid, mid + 3.0 * (run.period if np.isfinite(run.period) else 0.8))
    s = (t >= zoom[0]) & (t <= zoom[1])
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(t[s], run.roll_e[s], lw=1.0, label="roll")
    ax[0].plot(t[s], run.pitch_e[s], lw=1.0, label="pitch")
    ax[0].axhline(0, color="0.7", lw=0.5)
    ax[0].set_ylabel("deg from setpoint")
    ax[0].legend(loc="upper right", fontsize=8)
    colors = ("C0", "C1", "C2", "C3")
    for i, leg in enumerate(LEGS):
        ax[1].plot(t[s], run.z_clear[s, i], lw=1.0, color=colors[i], label=leg)
        sw = ~run.contact[s, i]
        ax[1].fill_between(t[s], -2, 30, where=sw, color=colors[i], alpha=0.08)
        ax[2].plot(t[s], run.cw[s, i], lw=1.0, color=colors[i])
        ax[3].plot(t[s], run.fz[s, i], lw=1.0, color=colors[i], label=leg)
    ax[1].axhline(CLEAR_MM, color="k", lw=0.5, ls="--")
    ax[1].set_ylabel("foot height above the planted plane, mm")
    ax[1].set_ylim(-3, 30)
    ax[1].legend(loc="upper right", ncol=4, fontsize=8)
    ax[2].set_ylabel("clock's contact weight")
    ax[3].set_ylabel("fz asked, N")
    ax[3].set_xlabel("s into the trot")
    ax[3].legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("%s: the clock against the feet, %.1f-%.1f s (shaded: the clock's swing)"
                 % (name, zoom[0], zoom[1]))
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_2_anatomy.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 3. swings and landings ---------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for i, leg in enumerate(LEGS):
        sw = [r for r in run.swings if r["leg"] == i]
        ld = [r for r in run.landings if r["leg"] == i]
        if sw:
            ax[0, 0].plot([r["t_lo"] for r in sw], [r["apex"] for r in sw], ".", ms=3,
                          color=colors[i], label=leg)
        if ld:
            ax[0, 1].hist([r["h_td"] for r in ld], bins=30, histtype="step",
                          color=colors[i], label=leg)
            ax[0, 2].plot([r["phi"] for r in ld], [r["dphi_toward"] for r in ld], ".",
                          ms=4, color=colors[i], label=leg)
            ax[1, 0].plot([r["t"] for r in ld], [r["dphi_toward"] for r in ld], ".",
                          ms=3, color=colors[i], label=leg)
        if sw:
            ax[1, 1].plot([r["t_lo"] for r in sw], [r["dy_floor"] for r in sw], ".", ms=3,
                          color=colors[i], label=leg)
            ax[1, 2].plot([r["t_lo"] for r in sw], [r["rms"] for r in sw], ".", ms=3,
                          color=colors[i], label=leg)
    ax[0, 0].axhline(CLEAR_MM, color="k", lw=0.5, ls="--")
    ax[0, 0].set_title("apex reached per swing, mm")
    ax[0, 0].set_xlabel("s")
    ax[0, 0].legend(fontsize=8, ncol=4)
    ax[0, 1].axvline(CLEAR_MM, color="k", lw=0.5, ls="--")
    ax[0, 1].set_title("h_td: foot height when the clock lands it, mm")
    ax[0, 1].legend(fontsize=8)
    ax[0, 2].set_title("roll step toward the landing foot vs Phi (N s)")
    ax[0, 2].set_xlabel("Phi, N s")
    ax[0, 2].set_ylabel("deg in 80 ms")
    ax[0, 2].axhline(0, color="0.7", lw=0.5)
    ax[1, 0].set_title("roll step toward the landing foot, per landing")
    ax[1, 0].set_xlabel("s")
    ax[1, 0].axhline(0, color="0.7", lw=0.5)
    ax[1, 1].set_title("the ratchet: landing y minus liftoff y over the floor, mm")
    ax[1, 1].set_xlabel("s")
    ax[1, 1].axhline(0, color="0.7", lw=0.5)
    ax[1, 2].set_title("swing tracking error rms, mm")
    ax[1, 2].set_xlabel("s")
    fig.suptitle("%s: swings and landings" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_3_swings.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 4. per cycle: the slow part ----------------------------------------------
    if cy:
        tc = np.array([r["t"] for r in cy])
        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        ax[0, 0].plot(tc, [r["width_front"] for r in cy], ".-", ms=2, lw=0.6, label="front")
        ax[0, 0].plot(tc, [r["width_rear"] for r in cy], ".-", ms=2, lw=0.6, label="rear")
        ax[0, 0].set_title("stance width, mm (levelled)")
        ax[0, 0].legend(fontsize=8)
        offy = np.array([r["off_y"] for r in cy])
        offx = np.array([r["off_x"] for r in cy])
        for i, leg in enumerate(LEGS):
            ax[0, 1].plot(tc, offy[:, i], ".-", ms=2, lw=0.6, color=colors[i], label=leg)
            ax[0, 2].plot(tc, offx[:, i], ".-", ms=2, lw=0.6, color=colors[i], label=leg)
            ax[1, 2].plot(tc, [r["fz"][i] for r in cy], ".-", ms=2, lw=0.6, color=colors[i], label=leg)
        ax[0, 1].set_title("foot y offset from the HOLD site, mm")
        ax[0, 1].legend(fontsize=8, ncol=4)
        ax[0, 2].set_title("foot x offset from the HOLD site, mm")
        eR = np.array([r["e_R"] for r in cy])
        M = np.array([r["M"] for r in cy])
        ax[1, 0].plot(tc, eR[:, 0], ".-", ms=2, lw=0.6, label="e_R roll, deg")
        ax[1, 0].plot(tc, eR[:, 1], ".-", ms=2, lw=0.6, label="e_R pitch, deg")
        ax[1, 0].plot(tc, 10.0 * M[:, 0], ".-", ms=2, lw=0.6, label="roll moment x10, N m")
        ax[1, 0].axhline(0, color="0.7", lw=0.5)
        ax[1, 0].set_title("attitude bias per cycle")
        ax[1, 0].legend(fontsize=8)
        v = np.array([r["v_est"] for r in cy])
        F = np.array([r["F"] for r in cy])
        ax[1, 1].plot(tc, 1e3 * v[:, 0], ".-", ms=2, lw=0.6, label="v_est x, mm/s")
        ax[1, 1].plot(tc, 1e3 * v[:, 1], ".-", ms=2, lw=0.6, label="v_est y, mm/s")
        ax[1, 1].plot(tc, 10.0 * F[:, 1], ".-", ms=2, lw=0.6, label="x/y rows' force y x10, N")
        ax[1, 1].axhline(0, color="0.7", lw=0.5)
        ax[1, 1].set_title("the estimator's velocity and the force it draws")
        ax[1, 1].legend(fontsize=8)
        ax[1, 2].set_title("fz per leg, cycle mean, N")
        ax[1, 2].legend(fontsize=8, ncol=4)
        for a in ax.flat:
            a.set_xlabel("s into the trot")
        fig.suptitle("%s: per cycle" % name)
        fig.tight_layout()
        p = os.path.join(out_dir, "%s_4_cycles.png" % name)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    # -- 5. the support diagonal ---------------------------------------------------
    tf = run.two_foot
    if tf:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        s2 = np.isfinite(run.proj_cmd)
        ax[0].plot(t[s2], run.proj_cmd[s2], ".", ms=1.5, label="commanded moment on the line")
        ax[0].plot(t[s2], run.proj_res[s2], ".", ms=1.5, label="delivered on the line")
        ax[0].plot(t[s2], BCFG.WEIGHT * run.d_off[s2], ".", ms=1.5, label="W d, the weight's toppling moment")
        ax[0].set_title("two-foot sweeps: moment about the support line, N m")
        ax[0].set_xlabel("s")
        ax[0].legend(fontsize=8)
        for pair in sorted(set(r["pair"] for r in tf)):
            rr = [r for r in tf if r["pair"] == pair]
            ax[1].plot([r["d_mm"] for r in rr], [r["excursion"] for r in rr], ".", ms=4, label=pair)
        ax[1].axhline(0, color="0.7", lw=0.5)
        ax[1].axvline(0, color="0.7", lw=0.5)
        ax[1].set_title("roll excursion over the two-foot phase vs CoM offset from the line")
        ax[1].set_xlabel("offset at the phase's start, mm (+ left of the line)")
        ax[1].set_ylabel("deg")
        ax[1].legend(fontsize=8)
        ax[2].plot([r["toppling"] for r in tf], [r["delivered"] for r in tf], ".", ms=4)
        lim = max(1e-3, np.nanmax(np.abs([r["toppling"] for r in tf] + [r["delivered"] for r in tf])))
        ax[2].plot([-lim, lim], [-lim, lim], color="0.6", lw=0.5)
        ax[2].set_xlabel("W d, N m")
        ax[2].set_ylabel("delivered on the line, N m")
        ax[2].set_title("what two feet deliver about their line is the toppling moment")
        fig.suptitle("%s: the support diagonal" % name)
        fig.tight_layout()
        p = os.path.join(out_dir, "%s_5_diagonal.png" % name)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    # -- 7. the foot's height at the clock's instants, per gait cycle ----------------
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    for i, leg in enumerate(LEGS):
        ld = [r for r in run.landings if r["leg"] == i]
        sw = [r for r in run.swings if r["leg"] == i]
        if ld:
            ax[i].plot(range(1, len(ld) + 1), [r["h_td"] for r in ld], "o-", ms=3, lw=0.7,
                       color=colors[i], label="h_td: height when the clock lands it")
        if sw:
            ax[i].plot(range(1, len(sw) + 1), [r["apex"] for r in sw], ".", ms=3,
                       color="0.5", label="apex reached in the swing")
            ax[i].plot(range(1, len(sw) + 1),
                       [run.z_clear[r["i_lo"], i] for r in sw], "x", ms=3,
                       color="0.3", label="height when the clock lifts it")
        ax[i].axhline(CLEAR_MM, color="k", lw=0.5, ls="--")
        ax[i].axhline(0, color="0.8", lw=0.5)
        ax[i].set_ylabel("%s  mm" % leg)
        if i == 0:
            ax[i].legend(fontsize=8, ncol=3, loc="upper right")
    ax[-1].set_xlabel("gait cycle since the trot started")
    fig.suptitle("%s: each foot's height above the planted plane at the clock's liftoff and "
                 "touchdown, per cycle (the dashed line is the 4 mm gate)" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_7_htd_per_cycle.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 8. every cycle on the leg's own clock: height against gait phase -------------
    per_ms = 1e3 * (run.period if np.isfinite(run.period) else 0.8)
    sw_ms = 1e3 * run.swing_s_median
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, leg in enumerate(LEGS):
        lo = run.lo[i]
        ncyc = max(1, lo.size - 1)
        for k in range(lo.size - 1):
            a, b = int(lo[k]), int(lo[k + 1])
            tt = 1e3 * (run.t[a:b] - run.t[a])
            ax[i].plot(tt, run.z_clear[a:b, i], lw=0.4, color=cmap(k / ncyc), alpha=0.7)
        ld = [r for r in run.landings if r["leg"] == i]
        if ld and lo.size:
            # each landing against its own cycle's liftoff
            xs, ys = [], []
            for r in ld:
                prev = lo[lo <= r["i_td"]]
                if prev.size:
                    xs.append(1e3 * (run.t[r["i_td"]] - run.t[prev[-1]]))
                    ys.append(r["h_td"])
            ax[i].plot(xs, ys, "k.", ms=3, label="h_td at the clock's touchdown")
        ax[i].axvline(0.0, color="k", lw=0.8, ls=":", label="clock: liftoff")
        ax[i].axvline(sw_ms, color="k", lw=0.8, ls="--", label="clock: touchdown")
        ax[i].axhline(CLEAR_MM, color="0.5", lw=0.5, ls="--")
        ax[i].set_ylabel("%s  mm" % leg)
        ax[i].set_xlim(-5.0, per_ms)
        if i == 0:
            ax[i].legend(fontsize=8, ncol=3, loc="upper right")
    ax[-1].set_xlabel("ms after the clock's liftoff of that leg (one line per cycle, "
                      "dark = early in the trot, yellow = late)")
    ax[0].set_ylim(-3, 30)
    fig.suptitle("%s: foot height against the gait clock, every cycle overlaid" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_8_height_vs_phase.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 9. height against the gait count, the clock's swing and stance in colour ----
    cyc = t / (run.period if np.isfinite(run.period) else 0.8)
    c0, c1 = (0.0, cyc[-1]) if cycles is None else cycles
    s = (cyc >= c0) & (cyc <= c1)
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True, sharey=True)
    for i, leg in enumerate(LEGS):
        z = run.z_clear[:, i]
        stance = np.where(run.contact[:, i], z, np.nan)
        swing = np.where(~run.contact[:, i], z, np.nan)
        # one sample of overlap at each edge, so the line does not break
        edge = np.flatnonzero(run.contact[1:, i] != run.contact[:-1, i]) + 1
        swing_e, stance_e = swing.copy(), stance.copy()
        for k in edge:
            swing_e[k - 1:k + 1] = z[k - 1:k + 1]
        ax[i].plot(cyc[s], stance_e[s], lw=0.9, color="C0", label="clock: STANCE")
        ax[i].plot(cyc[s], swing_e[s], lw=0.9, color="C3", label="clock: SWING")
        ax[i].axhline(CLEAR_MM, color="k", lw=0.6, ls="--", label="4 mm gate")
        ax[i].axhline(0, color="0.8", lw=0.5)
        ax[i].set_ylabel("%s height, mm" % leg)
        ax[i].grid(axis="x", color="0.9", lw=0.5)
        if i == 0:
            ax[i].legend(fontsize=8, ncol=3, loc="upper right")
    ax[-1].set_xlabel("gait cycle since the trot started (t / period, period %.2f s)"
                      % run.period)
    ax[0].set_ylim(-3, 32)
    if run.t_tip is not None:
        for a in ax:
            a.axvline(run.t_tip / run.period, color="k", lw=0.8, ls=":")
    fig.suptitle("%s: each foot's FK height above the planted plane, by gait cycle; "
                 "red where the clock has the leg in swing, blue in stance" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_9_height_by_gait.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 10. the same on ONE axis: four legs, the clock's swing windows shaded -------
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    pairs = ((0, 3), (1, 2))                       # FL+RR, FR+RL: the trot's diagonals
    pair_col = ("#f3e6b0", "#e8cde0")
    ymax = 32.0
    for (i, j), col in zip(pairs, pair_col):
        sw = (~run.contact[:, i]) | (~run.contact[:, j])
        ax.fill_between(cyc[s], -3, ymax, where=sw[s], color=col, alpha=0.9, lw=0,
                        label="clock: %s+%s SWING" % (LEGS[i], LEGS[j]))
    for i, leg in enumerate(LEGS):
        ax.plot(cyc[s], run.z_ref[s, i], lw=1.2, ls="--", color=colors[i], alpha=0.7,
                label="%s commanded arc" % leg)
        ax.plot(cyc[s], run.z_clear[s, i], lw=0.9, color=colors[i], label="%s measured" % leg)
    ax.axhline(CLEAR_MM, color="k", lw=0.6, ls="--", label="4 mm gate")
    ax.axhline(0, color="0.6", lw=0.5)
    if run.t_tip is not None:
        ax.axvline(run.t_tip / run.period, color="k", lw=0.8, ls=":")
    ax.set_ylim(-3, ymax)
    ax.set_xlim(c0, c1)
    ax.set_xlabel("gait cycle since the trot started (t / period, period %.2f s)" % run.period)
    ax.set_ylabel("foot height above the planted plane, mm")
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.legend(fontsize=7, ncol=6, loc="upper right")
    ax.set_title("%s: each foot's height by gait cycle -- dashed: the arc the clock commands, "
                 "solid: the foot (FK); shaded = the clock has that diagonal in swing" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_10_height_one_axis.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 11. the same shading, roll / pitch / yaw on the axis ------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    rpy = (run.roll_e, run.pitch_e, run.yaw - run.yaw[0])
    lim = float(np.nanmax(np.abs(np.concatenate([r[s] for r in rpy])))) * 1.1 + 0.5
    for (i, j), col in zip(pairs, pair_col):
        sw = (~run.contact[:, i]) | (~run.contact[:, j])
        ax.fill_between(cyc[s], -lim, lim, where=sw[s], color=col, alpha=0.9, lw=0,
                        label="clock: %s+%s SWING" % (LEGS[i], LEGS[j]))
    for arr, lab, col in zip(rpy, ("roll (from setpoint)", "pitch (from setpoint)",
                                   "yaw (from the trot's start)"), ("C0", "C1", "C2")):
        ax.plot(cyc[s], arr[s], lw=0.9, color=col, label=lab)
    ax.axhline(0, color="0.6", lw=0.5)
    if run.t_tip is not None:
        ax.axvline(run.t_tip / run.period, color="k", lw=0.8, ls=":")
    ax.set_ylim(-lim, lim)
    ax.set_xlim(c0, c1)
    ax.set_xlabel("gait cycle since the trot started (t / period, period %.2f s)" % run.period)
    ax.set_ylabel("deg")
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.set_title("%s: roll, pitch, yaw by gait cycle; shaded = the clock has that diagonal "
                 "in swing (roll > 0: right side down, pitch > 0: nose down, yaw > 0: nose left)"
                 % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_11_rpy_one_axis.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # -- 6. torque -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["%s %s" % (LEGS[i], j) for i in range(4) for j in ("abd", "pitch", "knee")]
    ax[0].bar(range(12), run.kt, color=[colors[i // 3] for i in range(12)])
    ax[0].set_xticks(range(12))
    ax[0].set_xticklabels(labels, rotation=60, fontsize=7)
    ax[0].axhline(1.0, color="0.5", lw=0.5)
    ax[0].set_title("|tau_meas| / |tau_cmd| on planted legs")
    for i, leg in enumerate(LEGS):
        ax[1].plot(t, np.abs(run.tau_cmd[:, 3 * i + 2]), lw=0.4, color=colors[i], label=leg + " knee")
    ax[1].set_title("|tau_cmd| knee, N m")
    ax[1].set_xlabel("s")
    ax[1].legend(fontsize=8, ncol=4)
    fig.suptitle("%s: torque" % name)
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_6_torque.png" % name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="hw.stand --log npz files")
    ap.add_argument("--out", default=None, help="figure directory (default: beside each log)")
    ap.add_argument("--no-fig", action="store_true", help="the summary only")
    ap.add_argument("--zoom", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="the anatomy figure's window, s into the trot (default: "
                         "the fall's last 2 s, or three cycles mid-run)")
    ap.add_argument("--cycles", type=float, nargs=2, default=None, metavar=("C0", "C1"),
                    help="the height-by-gait figure's window, in gait cycles "
                         "(default: the whole trot)")
    ap.add_argument("--setpoint", type=float, nargs=2, default=(None, None),
                    metavar=("ROLL_DEG", "PITCH_DEG"),
                    help="the attitude setpoint the law held (default: config's "
                         "statics, %.2f / %.2f, the fold stands' fixed datum)"
                         % (BCFG.SETPOINT_ROLL_DEG, BCFG.SETPOINT_PITCH_DEG))
    args = ap.parse_args(argv)
    for path in args.logs:
        run = Run(load(path), tuple(args.setpoint))
        print(run.summary())
        if not args.no_fig:
            out = args.out or os.path.dirname(os.path.abspath(path))
            for p in figures(run, out, args.zoom, args.cycles):
                print("  fig: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
