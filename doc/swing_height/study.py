"""Swing height with the inertial feedforward OFF and ON, 20 mm apex: one leg,
hung up, the robot's swing code against MuJoCo through the hardware chain.

    D:\\mujoco\\.venv\\Scripts\\python.exe doc\\swing_height\\study.py            run, tables, figures
    D:\\mujoco\\.venv\\Scripts\\python.exe doc\\swing_height\\study.py --figures  tables and figures
                                                                         from the saved runs

Writes data/<case>.npz, results.txt (every number the report quotes),
tex/*.tex (its tables and macros) and fig/*.png (its figures, drawn the way
doc/gait_study/analyse.py draws the robot's: fig 7, 8, 9 and 10 there).

THE QUESTION.  The gait study (doc/gait_study, branch gait-study-data) flew
the walk's swing with the feedforward OFF and measured the swing foot on the
robot: apex ~20 mm, lifting 30-40 ms after the clock, and still 9-15 mm in
the air when the clock hands it the load (`h_td`).  What does `--swing-ff`
change at the same 20 mm, with everything else as flown?

THE METRICS are the study's, per swing, from the same columns (`analyse.Run.
_swings` / `_landings`): the swing window is the clock's, liftoff to
touchdown; heights are the foot's FK z above its resting site (the hold's
target, the arc's own zero -- in the air there is no floor plane).

    apex     max height in the window
    lag      clock liftoff -> first sample above 4 mm (`CLEAR_MM`)
    rms      |p_swing - x_b| over the window, 3-D
    h_td     height on the sweep the clock lands the foot

The first SKIP swings are not summarised (the hold's state, not the swing's).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "doc", "gait_study"))

import mujoco                                   # noqa: E402

from sim import leg_dynamics as LD              # noqa: E402
from sim import params as P                     # noqa: E402
from hw import kinematics as HK                 # noqa: E402
from hw.balance import swing as SWING           # noqa: E402
from hw.balance import swing_control as SC      # noqa: E402
from hw.balance import torque as TRQ            # noqa: E402

from doc.swing_height import hwsim_leg as HL    # noqa: E402

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "fig")
TEX = os.path.join(HERE, "tex")
LEG = HL.LEG
H_MM = 20.0
SWINGS = 40
SKIP = 2
CLEAR_MM = 4.0
#: kg m^2, A SENSITIVITY ONLY: the effective rotor term the 2026-09-25 bench
#: reading would imply taken at face value (the leg answered M0's torque with
#: ~1.7x M0's motion, swing.py).  The rotor term itself is the MG5010-i10
#: datasheet's, `params.ARMATURE` = 8.5e-5 x 10^2, the same motor on DOG5
#: and DOG6 -- what the model and M0 both use.
ARM_BENCH = 0.0085 / 1.7
ROBOT_LOGS = [os.path.join(REPO, "doc", "gait_study", "runs", "p1_lat_%d.npz" % k)
              for k in (1, 2, 3)]


# ===========================================================================
# the model, from MuJoCo
# ===========================================================================
def full_mass(m, d) -> np.ndarray:
    """M(q) after `mj_forward`: MuJoCo's CRBA, sparse in `qM`, made dense."""
    M = np.zeros((m.nv, m.nv))
    if hasattr(d, "qM"):
        mujoco.mj_fullM(m, M, d.qM)
    else:                                       # mujoco >= 3.10 signature
        mujoco.mj_fullM(m, d, M)
    return M


def leg_model():
    m = mujoco.MjModel.from_xml_path(HL.MODEL)
    return m, mujoco.MjData(m)


def mass_and_gravity(q):
    """``(M, g)`` of the weighed leg at `q`, from MuJoCo: M by CRBA, g as the
    bias force at zero velocity (RNE with qd = 0, qdd = 0)."""
    m, d = leg_model()
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return full_mass(m, d), d.qfrc_bias.copy()


def arc_dynamics(q_hold, T, H, n=400):
    """The arc through the IK and MuJoCo's inverse dynamics along it.

    Returns s, q, qd, qdd and the torque split tau = M qdd + c(q, qd) + g(q),
    each (n+1, 3), plus what the law feeds forward along the same arc: its
    M0 J+ (a - Jdot qd_ref) and its closed-form gravity."""
    m, d = leg_model()
    rest = HK.foot_position(LEG, q_hold)
    hip = np.asarray(P.HIP_OFFSET[LEG], float)
    s = np.linspace(0.0, 1.0, n + 1)
    q = np.zeros((n + 1, 3))
    ff = np.zeros((n + 1, 3))
    g_law = np.zeros((n + 1, 3))
    qs = q_hold.copy()
    for k, x in enumerate(s):
        p, v, a = SWING.swing_reference_pva(rest, float(x), T, height=H)
        qs = HK.leg_ik(LEG, p - hip, q_seed=qs)
        q[k] = qs
        ff[k] = SWING.swing_feedforward(LEG, qs, v, a)
        g_law[k] = TRQ.leg_gravity_torque(LEG, qs)
    dt = T / n
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    Mqdd = np.zeros((n + 1, 3))
    cqd = np.zeros((n + 1, 3))
    g = np.zeros((n + 1, 3))
    tau = np.zeros((n + 1, 3))
    for k in range(n + 1):
        d.qpos[:] = q[k]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        g[k] = d.qfrc_bias
        mujoco.mj_mulM(m, d, Mqdd[k], qdd[k])
        d.qvel[:] = qd[k]
        d.qacc[:] = qdd[k]
        mujoco.mj_inverse(m, d)
        tau[k] = d.qfrc_inverse
        cqd[k] = tau[k] - Mqdd[k] - g[k]
    return dict(s=s, q=q, qd=qd, qdd=qdd, tau=tau, Mqdd=Mqdd, cqd=cqd, g=g,
                ff=ff, g_law=g_law, T=T)


def dynamics_report(q_hold, out) -> dict:
    """M, g, Lambda at the hold, the armature's share, the feedforward check."""
    w = out.append
    M_mj, g_mj = mass_and_gravity(q_hold)
    M_law = LD.mass_matrix(LEG, q_hold, armature=True)
    M0 = SWING.JOINT_INERTIA_M0
    links_mj = M_mj - P.ARMATURE * np.eye(3)
    J = HK.foot_jacobian(LEG, q_hold)
    lam = lambda M: np.linalg.inv(J @ np.linalg.solve(M, J.T))     # noqa: E731
    L_mj, L_links = lam(M_mj), lam(links_mj)
    g_law = TRQ.leg_gravity_torque(LEG, q_hold)
    w("THE LEG AT THE STUDY'S HOLD (FOLD2, 160 mm): q = %s deg, foot at %s mm (trunk)"
      % (np.round(np.degrees(q_hold), 2), np.round(1e3 * HK.foot_position(LEG, q_hold), 1)))
    w("  M(q) from MuJoCo, weighed leg, armature in:\n%s" % np.array2string(M_mj, precision=5))
    w("  M(q) the law's RNEA, CAD masses, armature in:\n%s" % np.array2string(M_law, precision=5))
    w("  M0 the feedforward multiplies by (FL at the lift pose, CAD):\n%s"
      % np.array2string(M0, precision=5))
    share = P.ARMATURE / np.diag(M_mj)
    w("  armature share of diag M (abd/pitch/knee): %s" % np.round(share, 3))
    w("  g(q) MuJoCo weighed %s   law (CAD, closed form) %s N*m"
      % (np.round(g_mj, 4), np.round(g_law, 4)))
    w("  Lambda = (J M^-1 J^T)^-1 diag, x/y/z: %s kg  (links only, no armature: %s kg)"
      % (np.round(np.diag(L_mj), 3), np.round(np.diag(L_links), 3)))
    w("  J at the hold:\n%s" % np.array2string(J, precision=4))
    # the bench's feedforward against the walk's, along the arc
    rest = HK.foot_position(LEG, q_hold)
    hip = np.asarray(P.HIP_OFFSET[LEG], float)
    worst = 0.0
    qs = q_hold.copy()
    for x in np.linspace(0, 1, 401):
        p, v, a = SWING.swing_reference_pva(rest, float(x), 0.16, height=1e-3 * H_MM)
        qs = HK.leg_ik(LEG, p - hip, q_seed=qs)
        Js = HK.foot_jacobian(LEG, qs)
        worst = max(worst, float(np.abs(SWING.swing_feedforward(LEG, qs, v, a, jac=Js)
                                        - SC.inertial_feedforward(LEG, qs, Js, v, a)).max()))
    w("  swing.swing_feedforward (bench) vs swing_control.inertial_feedforward (walk), "
      "160 ms / 20 mm arc: max |diff| %.1e N*m" % worst)
    return dict(M_mj=M_mj, g_mj=g_mj, M_law=M_law, M0=M0, share=share, g_law=g_law,
                L_mj=L_mj, L_links=L_links, J=J, ff_equiv=worst)


def linear_model(lam, lam0, ff, T=0.16, H=1e-3 * H_MM, kp=400.0, kd=40.0,
                 dt=1e-5, t_end=0.8) -> dict:
    """The swing reduced to its z axis: a mass `lam` (the foot's apparent mass
    in z, Lambda_zz) on the Cartesian PD, plus `lam0` z''_ref if `ff` (the
    feedforward's own idea of that mass).  No delay, no slew, no x/y, no
    gravity error -- what is left when the chain and the coupling are taken
    out, to say what sets the height the foot reaches:

        Z / Z_ref = (ff lam0 s^2 + kd s + kp) / (lam s^2 + kd s + kp)."""
    z = zd = 0.0
    t = 0.0
    rows = []
    while t < t_end:
        if 0.0 <= t <= T:
            p, v, a = SWING.swing_reference_pva(np.zeros(3), t / T, T, height=H)
            zr, vr, ar = p[2], v[2], a[2]
        else:
            zr = vr = ar = 0.0
        zdd = (kp * (zr - z) + kd * (vr - zd) + (lam0 * ar if ff else 0.0)) / lam
        zd += zdd * dt
        z += zd * dt
        rows.append((t, z))
        t += dt
    o = np.array(rows)
    sw = o[:, 0] <= T
    up = np.flatnonzero(o[:, 1] > 1e-3 * CLEAR_MM)
    return dict(apex=1e3 * o[sw, 1].max(), s_apex=o[sw][np.argmax(o[sw, 1]), 0] / T,
                h_td=1e3 * o[np.argmin(np.abs(o[:, 0] - T)), 1],
                lag=1e3 * o[up[0], 0] if up.size else np.inf,
                wn=float(np.sqrt(kp / lam)), zeta=float(kd / (2.0 * np.sqrt(kp * lam))))


# ===========================================================================
# the runs
# ===========================================================================
def cases(q_hold):
    M_mj, _ = mass_and_gravity(q_hold)
    arm5 = HL.Chain(armature_scale=ARM_BENCH / P.ARMATURE)
    return {
        "off": ("FF off", HL.Setup(swing_ff=False), HL.Chain()),
        "on": ("FF on", HL.Setup(swing_ff=True), HL.Chain()),
        "off_ideal": ("FF off, ideal chain", HL.Setup(swing_ff=False), HL.Chain(ideal=True)),
        "on_ideal": ("FF on, ideal chain", HL.Setup(swing_ff=True), HL.Chain(ideal=True)),
        "on_mj": ("FF on, M0 = MuJoCo M(q_hold)", HL.Setup(swing_ff=True, ff_inertia=M_mj),
                  HL.Chain()),
        "off_arm": ("FF off, rotor 0.005", HL.Setup(swing_ff=False), arm5),
        "on_arm": ("FF on, rotor 0.005, M0 on 0.0085", HL.Setup(swing_ff=True), arm5),
        "on_arm_m": ("FF on, rotor 0.005, M0 on 0.005",
                     HL.Setup(swing_ff=True, ff_inertia=SWING.feedforward_inertia(ARM_BENCH)),
                     arm5),
        "off_damp": ("FF off, joint damping 0.1", HL.Setup(swing_ff=False),
                     HL.Chain(damping=0.1)),
        "on_damp": ("FF on, joint damping 0.1", HL.Setup(swing_ff=True),
                    HL.Chain(damping=0.1)),
    }


def run_all(q_hold) -> dict:
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for key, (label, setup, chain) in cases(q_hold).items():
        res = HL.run_bench(setup, chain, swings=SWINGS)
        if res["stop"]:
            print("  %s stopped: %s" % (key, res["stop"]))
        res["label"] = label
        HL.save(res, os.path.join(DATA, key + ".npz"))
        out[key] = res
        print("  ran %-10s %s" % (key, label))
    return out


def load_all(q_hold) -> dict:
    """The saved runs, in `cases` order."""
    return {k: HL.load(os.path.join(DATA, k + ".npz")) for k in cases(q_hold)
            if os.path.exists(os.path.join(DATA, k + ".npz"))}


# ===========================================================================
# the metrics: analyse.Run._swings / _landings, on the bench's columns
# ===========================================================================
def heights(res):
    rest = np.asarray(res["rest"], float)
    z = 1e3 * (res["x_b"][:, 2] - rest[2])
    z_true = 1e3 * (res["x_true"][:, 2] - rest[2])
    z_ref = 1e3 * (res["p_swing"][:, 2] - rest[2])
    return z, z_true, z_ref


def swings(res) -> list:
    t = res["t"]
    c = res["contact"] > 0.5
    lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
    td = np.flatnonzero(~c[:-1] & c[1:]) + 1
    z, z_true, z_ref = heights(res)
    rest = np.asarray(res["rest"], float)
    T = float(res["swing_duration"])
    rows = []
    for a in lo:
        b = td[td > a]
        if not b.size:
            continue
        b = int(b[0])
        zc = z[a:b]
        k = int(np.argmax(zc))
        up = np.flatnonzero(zc > CLEAR_MM)
        err = res["p_swing"][a:b] - res["x_b"][a:b]
        after = slice(b, min(len(t), b + int(round(0.15 / np.median(np.diff(t))))))
        req, cmd = res["tau_req"][a:b], res["tau_cmd"][a:b]
        rows.append(dict(
            i_lo=int(a), i_td=b, t_lo=float(t[a]),
            apex=float(zc.max()), s_apex=float((t[a + k] - t[a]) / T),
            apex_true=float(z_true[a:b].max()),
            lag=1e3 * float(t[a + up[0]] - t[a]) if up.size else np.inf,
            rms=1e3 * float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
            rms_z=1e3 * float(np.sqrt(np.mean(err[:, 2] ** 2))),
            h_td=float(z[b]), h_td_true=float(z_true[b]),
            vz_td=float((z_true[b] - z_true[b - 1]) / (t[b] - t[b - 1]) * 1e-3),
            x_dev=1e3 * float(np.abs(res["x_b"][a:b, 0] - rest[0]).max()),
            under=float(z_true[after].min()),
            tau_pk=float(np.abs(cmd).max()), ff_pk=float(np.abs(res["tau_ff"][a:b]).max()),
            pd_pk=float(np.abs(res["tau_pd"][a:b]).max()),
            clip=float(np.abs(req - cmd).max())))
    return rows


KEYS = ("apex", "s_apex", "lag", "rms", "rms_z", "h_td", "vz_td", "x_dev", "under",
        "tau_pk", "ff_pk", "pd_pk", "clip")


def summary(rows, skip=SKIP) -> dict:
    rows = rows[skip:]
    out = {"n": len(rows)}
    for k in KEYS:
        v = np.array([r[k] for r in rows if k in r], float)
        v = v[np.isfinite(v)]
        out[k] = (float(np.median(v)), float(np.percentile(v, 10)),
                  float(np.percentile(v, 90))) if v.size else (np.nan,) * 3
    return out


# ===========================================================================
# the robot: the study's own logs, RR, through the study's own analyser
# ===========================================================================
def robot_rr() -> dict:
    """RR's swings in the 2026-10-21 lateral runs (FF off), before any tip,
    through the study's own analyser (`analyse.Run`).

    HEIGHTS FROM THE ARC'S OWN START, NOT THE FLOOR PLANE.  On the robot the
    clearance is read above the plane of the planted feet, whose datum the
    study corrects by a per-leg offset (RR's is -4.3 mm), and the walk's arc
    starts at the commanded stance height, 1-3 mm above that plane: the
    commanded RR arc peaks at 22-23.5 mm in the study's figure 10, and the
    foot sits at 1-4 mm when the clock lifts it -- above the 4 mm gate on
    some swings, so the study's absolute `lag` reads 0 there.  On the bench
    the arc starts at the resting site, which is zero.  So both are read
    from z_ref at the clock's liftoff: apex, h_td and the 4 mm lift gate.
    The study's absolute numbers are kept beside them (`summary_abs`)."""
    import analyse as GA
    segs, rows, htd, refs, cyc = [], [], [], [], []
    period = sw = None
    for path in ROBOT_LOGS:
        if not os.path.exists(path):
            continue
        run = GA.Run(GA.load(path))
        period, sw = run.period, run.swing_s_median
        t_end = (run.t_tip - 1.0) if run.t_tip is not None else run.t[-1]
        lo = run.lo[LEG]
        for r in run.swings:
            if r["leg"] != LEG or r["t_td"] >= t_end:
                continue
            a, b = r["i_lo"], r["i_td"]
            z0 = float(run.z_ref[a, LEG])
            zc = run.z_clear[a:b, LEG] - z0
            up = np.flatnonzero(zc > CLEAR_MM)
            nxt = lo[lo > a]
            if nxt.size and run.t[nxt[0]] <= t_end:
                c1 = int(nxt[0])
                segs.append((1e3 * (run.t[a:c1] - run.t[a]), run.z_clear[a:c1, LEG] - z0))
                refs.append((1e3 * (run.t[a:c1] - run.t[a]), run.z_ref[a:c1, LEG] - z0))
            htd.append((1e3 * (run.t[b] - run.t[a]), float(run.z_clear[b, LEG] - z0)))
            cyc.append(dict(apex=float(zc.max()),
                            lag=1e3 * float(run.t[a + up[0]] - run.t[a]) if up.size else np.inf,
                            rms=r["rms"], h_td=float(run.z_clear[b, LEG] - z0),
                            apex_abs=r["apex"], h_td_abs=float(run.z_clear[b, LEG]),
                            z_lo=float(run.z_clear[a, LEG]), ref_pk=float(np.nanmax(
                                run.z_ref[a:b, LEG]) - z0)))
            rows.append(r)
    if not cyc:
        return {}

    def stat(key):
        v = np.array([c[key] for c in cyc], float)
        v = v[np.isfinite(v)]
        return (float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)))
    summ = {"n": len(cyc)}
    for k in ("apex", "lag", "rms", "h_td", "ref_pk", "z_lo"):
        summ[k] = stat(k)
    summ_abs = {k: stat(k + "_abs") for k in ("apex", "h_td")}
    return dict(segs=segs, refs=refs, htd=htd, rows=cyc, summary=summ,
                summary_abs=summ_abs, period=period, swing=sw)


# ===========================================================================
# tables
# ===========================================================================
def _fmt(x, nd=1):
    m, lo, hi = x
    return "%.*f [%.*f, %.*f]" % (nd, m, nd, lo, nd, hi)


def tables(runs, dyn, robot, out) -> dict:
    w = out.append
    S = {k: summary(swings(r)) for k, r in runs.items()}
    w("")
    w("PER SWING, median [p10, p90] over swings %d-%d (study metrics; heights above the "
      "resting site, mm)" % (SKIP + 1, SWINGS))
    for k, r in runs.items():
        s = S[k]
        w("  %-36s n %2d  apex %s at s=%.2f  lag %s ms  rms %s  h_td %s  vz_td %+.2f m/s  "
          "x %s  under %+.1f  tau_pk %.2f  ff %.2f  pd %.2f  clip %.2f"
          % (r["label"], s["n"], _fmt(s["apex"]), s["s_apex"][0], _fmt(s["lag"], 0),
             _fmt(s["rms"]), _fmt(s["h_td"]), s["vz_td"][0], _fmt(s["x_dev"]),
             s["under"][0], s["tau_pk"][0], s["ff_pk"][0], s["pd_pk"][0], s["clip"][0]))
    if robot:
        rs, ra = robot["summary"], robot["summary_abs"]
        w("  %-36s n %2d  apex %s  lag %s ms  rms %s  h_td %s   (period %.2f s, swing %.0f ms)"
          % ("ROBOT, FF off (p1_lat_1..3, RR)", rs["n"], _fmt(rs["apex"]), _fmt(rs["lag"], 0),
             _fmt(rs["rms"]), _fmt(rs["h_td"]), robot["period"], 1e3 * robot["swing"]))
        w("  %-36s heights above the arc's start (z_ref at the clock's liftoff); the commanded "
          "arc peaks %s above it; the foot sits %s above the floor plane at liftoff; on the "
          "study's absolute scale apex %s, h_td %s"
          % ("", _fmt(rs["ref_pk"]), _fmt(rs["z_lo"]), _fmt(ra["apex"]), _fmt(ra["h_td"])))
    _tex(runs, S, dyn, robot)
    return S


def _row(cells):
    return " & ".join(_signed(c) for c in cells) + r" \\"


def _signed(text: str) -> str:
    """Every number's sign as a math sign, not a hyphen: '-1.9 [-3.8, 0.1]'
    -> '$-$1.9 [$-$3.8, 0.1]'.  Signs inside math are left alone."""
    if "$" in text:
        return text
    return re.sub(r"(?<![\w.])([+-])(?=\d)", r"$\1$", text)


#: the case labels as LaTeX (the runs' own carry underscores)
TEX_LABEL = {
    "off": "FF off", "on": "FF on",
    "off_ideal": "FF off, ideal chain", "on_ideal": "FF on, ideal chain",
    "on_mj": r"FF on, $\mathbf{M}_0 = \mathbf{M}_{\mathrm{MuJoCo}}(\mathbf{q}_{hold})$",
    "off_arm": "FF off, rotor 0.005",
    "on_arm": r"FF on, rotor 0.005, $\mathbf{M}_0$ on 0.0085",
    "on_arm_m": r"FF on, rotor 0.005, $\mathbf{M}_0$ on 0.005",
    "off_damp": "FF off, joint damping 0.1", "on_damp": "FF on, joint damping 0.1",
}


def _tex(runs, S, dyn, robot):
    os.makedirs(TEX, exist_ok=True)
    m = lambda k, q: S[k][q][0]                    # noqa: E731

    def line(k, label=None):
        s = S[k]
        return _row([label or TEX_LABEL[k], "%.1f" % s["apex"][0],
                     "%.2f" % s["s_apex"][0], "%.0f" % s["lag"][0], "%.1f" % s["rms"][0],
                     "%.1f" % s["h_td"][0], "%+.2f" % s["vz_td"][0], "%.1f" % s["x_dev"][0],
                     "%.2f" % s["tau_pk"][0], "%.2f" % s["clip"][0]])
    head = (r"\begin{tabular}{lrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & apex & $s_{apex}$ & lag & rms & $h_{td}$ & $v_{z,td}$ & $|x|$ & "
            r"$\hat\tau$ & clip \\" "\n"
            r" & mm & & ms & mm & mm & m/s & mm & N\,m & N\,m \\" "\n" r"\midrule")
    with open(os.path.join(TEX, "tab_results.tex"), "w") as fh:
        fh.write(head + "\n")
        fh.write(line("off", r"\textbf{FF off}") + "\n")
        fh.write(line("on", r"\textbf{FF on}") + "\n")
        fh.write(r"\midrule" "\n")
        fh.write(line("off_ideal", "FF off, ideal chain") + "\n")
        fh.write(line("on_ideal", "FF on, ideal chain") + "\n")
        fh.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
    with open(os.path.join(TEX, "tab_sens.tex"), "w") as fh:
        fh.write(head + "\n")
        for k in ("off", "on", "on_mj"):
            fh.write(line(k) + "\n")
        fh.write(r"\midrule" "\n")
        for k in ("off_arm", "on_arm", "on_arm_m"):
            fh.write(line(k) + "\n")
        fh.write(r"\midrule" "\n")
        for k in ("off_damp", "on_damp"):
            fh.write(line(k) + "\n")
        fh.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
    if robot:
        rs = robot["summary"]
        with open(os.path.join(TEX, "tab_robot.tex"), "w") as fh:
            fh.write(r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
                     r" & apex, mm & lag, ms & rms, mm & $h_{td}$, mm \\" "\n" r"\midrule" "\n")
            fh.write(_row(["Robot, FF off (%d swings)" % rs["n"], _fmt(rs["apex"]),
                           _fmt(rs["lag"], 0), _fmt(rs["rms"]), _fmt(rs["h_td"])]) + "\n")
            for k, lab in (("off", "Sim, FF off"), ("on", "Sim, FF on")):
                s = S[k]
                fh.write(_row([lab, _fmt(s["apex"]), _fmt(s["lag"], 0), _fmt(s["rms"]),
                               _fmt(s["h_td"])]) + "\n")
            fh.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
    M_mj, M_law, M0 = dyn["M_mj"], dyn["M_law"], dyn["M0"]

    def mat(M):
        return r"\begin{bmatrix}" + r"\\".join(
            " & ".join("%.4f" % v for v in row) for row in M) + r"\end{bmatrix}"
    mac = {
        "ApexOff": "%.1f" % m("off", "apex"), "ApexOn": "%.1f" % m("on", "apex"),
        "LagOff": "%.0f" % m("off", "lag"), "LagOn": "%.0f" % m("on", "lag"),
        "RmsOff": "%.1f" % m("off", "rms"), "RmsOn": "%.1f" % m("on", "rms"),
        "HtdOff": "%.1f" % m("off", "h_td"), "HtdOn": "%.1f" % m("on", "h_td"),
        "SapexOff": "%.2f" % m("off", "s_apex"), "SapexOn": "%.2f" % m("on", "s_apex"),
        "VtdOff": "%+.2f" % m("off", "vz_td"), "VtdOn": "%+.2f" % m("on", "vz_td"),
        "XOff": "%.1f" % m("off", "x_dev"), "XOn": "%.1f" % m("on", "x_dev"),
        "TauOff": "%.2f" % m("off", "tau_pk"), "TauOn": "%.2f" % m("on", "tau_pk"),
        "FfOn": "%.2f" % m("on", "ff_pk"), "ClipOn": "%.2f" % m("on", "clip"),
        "PdOff": "%.2f" % m("off", "pd_pk"), "PdOn": "%.2f" % m("on", "pd_pk"),
        "ApexOnIdeal": "%.1f" % m("on_ideal", "apex"), "RmsOnIdeal": "%.1f" % m("on_ideal", "rms"),
        "ApexOffIdeal": "%.1f" % m("off_ideal", "apex"), "HtdOnIdeal": "%.1f" % m("on_ideal", "h_td"),
        "ApexArmOff": "%.1f" % m("off_arm", "apex"), "ApexArmOn": "%.1f" % m("on_arm", "apex"),
        "ApexArmMatch": "%.1f" % m("on_arm_m", "apex"), "HtdArmOn": "%.1f" % m("on_arm", "h_td"),
        "RmsArmMatch": "%.1f" % m("on_arm_m", "rms"), "HtdArmMatch": "%.1f" % m("on_arm_m", "h_td"),
        "RmsArmOn": "%.1f" % m("on_arm", "rms"), "HtdArmOff": "%.1f" % m("off_arm", "h_td"),
        "UnderOn": "%.1f" % m("on", "under"), "UnderArmOn": "%.1f" % m("on_arm", "under"),
        "MMj": mat(M_mj), "MLaw": mat(M_law), "MZero": mat(M0),
        "GMj": ", ".join("%.3f" % v for v in dyn["g_mj"]),
        "GLaw": ", ".join("%.3f" % v for v in dyn["g_law"]),
        "LamZ": "%.2f" % dyn["L_mj"][2, 2], "LamX": "%.2f" % dyn["L_mj"][0, 0],
        "LamY": "%.2f" % dyn["L_mj"][1, 1], "LamZLinks": "%.2f" % dyn["L_links"][2, 2],
        "SharePitch": "%.0f" % (100 * dyn["share"][1]), "ShareKnee": "%.0f" % (100 * dyn["share"][2]),
        "ShareAbd": "%.0f" % (100 * dyn["share"][0]),
        "FfEquiv": r"\ensuremath{%d\times10^{%d}}" % _mant_exp(max(dyn["ff_equiv"], 1e-16)),
        "UnderArmOnAbs": "%.1f" % abs(m("on_arm", "under")),
        "HtdArmOnAbs": "%.1f" % abs(m("on_arm", "h_td")),
        "LinHtdArmOnAbs": "%.1f" % abs(dyn["lin"]["on_arm"]["h_td"]),
        "XArmOn": "%.0f" % m("on_arm", "x_dev"),
        "LamZFive": "%.2f" % dyn["lz5"],
        "Wn": "%.1f" % dyn["lin"]["off"]["wn"], "WnHz": "%.2f" % (dyn["lin"]["off"]["wn"] / 2 / np.pi),
        "Zeta": "%.2f" % dyn["lin"]["off"]["zeta"],
        "LinApexOff": "%.1f" % dyn["lin"]["off"]["apex"], "LinHtdOff": "%.1f" % dyn["lin"]["off"]["h_td"],
        "LinLagOff": "%.0f" % dyn["lin"]["off"]["lag"], "LinSapexOff": "%.2f" % dyn["lin"]["off"]["s_apex"],
        "LinApexOn": "%.1f" % dyn["lin"]["on"]["apex"], "LinHtdOn": "%.1f" % abs(dyn["lin"]["on"]["h_td"]),
        "LinApexArmOff": "%.1f" % dyn["lin"]["off_arm"]["apex"],
        "LinApexArmOn": "%.1f" % dyn["lin"]["on_arm"]["apex"],
        "LinHtdArmOn": "%.1f" % dyn["lin"]["on_arm"]["h_td"],
    }
    if robot:
        rs, ra = robot["summary"], robot["summary_abs"]
        mac.update(RobApex="%.1f" % rs["apex"][0], RobLag="%.0f" % rs["lag"][0],
                   RobRms="%.1f" % rs["rms"][0], RobHtd="%.1f" % rs["h_td"][0],
                   RobN="%d" % rs["n"], RobRefPk="%.1f" % rs["ref_pk"][0],
                   RobZlo="%.1f" % rs["z_lo"][0], RobApexAbs="%.1f" % ra["apex"][0],
                   RobHtdAbs="%.1f" % ra["h_td"][0])
    with open(os.path.join(TEX, "macros.tex"), "w") as fh:
        fh.write("%% GENERATED by doc/swing_height/study.py\n")
        for k, v in mac.items():
            if v[:1] in "+-" and v[1:2].isdigit():
                v = r"\ensuremath{%s}" % v       # a minus, not a hyphen, in text
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))


def _mant_exp(x: float):
    e = int(np.floor(np.log10(x)))
    return int(round(x / 10.0 ** e)), e


# ===========================================================================
# figures: doc/gait_study/analyse.py's, for one leg on the bench
# ===========================================================================
RR = "C3"                                       # analyse.py's colour for RR
PAIR_COL = "#f3e6b0"                            # its FL+RR swing shading


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _cycles(res):
    T = float(res["period"])
    return (res["t"] - float(res["t_swing0"])) / T


def fig_one_axis(runs, keys, path, c0=10.0, c1=16.0):
    """analyse.py figure 10: the foot by gait cycle, the commanded arc dashed,
    the clock's swing shaded -- one panel per case."""
    plt = _plt()
    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 3.6 * len(keys)), sharex=True,
                             sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        r = runs[k]
        cyc = _cycles(r)
        s = (cyc >= c0) & (cyc <= c1)
        z, _, z_ref = heights(r)
        sw = r["contact"] < 0.5
        ax.fill_between(cyc[s], -12, 40, where=sw[s], color=PAIR_COL, alpha=0.9, lw=0,
                        label="clock: FL+RR SWING")
        ax.plot(cyc[s], z_ref[s], lw=1.2, ls="--", color=RR, alpha=0.7, label="RR commanded arc")
        ax.plot(cyc[s], z[s], lw=1.1, color=RR, label="RR measured (FK of the encoders)")
        ax.axhline(CLEAR_MM, color="k", lw=0.6, ls="--", label="4 mm gate")
        ax.axhline(H_MM, color="0.5", lw=0.5, ls=":", label="apex asked, 20 mm")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_ylim(-6, 28)
        ax.set_xlim(c0, c1)
        ax.set_ylabel("foot height above its rest site, mm")
        ax.grid(axis="x", color="0.9", lw=0.5)
        ax.legend(fontsize=7, ncol=5, loc="upper right")
        ax.set_title("%s: RR foot height by gait cycle -- dashed: the arc the clock commands, "
                     "solid: the foot (FK); shaded = the clock has the leg in swing" % r["label"])
    np.atleast_1d(axes)[-1].set_xlabel("gait cycle since the swing started (t / period, "
                                       "period %.2f s)" % float(runs[keys[0]]["period"]))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_vs_phase(runs, keys, robot, path):
    """analyse.py figure 8: every cycle on the leg's own clock, overlaid."""
    plt = _plt()
    n = len(keys) + (1 if robot else 0)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.3 * n), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    period = float(runs[keys[0]]["period"])
    T = float(runs[keys[0]]["swing_duration"])
    for ax, k in zip(axes, keys):
        r = runs[k]
        c = r["contact"] > 0.5
        lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
        z, _, z_ref = heights(r)
        rows = swings(r)
        ncyc = max(1, lo.size - 1)
        for j in range(SKIP, lo.size - 1):
            a, b = int(lo[j]), int(lo[j + 1])
            ax.plot(1e3 * (r["t"][a:b] - r["t"][a]), z[a:b], lw=0.5, alpha=0.7,
                    color=cmap(j / ncyc))
        a, b = int(lo[SKIP]), int(lo[SKIP + 1])
        ax.plot(1e3 * (r["t"][a:b] - r["t"][a]), z_ref[a:b], lw=1.4, ls="--", color="k",
                alpha=0.6, label="commanded arc")
        ax.plot([1e3 * (r["t"][x["i_td"]] - r["t"][x["i_lo"]]) for x in rows[SKIP:]],
                [x["h_td"] for x in rows[SKIP:]], "k.", ms=4, label="h_td at the clock's touchdown")
        ax.set_title("%s (sim, hung-up leg): apex %.1f mm, lag %.0f ms, h_td %.1f mm (medians)"
                     % (r["label"], *[summary(rows)[q][0] for q in ("apex", "lag", "h_td")]),
                     fontsize=10, loc="left")
    if robot:
        ax = axes[-1]
        ns = max(1, len(robot["segs"]))
        for j, (tt, zz) in enumerate(robot["segs"]):
            ax.plot(tt, zz, lw=0.5, alpha=0.7, color=cmap(j / ns))
        n = min(len(s[1]) for s in robot["refs"])
        ax.plot(robot["refs"][0][0][:n], np.nanmedian([s[1][:n] for s in robot["refs"]], axis=0),
                lw=1.4, ls="--", color="k", alpha=0.6, label="commanded arc (median)")
        ax.plot([x for x, _ in robot["htd"]], [h for _, h in robot["htd"]], "k.", ms=4,
                label="h_td at the clock's touchdown")
        rs = robot["summary"]
        ax.set_title("ROBOT, FF off (gait study p1_lat_1..3, RR, on the floor, %d swings; "
                     "heights from the arc's start): apex %.1f mm, lag %.0f ms, h_td %.1f mm"
                     % (rs["n"], rs["apex"][0], rs["lag"][0], rs["h_td"][0]),
                     fontsize=10, loc="left")
    for ax in axes:
        ax.axvline(0.0, color="k", lw=0.8, ls=":", label="clock: liftoff")
        ax.axvline(1e3 * T, color="k", lw=0.8, ls="--", label="clock: touchdown")
        ax.axhline(CLEAR_MM, color="0.5", lw=0.5, ls="--")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_ylabel("RR  mm")
        ax.set_xlim(-5.0, 1e3 * period)
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[0].set_ylim(-8, 30)
    axes[-1].set_xlabel("ms after the clock's liftoff of that leg (one line per cycle, "
                        "dark = early, yellow = late)")
    fig.suptitle("RR foot height against the gait clock, every cycle overlaid")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_by_gait(runs, keys, path, c0=8.0, c1=16.0):
    """analyse.py figure 9: red where the clock has the leg in swing."""
    plt = _plt()
    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 3.2 * len(keys)), sharex=True,
                             sharey=True)
    for ax, k in zip(axes, keys):
        r = runs[k]
        cyc = _cycles(r)
        s = (cyc >= c0) & (cyc <= c1)
        z, _, _ = heights(r)
        c = r["contact"] > 0.5
        stance = np.where(c, z, np.nan)
        swing = np.where(~c, z, np.nan)
        for e in np.flatnonzero(c[1:] != c[:-1]) + 1:
            swing[e - 1:e + 1] = z[e - 1:e + 1]
        ax.plot(cyc[s], stance[s], lw=0.9, color="C0", label="clock: STANCE")
        ax.plot(cyc[s], swing[s], lw=0.9, color="C3", label="clock: SWING")
        ax.axhline(CLEAR_MM, color="k", lw=0.6, ls="--", label="4 mm gate")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_ylabel("%s\nRR height, mm" % r["label"])
        ax.grid(axis="x", color="0.9", lw=0.5)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[0].set_ylim(-6, 28)
    axes[-1].set_xlabel("gait cycle since the swing started (t / period, period %.2f s)"
                        % float(runs[keys[0]]["period"]))
    fig.suptitle("RR foot's FK height above its rest site, by gait cycle; red where the "
                 "clock has the leg in swing, blue in stance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_per_cycle(runs, keys, robot, path):
    """analyse.py figure 7: the clock's instants, per cycle."""
    plt = _plt()
    n = len(keys) + (1 if robot else 0)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharey=True)
    for ax, k in zip(axes, keys):
        rows = swings(runs[k])
        i = np.arange(1, len(rows) + 1)
        z, _, _ = heights(runs[k])
        ax.plot(i, [x["h_td"] for x in rows], "o-", ms=3, lw=0.7, color=RR,
                label="h_td: height when the clock lands it")
        ax.plot(i, [x["apex"] for x in rows], ".", ms=4, color="0.5",
                label="apex reached in the swing")
        ax.plot(i, [z[x["i_lo"]] for x in rows], "x", ms=3, color="0.3",
                label="height when the clock lifts it")
        ax.set_ylabel("%s\nRR  mm" % runs[k]["label"])
        ax.set_xlabel("swing count")
    if robot:
        ax = axes[-1]
        hs = [h for _, h in robot["htd"]]
        ax.plot(np.arange(1, len(hs) + 1), hs, "o-", ms=3, lw=0.7, color=RR,
                label="h_td: height when the clock lands it")
        ax.plot(np.arange(1, len(robot["rows"]) + 1), [x["apex"] for x in robot["rows"]],
                ".", ms=4, color="0.5", label="apex reached in the swing")
        ax.set_ylabel("ROBOT, FF off\nRR  mm")
        ax.set_xlabel("swing count (p1_lat_1..3 back to back, before any tip)")
    for ax in axes:
        ax.axhline(CLEAR_MM, color="k", lw=0.5, ls="--")
        ax.axhline(H_MM, color="0.6", lw=0.5, ls=":")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[0].set_ylim(-6, 30)
    fig.suptitle("RR foot height at the clock's liftoff and touchdown, and the apex, per swing "
                 "(dashed: the 4 mm gate, dotted: the 20 mm apex)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _aligned(res, col, j, pre=4, post=60):
    """(n_swings, pre+post) of `col`[:, j] from `pre` sweeps before each liftoff."""
    c = res["contact"] > 0.5
    lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
    lo = lo[SKIP:]
    lo = lo[lo + post < len(c)]
    return np.array([res[col][a - pre:a + post, j] for a in lo])


def fig_torque(runs, keys, path):
    """The torque split through a swing: what the law asked, what the slew let
    through, the feedforward and the PD (analyse.py figures 15 and 20)."""
    plt = _plt()
    fig, axes = plt.subplots(2, len(keys), figsize=(13, 7), sharex=True, sharey="row")
    sweep = float(runs[keys[0]]["chain"]["sweep_s"]) if isinstance(runs[keys[0]]["chain"], dict) \
        else HL.Chain().sweep_s
    T = float(runs[keys[0]]["swing_duration"])
    pre, post = 4, 60
    tt = 1e3 * sweep * (np.arange(pre + post) - pre)
    for col, k in enumerate(keys):
        r = runs[k]
        for row, (j, name) in enumerate(((1, "pitch"), (2, "knee"))):
            ax = axes[row, col]
            cmd = _aligned(r, "tau_cmd", j, pre, post)
            for y in cmd:
                ax.plot(tt, y, color="C1", lw=0.4, alpha=0.35)
            ax.plot(tt, np.median(cmd, axis=0), color="C1", lw=1.8, label="sent (after the slew)")
            ax.plot(tt, np.median(_aligned(r, "tau_req", j, pre, post), axis=0), color="C0",
                    lw=1.4, label="requested")
            ax.plot(tt, np.median(_aligned(r, "tau_ff", j, pre, post), axis=0), color="C2",
                    lw=1.2, ls="--", label="feedforward $M_0\\ddot q_{ref}$")
            ax.plot(tt, np.median(_aligned(r, "tau_pd", j, pre, post), axis=0), color="C4",
                    lw=1.2, ls="-.", label="Cartesian PD $J^T F$")
            ax.axvline(0, color="k", lw=0.8, ls=":")
            ax.axvline(1e3 * T, color="k", lw=0.8, ls="--")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_title("%s -- %s" % (r["label"], name), loc="left", fontsize=10)
            if col == 0:
                ax.set_ylabel("%s torque, N m" % name)
            if row == 1:
                ax.set_xlabel("ms after the clock's liftoff (dotted) -- dashed: its touchdown")
    axes[0, -1].legend(fontsize=8, loc="upper right")
    fig.suptitle("RR torque through a swing, every swing (thin) and the median (thick): "
                 "requested = gravity + PD (+ FF); sent = after the 300 N m/s slew")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_error(runs, keys, path):
    """analyse.py figure 18: the swing's tracking error, x and z, over the swing."""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    sweep = HL.Chain().sweep_s
    T = float(runs[keys[0]]["swing_duration"])
    pre, post = 2, 40
    tt = 1e3 * sweep * (np.arange(pre + post) - pre)
    cols = ("C0", "C3", "C2", "C1")
    for k, colr in zip(keys, cols):
        r = runs[k]
        for ax, j, nm in ((axes[0], 2, "z"), (axes[1], 0, "x")):
            ref = r["p_swing"][:, j].copy()
            rest = np.asarray(r["rest"], float)[j]
            ref[~np.isfinite(ref)] = rest
            e = 1e3 * (ref - r["x_b"][:, j])
            c = r["contact"] > 0.5
            lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
            lo = lo[SKIP:]
            lo = lo[lo + post < len(c)]
            E = np.array([e[a - pre:a + post] for a in lo])
            for y in E:
                ax.plot(tt, y, color=colr, lw=0.3, alpha=0.3)
            ax.plot(tt, np.median(E, axis=0), color=colr, lw=1.8, label=r["label"])
            ax.set_ylabel("%s error, ref - foot, mm" % nm)
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axvline(1e3 * T, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel("ms after the clock's liftoff (dotted) -- dashed: its touchdown")
    fig.suptitle("RR swing tracking error, every swing (thin) and the median (thick)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_dynamics(arc, path):
    """M(q), g(q) and the inverse dynamics along the 20 mm arc, from MuJoCo,
    against what the law feeds forward."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    t = 1e3 * arc["s"] * arc["T"]
    m, d = leg_model()
    Md = []
    for q in arc["q"]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        Md.append(full_mass(m, d))
    Md = np.array(Md)
    ax = axes[0]
    for (i, j), cc, nm in (((0, 0), "C0", "$M_{11}$ abd"), ((1, 1), "C1", "$M_{22}$ pitch"),
                           ((2, 2), "C3", "$M_{33}$ knee"), ((1, 2), "C2", "$M_{23}$")):
        ax.plot(t, 1e3 * Md[:, i, j], color=cc, lw=1.6, label=nm + " MuJoCo")
        ax.axhline(1e3 * SWING.JOINT_INERTIA_M0[i, j], color=cc, lw=0.9, ls="--")
    ax.axhline(1e3 * P.ARMATURE, color="k", lw=0.6, ls=":", label="armature 0.0085")
    ax.set_xlabel("ms into the swing")
    ax.set_ylabel("$10^{-3}$ kg m$^2$")
    ax.set_title("M(q) along the arc (solid, MuJoCo, weighed leg)\nvs the law's constant "
                 "$M_0$ (dashed)", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2)
    for ax, j, nm in ((axes[1], 1, "pitch"), (axes[2], 2, "knee")):
        ax.plot(t, arc["tau"][:, j], color="k", lw=2.0, label="inverse dynamics (MuJoCo)")
        ax.plot(t, arc["Mqdd"][:, j], color="C0", lw=1.3, label="$M(q)\\ddot q$")
        ax.plot(t, arc["g"][:, j], color="C2", lw=1.3, label="$g(q)$")
        ax.plot(t, arc["cqd"][:, j], color="C4", lw=1.3, label="$c(q,\\dot q)$")
        ax.plot(t, arc["ff"][:, j] + arc["g_law"][:, j], color="C3", lw=1.3, ls="--",
                label="the law: $M_0\\ddot q_{ref}$ + its $g$")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel("ms into the swing")
        ax.set_ylabel("%s torque, N m" % nm)
        ax.set_title("what the 160 ms / 20 mm arc asks of the %s" % nm, fontsize=10, loc="left")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _median_curve(res, post=None):
    z, _, _ = heights(res)
    c = res["contact"] > 0.5
    lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
    n = int(np.min(np.diff(lo)))
    lo = lo[SKIP:-1]
    Z = np.array([z[a:a + n] for a in lo])
    tt = 1e3 * (res["t"][lo[0]:lo[0] + n] - res["t"][lo[0]])
    return tt, np.median(Z, axis=0)


def fig_sensitivity(runs, robot, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    T = float(runs["off"]["swing_duration"])
    _, _, z_ref = heights(runs["off"])
    c = runs["off"]["contact"] > 0.5
    lo = np.flatnonzero(c[:-1] & ~c[1:]) + 1
    a = int(lo[SKIP])
    b = int(np.flatnonzero(c[a:])[0]) + a
    ax.plot(1e3 * (runs["off"]["t"][a:b] - runs["off"]["t"][a]), z_ref[a:b], color="k",
            lw=1.6, ls="--", label="commanded arc")
    for k, cc, ls in (("off", "C0", "-"), ("on", "C3", "-"), ("off_arm", "C0", ":"),
                      ("on_arm", "C1", ":"), ("on_arm_m", "C3", ":")):
        tt, zz = _median_curve(runs[k])
        ax.plot(tt, zz, color=cc, lw=1.8, ls=ls, label="sim: " + runs[k]["label"])
    if robot:
        n = min(len(s[1]) for s in robot["segs"])
        Z = np.array([s[1][:n] for s in robot["segs"]])
        ax.plot(robot["segs"][0][0][:n], np.median(Z, axis=0), color="0.35", lw=2.2,
                label="ROBOT, FF off (median of %d swings)" % len(robot["segs"]))
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.axvline(1e3 * T, color="k", lw=0.8, ls="--")
    ax.axhline(CLEAR_MM, color="0.5", lw=0.5, ls="--")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlim(-5, 400)
    ax.set_xlabel("ms after the clock's liftoff (dotted) -- dashed: its touchdown")
    ax.set_ylabel("RR foot height, mm (median over swings)")
    ax.set_title("The armature is the feedforward's gain: rotor term 0.0085 (the datasheet, "
                 "solid) vs a sensitivity at 0.005 (dotted)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_model(q_hold, path):
    """The robot at the study's hold, the RR leg's joint axes drawn."""
    plt = _plt()
    from sim import coordinates as C
    from hw.balance import config as BCFG
    m = mujoco.MjModel.from_xml_path(P.XML_PATH)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1200)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 900)
    d = mujoco.MjData(m)
    q4 = HL.fold2_pose(0.160)
    d.qpos[:] = C.qpos_from_q(q4, root_pos=(0.0, 0.0, 0.160 + BCFG.TRUNK_BOTTOM_OFFSET))
    mujoco.mj_forward(m, d)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    imgs = []
    with mujoco.Renderer(m, 600, 800) as r:
        # azimuth 90 looks along +y: from the robot's right, RR's swing plane
        for az, el, dist, look in ((215, -18, 0.75, (-0.05, 0.0, 0.10)),
                                   (90, -4, 0.40, (-0.20, -0.07, 0.07))):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, cam, scene_option=opt)
            imgs.append(r.render())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, im, tt in zip(axes, imgs, ("DOG6 at the study's hold (FOLD2, 160 mm)",
                                       "RR from the robot's right: the swing plane (trunk x-z);\n"
                                       "cones = joint axes (abd along x, pitch and knee along y here)")):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(tt, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figures(runs, robot, arc, q_hold):
    os.makedirs(FIG, exist_ok=True)
    main = ["off", "on"]
    fig_one_axis(runs, main, os.path.join(FIG, "height_one_axis.png"))
    fig_vs_phase(runs, main, robot, os.path.join(FIG, "height_vs_phase.png"))
    fig_by_gait(runs, main, os.path.join(FIG, "height_by_gait.png"))
    fig_per_cycle(runs, main, robot, os.path.join(FIG, "htd_per_cycle.png"))
    fig_torque(runs, main, os.path.join(FIG, "torque_split.png"))
    fig_error(runs, ["off", "on", "on_ideal"], os.path.join(FIG, "swing_error.png"))
    fig_dynamics(arc, os.path.join(FIG, "arc_dynamics.png"))
    fig_sensitivity(runs, robot, os.path.join(FIG, "sensitivity.png"))
    try:
        fig_model(q_hold, os.path.join(FIG, "model.png"))
    except Exception as failure:                # no GL context: the figure only
        print("  model render skipped: %s" % failure)
    print("figures ->", FIG)


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures", action="store_true",
                    help="tables and figures from data/ without re-running")
    args = ap.parse_args(argv)
    q_hold = HL.fold2_pose(0.160)[LEG]
    out = []
    dyn = dynamics_report(q_hold, out)
    arc = arc_dynamics(q_hold, 0.16, 1e-3 * H_MM)
    rate = np.abs(np.gradient(arc["ff"], arc["T"] / (len(arc["s"]) - 1), axis=0)).max(axis=0)
    out.append("  along the arc: peak |M qdd| (MuJoCo) %s, peak |g| %s, peak |c qd| %s N*m; "
               "the law's feedforward peak %s N*m, its peak rate %s N*m/s against the 300 slew"
               % (np.round(np.abs(arc["Mqdd"]).max(axis=0), 2), np.round(np.abs(arc["g"]).max(axis=0), 2),
                  np.round(np.abs(arc["cqd"]).max(axis=0), 3), np.round(np.abs(arc["ff"]).max(axis=0), 2),
                  np.round(rate, 0)))
    lz = dyn["L_mj"][2, 2]
    m_arm = mass_and_gravity(q_hold)[0] - (P.ARMATURE - ARM_BENCH) * np.eye(3)
    J = dyn["J"]
    lz5 = np.linalg.inv(J @ np.linalg.solve(m_arm, J.T))[2, 2]
    lin = {"off": linear_model(lz, lz, False), "on": linear_model(lz, lz, True),
           "off_arm": linear_model(lz5, lz, False), "on_arm": linear_model(lz5, lz, True)}
    dyn.update(lin=lin, lz5=lz5)
    out.append("THE z AXIS ALONE (linear_model): Lambda_z %.2f kg (rotor 0.005: %.2f), Kp 400, "
               "Kd 40 -> wn %.2f rad/s (%.2f Hz), zeta %.2f" % (lz, lz5, lin["off"]["wn"],
                                                              lin["off"]["wn"] / 2 / np.pi,
                                                              lin["off"]["zeta"]))
    for k, lab in (("off", "FF off"), ("on", "FF on, lam0 = lam"),
                   ("off_arm", "FF off, rotor 0.005"), ("on_arm", "FF on, rotor 0.005, lam0 on 0.0085")):
        r = lin[k]
        out.append("  %-36s apex %.1f mm at s=%.2f  h_td %.1f mm  lag %.0f ms"
                   % (lab, r["apex"], r["s_apex"], r["h_td"], r["lag"]))
    print("\n".join(out))
    runs = load_all(q_hold) if args.figures else run_all(q_hold)
    robot = robot_rr()
    tables(runs, dyn, robot, out)
    print("\n".join(out[-(len(runs) + 3):]))
    with open(os.path.join(HERE, "results.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    figures(runs, robot, arc, q_hold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
