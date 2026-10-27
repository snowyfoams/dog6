"""The hung-up swing logs of 2026-10-27 (`hw.swing_ff`, round 1) against the
operator's three guesses: period, the feedforward at 2.0 s, the feedforward
at 0.8 s.  Figures to fig/, every number to results.txt.

    python doc/swing_hw/analyse.py                 reads doc/swing_height/runs (branch
                                                   swing-analysis), writes fig/ and results.txt
    python doc/swing_hw/analyse.py --runs DIR      the logs elsewhere

THE RUNS, all four legs, 12 swings a leg, FOLD2's hold at 160 mm, duty 0.8,
ramp 0.15, 20 mm arc, the walk's swing gains, the joint layer on the stance
legs, cap 9 N*m, slew 300 N*m/s (the data commit's table):

    b_off_T06_1    B1  0.6 s  FF off
    b_off_T08_1    B2  0.8 s  FF off
    b_off_T10_1    B3  1.0 s  FF off
    b_off_T14_1    B4  1.4 s  FF off
    b_off_T20_1    B5  2.0 s  FF off
    b_ff003_T20_1  B6  2.0 s  FF on, armature 0.003
    b_ff001_T08_1  B7  0.8 s  FF on, armature 0.001
    b_ff003_T08_1  B8  0.8 s  FF on, armature 0.003

THE METRICS, per swing, the swing window the clock's (liftoff to touchdown),
heights the foot's FK z above its RESTING SITE (the hold's target, the arc's
zero -- hung up there is no floor; on the floor the site IS the floor):

    apex      max height in the window, and s where it happens (the arc: 20 mm at 0.5)
    lag       clock liftoff -> first sweep above 4 mm
    rms       |p_swing - x_b| over the window, 3-D
    h_td      height on the sweep the clock lands the foot (the arc: 0)
    vz_td     the foot's vertical speed on that sweep
    t_arr     when the foot actually gets back below 2 mm, relative to the clock's
              touchdown: + late, - early.  ON THE FLOOR THIS IS WHEN IT HITS.
    vz_arr    its vertical speed at that moment -- the impact speed on the floor
    dip       the lowest the foot goes in the 150 ms after the clock's touchdown
              (negative: below the site -- on the floor, pushed INTO it)
    x, y      the largest excursion from the site, in the trunk's x and y
    tau       peak |tau| sent per leg; ff / pd the feedforward's and the PD's peaks;
              clip what the slew shaved; rate the peak |dtau/dt| sent
    jerk      median |d^3 z / dt^3| of the foot's FK z through the swing (smoothness)

The first two swings of every leg are skipped (the hold's state, not the
swing's), as doc/swing_height does: 40 swings a run are summarised.

THE EFFECTIVE INERTIA.  Over every swing window, per joint, least squares of
tau_meas - g (and tau_cmd - g) on (qdd, qd, 1): what the leg answered the
torque with.  The links' share is `leg_dynamics.mass_matrix` at the hold with
the armature out; the rest is the rotor term the feedforward should carry.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from sim import leg_dynamics as LD      # noqa: E402
from sim import params as P             # noqa: E402

RUNS = os.path.join(REPO, "doc", "swing_height", "runs")
FIG = os.path.join(HERE, "fig")
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("abd", "pitch", "knee")
SKIP = 2                      # swings per leg not summarised
CLEAR_MM = 4.0
ARRIVE_MM = 2.0
AFTER_S = 0.150

#: file stem, group, label, period, feedforward armature (None: off)
GROUPS = [
    ("b_off_T06_1", "B1", "FF off, 0.6 s", 0.6, None),
    ("b_off_T08_1", "B2", "FF off, 0.8 s", 0.8, None),
    ("b_off_T10_1", "B3", "FF off, 1.0 s", 1.0, None),
    ("b_off_T14_1", "B4", "FF off, 1.4 s", 1.4, None),
    ("b_off_T20_1", "B5", "FF off, 2.0 s", 2.0, None),
    ("b_ff003_T20_1", "B6", "FF on 0.003, 2.0 s", 2.0, 0.003),
    ("b_ff001_T08_1", "B7", "FF on 0.001, 0.8 s", 0.8, 0.001),
    ("b_ff003_T08_1", "B8", "FF on 0.003, 0.8 s", 0.8, 0.003),
]
COL = {"B1": "#9e9e9e", "B2": "#1f77b4", "B3": "#2ca02c", "B4": "#ff7f0e", "B5": "#d62728",
       "B6": "#9467bd", "B7": "#17becf", "B8": "#e377c2"}


# ===========================================================================
# the logs
# ===========================================================================
def load(stem: str, runs: str = RUNS) -> dict:
    L = np.load(os.path.join(runs, stem + ".npz"))
    n = len(L["t"])
    r = {k: L[k] for k in L.files}
    for k in ("tau_cmd", "tau_req", "tau_meas", "tau_ff", "tau_pd", "tau_g", "tau_hold",
              "q", "qd"):
        r[k] = r[k].reshape(n, 4, 3)
    r["stem"] = stem
    return r


def swings(r: dict) -> list:
    """Every swing in the log, with its metrics and its aligned traces."""
    t = r["t"]; x = r["x_b"]; rest = r["rest"]; sw = r["swing_s"]; p = r["p_swing"]
    dt_med = float(np.median(np.diff(t)))
    n_after = int(round(AFTER_S / dt_med))
    out = []
    for i in range(4):
        up = sw[:, i] > 0
        d = np.diff(up.astype(int))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        for a in starts:
            b = next((e for e in ends if e > a), None)
            if b is None:
                continue
            z = x[a:b, i, 2] - rest[i, 2]
            zr = p[a:b, i, 2] - rest[i, 2]
            err = np.linalg.norm(p[a:b, i] - x[a:b, i], axis=1)
            k = int(z.argmax())
            above = np.flatnonzero(z > 1e-3 * CLEAR_MM)
            lag = t[a + above[0]] - t[a] if len(above) else np.nan
            vz_td = (z[-1] - z[-2]) / (t[b - 1] - t[b - 2])
            # the arrival: after the apex, the first sweep back below ARRIVE_MM
            c = min(len(t), b + n_after)
            z_long = x[a:c, i, 2] - rest[i, 2]
            arr = np.flatnonzero(z_long[k:] < 1e-3 * ARRIVE_MM)
            if len(arr):
                j = k + arr[0]
                t_arr = t[a + j] - t[b]
                vz_arr = (z_long[j] - z_long[j - 1]) / (t[a + j] - t[a + j - 1])
            else:
                t_arr = vz_arr = np.nan
            dip = z_long[b - a:].min() if c > b else np.nan
            tau_cmd = r["tau_cmd"][a:b, i]
            rate = np.abs(np.diff(tau_cmd, axis=0)) / np.diff(t[a:b])[:, None]
            jerk = np.abs(np.diff(z, 3)) / dt_med ** 3
            out.append(dict(
                leg=i, a=a, b=b, t0=t[a], T=t[b - 1] - t[a],
                apex=z.max(), s_apex=float(sw[a + k, i]), apex_ref=zr.max(),
                lag=lag, rms=float(np.sqrt(np.mean(err ** 2))), h_td=z[-1], vz_td=vz_td,
                t_arr=t_arr, vz_arr=vz_arr, dip=dip, z0=z[0],
                x_dev=np.abs(x[a:b, i, 0] - rest[i, 0]).max(),
                y_dev=np.abs(x[a:b, i, 1] - rest[i, 1]).max(),
                tau_pk=np.abs(tau_cmd).max(), ff_pk=np.abs(r["tau_ff"][a:b, i]).max(),
                pd_pk=np.abs(r["tau_pd"][a:b, i]).max(),
                clip=np.abs(r["tau_req"][a:b, i] - tau_cmd).max(),
                meas_pk=np.abs(r["tau_meas"][a:b, i]).max(),
                rate_pk=rate.max(), rate_knee_p90=np.percentile(rate[:, 2], 90),
                jerk=float(np.median(jerk)) if len(jerk) else np.nan,
                tau_td=tau_cmd[-1].copy(),
            ))
    out.sort(key=lambda s: s["t0"])
    return out


def kept(rows: list) -> list:
    """Drop the first SKIP swings of every leg."""
    seen = {i: 0 for i in range(4)}
    out = []
    for s in rows:
        seen[s["leg"]] += 1
        if seen[s["leg"]] > SKIP:
            out.append(s)
    return out


def stat(rows, key):
    v = np.array([s[key] for s in rows], dtype=float)
    return (float(np.nanmedian(v)), float(np.nanpercentile(v, 10)),
            float(np.nanpercentile(v, 90)))


def aligned(r: dict, rows: list, col, leg_axis=None, pre_s=0.02, post_s=AFTER_S):
    """Traces on one grid from `pre_s` before the clock's liftoff to `post_s`
    after its touchdown, time in ms since liftoff.  `col(r, a, c, i)` gives the
    trace between sweeps a and c for leg i.  Returns (grid_ms, (n, m) array)."""
    t = r["t"]
    T = float(np.median([s["T"] for s in rows]))
    grid = np.arange(-pre_s, T + post_s, 0.004)
    traces = []
    cover = np.zeros(len(grid), dtype=bool)
    for s in rows:
        a, b, i = s["a"], s["b"], s["leg"]
        a0 = max(0, a - int(pre_s / 0.004) - 2)
        c = min(len(t), b + int(post_s / 0.004) + 2)
        tt = t[a0:c] - t[a]
        cover |= (grid >= tt[0]) & (grid <= tt[-1])
        traces.append(np.interp(grid, tt, col(r, a0, c, i), left=np.nan, right=np.nan))
    # the grid where at least one swing has samples -- the same mask whatever
    # the column, so traces of one run share a grid
    return 1e3 * grid[cover], np.array(traces)[:, cover]


def z_col(r, a, c, i):
    return 1e3 * (r["x_b"][a:c, i, 2] - r["rest"][i, 2])


def zref_col(r, a, c, i):
    z = r["p_swing"][a:c, i, 2] - r["rest"][i, 2]
    return 1e3 * np.where(np.isfinite(z), z, 0.0)


def err_col(r, a, c, i):
    e = np.linalg.norm(r["p_swing"][a:c, i] - r["x_b"][a:c, i], axis=1)
    return 1e3 * np.where(np.isfinite(e), e, np.nan)


def leg_sign(r, i, j):
    """The joint's sign on leg i relative to FL's: the hold's joint angles
    mirror between legs (q_pose), so pitch and knee torques do too."""
    return float(np.sign(r["q_pose"][i, j] * r["q_pose"][0, j])) or 1.0


def tau_col(name, j):
    return lambda r, a, c, i: leg_sign(r, i, j) * r[name][a:c, i, j]


# ===========================================================================
# the effective inertia
# ===========================================================================
def fit_inertia(r: dict, rows: list) -> dict:
    """Per joint (pitch, knee): tau - g = I qdd + b qd + c over the swing
    windows, pooled over the run's kept swings; from tau_meas and tau_cmd.
    qd is the log's (the runner's 40 Hz low-pass of the encoder difference,
    the velocity the law itself used), differenced once for qdd; the first
    and last three sweeps of each window are left out.
    The torque is allowed to LEAD the acceleration by 0-5 sweeps (the current
    loop and the chain: tau_meas itself follows tau_cmd by ~12 ms); the shift
    with the best R^2 is kept and reported."""
    t = r["t"]
    out = {}
    for src in ("tau_meas", "tau_cmd"):
        res = {}
        for j in (1, 2):
            best = None
            for shift in range(0, 6):
                X = []; Y = []
                for s in rows:
                    a, b, i = s["a"], s["b"], s["leg"]
                    if b - a < 12 + shift:
                        continue
                    tt = t[a:b + shift]
                    qd = r["qd"][a:b + shift, i]
                    qdd = np.gradient(qd, tt, axis=0)
                    y = r[src][a:b, i, j] - r["tau_g"][a:b, i, j]
                    k = np.arange(3, b - a - 3)             # the torque's sweeps
                    X.append(np.column_stack([qdd[k + shift, j], qd[k + shift, j],
                                              np.ones(len(k))]))
                    Y.append(y[k])
                A = np.vstack(X); yy = np.concatenate(Y)
                coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
                resid = yy - A @ coef
                r2 = 1.0 - resid.var() / max(yy.var(), 1e-12)
                if best is None or r2 > best["r2"]:
                    best = dict(I=float(coef[0]), b=float(coef[1]), c=float(coef[2]),
                                r2=float(r2), n=len(yy), A=A, y=yy, shift_ms=4.0 * shift)
            res[JOINTS[j]] = best
        out[src] = res
    return out


def links_diag(r: dict) -> np.ndarray:
    """The links' own inertia about each joint at the hold, armature out,
    averaged over the four legs (they differ only by mirroring)."""
    return np.mean([np.diag(LD.mass_matrix(i, r["q_pose"][i], armature=False))
                    for i in range(4)], axis=0)


def following(r: dict, rows: list) -> dict:
    """The driver after the command: the lag (ms) that best aligns tau_meas on
    tau_cmd for the knee over the swings, and the gain at that lag."""
    t = r["t"]
    dt = float(np.median(np.diff(t)))
    best = None
    for shift in range(0, 8):
        num = den = 0.0
        for s in rows:
            a, b, i = s["a"], s["b"], s["leg"]
            c = r["tau_cmd"][a:b - shift, i, 2]
            m = r["tau_meas"][a + shift:b, i, 2]
            num += float(np.dot(c, m)); den += float(np.dot(c, c))
        gain = num / max(den, 1e-12)
        err = 0.0
        for s in rows:
            a, b, i = s["a"], s["b"], s["leg"]
            c = r["tau_cmd"][a:b - shift, i, 2]
            m = r["tau_meas"][a + shift:b, i, 2]
            err += float(np.sum((m - gain * c) ** 2))
        if best is None or err < best[2]:
            best = (shift, gain, err)
    return dict(lag_ms=1e3 * best[0] * dt, gain=best[1])


# ===========================================================================
# the tables
# ===========================================================================
def _fmt3(v, scale=1.0, nd=1):
    return "%*.*f [%.*f, %.*f]" % (5, nd, scale * v[0], nd, scale * v[1], nd, scale * v[2])


def tables(runs: dict, out) -> None:
    def w(line=""):
        print(line); out.append(line)

    w("THE HUNG-UP SWING, 2026-10-27, ROUND 1: hw.swing_ff, all four legs, 12 swings a leg, "
      "FOLD2 at 160 mm, 20 mm arc, duty 0.8, ramp 0.15, Kp 150/150/400, Kd 6/6/40, "
      "cap 9 N*m, slew 300 N*m/s")
    w("per run, median [p10, p90] over the 40 swings after the first two of each leg; "
      "heights above the resting site, mm")
    w()
    w("  %-5s %-20s %4s %5s | %-18s %5s  %-17s %-5s  %-18s %6s  %-16s %6s  %6s  %5s"
      % ("group", "run", "T s", "swing", "apex", "s", "lag ms", "rms", "h_td", "vz_td",
         "t_arr ms", "vz_arr", "dip", "x mm"))
    for g in GROUPS:
        r = runs[g[0]]; rows = r["kept"]
        w("  %-5s %-20s %4.1f %4.0fms | %s  %.2f  %s  %4.1f  %s  %+5.2f  %s  %+5.2f  %+5.1f  %5.1f"
          % (g[1], g[2], g[3], 1e3 * stat(rows, "T")[0], _fmt3(stat(rows, "apex"), 1e3),
             stat(rows, "s_apex")[0], _fmt3(stat(rows, "lag"), 1e3, 0), 1e3 * stat(rows, "rms")[0],
             _fmt3(stat(rows, "h_td"), 1e3), stat(rows, "vz_td")[0],
             _fmt3(stat(rows, "t_arr"), 1e3, 0), stat(rows, "vz_arr")[0],
             1e3 * stat(rows, "dip")[0], 1e3 * stat(rows, "x_dev")[0]))
    w()
    w("  the torque, per run: peaks over a swing (median over swings), N*m; rate the peak |dtau/dt| "
      "sent, N*m/s; jerk the foot's median |z'''| through the swing, m/s^3")
    w("  %-5s %-20s %5s %5s %5s %5s %6s %6s %8s %6s   %s"
      % ("group", "run", "tau", "ff", "pd", "clip", "meas", "rate", "knee p90", "jerk",
         "driver: knee tau_meas on tau_cmd, lag / gain"))
    for g in GROUPS:
        r = runs[g[0]]; rows = r["kept"]; f = r["follow"]
        w("  %-5s %-20s %5.2f %5.2f %5.2f %5.2f %6.2f %6.0f %8.0f %6.0f   %.0f ms / %.2f"
          % (g[1], g[2], stat(rows, "tau_pk")[0], stat(rows, "ff_pk")[0], stat(rows, "pd_pk")[0],
             stat(rows, "clip")[0], stat(rows, "meas_pk")[0], stat(rows, "rate_pk")[0],
             stat(rows, "rate_knee_p90")[0], stat(rows, "jerk")[0], f["lag_ms"], f["gain"]))
    w()
    w("  per leg, apex / h_td / t_arr (mm, mm, ms):")
    for g in GROUPS:
        r = runs[g[0]]
        cells = []
        for i in range(4):
            mine = [s for s in r["kept"] if s["leg"] == i]
            cells.append("%s %4.1f / %+4.1f / %+3.0f" % (LEGS[i], 1e3 * stat(mine, "apex")[0],
                                                           1e3 * stat(mine, "h_td")[0],
                                                           1e3 * stat(mine, "t_arr")[0]))
        w("  %-5s %-20s  %s" % (g[1], g[2], "   ".join(cells)))
    w()
    w("THE EFFECTIVE INERTIA the leg answered with, tau - g = I qdd + b qd + c over the swing "
      "windows (kg m^2; R^2 of the fit); the links' own at the hold, armature out, pitch %.4f "
      "knee %.4f; the inherited rotor term %.4f; M0's diag with it pitch %.4f knee %.4f"
      % (runs[GROUPS[0][0]]["links"][1], runs[GROUPS[0][0]]["links"][2], P.ARMATURE,
         runs[GROUPS[0][0]]["links"][1] + P.ARMATURE, runs[GROUPS[0][0]]["links"][2] + P.ARMATURE))
    w("  %-5s %-20s | %-42s | %-42s | fit / M0 (cmd) pitch, knee"
      % ("group", "run", "from tau_meas: pitch, knee (R^2, torque lead)",
         "from tau_cmd: pitch, knee (R^2, torque lead)"))
    for g in GROUPS:
        r = runs[g[0]]; fi = r["fit"]; lk = r["links"]
        m, c = fi["tau_meas"], fi["tau_cmd"]
        w("  %-5s %-20s | %.4f (%.2f, %2.0f ms), %.4f (%.2f, %2.0f ms) | %.4f (%.2f, %2.0f ms), "
          "%.4f (%.2f, %2.0f ms) | %.2f, %.2f"
          % (g[1], g[2], m["pitch"]["I"], m["pitch"]["r2"], m["pitch"]["shift_ms"],
             m["knee"]["I"], m["knee"]["r2"], m["knee"]["shift_ms"],
             c["pitch"]["I"], c["pitch"]["r2"], c["pitch"]["shift_ms"],
             c["knee"]["I"], c["knee"]["r2"], c["knee"]["shift_ms"],
             c["pitch"]["I"] / (lk[1] + P.ARMATURE), c["knee"]["I"] / (lk[2] + P.ARMATURE)))
    w()
    # the guesses, in numbers
    off = {g[3]: runs[g[0]]["kept"] for g in GROUPS if g[4] is None}
    w("GUESS 1 -- period, FF off: apex %s mm, h_td %s mm, lag as a share of the swing %s %%, "
      "t_arr %s ms, vz_arr %s m/s for T = %s s"
      % (", ".join("%.1f" % (1e3 * stat(v, "apex")[0]) for v in off.values()),
         ", ".join("%+.1f" % (1e3 * stat(v, "h_td")[0]) for v in off.values()),
         ", ".join("%.0f" % (100 * stat(v, "lag")[0] / stat(v, "T")[0]) for v in off.values()),
         ", ".join("%+.0f" % (1e3 * stat(v, "t_arr")[0]) for v in off.values()),
         ", ".join("%+.2f" % stat(v, "vz_arr")[0] for v in off.values()),
         ", ".join("%.1f" % k for k in off)))
    b5, b6 = runs["b_off_T20_1"]["kept"], runs["b_ff003_T20_1"]["kept"]
    w("GUESS 2 -- 2.0 s, FF off -> on 0.003: apex %.1f -> %.1f mm (at s %.2f -> %.2f), lag %.0f -> "
      "%.0f ms, rms %.1f -> %.1f mm, h_td %+.1f -> %+.1f mm, t_arr %+.0f -> %+.0f ms, "
      "vz_arr %+.2f -> %+.2f m/s"
      % (1e3 * stat(b5, "apex")[0], 1e3 * stat(b6, "apex")[0], stat(b5, "s_apex")[0],
         stat(b6, "s_apex")[0], 1e3 * stat(b5, "lag")[0], 1e3 * stat(b6, "lag")[0],
         1e3 * stat(b5, "rms")[0], 1e3 * stat(b6, "rms")[0], 1e3 * stat(b5, "h_td")[0],
         1e3 * stat(b6, "h_td")[0], 1e3 * stat(b5, "t_arr")[0], 1e3 * stat(b6, "t_arr")[0],
         stat(b5, "vz_arr")[0], stat(b6, "vz_arr")[0]))
    b2, b7, b8 = (runs[k]["kept"] for k in ("b_off_T08_1", "b_ff001_T08_1", "b_ff003_T08_1"))
    w("GUESS 3 -- 0.8 s, FF off / 0.001 / 0.003: apex %.1f / %.1f / %.1f mm, rms %.1f / %.1f / %.1f mm, "
      "h_td %+.1f / %+.1f / %+.1f mm, t_arr %+.0f / %+.0f / %+.0f ms, vz_arr %+.2f / %+.2f / %+.2f m/s, "
      "dip %+.1f / %+.1f / %+.1f mm, x %.1f / %.1f / %.1f mm, rate %.0f / %.0f / %.0f N*m/s"
      % (1e3 * stat(b2, "apex")[0], 1e3 * stat(b7, "apex")[0], 1e3 * stat(b8, "apex")[0],
         1e3 * stat(b2, "rms")[0], 1e3 * stat(b7, "rms")[0], 1e3 * stat(b8, "rms")[0],
         1e3 * stat(b2, "h_td")[0], 1e3 * stat(b7, "h_td")[0], 1e3 * stat(b8, "h_td")[0],
         1e3 * stat(b2, "t_arr")[0], 1e3 * stat(b7, "t_arr")[0], 1e3 * stat(b8, "t_arr")[0],
         stat(b2, "vz_arr")[0], stat(b7, "vz_arr")[0], stat(b8, "vz_arr")[0],
         1e3 * stat(b2, "dip")[0], 1e3 * stat(b7, "dip")[0], 1e3 * stat(b8, "dip")[0],
         1e3 * stat(b2, "x_dev")[0], 1e3 * stat(b7, "x_dev")[0], 1e3 * stat(b8, "x_dev")[0],
         stat(b2, "rate_pk")[0], stat(b7, "rate_pk")[0], stat(b8, "rate_pk")[0]))


# ===========================================================================
# the figures
# ===========================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 120, "savefig.dpi": 150})
    return plt


def _band(ax, grid, tr, color, label, lw=1.8):
    med = np.nanmedian(tr, axis=0)
    lo, hi = np.nanpercentile(tr, 10, axis=0), np.nanpercentile(tr, 90, axis=0)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(grid, med, color=color, lw=lw, label=label)
    return med


def fig_period(runs, path):
    """Guess 1: the foot's height through the swing, one panel per period, FF off."""
    plt = _plt()
    groups = [g for g in GROUPS if g[4] is None]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.4), sharey=True)
    for ax, g in zip(axes, groups):
        r = runs[g[0]]; rows = r["kept"]
        grid, z = aligned(r, rows, z_col)
        _, zr = aligned(r, rows, zref_col)
        T = 1e3 * stat(rows, "T")[0]
        for tr in z:
            ax.plot(grid, tr, color=COL[g[1]], lw=0.5, alpha=0.25)
        ax.plot(grid, np.nanmedian(z, axis=0), color=COL[g[1]], lw=2.0, label="foot, median of 40")
        ax.plot(grid, np.nanmedian(zr, axis=0), "k--", lw=1.2, label="the arc")
        ax.axvline(0, color="k", lw=0.6); ax.axvline(T, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title("%s: swing %.0f ms" % (g[2], T))
        ax.set_xlabel("ms since the clock's liftoff")
        ax.text(0.02, 0.97, "apex %.1f mm at s=%.2f\nlag %.0f ms\nh_td %+.1f mm\narrives %+.0f ms"
                % (1e3 * stat(rows, "apex")[0], stat(rows, "s_apex")[0], 1e3 * stat(rows, "lag")[0],
                   1e3 * stat(rows, "h_td")[0], 1e3 * stat(rows, "t_arr")[0]),
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    axes[0].set_ylabel("foot height above its site, mm")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Guess 1 -- the period, FF off: the foot against the arc; vertical lines the "
                 "clock's liftoff and touchdown", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_metrics(runs, path):
    """Guess 1 in numbers: the metrics against the period, FF off, FF on beside."""
    plt = _plt()
    keys = [("apex", 1e3, "apex, mm (arc 20)"), ("lag", 1e3, "lag, ms"),
            ("rms", 1e3, "rms tracking error, mm"), ("h_td", 1e3, "h_td, mm (arc 0)"),
            ("t_arr", 1e3, "arrival after the clock, ms"), ("vz_arr", 1.0, "speed at arrival, m/s")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (k, sc, lab) in zip(axes.ravel(), keys):
        off = [g for g in GROUPS if g[4] is None]
        T = [g[3] for g in off]
        v = np.array([stat(runs[g[0]]["kept"], k) for g in off])
        ax.errorbar(T, sc * v[:, 0], yerr=[sc * (v[:, 0] - v[:, 1]), sc * (v[:, 2] - v[:, 0])],
                    fmt="o-", color="k", capsize=3, label="FF off")
        for g in GROUPS:
            if g[4] is None:
                continue
            s = stat(runs[g[0]]["kept"], k)
            ax.errorbar([g[3]], [sc * s[0]], yerr=[[sc * (s[0] - s[1])], [sc * (s[2] - s[0])]],
                        fmt="s", color=COL[g[1]], capsize=3, ms=7, label=g[2])
        if k == "apex":
            ax.axhline(20, color="k", ls="--", lw=0.8)
        if k in ("h_td", "t_arr"):
            ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("period, s"); ax.set_ylabel(lab)
        ax.set_xticks([0.6, 0.8, 1.0, 1.4, 2.0])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("The study's metrics against the period: FF off (black), FF on beside; "
                 "median with p10-p90", y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_ff(runs, stems, path, title):
    """Guesses 2 and 3: the runs in `stems` overlaid -- height, error, the knee torque split."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_z, ax_e, ax_k, ax_p = axes.ravel()
    for stem in stems:
        g = next(g for g in GROUPS if g[0] == stem)
        r = runs[stem]; rows = r["kept"]
        grid, z = aligned(r, rows, z_col)
        med = _band(ax_z, grid, z, COL[g[1]], g[2])
        _, e = aligned(r, rows, err_col)
        _band(ax_e, grid, e, COL[g[1]], g[2])
        for ax, j in ((ax_k, 2), (ax_p, 1)):
            _, c = aligned(r, rows, tau_col("tau_cmd", j))
            _, m = aligned(r, rows, tau_col("tau_meas", j))
            _, f = aligned(r, rows, tau_col("tau_ff", j))
            ax.plot(grid, np.nanmedian(c, axis=0), color=COL[g[1]], lw=1.8, label="%s: sent" % g[2])
            ax.plot(grid, np.nanmedian(m, axis=0), color=COL[g[1]], lw=1.0, ls=":",
                    label="%s: measured" % g[2])
            if g[4] is not None:
                ax.plot(grid, np.nanmedian(f, axis=0), color=COL[g[1]], lw=1.0, ls="--",
                        label="%s: tau_ff" % g[2])
    r0 = runs[stems[0]]
    grid, zr = aligned(r0, r0["kept"], zref_col)
    ax_z.plot(grid, np.nanmedian(zr, axis=0), "k--", lw=1.2, label="the arc")
    T = 1e3 * stat(r0["kept"], "T")[0]
    for ax in axes.ravel():
        ax.axvline(0, color="k", lw=0.6); ax.axvline(T, color="k", lw=0.6)
        ax.set_xlabel("ms since the clock's liftoff")
    ax_z.axhline(0, color="k", lw=0.4)
    ax_z.set_ylabel("foot height above its site, mm"); ax_z.legend(fontsize=7)
    ax_e.set_ylabel("|p_swing - foot|, mm"); ax_e.legend(fontsize=7)
    ax_k.set_ylabel("knee torque, N*m"); ax_k.legend(fontsize=6, ncol=2)
    ax_p.set_ylabel("pitch torque, N*m"); ax_p.legend(fontsize=6, ncol=2)
    ax_z.set_title("height, median and p10-p90"); ax_e.set_title("tracking error")
    ax_k.set_title("knee: sent, measured, feedforward"); ax_p.set_title("pitch: the same")
    fig.suptitle(title, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_impact(runs, path):
    """The touchdown and the smoothness at 0.8 s: the foot through the clock's
    touchdown, its speed, the knee torque's rate, and the per-swing spread."""
    plt = _plt()
    stems = ["b_off_T08_1", "b_ff001_T08_1", "b_ff003_T08_1"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    ax_z, ax_v, ax_r, ax_b1, ax_b2, ax_b3 = axes.ravel()
    data = {k: [] for k in ("vz_arr", "t_arr", "dip", "rate_pk", "jerk", "x_dev")}
    labels = []
    for stem in stems:
        g = next(g for g in GROUPS if g[0] == stem)
        r = runs[stem]; rows = r["kept"]
        grid, z = aligned(r, rows, z_col, pre_s=0.0, post_s=0.25)
        T = 1e3 * stat(rows, "T")[0]
        _band(ax_z, grid - T, z, COL[g[1]], g[2])
        vz = np.gradient(z, 0.004, axis=1) / 1e3          # mm -> m/s
        _band(ax_v, grid - T, vz, COL[g[1]], g[2])
        _, c = aligned(r, rows, tau_col("tau_cmd", 2), pre_s=0.0, post_s=0.25)
        rate = np.abs(np.gradient(c, 0.004, axis=1))
        _band(ax_r, grid - T, rate, COL[g[1]], g[2])
        labels.append(g[2].replace(", 0.8 s", ""))
        for k in data:
            data[k].append([s[k] for s in rows])
    for ax in (ax_z, ax_v, ax_r):
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("ms since the clock's touchdown")
        ax.set_xlim(-160, 250)
    ax_z.axhline(0, color="k", lw=0.4); ax_z.set_ylabel("foot height above its site, mm")
    ax_z.set_title("the foot through the clock's touchdown"); ax_z.legend(fontsize=7)
    ax_v.axhline(0, color="k", lw=0.4); ax_v.set_ylabel("foot vertical speed, m/s")
    ax_v.set_title("its vertical speed (on the floor: the impact speed at z = 0)")
    ax_r.set_ylabel("|d tau_knee / dt| sent, N*m/s"); ax_r.set_title("the knee torque's rate")
    ax_r.axhline(300, color="k", ls="--", lw=0.8)
    for ax, k, sc, lab in ((ax_b1, "vz_arr", 1.0, "speed at arrival (z < 2 mm), m/s"),
                           (ax_b2, "dip", 1e3, "lowest point after the clock's touchdown, mm"),
                           (ax_b3, "rate_pk", 1.0, "peak |d tau / dt| in the swing, N*m/s")):
        ax.boxplot([sc * np.array(v, float)[np.isfinite(v)] for v in data[k]], tick_labels=labels,
                   widths=0.5)
        ax.set_ylabel(lab); ax.axhline(0, color="k", lw=0.4)
    ax_b3.axhline(300, color="k", ls="--", lw=0.8)
    fig.suptitle("0.8 s: the touchdown and the smoothness, FF off / 0.001 / 0.003 "
                 "(median with p10-p90; boxes over 40 swings)", y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_xy(runs, path):
    """The foot's path in the trunk's x-z and x-y through the swing, 0.8 s and 2.0 s."""
    plt = _plt()
    stems = ["b_off_T08_1", "b_ff001_T08_1", "b_ff003_T08_1", "b_off_T20_1", "b_ff003_T20_1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for stem in stems:
        g = next(g for g in GROUPS if g[0] == stem)
        r = runs[stem]; rows = r["kept"]
        _, x = aligned(r, rows, lambda r, a, c, i: 1e3 * (r["x_b"][a:c, i, 0] - r["rest"][i, 0])
                       * (1.0 if i < 2 else -1.0), pre_s=0.0, post_s=0.0)
        _, y = aligned(r, rows, lambda r, a, c, i: 1e3 * (r["x_b"][a:c, i, 1] - r["rest"][i, 1])
                       * (1.0 if i % 2 == 0 else -1.0), pre_s=0.0, post_s=0.0)
        _, z = aligned(r, rows, z_col, pre_s=0.0, post_s=0.0)
        ls = "-" if g[3] == 0.8 else "--"
        axes[0].plot(np.nanmedian(x, axis=0), np.nanmedian(z, axis=0), color=COL[g[1]], ls=ls,
                     lw=1.8, label=g[2])
        axes[1].plot(np.nanmedian(x, axis=0), np.nanmedian(y, axis=0), color=COL[g[1]], ls=ls,
                     lw=1.8, label=g[2])
    axes[0].set_xlabel("x from the site, mm (+ outboard: front legs forward, rear back)")
    axes[0].set_ylabel("height, mm"); axes[0].set_title("side view: the arc has no x")
    axes[1].set_xlabel("x from the site, mm"); axes[1].set_ylabel("y from the site, mm (+ outboard)")
    axes[1].set_title("top view")
    axes[0].legend(fontsize=7); axes[0].axvline(0, color="k", lw=0.4); axes[1].axvline(0, color="k", lw=0.4)
    axes[1].axhline(0, color="k", lw=0.4)
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("The foot's path through the swing (median over 40, the legs mirrored onto one)",
                 y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_inertia(runs, path):
    """tau_cmd - g against qdd, knee and pitch, the 0.8 s runs: the slope is
    the inertia the leg answered the command with -- what M0 should be."""
    plt = _plt()
    stems = ["b_off_T08_1", "b_ff001_T08_1", "b_ff003_T08_1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lk = runs[stems[0]]["links"]
    for ax, j in zip(axes, ("pitch", "knee")):
        jj = JOINTS.index(j)
        for stem in stems:
            g = next(g for g in GROUPS if g[0] == stem)
            f = runs[stem]["fit"]["tau_cmd"][j]
            A, y = f["A"], f["y"]
            ax.plot(A[:, 0], y - f["b"] * A[:, 1] - f["c"], ".", ms=2, alpha=0.3, color=COL[g[1]])
            xx = np.array([A[:, 0].min(), A[:, 0].max()])
            ax.plot(xx, f["I"] * xx, color=COL[g[1]], lw=2,
                    label="%s: I = %.4f (R^2 %.2f, torque %.0f ms ahead)"
                    % (g[2], f["I"], f["r2"], f["shift_ms"]))
        xx = np.array(ax.get_xlim())
        ax.plot(xx, (lk[jj] + P.ARMATURE) * xx, "k--", lw=1.2,
                label="M0's diag, armature %.4f: %.4f" % (P.ARMATURE, lk[jj] + P.ARMATURE))
        ax.plot(xx, lk[jj] * xx, "k:", lw=1.0, label="the links alone: %.4f" % lk[jj])
        ax.set_xlabel("%s qdd, rad/s^2 (the log's 40 Hz-filtered qd, differenced once)" % j)
        ax.set_ylabel("tau_cmd - g - b qd - c, N*m"); ax.set_title("%s: the inertia the joint answered with" % j)
        ax.legend(fontsize=7)
    fig.suptitle("0.8 s: the torque sent against the joint acceleration over the swing windows "
                 "(the measured channel reads 0.68x and 12 ms late: results.txt)", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_legs(runs, path):
    """Every leg on its own, 0.8 s FF off and FF on 0.003: do the four agree?"""
    plt = _plt()
    stems = ["b_off_T08_1", "b_ff003_T08_1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    leg_col = ["C0", "C1", "C2", "C3"]
    for ax, stem in zip(axes, stems):
        g = next(g for g in GROUPS if g[0] == stem)
        r = runs[stem]
        for i in range(4):
            rows = [s for s in r["kept"] if s["leg"] == i]
            grid, z = aligned(r, rows, z_col)
            _band(ax, grid, z, leg_col[i], "%s: apex %.1f, h_td %+.1f" % (
                LEGS[i], 1e3 * stat(rows, "apex")[0], 1e3 * stat(rows, "h_td")[0]), lw=1.4)
        _, zr = aligned(r, r["kept"], zref_col)
        ax.plot(grid, np.nanmedian(zr, axis=0), "k--", lw=1.2)
        T = 1e3 * stat(r["kept"], "T")[0]
        ax.axvline(0, color="k", lw=0.6); ax.axvline(T, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(g[2]); ax.set_xlabel("ms since the clock's liftoff"); ax.legend(fontsize=7)
    axes[0].set_ylabel("foot height above its site, mm")
    fig.suptitle("The four legs separately (median with p10-p90 over 10 swings each)", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    runs = {}
    for g in GROUPS:
        r = load(g[0], args.runs)
        r["swings"] = swings(r)
        r["kept"] = kept(r["swings"])
        r["fit"] = fit_inertia(r, r["kept"])
        r["links"] = links_diag(r)
        r["follow"] = following(r, r["kept"])
        runs[g[0]] = r
        assert len(r["kept"]) == 40, (g[0], len(r["kept"]))
    out = []
    tables(runs, out)
    with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    if not args.no_figures:
        os.makedirs(FIG, exist_ok=True)
        fig_period(runs, os.path.join(FIG, "1_period_height.png"))
        fig_metrics(runs, os.path.join(FIG, "2_metrics_vs_period.png"))
        fig_ff(runs, ["b_off_T20_1", "b_ff003_T20_1"], os.path.join(FIG, "3_T20_ff.png"),
               "Guess 2 -- 2.0 s: the feedforward off against on (armature 0.003)")
        fig_ff(runs, ["b_off_T08_1", "b_ff001_T08_1", "b_ff003_T08_1"],
               os.path.join(FIG, "4_T08_ff.png"),
               "Guess 3 -- 0.8 s: the feedforward off, on with armature 0.001, on with 0.003")
        fig_impact(runs, os.path.join(FIG, "5_T08_touchdown_smoothness.png"))
        fig_xy(runs, os.path.join(FIG, "6_foot_path.png"))
        fig_inertia(runs, os.path.join(FIG, "7_inertia_fit.png"))
        fig_legs(runs, os.path.join(FIG, "8_legs.png"))
        print("figures in %s" % FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
