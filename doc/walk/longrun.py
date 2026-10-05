"""longrun -- the trot and the walk for 30 s in place, and the keys' box, on a
ROBOT-LIKE plant: what fails "after a while", and which hardware unknown
does it.  doc/walk/README.md section 11 quotes the tables.

    python doc/walk/longrun.py inplace   30 s in place: 8 swing/clock configs x 2 plants
    python doc/walk/longrun.py box       the keys' box (0.06 / 0.03 m/s, 12 deg/s), robot-like
    python doc/walk/longrun.py single    ONE unknown at a time, 30 s in place
    python doc/walk/longrun.py corr      the robot-like plant with the CoM pin corrected
    python doc/walk/longrun.py all       inplace + box
    python doc/walk/longrun.py L_trot_hwlike C_walk_osc_fwd06 ...   named runs

THE PLANT.  `hwlike` is every unknown the bench and the robustness study
name, together: the rotor 0.6x DOG5's (`armature_scale`), 0.1 N*m of joint
Coulomb, the IMU 15 ms late, floor mu 0.7, the CoM 3 / 5 mm off the CAD's in
x / y, and 10 % less torque per amp.  `ideal` is the model as drawn.
`single` takes them one at a time.  `corr` is `hwlike` with the law's SRB
pin shifted by the same 3 / 5 mm -- `hw.stand --com-offset 3 5` once
`hw.com_check` has measured it.

THE CONFIGS, all on `walksim.entry()` (the entry points as `hw.stand.main`
builds them): `trot` is hw.fold_trot as shipped; `trot150` the same with
the walk's 150/150/400 swing gains; `walk_osc` hw.fold_walk as shipped;
`walk_osc_fit` with M0 fitted to the plant's rotor; `walk_imp150_d80` the
walk's impedance on the trot's clock (--swing-law impedance --duty 0.8
--settle 0.2), `walk_imp10_d80` the trot's own 10/10/400 gains there, and
the two `_d70` the same on the walk's clock.

WHAT IT REPORTS beyond walksim's metrics: the tilt rms per 10 s of trot (a
number that GROWS is the failure that takes a while), the trunk's drift
over the trot and the yaw drift.  Runs write data/longrun/<name>.json and
.npz (git-ignored), ~25-30 s each, four at a time.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "doc", "crouch_trot"))

import walksim as W                  # noqa: E402
import hwsim as H                    # noqa: E402
from sim import params as P          # noqa: E402
from hw.balance import config as BCFG   # noqa: E402

OUT = os.path.join(HERE, "data", "longrun")
ARM_FIT = 0.6 * P.ARMATURE

#: the plant as the bench and the robustness study describe DOG6: a rotor 0.6
#: of DOG5's, 0.1 N*m of joint Coulomb, IMU 15 ms late, floor mu 0.7, the CoM
#: 3 / 5 mm off, 10 % less torque per amp.
HWLIKE = dict(armature_scale=0.6, frictionloss=0.1, imu_ms=15.0,
              friction=0.7, com_err_mm=(3.0, 5.0), kt_scale=0.9)
IDEAL = {}

TROT_KP, TROT_KD = list(BCFG.KP_SWING), list(BCFG.KD_SWING)
WALK_KP, WALK_KD = list(BCFG.KP_SWING_WALK), list(BCFG.KD_SWING_WALK)

#: the six configurations, in place
CONFIGS = {
    # hw.fold_trot as shipped: impedance 10/10/400, duty 0.80, settle 0.2
    "trot":           dict(walk=False, duty=0.8, settle=BCFG.SETTLE_S),
    # hw.fold_trot with --kp-swing 150 150 400 --kd-swing 6 6 40 (the walk's impedance gains)
    "trot150":        dict(walk=False, duty=0.8, settle=BCFG.SETTLE_S,
                           trot_kp=WALK_KP, trot_kd=WALK_KD),
    # hw.fold_walk as shipped: osc 25/25/20, duty 0.70, no settle, M0 DOG5's
    "walk_osc":       dict(),
    # ... with M0 fitted to the plant's rotor (--ff-armature)
    "walk_osc_fit":   dict(ff_armature=ARM_FIT),
    # the "next flight's flags": impedance at the walk's 150/150/400, the trot's clock
    "walk_imp150_d80": dict(swing_law="impedance", duty=0.8, settle=BCFG.SETTLE_S),
    # impedance at the TROT's 10/10/400 and the trot's clock: fold_trot + the walk's layers
    "walk_imp10_d80": dict(swing_law="impedance", kp_swing=TROT_KP, kd_swing=TROT_KD,
                           duty=0.8, settle=BCFG.SETTLE_S),
    # impedance at the trot's gains on the walk's clock
    "walk_imp10_d70": dict(swing_law="impedance", kp_swing=TROT_KP, kd_swing=TROT_KD),
    # impedance at the walk's gains, walk's clock, no settle (duty 0.70)
    "walk_imp150_d70": dict(swing_law="impedance"),
}
PLANTS = {"ideal": IDEAL, "hwlike": HWLIKE}

T_INPLACE = 30.0
INPLACE = [(0.0, 0.0, 0.0, 0.0)]


def _box(vx=0.0, vy=0.0, r=0.0, hold=6.0, t_cmd=1.5, **opts):
    return dict(schedule=[(0.0, 0.0, 0.0, 0.0), (t_cmd, vx, vy, r),
                          (t_cmd + hold, 0.0, 0.0, 0.0)],
                trot_s=t_cmd + hold + 2.0, opts=opts)


LONG = {}
for cname, copts in CONFIGS.items():
    for pname, popts in PLANTS.items():
        LONG["L_%s_%s" % (cname, pname)] = dict(
            schedule=INPLACE, trot_s=T_INPLACE, opts=dict(copts, **popts))

#: the keys' box as shipped since 2026-10-05: 0.06 / 0.03 m/s and 12 deg/s
BOXCASES = dict(fwd06=dict(vx=0.06), back06=dict(vx=-0.06), lat03=dict(vy=0.03),
                yaw12=dict(r=12.0), combo=dict(vx=0.06, r=12.0))
BOX = {}
for cname in ("walk_osc", "walk_osc_fit", "walk_imp150_d80", "walk_imp10_d80",
              "walk_imp10_d70", "walk_imp150_d70"):
    for case, ckw in BOXCASES.items():
        BOX["B_%s_%s" % (cname, case)] = _box(**ckw, **CONFIGS[cname], **HWLIKE)


#: ONE unknown at a time, 30 s in place, three configs
SINGLES = {
    "arm06":   dict(armature_scale=0.6),
    "arm06fit": dict(armature_scale=0.6, ff_armature=ARM_FIT),
    "com35":   dict(com_err_mm=(3.0, 5.0)),
    "kt09":    dict(kt_scale=0.9),
    "fric":    dict(frictionloss=0.1, imu_ms=15.0),
    "mu07":    dict(friction=0.7),
}
SINGLE = {}
for cname in ("trot", "trot150", "walk_osc", "walk_imp10_d80", "walk_imp150_d80"):
    for sname, sopts in SINGLES.items():
        if sname == "arm06fit" and cname != "walk_osc":
            continue
        SINGLE["S_%s_%s" % (cname, sname)] = dict(
            schedule=INPLACE, trot_s=T_INPLACE, opts=dict(CONFIGS[cname], **sopts))
W.SCENARIOS.update(SINGLE)

#: THE CoM CORRECTED IN THE LAW: the plant's CoM 3/5 mm off (hwlike) and the
#: SRB pin shifted by the same amount -- what `hw.stand --com-offset 3 5`
#: would do once the offset is measured.
CORR = {}
for cname in ("trot", "walk_osc", "walk_imp150_d80", "walk_imp10_d80"):
    CORR["C_%s_inplace" % cname] = dict(
        schedule=INPLACE, trot_s=T_INPLACE,
        opts=dict(CONFIGS[cname], com_offset_mm=(3.0, 5.0), **HWLIKE))
for case, ckw in BOXCASES.items():
    CORR["C_walk_osc_%s" % case] = _box(**ckw, com_offset_mm=(3.0, 5.0),
                                        **CONFIGS["walk_osc"], **HWLIKE)
    CORR["C_walk_imp150_d80_%s" % case] = _box(
        **ckw, com_offset_mm=(3.0, 5.0), **CONFIGS["walk_imp150_d80"], **HWLIKE)
W.SCENARIOS.update(CORR)

W.SCENARIOS.update(LONG)
W.SCENARIOS.update(BOX)


def run(name: str) -> dict:
    sc = W.SCENARIOS[name]
    opts = dict(sc["opts"])
    hwkw = dict(frictionloss=opts.pop("frictionloss", 0.0),
                imu_latency=1e-3 * opts.pop("imu_ms", 3.0),
                friction=opts.pop("friction", None),
                armature_scale=opts.pop("armature_scale", 1.0),
                com_err_mm=tuple(opts.pop("com_err_mm", (0.0, 0.0))),
                kt_scale=opts.pop("kt_scale", 1.0))
    t0 = time.time()
    trot_kp = opts.pop("trot_kp", None)
    trot_kd = opts.pop("trot_kd", None)
    saved = (BCFG.KP_SWING, BCFG.KD_SWING)
    if trot_kp is not None:
        BCFG.KP_SWING = np.array(trot_kp, dtype=float)
        BCFG.KD_SWING = np.array(trot_kd, dtype=float)
    com_offset = opts.pop("com_offset_mm", None)
    try:
        e, plan = W.entry(**opts)
    finally:
        BCFG.KP_SWING, BCFG.KD_SWING = saved
    if com_offset is not None:
        import dataclasses
        srb = e.law_kw["srb"]
        com = np.array(srb.com_body, dtype=float)
        com[:2] += 1e-3 * np.asarray(com_offset, dtype=float)
        com.flags.writeable = False
        e.law_kw["srb"] = dataclasses.replace(srb, com_body=com)
    sim = H.HwSim(e, H.HwParams(**hwkw))
    sim.place(e.crouch.q, e.crouch.z_origin)
    pushes = [(-1.0, -1.0, (0.0, 0.0, 0.0))]
    drv = W.Driver(sim, sc["schedule"], push=None, pushes=pushes)
    op = H.Operator(hold_s=W.T_HOLD, trot_s=sc["trot_s"], park=False,
                    hold_after_s=1.5)
    res = sim.run(op, t_max=3.0 + 1.0 + 3.0 + W.T_HOLD + sc["trot_s"] + 3.0,
                  push=pushes, quiet=True, pre_update=drv)
    out = W.metrics(name, res, drv, sim)
    out["t_tilt15"] = H.trot_metrics(res).get("t_tilt15_after_T")
    out.update(_time_course(res))
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "%s.json" % name), "w") as fh:
        json.dump(out, fh, indent=1)
    keep = ("t", "phase", "roll", "pitch", "yaw", "xy_trunk", "z_trunk", "foot_w",
            "foot_fn", "tau_act", "tau_req", "swing_s", "x_b", "contact_w", "fz",
            "b_d", "com_w", "q", "q_des")
    np.savez_compressed(os.path.join(OUT, "%s.npz" % name),
                        **{k: res[k] for k in keep if k in res})
    return out


def _time_course(res) -> dict:
    """Tilt rms per 10 s of trot, trunk drift and yaw drift over the trot."""
    ph, t = res["phase"], res["t"]
    trot = np.flatnonzero(ph == "trot")
    if not trot.size:
        return {}
    i0, i1 = int(trot[0]), int(trot[-1])
    tilt = np.degrees(np.maximum(np.abs(res["roll"]), np.abs(res["pitch"])))
    tt = t[i0:i1 + 1] - t[i0]
    wins = []
    for a in np.arange(0.0, tt[-1] + 1e-9, 10.0):
        sel = (tt >= a) & (tt < a + 10.0)
        if sel.sum() > 50:
            wins.append(round(float(np.sqrt(np.mean(tilt[i0:i1 + 1][sel] ** 2))), 2))
    xy = res["xy_trunk"]
    yaw = np.degrees(np.unwrap(res["yaw"]))
    return dict(tilt_rms_per_10s=wins,
                drift_xy_mm=[round(1e3 * float(v), 0) for v in (xy[i1] - xy[i0])],
                drift_yaw_deg=round(float(yaw[i1] - yaw[i0]), 1),
                t_trot_end=round(float(tt[-1]), 1))


def table(results) -> str:
    lines = ["%-28s %-4s %-6s %6s %6s %6s  %-26s %10s %6s %6s"
             % ("scenario", "fell", "t15", "tilt", "tau", "slip",
                "tilt rms /10s", "drift xy mm", "dyaw", "wall")]
    for r in results:
        segs = [s for s in r.get("segments", []) if any(s["cmd"])]
        vt = ("" if not segs else " v %+.3f %+.3f r %+.1f"
              % (segs[0]["v_true"][0], segs[0]["v_true"][1], segs[0]["r_true"]))
        lines.append("%-28s %-4s %-6s %6.1f %6.2f %6.1f  %-26s %10s %6.1f %6.0f%s"
                     % (r["name"], "YES" if r["fell"] else "no",
                        str(r.get("t_tilt15", "-")),
                        r.get("max_tilt", float("nan")),
                        r.get("peak_tau", float("nan")),
                        r.get("slip_mm_per_s", float("nan")),
                        str(r.get("tilt_rms_per_10s", "")),
                        str(r.get("drift_xy_mm", "")),
                        r.get("drift_yaw_deg", float("nan")),
                        r.get("wall_s", 0), vt))
    return "\n".join(lines)


def main(argv) -> int:
    names = []
    if "inplace" in argv or "all" in argv:
        names += list(LONG)
    if "box" in argv or "all" in argv:
        names += list(BOX)
    if "single" in argv:
        names += list(SINGLE)
    if "corr" in argv:
        names += list(CORR)
    if "extra" in argv:
        names += ["L_trot150_ideal", "L_trot150_hwlike"]
    names += [a for a in argv if a in W.SCENARIOS]
    if not names:
        print(__doc__)
        return 2
    t0 = time.time()
    with Pool(min(4, len(names))) as pool:
        results = pool.map(run, names)
    print(table(results))
    print("wall %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
