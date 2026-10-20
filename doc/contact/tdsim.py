"""tdsim -- touchdown detection (scheme A) in MuJoCo: does it see the floor?

    python doc/contact/tdsim.py                 the standard set, 4 in parallel
    python doc/contact/tdsim.py lat05 fwd10     named cases only (each off + on)
    python doc/contact/tdsim.py --list
    python doc/contact/tdsim.py --quick         inplace + lat05, for a smoke run

Every case runs TWICE: `hw.fold2_walk` as shipped with `--no-touchdown` (the
clock's word for which foot is down -- the walk flown on 2026-10-07) and
with the detector ON (the default since 2026-10-20).  The plant, the loop,
the estimator and the operator are doc/walk/walksim.py's (FOLD2, `entry`,
`Driver`); the detector is the one `hw.stand` attaches
(`hw.balance.contact.ContactDetector()` at config's TD_*).

WHAT IT REPORTS, per run
    fell / tilt / tau / slip   walksim's, over the trot
    late                       walksim's: ms from the law's contact (weight
                               > 0) to the floor's first newton -- with the
                               detector, from the DETECTION to the floor
    det err                    ms from the TRUE touchdown (MuJoCo's foot
                               normal force >= 1 N, first sweep after the
                               swing) to the detector's LANDED: + after the
                               floor, - before it.  median / p90 / max
    early (false)              detections more than 8 ms BEFORE the foot
                               touched -- a foot handed to the allocator in
                               the air, the thing the clearance gate exists
                               to prevent
    probes                     the detector's own account: probes started,
                               given up at the limit, cut by the two-foot
                               rule, re-probes of a landed foot that left
                               the floor, lost; detections by the push / the
                               stopped foot
Everything is simulation on the CAD model (doc/walk/README.md says what the
plant has and has not); nothing has flown.  `data/td_*.json` keep the runs.
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
sys.path.insert(0, os.path.join(REPO, "doc", "crouch_trot"))
sys.path.insert(0, os.path.join(REPO, "doc", "walk"))

import hwsim as H                    # noqa: E402
import walksim as W                  # noqa: E402
from hw.balance import contact as CONTACT   # noqa: E402

DATA = os.path.join(HERE, "data")
F2 = dict(posture="fold2")
#: the robot-like plant of doc/walk section 11: joint friction, a late IMU,
#: DOG6's lighter rotor, a mu 0.7 floor
ROBOTLIKE = dict(frictionloss=0.1, imu_ms=15.0, armature_scale=0.6,
                 friction=0.7)


def _walk(vx=0.0, vy=0.0, r=0.0, hold=5.0, t_cmd=1.5, push=None, **opts):
    sc = W._walk(vx=vx, vy=vy, r=r, hold=hold, t_cmd=t_cmd, **F2, **opts)
    if push is not None:
        sc["push"] = push
    return sc


CASES = {
    "inplace": dict(schedule=W.INPLACE, trot_s=8.0, opts=dict(F2)),
    "fwd10":   _walk(vx=0.10),
    "lat05":   _walk(vy=0.05),
    "latm05":  _walk(vy=-0.05),
    "yaw20":   _walk(r=20.0),
    "combo":   _walk(vx=0.10, r=20.0),
    "fwd15":   _walk(vx=0.15),
    # the hardware unknowns
    "fwd10_rl":  _walk(vx=0.10, **ROBOTLIKE),
    "lat05_rl":  _walk(vy=0.05, **ROBOTLIKE),
    "yaw20_rl":  _walk(r=20.0, **ROBOTLIKE),
    # the CoM 3 / 5 mm off the CAD's: the one unknown that rocked every
    # walk in doc/walk section 11 (the law pinned at the CAD's, no --com-auto
    # here: the harness's HOLD applies none)
    "fwd10_com": _walk(vx=0.10, com_err_mm=(3.0, 5.0)),
    "lat05_com": _walk(vy=0.05, com_err_mm=(3.0, 5.0)),
    # a sideways shove, 8 N for 0.25 s, 3 s into the trot, walking left at
    # the box speed
    "lat05_push": _walk(vy=0.05, push=("trot", 3.0, 0.25, (0.0, 8.0, 0.0))),
    # the trot in place of hw.fold2_trot itself (no walk plan), duty 0.80
    "trot_inplace": dict(schedule=W.INPLACE, trot_s=8.0,
                         opts=dict(F2, walk=False, duty=0.8,
                                   settle=W.SETTLE_TROT)),
}
STANDARD = list(CASES)
QUICK = ["inplace", "lat05"]

#: name -> (case, detector on?)
RUNS = {}
for _c in CASES:
    RUNS["%s_off" % _c] = (_c, False)
    RUNS["%s_on" % _c] = (_c, True)
    # walksim.metrics reads the schedule from its own table
    for _tag in ("off", "on"):
        W.SCENARIOS["td_%s_%s" % (_c, _tag)] = CASES[_c]


def run(name: str) -> dict:
    case, on = RUNS[name]
    sc = CASES[case]
    opts = dict(sc["opts"])
    hwkw = dict(frictionloss=opts.pop("frictionloss", 0.0),
                imu_latency=1e-3 * opts.pop("imu_ms", 3.0),
                friction=opts.pop("friction", None),
                armature_scale=opts.pop("armature_scale", 1.0),
                com_err_mm=tuple(opts.pop("com_err_mm", (0.0, 0.0))))
    t0 = time.time()
    e, plan = W.entry(**opts)
    if on:
        e.law_kw["contact_detector"] = CONTACT.ContactDetector()
    sim = H.HwSim(e, H.HwParams(**hwkw))
    sim.place(e.crouch.q, e.crouch.z_origin)
    pushes = [(-1.0, -1.0, (0.0, 0.0, 0.0))]
    drv = W.Driver(sim, sc["schedule"], push=sc.get("push"), pushes=pushes)
    op = H.Operator(hold_s=W.T_HOLD, trot_s=sc["trot_s"], park=False,
                    hold_after_s=1.5)
    res = sim.run(op, t_max=3.0 + 1.0 + 3.0 + W.T_HOLD + sc["trot_s"] + 3.0,
                  push=pushes, quiet=True, pre_update=drv)
    out = W.metrics("td_%s_%s" % (case, "on" if on else "off"), res, drv, sim)
    out["name"] = name
    out["td"] = on
    out.update(detection_metrics(res))
    det = sim.stand.balance.contact_detector
    if det is not None:
        out.update(swings=det.swings, early_n=len(det.early),
                   late_n=len(det.late), probes=det.probes, limit=det.limit,
                   window=det.window, lifted=det.lifted, lost=det.lost,
                   by_push=det.by_push, by_stop=det.by_stop,
                   lead_ms=[round(1e3 * x, 1) for x in det.early],
                   lag_ms=[round(1e3 * x, 1) for x in det.late])
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "td_%s.json" % name), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(DATA, "td_%s.npz" % name),
                        **{k: v for k, v in res.items()
                           if isinstance(v, np.ndarray)})
    return out


def detection_metrics(res) -> dict:
    """The detector's LANDED against the floor's first newton, per swing."""
    ph, t = res["phase"], res["t"]
    trot = np.flatnonzero(ph == "trot")
    out = {}
    if not trot.size:
        return out
    i0, i1 = int(trot[0]), int(trot[-1]) + 1
    fn = res["foot_fn"]
    st = res.get("td_state")
    if st is None or not np.isfinite(st[i0:i1]).any():
        return out
    errs, false_early, n_sw = [], 0, 0
    for leg in range(4):
        s = st[i0:i1, leg]
        f = fn[i0:i1, leg]
        k = 1
        while k < s.size:
            if s[k] == CONTACT.SWING and s[k - 1] != CONTACT.SWING:
                # a swing starts at k: the true touchdown is the first sweep
                # after the foot has left the floor that it is loaded again
                j = k
                while j < s.size and f[j] >= 1.0:
                    j += 1                      # still loaded at liftoff
                while j < s.size and f[j] < 1.0:
                    j += 1
                k_true = j
                m = k
                while m < s.size and s[m] != CONTACT.LANDED:
                    m += 1
                if m < s.size and k_true < s.size:
                    n_sw += 1
                    err = (m - k_true) * H.T_SWEEP
                    errs.append(err)
                    if err < -0.008:
                        false_early += 1
                k = max(m, k_true, k + 1)
            else:
                k += 1
    if errs:
        a = 1e3 * np.asarray(errs)
        out.update(det_n=n_sw, det_err_med_ms=round(float(np.median(a)), 0),
                   det_err_p90_ms=round(float(np.percentile(a, 90)), 0),
                   det_err_max_ms=round(float(a.max()), 0),
                   det_err_min_ms=round(float(a.min()), 0),
                   false_early=false_early)
    return out


def table(results) -> str:
    lines = ["%-13s %-3s %-4s %5s %5s %5s %5s  %-18s %5s  %-24s %s"
             % ("case", "td", "fell", "tilt", "tau", "slip", "late",
                "det err med/p90/max", "false", "probes lim/win/lift/lost",
                "push/stop  law p50")]
    for r in results:
        lines.append("%-13s %-3s %-4s %5.1f %5.2f %5.1f %5s  %-18s %5s  %-24s %s"
                     % (r["name"].rsplit("_", 1)[0], "on" if r["td"] else "off",
                        "YES" if r["fell"] else "no",
                        r.get("max_tilt", float("nan")),
                        r.get("peak_tau", float("nan")),
                        r.get("slip_mm_per_s", float("nan")),
                        str(r.get("late_ms_mean", "-")),
                        ("%+.0f / %+.0f / %+.0f" % (r["det_err_med_ms"],
                                                    r["det_err_p90_ms"],
                                                    r["det_err_max_ms"])
                         if "det_err_med_ms" in r else "-"),
                        str(r.get("false_early", "-")),
                        ("%d  %d/%d/%d/%d" % (r["probes"], r["limit"],
                                              r["window"], r.get("lifted", 0),
                                              r["lost"])
                         if "probes" in r else "-"),
                        ("%d/%d  %.0f us" % (r["by_push"], r["by_stop"],
                                             r["law_p50_us"])
                         if "by_push" in r else "%.0f us" % r["law_p50_us"])))
    return "\n".join(lines)


def main(argv) -> int:
    if "--list" in argv:
        for k, v in CASES.items():
            print("%-13s %s %s" % (k, v["schedule"], v["opts"]))
        return 0
    cases = ([a for a in argv if a in CASES]
             or (QUICK if "--quick" in argv else STANDARD))
    names = []
    for c in cases:
        names += ["%s_off" % c, "%s_on" % c]
    t0 = time.time()
    with Pool(min(4, len(names))) as pool:
        results = pool.map(run, names)
    print(table(results))
    print("wall %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
