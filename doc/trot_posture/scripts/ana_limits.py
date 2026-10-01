"""ana: what limits each candidate stance (from cand_<NAME>.json), plus a
closer look at the worst-case npz logs for the best 8 by worst-case tilt and
FOLD_asis.  Writes data/ana_limits.json and prints the table."""
import json, glob, os, sys
import numpy as np

D = "doc/trot_posture/data"
CAP = 9.0
summ = json.load(open(os.path.join(D, "candidates_summary.json")))
summ = {r["name"]: r for r in summ}


def per_clock(tr):
    """worst survivor, failures, rms-vs-max, walk axis, fn, tau for one clock."""
    rows = {p: v for p, v in tr.items() if not p.startswith("_")}
    surv = {p: v for p, v in rows.items() if not v["fell"]}
    fails = [p for p, v in rows.items() if v["fell"]]
    out = dict(n_surv=len(surv), fails=fails)
    if not surv:
        return out
    pw = max(surv, key=lambda p: surv[p]["max_tilt_trot"])
    w = surv[pw]
    rr, pr = w["trot_roll_rms_deg"], w["trot_pitch_rms_deg"]
    dx, dy = w["trot_xy_drift_mm"]
    out.update(worst=pw, worst_tilt=w["max_tilt_trot"], roll_rms=rr, pitch_rms=pr,
               ratio_max_over_rms=round(w["max_tilt_trot"] / max(rr, pr, 1e-3), 1),
               walk_mm=[dx, dy], walk_axis=("x" if abs(dx) > abs(dy) else "y"),
               walk_norm=round(float(np.hypot(dx, dy)), 1),
               yaw_drift=w["trot_yaw_drift_deg"], fn_peak=w["trot_foot_fn_peak_n"],
               peak_tau=w["peak_tau_trot"], peak_tau_frac_cap=round(w["peak_tau_trot"] / CAP, 2),
               hold_after=w["hold_after_tilt_deg"], ideal_tilt=rows["ideal"]["max_tilt_trot"])
    # ranked list of the 3 worst perturbations
    out["rank"] = sorted(((p, v["max_tilt_trot"]) for p, v in surv.items()),
                         key=lambda x: -x[1])[:3]
    out["fn_peak_max"] = max(v["trot_foot_fn_peak_n"] for v in surv.values())
    out["tau_peak_max"] = max(v["peak_tau_trot"] for v in surv.values())
    out["walk_max"] = max(float(np.hypot(*v["trot_xy_drift_mm"])) for v in surv.values())
    # sensitivity: tilt(com_y_+10) - tilt(ideal), tilt(imu_lat_30) - ideal
    sens = {}
    for p in ("com_y_+10mm", "imu_lat_30ms", "kt_0.85", "joint_fric_0.3", "mass_+0.4kg"):
        if p in surv:
            sens[p] = round(surv[p]["max_tilt_trot"] - rows["ideal"]["max_tilt_trot"], 2)
    out["sens"] = sens
    return out


res = {}
for f in sorted(glob.glob(os.path.join(D, "cand_*.json"))):
    d = json.load(open(f))
    name = d["info"]["spec"]["name"]
    r = dict(spec=d["info"]["spec"], stand=dict(
        hold=d["stand"].get("tau_hold_mean_max_nm"), rise_peak=d["stand"].get("tau_rise_peak_nm"),
        rise_joint=d["stand"].get("tau_rise_peak_joint"), drift=d["stand"].get("drift_x_mm"),
        slip=d["stand"].get("foot_slip_max_mm"), tilt_rise=d["stand"].get("tilt_max_rise_deg"),
        h_min_rise=d["stand"].get("h_min_rise_mm")),
        crouch=d["info"]["crouch_h_mm"], reach=d["info"].get("reach_u"), knee=d["info"].get("knee_from_straight_deg"))
    for ck, tr in d["trots"].items():
        r[ck] = per_clock(tr)
    res[name] = r

# ---- print table ------------------------------------------------------------
order = sorted(res, key=lambda n: (res[n]["P0.8_A40"].get("worst_tilt", 99), res[n]["P0.5_A20"].get("worst_tilt", 99)))
print("%-14s | %-13s w08  ideal rRMS pRMS mx/rms walk(x,y)   fn   tau  | %-13s w05  rRMS pRMS walk  fn   tau | hold rise" % ("name", "worst@0.8", "worst@0.5"))
for n in order:
    r = res[n]; a = r["P0.8_A40"]; b = r["P0.5_A20"]
    if "worst" in a:
        s8 = "%-13s %5.2f %5.2f %4.2f %4.2f %5.1f (%+4.0f,%+4.0f) %5.1f %4.2f" % (
            a["worst"][:13], a["worst_tilt"], a["ideal_tilt"], a["roll_rms"], a["pitch_rms"], a["ratio_max_over_rms"],
            a["walk_mm"][0], a["walk_mm"][1], a["fn_peak"], a["peak_tau"])
    else:
        s8 = "FAILED %d/13 %s" % (13 - len(a["fails"]), ",".join(a["fails"])[:40])
    if "worst" in b:
        s5 = "%-13s %5.2f %4.2f %4.2f (%+4.0f,%+4.0f) %5.1f %4.2f" % (
            b["worst"][:13], b["worst_tilt"], b["roll_rms"], b["pitch_rms"], b["walk_mm"][0], b["walk_mm"][1], b["fn_peak"], b["peak_tau"])
    else:
        s5 = "FAILED"
    print("%-14s | %s | %s | %.2f %.2f" % (n, s8, s5, r["stand"]["hold"], r["stand"]["rise_peak"]))

print("\nsensitivity (tilt - ideal tilt, deg) at 0.8/40:")
for n in order:
    a = res[n]["P0.8_A40"]
    if "sens" in a:
        print("%-14s %s  rank %s" % (n, " ".join("%s=%+.2f" % (k[:9], v) for k, v in a["sens"].items()),
                                     " ".join("%s:%.2f" % (p[:9], t) for p, t in a["rank"])))

json.dump(res, open(os.path.join(D, "ana_limits.json"), "w"), indent=1)
