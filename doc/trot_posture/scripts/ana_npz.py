"""ana: open the worst-case npz logs (best 8 by worst-case tilt + FOLD_asis)
and say what the tilt IS: steady offset vs per-swing rocking, walk rate and
axis, touchdown force, torque peak and joint.  Writes data/ana_npz.json."""
import json, os, sys
import numpy as np

D = "doc/trot_posture/data"
JN = ["abd_FL", "hip_FL", "knee_FL", "abd_FR", "hip_FR", "knee_FR",
      "abd_RL", "hip_RL", "knee_RL", "abd_RR", "hip_RR", "knee_RR"]
LEG = ["FL", "FR", "RL", "RR"]
DT = 0.004


def analyse(path, period):
    z = np.load(path, allow_pickle=True)
    ph = z["phase"]; t = z["t"]
    trot = ph == "trot"
    out = dict(file=os.path.basename(path))
    if not trot.any():
        return out
    i0 = int(np.flatnonzero(trot)[0]); i1 = int(np.flatnonzero(trot)[-1])
    T0 = t[i0]
    roll = np.degrees(z["roll"][trot]); pitch = np.degrees(z["pitch"][trot])
    tt = t[trot] - T0
    out["T_start"] = round(float(T0), 2); out["trot_s"] = round(float(tt[-1]), 2)
    # offset vs rocking: one-cycle moving mean (slow) and the oscillation about it
    w = max(1, int(round(period / DT)))
    k = np.ones(w) / w
    slow = np.convolve(roll, k, mode="same")
    osc = roll - slow
    if len(roll) <= 2 * w + 2:
        w = max(1, len(roll) // 3)
    out["roll_mean"] = round(float(roll.mean()), 2)
    out["roll_max_abs"] = round(float(np.abs(roll).max()), 2)
    out["roll_slow_max_abs"] = round(float(np.abs(slow[w:-w]).max()), 2)
    out["roll_osc_amp"] = round(float(np.abs(osc[w:-w]).max()), 2)
    out["roll_osc_rms"] = round(float(np.sqrt(np.mean(osc[w:-w] ** 2))), 2)
    out["pitch_mean"] = round(float(pitch.mean()), 2)
    out["pitch_max_abs"] = round(float(np.abs(pitch).max()), 2)
    # spectrum of roll at 1/T (per cycle) and 2/T (per swing)
    n = len(roll); f = np.fft.rfftfreq(n, DT); R = np.abs(np.fft.rfft(roll - roll.mean())) * 2 / n
    fg = 1.0 / period

    def amp_at(fq):
        sel = (f > 0.8 * fq) & (f < 1.2 * fq)
        return round(float(R[sel].max()), 2) if sel.any() else None
    out["roll_amp_at_1/T"] = amp_at(fg); out["roll_amp_at_2/T"] = amp_at(2 * fg)
    sel = (f > 0) & (f < 0.5 * fg)
    out["roll_amp_lowf"] = round(float(R[sel].max()), 2) if sel.any() else None
    nc = int(3 * period / DT)
    out["roll_max_first3cyc"] = round(float(np.abs(roll[:nc]).max()), 2)
    out["roll_max_last3cyc"] = round(float(np.abs(roll[-nc:]).max()), 2)
    # walk
    xy = 1e3 * z["xy_trunk"][trot]; dxy = xy - xy[0]
    out["walk_total_mm"] = np.round(dxy[-1], 1).tolist()
    half = len(tt) // 2
    out["walk_rate_1st_half"] = np.round((dxy[half] - dxy[0]) / (tt[half] - tt[0]), 1).tolist()
    out["walk_rate_2nd_half"] = np.round((dxy[-1] - dxy[half]) / (tt[-1] - tt[half]), 1).tolist()
    out["yaw_drift_deg"] = round(float(np.degrees(z["yaw"][i1] - z["yaw"][i0])), 2)
    # touchdown force
    fn = z["foot_fn"][trot]; cw = z["contact_w"][trot]
    out["fn_peak_n"] = round(float(fn.max()), 1)
    ipk = np.unravel_index(int(fn.argmax()), fn.shape)
    leg = ipk[1]; c = cw[:, leg] > 0.5
    rises = np.flatnonzero(np.diff(c.astype(int)) == 1)
    prev = rises[rises <= ipk[0]]
    out["fn_peak_leg"] = LEG[leg]
    out["fn_peak_after_td_ms"] = round(float(1e3 * (ipk[0] - prev[-1]) * DT), 0) if prev.size else None
    n_on = (cw > 0.5).sum(axis=1)
    sel2 = n_on == 2
    out["fn_mean_2foot_n"] = round(float(fn[sel2][cw[sel2] > 0.5].mean()), 1) if sel2.any() else None
    out["fn_99pct_n"] = round(float(np.percentile(fn[cw > 0.5], 99)), 1)
    # torque
    tau = np.abs(z["tau_act"][trot])
    out["tau_peak"] = round(float(tau.max()), 2)
    out["tau_peak_joint"] = JN[int(np.argmax(tau.max(axis=0)))]
    out["tau_99pct"] = round(float(np.percentile(tau, 99)), 2)
    out["tau_frac_gt6"] = round(float((tau.max(axis=1) > 6).mean()), 3)
    out["h_min_mm"] = round(float(1e3 * z["z_trunk"][trot].min()) - 35, 1)
    out["h_mean_mm"] = round(float(1e3 * z["z_trunk"][trot].mean()) - 35, 1)
    out["cap_max"] = round(float(z["cap"][trot].max()), 2)
    return out


def failure_timeline(path, period):
    z = np.load(path, allow_pickle=True)
    ph = z["phase"]; t = z["t"]; mode = z["mode"]
    trot = ph == "trot"
    i0 = int(np.flatnonzero(trot)[0]); T0 = t[i0]
    print("\n==== %s" % os.path.basename(path))
    print("meta:", str(z["meta"])[:600])
    print("phases:", {p: (round(float(t[ph == p][0]), 2), round(float(t[ph == p][-1]), 2)) for p in dict.fromkeys(ph.tolist())})
    print("modes:", list(dict.fromkeys(mode.tolist())))
    last = int(np.flatnonzero(ph != "STOP")[-1])
    roll = np.degrees(z["roll"]); pitch = np.degrees(z["pitch"]); yaw = np.degrees(z["yaw"])
    xy = 1e3 * z["xy_trunk"]; zt = 1e3 * z["z_trunk"] - 35
    fn = z["foot_fn"]; cw = z["contact_w"]; tau = np.abs(z["tau_act"]); ss = z["swing_s"]
    print(" t-T   roll   pitch   yaw |    x     y     h | fn FL FR RL RR | cw   | swing_s             | tau_max joint   | cap")
    times = list(np.arange(0, 2.0, 0.1)) + list(np.arange(2.0, t[last] - T0 + 0.01, 0.25))
    for dt in times:
        i = int(np.searchsorted(t, T0 + dt))
        if i > last:
            break
        print("%5.2f %+6.2f %+6.2f %+6.2f | %+5.0f %+5.0f %5.1f | %s | %s | %s | %4.2f %-7s | %.2f" % (
            dt, roll[i], pitch[i], yaw[i], xy[i, 0] - xy[i0, 0], xy[i, 1] - xy[i0, 1], zt[i],
            " ".join("%4.0f" % v for v in fn[i]), "".join("%d" % (c > 0.5) for c in cw[i]),
            " ".join("%.2f" % v for v in ss[i]), tau[i].max(), JN[int(tau[i].argmax())], z["cap"][i]))
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    for th in (3, 5, 10, 15, 30):
        j = np.flatnonzero(tilt[i0:last + 1] > th)
        print("tilt>%2d deg first at T+%.2f s (roll %+.1f pitch %+.1f)" % (
            th, t[i0 + j[0]] - T0, roll[i0 + j[0]], pitch[i0 + j[0]]) if j.size else "tilt>%d never" % th)
    print("stop at T+%.2f s  (t=%.2f)" % (t[last] - T0, t[last]))
    hp = period / 2
    nwin = int((t[last] - T0) / hp)
    idx = np.arange(len(t))
    parts = []
    for k in range(min(nwin, 40)):
        m = (t >= T0 + k * hp) & (t < T0 + (k + 1) * hp) & (idx <= last)
        if not m.any():
            break
        r = roll[m]; p = pitch[m]
        parts.append("%+.1f/%+.1f/%.0f" % (r[np.argmax(np.abs(r))], p[np.argmax(np.abs(p))], zt[m].min()))
    print("per-half-cycle (roll_pk/pitch_pk/h_min):", " ".join(parts))


if __name__ == "__main__":
    top = ["Y95_h175", "WIDE_h175", "Y105_h160", "WIDE_h170", "Y95_h160", "X100_y95_h160", "X100_y85_h160", "PAR60_w85_160"]
    res = {}
    for n in top:
        for ck, per in (("P0.8_A40", 0.8), ("P0.5_A20", 0.5)):
            for pert in ("com_y_+10mm", "ideal"):
                f = os.path.join(D, "%s__%s_%s.npz" % (n, ck, pert))
                res["%s/%s/%s" % (n, ck, pert)] = analyse(f, per)
    for pert in ("com_y_+10mm", "ideal", "joint_fric_0.3"):
        res["FOLD_asis/P0.5_A20/%s" % pert] = analyse(os.path.join(D, "FOLD_asis__P0.5_A20_%s.npz" % pert), 0.5)
    for pert in ("ideal", "com_y_+5mm", "joint_fric_0.1"):
        res["FOLD_asis/P0.8_A40/%s" % pert] = analyse(os.path.join(D, "FOLD_asis__P0.8_A40_%s.npz" % pert), 0.8)
    json.dump(res, open(os.path.join(D, "ana_npz.json"), "w"), indent=1)
    keys = ["roll_mean", "roll_slow_max_abs", "roll_osc_amp", "roll_osc_rms", "roll_max_abs", "roll_amp_at_1/T", "roll_amp_at_2/T", "roll_amp_lowf",
            "roll_max_first3cyc", "roll_max_last3cyc", "pitch_mean", "pitch_max_abs", "walk_total_mm", "walk_rate_1st_half", "walk_rate_2nd_half",
            "yaw_drift_deg", "fn_peak_n", "fn_peak_leg", "fn_peak_after_td_ms", "fn_mean_2foot_n", "fn_99pct_n", "tau_peak", "tau_peak_joint", "tau_99pct", "tau_frac_gt6", "h_min_mm", "h_mean_mm", "cap_max"]
    print("%-34s" % "run" + " ".join("%13s" % k[:13] for k in keys))
    for k, v in res.items():
        print("%-34s" % k + " ".join("%13s" % (str(v.get(kk)) if kk in v else "-") for kk in keys))
    failure_timeline(os.path.join(D, "FOLD_asis__P0.8_A40_ideal.npz"), 0.8)
    failure_timeline(os.path.join(D, "FOLD_asis__P0.8_A40_com_y_+5mm.npz"), 0.8)
