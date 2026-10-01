"""One line per simulated candidate: data/cand_*.json -> stdout table and data/cand_summary.json."""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
rows = []
for p in sorted(glob.glob(os.path.join(DATA, "cand_*.json"))):
    d = json.load(open(p))
    if not isinstance(d, dict) or "info" not in d:
        continue
    i = d["info"]; s = d.get("stand", {}); t = d.get("trots", {})
    sp = i["spec"]
    def surv(k):
        return t.get(k, {}).get("_survived", "-")
    def worst(k):
        v = [m.get("max_tilt_trot") for n, m in t.get(k, {}).items() if not n.startswith("_") and not m.get("fell")]
        return max(v) if v else None
    def ideal(k, key):
        return t.get(k, {}).get("ideal", {}).get(key)
    def walk(k):
        v = [np.hypot(*m.get("trot_xy_drift_mm", [0, 0])) for n, m in t.get(k, {}).items() if not n.startswith("_") and not m.get("fell")]
        return max(v) if v else None
    rows.append(dict(name=sp["name"], family=sp["family"], xf=sp["xf"], xr=sp["xr"], y=sp["y"], h=sp["h"],
                     crouch=i.get("crouch_h_mm"), floor_rest=(i.get("crouch_h_mm") == 0),
                     drift=s.get("drift_x_mm"), hold_tau=s.get("tau_hold_mean_max_nm"), rise_peak=s.get("tau_rise_peak_nm"),
                     diag_knee=i.get("stand_diag_FL_RR_peak_nm"), reach=max(i.get("reach_used_stand", [0])), knee=abs(i.get("q_stand_deg", [[0,0,0]])[0][2]), com_diag=i.get("com_to_diag_FL_RR_mm"),
                     clear_stand=i.get("collide_stand", {}).get("min_mm"), clear_crouch=i.get("collide_crouch", {}).get("min_mm"),
                     s05=surv("P0.5_A20"), s08=surv("P0.8_A40"), w05=worst("P0.5_A20"), w08=worst("P0.8_A40"),
                     i05=ideal("P0.5_A20", "max_tilt_trot"), i08=ideal("P0.8_A40", "max_tilt_trot"),
                     walk05=walk("P0.5_A20"), walk08=walk("P0.8_A40"),
                     peak05=max([m.get("peak_tau_trot", 0) or 0 for n, m in t.get("P0.5_A20", {}).items() if not n.startswith("_")] or [None]),
                     peak08=max([m.get("peak_tau_trot", 0) or 0 for n, m in t.get("P0.8_A40", {}).items() if not n.startswith("_")] or [None])))
def f(v, fmt):
    return ("%" + fmt) % v if v is not None else "   -"
print("%-14s %-8s %5s %5s %3s %3s %6s %6s %5s %5s | %5s %5s | %5s %5s %5s %5s | %4s %4s | %5s %5s" % (
    "name", "family", "xf", "xr", "y", "h", "crouch", "drift", "hold", "diagK", "s0.5", "s0.8", "w0.5", "w0.8", "i0.5", "i0.8", "wk05", "wk08", "pk05", "pk08"))
print("%-14s %-8s %5s %5s %3s %3s %6s %6s %5s %5s | reach knee" % ("", "", "", "", "", "", "", "", "", ""))
for r in sorted(rows, key=lambda r: (r["w08"] is None, r["w08"] or 99, r["w05"] or 99)):
    print("%-14s %-8s %5.0f %5.0f %3.0f %3.0f %6s %6s %5s %5s | %5s %5s | %5s %5s %5s %5s | %4s %4s | %5s %5s" % (
        r["name"], r["family"], r["xf"], r["xr"], r["y"], r["h"], f(r["crouch"], ".0f"), f(r["drift"], "+.1f"), f(r["hold_tau"], ".2f"), f(r["diag_knee"], ".2f"),
        r["s05"], r["s08"], f(r["w05"], ".2f"), f(r["w08"], ".2f"), f(r["i05"], ".2f"), f(r["i08"], ".2f"), f(r["walk05"], ".0f"), f(r["walk08"], ".0f"), f(r["peak05"], ".2f"), f(r["peak08"], ".2f")) + "  reach %.2f knee %3.0f" % (r["reach"], r["knee"]))
print("\ncolumns: crouch = lowest feasible crouch (mm, floor to belly); drift = trunk x shift over the rise (mm); hold = max mean |tau| in HOLD (N m); diagK = static peak torque on a diagonal pair (N m);\n s = perturbations survived of 13 at 0.5 s/20 mm and 0.8 s/40 mm; w = worst max tilt among survivors (deg); i = ideal-plant max tilt (deg); wk = worst trunk walk among survivors (mm); pk = peak torque in the trot (N m)")
json.dump(rows, open(os.path.join(DATA, "candidates_summary.json"), "w"), indent=1)
