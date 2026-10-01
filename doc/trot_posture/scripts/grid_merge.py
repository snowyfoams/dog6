"""grid_merge -- merge grid_shard_*.json into grid_table.json / grid_table.csv
and analyse: per family and (y, h) the centred line in the (xf, xr) plane,
the best torque / crouch / clearance along it, and infeasible regions.

    python doc/trot_posture/scripts/grid_merge.py            -> data/grid_table.{json,csv}
                                                                data/grid_analysis.json
"""
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

CSV_COLS = ["name", "family", "xf", "xr", "y", "h", "feasible_stand", "feasible_stand_8mm",
            "stand_reachable", "family_ok", "stand_soft_limit_margin_deg",
            "stand_min_mm", "stand_leg_floor_mm", "stand_leg_trunk_mm", "stand_leg_leg_mm",
            "stand_penetrating", "stand_worst",
            "crouch_h_mm", "crouch_zero_torque", "crouch_min_mm", "crouch_leg_floor_mm",
            "crouch_leg_trunk_mm", "crouch_worst",
            "stand_tau_4foot_peak_nm", "stand_diag_FL_RR_peak_nm", "stand_diag_FR_RL_peak_nm",
            "stand_diag_peak_nm", "stand_diag_knee_max_nm", "stand_front_share",
            "com_to_diag_FL_RR_mm", "com_to_diag_FR_RL_mm", "com_minus_centre_x_mm", "centred",
            "wheelbase_mm", "track_mm", "diag_angle_deg", "reach_used_F", "reach_used_R",
            "stand_knee_rel_hinge_x_F_mm", "stand_knee_rel_hinge_x_R_mm",
            "stand_knee_motor_floor_clear_F_mm", "stand_knee_motor_floor_clear_R_mm",
            "crouch_knee_rel_hinge_x_F_mm", "crouch_knee_rel_hinge_x_R_mm",
            "crouch_knee_motor_floor_clear_F_mm", "crouch_knee_motor_floor_clear_R_mm",
            "inertia_diag", "stand_tau_4foot_max_nm", "stand_tau_diag_FL_RR_max_nm",
            "stand_tau_diag_FR_RL_max_nm", "crouch_tau_4foot_max_nm", "build_s"]


def merge():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "grid_shard_*.json"))):
        rows += json.load(open(f))
    order = {"parallel": 0, "x": 1, "xin": 2}
    rows.sort(key=lambda r: (order[r["family"]], r["h"], r["y"], r["xf"], r["xr"]))
    json.dump(rows, open(os.path.join(DATA, "grid_table.json"), "w"), indent=0, default=str)
    with open(os.path.join(DATA, "grid_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()
                        if k in CSV_COLS})
    return rows


def fmt(r):
    return ("%-9s xf %+4d xr %+4d y %d h %d | centre dx %+6.1f diag %+6.1f | diag peak %.2f "
            "(knee %.2f) 4ft %.2f | crouch %s%s | stand min %.1f (%s) | share %.2f | wb %.0f" % (
                r["family"], r["xf"], r["xr"], r["y"], r["h"], r["com_minus_centre_x_mm"],
                r["com_to_diag_FL_RR_mm"], r["stand_diag_peak_nm"], r["stand_diag_knee_max_nm"],
                r["stand_tau_4foot_peak_nm"],
                "none" if r["crouch_h_mm"] is None else "%.0f" % r["crouch_h_mm"],
                " (floor rest)" if r.get("crouch_zero_torque") else "",
                r["stand_min_mm"], r["stand_worst"], r["stand_front_share"], r["wheelbase_mm"]))


def short_pair(worst):
    if not worst:
        return "?"
    a, b = (worst.split(" <-> ") + ["?"])[:2]
    return a.split(":")[0] + "|" + b.split(":")[0]


def analyse(rows):
    out = {"n_specs": len(rows)}
    ok = [r for r in rows if not r.get("error")]
    out["n_errors"] = len(rows) - len(ok)
    feas = [r for r in ok if r["feasible_stand"]]
    feas8 = [r for r in ok if r["feasible_stand_8mm"]]
    out["n_feasible_stand"] = len(feas)
    out["n_feasible_stand_8mm"] = len(feas8)
    out["n_feasible_with_crouch"] = sum(1 for r in feas if r["crouch_h_mm"] is not None)
    out["n_floor_rest"] = sum(1 for r in feas if r["crouch_zero_torque"])
    cen = [r for r in feas8 if r["centred"]]
    out["n_centred_feasible8"] = len(cen)
    reasons = defaultdict(lambda: defaultdict(int))
    for r in ok:
        if r["feasible_stand"]:
            continue
        f = r["family"]
        if not r["stand_reachable"]:
            reasons[f]["unreachable"] += 1
        elif not r["family_ok"]:
            reasons[f]["wrong_knee_branch"] += 1
        elif r["stand_soft_limit_margin_deg"] <= 0:
            reasons[f]["soft_limit"] += 1
        elif r["stand_penetrating"]:
            reasons[f]["penetrating:" + short_pair(r["stand_worst"])] += 1
        else:
            reasons[f]["clearance<5:" + short_pair(r["stand_worst"])] += 1
    out["infeasible_reasons"] = {f: dict(v) for f, v in reasons.items()}
    cnt = defaultdict(lambda: [0, 0, 0, 0])
    for r in ok:
        k = "%s y%d h%d" % (r["family"], r["y"], r["h"])
        cnt[k][0] += int(r["feasible_stand"])
        cnt[k][1] += int(r["feasible_stand_8mm"])
        cnt[k][2] += int(r["feasible_stand_8mm"] and r["centred"])
        cnt[k][3] += int(r["feasible_stand"] and r["crouch_zero_torque"])
    out["counts_by_cell"] = dict(cnt)
    env = {}
    for f in ("parallel", "x", "xin"):
        for h in (125, 145, 160):
            sel = [r for r in feas if r["family"] == f and r["h"] == h]
            if sel:
                env["%s h%d" % (f, h)] = dict(
                    xf=[min(r["xf"] for r in sel), max(r["xf"] for r in sel)],
                    xr=[min(r["xr"] for r in sel), max(r["xr"] for r in sel)],
                    n=len(sel), n_grid=len([r for r in ok if r["family"] == f and r["h"] == h]))
    out["feasible_envelope"] = env
    lines = {}
    for f in ("parallel", "x", "xin"):
        for h in (125, 145, 160):
            for y in (65, 75, 85):
                key = "%s y%d h%d" % (f, y, h)
                sel = [r for r in ok if r["family"] == f and r["h"] == h and r["y"] == y]
                line = []
                for xf in sorted(set(r["xf"] for r in sel)):
                    col = [r for r in sel if r["xf"] == xf]
                    best = min(col, key=lambda r: abs(r["com_minus_centre_x_mm"]))
                    col.sort(key=lambda r: r["xr"])
                    xr_c = None
                    for a, b in zip(col, col[1:]):
                        da, db = a["com_minus_centre_x_mm"], b["com_minus_centre_x_mm"]
                        if da == 0:
                            xr_c = a["xr"]
                        elif da * db < 0:
                            xr_c = a["xr"] + (b["xr"] - a["xr"]) * da / (da - db)
                    line.append(dict(xf=xf, xr_nearest=best["xr"], dx=best["com_minus_centre_x_mm"],
                                     xr_centred_interp=None if xr_c is None else round(xr_c, 1),
                                     feasible8=best["feasible_stand_8mm"], centred=best["centred"],
                                     diag_peak=best["stand_diag_peak_nm"],
                                     diag_knee=best["stand_diag_knee_max_nm"],
                                     crouch=best["crouch_h_mm"], stand_min=best["stand_min_mm"],
                                     name=best["name"]))
                cands = [r for r in sel if r["feasible_stand_8mm"] and r["centred"]]
                pick = {}
                if cands:
                    pick["lowest_diag_knee"] = fmt(min(cands, key=lambda r: r["stand_diag_knee_max_nm"]))
                    pick["lowest_diag_peak"] = fmt(min(cands, key=lambda r: r["stand_diag_peak_nm"]))
                    wc = [r for r in cands if r["crouch_h_mm"] is not None]
                    if wc:
                        pick["lowest_crouch"] = fmt(min(wc, key=lambda r: r["crouch_h_mm"]))
                    pick["best_stand_clearance"] = fmt(max(cands, key=lambda r: r["stand_min_mm"]))
                lines[key] = dict(line=line, n_centred_feasible8=len(cands), picks=pick,
                                  centred_candidates=[fmt(r) for r in sorted(
                                      cands, key=lambda r: r["stand_diag_knee_max_nm"])])
    out["centred_lines"] = lines
    return out


def main():
    rows = merge()
    an = analyse(rows)
    json.dump(an, open(os.path.join(DATA, "grid_analysis.json"), "w"), indent=1)
    print("specs %d  errors %d  feasible %d  feasible(8mm) %d  with crouch %d  floor rest %d  centred&feasible8 %d" % (
        an["n_specs"], an["n_errors"], an["n_feasible_stand"], an["n_feasible_stand_8mm"],
        an["n_feasible_with_crouch"], an["n_floor_rest"], an["n_centred_feasible8"]))
    print("\ninfeasible reasons:")
    for f, v in an["infeasible_reasons"].items():
        print("  %-9s %s" % (f, dict(sorted(v.items(), key=lambda kv: -kv[1]))))
    print("\nfeasible envelope:")
    for k, v in an["feasible_envelope"].items():
        print("  %-14s %s" % (k, v))
    print("\ncounts (feasible, feasible8, centred8, floor_rest):")
    for k, v in an["counts_by_cell"].items():
        print("  %-20s %s" % (k, v))
    print("\ncentred lines (xf -> xr nearest centre [dx | interpolated centred xr], * feasible8 / x not):")
    for k, v in an["centred_lines"].items():
        print("  %s  (centred & feasible8: %d)" % (k, v["n_centred_feasible8"]))
        print("    " + "  ".join("%+d->%+d[%+.0f|%s]%s" % (
            p["xf"], p["xr_nearest"], p["dx"],
            "-" if p["xr_centred_interp"] is None else "%.0f" % p["xr_centred_interp"],
            "*" if p["feasible8"] else "x") for p in v["line"]))
        for kk, vv in v["picks"].items():
            print("    %-20s %s" % (kk, vv))


if __name__ == "__main__":
    main()
