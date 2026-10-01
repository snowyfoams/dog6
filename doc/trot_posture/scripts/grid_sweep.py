"""grid_sweep -- statics-only sweep of the stand-posture design space.

    python doc/trot_posture/scripts/grid_sweep.py --shard I --nshards N

Builds every spec of the grid (family x xf x xr x y x h) with
evalpose.build (IK, lowest feasible crouch, mesh collisions, static torques,
support geometry) and writes doc/trot_posture/data/grid_shard_I.json.
grid_merge.py merges the shards into grid_table.json / .csv and analyses.
No dynamic simulation.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, TP)
import evalpose as E                                  # noqa: E402

FAMILIES = ("parallel", "x", "xin")
XF = (-20, 0, 20, 40, 60, 80, 100)
XR = (-100, -80, -60, -40, -20, 0, 20, 40)
Y = (65, 75, 85)
H = (125, 145, 160)
EXPECT_KD = {"parallel": [-1, -1, -1, -1], "x": [1, 1, -1, -1], "xin": [-1, -1, 1, 1]}


def grid():
    for fam in FAMILIES:
        for h in H:
            for y in Y:
                for xf in XF:
                    for xr in XR:
                        yield fam, xf, xr, y, h


def name_of(fam, xf, xr, y, h):
    s = lambda v: ("m%g" % -v) if v < 0 else "%g" % v
    return "%s_%s_%s_%s_%s" % (fam[:3], s(xf), s(xr), y, h)


def knees(rows, tag):
    out = {}
    for lab, i in (("F", 0), ("R", 2)):
        r = rows[i]
        out["%s_knee_rel_hinge_x_%s_mm" % (tag, lab)] = r["knee_rel_hinge_x_mm"]
        out["%s_knee_hinge_floor_%s_mm" % (tag, lab)] = r["knee_hinge_above_floor_mm"]
        out["%s_knee_motor_floor_clear_%s_mm" % (tag, lab)] = r["knee_motor_floor_clear_mm"]
        out["%s_knee_motor_abd_clear_%s_mm" % (tag, lab)] = r["knee_motor_abd_clear_mm"]
        out["%s_knee_motor_box_clear_%s_mm" % (tag, lab)] = r["knee_motor_box_clear_mm"]
    return out


def one(fam, xf, xr, y, h, name=None, crouch_h=None):
    t0 = time.time()
    name = name or name_of(fam, xf, xr, y, h)
    spec = E.Spec(name, fam, xf, xr, y, h, crouch_h)
    pose, info = E.build(spec)
    crouch_found = pose is not None
    if not crouch_found:
        # no feasible crouch: build with crouch = stand so the stand statics exist
        pose, info = E.build(E.Spec(name, fam, xf, xr, y, h, float(h)))
    kd = [int(v) for v in info["knee_dir"]]
    fam_ok = kd == EXPECT_KD[fam]
    cs = info["collide_stand"]
    cc = info["collide_crouch"]
    row = dict(name=name, family=fam, xf=xf, xr=xr, y=y, h=h,
               stand_reachable=info["stand_reachable"], knee_dir=kd, family_ok=fam_ok,
               stand_soft_limit_margin_deg=info["stand_soft_limit_margin_deg"],
               stand_min_mm=cs["min_mm"], stand_worst=cs["worst"],
               stand_leg_floor_mm=cs["leg_floor_mm"], stand_leg_trunk_mm=cs["leg_trunk_mm"],
               stand_leg_leg_mm=cs["leg_leg_mm"], stand_penetrating=bool(cs["penetrating"]),
               stand_penetrating_pairs=cs["penetrating"])
    row["feasible_stand"] = bool(info["stand_reachable"] and not cs["penetrating"]
                                 and cs["min_mm"] >= 5.0 and fam_ok
                                 and info["stand_soft_limit_margin_deg"] > 0)
    row["feasible_stand_8mm"] = bool(row["feasible_stand"] and cs["min_mm"] >= 8.0)
    if crouch_found:
        hc = info["crouch_h_mm"]
        row.update(crouch_h_mm=hc, crouch_zero_torque=bool(hc <= 0.5),
                   crouch_min_mm=cc["min_mm"], crouch_worst=cc["worst"],
                   crouch_leg_floor_mm=cc["leg_floor_mm"], crouch_leg_trunk_mm=cc["leg_trunk_mm"],
                   crouch_leg_leg_mm=cc["leg_leg_mm"],
                   crouch_soft_limit_margin_deg=info["crouch_soft_limit_margin_deg"],
                   crouch_tau_4foot_max_nm=info["crouch_tau_4foot_max_nm"],
                   q_crouch_deg=info["q_crouch_deg"])
        row.update(knees(info["knees_crouch"], "crouch"))
    else:
        row.update(crouch_h_mm=None, crouch_zero_torque=False, crouch_min_mm=None,
                   crouch_worst=None, crouch_leg_floor_mm=None, crouch_leg_trunk_mm=None,
                   crouch_leg_leg_mm=None, crouch_soft_limit_margin_deg=None,
                   crouch_tau_4foot_max_nm=None, q_crouch_deg=None)
        for k in knees(info["knees_crouch"], "crouch"):
            row[k] = None
    row.update(knees(info["knees_stand"], "stand"))
    row.update(stand_tau_4foot_max_nm=info["stand_tau_4foot_max_nm"],
               stand_tau_4foot_peak_nm=max(info["stand_tau_4foot_max_nm"]),
               stand_tau_diag_FL_RR_max_nm=info["stand_tau_diag_FL_RR_max_nm"],
               stand_tau_diag_FR_RL_max_nm=info["stand_tau_diag_FR_RL_max_nm"],
               stand_diag_FL_RR_peak_nm=info["stand_diag_FL_RR_peak_nm"],
               stand_diag_FR_RL_peak_nm=info["stand_diag_FR_RL_peak_nm"],
               stand_diag_peak_nm=max(info["stand_diag_FL_RR_peak_nm"], info["stand_diag_FR_RL_peak_nm"]),
               stand_diag_knee_max_nm=max(info["stand_tau_diag_FL_RR_max_nm"][2],
                                          info["stand_tau_diag_FR_RL_max_nm"][2]),
               stand_diag_FL_RR_residual_nm=info["stand_diag_FL_RR_residual_nm"],
               stand_diag_FR_RL_residual_nm=info["stand_diag_FR_RL_residual_nm"],
               stand_fz_4foot_n=info["stand_fz_4foot_n"],
               stand_front_share=info["stand_front_share"],
               com_mm=info["com_mm"],
               com_to_diag_FL_RR_mm=info["com_to_diag_FL_RR_mm"],
               com_to_diag_FR_RL_mm=info["com_to_diag_FR_RL_mm"],
               com_minus_centre_mm=info["com_minus_centre_mm"],
               com_minus_centre_x_mm=info["com_minus_centre_mm"][0],
               centred=bool(abs(info["com_minus_centre_mm"][0]) < 3.0
                            and abs(info["com_minus_centre_mm"][1]) < 3.0),
               inertia_diag=info["inertia_diag"],
               wheelbase_mm=info["wheelbase_mm"], track_mm=info["track_mm"],
               diag_angle_deg=info["diag_angle_deg"],
               reach_used_stand=info["reach_used_stand"],
               reach_used_F=info["reach_used_stand"][0], reach_used_R=info["reach_used_stand"][2],
               q_stand_deg=info["q_stand_deg"],
               build_s=round(time.time() - t0, 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="fam xf xr y h (one spec)")
    a = ap.parse_args()
    specs = list(grid())
    if a.only:
        fam, xf, xr, y, h = a.only[0], int(a.only[1]), int(a.only[2]), int(a.only[3]), int(a.only[4])
        print(json.dumps(one(fam, xf, xr, y, h), indent=1))
        return
    mine = specs[a.shard::a.nshards]
    rows = []
    t0 = time.time()
    for k, s in enumerate(mine):
        try:
            rows.append(one(*s))
        except Exception as ex:          # record, keep going
            rows.append(dict(name=name_of(*s), family=s[0], xf=s[1], xr=s[2], y=s[3], h=s[4],
                             error=repr(ex), feasible_stand=False))
        if k % 20 == 0:
            print("shard %d: %d/%d  %.0f s" % (a.shard, k + 1, len(mine), time.time() - t0),
                  file=sys.stderr, flush=True)
    out = os.path.join(TP, "data", "grid_shard_%d.json" % a.shard)
    json.dump(rows, open(out, "w"), indent=0, default=str)
    print("shard %d done: %d rows in %.0f s -> %s" % (a.shard, len(rows), time.time() - t0, out))


if __name__ == "__main__":
    main()
