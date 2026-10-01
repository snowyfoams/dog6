"""grid_refine -- off-grid centred candidates and the fixed references.

    python doc/trot_posture/scripts/grid_refine.py   -> data/grid_refine.json

For each (family, xf, y, h) in REFINE, bisect xr (continuous) until the CoM
sits on the support centre (|com - centre| x < 0.5 mm), then build the
spec fully (crouch search) and record the grid_sweep row.  The REFERENCES
are built as given.  Statics only.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import evalpose as E                                  # noqa: E402
import grid_sweep as G                                # noqa: E402

REFINE = [  # family, xf, y, h
    ("parallel", 80, 71, 145), ("parallel", 80, 65, 145), ("parallel", 80, 85, 145),
    ("parallel", 70, 71, 145), ("parallel", 60, 71, 145), ("parallel", 100, 71, 145),
    ("parallel", 80, 71, 125), ("parallel", 80, 71, 160), ("parallel", 60, 85, 160),
    ("parallel", 60, 65, 160), ("parallel", 70, 85, 160),
    ("xin", 80, 65, 145), ("xin", 60, 65, 160),
]
REFERENCES = [  # name, family, xf, xr, y, h, crouch_h
    ("FOLD_asis", "parallel", 79, 23, 71, 145, 60.0),
    ("NOMINAL", "x", 81, -81, 65, 145, None),
    ("WIDE", "x", 81, -81, 85, 145, 40.0),
    ("UNDER_x", "x", 0, 0, 65, 145, None),
]


def dx_of(fam, xf, xr, y, h):
    _, info = E.build(E.Spec("probe", fam, xf, xr, y, h, float(h)))
    return info["com_minus_centre_mm"][0], info


def centred_xr(fam, xf, y, h, lo=-150.0, hi=40.0):
    dlo, _ = dx_of(fam, xf, lo, y, h)
    dhi, _ = dx_of(fam, xf, hi, y, h)
    if dlo * dhi > 0:
        return None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        dm, _ = dx_of(fam, xf, mid, y, h)
        if dm * dlo > 0:
            lo, dlo = mid, dm
        else:
            hi, dhi = mid, dm
    return round(0.5 * (lo + hi), 1)


def main():
    rows = []
    for fam, xf, y, h in REFINE:
        xr = centred_xr(fam, xf, y, h)
        tag = "ref_%s_%d_c_%d_%d" % (fam[:3], xf, y, h)
        if xr is None:
            rows.append(dict(name=tag, family=fam, xf=xf, y=y, h=h, error="no centred xr in [-150, 40]"))
            print(tag, "no root")
            continue
        row = G.one(fam, xf, xr, y, h, name=tag)
        row["xr_centred"] = xr
        rows.append(row)
        print("%-24s xr %+6.1f  feas %d/%d dx %+.1f  diag peak %.2f (FLRR %s) 4ft %s crouch %s  stand min %.1f (%s)  reach F %.3f R %.3f" % (
            tag, xr, row["feasible_stand"], row["feasible_stand_8mm"], row["com_minus_centre_x_mm"],
            row["stand_diag_peak_nm"], row["stand_tau_diag_FL_RR_max_nm"], row["stand_tau_4foot_max_nm"],
            row["crouch_h_mm"], row["stand_min_mm"], row["stand_worst"], row["reach_used_F"], row["reach_used_R"]))
    for name, fam, xf, xr, y, h, hc in REFERENCES:
        row = G.one(fam, xf, xr, y, h, name=name, crouch_h=hc)
        row["crouch_h_given"] = hc
        rows.append(row)
        print("%-24s xr %+6.1f  feas %d/%d dx %+.1f  diag peak %.2f (FLRR %s) 4ft %s crouch %s (min %s)  stand min %.1f (%s)  reach F %.3f R %.3f" % (
            name, xr, row["feasible_stand"], row["feasible_stand_8mm"], row["com_minus_centre_x_mm"],
            row["stand_diag_peak_nm"], row["stand_tau_diag_FL_RR_max_nm"], row["stand_tau_4foot_max_nm"],
            row["crouch_h_mm"], row["crouch_min_mm"], row["stand_min_mm"], row["stand_worst"], row["reach_used_F"], row["reach_used_R"]))
    out = os.path.join(HERE, "..", "data", "grid_refine.json")
    json.dump(rows, open(out, "w"), indent=1, default=str)
    print("->", os.path.abspath(out))


if __name__ == "__main__":
    main()
