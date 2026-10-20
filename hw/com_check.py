"""Where the built robot's CoM really is, read off a log -- EVERY HOLD in it.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.com_check fold_trot.npz                  a `hw.fold_trot --log` run
    $V -m hw.com_check fold2_trot.npz --entry fold2_trot
    $V -m hw.com_check trot.npz --entry trot --height 145

SINCE 2026-10-06 THE RUN DOES THIS ITSELF (`--com-auto`, the default on every
trot that tracks its rise): each HOLD measures the CoM and T applies it
(`hw.balance.com_estimate`, where `com_from_hold` now lives).  This script
reads the same thing back off the npz, one row per HOLD -- the operator found
each HOLD gives its own -- beside the pin the law flew (`com_pin`) and each
T's re-centring.

WHY.  The balance law pins the whole-robot CoM at the CAD's number
(`config.SrbModel`, `posture.CrouchPose.srb_at`) and never measures it.  A
battery, a cable loom or a bracket the CAD does not carry moves the real one
by millimetres, and the attitude loop has no integrator to absorb that: it
parks the trunk a few degrees off level, the two diagonals load unequally,
and the trot in place rocks about the heavier one.  MuJoCo, 2026-10-05
(doc/walk/README.md, "the CoM error"): a 3 / 5 mm x / y error ALONE takes
`hw.fold_trot` in place from 2.3 deg of tilt to 9.7 with the tilt rms
growing 2.3 -> 4.8 -> 4.9 deg over three 10 s windows, and fells every
`hw.fold_walk` configuration within 5 s; every other hardware unknown tried
(rotor 0.6x, 0.1 N*m joint friction + IMU 15 ms, floor mu 0.7, 10 % less
torque per amp) costs under a degree.  `hw.stand --com-offset DX DY` shifts
the pin; this script says by how much.

WHAT IT COMPUTES.  In HOLD the robot is static on four feet, so the ground
reactions balance the weight about the CoM.  Each foot's force comes from
the torque the drivers MEASURED (`tau_meas`, the q-axis current) through
the leg's Jacobian, less the leg's own weight -- `torque.stance_torque`
run backwards:

    f_i^b = -(J_i^T)^-1 (tau_i - g_i(q_i, R))         trunk frame
    f_i^w = R f_i^b                                   world, R from roll/pitch
    sum_i r_i x f_i = c x W z_hat   ->   c_x = -M_y / F_z,   c_y = M_x / F_z

with r_i the feet from the trunk origin (FK of the encoders) and F_z the
measured vertical total, which also says what fraction of the weight the
drivers' torque constant accounts for (`kt`).  Every HOLD sweep gives one
estimate; the median over the phase is the answer, the spread its noise.
What is NOT in it: joint friction (a Coulomb term in `tau_meas` that is not
a foot force), and a driver whose reported current is off.  Both show up
as spread, and as a `kt` far from 1.

OUTPUT: the measured CoM against the entry point's pin, and the flag to
pass.  The sign is the one `--com-offset` takes: measured minus pinned.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/com_check.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import argparse                      # noqa: E402

import numpy as np                   # noqa: E402

from .balance import com_estimate as COM  # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import posture as POSE  # noqa: E402

#: One sweep's statics: `hw.balance.com_estimate`'s, which the run uses too.
com_from_hold = COM.com_from_hold

__all__ = ["com_from_hold", "pinned_srb", "segments", "hold_reading",
           "analyse", "main"]

#: entry point -> (crouch, default hold height m, stand feet or None)
ENTRIES = {
    "fold_trot": lambda: _fold_trot(),
    "fold2_trot": lambda: _fold2_trot(),
    "trot": lambda: (POSE.NOMINAL, BCFG.H_LIFT, None),
}


def _fold_trot():
    from . import fold_trot as FT
    return POSE.FOLD, FT.HEIGHT, FT.STAND_XY


def _fold2_trot():
    from . import fold2_trot as F2
    return POSE.FOLD2, F2.HEIGHT, None


def pinned_srb(entry: str, height: float | None = None) -> BCFG.SrbModel:
    """The SRB model `entry` pins, at `height` (m, floor to trunk bottom;
    None is the entry point's own)."""
    crouch, h, stand_xy = ENTRIES[entry]()
    return crouch.srb_at(h if height is None else float(height), stand_xy)


def segments(phase_col, name: str) -> list[tuple[int, int]]:
    """``[(i0, i1), ...]``: each contiguous run of `name` in the log's phase
    column, as sweep index ranges (i1 exclusive)."""
    on = (np.asarray(phase_col) == name).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on, [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def hold_reading(L, i0: int, i1: int, z_b: float,
                 settle_s: float = BCFG.COM_SETTLE_S) -> dict:
    """One HOLD's CoM, read off the log as the run read it (`com_estimate`):
    trunk-frame x/y per sweep after the settle, its median and p10/p90 (m),
    the median Fz/W, the mean roll/pitch (rad), and the sweep count."""
    idx = np.arange(i0, i1)
    idx = idx[(L["t"][idx] >= settle_s)
              & np.isfinite(L["tau_meas"][idx]).all(axis=1)]
    c, kt = [], []
    for k in idx:
        cl, kk = com_from_hold(L["q"][k], L["tau_meas"][k], L["roll"][k],
                               L["pitch"][k])
        if np.all(np.isfinite(cl)):
            c.append(COM.level_to_trunk_xy(cl, L["roll"][k], L["pitch"][k],
                                           z_b))
            kt.append(kk)
    if not c:
        return dict(n=0, idx=idx)
    c, kt = np.array(c), np.array(kt)
    return dict(n=len(c), idx=idx, c=np.median(c, axis=0),
                p10=np.percentile(c, 10, axis=0),
                p90=np.percentile(c, 90, axis=0), kt=float(np.median(kt)),
                roll=float(np.mean(L["roll"][idx])),
                pitch=float(np.mean(L["pitch"][idx])))


def analyse(L, entry: str, height: float | None = None,
            phase: str = "hold") -> str:
    """Every HOLD in the log, one row each, against the pin the law FLEW
    there (`com_pin`, a `--com-auto` / `--com-offset` run's own) or, in a log
    from before 2026-10-06, the entry point's; then each T's re-centring."""
    srb = pinned_srb(entry, height)
    cad = np.asarray(srb.com_body, dtype=float)[:2]
    z_b = float(srb.com_body[2])
    ph = L["phase"]
    segs = segments(ph, phase)
    if not segs:
        return "no %r sweeps in this log (phases: %s)" % (
            phase, ", ".join(sorted(set(ph.tolist()))))
    pins = L["com_pin"] if "com_pin" in L.files else None
    online = L["com_meas"] if "com_meas" in L.files else None
    out = ["%s -- the entry point's pin at that height" % srb.describe(),
           "each %s from %.1f s in (the settle the run skips too), trunk "
           "frame, mm:" % (phase.upper(), BCFG.COM_SETTLE_S),
           "   #  starts   s     CoM x / y     p10..p90 x / y            Fz/W  "
           "roll/pitch deg   pin flown x / y   measured - flown"]
    last = None
    for j, (i0, i1) in enumerate(segs, 1):
        r = hold_reading(L, i0, i1, z_b)
        if r["n"] == 0:
            out.append("  %2d  sweep %d: under %.1f s, nothing after the "
                       "settle" % (j, i0, BCFG.COM_SETTLE_S))
            continue
        pin = cad
        if pins is not None and np.isfinite(pins[r["idx"]]).all():
            pin = np.median(pins[r["idx"]], axis=0)
        d = r["c"] - pin
        out.append("  %2d  %6d %4.1f  %+6.1f %+6.1f  %+6.1f..%+6.1f / "
                   "%+5.1f..%+5.1f  %4.2f  %+5.2f/%+5.2f    %+6.1f %+6.1f    "
                   "%+5.1f %+5.1f"
                   % (j, i0, r["n"] / 250.0, *(1e3 * r["c"]),
                      1e3 * r["p10"][0], 1e3 * r["p90"][0], 1e3 * r["p10"][1],
                      1e3 * r["p90"][1], r["kt"], np.degrees(r["roll"]),
                      np.degrees(r["pitch"]), *(1e3 * pin), *(1e3 * d)))
        if online is not None:
            got = online[r["idx"]]
            got = got[np.isfinite(got).all(axis=1)]
            if got.size:
                out.append("        the run's own reading (com_meas): %+.1f "
                           "%+.1f" % tuple(1e3 * np.median(got, axis=0)))
        last = r
    if pins is not None:
        rc = segments(ph, "recentre")
        if rc:
            out.append("")
            out.append("re-centred at T (the pin flown, before -> after the "
                       "slide), mm:")
            for i0, i1 in rc:
                before = pins[max(0, i0 - 1)]
                after = pins[min(len(pins) - 1, i1)]
                out.append("  sweep %6d, %.1f s: %+.1f/%+.1f -> %+.1f/%+.1f  "
                           "(%+.1f/%+.1f)"
                           % (i0, (i1 - i0) / 250.0, *(1e3 * before),
                              *(1e3 * after), *(1e3 * (after - before))))
    out.append("")
    out.append("  measured vertical force / weight is the drivers' torque "
               "constant, roughly: far from 1 means the torque readback or "
               "the gravity model is off.  The CoM is a ratio and does not "
               "depend on it.")
    if last is not None:
        d = last["c"] - cad
        out.append("  -> the LAST hold, against the CAD's pin: --com-offset "
                   "%+.1f %+.1f  (by hand, with --no-com-auto)"
                   % tuple(1e3 * d))
        if np.hypot(*(last["p90"] - last["p10"])) > 2.0 * np.hypot(*d):
            out.append("     its spread is larger than the offset: hold "
                       "longer, or check the torque readback")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="the built robot's CoM from a HOLD log's measured torques")
    ap.add_argument("log", help="an hw.stand --log FILE.npz")
    ap.add_argument("--entry", choices=sorted(ENTRIES), default="fold_trot",
                    help="whose pin to compare against (default fold_trot)")
    ap.add_argument("--height", type=float, default=None, metavar="MM",
                    help="the hold height the run used, if not the entry "
                         "point's default")
    ap.add_argument("--phase", default="hold",
                    help="which phase's sweeps to read (default hold)")
    args = ap.parse_args(argv)
    L = np.load(args.log, allow_pickle=True)
    print(analyse(L, args.entry,
                  None if args.height is None else 1e-3 * args.height,
                  args.phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
