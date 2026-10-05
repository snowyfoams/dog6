"""Where the built robot's CoM really is, read off a HOLD log.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.com_check fold_trot.npz                  a `hw.fold_trot --log` run
    $V -m hw.com_check fold2_trot.npz --entry fold2_trot
    $V -m hw.com_check trot.npz --entry trot --height 145

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

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import kinematics as HK       # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .balance import torque as TRQ   # noqa: E402

__all__ = ["com_from_hold", "pinned_srb", "main"]

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


def com_from_hold(q, tau_meas, roll, pitch) -> tuple[np.ndarray, float]:
    """``(c_xy, kt)`` for ONE sweep: the CoM in the LEVEL trunk frame (the
    trunk's roll and pitch taken out, yaw left in), m, and the measured
    vertical force over the weight.  `q`, `tau_meas` (12,); angles rad."""
    R = C.rot_y(float(pitch)) @ C.rot_x(float(roll))
    q4 = C.unflat(np.asarray(q, dtype=float))
    tau4 = C.unflat(np.asarray(tau_meas, dtype=float))
    grav = TRQ.all_leg_gravity_torque(np.asarray(q, dtype=float), R)
    M = np.zeros(3)
    F = np.zeros(3)
    for i in range(C.N_LEGS):
        jac = HK.foot_jacobian(i, q4[i])
        f_b = -np.linalg.solve(jac.T, tau4[i] - grav[i])
        f_w = R @ f_b
        r_w = R @ HK.foot_position(i, q4[i])
        M += np.cross(r_w, f_w)
        F += f_w
    fz = float(F[2])
    if abs(fz) < 1e-6:
        return np.full(2, np.nan), 0.0
    # sum r x f = c x (F_z z_hat) = F_z (c_y, -c_x, 0)
    return np.array([-M[1] / fz, M[0] / fz]), fz / BCFG.WEIGHT


def analyse(L, entry: str, height: float | None = None,
            phase: str = "hold") -> str:
    srb = pinned_srb(entry, height)
    ph = L["phase"]
    sel = np.flatnonzero(ph == phase)
    if not sel.size:
        return "no %r sweeps in this log (phases: %s)" % (
            phase, ", ".join(sorted(set(ph.tolist()))))
    finite = sel[np.isfinite(L["tau_meas"][sel]).all(axis=1)]
    c = np.array([com_from_hold(L["q"][k], L["tau_meas"][k],
                                L["roll"][k], L["pitch"][k])[0]
                  for k in finite])
    kt = np.array([com_from_hold(L["q"][k], L["tau_meas"][k],
                                 L["roll"][k], L["pitch"][k])[1]
                   for k in finite])
    ok = np.isfinite(c).all(axis=1)
    c, kt = c[ok], kt[ok]
    med = np.median(c, axis=0)
    p10, p90 = np.percentile(c, 10, axis=0), np.percentile(c, 90, axis=0)
    pin = np.asarray(srb.com_body, dtype=float)[:2]
    d = med - pin
    out = [
        "%d %s sweeps (%.1f s), %s" % (len(c), phase, len(c) / 250.0,
                                       srb.describe()),
        "  measured CoM, trunk x / y:  %+.1f / %+.1f mm  (p10..p90 %+.1f..%+.1f "
        "/ %+.1f..%+.1f)" % (1e3 * med[0], 1e3 * med[1], 1e3 * p10[0],
                             1e3 * p90[0], 1e3 * p10[1], 1e3 * p90[1]),
        "  pinned CoM, trunk x / y:    %+.1f / %+.1f mm" % (1e3 * pin[0],
                                                             1e3 * pin[1]),
        "  measured vertical force / weight: median %.2f (p10 %.2f, p90 %.2f) "
        "-- the drivers' torque constant, roughly; far from 1 means the "
        "torque readback or the gravity model is off, and the CoM above is a "
        "ratio that does not depend on it" % (np.median(kt),
                                               np.percentile(kt, 10),
                                               np.percentile(kt, 90)),
        "  roll / pitch over the phase: %+.2f / %+.2f deg mean -- an "
        "integrator-less attitude loop parks a few degrees off for a CoM "
        "off the pin" % (np.degrees(L["roll"][finite].mean()),
                         np.degrees(L["pitch"][finite].mean())),
        "",
        "  -> --com-offset %+.1f %+.1f   (measured minus pinned, mm)"
        % (1e3 * d[0], 1e3 * d[1]),
    ]
    if np.hypot(*d) < 1.0e-3:
        out.append("     under a millimetre: the pin is right; leave it")
    elif np.hypot(*(p90 - p10)) > 2.0 * np.hypot(*d):
        out.append("     the spread is larger than the offset: hold longer, "
                   "or check the torque readback before trusting this")
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
