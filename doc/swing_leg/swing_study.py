"""One swing leg, trunk pinned: the simulator's actuation chain against the robot's.

    D:\\mujoco\\.venv\\Scripts\\python.exe doc\\swing_leg\\swing_study.py

Prints every table in doc/dog6_swing_leg.tex and writes its figures to
doc/swing_leg/fig/.  The leg is FL at the nominal lift pose, its dynamics the
repo's own RNEA (`sim.leg_dynamics`, armature in); the swing law is the repo's
own (`hw.balance.swing`: the arc, the Cartesian PD, the feedforward).  What
this file adds is the chain between the law and the joint:

    SIM  the law sees this sweep's q and the true qd; torque is held 4 ms
    HW   the law sees last sweep's q (one 4 ms sweep old) and a finite-
         differenced qd through the runner's 40 Hz low-pass; the request is
         clipped at 9 N*m and rate-limited at `slew` N*m/s, as the gate does

No floor: "z at touchdown" is how high the foot still is when the clock
hands it the load.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hw import kinematics as HK            # noqa: E402
from hw.balance import config as cfg       # noqa: E402
from hw.balance import swing as S          # noqa: E402
from sim import leg_dynamics as LD         # noqa: E402
from sim import params as P                # noqa: E402

LEG = 0
Q0 = np.asarray(cfg.NOMINAL_POSE, float)[LEG]
HIP = np.asarray(P.HIP_OFFSET[LEG], float)
DT_PHYS = 2.5e-4
SWEEP = 0.004
LPF_HZ = 40.0
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")


# ---------------------------------------------------------------------------
# the two arcs
# ---------------------------------------------------------------------------
def jerk_table(T, n=4000):
    """Bang-bang jerk, whole swing: +j, -j, +j, -j with switches at t1, T/2,
    T - t1, t1 = (1 - 1/sqrt 2) T / 2.  Zero v and a at both ends, symmetric.
    Returns (t, z, v, a) for UNIT jerk."""
    t1 = (1.0 - 1.0 / np.sqrt(2.0)) / 2.0 * T
    ts = np.linspace(0.0, T, n + 1)
    dt = T / n
    j = np.where(ts < t1, 1.0, np.where(ts < T / 2, -1.0,
                 np.where(ts < T - t1, 1.0, -1.0)))
    a = np.concatenate([[0.0], np.cumsum((j[:-1] + j[1:]) / 2 * dt)])
    v = np.concatenate([[0.0], np.cumsum((a[:-1] + a[1:]) / 2 * dt)])
    z = np.concatenate([[0.0], np.cumsum((v[:-1] + v[1:]) / 2 * dt)])
    return ts, z, v, a


_JT = {}


def arc_pva(x0, s, T, H, arc="quintic"):
    """(p, v, a) at progress s, trunk frame, start == end == x0."""
    if arc == "quintic":
        return S.swing_reference_pva(x0, float(s), T, height=H)
    key = (T, H)
    if key not in _JT:
        ts, z, v, a = jerk_table(T)
        k = H / z.max()
        _JT[key] = (ts, z * k, v * k, a * k)
    ts, z, v, a = _JT[key]
    t = min(max(s, 0.0), 1.0) * T
    p = np.array(x0, float)
    p[2] += np.interp(t, ts, z)
    return p, np.array([0, 0, np.interp(t, ts, v)]), np.array([0, 0, np.interp(t, ts, a)])


def demand(T, H, arc="quintic", n=400):
    """Feedforward torque along the arc: (s, tau (n+1, 3), dtau/dt (n+1, 3))."""
    x0 = HK.foot_position(LEG, Q0)
    s = np.linspace(0.0, 1.0, n + 1)
    tau = []
    for x in s:
        p, v, a = arc_pva(x0, x, T, H, arc)
        q = HK.leg_ik(LEG, p - HIP, q_seed=Q0)
        tau.append(S.swing_feedforward(LEG, q, v, a))
    tau = np.array(tau)
    return s, tau, np.gradient(tau, T / n, axis=0)


def slew_needed(T, H, arc="quintic"):
    _, tau, rate = demand(T, H, arc)
    return float(np.abs(rate).max()), float(np.abs(tau).max())


# ---------------------------------------------------------------------------
# the one-leg run
# ---------------------------------------------------------------------------
def run(T, H, *, hw=True, slew=None, ff=True, kp=(140, 140, 180), kd=(20, 20, 40),
        kd_hold=0.0, hold_relative=False, arm_plant=None, ff_arm=None,
        cap=9.0, arc="quintic", pre=0.05, post=0.10):
    kp = np.asarray(kp, float)
    kd = np.asarray(kd, float)
    inertia = None if ff_arm is None else S.feedforward_inertia(ff_arm)
    arm = P.ARMATURE if arm_plant is None else arm_plant
    q, qd = Q0.copy(), np.zeros(3)
    x0 = HK.foot_position(LEG, q)
    tau_sent = LD.gravity_torque(LEG, q).copy()
    q_prev, qd_f = q.copy(), np.zeros(3)
    last = (q.copy(), np.zeros(3))
    t, next_sweep = -pre, -pre
    rows = []
    while t < T + post:
        if t >= next_sweep - 1e-12:
            q_now = q.copy()
            if hw:
                fd = (q_now - q_prev) / SWEEP
                qd_f = qd_f + (1 - np.exp(-2 * np.pi * LPF_HZ * SWEEP)) * (fd - qd_f)
                use_q, use_qd = last
                last = (q_now, qd_f.copy())
            else:
                use_q, use_qd = q_now, qd.copy()
            q_prev = q_now
            s = min(max(t / T, 0.0), 1.0)
            swinging = 0.0 <= t <= T
            jac = HK.foot_jacobian(LEG, use_q)
            x = HK.foot_position(LEG, use_q)
            req = LD.gravity_torque(LEG, use_q).copy()
            p, v, a = arc_pva(x0, s, T, H, arc) if swinging else (x0, np.zeros(3), np.zeros(3))
            req += jac.T @ (kp * (p - x) + kd * (v - jac @ use_qd))
            if swinging and ff:
                req += S.swing_feedforward(LEG, use_q, v, a, jac=jac, inertia=inertia)
            if swinging and kd_hold:
                qd_ref = np.zeros(3)
                if hold_relative:
                    j2 = jac[:, 1:]
                    qd_ref[1:] = np.linalg.solve(j2.T @ j2, j2.T @ v)
                req += -kd_hold * (use_qd - qd_ref)
            req = np.clip(req, -cap, cap)
            if slew is None:
                tau_sent = req
            else:
                step = slew * SWEEP
                tau_sent = tau_sent + np.clip(req - tau_sent, -step, step)
            rows.append(np.concatenate([[t], HK.foot_position(LEG, q), p, req, tau_sent]))
            next_sweep += SWEEP
        M = LD.mass_matrix(LEG, q, armature=False) + arm * np.eye(3)
        qdd = np.linalg.solve(M, tau_sent - LD.bias(LEG, q, qd))
        qd = qd + qdd * DT_PHYS
        q = q + qd * DT_PHYS
        t += DT_PHYS
    R = np.array(rows)
    tt, X, Pr = R[:, 0], R[:, 1:4], R[:, 4:7]
    sw = (tt >= 0) & (tt <= T)
    dz = X[:, 2] - x0[2]
    k = int(np.argmin(np.abs(tt - T)))
    return dict(apex=1e3 * dz[sw].max(), s_apex=tt[sw][dz[sw].argmax()] / T,
                rms=1e3 * np.sqrt(np.mean((X[sw, 2] - Pr[sw, 2]) ** 2)),
                x=1e3 * np.abs(X[sw, 0] - x0[0]).max(), z_td=1e3 * dz[k],
                vz_td=(X[k, 2] - X[k - 1, 2]) / SWEEP,
                tau_pk=float(np.abs(R[sw, 10:13]).max()),
                clip=float(np.abs(R[:, 7:10] - R[:, 10:13]).max()),
                t=tt, z=1e3 * dz, z_ref=1e3 * (Pr[:, 2] - x0[2]),
                knee_req=R[:, 9], knee_sent=R[:, 12], T=T)


def row(tag, r):
    return ("  %-40s apex %5.1f mm at s=%.2f  rms %4.1f  x %4.1f  z_td %5.1f  "
            "vz_td %+.2f  tau_pk %.2f  clip %.2f"
            % (tag, r["apex"], r["s_apex"], r["rms"], r["x"], r["z_td"],
               r["vz_td"], r["tau_pk"], r["clip"]))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
CASES = [("bench 140 ms / 20 mm", 0.14, 0.020),
         ("trot 0.8 s: 160 ms / 40 mm", 0.16, 0.040),
         ("trot 0.8 s: 160 ms / 20 mm", 0.16, 0.020),
         ("fold trot 0.5 s: 100 ms / 20 mm", 0.10, 0.020)]


def tables():
    print("THE SLEW THE ARC ASKS FOR (feedforward torque rate, knee or pitch, max)")
    for name, T, H in CASES + [("DOG5 1.2 s: 240 ms / 40 mm", 0.24, 0.040),
                               ("sim cMPC: 250 ms / 30 mm", 0.25, 0.030)]:
        need, tau = slew_needed(T, H)
        print("  %-34s tau_ff peak %.2f N*m   rise rule 4 tau/T %4.0f   "
              "exact %5.0f N*m/s   kappa %.0f" % (name, tau, 4 * tau / T, need,
                                                  need * T ** 3 / H))
    for T, H in [(0.16, 0.02), (0.24, 0.02), (0.30, 0.02)]:
        nq, tq = slew_needed(T, H)
        nj, tj = slew_needed(T, H, "jerk")
        print("  T %.2f H %2.0f: quintic %4.0f N*m/s (tau %.2f)  jerk-limited %4.0f "
              "(tau %.2f)  ratio %.2f" % (T, 1e3 * H, nq, tq, nj, tj, nq / nj))

    print("\nSIM CHAIN AGAINST HW CHAIN (Kp 140/140/180, Kd 20/20/40)")
    for name, T, H in CASES:
        print(" ", name)
        print(row("sim: PD only", run(T, H, hw=False, ff=False)))
        print(row("sim: PD + ff", run(T, H, hw=False)))
        print(row("hw, no slew: PD + ff", run(T, H)))
        print(row("hw, slew 60: PD only", run(T, H, slew=60, ff=False)))
        print(row("hw, slew 60: PD + ff", run(T, H, slew=60)))
        print(row("hw, slew 60: PD only + hold damper", run(T, H, slew=60, ff=False, kd_hold=0.2)))

    print("\nHOW MUCH OF THE ARC'S PEAK RATE THE SLEW MUST PASS (hw chain, PD + ff)")
    for arc, T, H in [("quintic", 0.16, 0.02), ("quintic", 0.10, 0.02),
                      ("quintic", 0.24, 0.02), ("quintic", 0.16, 0.04),
                      ("jerk", 0.16, 0.02), ("jerk", 0.24, 0.03)]:
        need, _ = slew_needed(T, H, arc)
        print("  %s %.0f ms / %.0f mm, peak rate %.0f N*m/s" % (arc, 1e3 * T, 1e3 * H, need))
        for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            print(row("S = %.1f x peak (%4.0f)" % (f, f * need),
                      run(T, H, slew=f * need, arc=arc)))

    print("\nDESIGN OPTIONS (hw chain throughout)")
    for T, H in [(0.16, 0.02), (0.10, 0.02)]:
        need, _ = slew_needed(T, H)
        print(row("swing slew %.0f, %.0f ms / %.0f mm" % (1.3 * need, 1e3 * T, 1e3 * H),
                  run(T, H, slew=1.3 * need)))
    for T, H in [(0.24, 0.012), (0.30, 0.020), (0.30, 0.030), (0.24, 0.020)]:
        print(row("slew 60, quintic %.0f ms / %.0f mm" % (1e3 * T, 1e3 * H),
                  run(T, H, slew=60)))
    for T, H in [(0.24, 0.020), (0.32, 0.030)]:
        print(row("slew 60, jerk-limited %.0f ms / %.0f mm" % (1e3 * T, 1e3 * H),
                  run(T, H, slew=60, arc="jerk")))
    print(row("slew 60, 300/20, hold damper absolute", run(0.30, 0.02, slew=60, kd_hold=0.2)))
    print(row("slew 60, 300/20, hold damper relative",
              run(0.30, 0.02, slew=60, kd_hold=0.2, hold_relative=True)))
    print(row("160/20 no slew, plant arm 0.0045, ff 0.0085", run(0.16, 0.02, arm_plant=0.0045)))
    print(row("160/20 no slew, plant arm 0.0045, ff 0.0045",
              run(0.16, 0.02, arm_plant=0.0045, ff_arm=0.0045)))
    print(row("140/20 slew 60, plant = model", run(0.14, 0.02, slew=60)))
    print(row("140/20 slew 150, plant = model", run(0.14, 0.02, slew=150)))

    J = HK.foot_jacobian(LEG, Q0)
    Ji = np.linalg.inv(J)
    print("\nhold damper 0.2 N*m*s/rad seen at the foot (x, y, z) N*s/m:",
          np.round(np.diag(Ji.T @ (0.2 * np.eye(3)) @ Ji), 1))
    for d in ((1, 0, 0), (0, 0, 1)):
        print("foot apparent mass along", d, "%.2f kg" % LD.foot_apparent_mass(LEG, Q0, d))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF = "#8a8984"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK2,
                         "axes.titlesize": 9, "axes.titlecolor": INK,
                         "pdf.fonttype": 42})
    os.makedirs(FIG, exist_ok=True)

    # -- 1: what the arc asks of the knee, and what 60 N*m/s lets through --
    T, H = 0.16, 0.020
    s, tau, _ = demand(T, H)
    knee = tau[:, 2]
    t = s * T
    passed = np.zeros_like(knee)
    for k in range(1, len(t)):
        step = 60.0 * (t[k] - t[k - 1])
        passed[k] = passed[k - 1] + np.clip(knee[k] - passed[k - 1], -step, step)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    ax = axes[0]
    _style(ax)
    ax.plot(1e3 * t, knee, color=BLUE, lw=1.6)
    ax.plot(1e3 * t, passed, color=ORANGE, lw=1.6)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.text(40, 2.2, "asked: $M_0\\ddot q_{ref}$", color=INK, fontsize=8, ha="left")
    ax.text(2, -1.75, "through a\n60 N m/s slew", color=INK, fontsize=8, ha="left")
    ax.set_xlabel("time in swing (ms)")
    ax.set_ylabel("knee feedforward (N m)")
    ax.set_title("160 ms / 20 mm quintic arc", loc="left")

    ax = axes[1]
    _style(ax)
    Ts = np.linspace(0.08, 0.40, 120)
    kq = slew_needed(0.16, 0.02)[0] * 0.16 ** 3 / 0.02
    # the foot tracks once S >= half the arc's peak torque rate: H <= 2 S T^3 / kappa
    for S_, col, lab, side in ((60, ORANGE, "S = 60 N m/s (now)", "right"),
                               (200, AQUA, "S = 200", "left"),
                               (700, BLUE, "S = 700", "left")):
        h = 1e3 * 2.0 * S_ * Ts ** 3 / kq
        ax.plot(1e3 * Ts, h, color=col, lw=1.6)
        y = 46.0
        x = 1e3 * (y * 1e-3 * kq / (2.0 * S_)) ** (1.0 / 3.0)
        ax.text(x + 6 if side == "right" else x - 6, y, lab, color=INK, fontsize=7.5,
                ha="left" if side == "right" else "right", va="center")
    pts = [(160, 40, "trot", "right"), (160, 20, "trot", "right"),
           (100, 20, "fold", "below"), (140, 20, "bench", "below"),
           (240, 40, "DOG5", "right")]
    for x, y, lab, side in pts:
        ax.plot(x, y, "o", ms=5, color=INK, mec="white", mew=1.0, zorder=3)
        if side == "right":
            ax.text(x + 6, y, lab, fontsize=7.5, color=INK2, va="center")
        elif side == "left":
            ax.text(x - 6, y, lab, fontsize=7.5, color=INK2, va="center", ha="right")
        else:
            ax.text(x, y - 3, lab, fontsize=7.5, color=INK2, va="top", ha="center")
    ax.set_xlim(80, 400)
    ax.set_ylim(0, 50)
    ax.set_xlabel("swing duration $T$ (ms)")
    ax.set_ylabel("largest trackable apex $H$ (mm)")
    ax.set_title("quintic arcs the slew lets the foot track", loc="left")
    fig.savefig(os.path.join(FIG, "slew_budget.pdf"))
    plt.close(fig)

    # -- 2: one leg, sim chain against hw chain ------------------------------
    sim = run(0.16, 0.02, hw=False)
    hw60 = run(0.16, 0.02, slew=60)
    fix = run(0.30, 0.02, slew=60)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    ax = axes[0]
    _style(ax)
    for r, col, lab in ((sim, BLUE, "sim chain"), (hw60, ORANGE, "hw chain, slew 60"),
                        (fix, AQUA, "hw chain, slew 60, T = 300 ms")):
        m = (r["t"] >= -0.01) & (r["t"] <= 1.15 * r["T"])
        ax.plot(r["t"][m] / r["T"], r["z"][m], color=col, lw=1.6, label=lab)
    m = (sim["t"] >= 0) & (sim["t"] <= sim["T"])
    ax.plot(sim["t"][m] / sim["T"], sim["z_ref"][m], color=REF, lw=1.2, ls="--",
            label="reference")
    ax.axvline(1.0, color=INK2, lw=0.6)
    ax.text(1.02, 27, "clock's\ntouchdown", fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(-0.05, 1.18)
    ax.set_ylim(-6, 29)
    ax.set_xlabel("swing progress $s = t/T$  (no floor in the model)")
    ax.set_ylabel("foot height above liftoff (mm)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("foot z, 20 mm apex", loc="left")

    ax = axes[1]
    _style(ax)
    m = (hw60["t"] >= -0.01) & (hw60["t"] <= 1.6 * hw60["T"])
    ax.plot(1e3 * hw60["t"][m], hw60["knee_req"][m], color=BLUE, lw=1.6, label="requested")
    ax.plot(1e3 * hw60["t"][m], hw60["knee_sent"][m], color=ORANGE, lw=1.6,
            label="sent (after slew)")
    ax.axvline(160, color=INK2, lw=0.6)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("knee torque (N m)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("hw chain, 160 ms / 20 mm, slew 60", loc="left")
    fig.savefig(os.path.join(FIG, "one_leg.pdf"))
    plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    tables()
    figures()
