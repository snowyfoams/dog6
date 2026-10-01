"""Figure: the forward drift during the rise from the fold crouch -- as flown, bumpless only, bumpless + stance xy spring."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "fig")
TBO = 0.03501
runs = [("FOLD60_asis__stand.npz", "as flown: torque ramps from 0, no xy spring", "C3"),
        ("FOLD60_prime_nospring__stand.npz", "bumpless handover, no xy spring", "C1"),
        ("FOLD60_fixed__stand.npz", "bumpless + stance xy spring 400 N/m, 25 N s/m", "C2")]
fig, ax = plt.subplots(4, 1, figsize=(7.5, 8.5), sharex=True)
for fname, label, col in runs:
    r = np.load(os.path.join(DATA, fname), allow_pickle=True)
    t = r["t"]; ph = r["phase"]
    i0 = np.flatnonzero(ph == "rise")[0]
    t0 = t[i0]
    sel = (t >= t0 - 0.5) & (ph != "STOP") & (t <= t0 + 6.0)
    tt = t[sel] - t0
    ax[0].plot(tt, 1e3 * (r["xy_trunk"][sel, 0] - r["xy_trunk"][i0, 0]), color=col, label=label)
    ax[1].plot(tt, 1e3 * (r["z_trunk"][sel] - TBO), color=col)
    ax[2].plot(tt, np.degrees(r["pitch"][sel]), color=col)
    feet = r["foot_w"][sel]
    front = feet[:, :2, 0].mean(axis=1) - r["xy_trunk"][sel, 0] - 0.1563
    rear = feet[:, 2:, 0].mean(axis=1) - r["xy_trunk"][sel, 0] + 0.1563
    ax[3].plot(tt, 1e3 * front, color=col)
    ax[3].plot(tt, 1e3 * rear, color=col, linestyle="--")
ax[0].set_ylabel("trunk x shift [mm]"); ax[0].legend(fontsize=8, loc="upper left")
ax[1].set_ylabel("h, floor to belly [mm]")
ax[2].set_ylabel("pitch [deg]")
ax[3].set_ylabel("foot x ahead of its hip [mm]\n(solid front, dashed rear)")
ax[3].set_xlabel("time since the crouch -> rise handover [s]  (rise 0-3 s, then HOLD)")
for a in ax: a.grid(alpha=0.3)
ax[0].set_title("Rise from the parallel fold (feet +79 / +23 mm ahead of the hips, crouch 60 mm)", fontsize=10)
fig.tight_layout()
out = os.path.join(FIG, "fig_rise_drift.png")
fig.savefig(out, dpi=130)
print("->", out)
