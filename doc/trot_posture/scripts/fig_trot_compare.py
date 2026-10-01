"""Figure: the in-place trot (0.5 s, 20 mm apex, ideal plant) from three stances -- roll, pitch, trunk walk."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); FIG = os.path.join(HERE, "..", "fig")
names = sys.argv[1:] or ["FOLD_asis", "NOMINAL", "WIDE"]
clock = os.environ.get("CLOCK", "P0.5_A20"); pert = os.environ.get("PERT", "ideal")
cols = ["C3", "C0", "C2", "C1", "C4", "C5"]
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for name, col in zip(names, cols):
    path = os.path.join(DATA, "%s__%s_%s.npz" % (name, clock, pert))
    if not os.path.exists(path):
        print("missing", path); continue
    r = np.load(path, allow_pickle=True)
    t = r["t"]; ph = r["phase"]
    i0 = np.flatnonzero(ph == "trot")[0]
    sel = (t >= t[i0] - 0.5) & (ph != "STOP")
    tt = t[sel] - t[i0]
    ax[0].plot(tt, np.degrees(r["roll"][sel]), color=col, label=name, lw=1.0)
    ax[1].plot(tt, np.degrees(r["pitch"][sel]), color=col, lw=1.0)
    xy = r["xy_trunk"][sel] - r["xy_trunk"][i0]
    ax[2].plot(tt, 1e3 * xy[:, 0], color=col, lw=1.0)
    ax[2].plot(tt, 1e3 * xy[:, 1], color=col, lw=1.0, ls="--")
ax[0].set_ylabel("roll [deg]"); ax[0].legend(fontsize=8, loc="upper right")
ax[1].set_ylabel("pitch [deg]")
ax[2].set_ylabel("trunk walk [mm]\n(solid x, dashed y)")
ax[2].set_xlabel("time since T [s]  (trot in place %s, %s)" % (clock, pert))
for a in ax: a.grid(alpha=0.3)
ax[0].set_title("Trot in place from three stances: %s" % ", ".join(names), fontsize=10)
fig.tight_layout()
out = os.path.join(FIG, "fig_trot_%s_%s.png" % (clock, pert))
fig.savefig(out, dpi=130); print("->", out)
