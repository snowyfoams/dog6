"""Drive the reference generator from the Xbox pad and watch what it makes.

    python -m sim.cmpc.teleop

WHAT IS RUNNING HERE, AND WHAT IS NOT
    This is `trajectory.ReferenceTrajectory` and `gait.contact`, and nothing
    else.  No MuJoCo, no QP, no robot.  The stick moves, the command is
    integrated, and the three things the MPC would be handed -- the xy path,
    the yaw, and the contact schedule c -- are drawn as they are produced.

    That separation is the point of having this module at all.  When a turn
    comes out backwards or a strafe drifts, the question is always "is the
    reference wrong or is the tracking wrong", and the only way to answer it
    without argument is to look at the reference with the robot taken out of
    the picture.  `run --gamepad` is the same command path with the robot put
    back in.

    So every panel here is an OPEN-LOOP quantity.  The path drawn is where the
    reference has been told to go, not where any robot went; if this looks
    perfect and the robot still wanders, the fault is downstream of here.

TIME IS WALL TIME, NOT FRAMES
    Each frame advances the reference by the wall-clock interval since the
    last one, not by a nominal `cfg.MPC_DT`.  A dropped frame under a fixed
    step would make the drawn path shorter than the seconds you spent holding
    the stick -- the animation would disagree with the clock, and the contact
    strip (which reads the same clock) would slide against the yaw trace.
    Advancing by the measured interval keeps every panel on one timebase.

    The interval is clamped to `MAX_FRAME` so that dragging the window or
    letting the machine stall does not teleport the reference.

THE CONTACT STRIP IS COMPUTED, NOT RECORDED
    `gait.contact` is a pure function of the clock, so the strip is evaluated
    over the whole visible window each frame -- including the part that is
    still in the FUTURE.  That is not a nicety: the shaded band to the right
    of "now" is exactly the `horizon_contacts` table the QP builds its
    constraint structure from, and seeing it move is seeing the assumption the
    convex formulation rests on.

WITHOUT A PAD, THE STICKS ARE THE MOUSE
    Click and drag inside either stick circle and it deflects; release and it
    springs back to centre, like the real one.  A connected pad always wins
    while it is off-centre, so a pad plugged in mid-session takes over with no
    mode switch.  This keeps the module runnable on a machine with no pad,
    which is what makes it testable.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "sim.cmpc"

from .. import coordinates as C           # noqa: E402
from .. import params as P                # noqa: E402
from . import config as cfg               # noqa: E402
from . import gait                        # noqa: E402
from . import trajectory                  # noqa: E402
from .gamepad import Gamepad, PadState, bindings  # noqa: E402

#: Seconds of history in the time panels, and how far past "now" they show.
WINDOW = 6.0
LOOKAHEAD = cfg.HORIZON * cfg.MPC_DT                 # 0.25 s

#: Longest interval a single frame may integrate.  See the module docstring.
MAX_FRAME = 0.10                                     # s

#: Redraw rate.  The reference is integrated once per frame, so this is also
#: the generator's rate here -- which is deliberately NOT MPC_HZ: the numbers
#: an integrator produces must not depend on how often it is asked, and 30 Hz
#: against the controller's 40 Hz is the cheapest standing check that they do
#: not.
FPS = 30.0

#: Half-width of the xy panel in FOLLOW mode, where the view is locked to the
#: robot instead of to the whole path.  Fit shows the shape of the route and
#: loses the body; follow shows the body, the hips and the horizon preview at
#: a size you can read, and loses the route.  F swaps them because both
#: questions get asked and no single scale answers them.
FOLLOW_SPAN = 0.75                                   # m

#: Heading arrows are dropped along the path this often, in seconds of stick.
ARROW_PERIOD = 0.4

#: How many heading arrows the quiver has room for.  A Quiver's arrow COUNT is
#: fixed when it is created -- `set_UVC` refuses any other length -- so the
#: collection is allocated once at this size and the unused slots are parked
#: at NaN, where nothing is drawn.  It doubles as the cap that keeps a long
#: walk from turning the path into a hedge: past this many the stride grows.
MAX_ARROWS = 240

LEG_COLOURS = ("#d1495b", "#edae49", "#00798c", "#66a182")   # FL FR RL RR
TRAIL = "#2b2d42"
FUTURE = "#8d99ae"


# ===========================================================================
# the thing being driven
# ===========================================================================
class Session:
    """The reference generator, a pad, and the history the panels draw.

    Holds no matplotlib: `step` is the whole simulation and can be called from
    a test with a synthetic `PadState`, which is how the mapping is checked
    without a human holding a stick.
    """

    def __init__(self) -> None:
        self.pad = Gamepad()
        self.ref = trajectory.ReferenceTrajectory()
        self.command = trajectory.Command()
        self.state = PadState()
        self.t = 0.0
        self.paused = False
        self.quit = False
        # Mouse-dragged stand-ins, in the same units as the pad's sticks.
        self.virtual = {"L": np.zeros(2), "R": np.zeros(2)}
        self.hist = {k: [] for k in ("t", "x", "y", "yaw", "vx", "vy", "wz")}
        self._record()

    # -- one frame ---------------------------------------------------------
    def step(self, dt: float, state: PadState | None = None) -> None:
        """Poll, map, integrate, record.  `state` overrides the hardware."""
        if state is None:
            state = self.pad.poll()
            if not (state.lx or state.ly or state.rx or state.ry):
                # Pad centred (or absent): the mouse has the sticks.  Buttons
                # still come from the pad, so A and B work with a hand off it.
                lx, ly = self.virtual["L"]
                rx, ry = self.virtual["R"]
                state = PadState(connected=state.connected, slot=state.slot,
                                 lx=lx, ly=ly, rx=rx, ry=ry,
                                 lt=state.lt, rt=state.rt,
                                 buttons=state.buttons, pressed=state.pressed,
                                 released=state.released)
        self.state = state

        if "A" in state.pressed:
            self.virtual["L"][:] = self.virtual["R"][:] = 0.0
        if "B" in state.pressed:
            self.reset()
        if "START" in state.pressed:
            self.quit = True

        self.command = state.command()
        if self.paused:
            return
        self.ref.advance(self.command, dt)
        self.t += dt
        self._record()

    def _record(self) -> None:
        h = self.hist
        h["t"].append(self.t)
        h["x"].append(self.ref.x)
        h["y"].append(self.ref.y)
        h["yaw"].append(self.ref.yaw)
        h["vx"].append(self.command.vx)
        h["vy"].append(self.command.vy)
        h["wz"].append(self.command.yaw_rate)

    def reset(self) -> None:
        """Back to the origin, facing +x, clock included.

        The CLOCK goes back too, which is the one non-obvious part: the gait
        is a function of absolute time, so leaving t alone would restart the
        path mid-stride and the strip would not line up with a fresh run.
        """
        self.ref = trajectory.ReferenceTrajectory()
        self.t = 0.0
        for v in self.hist.values():
            v.clear()
        self._record()

    # -- what the panels ask for -------------------------------------------
    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.hist.items()}

    def horizon_xy(self) -> np.ndarray:
        """The (HORIZON, 2) xy the MPC would be handed at this instant."""
        return self.ref.horizon(self.command)[:, cfg.POS][:, :2]

    def contacts(self) -> np.ndarray:
        return gait.contact(self.t)

    def source(self) -> str:
        if self.state.connected and any(
                (self.state.lx, self.state.ly, self.state.rx, self.state.ry)):
            return "pad slot %d" % self.state.slot
        if self.state.connected:
            return "pad slot %d (centred)" % self.state.slot
        return "mouse" if any(np.abs(v).sum() for v in self.virtual.values()) \
            else "no pad"


# ===========================================================================
# the figure
# ===========================================================================
class Animation:
    """Six panels over one `Session`.  Everything is a live artist update."""

    def __init__(self, session: Session, window: float = WINDOW) -> None:
        import matplotlib.pyplot as plt

        self.s = session
        self.window = float(window)
        self.plt = plt
        self._last = time.monotonic()
        self._drag = None                    # which stick the mouse owns
        self.follow = False

        self.fig = plt.figure(figsize=(13.0, 7.6))
        self.fig.canvas.manager.set_window_title("DOG6  reference from the pad")
        gs = self.fig.add_gridspec(
            3, 3, width_ratios=[1.0, 0.42, 1.15], height_ratios=[1.0, 1.0, 0.9],
            left=0.05, right=0.935, top=0.90, bottom=0.11,
            wspace=0.30, hspace=0.50)

        self.ax_xy = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_sl = self.fig.add_subplot(gs[2, 0])
        self.ax_sr = self.fig.add_subplot(gs[2, 1])
        self.ax_yaw = self.fig.add_subplot(gs[0, 2])
        self.ax_v = self.fig.add_subplot(gs[1, 2])
        self.ax_c = self.fig.add_subplot(gs[2, 2])

        self._build_xy()
        self._build_sticks()
        self._build_time_panels()
        self._build_contacts()

        self.status = self.fig.text(0.05, 0.962, "", family="monospace",
                                    fontsize=10.0, va="center")
        self.fig.text(0.5, 0.025,
                      "left stick  vx vy      right stick  yaw rate      "
                      "LT precision      A stop      B reset      "
                      "F follow/fit      P pause      R reset",
                      family="monospace", fontsize=8.5, color="#666666",
                      va="center", ha="center")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # -- panels ------------------------------------------------------------
    def _build_xy(self) -> None:
        ax = self.ax_xy
        ax.set_title("reference path   x, y, yaw", fontsize=10, loc="left")
        ax.set_xlabel("x  [m]")
        ax.set_ylabel("y  [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="#cccccc", linewidth=0.8, zorder=0)

        (self.trail,) = ax.plot([], [], color=TRAIL, linewidth=1.6,
                                label="integrated reference")
        (self.future,) = ax.plot([], [], color=FUTURE, linewidth=1.4,
                                 linestyle="--", marker="o", markersize=2.5,
                                 zorder=6,
                                 label="MPC horizon, %.2f s" % LOOKAHEAD)
        nan = np.full(MAX_ARROWS, np.nan)
        self.arrows = ax.quiver(nan, nan, np.zeros(MAX_ARROWS),
                                np.zeros(MAX_ARROWS), color=TRAIL,
                                angles="xy", scale_units="xy", scale=5.0,
                                width=0.005, alpha=0.45, zorder=3)
        # The trunk box, drawn at the reference pose, with the hips on it
        # coloured by the contact schedule -- the same c the strip shows,
        # placed where the feet are.
        from matplotlib.patches import Polygon
        self.body = Polygon(np.zeros((5, 2)), closed=True, facecolor="#ffffff",
                            edgecolor=TRAIL, linewidth=1.6, zorder=4)
        ax.add_patch(self.body)
        self.hips = ax.scatter(np.zeros(4), np.zeros(4), s=42,
                               c=list(LEG_COLOURS), edgecolors=TRAIL,
                               linewidths=0.8, zorder=5)
        (self.nose,) = ax.plot([], [], color=TRAIL, linewidth=1.8, zorder=5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _build_sticks(self) -> None:
        from matplotlib.patches import Circle

        self.stick_dots = {}
        self.stick_lines = {}
        for ax, key, title, dead in (
                (self.ax_sl, "L", "left stick   vx / vy", 0.240),
                (self.ax_sr, "R", "right stick   yaw rate", 0.265)):
            ax.set_title(title, fontsize=9, loc="left")
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#999999",
                                linewidth=1.2))
            ax.add_patch(Circle((0, 0), dead, fill=True, facecolor="#f0f0f0",
                                edgecolor="#cccccc", linewidth=0.8))
            ax.axhline(0.0, color="#dddddd", linewidth=0.8)
            ax.axvline(0.0, color="#dddddd", linewidth=0.8)
            (line,) = ax.plot([0, 0], [0, 0], color=TRAIL, linewidth=1.4)
            (dot,) = ax.plot([0], [0], marker="o", markersize=9,
                             color="#d1495b")
            self.stick_lines[key] = line
            self.stick_dots[key] = dot

    def _build_time_panels(self) -> None:
        ax = self.ax_yaw
        ax.set_title("yaw   integral of the commanded yaw rate",
                     fontsize=10, loc="left")
        ax.set_ylabel("yaw  [deg]")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelbottom=False)
        (self.yaw_line,) = ax.plot([], [], color="#00798c", linewidth=1.6)
        self.yaw_now = ax.axvline(0.0, color="#d1495b", linewidth=1.0)

        ax = self.ax_v
        ax.set_title("command   body axes", fontsize=10, loc="left")
        ax.set_ylabel("v  [m/s]")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelbottom=False)
        ax.set_ylim(-cfg.V_MAX * 1.15, cfg.V_MAX * 1.15)
        (self.vx_line,) = ax.plot([], [], color="#2b2d42", linewidth=1.5,
                                  label="vx")
        (self.vy_line,) = ax.plot([], [], color="#edae49", linewidth=1.5,
                                  label="vy")
        self.ax_w = ax.twinx()
        self.ax_w.set_ylabel("yaw rate  [deg/s]")
        limit = np.rad2deg(cfg.YAW_RATE_MAX) * 1.15
        self.ax_w.set_ylim(-limit, limit)
        (self.wz_line,) = self.ax_w.plot([], [], color="#d1495b",
                                         linewidth=1.5, linestyle="--",
                                         label="yaw rate")
        ax.legend(handles=[self.vx_line, self.vy_line, self.wz_line],
                  loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
        self.v_now = ax.axvline(0.0, color="#d1495b", linewidth=1.0)

    def _build_contacts(self) -> None:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Polygon

        ax = self.ax_c
        ax.set_title("contact schedule c   stance filled   the band right of "
                     "now is the horizon", fontsize=9, loc="left")
        ax.set_xlabel("t  [s]")
        ax.set_yticks(range(4))
        ax.set_yticklabels(C.LEGS, fontsize=9)
        ax.set_ylim(3.5, -0.5)
        self._cmap = ListedColormap(("#f4f4f4",) + LEG_COLOURS)
        self._grid = np.linspace(-self.window, LOOKAHEAD, 480)
        self.strip = ax.imshow(np.zeros((4, self._grid.size)), aspect="auto",
                               cmap=self._cmap, vmin=0, vmax=4,
                               interpolation="nearest",
                               extent=(-self.window, LOOKAHEAD, 3.5, -0.5))
        self.horizon_band = Polygon(np.zeros((4, 2)), closed=True,
                                    facecolor="#000000", alpha=0.06,
                                    edgecolor="none", zorder=3)
        ax.add_patch(self.horizon_band)
        self.c_now = ax.axvline(0.0, color="#d1495b", linewidth=1.2, zorder=4)

    # -- per-frame updates -------------------------------------------------
    def update(self, _frame=None):
        now = time.monotonic()
        dt = min(now - self._last, MAX_FRAME)
        self._last = now
        self.s.step(dt)
        if self.s.quit:
            self.plt.close(self.fig)
            return []

        h = self.s.arrays()
        self._update_xy(h)
        self._update_sticks()
        self._update_time_panels(h)
        self._update_contacts()
        self.status.set_text(
            "t %6.2f s   %s   |  x %+.2f  y %+.2f  yaw %+6.1f deg   |  %s%s"
            % (self.s.t, self.s.command, self.s.ref.x, self.s.ref.y,
               np.rad2deg(self.s.ref.yaw), self.s.source(),
               "   PAUSED" if self.s.paused else ""))
        return []

    def _update_xy(self, h) -> None:
        self.trail.set_data(h["x"], h["y"])

        future = self.s.horizon_xy()
        self.future.set_data(np.r_[h["x"][-1], future[:, 0]],
                             np.r_[h["y"][-1], future[:, 1]])

        # Heading arrows every ARROW_PERIOD of stick time, thinned by index
        # because the frame interval is wall time and therefore not uniform,
        # and thinned further once the path outgrows the quiver.
        n = h["x"].size
        stride = max(1, int(round(ARROW_PERIOD * FPS)),
                     int(np.ceil(n / MAX_ARROWS)))
        xs, ys, yaws = h["x"][::stride], h["y"][::stride], h["yaw"][::stride]
        offsets = np.full((MAX_ARROWS, 2), np.nan)
        u, v = np.zeros(MAX_ARROWS), np.zeros(MAX_ARROWS)
        k = min(xs.size, MAX_ARROWS)
        offsets[:k] = np.c_[xs[:k], ys[:k]]
        u[:k], v[:k] = np.cos(yaws[:k]), np.sin(yaws[:k])
        # BOTH, and the second one is not redundant.  `set_offsets` moves the
        # arrows, but a Quiver built with angles="xy" computes each arrow's
        # DIRECTION from its own `XY` attribute, which set_offsets does not
        # touch -- leave it at the NaN it was constructed with and every arrow
        # gets a NaN angle and silently draws nothing at all.
        self.arrows.XY = offsets
        self.arrows.set_offsets(offsets)
        self.arrows.set_UVC(u, v)

        x, y, yaw = h["x"][-1], h["y"][-1], h["yaw"][-1]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        half = P.TRUNK_BOX_HALF[:2]
        corners = np.array([[+half[0], +half[1]], [+half[0], -half[1]],
                            [-half[0], -half[1]], [-half[0], +half[1]],
                            [+half[0], +half[1]]])
        self.body.set_xy(corners @ rot.T + (x, y))
        self.nose.set_data(*zip((x, y), (x + 0.18 * np.cos(yaw),
                                         y + 0.18 * np.sin(yaw))))

        hips = P.HIP_OFFSET[:, :2] @ rot.T + (x, y)
        self.hips.set_offsets(hips)
        # Stance feet are drawn solid, swing feet hollow: the same c the strip
        # reports, in the place on the robot it refers to.
        contact = self.s.contacts()
        self.hips.set_facecolor([LEG_COLOURS[i] if contact[i] else "#ffffff"
                                 for i in range(4)])

        self._fit_xy(h)

    def _fit_xy(self, h) -> None:
        """The view: locked to the robot in follow mode, else around the path.

        Equal aspect plus independently fitted limits is the classic way to
        get a path that changes shape as it grows; the span is taken as one
        number for both axes so a straight-line walk does not get stretched.
        """
        if self.follow:
            x, y = float(h["x"][-1]), float(h["y"][-1])
            self.ax_xy.set_xlim(x - FOLLOW_SPAN, x + FOLLOW_SPAN)
            self.ax_xy.set_ylim(y - FOLLOW_SPAN, y + FOLLOW_SPAN)
            return
        pad = 0.35
        x0, x1 = float(h["x"].min()), float(h["x"].max())
        y0, y1 = float(h["y"].min()), float(h["y"].max())
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        span = max(x1 - x0, y1 - y0, 1.0) * 0.5 + pad
        self.ax_xy.set_xlim(cx - span, cx + span)
        self.ax_xy.set_ylim(cy - span, cy + span)

    def _update_sticks(self) -> None:
        s = self.s.state
        for key, (x, y) in (("L", (s.lx, s.ly)), ("R", (s.rx, s.ry))):
            self.stick_dots[key].set_data([x], [y])
            self.stick_lines[key].set_data([0, x], [0, y])

    def _update_time_panels(self, h) -> None:
        t = h["t"]
        t0, t1 = self.s.t - self.window, self.s.t + LOOKAHEAD
        yaw = np.rad2deg(h["yaw"])
        self.yaw_line.set_data(t, yaw)
        self.vx_line.set_data(t, h["vx"])
        self.vy_line.set_data(t, h["vy"])
        self.wz_line.set_data(t, np.rad2deg(h["wz"]))
        for ax in (self.ax_yaw, self.ax_v, self.ax_w):
            ax.set_xlim(t0, t1)
        self.yaw_now.set_xdata([self.s.t, self.s.t])
        self.v_now.set_xdata([self.s.t, self.s.t])

        visible = yaw[t >= t0]
        if visible.size:
            lo, hi = float(visible.min()), float(visible.max())
            pad = max(10.0, 0.1 * (hi - lo))
            self.ax_yaw.set_ylim(lo - pad, hi + pad)

    def _update_contacts(self) -> None:
        times = self.s.t + self._grid
        # gait.contact takes an array of times and returns (N, 4); the strip
        # wants a leg per ROW, and a leg's own colour index where it is down.
        c = gait.contact(times).T
        self.strip.set_data(c * (np.arange(4)[:, None] + 1))
        self.strip.set_extent((times[0], times[-1], 3.5, -0.5))
        self.ax_c.set_xlim(times[0], times[-1])
        self.c_now.set_xdata([self.s.t, self.s.t])
        self.horizon_band.set_xy([[self.s.t, -0.5], [self.s.t + LOOKAHEAD, -0.5],
                                  [self.s.t + LOOKAHEAD, 3.5], [self.s.t, 3.5]])

    # -- input -------------------------------------------------------------
    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == " ":
            self.s.virtual["L"][:] = self.s.virtual["R"][:] = 0.0
        elif key == "r":
            self.s.reset()
        elif key == "p":
            self.s.paused = not self.s.paused
        elif key == "f":
            self.follow = not self.follow

    def _stick_at(self, event):
        if event.inaxes is self.ax_sl:
            return "L"
        if event.inaxes is self.ax_sr:
            return "R"
        return None

    def _on_press(self, event) -> None:
        self._drag = self._stick_at(event)
        self._on_motion(event)

    def _on_motion(self, event) -> None:
        if self._drag is None or event.xdata is None:
            return
        v = np.array([event.xdata, event.ydata], float)
        r = float(np.hypot(*v))
        if r > 1.0:                       # the gate is round, so clamp radially
            v /= r
        self.s.virtual[self._drag][:] = v

    def _on_release(self, event) -> None:
        if self._drag is not None:
            # Spring back to centre, like the real stick.  A virtual stick that
            # held its last value would reintroduce exactly the latching the
            # absolute mapping exists to avoid.
            self.s.virtual[self._drag][:] = 0.0
            self._drag = None

    # -- run ---------------------------------------------------------------
    def run(self) -> int:
        from matplotlib.animation import FuncAnimation

        # Kept on the instance: a FuncAnimation nobody holds is collected and
        # the figure simply stops updating, with no error anywhere.
        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=int(1000.0 / FPS),
                                  blit=False, cache_frame_data=False)
        self._last = time.monotonic()
        self.plt.show()
        return 0


# ===========================================================================
# headless: the same session, on a scripted stick
# ===========================================================================
def dry_run(seconds: float = 4.0, dt: float = 1.0 / FPS) -> str:
    """Drive `Session` from a synthetic stick and print what came out.

    No figure, no pad, no window -- this is the part of the module that can be
    checked on a machine with none of the three, and it exercises the same
    `step` the animation calls.
    """
    session = Session()
    rows = ["DOG6 teleop dry run   %.1f s at %.0f Hz" % (seconds, 1.0 / dt),
            "  stick: left fully up, right half right",
            "",
            "     t      x       y     yaw     vx     vy     wz    c"]
    stick = PadState(connected=True, slot=0, ly=1.0, rx=0.5)
    n = int(round(seconds / dt))
    # 0.35 s between printed rows, NOT a round fraction of the 0.50 s gait
    # period: sampling on the period prints the same contact pair every row
    # and the schedule looks frozen.
    every = max(1, int(round(0.35 / dt)))
    for i in range(n):
        session.step(dt, stick)
        if i % every == 0 or i == n - 1:
            c = "".join("1" if x else "0" for x in session.contacts())
            rows.append("  %5.2f  %6.3f  %6.3f  %6.1f  %5.2f  %5.2f  %5.1f  %s"
                        % (session.t, session.ref.x, session.ref.y,
                           np.rad2deg(session.ref.yaw), session.command.vx,
                           session.command.vy,
                           np.rad2deg(session.command.yaw_rate), c))
    rows += ["",
             "  the path curves CLOCKWISE: right stick right is a negative",
             "  yaw rate, which is a right-hand turn.  See gamepad's docstring."]
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=float, default=WINDOW,
                        help="seconds of history in the time panels")
    parser.add_argument("--dry-run", action="store_true",
                        help="no figure: drive a synthetic stick and print")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(dry_run())
        return 0

    session = Session()
    print("\n  DOG6 reference generator, driven by the pad\n")
    print("  pad             %s" % (session.pad.backend or "none"))
    print("  connected       %s" % ("slot %d" % session.pad.slot
                                    if session.pad.slot >= 0
                                    else "no -- drag the stick circles with "
                                         "the mouse instead"))
    print()
    print(bindings())
    print("\n  in the window:  SPACE stop    P pause    R reset    Q close\n")
    return Animation(session, window=args.window).run()


if __name__ == "__main__":
    raise SystemExit(main())
