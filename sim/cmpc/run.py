"""Drive the convex MPC in MuJoCo -- interactively, or on a script.

    python -m sim.cmpc.run                      # viewer; pad if one is on
    python -m sim.cmpc.run --no-gamepad         # viewer, keyboard only
    python -m sim.cmpc.run --headless           # scripted, prints a report
    python -m sim.cmpc.run --headless --script turn --duration 8

KEYS
    W / S     x-velocity  -/+ 0.1 m/s        Q / E   yaw rate -/+ 20 deg/s
    A / D     y-velocity  -/+ 0.1 m/s        SPACE   stop (all commands 0)
    R         reset the robot to the stand keyframe

AN XBOX PAD, IF ONE IS CONNECTED, TAKES THE COMMAND OVER
    A pad found at startup drives the command ABSOLUTELY -- left stick xy,
    right stick yaw rate, centred means stop -- and the keys stop mattering
    while it is connected.  `gamepad` has the mapping and the reasons; the
    short version is that a stick has a magnitude and a key does not, so the
    pad commands a velocity where the keys accumulate one.

    Switch the pad off and the keyboard is live again on the next sweep, at
    whatever the keys last accumulated -- which is why the command is zeroed
    on the way out.  A pad going quiet must not leave a robot walking.

    The pad is polled at PAD_HZ, NOT once per sweep.  A connected slot costs
    19 us to read (an empty one 48, which is why hunting for a vanished pad is
    rate-limited inside `gamepad`); at 250 Hz that is a measurable slice of a
    4 ms budget spent resampling a device that reports at about 125 Hz over
    Bluetooth.  50 Hz is already faster than the stick can be moved.

    `teleop` is the same command path with the robot taken out: the pad drives
    the reference generator alone, and the xy path, the yaw and the contact
    schedule are drawn as they are produced.  It is the place to look when the
    question is whether the reference or the tracking is at fault.

THREE RATES, SET BY THREE SOURCES
    physics   500 Hz   model/dog6.xml's timestep, 0.002 s
    MPC        20 Hz   the QP re-solved for f in WORLD axes; HELD in between
    IMU       200 Hz   R arrives, so fb = R' f is refreshed; HELD in between
    encoder   250 Hz   q arrives, so tau = J' fb is computed and written

    The two sensor rates are hardware, not choices.  R and q come from
    different devices on different clocks, so the world-to-body force rotation
    and the torque map cannot share a rate: every torque written between IMU
    samples goes out through an attitude up to 8 ms old.  `controller._sensed`
    is where the simulator's perfect state is split along that boundary.

    The MPC's hold is the zero-order hold `dynamics` assumes.  Writing a fresh
    force every sweep would make the plant see an input the model did not
    predict.

    Neither 20 Hz nor 200 Hz divides 250 Hz (12.5 and 1.25 sweeps), so both
    deadlines are counted from a fixed origin and the intervals alternate
    about an exact mean -- see `controller.Ticker`.
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

from .. import coordinates as C            # noqa: E402
from .. import params as P                 # noqa: E402
from . import config as cfg                # noqa: E402
from . import controller as ctl            # noqa: E402
from . import gamepad as gp                # noqa: E402
from . import trajectory                   # noqa: E402

try:
    import mujoco
except ImportError:                        # pragma: no cover
    sys.exit("mujoco is not installed in this interpreter.\n"
             "  D:\\mujoco\\.venv\\Scripts\\python.exe -m sim.cmpc.run")


#: How often the pad is read, in simulated seconds.  See the module docstring.
PAD_HZ = 50.0


# ===========================================================================
# scripted command timelines, for headless runs
# ===========================================================================
#: (start time, Command).  The last one holds to the end of the run.
SCRIPTS = {
    "stand": [(0.0, trajectory.Command())],
    "forward": [
        (0.0, trajectory.Command()),
        (1.0, trajectory.Command(vx=0.1)),
        (3.0, trajectory.Command(vx=0.2)),
        (6.0, trajectory.Command(vx=0.3)),
    ],
    "turn": [
        (0.0, trajectory.Command()),
        (1.0, trajectory.Command(yaw_rate=np.deg2rad(20.0))),
        (4.0, trajectory.Command(yaw_rate=np.deg2rad(40.0))),
    ],
    "strafe": [
        (0.0, trajectory.Command()),
        (1.0, trajectory.Command(vy=0.1)),
        (4.0, trajectory.Command(vy=0.2)),
    ],
    "box": [
        (0.0, trajectory.Command()),
        (1.0, trajectory.Command(vx=0.2)),
        (4.0, trajectory.Command(vy=0.2)),
        (7.0, trajectory.Command(vx=-0.2)),
        (10.0, trajectory.Command(vy=-0.2)),
    ],
}


def command_at(script, t: float) -> trajectory.Command:
    out = script[0][1]
    for start, command in script:
        if t >= start:
            out = command
    return out


# ===========================================================================
# the simulation
# ===========================================================================
def reset(model, data) -> None:
    """Put the robot at the drawn stance, feet exactly on the floor."""
    mujoco.mj_resetDataKeyframe(model, data, C.KEYFRAMES["stand"])
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


class Run:
    """One simulation, driven either by a script or by the keyboard."""

    def __init__(self, duration: float = 10.0, script: str = "forward"):
        self.model = mujoco.MjModel.from_xml_path(P.XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.controller = ctl.Controller()
        self.duration = float(duration)
        self.script = SCRIPTS[script]
        # The command the operator is asking for, whatever produced it --
        # keys or a stick.
        self.operator = trajectory.Command()
        self.live = False                  # the operator drives, not a script
        self.pad = None
        self.diverged = False              # mj_step raised, see `step`
        self.stop = False                  # START on the pad ends the run
        self._pad_due = 0.0
        self._pad_on = False

        self.steps_per_control = max(1, int(round(
            cfg.CONTROL_DT / self.model.opt.timestep)))
        self.log: list[dict] = []
        self._solve_times: list[float] = []
        self._sweep_times: list[float] = []
        reset(self.model, self.data)

    # -- keyboard ----------------------------------------------------------
    def on_key(self, keycode: int) -> None:
        key = chr(keycode) if 0 <= keycode < 0x110000 else ""
        command = self.operator
        if key in ("W", "w"):
            command.vx += cfg.V_STEP
        elif key in ("S", "s"):
            command.vx -= cfg.V_STEP
        elif key in ("A", "a"):
            command.vy += cfg.V_STEP
        elif key in ("D", "d"):
            command.vy -= cfg.V_STEP
        elif key in ("Q", "q"):
            command.yaw_rate += cfg.YAW_RATE_STEP
        elif key in ("E", "e"):
            command.yaw_rate -= cfg.YAW_RATE_STEP
        elif key == " ":
            command.vx = command.vy = command.yaw_rate = 0.0
        elif key in ("R", "r"):
            self.recover()
            command = self.operator
        else:
            return
        self.operator = command.clipped()
        print("  command  %s" % self.operator, flush=True)

    # -- gamepad -----------------------------------------------------------
    def attach_pad(self):
        """Start watching for a pad.  True if one is answering right now.

        A pad that is ASLEEP is not a pad that is absent, and the difference
        bites: an Xbox pad idles itself out after a few minutes and reports
        ERROR_DEVICE_NOT_CONNECTED until a button wakes it.  Deciding at
        startup that there is no pad would mean the operator who reaches for
        it thirty seconds in has to restart the simulation to be noticed.

        So the watcher is attached whenever the BACKEND exists, and the pad
        itself may show up later -- `Gamepad.poll` re-scans the slots once a
        second, and `poll_pad` takes over the command on the edge.
        """
        pad = gp.Gamepad()
        self.pad = pad if pad.backend else None
        self._pad_on = bool(self.pad) and pad.poll().connected
        return self._pad_on

    def poll_pad(self, t: float) -> None:
        """Read the stick into `self.operator`, at PAD_HZ.

        The command is written ONLY while the pad reports connected, and is
        ZEROED on the EDGE where it stops.  Both halves matter.  Zeroing is
        not left to the absolute mapping because a mapping can only speak for
        a device that is still answering, and the last thing a pad says before
        its battery dies is whatever the stick happened to be at.  Zeroing on
        the edge rather than on every disconnected poll is what lets the
        keyboard keep working when there is no pad at all -- otherwise every
        key press would be wiped 50 times a second by a pad that never was.
        """
        if self.pad is None or t < self._pad_due:
            return
        self._pad_due = t + 1.0 / PAD_HZ

        state = self.pad.poll()
        if state.connected:
            if "B" in state.pressed:
                self.recover()
            if "START" in state.pressed:
                self.stop = True
            if not self._pad_on:
                print("  pad awake on slot %d -- the stick has the command"
                      % state.slot, flush=True)
            self.operator = state.command()
        elif self._pad_on:
            self.operator = trajectory.Command()
            print("  pad gone -- command zeroed, the keys are live",
                  flush=True)
        self._pad_on = state.connected

    # -- one control sweep -------------------------------------------------
    def sweep(self) -> None:
        t = self.data.time
        if self.live:
            self.poll_pad(t)
        self.controller.command = (self.operator if self.live
                                   else command_at(self.script, t))

        state = ctl.state_from_mujoco(self.model, self.data)
        started = time.perf_counter()
        torque = self.controller.update(state, t)
        self._sweep_times.append(time.perf_counter() - started)
        if self.controller.telemetry.solved:
            # The controller times its own solve.  Timing it from out here
            # would include the leg loop that runs in the same call.
            self._solve_times.append(self.controller.telemetry.solve_ms * 1e-3)
            self.controller.telemetry.solved = False

        self.data.ctrl[:] = C.flat(torque)
        self._record(t, state, torque)

    def _record(self, t, state, torque) -> None:
        tele = self.controller.telemetry
        command = self.controller.command
        # The command is in BODY axes; the measured velocity is in world.
        # Comparing them directly would call a perfectly tracked turn a
        # tracking failure as soon as the heading left zero.
        rot = state.rotation[:2, :2]
        self.log.append({
            "t": t,
            "z": float(state.position[2]),
            "rpy": state.rpy.copy(),
            "v": state.velocity.copy(),
            "v_body": rot.T @ state.velocity[:2],
            "v_cmd": np.array([command.vx, command.vy]),
            "omega": state.omega.copy(),
            "yaw_rate_cmd": command.yaw_rate,
            "tau_max": float(np.abs(torque).max()),
            "clipped": tele.clipped,
            "status": tele.status,
            "fz": tele.forces[:, 2].copy(),
            "contacts": tele.contacts.copy(),
        })

    def step(self) -> None:
        """Advance physics by one control period.

        THE PHYSICS CAN RAISE, AND IT RAISES FATALLY.  MuJoCo reports a state
        it cannot integrate -- most visibly "mju_makeFrame: xaxis of contact
        frame undefined", a contact whose normal came out undefined -- by
        calling its error handler, which the Python bindings turn into
        `mujoco.FatalError`.  Uncaught, that ends the process: a long driving
        session dies at the prompt and takes its report with it.

        It is caught here rather than in the viewer loop because THIS is the
        call that knows how far through a control period it got.  The
        remaining substeps are abandoned, the run is marked diverged, and
        `fallen` reports it -- so the recovery is the one the interactive loop
        already performs for a fall, and there is only one way back.
        """
        self.sweep()
        try:
            for _ in range(self.steps_per_control):
                mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError as exc:
            self.diverged = True
            print("  PHYSICS DIVERGED at t = %.2f s: %s"
                  % (self.data.time, exc), flush=True)

    def recover(self) -> None:
        """Back to the stand keyframe, with a fresh controller and no command.

        THE COMMAND IS ZEROED AND THAT IS THE POINT OF HAVING THIS AS ONE
        METHOD.  A key-driven command ACCUMULATES, so a robot reset after
        falling at 0.5 m/s would otherwise walk straight back into the same
        fall with nobody touching anything -- which is what the repeated,
        identically-timed falls in a session log look like.  A pad, being
        absolute, simply re-commands the stick within one poll if the operator
        is still holding it, which is the honest behaviour: the stick says
        what it says.
        """
        reset(self.model, self.data)
        self.controller = ctl.Controller()
        self.operator = trajectory.Command()
        self.diverged = False

    def fallen(self) -> bool:
        """Height or attitude outside anything a trot could recover from.

        THE ATTITUDE TEST IS NOT OPTIONAL.  Height alone misses the failure
        that actually happens here: a robot rolling onto its side keeps its
        trunk origin well above 0.08 m the whole way over, so a height-only
        check reported a 60-degree-rms tumble as a completed run.  The trunk's
        own +z against world +z is the direct question -- below 0.5 is 60
        degrees off vertical, which no trot recovers from.
        """
        if self.diverged or not np.all(np.isfinite(self.data.qpos)):
            return True
        upright = self.data.xmat[1].reshape(3, 3)[2, 2]
        return (self.data.qpos[2] < 0.08
                or upright < 0.5
                or abs(self.controller.telemetry.torque).max() > 1e6)

    # -- drivers -----------------------------------------------------------
    def headless(self) -> int:
        while self.data.time < self.duration:
            self.step()
            if self.fallen():
                break
        print(self.report())
        return 0 if not self.fallen() else 1

    def interactive(self) -> int:
        import mujoco.viewer

        self.live = True
        # Spelled out rather than sliced out of __doc__: the docstring's
        # headings are prose and have been reworded once already, which
        # silently turned this into "print the rest of the file".
        if self.pad is not None:
            print("\n".join([
                "",
                "  DOG6 convex MPC -- Xbox pad, %s" % self.pad.backend,
                "  %s" % ("awake on slot %d" % self.pad.slot if self._pad_on
                          else "asleep or off -- press a button on it and it "
                               "is picked up within a second"),
                "",
                gp.bindings(),
                "",
                "  The keys below still work, but the stick overwrites them",
                "  %.0f times a second while the pad is connected." % PAD_HZ,
                ""]))
        print("\n".join([
            "",
            "  DOG6 convex MPC -- click the viewer window, then:",
            "",
            "    W / S     forward  -/+ %.2f m/s" % cfg.V_STEP,
            "    A / D     sideways -/+ %.2f m/s" % cfg.V_STEP,
            "    Q / E     turn     -/+ %.0f deg/s" % np.rad2deg(cfg.YAW_RATE_STEP),
            "    SPACE     stop     (all commands to zero)",
            "    R         reset    (robot back to the stand pose)",
            "",
            "  Presses ACCUMULATE: three taps of W is %.1f m/s.  It walks to"
            % (3 * cfg.V_STEP),
            "  about %.1f m/s and turns to %.0f deg/s; past that the 8 N*m"
            % (0.3, 90.0),
            "  torque clamp saturates and it falls.",
            "",
        ]))
        print("  command  %s" % self.operator, flush=True)
        with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.on_key) as viewer:
            wall = time.perf_counter()
            while viewer.is_running() and not self.stop:
                self.step()
                if self.fallen():
                    print("  FELL at t = %.2f s -- reset, command zeroed"
                          % self.data.time, flush=True)
                    self.recover()
                viewer.sync()
                # Keep wall time roughly in step with sim time.
                wall += self.steps_per_control * self.model.opt.timestep
                lag = wall - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        print(self.report())
        return 0

    # -- reporting ---------------------------------------------------------
    def report(self) -> str:
        if not self.log:
            return "nothing ran"
        t = np.array([row["t"] for row in self.log])
        z = np.array([row["z"] for row in self.log])
        rpy = np.array([row["rpy"] for row in self.log])
        vel = np.array([row["v"] for row in self.log])
        tau = np.array([row["tau_max"] for row in self.log])
        clipped = np.array([row["clipped"] for row in self.log])
        fz = np.array([row["fz"] for row in self.log])

        v_body = np.array([row["v_body"] for row in self.log])
        v_cmd = np.array([row["v_cmd"] for row in self.log])

        # UNWRAPPED.  rpy[:, 2] comes from arctan2 and lives in (-pi, pi], so a
        # robot that turns more than half a circle reports a yaw that has
        # jumped by 2 pi.  Differencing the raw ends called a clean 248 deg
        # turn "-112 deg" -- a reporting bug that reads exactly like a sign
        # error in the controller.
        yaw = np.unwrap(rpy[:, 2])
        yaw_cmd = np.trapezoid(np.array([row["yaw_rate_cmd"] for row in self.log]), t)

        settle = t > min(1.0, 0.3 * t[-1])       # ignore the first second
        solves = np.array(self._solve_times) * 1e3
        sweeps = np.array(self._sweep_times) * 1e3
        statuses = {row["status"] for row in self.log}

        lines = [
            "",
            "DOG6 convex MPC -- %.2f s simulated%s"
            % (t[-1], "  (FELL)" if self.fallen() else ""),
            "  height          %.4f m mean, %.4f min, %.4f max  (ref %.4f)"
            % (z[settle].mean(), z.min(), z.max(), cfg.Z_REF),
            "  height error    %.1f mm rms" % (1000 * np.sqrt(
                np.mean((z[settle] - cfg.Z_REF) ** 2))),
            "  roll / pitch    %.2f / %.2f deg rms"
            % (np.rad2deg(np.sqrt(np.mean(rpy[settle, 0] ** 2))),
               np.rad2deg(np.sqrt(np.mean(rpy[settle, 1] ** 2)))),
            "  velocity        vx %+.3f / %+.3f  vy %+.3f / %+.3f m/s "
            "(achieved / commanded, body axes)"
            % (v_body[settle, 0].mean(), v_cmd[settle, 0].mean(),
               v_body[settle, 1].mean(), v_cmd[settle, 1].mean()),
            "  yaw             %+.1f / %+.1f deg turned (achieved / commanded)"
            % (np.rad2deg(yaw[-1] - yaw[0]), np.rad2deg(yaw_cmd)),
            "  peak torque     %.2f N*m of the %.1f cap, clipped on %d sweeps"
            % (tau.max(), cfg.TAU_MAX, int((clipped > 0).sum())),
            "  normal force    %.1f N peak on one foot, %.1f N mean total"
            % (fz.max(), fz.sum(axis=1)[settle].mean()),
            # MEDIAN, NOT MEAN.  The first solve pays OSQP's symbolic setup and
            # every run picks up scheduling spikes an order of magnitude above
            # the body of the distribution -- reported as a mean, the leg loop
            # looked like 156 % of its budget when its median is 56 %.  The
            # max is kept because a hard-real-time loop does care about it, but
            # on a desktop OS it measures the OS.
            "  rates           MPC %.1f Hz / IMU %.1f Hz / encoder %.1f Hz"
            % (self.controller._mpc_tick.count / max(t[-1], 1e-9),
               self.controller._imu_tick.count / max(t[-1], 1e-9),
               len(self.log) / max(t[-1], 1e-9)),
            "  MPC solve       %s, %.2f ms median, %.2f ms worst (%d solves)"
            % ("/".join(sorted(statuses)), np.median(solves), solves.max(),
               len(solves)),
            "  leg sweep       %.2f ms median, %.2f ms worst (%d sweeps)"
            % (np.median(sweeps), sweeps.max(), len(sweeps)),
            "  budget          MPC %.0f%% of %.0f ms at %.0f Hz; "
            "legs %.0f%% of %.1f ms at %.0f Hz  (medians)"
            % (100 * np.median(solves) / (1000 * cfg.MPC_DT), 1000 * cfg.MPC_DT,
               cfg.MPC_HZ,
               100 * np.median(sweeps) / (1000 * cfg.CONTROL_DT),
               1000 * cfg.CONTROL_DT, cfg.CONTROL_HZ),
        ]
        return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run DOG6's convex MPC in MuJoCo.")
    parser.add_argument("--headless", action="store_true",
                        help="no viewer; run a script and print a report")
    parser.add_argument("--script", default="forward", choices=sorted(SCRIPTS),
                        help="command timeline for a headless run")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds of simulated time")
    parser.add_argument("--vx", type=float, help="constant x-velocity command")
    parser.add_argument("--vy", type=float, help="constant y-velocity command")
    parser.add_argument("--yaw-rate", type=float,
                        help="constant yaw-rate command, deg/s")
    parser.add_argument("--no-gamepad", action="store_true",
                        help="ignore any pad; drive from the keyboard")
    args = parser.parse_args(argv)

    run = Run(duration=args.duration, script=args.script)
    if not args.headless and not args.no_gamepad:
        run.attach_pad()
    if args.vx is not None or args.vy is not None or args.yaw_rate is not None:
        # A constant command after a second of standing: the robot starts from
        # the stance keyframe with zero velocity, and stepping straight to a
        # command makes the first stride a transient that pollutes the report.
        held = trajectory.Command(
            vx=args.vx or 0.0, vy=args.vy or 0.0,
            yaw_rate=np.deg2rad(args.yaw_rate or 0.0))
        run.script = [(0.0, trajectory.Command()), (1.0, held)]
    return run.headless() if args.headless else run.interactive()


if __name__ == "__main__":
    sys.exit(main())
