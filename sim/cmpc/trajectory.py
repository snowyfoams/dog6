"""Operator command -> the reference trajectory the MPC tracks.

THE RULE, VERBATIM FROM THE PAPER
    The reference contains non-zero xy-velocity, xy-position, z position, yaw
    and yaw rate, and NOTHING ELSE.  Roll, pitch, roll rate, pitch rate and
    z-velocity are always zero.  Every parameter is commanded directly by the
    operator except yaw and xy-position, which are obtained by integrating the
    commanded yaw rate and xy-velocity.

    That is the entire trajectory generator.  It is worth being precise about
    why it is allowed to be this thin: the MPC is not being asked to follow a
    path through space, it is being asked to hold a velocity.  A quadruped
    tracking a commanded velocity has no absolute xy reference to speak of --
    the integral below is a bookkeeping device that turns "go forward at
    0.3 m/s" into something the position rows of Q can penalise, not a claim
    about where the robot is supposed to be in the world.

WHY THE INTEGRAL IS A *STATE*, NOT A RECOMPUTATION
    `ReferenceTrajectory` holds x, y and yaw and advances them by the
    commanded velocity each solve.  It is therefore path-dependent: the
    reference remembers where it has been told to go, and the robot's actual
    position never feeds back into it.

    That is deliberate and it is the source of one honest limitation.  If the
    robot is pushed and falls behind, the reference does not wait -- the
    position error grows, and the position rows of Q pull the robot back
    toward a point it may be a long way from.  The paper's answer is that the
    position weights are low (2.0 against z's 50.0) precisely so this pull is
    gentle.  The alternative -- re-anchoring the reference on the measured
    position every solve -- makes the position term do nothing at all, since
    the error would be reset to zero each time.  Neither is obviously right;
    this is the paper's choice.

DIRECTION: THE COMMAND IS IN BODY AXES, THE REFERENCE IS IN WORLD AXES
    An operator pressing W means "forward", which is the robot's own forward,
    not the world's +x.  So the commanded velocity is rotated by the REFERENCE
    yaw -- not the measured yaw -- before it is integrated or written into the
    velocity rows.  Using the measured yaw would couple the reference to the
    robot's own heading error and let a yaw disturbance steer the reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg


@dataclass
class Command:
    """What the operator's stick is asking for, in BODY axes.

    `vx` forward, `vy` left, `yaw_rate` counter-clockwise.  The keys that move
    these are W/S, A/D and Q/E, one `cfg.V_STEP` or `cfg.YAW_RATE_STEP` each.
    """

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    z: float = cfg.Z_REF

    def clipped(self) -> "Command":
        return Command(
            vx=float(np.clip(self.vx, -cfg.V_MAX, cfg.V_MAX)),
            vy=float(np.clip(self.vy, -cfg.V_MAX, cfg.V_MAX)),
            yaw_rate=float(np.clip(self.yaw_rate, -cfg.YAW_RATE_MAX,
                                   cfg.YAW_RATE_MAX)),
            z=float(self.z),
        )

    def is_zero(self, tol: float = 1e-9) -> bool:
        return (abs(self.vx) < tol and abs(self.vy) < tol
                and abs(self.yaw_rate) < tol)

    def __str__(self) -> str:
        return "vx %+.2f  vy %+.2f  yaw_rate %+5.1f deg/s  z %.3f" % (
            self.vx, self.vy, np.rad2deg(self.yaw_rate), self.z)


@dataclass
class ReferenceTrajectory:
    """The integrated part of the reference: x, y and yaw.

    Everything else in the reference is either commanded directly or
    identically zero, so these three are the whole memory of the generator.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    _initialised: bool = field(default=False, repr=False)

    def anchor(self, position, yaw: float) -> None:
        """Place the reference on the robot's current pose.

        Called ONCE, at the start of a run, so the reference does not begin
        with a position error the robot never had a chance to cause.  Calling
        it every solve would make the position terms of the cost identically
        zero -- see the module docstring.
        """
        self.x, self.y = float(position[0]), float(position[1])
        self.yaw = float(yaw)
        self._initialised = True

    def advance(self, command: Command, dt: float) -> None:
        """Integrate the commanded velocities forward by one MPC period."""
        command = command.clipped()
        # Rotate the body-frame command into world axes using the REFERENCE
        # yaw.  Advancing yaw first, or using the measured yaw, both let the
        # heading error leak into the translation reference.
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * command.vx - s * command.vy) * dt
        self.y += (s * command.vx + c * command.vy) * dt
        self.yaw += command.yaw_rate * dt

    def world_velocity(self, command: Command) -> np.ndarray:
        """The commanded body velocity expressed in world axes.  vz is always 0."""
        command = command.clipped()
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * command.vx - s * command.vy,
                         s * command.vx + c * command.vy,
                         0.0])

    def horizon(self, command: Command, horizon: int | None = None,
                dt: float | None = None) -> np.ndarray:
        """(horizon, STATE_DIM) reference for steps 1..horizon ahead.

        Row i is the desired state at ``t + (i + 1) * dt`` -- the state the
        dynamics reach after applying u_i -- which is what the cost
        ``||x_{i+1} - x_{i+1,ref}||_Q`` compares against.

        The generator's own x, y and yaw are NOT advanced by this call.  It
        rolls a local copy forward across the horizon and leaves the committed
        reference where it was; `advance` is what moves that, once per solve.
        """
        horizon = cfg.HORIZON if horizon is None else int(horizon)
        dt = cfg.MPC_DT if dt is None else float(dt)
        command = command.clipped()

        ref = np.zeros((horizon, cfg.STATE_DIM))
        x, y, yaw = self.x, self.y, self.yaw

        for i in range(horizon):
            # Step the local copy first: row i describes t + (i+1)*dt.
            c, s = np.cos(yaw), np.sin(yaw)
            vx_w = c * command.vx - s * command.vy
            vy_w = s * command.vx + c * command.vy
            x += vx_w * dt
            y += vy_w * dt
            yaw += command.yaw_rate * dt

            ref[i, cfg.RPY] = (0.0, 0.0, yaw)     # roll and pitch: always 0
            ref[i, cfg.POS] = (x, y, command.z)
            ref[i, cfg.OMEGA] = (0.0, 0.0, command.yaw_rate)   # rates: only yaw
            ref[i, cfg.VEL] = (vx_w, vy_w, 0.0)   # vz: always 0
            ref[i, cfg.GRAV] = -9.81

        return ref


def initial_state(position, rpy, velocity, omega) -> np.ndarray:
    """Pack an observed body pose into the 13-vector the MPC integrates.

    The thirteenth entry is gravity, a constant the dynamics read and nothing
    ever writes -- see `config` for why it is carried as a state at all.
    """
    x = np.zeros(cfg.STATE_DIM)
    x[cfg.RPY] = np.asarray(rpy, dtype=float)
    x[cfg.POS] = np.asarray(position, dtype=float)
    x[cfg.OMEGA] = np.asarray(omega, dtype=float)
    x[cfg.VEL] = np.asarray(velocity, dtype=float)
    x[cfg.GRAV] = -9.81
    return x


def describe(command: Command | None = None) -> str:
    command = Command(vx=0.3, yaw_rate=np.deg2rad(20.0)) if command is None else command
    ref = ReferenceTrajectory()
    rows = ["DOG6 reference trajectory", "  command         %s" % command,
            "",
            "  step   yaw     x       y       z      vx      vy    wz"]
    for i, row in enumerate(ref.horizon(command)):
        rows.append("  %4d  %6.3f  %6.3f  %6.3f  %.4f  %6.3f  %6.3f  %5.2f"
                    % (i + 1, row[2], row[3], row[4], row[5],
                       row[9], row[10], row[8]))
    rows.append("")
    rows.append("  roll, pitch, roll rate, pitch rate and vz are 0 in every row.")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
