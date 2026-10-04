"""The walking reference: the operator's velocity in, a trunk reference out.

    WalkReference.reset(p_com_xy, yaw)                  anchor on the robot
    WalkReference.update(now, command, p_com_hat, yaw_hat)  -> RefSample

THE GENERATOR IS `sim.cmpc.trajectory`'S, IMPORTED, NOT COPIED
    `Command` (body-axis vx, vy, yaw rate) and `ReferenceTrajectory` (x, y and
    yaw integrated with the command rotated by the REFERENCE yaw -- never the
    measured one, so a heading disturbance cannot steer the reference).  The
    paper's rule holds here as there: the reference has xy velocity, xy
    position, yaw and yaw rate, and nothing else -- roll, pitch and their
    rates are zero, and height is the stand's.

THREE THINGS THE MPC DID NOT NEED AND A LAW WITHOUT A HORIZON DOES
    1. THE COMMAND IS SLEWED.  cMPC sees a velocity step coming over its
       horizon and spreads it; the SRB PD sees it as a step in its velocity
       error and the footstep planner as a step in the foothold.  So the
       command moves toward what the keys ask at `WALK_ACC_MAX` and
       `WALK_YAW_ACC_MAX`, and the slew's rate comes back as `a`, the
       feedforward acceleration the x/y rows add.
    2. HARDWARE LIMITS, not cMPC's 0.6 m/s / 90 deg/s, before
       `Command.clipped()` ever sees the command -- so cMPC's clip is a no-op.
    3. THE LEASH.  The integrated position has no physical meaning to track
       tightly (cMPC's Q weighs it 2 against z's 50), and with a PD on it a
       robot that falls behind is chased by a setpoint that keeps running.
       MIT's implementation clamps the desired position to within
       `max_pos_error` of the measured one; this does the same against the
       filter's CoM, and the same for yaw against the IMU's heading.

FRAMES
    p and v are the CoM's, WORLD x/y -- the run's world, whose x axis is the
    heading latched at the handover (`state.rezero_yaw`), which is the world
    the state estimator works in.  yaw is unwrapped: it is fed to
    `controller.latched_attitude` as an angle of a rotation matrix, so 400
    deg is as good as 40 and there is no branch cut to fall into (the one
    that toppled cMPC's turns was in an Euler-angle difference).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/trajectory.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim.cmpc.trajectory import Command, ReferenceTrajectory   # noqa: E402

from . import config as cfg          # noqa: E402

__all__ = ["Command", "RefSample", "WalkReference", "clip_command",
           "slew_command"]

#: s, the longest step the reference integrates.  A sweep that arrives late
#: (an overrun, a GC pause) integrates this much and no more, so one long
#: sweep cannot throw the reference forward.
DT_MAX = 0.02


class RefSample(NamedTuple):
    """The reference at one sweep.  WORLD frame, the CoM's."""

    p: np.ndarray            # (2,) m, CoM x/y
    v: np.ndarray            # (2,) m/s
    a: np.ndarray            # (2,) m/s^2, feedforward: the slew, rotated, and
                             #       the turn's r x v
    yaw: float               # rad, unwrapped
    yaw_rate: float          # rad/s
    command: Command         # body axes, as slewed -- what was integrated
    leashed: bool            # the leash moved p or yaw this sweep


def clip_command(command: Command, vx_max: float = cfg.WALK_VX_MAX,
                 vy_max: float = cfg.WALK_VY_MAX,
                 yaw_rate_max: float = cfg.WALK_YAW_RATE_MAX) -> Command:
    """`command` inside the HARDWARE box, z untouched."""
    def box(x, m):
        return min(max(float(x), -m), m)
    return Command(vx=box(command.vx, vx_max), vy=box(command.vy, vy_max),
                   yaw_rate=box(command.yaw_rate, yaw_rate_max), z=command.z)


def slew_command(current: Command, target: Command, dt: float,
                 acc_max: float = cfg.WALK_ACC_MAX,
                 yaw_acc_max: float = cfg.WALK_YAW_ACC_MAX) -> Command:
    """`current` moved toward `target` by at most acc * dt per axis."""
    def toward(a: float, b: float, step: float) -> float:
        return a + min(max(b - a, -step), step)
    return Command(vx=toward(current.vx, target.vx, acc_max * dt),
                   vy=toward(current.vy, target.vy, acc_max * dt),
                   yaw_rate=toward(current.yaw_rate, target.yaw_rate,
                                   yaw_acc_max * dt),
                   z=current.z)


class WalkReference:
    """cMPC's reference generator, slewed, clipped and leashed, per sweep."""

    def __init__(self, *, vx_max: float = cfg.WALK_VX_MAX,
                 vy_max: float = cfg.WALK_VY_MAX,
                 yaw_rate_max: float = cfg.WALK_YAW_RATE_MAX,
                 acc_max: float = cfg.WALK_ACC_MAX,
                 yaw_acc_max: float = cfg.WALK_YAW_ACC_MAX,
                 leash: float = cfg.WALK_LEASH,
                 yaw_leash: float = cfg.WALK_YAW_LEASH):
        self.vx_max, self.vy_max = float(vx_max), float(vy_max)
        self.yaw_rate_max = float(yaw_rate_max)
        self.acc_max, self.yaw_acc_max = float(acc_max), float(yaw_acc_max)
        self.leash, self.yaw_leash = float(leash), float(yaw_leash)
        self.traj = ReferenceTrajectory()
        self.command = Command(vx=0.0, vy=0.0, yaw_rate=0.0)
        self.t_prev: float | None = None
        #: How often the leash had to act, for the exit report.
        self.leash_sweeps = 0

    def reset(self, p_com_xy, yaw: float) -> None:
        """Anchor the reference ON THE ROBOT and stop it: the first sweep after
        this has zero position, heading and velocity error -- so engaging the
        walk cannot step the wrench."""
        self.traj = ReferenceTrajectory()
        self.traj.anchor((float(p_com_xy[0]), float(p_com_xy[1])), float(yaw))
        self.command = Command(vx=0.0, vy=0.0, yaw_rate=0.0)
        self.t_prev = None
        self.leash_sweeps = 0

    def sample(self, leashed: bool = False,
               a_body=(0.0, 0.0)) -> RefSample:
        """The reference as it stands, without advancing it."""
        c = self.command
        yaw = float(self.traj.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        v = np.array([cy * c.vx - sy * c.vy, sy * c.vx + cy * c.vy])
        # d/dt [Rz(yaw) v_b] = Rz(yaw) (a_b + r x v_b): the slew, plus the
        # turn carrying the velocity round with the heading.
        ab = np.array([a_body[0] - c.yaw_rate * c.vy,
                       a_body[1] + c.yaw_rate * c.vx])
        a = np.array([cy * ab[0] - sy * ab[1], sy * ab[0] + cy * ab[1]])
        return RefSample(p=np.array([self.traj.x, self.traj.y]), v=v, a=a,
                         yaw=yaw, yaw_rate=float(c.yaw_rate), command=c,
                         leashed=bool(leashed))

    def update(self, now: float, target: Command, p_com_hat=None,
               yaw_hat: float | None = None) -> RefSample:
        """Slew toward `target`, integrate one step, leash to the estimate.

        `p_com_hat` (2,) is the filter's CoM x/y and `yaw_hat` the IMU's
        heading, both in the run's world; None skips that half of the leash
        (no estimate fed yet, or one refused this sweep).
        """
        now = float(now)
        dt = 0.0 if self.t_prev is None else min(max(now - self.t_prev, 0.0),
                                                  DT_MAX)
        self.t_prev = now
        target = clip_command(target, self.vx_max, self.vy_max,
                              self.yaw_rate_max)
        before = self.command
        self.command = slew_command(before, target, dt, self.acc_max,
                                    self.yaw_acc_max)
        a_body = ((0.0, 0.0) if dt <= 0.0 else
                  ((self.command.vx - before.vx) / dt,
                   (self.command.vy - before.vy) / dt))
        if dt > 0.0:
            # cMPC's integrator: the body command rotated by the REFERENCE
            # yaw, the yaw advanced after.
            self.traj.advance(self.command, dt)
        leashed = False
        if p_com_hat is not None:
            e = np.array([self.traj.x - float(p_com_hat[0]),
                          self.traj.y - float(p_com_hat[1])])
            n = float(np.hypot(e[0], e[1]))
            if n > self.leash:
                e *= self.leash / n
                self.traj.x = float(p_com_hat[0]) + float(e[0])
                self.traj.y = float(p_com_hat[1]) + float(e[1])
                leashed = True
        if yaw_hat is not None:
            e_yaw = float(self.traj.yaw) - float(yaw_hat)
            # Compare on the circle, keep the reference's own branch.
            wrapped = math.remainder(e_yaw, 2.0 * math.pi)
            if abs(wrapped) > self.yaw_leash:
                self.traj.yaw = float(self.traj.yaw) - wrapped + math.copysign(
                    self.yaw_leash, wrapped)
                leashed = True
        if leashed:
            self.leash_sweeps += 1
        return self.sample(leashed, a_body)


def describe() -> str:
    ref = WalkReference()
    ref.reset((0.0, 0.0), 0.0)
    target = Command(vx=0.15, vy=0.0, yaw_rate=float(np.radians(30.0)))
    rows = ["DOG6 walking reference (sim.cmpc.trajectory, slewed, clipped, "
            "leashed)",
            "  limits      vx %.2f  vy %.2f m/s  yaw %.0f deg/s"
            % (cfg.WALK_VX_MAX, cfg.WALK_VY_MAX,
               np.degrees(cfg.WALK_YAW_RATE_MAX)),
            "  slew        %.2f m/s^2  %.0f deg/s^2    leash %.0f mm / %.0f deg"
            % (cfg.WALK_ACC_MAX, np.degrees(cfg.WALK_YAW_ACC_MAX),
               1e3 * cfg.WALK_LEASH, np.degrees(cfg.WALK_YAW_LEASH)),
            "  command %s, from rest, no estimate:" % target,
            "     t (s)    x (mm)   y (mm)  yaw (deg)  vx_b    r (deg/s)"]
    for k in range(0, 501):
        s = ref.update(0.004 * k, target)
        if k % 50 == 0:
            rows.append("    %5.2f  %8.1f %8.1f  %8.2f  %6.3f  %8.2f"
                        % (0.004 * k, 1e3 * s.p[0], 1e3 * s.p[1],
                           np.degrees(s.yaw), s.command.vx,
                           np.degrees(s.yaw_rate)))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
