"""The controller: three rates, set by three sources.

    20 Hz   MPC      solve the QP, keep the first force vector -- WORLD axes
    200 Hz  IMU      R arrives; fb = R' f is refreshed and then HELD
    250 Hz  encoder  q arrives; tau = J' fb is computed and written

THE MIDDLE RATE IS A SENSOR, NOT A DESIGN CHOICE
    R comes from the IMU at 200 Hz and q comes back from the encoders with the
    CAN sweep at 250 Hz.  They are different devices on different clocks, and
    the controller must not pretend otherwise: between IMU samples there is no
    new orientation, so the world-to-body rotation of the MPC's force can only
    be refreshed 200 times a second, and every torque written in between goes
    out through an attitude up to 8 ms old.

    Simulating that is the point.  Recomputing fb every sweep would invent IMU
    samples the hardware never produced, and would do it in exactly the axis
    -- orientation -- that decides whether a stance force pushes the robot up
    or sideways.  `_sensed` splits the simulator's perfect state along the
    sensor boundary and hands the controller only what the robot would have.

    It also means a freshly solved force waits up to one IMU period before it
    reaches a joint.  That latency is real and it is now in the loop.

WHY THE MPC IS ALLOWED TO BE THE SLOWEST
    The QP takes about 6 ms.  Running it at 250 Hz would be possible here and
    is not the point -- the paper's whole argument is that a horizon of
    planned wrenches beats a fast reaction.  Between solves the force is HELD,
    which is exactly the zero-order hold the discretisation assumes, so the
    plant sees the input the model predicted.

    The swing legs hold nothing: their reference is a continuous arc evaluated
    at the current time, so they move smoothly between solves.

THE STATE IS NOT ESTIMATED
    What `_sensed` hands out is still ground truth, just rate-limited.  There
    is no filter and no leg odometry -- the IMU split models the hardware's
    TIMING, not its noise.  This is the assumption the whole reproduction is
    built on (see `config`), and it means these numbers remain an upper bound
    on what the same controller does behind a real estimator.

THE ONE PIECE OF MEMORY
    Everything else here is a function of time and the measured state, but the
    swing arcs are not: a foot's arc is anchored at WHERE IT LIFTED OFF, which
    is a fact about the past.  `_liftoff` latches it on the stance -> swing
    transition.  Recomputing the arc from the current foot position each sweep
    instead would make the trajectory chase its own tail -- the foot would
    never reach the apex because the arc's start keeps moving to meet it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .. import coordinates as C
from .. import kinematics as K
from .. import params as P
from . import config as cfg
from . import dynamics as dyn
from . import gait
from . import qp
from . import swing
from . import trajectory


@dataclass
class BodyState:
    """Everything the controller reads, all of it ground truth.

    `position` and `velocity` are the TRUNK ORIGIN's, in world axes -- not the
    CoM's.  The controller converts to the CoM where the body model needs it
    and nowhere else, because the leg kinematics are written in the trunk
    frame and mixing the two is a silent 29 mm offset.
    """

    position: np.ndarray                 # trunk origin, world
    rotation: np.ndarray                 # R, body -> world
    rpy: np.ndarray                      # ZYX Euler of R
    velocity: np.ndarray                 # trunk origin, world
    omega: np.ndarray                    # world axes
    q: np.ndarray                        # (4, 3) joint angles
    qd: np.ndarray                       # (4, 3) joint rates

    def com_position(self) -> np.ndarray:
        return self.position + self.rotation @ dyn.COM_OFFSET_BODY

    def com_velocity(self) -> np.ndarray:
        # The CoM offset is fixed in the body frame, so the CoM's world
        # velocity picks up an w x r term relative to the trunk origin's.
        return self.velocity + np.cross(self.omega,
                                        self.rotation @ dyn.COM_OFFSET_BODY)

    def foot_world(self, leg: int) -> np.ndarray:
        return self.position + self.rotation @ K.foot_position(leg, self.q[leg])

    def hip_world(self, leg: int) -> np.ndarray:
        return self.position + self.rotation @ P.HIP_OFFSET[leg]

    def all_feet_world(self) -> np.ndarray:
        return np.stack([self.foot_world(i) for i in range(C.N_LEGS)])


@dataclass
class Telemetry:
    """What the last update did, for logging and for the self-test to read."""

    solved: bool = False
    status: str = "not solved"
    iterations: int = 0
    forces: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    contacts: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=bool))
    torque: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    clipped: int = 0
    non_finite: int = 0
    imu_ticks: int = 0
    solve_ms: float = 0.0
    foot_target: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))


class Ticker:
    """Fires on a FIXED-ORIGIN schedule: origin, origin+T, origin+2T, ...

    Used for both the 20 Hz MPC and the 200 Hz IMU, neither of whose periods
    is a multiple of the 4 ms control sweep (12.5 and 1.25 sweeps).

    THE OBVIOUS ALTERNATIVE IS WRONG AND LOOKS RIGHT.  Re-anchoring on the
    moment the tick actually fired -- `next = t + period` -- rounds the period
    UP to a whole sweep every time, and the rounding compounds instead of
    cancelling.  Measured on the MPC, that turned 50 ms into a rock-steady
    52 ms: a 19.23 Hz loop calling itself 20 Hz, with no jitter to hint at it.
    Counting from a fixed origin lets the intervals alternate about an exact
    mean instead.
    """

    def __init__(self, period: float):
        self.period = float(period)
        self.origin: float | None = None
        self.count = 0

    def due(self, t: float) -> bool:
        """True at most once per sweep, when this tick's deadline has passed."""
        if self.origin is None:
            self.origin = float(t)
        if t + 1e-9 >= self.origin + self.count * self.period:
            self.count += 1
            return True
        return False


class Controller:
    """Convex MPC + swing control.  `update(state, t)` -> (4, 3) joint torques.

    THREE RATES, SET BY THREE SOURCES:

        20 Hz   the MPC solves for f in WORLD axes
        200 Hz  the IMU delivers R, so fb = R' f is refreshed and then HELD
        250 Hz  the encoders deliver q, so tau = J' fb is written

    The middle one is a hardware fact, not a design choice: there is no R
    between IMU samples, so a torque written at 250 Hz is applied through an
    attitude up to 8 ms old.  See `cfg.IMU_HZ`.
    """

    def __init__(self, horizon: int | None = None):
        self.horizon = cfg.HORIZON if horizon is None else int(horizon)
        self.solver = qp.Solver(horizon=self.horizon)
        self.reference = trajectory.ReferenceTrajectory()
        self.command = trajectory.Command()
        self.telemetry = Telemetry()

        self._mpc_tick = Ticker(cfg.MPC_DT)      # 20 Hz
        self._imu_tick = Ticker(cfg.IMU_DT)      # 200 Hz

        self._forces = np.zeros((4, 3))          # MPC solution, WORLD axes
        self._forces_body = np.zeros((4, 3))     # fb = R' f, held between IMU
        self._imu: BodyState | None = None       # the held IMU sample
        self._liftoff = np.zeros((4, 3))         # world, latched at liftoff
        self._target = np.zeros((4, 3))          # world, planned touchdown
        self._contact_prev = np.ones(4, dtype=bool)
        self._q_hold = C.Q_STAND.copy()          # joint PD reference
        self._started = False
        self._started_body = False

    # -- the sensor split --------------------------------------------------
    def _sensed(self, truth: BodyState, t: float) -> tuple[BodyState, bool]:
        """One sweep's view of the robot: held IMU fields, fresh encoder fields.

        `truth` is everything the simulator can report, all of it perfectly
        current.  A real controller does not get that.  This splits it along
        the sensor boundary and hands back what the hardware would actually
        have at time `t`:

            from the IMU, refreshed at 200 Hz and held in between
                rotation, rpy, omega, and -- because the body position and
                velocity come from an IMU-driven estimator, not from any
                direct measurement -- position and velocity too
            from the encoders, fresh every 250 Hz sweep
                q, qd

        Returns the merged state and whether the IMU refreshed this sweep.
        """
        fresh = self._imu_tick.due(t)
        if fresh or self._imu is None:
            self._imu = truth
        held = self._imu
        return BodyState(position=held.position, rotation=held.rotation,
                         rpy=held.rpy, velocity=held.velocity,
                         omega=held.omega, q=truth.q, qd=truth.qd), fresh

    # -- the 20 Hz half ----------------------------------------------------
    def _predict_feet(self, state: BodyState, contacts) -> np.ndarray:
        """(horizon, 4, 3) foot positions relative to the CoM, world axes.

        A foot in stance stays where it is -- it is pinned to the ground.  A
        foot in swing is ASSUMED TO BE AT ITS PLANNED TOUCHDOWN for the steps
        after it lands, which is the one place this implementation is more
        careful than the reference one: holding today's swing-foot position
        across the horizon tells the MPC it can push from a point in mid-air.

        The CoM itself is propagated by the commanded velocity, so the moment
        arms shrink and grow as the body travels over the stance foot.  That
        coupling is most of what makes a horizon worth having.
        """
        feet_world = state.all_feet_world()
        com = state.com_position()
        velocity = self.reference.world_velocity(self.command)

        out = np.zeros((self.horizon, 4, 3))
        for step in range(self.horizon):
            com_at = com + velocity * (step + 1) * cfg.MPC_DT
            for leg in range(C.N_LEGS):
                if contacts[step, leg] and not self._contact_prev[leg]:
                    point = self._target[leg]       # will have landed by then
                elif contacts[step, leg]:
                    point = feet_world[leg]
                else:
                    point = self._target[leg]       # about to land there
                out[step, leg] = point - com_at
        return out

    def _command_for_body_model(self) -> trajectory.Command:
        """The operator's command, with z moved from the trunk to the CoM.

        `Command.z` and `cfg.Z_REF` are TRUNK ORIGIN heights -- that is what
        `params.STAND_HEIGHT` measures and what a ruler against the robot
        reads.  The MPC's state is the CoM, which sits 29 mm lower.  Handing
        the trunk height straight to the body model asks the CoM to stand
        where the trunk should be, and the robot rides high by most of that
        offset: measured, 18 mm of steady error and 32 % more vertical force
        than its own weight, with nothing in the logs that looks like a bug.
        """
        return replace(self.command,
                       z=self.command.z + float(dyn.COM_OFFSET_BODY[2]))

    def solve(self, state: BodyState, t: float) -> None:
        """Run one MPC solve and latch its first force vector."""
        started = time.perf_counter()
        contacts = gait.horizon_contacts(t, self.horizon, cfg.MPC_DT)
        reference = self.reference.horizon(self._command_for_body_model(),
                                           self.horizon, cfg.MPC_DT)

        # WRAP THE REFERENCE YAW ONTO THE MEASURED YAW'S BRANCH.
        #
        # The reference integrates yaw without bound -- after two full turns it
        # reads 12.6 rad -- while the measured yaw comes from `arctan2` and
        # lives in (-pi, pi].  The moment the robot passes half a turn the two
        # descriptions of the SAME heading differ by 2 pi, the yaw row of Q
        # (weighted 10, the largest in the vector) sees a 6.28 rad error, and
        # the MPC throws everything it has at correcting a heading that is
        # already correct.
        #
        # It is not subtle once seen: the robot fell at 7.9 s at 30 deg/s, 6.2 s
        # at 40, 5.3 s at 60 and 3.9 s at 90 -- which is 180 degrees of yaw in
        # every case.  Before the fix it looked like a turning-rate limit.
        #
        # The horizon is continuous in yaw, so ONE offset fixes the whole
        # block; wrapping each row separately would tear it at the branch cut.
        offset = 2.0 * np.pi * np.round(
            (reference[0, 2] - state.rpy[2]) / (2.0 * np.pi))
        reference[:, 2] -= offset

        x0 = trajectory.initial_state(state.com_position(), state.rpy,
                                      state.com_velocity(), state.omega)
        r_feet = self._predict_feet(state, contacts)
        a_seq, b_seq = dyn.discrete_sequence(reference[:, cfg.RPY][:, 2], r_feet)

        forces, _ = qp.solve_once(x0, reference, a_seq, b_seq, contacts,
                                  solver=self.solver)
        self._forces = forces
        self.reference.advance(self.command, cfg.MPC_DT)

        self.telemetry.solved = True
        self.telemetry.status = self.solver.last_status
        self.telemetry.iterations = self.solver.last_iterations
        self.telemetry.forces = forces.copy()
        # Timed HERE and not around `update`, which would fold the 250 Hz leg
        # loop into a number labelled "QP" and quietly triple it.
        self.telemetry.solve_ms = 1e3 * (time.perf_counter() - started)

    # -- the 250 Hz half ---------------------------------------------------
    def _update_swing_plan(self, state: BodyState, t: float, contacts) -> None:
        """Latch liftoff points and choose touchdown targets."""
        com_velocity = state.com_velocity()
        com = state.com_position()
        for leg in range(C.N_LEGS):
            lifting = self._contact_prev[leg] and not contacts[leg]
            if lifting or not self._started:
                self._liftoff[leg] = state.foot_world(leg)
            if not contacts[leg]:
                hip = state.hip_world(leg)
                # (33) asks for "the CoM velocity".  Under a yaw rate the hip
                # does not travel with the CoM, and using the CoM's velocity
                # makes the turn diverge -- see `swing.hip_velocity`.  With
                # w = 0 the two are identical, so this is (33) unchanged for
                # everything except turning.
                velocity = (swing.hip_velocity(com_velocity, state.omega, hip, com)
                            if cfg.YAW_PLACEMENT_CORRECTION else com_velocity)
                # Re-planned every sweep: the placement depends on the CURRENT
                # velocity, so a foot in the air keeps aiming at where the
                # body is going rather than where it was going at liftoff.
                self._target[leg] = swing.foot_placement(hip, velocity)
            else:
                self._target[leg] = state.foot_world(leg)
        self._contact_prev = np.asarray(contacts, dtype=bool).copy()

    def update(self, truth: BodyState, t: float) -> np.ndarray:
        """One 250 Hz sweep.  Returns (4, 3) joint torques, already clamped.

        `truth` is the simulator's full state; what the controller then works
        from is `_sensed`, which holds the IMU fields at 200 Hz and takes the
        encoder fields fresh.
        """
        state, imu_fresh = self._sensed(truth, t)

        if not self._started:
            self.reference.anchor(state.com_position(), state.rpy[2])
            self._q_hold = np.asarray(state.q, dtype=float).copy()

        contacts = gait.contact(t)
        self._update_swing_plan(state, t, contacts)
        self._started = True

        # 20 Hz: solve for f in WORLD axes.
        if self._mpc_tick.due(t):
            self.solve(state, t)

        # 200 Hz: rotate that force into body axes and HOLD it.
        #
        # ON THE IMU'S TICK, NOT THE SWEEP'S, AND NOT THE MPC'S.  R only
        # exists at 200 Hz, so this is the fastest fb can legitimately change.
        # A newly solved force therefore waits up to one IMU period before it
        # reaches the joints -- a real latency on the real robot, and one a
        # sim that recomputed fb every sweep would hide.
        if imu_fresh or not self._started_body:
            self._forces_body = swing.body_frame_force(self._forces,
                                                       state.rotation)
            self._started_body = True
            self.telemetry.imu_ticks += 1

        progress = gait.swing_phase(t)
        torque = np.zeros((4, 3))
        for leg in range(C.N_LEGS):
            q, qd = state.q[leg], state.qd[leg]
            if contacts[leg]:
                # 250 Hz: fresh encoder q, against the body-frame force that
                # the 200 Hz IMU tick last produced.
                torque[leg] = swing.stance_torque(leg, q, self._forces_body[leg])
                # THE JOINT FLOOR IS ON STANCE LEGS ONLY, AND IT IS PURE
                # DAMPING.  A stance leg is driven by a force law, which is
                # velocity-level and says nothing about where the joint should
                # be; a little damping is what keeps it from drifting.  A
                # SWING leg already has a complete operational-space law --
                # Cartesian PD plus Lambda plus the bias -- and adding joint
                # damping there opposes qd, which during a swing is several
                # rad/s.  That is a drag the feedforward does not know about,
                # so the swing tracks worse with the floor than without it.
                torque[leg] = torque[leg] + swing.joint_pd(q, qd, q)
            else:
                arc = swing.SwingTrajectory(self._liftoff[leg], self._target[leg])
                p_w, v_w, a_w = arc.at(progress[leg])
                p_b, v_b = swing.to_body(p_w, v_w, state.position,
                                         state.velocity, state.rotation)
                a_b = state.rotation.T @ a_w
                torque[leg] = swing.swing_torque(leg, q, qd, p_b, v_b, a_b)

        # np.clip DOES NOT FILTER NaN, AND THAT IS NOT AN ACADEMIC POINT.
        # NaN is neither above a bound nor below one, so `np.clip` returns it
        # unchanged: a single NaN arriving here leaves this function looking
        # clamped, lands in `data.ctrl`, and MuJoCo turns it into a NaN pose
        # and then dies INSIDE mj_step -- "mju_makeFrame: xaxis of contact
        # frame undefined" -- which is a fatal error, not an exception the
        # step returns.  On hardware the same value would be written to a
        # motor.  Zero is the only defensible substitute: the leg is told to
        # do nothing, and the count says it happened rather than hiding it.
        #
        # Nothing in this controller is KNOWN to produce a NaN -- the QP
        # guards its own solution and `operational_inertia` regularises the
        # one inverse that can go singular -- which is exactly why the guard
        # belongs at the boundary, where it costs one comparison per torque.
        finite = np.isfinite(torque)
        self.telemetry.non_finite = int(finite.size - int(finite.sum()))
        if self.telemetry.non_finite:
            torque = np.where(finite, torque, 0.0)

        clamped = np.clip(torque, -cfg.TAU_MAX, cfg.TAU_MAX)
        self.telemetry.clipped = int(np.count_nonzero(
            np.abs(torque) > cfg.TAU_MAX + 1e-9))
        self.telemetry.torque = clamped.copy()
        self.telemetry.contacts = np.asarray(contacts, dtype=bool).copy()
        self.telemetry.foot_target = self._target.copy()
        return clamped


# ===========================================================================
# reading a MuJoCo state without an estimator
# ===========================================================================
def state_from_mujoco(model, data) -> BodyState:
    """Ground truth out of MuJoCo, packed into `BodyState`.

    THE ANGULAR VELOCITY NEEDS CARE.  MuJoCo's free-joint `qvel[3:6]` is the
    body's angular velocity in the BODY frame, while everything in this
    controller -- and in the MPC's state vector -- is in world axes.  It has
    to be rotated.  Reading it as world is a bug that is invisible while the
    robot is level and steers it as soon as it is not.
    """
    import mujoco

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.TRUNK_BODY)
    rotation = data.xmat[trunk].reshape(3, 3).copy()

    quat = data.qpos[C.QPOS_ROOT_QUAT]
    rpy = _quat_to_rpy(quat)

    return BodyState(
        position=data.qpos[C.QPOS_ROOT_POS].copy(),
        rotation=rotation,
        rpy=rpy,
        velocity=data.qvel[C.QVEL_ROOT_LIN].copy(),
        omega=rotation @ data.qvel[C.QVEL_ROOT_ANG],
        q=C.q_from_qpos(data.qpos),
        qd=data.qvel[C.QVEL_JOINTS].reshape(C.N_LEGS, 3).copy(),
    )


def _quat_to_rpy(quat) -> np.ndarray:
    """ZYX Euler angles from a MuJoCo (w, x, y, z) quaternion.

    The pitch extraction is clamped: a quaternion that round-trips through
    floating point can leave the sine argument a few ulps outside [-1, 1], and
    `arcsin` then returns NaN, which reaches the QP and makes every force NaN.
    """
    w, x, y, z = np.asarray(quat, dtype=float)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return np.array([
        np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        np.arcsin(sin_pitch),
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    ])
