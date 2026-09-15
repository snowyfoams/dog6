"""DOG6 stand: position mode -> Cartesian compliance -> position mode.  NO IMU.

Four phases, stepped by pressing ENTER:

    0 settle   position mode, holding whatever pose the model started in
    1 crouch   position mode, interpolating to Q_CROUCH -- trunk down on the
               floor, shins vertical, a pose that holds at zero torque
    2 lift     TORQUE mode, Cartesian compliance at each foot: xy pinned,
               z driven down under the hip, which is what lifts the trunk
    3 park     position mode, back to Q_CROUCH

The point of the sequence is the handover in the middle.  Phases 0, 1 and 3
are a joint-space servo that does not know the robot has a body; phase 2 is a
force law at the feet that does not know what the joints are doing.  They meet
at one pose, and if the model, the frames or the sign of a Jacobian is wrong,
they disagree there visibly rather than subtly.


THERE IS NO IMU IN THIS FILE, ON PURPOSE
    `model/dog6.xml` carries `imu_quat`, `imu_gyro` and `imu_acc`, and nothing
    here reads them.  Trunk height comes from the twelve JOINT ENCODERS alone:

        h = FOOT_RADIUS - mean_over_legs( foot_position(leg, q).z )

    `foot_position` returns the foot site in the TRUNK frame, so for a level
    trunk with all four balls on the floor that inverts to the trunk origin's
    height above the floor.  `height_from_fk` is the one function that does it.

    BOTH OF THOSE ARE ASSUMPTIONS, NOT MEASUREMENTS.  Leg FK cannot see the
    trunk's ORIENTATION and cannot see whether a foot is actually loaded.  Tip
    the robot and this number is the height of a trunk that is not level; lift
    a foot and it averages in a leg that is measuring nothing.  It is honest
    here because this sequence never leaves flat ground with four feet planted
    -- and it is exactly the quantity a real DOG6 can compute before its IMU is
    mounted, which is why the sequence is built to need only this.

    `coordinates.R_BODY_IMU` is still the identity placeholder.  When the board
    is mounted, the estimator that replaces this function belongs in its own
    module, not bolted on here.


POSITION MODE IS EMULATED, AND THAT IS A REAL DIFFERENCE FROM HARDWARE
    Every actuator in `model/dog6.xml` is a plain `<motor>` with gear=1, so the
    only thing this file can write is TORQUE.  "Position mode" below is a stiff
    joint-space PD closed in Python at the sim rate:

        tau = KP_JOINT (q_des - q) - KD_JOINT qd

    The MG5010 drivers have their own position mode, closed in the driver at a
    rate this loop does not see and with gains that are NOT KP_JOINT/KD_JOINT
    (`hw/motor/motor_gains.py` holds the driver's own numbers).  So the gains
    here are a SIM tuning, not a hardware one, and the phase-1 and phase-3
    trajectories are what transfers -- not the servo that follows them.

    Deliberately there is no gravity compensation in position mode: a driver's
    position loop does not have any, and the steady-state droop it produces is
    real.  Phase 2 does compensate, because a compliance law without a gravity
    feedforward parks at the wrong height by weight/stiffness and would hide
    the very error this sequence is meant to expose.


THE COMPLIANCE LAW
    Per leg, everything in that leg's own HIP frame:

        f = KP_CART (p_des - p) - KD_CART (J qd) + f_ff
        tau = J^T f + leg_gravity_torque

    with ``f_ff = (0, 0, -WEIGHT/4)``.  THE MINUS IS THE WHOLE SIGN STORY: `f`
    is the force the LEG APPLIES TO THE WORLD, so holding the robot up means
    pushing DOWN on the ground.  Passing +WEIGHT/4 does not sag, it commands
    the robot to pull itself into the floor -- measured, it drops from 0.1925 m
    to 0.0349 m in 1.5 s, which is the trunk box on the ground.

    `p_des` keeps the xy it had at the end of phase 1 and moves only in z.  A
    foot that is planted cannot follow z, so the spring compresses and its
    reaction lifts the trunk; the commanded z is therefore a HEIGHT command
    wearing a position command's clothes.  That is the point of doing it in
    Cartesian space rather than in joints: the same three gains mean the same
    thing at every pose, where a joint-space stiffness does not.

Run it:

    python -m sim.stand                 # viewer; ENTER steps the phases,
                                        # pressed IN THE WINDOW or on the terminal
    python -m sim.stand --headless      # no window, auto-steps, prints a table
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

import mujoco

if __package__ in (None, ""):        # allow `python sim/stand.py` too
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402
from . import kinematics as K        # noqa: E402
from . import params as P            # noqa: E402

__all__ = ["CROUCH_HEIGHT", "LIFT_HEIGHT", "foot_targets", "pose_for_height",
           "height_from_fk", "StandController", "run"]


# ===========================================================================
# the crouch, the lift, and the gains
# ===========================================================================
#: THE CROUCH: TRUNK ON THE FLOOR, SHINS VERTICAL.  Two conditions, one per
#: planar degree of freedom, and both are there for the same reason -- a park
#: pose must survive losing torque.
#:
#:   trunk on the floor   the trunk origin at HOME_HEIGHT puts the collision
#:                        box on the ground, so the BODY carries the weight and
#:                        the twelve motors carry only their own links
#:   shin vertical        the ground reaction under each foot runs THROUGH the
#:                        knee axis, so the knee sits at a dead point
#:
#: MEASURED, RELEASED TO ZERO TORQUE FOR 3 s: the trunk drops 0.1 mm and the
#: joints drift 1.5 deg.  Holding the legs' own weight needs 0.229 N*m and
#: even that is not required to stay put.
#:
#: This is the third crouch this file has had, and the first that is actually a
#: park.  The first was an IK solve for a chosen height with the feet at their
#: Q_STAND xy: it holds, but it leaves the shin 58 deg off vertical, so the
#: KNEE carries everything -- 1.250 N*m of a 1.250 N*m maximum -- and folds the
#: moment torque is lost.  The second stood the shin up but left the thigh
#: horizontal, which zeroes the knee (0.013 N*m, measured) and moves 1.02 N*m
#: onto the pitch joint, because the trunk is still in the air and something
#: has to hold it.  Only putting the trunk down removes the load entirely.
def _crouch_pose() -> np.ndarray:
    """Q_CROUCH, derived from the two conditions rather than tabulated.

    SHIN VERTICAL fixes the sum.  ``r_shin = rot_x(abd) rot_z(pitch + knee)``,
    so the shin link's own axis lands at ``(sx cos th, sx sin th cos abd,
    sx sin th sin abd)`` with ``sx = sign(knee_to_foot.x)`` -- the sign that
    carries the front/rear mirror, since `link_rotations` has no per-leg sign
    of its own and the mirror lives in the offsets.  With abd = -+90 deg that
    is (0, 0, -1) for exactly one value:

        theta = -sign(abd) * sx * pi/2

    TRUNK ON THE FLOOR fixes the split.  With the shin vertical the only z the
    chain gains from the thigh is ``L2 sx sin(pitch) sin(abd)``, the hip elbow
    is pure x and contributes none, and the shin contributes a fixed
    ``-|knee_to_foot.x|``.  So the height condition is one arcsin, not a solve:

        sin(pitch) = (FOOT_RADIUS + |knee_to_foot.x| - HOME_HEIGHT)
                     / (L2 sx sin(abd))

    The principal branch is the one with the knee ABOVE the hip, which is the
    pose that folds the leg up out of the way rather than under the body.
    """
    q = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
    for i in range(C.N_LEGS):
        abd = C.Q_STAND[i, 0]
        sx = float(np.sign(P.KNEE_TO_FOOT[i, 0]))
        theta = -np.sign(abd) * sx * np.pi / 2
        drop = P.FOOT_RADIUS + abs(P.KNEE_TO_FOOT[i, 0]) - P.HOME_HEIGHT
        pitch = float(np.arcsin(drop / (P.L2 * sx * np.sin(abd))))
        q[i] = [abd, pitch, theta - pitch]
    return q


Q_CROUCH = _crouch_pose()

#: Where Q_CROUCH puts the trunk origin.  DERIVED by leg FK from the pose, and
#: it comes out at HOME_HEIGHT because that is what the pose was solved for --
#: the two agreeing is the cheap check that the derivation above is right.
CROUCH_HEIGHT = float(P.FOOT_RADIUS - K.all_foot_positions(Q_CROUCH)[:, 2].mean())

#: The foot xy the whole sequence pins, read off Q_CROUCH.  A vertical shin
#: puts the foot under the knee, which the folded thigh carries 80.9 mm out
#: from the hip -- 55 mm further out than Q_STAND's stance.  That splay is not
#: a tuning choice, it is what "shin vertical" means on this geometry.
FOOT_XY = K.hip_to_foot_stance(Q_CROUCH)[:, :2].copy()

#: Where phase 2 lifts to.  STAND_HEIGHT is reachable from this crouch at
#: 83.6 % of LEG_REACH -- more than Q_STAND's own 76.9 %, because of the splay
#: above, but nowhere near the 93.9 % the previous thigh-horizontal crouch
#: needed, which was DOG5's number and the thing DOG6 exists to get away from.
LIFT_HEIGHT = P.STAND_HEIGHT

#: Position-mode joint PD.  [SIM TUNING -- NOT THE DRIVER'S GAINS]
KP_JOINT = 120.0                    # N*m/rad
KD_JOINT = 3.0                      # N*m*s/rad

#: Cartesian compliance at the foot, in the hip frame.  [SIM TUNING]
#: xy is stiffer than z on purpose: xy is a constraint the sequence never asks
#: to move, so it should hold hard, while z is the axis being commanded and a
#: softer z is what makes the lift compliant rather than a position servo.
KP_CART = np.array([1500.0, 1500.0, 1000.0])    # N/m
KD_CART = np.array([30.0, 30.0, 25.0])          # N*s/m

#: Phase durations, seconds.  The lift is slower because it is the phase with
#: contact in the loop.
RAMP_POSITION = 1.5
RAMP_LIFT = 3.0        # 157 mm of lift, against the crouch's 0 mm


# ===========================================================================
# poses
# ===========================================================================
def foot_targets(height: float) -> np.ndarray:
    """(4, 3) foot positions in each leg's OWN HIP frame for a level trunk.

    xy is FOOT_XY -- the crouch's, pinned -- and only z carries `height`.
    Every hip offset has z = 0 (the four abduction hinges lie in the trunk
    origin's plane, see `coordinates`, "THE TRUNK FRAME"), so a foot's z is the
    same number in the hip frame and in the trunk frame and needs no per-leg
    correction here.
    """
    p = np.zeros((C.N_LEGS, 3))
    p[:, :2] = FOOT_XY
    p[:, 2] = -(float(height) - P.FOOT_RADIUS)
    return p


def pose_for_height(height: float, q_seed=None) -> np.ndarray:
    """(4, 3) joint angles putting all four feet at `foot_targets(height)`."""
    return K.all_leg_ik(foot_targets(height), q_seed=q_seed)


def height_from_fk(q) -> float:
    """Trunk-origin height above the floor, from the JOINT ENCODERS ONLY.

    No IMU, no base state, no contact sensing.  Assumes a level trunk and four
    planted feet -- see this module's docstring for why that is honest here and
    where it stops being so.
    """
    return float(P.FOOT_RADIUS - K.all_foot_positions(q)[:, 2].mean())


# ===========================================================================
# the controller
# ===========================================================================
class StandController:
    """The four-phase sequence, as a MuJoCo control callback.

    Install with ``mujoco.set_mjcb_control(controller)`` and it runs inside
    MuJoCo's own step, which is what lets the managed viewer drive it -- the
    viewer owns the physics thread and will not call user code any other way.
    ``request_advance()`` is safe to call from another thread: it only sets a
    flag, which the callback consumes at a step boundary.
    """

    PHASES = ("settle", "crouch", "lift", "park", "done")
    BLURB = {
        "settle": "position mode, holding the start pose",
        "crouch": "position mode -> crouch",
        "lift":   "TORQUE mode, Cartesian compliance, xy pinned, z lifts",
        "park":   "position mode -> crouch",
        "done":   "position mode, holding crouch",
    }

    def __init__(self) -> None:
        self.q_crouch = Q_CROUCH.copy()
        # The xy the feet keep for the whole of phase 2.  Read off Q_CROUCH
        # rather than off the achieved pose, so a droopy phase 1 cannot smuggle
        # its steady-state error into the compliance reference.
        self.foot_xy = FOOT_XY.copy()

        self.phase = 0
        self.t_phase = 0.0
        self.q_ref0 = self.q_crouch.copy()
        self.h_cmd = CROUCH_HEIGHT
        self.h_fk = float("nan")
        self.tau = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        self._advance = False

    # -- lifecycle ---------------------------------------------------------
    def reset(self, data) -> None:
        """Latch the pose the sequence starts from.  Call once, before step 1."""
        self.q_ref0 = C.q_from_qpos(data.qpos)
        self.phase = 0
        self.t_phase = float(data.time)

    def request_advance(self) -> None:
        """Ask for the next phase.  Thread-safe; takes effect at a step edge."""
        self._advance = True

    @property
    def phase_name(self) -> str:
        return self.PHASES[self.phase]

    @property
    def finished(self) -> bool:
        return self.phase_name == "done"

    def _enter_next(self, data, q) -> None:
        if self.finished:
            return
        self.phase += 1
        self.t_phase = float(data.time)
        # Every position-mode phase interpolates FROM WHERE THE ROBOT IS, not
        # from where the last phase wished it were.  After phase 2 those differ
        # by whatever the compliance law settled at, and starting a stiff PD
        # from a stale reference is a step input into a 120 N*m/rad spring.
        self.q_ref0 = q.copy()

    # -- the two laws ------------------------------------------------------
    def _position(self, q_des, q, qd) -> np.ndarray:
        """Stiff joint PD.  No gravity term -- a driver's position loop has none."""
        return KP_JOINT * (q_des - q) - KD_JOINT * qd

    def _compliance(self, q, qd, height) -> np.ndarray:
        """Cartesian spring-damper at each foot, plus the weight feedforward."""
        tau = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        z_des = -(height - P.FOOT_RADIUS)
        for i in range(C.N_LEGS):
            p = K.foot_position_hip(i, q[i])
            jac = K.foot_jacobian(i, q[i])
            p_des = np.array([self.foot_xy[i, 0], self.foot_xy[i, 1], z_des])
            # f is what the LEG APPLIES TO THE WORLD.  f_ff pushes DOWN.
            force = (KP_CART * (p_des - p) - KD_CART * (jac @ qd[i])
                     + np.array([0.0, 0.0, -P.WEIGHT / C.N_LEGS]))
            tau[i] = jac.T @ force + K.leg_gravity_torque(i, q[i])
        return tau

    # -- the callback ------------------------------------------------------
    def __call__(self, model, data) -> None:
        q = C.q_from_qpos(data.qpos)
        qd = data.qvel[C.QVEL_JOINTS].reshape(C.N_LEGS, C.N_JOINTS_PER_LEG)
        self.h_fk = height_from_fk(q)

        # THE VIEWER CAN REWIND TIME UNDER US.  Its reset key puts `data` back
        # at qpos0 -- the robot flat on its belly -- while this object happily
        # keeps whatever phase it was in, and a stiff PD aimed at a standing
        # reference from a belly-flat start saturates all twelve motors at
        # TAU_MAX_SIM.  That is what a reset looked like in an earlier run.
        # Time running backwards is the one signal that says it happened.
        if float(data.time) < self.t_phase:
            self.reset(data)

        if self._advance:
            self._advance = False
            self._enter_next(data, q)

        elapsed = float(data.time) - self.t_phase
        name = self.phase_name

        if name == "settle":
            self.h_cmd = height_from_fk(self.q_ref0)   # whatever it started at
            tau = self._position(self.q_ref0, q, qd)
        elif name in ("crouch", "park"):
            alpha = min(1.0, elapsed / RAMP_POSITION)
            self.h_cmd = CROUCH_HEIGHT
            tau = self._position(self.q_ref0 + alpha * (self.q_crouch - self.q_ref0),
                                 q, qd)
        elif name == "lift":
            alpha = min(1.0, elapsed / RAMP_LIFT)
            self.h_cmd = CROUCH_HEIGHT + alpha * (LIFT_HEIGHT - CROUCH_HEIGHT)
            tau = self._compliance(q, qd, self.h_cmd)
        else:                                            # done
            self.h_cmd = CROUCH_HEIGHT
            tau = self._position(self.q_crouch, q, qd)

        # params.JOINT_LIMITS is a policy the MJCF does not enforce; the torque
        # cap is the one limit this loop is responsible for.
        self.tau = np.clip(tau, -P.TAU_MAX_SIM, P.TAU_MAX_SIM)
        data.ctrl[:] = self.tau.reshape(-1)

    # -- reporting ---------------------------------------------------------
    def status(self, data) -> str:
        return ("t=%6.2f s  phase %d %-6s  h_cmd=%.4f  h_fk=%.4f  "
                "err=%+7.2f mm  |tau|max=%5.2f N*m"
                % (data.time, self.phase, self.phase_name, self.h_cmd, self.h_fk,
                   1000.0 * (self.h_fk - self.h_cmd), np.abs(self.tau).max()))


# ===========================================================================
# running it
# ===========================================================================
def _load():
    model = mujoco.MjModel.from_xml_path(P.XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, C.KEYFRAMES["stand"])
    mujoco.mj_forward(model, data)
    return model, data


def run_headless(seconds_per_phase: float = 4.0, quiet: bool = False):
    """Auto-step the whole sequence with no window.  Returns the controller.

    This is the form the self-test and any regression check should use: it is
    deterministic, it needs no display, and it exercises exactly the callback
    the viewer runs.
    """
    model, data = _load()
    ctrl = StandController()
    ctrl.reset(data)
    steps = int(round(seconds_per_phase / model.opt.timestep))

    rows = []
    for _ in range(len(StandController.PHASES)):
        for k in range(steps):
            ctrl(model, data)                    # called directly: no callback
            mujoco.mj_step(model, data)
            if not quiet and k % (steps // 2) == 0:
                print("   " + ctrl.status(data))
        rows.append((ctrl.phase_name, ctrl.h_cmd, ctrl.h_fk, float(data.qpos[2])))
        if ctrl.finished:
            break
        if not quiet:
            print("-- ENTER --")
        ctrl.request_advance()
    return ctrl, data, rows


#: GLFW key codes.  ENTER is not one of the viewer's own shortcuts, so it
#: reaches a user callback untouched.  SPACE is the viewer's PAUSE and must
#: not be borrowed for this.
KEY_ENTER = (257, 335)                  # Return, keypad Return


def run():
    """Open the viewer and step the sequence with ENTER.

    ENTER is wired up TWO ways, because neither alone survives every way this
    gets launched:

      * in the VIEWER WINDOW, through a key callback.  This is the one that
        works when the process was started detached -- `nohup`, a background
        job, anything whose stdin is /dev/null and where a terminal Enter has
        nowhere to go.
      * on the TERMINAL, through a reader thread, for a shell that does have a
        keyboard on stdin.

    The window hook needs `mujoco.viewer._launch_internal`, because the public
    `launch()` does not forward `key_callback` and `launch_passive`, which
    does, hard-exits under mjpython on macOS.  If that private door ever
    closes this falls back to `launch()` and the terminal path still works.
    """
    import mujoco.viewer

    model, data = _load()
    ctrl = StandController()
    ctrl.reset(data)
    mujoco.set_mjcb_control(ctrl)

    def step_phase():
        if ctrl.finished:
            print("\n   sequence finished; close the window to quit.", flush=True)
            return
        ctrl.request_advance()
        time.sleep(0.05)                 # let the control callback consume it
        print("\n>> phase %d %s : %s"
              % (ctrl.phase, ctrl.phase_name,
                 StandController.BLURB[ctrl.phase_name]), flush=True)

    def on_key(keycode):
        if keycode in KEY_ENTER:
            step_phase()

    def on_stdin():
        try:
            while sys.stdin.readline() != "":
                step_phase()
        except Exception:                # stdin closed or is /dev/null
            return

    def report():
        while True:
            time.sleep(0.25)
            print("   " + ctrl.status(data), end="\r", flush=True)

    print("DOG6 stand -- ENTER steps the phase, IN THE WINDOW or on this terminal.")
    print("Close the window to quit.")
    print("   phase 0 settle : " + StandController.BLURB["settle"], flush=True)

    for target in (on_stdin, report):
        threading.Thread(target=target, daemon=True).start()

    try:
        mujoco.viewer._launch_internal(
            model, data, run_physics_thread=True, key_callback=on_key)
    except AttributeError:               # private API moved; terminal only
        print("   (no window key hook in this mujoco -- use ENTER on the terminal)")
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)    # it is a GLOBAL callback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DOG6 stand: position -> compliance -> park")
    ap.add_argument("--headless", action="store_true",
                    help="no window; auto-step the phases and print a table")
    ap.add_argument("--dwell", type=float, default=4.0,
                    help="seconds per phase in --headless (default 4)")
    args = ap.parse_args(argv)

    if args.headless:
        print("DOG6 stand, headless.  crouch %.3f m -> lift %.4f m, no IMU.\n"
              % (CROUCH_HEIGHT, LIFT_HEIGHT))
        _, _, rows = run_headless(args.dwell)
        print("\n%-8s %10s %10s %10s %10s" % ("phase", "h_cmd", "h_fk", "h_true", "fk err"))
        for name, h_cmd, h_fk, h_true in rows:
            print("%-8s %10.4f %10.4f %10.4f %+9.2f mm"
                  % (name, h_cmd, h_fk, h_true, 1000.0 * (h_fk - h_true)))
        return 0

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
