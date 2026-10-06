"""The convex MPC on the robot, entered from -- and parked by -- the stand.

    python -m hw.cmpc.selftest                     gate it, no robot, no IMU
    python -m hw.cmpc.run --fake --auto 1 --no-imu   the whole path, no robot
    python -m hw.cmpc.run --tau-cap 2.5 --log run.npz

`sim.cmpc` is the controller: the linearised body model, the condensed QP,
the force-to-torque map, 83 gates in simulation.  This package is what it
takes to run THAT OBJECT on twelve MG5010 drivers, and it is deliberately
thin -- every number the controller reads is still `sim.cmpc.config`'s, and
every line of the QP is still `sim.cmpc.qp`'s.

THE STAND'S JOINT HOLD IS KEPT, AND THE MPC IS A PHASE INSIDE IT
    `hw.stand` is six phases: limp, settle, crouch, lift, park, done.  Five of
    them are the drivers' own 0xA4 position loops holding the joints -- the
    thing that lets a robot be put down, picked up, stepped and parked without
    a torque law in the loop.  The MPC does not replace that sequence.  It is
    inserted as ONE phase, between the lift and the park:

        limp  settle  crouch  lift  [mpc]  park  done
         0xA1  0xA4    0xA4   0xA1   0xA1   0xA4  0xA4

    so the robot stands up under `hw.balance`'s closed-form SRB law, hands
    over to the MPC at the lift height, and when ENTER is pressed -- or the
    MPC trips -- parks through the same position-mode ramp the stand always
    had.  `run.HardwareCmpc` is a subclass of `hw.stand.HardwareStand` that
    adds exactly that phase; the five position phases, their tracking trips
    and their speed caps are inherited, not copied.  The gate is the same
    `SafetyGate` object across the handover, so its slew limiter blends the
    two laws' torques rather than stepping between them.

    Inside the MPC phase the controller's own joint-space floor on the stance
    legs (`sim.cmpc.config.KP_JOINT` / `KD_JOINT`, pure damping) is kept as
    it is in simulation.

WHAT IS DIFFERENT ON THE ROBOT, AND WHERE EACH DIFFERENCE LIVES
    the gait          `--gait stand` by default: four feet down, always.
                      `hw.safety.TAU_STAGED_MAX` is 3.0 N*m, which stands
                      (2.20 measured) and does not carry a trot diagonal
                      (4.46).  A standing MPC is the same QP with a constant
                      constraint structure, and it is the A/B against the
                      SRB law that lifted the robot.         [sim.cmpc.gait]
    the state         there is no ground truth.  `state.KinematicOdometry`
                      builds the controller's `BodyState` from the IMU's roll
                      and pitch, a gyro-integrated yaw, and the stance feet's
                      kinematics for height and velocity.  It is the simplest
                      estimator that closes the loop and it is labelled as
                      such.                                   [hw.cmpc.state]
    the kinematics    `hw.kinematics`' closed forms, handed to the controller
                      through `BodyState.feet_body` / `jacobians_body`.  The
                      chain walk `sim.kinematics` does costs 810 us a sweep,
                      2.4 CAN slots.                 [sim.cmpc.controller]
    the IMU hold      real, so not emulated: `Controller(sensor_split=False)`
                      and `imu_fresh` per sweep.     [sim.cmpc.controller]
    leg gravity       `hw.balance.torque`'s closed-form leg-weight term is
                      ADDED to the stance torque, as the SRB and per-leg laws
                      do.  The simulator's controller has no such term and
                      absorbs the 0.19 N*m into its height error; on the
                      robot it is what the other two laws carry, so the A/B
                      is between wrench laws and not between gravity models.
                      `--no-leg-gravity` is the simulator's law exactly.
                                                              [hw.cmpc.run]
    the torque        shaped by `hw.safety.SafetyGate` AFTER the controller's
                      own 8 N*m clamp: cap, limit block, slew.  [hw.cmpc.run]
    the solve         runs at slot 0 and delays every motor behind it.  It
                      is timed every solve; a sustained overrun of
                      `config.SOLVE_BUDGET_S` is a trip, because the CAN gap
                      stop line is 25 ms and a QP that takes 20 is already
                      inside it.                              [hw.cmpc.run]

WHAT A GREEN `hw.cmpc.selftest` DOES NOT MEAN
    That the robot will stand under the MPC.  It says the controller on the
    hardware path computes what the simulator's computes given the same
    state, that the estimator recovers the states it is built to recover, that
    the phase machine keeps the stand's joint hold around the MPC, and that
    every trip fires on the state that should fire it.  Nothing in it has
    seen a motor.
"""
from __future__ import annotations

import importlib

__all__ = ["config", "state", "run", "selftest"]


def __getattr__(name: str):
    """Resolve submodules on first use, as `sim.cmpc` does."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
