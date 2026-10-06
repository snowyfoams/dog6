# `hw.cmpc` — the convex MPC on the robot

```
python -m hw.cmpc.selftest                        73 checks, no robot, no IMU, no MuJoCo
python -m hw.cmpc.run --fake --auto 1 --no-imu --law per-leg   the whole path on hw.fake_bus
python -m hw.cmpc.run --tau-cap 2.5 --log run.npz              the first run: stand gait
python -m hw.cmpc.run --tau-cap 2.5 --mpc-height 140           ...and rise to 140 mm
python -m hw.cmpc.config                          every number this path adds
```

`sim.cmpc` is the controller. This is what it takes to run **that object** on
the drivers, and it is thin on purpose: every constant the MPC reads is still
`sim.cmpc.config`'s, every line of the QP is still `sim.cmpc.qp`'s, and the
self-test's central gate is that the hardware path's torques equal the
simulator's to 1e-12 N·m from the same state.

## The stand's joint hold is kept; the MPC is one phase inside it

```
 limp    settle   crouch   lift    [mpc]    park    done
 0xA1    0xA4     0xA4     0xA1    0xA1     0xA4    0xA4
 iq=0    hold     ramp     SRB     MPC      ramp    hold
```

Five of `hw.stand`'s six phases are the drivers' own 0xA4 position loops
holding the joints — what lets the robot be put down, picked up, stepped and
parked with no torque law in the loop. The MPC does not replace that sequence.
`run.HardwareCmpc` subclasses `hw.stand.HardwareStand` and inserts **one**
torque phase between the lift and the park; limp, settle, crouch, park and
done — their joint targets, their speed caps, their tracking trips — are
inherited, and the self-test compares them sweep for sweep against a plain
`HardwareStand` (2327 position sweeps, worst difference exactly 0).

Both handovers are latches, not steps:

| | |
|---|---|
| lift → mpc | the height command is latched at the trunk height the estimator measures; the reference's xy and yaw anchor at zero. The `SafetyGate` is the **same object** the lift used — its slew limiter (5 N·m/s) is what blends the SRB law's last torque into the MPC's first |
| mpc → park | `HardwareStand.advance` latches the measured pose and ramps to `Q_CROUCH` in position mode, exactly as after the lift. ENTER does it; so does any trip |

Inside the MPC phase the controller's own joint-space floor on the stance
legs (`KP_JOINT` / `KD_JOINT`, pure damping about the measured angle) is kept
as in simulation.

## What is different on the robot

| | sim | robot | where |
|---|---|---|---|
| gait | trot timetable | **stand**: four feet down at every step of the horizon. `TAU_STAGED_MAX` = 3.0 N·m stands (2.20) and cannot carry a trot diagonal (4.46), so `--gait trot` is refused without `--trot-anyway "why"` | `sim.cmpc.gait.STAND` |
| state | ground truth from MuJoCo | `state.KinematicOdometry`: IMU roll/pitch, **gyro-integrated yaw** (the magnetometer is logged, never used), height and velocity from the stance feet, xy integrated | `hw/cmpc/state.py` |
| kinematics | `sim.kinematics` chain walk, 810 µs/sweep | `hw.kinematics` closed form, handed in through `BodyState.feet_body` / `jacobians_body` | `sim.cmpc.controller` |
| IMU hold | emulated at 200 Hz | real: `Controller(sensor_split=False)`, `imu_fresh` per sweep | `sim.cmpc.controller` |
| leg gravity | none | `hw.balance.torque`'s closed form added, as the SRB and per-leg laws do. `--no-leg-gravity` is the simulator's law exactly | `run.HardwareCmpc._mpc` |
| torque limit | 8 N·m clamp | that clamp, then `SafetyGate`: cap, limit block, slew | `run.HardwareCmpc._mpc` |
| OSQP polishing | on | off: OSQP 1.x prints a line from C on every solve with no active set, `verbose` or not, and a standing QP never has one. `--polish` turns it back on | `Controller(solver_settings=)` |

None of the simulator's own behaviour changes: `Controller()` with no
arguments is the trot, the emulated hold and the chain walk, as before.

## The estimator, and what it assumes

| output | provenance |
|---|---|
| R, roll, pitch | **measured** (DETA10 through `hw.imu`) |
| yaw | **integrated** from the gyro; world x is the heading at the handover |
| ω (world) | **measured**, R ω^b |
| z | **derived**: `FOOT_RADIUS − mean_stance (R x_i^b)_z` — a planted foot is a height sensor |
| v | **derived**: `−mean_stance (R J_i q̇_i + ω × R x_i^b)`, low-passed at 20 Hz |
| x, y | **integrated** from v. Leg odometry; drifts with every slip |

Every derived row is wrong the moment a foot the schedule calls planted is
not. That is `hw.balance.state`'s assumption too, the one the 2026-09-15 runs
showed the robot can violate, and nothing here can detect it. The
accelerometer is not fused; a filter that uses it is its own module, when the
kinematic estimate has been seen to be the limit.

## The solve is inside the CAN schedule, so it is a trip

The whole controller runs at slot 0 of the 4 ms sweep and delays every motor
behind it. The 250 Hz half is cheap on this path (the kinematics arrive
precomputed; measured no-solve sweep **0.48 ms p50** here against the SRB
law's 0.39 on the same machine). The QP is not: every `mpc_dt` it runs in the
same slot, and `hw.stand.run` stops the run when any motor goes 25 ms without
a frame. So a solve past `SOLVE_BUDGET_S` = 12 ms for three solves in a row
is a trip here, before a driver's input-lost latch reports it.

```
desktop (sim README)   4.25 ms median
this container         5.7 ms p50, 8.4 p95, 27 max     4 of 80 over budget
the Pi                 NOT MEASURED -- run --fake first; the exit report prints it
```

If it does not fit, `--mpc-hz 20` halves the load. A **trot** must not go
below 40 Hz (`sim.cmpc.config.MPC_HZ` has the measured subharmonic); a
four-foot **stand** has no diagonal pendulum mode to excite and may.

## Trips the MPC phase adds to the stand's

| | |
|---|---|
| tilt | `|roll|` or `|pitch|` past 12° (`hw.balance.config`'s) |
| height | 40 mm from the commanded height for 25 sweeps |
| QP | status not `solved` / `solved inaccurate` three solves running |
| solve budget | over 12 ms three solves running |
| non-finite | any NaN torque out of the controller |
| IMU stale | **held**, not tripped, as in `hw.balance`; counted in the report |

The stand's own trips — joint limits, overspeed, over-temperature, CAN miss,
driver faults, the torque readback — run in `hw.stand.run` around all of this,
unchanged.

## Operator

ENTER steps the phase; from `mpc` it parks. X is an e-stop everywhere and from
`mpc` it **drops the robot** from the lift height; run the first MPC phases
supported, as the first lifts were. `+` / `-` nudge the height 5 mm through a
C2 ramp (`hw.balance.reference.Quintic`). W/S/A/D/Q/E and SPACE drive the
velocity command exactly as in `sim.cmpc.run`, in the trot gait only, under
`V_MAX_HW` = 0.2 m/s.

## The log

`CmpcLog` is `hw.stand.StandLog`'s columns — so an MPC run and an SRB run
read with one script — plus `f_mpc`, `fz_mpc`, `contacts`, `solve_ms`, `qp_ok`,
`clipped`, `est_p`, `est_v`, `est_yaw`, `yaw_mag`, `z_cmd`, `tau_grav`,
`tau_mpc`. NaN in every non-MPC sweep.

## What is still open

- **Nothing here has seen a motor.** The 73 checks say the hardware path
  computes what the simulator's computes, not that the robot will stand.
- **The Pi's solve time.** The budget is derived from the CAN stop line, not
  measured against the machine that has to meet it.
- **The estimator is kinematic.** No accelerometer, no slip detection.
- **The trot** needs a torque ceiling above 4.46 N·m and a swing law whose
  250 Hz half is 2.1 ms per leg on `sim.leg_dynamics`. Neither is this
  branch's; both are known.
- **`R_BODY_IMU` is still the identity placeholder**, as everywhere else.
