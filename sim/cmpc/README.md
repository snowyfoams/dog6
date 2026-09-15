# `sim.cmpc` — convex MPC

A reproduction of Di Carlo, Wensing, Katz, Bledt and Kim, *Dynamic Locomotion
in the MIT Cheetah 3 Through Convex Model-Predictive Control* (IROS 2018),
against DOG6's model.

```
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.selftest          # 83 gates
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.run               # viewer; Xbox pad or keys
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.teleop            # the reference alone, animated
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.gamepad           # is the pad awake?
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.run --headless --script box
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.run --headless --vx 0.2 --yaw-rate 30
```

**Xbox pad**, if one is connected: left stick is vx/vy, right stick is yaw
rate, centred is stop, left trigger is a precision scale, A stops, B resets.
The mapping is **absolute** — the stick commands a velocity, where the keys
accumulate one — and a pad that goes quiet zeroes the command rather than
leaving the robot walking. It is XInput through `ctypes`, so there is nothing
to install; a pad asleep at startup is picked up within a second of waking.

**Keys**, always live when no pad is: W/S drive x-velocity, A/D drive
y-velocity, Q/E drive yaw rate — 0.1 m/s and 20 °/s per press. SPACE stops,
R resets.

`teleop` runs the reference generator *without the robot* and animates what it
produces — the xy path with the heading along it, the yaw, and the contact
schedule c over a window that extends past "now" into the MPC's horizon. It is
the way to answer "is the reference wrong, or is the tracking wrong" without
arguing about it. With no pad, the stick circles are draggable with the
mouse.

## What it does

14 s runs, ground-truth state, timetable gait, MPC at 40 Hz:

| case | commanded | achieved | height err | roll rms | pitch rms | τ clipped |
|---|---|---|---|---|---|---|
| stand | — | — | 2.3 mm | 0.18° | 0.08° | 0 % |
| forward | 0.20 m/s | 0.20 | 6.6 mm | 1.36° | 3.17° | 0.1 % |
| forward | 0.30 m/s | 0.30 | 10.9 mm | 1.70° | 4.47° | 2.1 % |
| forward | 0.40 m/s | 0.39 | 20.7 mm | 5.34° | 3.98° | 16.2 % |
| forward | 0.50 m/s | — | — | — | — | **falls at 4.0 s** |
| strafe | 0.20 m/s | 0.20 | 3.5 mm | 1.56° | 0.72° | 0 % |
| turn | 90 °/s | 89.2 | 11.4 mm | 0.42° | 0.31° | 0 % |
| forward + turn | 0.2 m/s, 40 °/s | 0.20, 39.7 | 8.5 mm | 3.13° | 3.18° | 1.1 % |

Velocity tracking is 97–100 %. The residual shortfall is structural, not a gain
that wants raising — the reference has no integrator, so a steady velocity
error is what the cost trades against everything else in `Q_DIAG`.

**The envelope is now set by torque.** 0.4 m/s works with the 8.0 N·m clamp
active on 16 % of sweeps; 0.5 m/s falls with it active on 23 %. That is the
known weak seam in this method and it is worth being precise about: the QP's
input is a *force at a contact point*, and it has no model of the legs, so it
cannot know that 120 N horizontally at DOG6's 0.18 m hip-to-foot lever asks for
21 N·m from a motor that saturates at 9.94. The limit is applied downstream in
`controller.update`, after the solve. A force the QP planned and the joints
then clip is a force the plan assumed and did not get.

### What 20 Hz cost, and how it hid

The MPC ran at 20 Hz until a rate sweep was done properly. At 20 Hz:

| | 20 Hz | 40 Hz |
|---|---|---|
| forward 0.30 m/s | 0.28 achieved, 22.0 mm, roll 8.28° | 0.30, 10.9 mm, roll 1.70° |
| forward 0.40 m/s | **fell at 3.8 s** | 0.39, roll 5.34° |
| turn 90 °/s | 85.6 | 89.2 |

The failure did not look like a rate problem. It looked like a gain problem:
roll would sit around 4–8° for ten seconds and then grow and tip over. What it
actually was is in `config.MPC_HZ` — a period-2 subharmonic riding the
inverted-pendulum mode the body has while standing on one diagonal, which the
MPC was sampling three times per time constant. Raising the rate moved the
roll's dominant frequency from **1.00 Hz — the unstable mode — to 2.00 Hz, the
stride**, its rms from 7.43° to 1.49°, and its stride-to-stride sign
alternations from 13 of 13 to 0 of 13. That is the signature of an instability
becoming an ordinary forced response.

**No leg-loop rate fixes it.** Sweeping the control loop 100 → 1000 Hz leaves
roll at 14–17° and every run still falls; sweeping the IMU 50 → 500 Hz does
nothing either. The stance force is held between solves, so a faster leg loop
re-maps the same force through a fresher Jacobian and never produces a new one.

**The roll weight is a second, independent fix** that was not taken.
`Q_DIAG`'s roll/pitch entry is 0.25 — the lowest non-zero weight in the vector,
against 10 for yaw and 50 for z — and raising it to 5.0 stabilises the same
case at 20 Hz (roll 14.64° → 1.41°). It is left at the paper's value because
this is a reproduction; the rate was the thing changed.

## The pieces

```
config.py       every constant, tagged [PAPER] / [DOG6] / [TUNED]
gait.py         the contact schedule -- a pure function of the clock
trajectory.py   operator command -> reference states
dynamics.py     the linearised body model, and its ZOH discretisation
qp.py           condensation, constraint assembly, the OSQP solve
swing.py        foot placement (33) and the swing law (1)-(3)
controller.py   the three rates: 40 Hz MPC, 200 Hz IMU, 250 Hz torques
gamepad.py      an Xbox pad as the operator, through XInput and ctypes
run.py          MuJoCo, driven by the pad, the keys or a script
teleop.py       the pad driving the reference alone, animated
selftest.py     83 gates
```

## The model

Thirteen states, twelve inputs:

```
x = [Theta(3)  p(3)  omega(3)  v(3)  g]        u = four world-frame foot forces
```

The thirteenth state is gravity. `m p̈ = Σf + mg` is *affine*, and no choice of
A and B produces a constant from x and u; carrying g as a state whose
derivative is zero makes `ẋ = Ax + Bu` exact, with `A[vz, g] = 1` doing the
work. The alternative is threading an affine offset through every step of the
condensation.

Three approximations turn the rigid-body dynamics into that linear form. The
self-test **measures** each rather than citing the paper for it, at the
attitude a trot actually holds (±1°, ω ≤ 0.5 rad/s) and with forces of the
shape the QP really returns:

| approximation | error at ±1° | at ±6° |
|---|---|---|
| `Θ̇ = Rz(ψ)ᵀω` instead of `T(Θ)⁻¹ω` | 1.4 % | 8 % |
| drop `ω × (Iω)`, carry I by yaw only | 1.3 % | 7 % |

What grows with tilt is the **inertia carry**, not the gyroscopic term: DOG6's
Ixx is 0.0362 and its Iyy 0.193, a factor of 5, so tilting mixes a large moment
into a small one and `I⁻¹` moves a long way. All three approximations buy the
same thing — A and B depend on the state only through yaw, so the dynamics are
linear time-varying with a schedule known at solve time. Give up any one and
the problem is a nonlinear program.

### The discretisation has a closed form

`A` here has exactly three non-zero blocks and none feeds back, so `A² ` has one
non-zero entry and **`A³ = 0` exactly**. The series for the ZOH block matrix
therefore terminates:

```
Ad = I + A dt + A² dt²/2
Bd = B dt + A B dt²/2 + A² B dt³/6
```

Not a truncation — the same number `expm` computes, reached by noticing the
series is finite, and ~40× faster for the ten of them each solve needs.
`discretize_expm` is kept and gated against (agreement 5.6e-17).

Exactness is not pedantry at `dt = 0.025 s`: Euler drops the `A B dt²/2` term
that couples force to *position*, leaving the model believing a force changes
velocity within a step but not position. Over ten steps that compounds into a
plan that under-predicts how far the body travels.

## The QP

Condensed, not sparse. Substituting the dynamics gives `X = A_qp x₀ + B_qp U`,
so the decision variables are the forces alone: 120 of them at a horizon of 10,
with a dense 120×120 Hessian. The dynamics are then satisfied *by construction*
rather than to solver tolerance, which matters at a 40 Hz outer loop where
there is no chance to correct a violation before the next solve.

Constraint (21), `D_i u_i = 0`, is enforced through the box rather than as an
equality block: a swing foot keeps its five friction rows and gets `fz ∈ [0,0]`,
which with the pyramid rows forces `|fx| ≤ 0` and `|fy| ≤ 0`. Identical
feasible set, constant sparsity pattern, no rank-changing equality block. The
measured residual is 4e-8 N.

```
variables       120   (10 steps x 12 forces)
constraint rows 200   (4 feet x 5 rows x 10 steps)
solve           ~4.3 ms median, 17% of the 25 ms budget at 40 Hz
```

`R = 1e-6 I` is not there to discourage force — the robot needs 57.7 N to
stand. It is there to make the Hessian positive *definite*: with four feet down
there are 12 force variables against 6 wrench equations, so a 6-dimensional
subspace of internal forces costs nothing and the QP would be free to return
any member of it. It also leaves H conditioned around 3e5, which is why the
self-test gates the objective's *value* tightly and its *argmin* loosely.

## Control rates

Three rates, set by three **sources** — not by three design choices. Measured
by counting calls over a 6 s run, not read off the config:

| layer | rate | source |
|---|---|---|
| `mj_step` physics | 500 Hz | model timestep 0.002 s |
| MPC solve → f (world) | **40.000 Hz** | the QP |
| `fb = Rᵀ f` | **200.000 Hz** | the **IMU** — R exists only at its rate |
| `τ = Jᵀ fb` | **250 Hz** | the **encoders** — q arrives with the CAN sweep |
| control sweep | 250 Hz | `params.CONTROL_HZ` |

R and q come from different devices on different clocks, so `Rᵀf` and `Jᵀfb`
**cannot share a rate**. Between IMU samples there is no new orientation: the
body-frame force is held, and every torque written in between goes out through
an attitude up to **8 ms old**. `controller._sensed` splits the simulator's
perfect state along that sensor boundary and hands the controller only what the
hardware would have — IMU fields held at 200 Hz, encoder fields fresh at 250 Hz.

Simulating this matters because recomputing `fb` every sweep would invent IMU
samples the robot never sends, and would do it in exactly the axis —
orientation — that decides whether a stance force pushes the robot up or
sideways. It also means a newly solved force waits up to one IMU period before
it reaches a joint. That latency is real and it is now in the loop.

**What it costs, measured:** almost nothing at these rates. Against an
idealised 250 Hz R, standing and turning are identical and forward tracking is
unchanged; only roll at 0.3 m/s degrades, 5.54° → 6.16° rms.

### Neither 40 Hz nor 200 Hz divides 250 Hz

6.25 and 1.25 sweeps respectively. Both deadlines are therefore counted from a
**fixed origin** (`controller.Ticker`), so the intervals alternate about an
exact mean — 24/28 ms for the MPC, 4/8 ms for the IMU.

Re-anchoring on the moment a tick actually fired — `next = t + period`, which
reads as obviously equivalent — rounds the period *up* to a whole sweep every
time, and the rounding compounds instead of cancelling. On the MPC that gave a
rock-steady **52 ms period: a 19.23 Hz loop calling itself 20 Hz**, with no
jitter to hint at it. Not just a mislabelled rate — `u₀` was then held 4 %
longer than the discretisation that chose it assumes. Both schedules are gated
now, including that R is genuinely observed stale on 151 of 751 sweeps.

## Timing

Python and NumPy, on this machine. Medians over five trials:

```
MPC solve    4.25 ms    17% of the 25 ms available at 40 Hz
leg sweep    2.23 ms    56% of the 4.0 ms available at 250 Hz
```

Both fit. **Report the median, not the mean** — the first solve pays OSQP's
symbolic setup, and a desktop OS adds scheduling spikes an order of magnitude
above the body of the distribution. As a mean the leg loop reads 6.3 ms and
156 % of budget; its median is 2.23 ms and does not vary (2.23, 2.23, 2.24,
2.23, 2.23 across trials). The mean was measuring Windows, not this code.

Two optimisations landed along the way, each worth about 2×: `mass_matrix`
assembles from per-link Jacobians in **one** chain walk instead of three RNEA
passes, and the joint-space floor came off the swing legs, which removed a
per-sweep IK call and also improved tracking — a swing leg already has a
complete operational-space law, and joint damping on top of it opposes a `qd`
of several rad/s that the feedforward knows nothing about. What remains is
dominated by `operational_inertia` and the four separate `leg_frames` walks
each swing leg makes per sweep.

## Deliberate simplifications

Both were requested, and both are what make a failure here attributable:

1. **The 12-state body pose is fully observed** — read from the simulator. No
   IMU, no leg odometry, no filter. The model carries an IMU site and three
   sensors and this controller ignores all of them. Everything below is
   therefore an *upper bound* on what the same controller does with a real
   estimator in front of it.
2. **The gait is a pure timetable.** No contact sensing, no early-touchdown
   detection, no schedule adaptation. A foot is in stance because the clock
   says so, and the MPC plans forces for it on that basis.

## Three bugs worth recording

Each is now a gate, because each was invisible in the logs.

**The CoM is not the trunk origin.** `Z_REF` is a trunk-origin height — what
`params.STAND_HEIGHT` measures and what a ruler reads — but the MPC's state is
the CoM, 29 mm lower. Handing the trunk height straight to the body model asked
the CoM to stand where the trunk should be: **18 mm of steady height error and
32 % more vertical force than the robot's own weight** (76 N against 57.7),
with nothing anywhere that looked like a bug. Fixing it took the height error
to 3.4 mm and the force to 60.5 N.

**The yaw branch cut.** The reference integrates yaw without bound; the
measurement comes from `arctan2` and lives in (−π, π]. The moment the robot
passed half a turn, the same heading had two descriptions 2π apart, the yaw row
of Q — weighted 10, the largest in the vector — saw a 6.28 rad error, and the
MPC threw everything it had at correcting a heading that was already right. It
fell at 7.9 s at 30 °/s, 6.2 s at 40, 5.3 s at 60, 3.9 s at 90 — which is 180°
of yaw in every case. Before the fix it read convincingly as a turning-rate
limit.

**Four feet down in a trot.** `mod(t/T + offset, 1)` forms a sum near 1.0 where
doubles are spaced 2⁻⁵³ and the operands 2⁻⁵⁴, so a phase of 0.49999999999999994
plus an offset of 0.5 *rounds up* to exactly 1.0 and leaves the modulo as 0.0. A
foot just before liftoff and its diagonal partner just before touchdown then
both read "in stance" and `contact()` returned all four feet down. It showed on
2 of 997 samples. Applying the offset in seconds, before the modulo, keeps both
operands at the period's scale where the modulo is exact; 0 wrong in 232,001
samples across four grids.

## One negative result

Equation (33) is `p_des = p_ref + v_CoM Δt/2`, and under a yaw rate the hip
does not travel with the CoM — it sits 0.17 m off the turn axis and moves at
`ω × r_hip`, which at 90 °/s displaces the target by 33 mm. That correction is
implemented (`swing.hip_velocity`, `cfg.YAW_PLACEMENT_CORRECTION`) and it is
**off by default, because it was measured and changed nothing**: 95–96 % yaw
tracking either way, under 0.3° rms roll either way, at 20/40/60/90 °/s.

That is the paper's own argument showing up in data rather than in its
abstract. A 33 mm error in a heuristic foot placement is something an optimiser
already planning a horizon of wrenches, with yaw-moment authority to spare,
simply absorbs. A reactive controller would not have that slack. The flag was
added while chasing the turn divergence above — which turned out to be the yaw
branch cut, not the placement law at all.

## What is still a heuristic

The QP is convex *given* the contact schedule and the foot positions over the
horizon. Both come from outside it: the schedule from `gait`, the positions
from predicting where the feet will be. Choosing **where to put a foot** —
equation (33) — is not part of the optimisation at all. The convexity is real,
but it is convexity of a subproblem, and the placement heuristic feeding it is
the part of this method that is still a heuristic. When this controller falls
over, look there first.
