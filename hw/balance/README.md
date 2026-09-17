# `hw.balance` — the stand's balance controller

```
python -m hw.balance.selftest       188 checks, no robot, no IMU, no simulator
python -m hw.balance.config          every number, with its provenance
python -m hw.stand --law srb         run it
python -m hw.stand --law per-leg     run what it replaces
python -m hw.fold_trot               the fold stance, then T trots in place
python -m hw.trot                    the nominal crouch, then T trots in place
```

Implements the design note in [`doc/dog6_stand_control.tex`](../../doc): an
SRB wrench law with a weighted-least-squares force allocator, replacing the
per-leg Cartesian compliance law in the **lift phase only**.

`sequence.py` runs the whole six-phase stand — `limp → settle → crouch → lift
→ park → done` — but the balance law drives one phase of it. The other five
are the drivers' own 0xA4 position loops and no gain in this package reaches
them. The sequence lives here rather than in `hw.stand` because every seam
between a position phase and the lift is a statement about the law: the lift
arms from the pose the **crouch** left, `BalanceLaw.arm` latches `h0` and the
heading from what is *measured* at that instant, and the **park** starts from
wherever the lift settled. `hw.stand` is the runner — CAN slots, keys, log,
command line — and the dividing line is I/O.

## What the old law could not see

| | per-leg compliance | this |
|---|---|---|
| trunk height | regulated | regulated |
| trunk roll / pitch | **emergent** — no sensor reads it, no variable names it | measured, SO(3) log-map error, its own gains |
| load sharing | fixed `mg/4` per foot | allocated by the actual geometry |
| internal force (6 DOF) | picked by accident | picked deliberately, min-weighted-norm |
| a foot pulling **up** on the floor | nothing forbids it | `fz_min > 0` forbids it |
| gravity | `mg/4` per foot, four times | once, as `m*g` in the wrench |

## The five stages, and the two frames

```
1 state       R, ω^b from the IMU; q, q̇ from the encoders      body / joint
2   …         ω^w = R ω^b,  r_i^w = R (x_i^b − c^b),  h        BODY → WORLD
3 controller  PD on (p_c,z, R) → accelerations → b_d            world
4 allocation  A f^w = b_d by weighted LS, then the cone         world
5 torque      τ_i = −J_i^T R^T f_i^w + τ_grav,i                WORLD → JOINT
```

Everything between 2 and 5 is world frame; everything outside is body or
joint. **Every array that has a frame carries the frame in its name** — `x_b`
and `r_w` are never spelled the same way.

| file | what it owns |
|---|---|
| `sequence.py` | the six phases and what each sends. The only file here that is not part of the law |
| `config.py` | every number. Gains, cone, weights, the pinned CoM/inertia, the trips |
| `state.py` | stages 1–2. Measurement only — no gain appears in it |
| `reference.py` | the C² quintic S-curve in `h`, and the CoM conversion |
| `controller.py` | stage 3. **The controller**: the only place a gain touches the trunk |
| `allocation.py` | stage 4. Pure transmission — no trunk feedback at all |
| `torque.py` | stage 5, plus the tilt-aware leg-gravity term |
| `law.py` | the five chained; one call per sweep. `sequence.py` owns *when*, this owns *what* |
| `selftest.py` | gates the lot offline |

## Three things that are exactly zero error at the identity

Each would pass every level bench test and then grow with tilt. All three are
checked **at a tilt**, never at `R = I`:

- the `R^T` in stage 5 — dropping it costs 0.35 N·m at 10°, 0 at 0°
- the world/body attitude-error pair — `e^w = R e^b`, identical at `R_d = I`
- the `ω × R x^b` term in `ż` — and at a *level square stance* it cancels by
  symmetry even when ω ≠ 0, which is why it is so easy to leave out

## Heights are floor to trunk **bottom**

The number a ruler reads. `sim.stand`'s `CROUCH_HEIGHT` / `LIFT_HEIGHT` /
`height_from_fk` are all trunk-**origin** heights, 35.01 mm higher, and
`state.origin_to_height` is the only conversion. DOG5 printed 191 mm where a
ruler read ~160 for want of exactly this distinction.

```
crouch  h =   0.0 mm     lift  h = 115.0 mm     CAD stand  h = 157.5 mm
```

## The CoM is pinned, and that is a decision

`config.COM_BODY` is evaluated once at the **lift pose** and used everywhere.
The true value swings 37 mm over the ramp (the legs are 60 % of the mass and
unfold downward), so this is an approximation — but it is the *consistent*
one: the same constant appears in the measured `p_c,z` and in the commanded
`p_c,z,d`, so it cancels out of the error entirely and the PD's z channel sees
precisely the trunk-height error. The chain-rule coefficient a pose-dependent
CoM would force into the velocity reference (0.677) becomes exactly 1.

What it costs is the moment arms, taken about a point up to 37 mm off early in
the ramp — a bias in the moment row that looks like a small constant pitch
offset and shrinks to zero at the top. **If a log shows a residual tilt that
shrinks as the robot rises, this is the first thing to unpin.**

## Timing, measured

Whole law, all four leg-gravity terms refreshed every sweep:

| | |
|---|---|
| `state.read` (stages 1–2) | 82 µs |
| reference | 4 µs |
| controller | 11 µs |
| allocation | 37 µs |
| leg gravity ×4, **closed form** | 36 µs |
| leg gravity ×4, chain walk | *465 µs* |
| `stance_torque` | 8 µs |
| IK for the tracking trip | 67 µs |
| **total** | **210 µs** of a 333 µs slot |

The closed-form leg gravity is what makes it fit. It agrees with
`sim.kinematics.leg_frames`' walk to 2.2e-16 over 300 random poses **and**
random orientations, which is the only licence for a second copy.

## The crouch is a parameter now: `posture.py`

`posture.NOMINAL` is `sim.stand.Q_CROUCH` — shins vertical, trunk on the floor
at `h = 0`, the pose the lift was *solved for*. `posture.FOLD` is a crouch
somebody folded the robot into by hand, and it agrees with the nominal one
about nothing:

| | nominal | fold |
|---|---|---|
| `h` at the crouch | 0 mm | **60 mm** — starts in the air |
| feet, trunk x | ±237 mm | **+215 front / −143 rear** |
| reach used | 0.358 all round | 0.428 front / 0.350 rear |
| support centroid | trunk origin | **36 mm ahead** of it |

A pose owes the sequence exactly three things: `q` (what CROUCH and PARK drive
to), `foot_xy` (where `ik_reference` **pins the feet** while the trunk rises,
in the hip frame), and `z_origin`. `BalanceLaw.arm` already latched `h0` from
the *measurement*, so a crouch that does not start at zero needed nothing
there — that was written to keep a ramp from stepping into `k_d,z`, and it
pays for itself here.

**`foot_xy` had to stop being a module constant.** It was
`sim.stand.FOOT_XY`, which from the fold crouch is a 59° joint error on the
first torque sweep against a 25° trip — the run would end before the ramp
started, reporting a tracking failure that is really just two postures being
different.

**A hand-captured pose is not a pose until it is regularised.** The raw
capture carried two defects belonging to the operator, not the posture: the
left and right feet sat 2.4 mm apart, and the four feet spanned 21.5 mm of
trunk-frame z — which on a flat floor is not four foot heights, it is the
**trunk pitched 3.43°**. `posture.regularise` averages x and |y| within each
axle, mirrors them, and gives all four one height. Front and rear are *not*
averaged together: the 45 mm the folded rear legs sit forward is the posture.
It is done in **foot space** and the IK solved afterwards, which is what makes
the joint symmetry exact — averaging joint angles instead leaves the feet off,
because the map between them is not linear.

It moved each foot 10–12 mm and each joint 4.8–7.5°, and left `h` where it
was. The payoff: with the feet at one height the level-trunk reference **is**
the pose, so tracking error starts at exactly zero instead of 6.33° of a 25°
budget spent before the robot moves. `FOLD_CAPTURED_Q` keeps the raw numbers
and the regularisation re-runs at import, so the two cannot drift apart.

### The SRB model is pinned per POSTURE

`c^b` and `I^b` are `[DERIVED, PINNED]` — one CoM, one inertia, constant for
the run. That is what "single rigid body" means here and it is not changing.
What was wrong is the **pose they were derived at**: `config`'s pair is solved
at `NOMINAL_POSE`, and holding a folded crouch is a different stance.

| | nominal | fold |
|---|---|---|
| `c^b` | (0.0, 0.0, −14.8) mm | **(−13.8, 0.0, −27.1) mm** |
| `I^b` diag | 0.0281 / 0.2238 / 0.2406 | 0.0346 / **0.1587** / 0.1705 |

(That fold column is the original, rear-tucked fold. Since 2026-09-17 the rear
legs fold like the front ones and `FOLD` derives c^b (0.0, 0.0, −29.9) mm,
I^b 0.0371 / 0.1399 / 0.1493 — no x offset left.)

The 13.8 mm of `c^b` x was not cosmetic: every moment arm is
`r_w = R(x_b − c^b)`, so on a 57.7 N robot it is **0.80 N·m of phantom pitch
moment**. With no integrator the attitude loop parks against it, and the first
fold run held **−2.1° of nose-up with a steady 0.67 N·m** of commanded pitch
moment — which is that bias, measured.

`CrouchPose.srb` derives the pair at the posture's own stance at `H_LIFT`, and
`config.SrbModel` carries them **as one object** because three places consume
them — `state.read`, `reference.com_command` and `controller.balance_wrench` —
and `com_command`'s docstring is explicit that the CoM offset only cancels out
of the height error *because both sides convert with the same constant*.
Mixing two models puts 12.3 mm of phantom height error straight into the z
channel; `selftest` §11 checks exactly that.

`NOMINAL` pins `config.SRB` itself rather than re-deriving it, so the path
`hw.stand` has always flown is bit-identical — `config.NOMINAL_POSE` is solved
with `sim.stand`'s damped least squares and `CrouchPose` uses the closed-form
inverse, and the two differ by about a nanometre.

### Roll has its own gain in the fold stance

With the SRB model corrected, the second fold run's pitch came back and roll
still did not. The PD produces an **acceleration**, so equal `kp_att` gives
roll and pitch equal bandwidth — but the restoring moment is `I · kp · e`, and
this robot is narrow: in the fold stance `I_xx` is 4.6× smaller than `I_yy`.
At kp 90 a 0.14 N·m bias parks roll at 2.6° and pitch at 0.6°.

`--kp-roll` / `--kd-roll` override roll only, in the same units as
`--kp-att`. `hw.fold_stand` flies **kp 290, kd 23**; `hw.stand` is unchanged.
kp 290 puts roll at the moment stiffness DOG5 held on both axes (10 N·m/rad).
kd 23 is chosen for latency: in a sampled model at 250 Hz it stays stable to
40 ms of attitude age, the most of any damping at that kp (today's roll gains
tolerate 68 ms). `IMU_MAX_AGE_S` freezes the attitude half at 50 ms, so the
hold line prints the IMU age. No integrator — DOG5 never had one either.

### Does it lift?

Offline, over a perfectly tracked rise (`selftest` §11):

| | nominal | fold |
|---|---|---|
| trips | none | **none** |
| peak torque | 0.79 N·m | **2.02 N·m** (RR.knee, at h ≈ 76 mm) |
| static hold | 0.54–0.66 N·m | 1.58–1.78 N·m |
| allocator residual | 0 | 0 |
| load, front/rear | 50 / 50 | **36 / 64** |

The rear bias is the support polygon: the CoM is pinned at the trunk origin
and the centroid is 36 mm ahead of it. **`--tau-cap 1.0` cannot lift the fold
crouch — it cannot even hold it statically.** `--tau-cap 3.0`
(`TAU_STAGED_MAX`) is required, and leaves about 1.6× margin.

`hw.fold_stand` is the entry point: `hw.stand`'s sequence, runner and law
unchanged, SRB only, IMU datum **fixed** rather than latched — a run meant to
be compared against a nominal one cannot have each side holding to its own
private "level".

## RISE and HOLD: where the controller is actually watched

```
limp -> settle -> crouch -> rise -> hold -> park -> done
                            ^^^^^^^^^^^^ the only torque phases
```

Physically the two are one phase — same law, same gains, same reference (the
quintic clamps past `T`, so it holds itself). **The split is so they can be
told apart afterwards.** The rise answers *can it get up*; the hold answers
*does the controller correct*, and that second question is only asked by
pushing the robot.

- **The edge is a clock, not a key.** `update` steps rise → hold the sweep the
  S-curve arrives — DOG5's `now - t0 >= T_RISE`. `advance` **refuses** to leave
  the rise by hand: a phase whose exit is a physical fact should not also be
  exitable early by a keystroke, or a half-finished lift becomes a hold that
  thinks it is at height.
- **A push is sliceable out of the log**: `d["phase"] == "hold"`.
- **`HoldWatch`** carries the three numbers that answer the question — worst
  tilt **from the setpoint**, the attitude moment the law asked for, and how
  long recovery took. Live on the status line, and in the exit report.

Measured from the setpoint and not from true level, because the setpoint is
what the law is holding the robot at; on a sloped floor a peak measured from
true level reads the slope as a disturbance that was never rejected.

**One trap lives at that edge.** The per-leg law's height ramp is a smoothstep
over time since the *rise*, and `t_phase` resets at the boundary. Authored
against `t_phase` it would restart at `alpha = 0`, command `CROUCH_HEIGHT` at
full height, and drop the robot. It is authored against `t_lift`, and
`selftest` section 10 is the regression for it. The SRB law is immune —
`arm` latches its own `t0`.

**DOG5 eventually flew two gain tables across this edge** (`_RISE` for
WAIT+RISE, `_TROT` for HOLD+TROT, smoothstepped over `GAIN_BLEND_S = 0.5`),
because it measured that `kd_att = 1.0` killed five rises and then trotted
fine: "a gain RISE cannot carry is not thereby a gain TROT cannot carry, and
one shared table cannot say so." DOG6 has **one** table. Nothing here has been
measured yet, so there is nothing to split it on.

## Trot in place: `gait.py`, `swing.py`, `hw.trot`, `hw.fold_trot`

Two entry points, one trot (`hw.trot.trot_options`). `hw.trot` is `hw.stand`
at its own defaults from the **nominal** crouch — setpoint latched, roll
90/17 — with the tilt stop at 45°, the torque-phase tracking trip off (it
feeds no torque), and a faster 0.8 s cycle
(`--period`, `--duty`, `--settle`, `--settle-every` on both entry points).
`hw.fold_trot` is `hw.fold_stand`'s, at 1.2 s. The fold
trot tipped in roll on the robot on 2026-09-17; the nominal stance puts both
diagonal support lines exactly through the CoM (fold: 17.2 mm behind), and
offline its trot peaks at 1.14 N·m against the fold's 3.30.

```
hold --T--> trot --T (latched)--> hold at the next four-foot window
```

The same law, the same gains, the same wrench. A trot changes three things in
`BalanceLaw.update`, all driven by one `gait.sample(now)` per sweep:

| | stand | trot |
|---|---|---|
| allocator | four feet, weight 1 | the clock's contact weights; a swing foot is out of the solve, `fz ≤ w·fz_max`, total Fz rescaled to the command |
| a swing leg | — | `swing.swing_torque`: Cartesian PD on cMPC's quintic arc, on top of its own gravity |
| height | mean of four feet | mean of the **stance** feet (`state.on_stance`) |
| residual trip | every sweep | four-foot sweeps only |

**Where each piece came from.** The clock is DOG5's `trot_demo`: 1.2 s, duty
0.80, contact ramp 0.15, a 0.2 s four-foot settle every 2 cycles, lead
alternating (`config`, `[DOG5 FLOWN]`). The phase arithmetic is `sim.cmpc.gait`'s
(offset in seconds, before the modulo). The arc is `sim.cmpc.swing.SwingTrajectory`,
imported. The gains are DOG5's 140/140/180 N/m, 8/8/15 N s/m.

**Operator decisions, 2026-09-16.** No foot placement: the foot rises 40 mm
in the trunk frame and lands where it left. Torque cap 9 N·m (`TAU_HARD_NM`);
`SafetyGate(ceiling=...)` is the only way past 3.0, and only `hw.fold_trot`
passes it. Tilt stop 45°, roll gains and tracking-off as `hw.fold_stand`.
Slew 60 N·m/s, DOG5's.

**Two things this port had to do that neither source did:**

- *Height over the stance feet.* Averaged over all four, a foot at its apex
  reads the trunk 10 mm low on DOG6 and the height loop answered with ~19 N it
  should not (76.5 N total on a 57.7 N robot, offline).
- *The weight bounds the normal force.* A diagonal pair cannot make the
  moment the wrench asks for, and least squares loads the unreachable part
  onto the foot ramping out — 3.7 N at weight 0.002, gone in one sweep. With
  `fz ≤ w·fz_max` the worst per-sweep step is 0.28 N·m (71 N·m/s).

And one DOG5 did differently: `reset` starts the clock mid the all-four
window, every weight already 1, so pressing T moves no load.

**The fold stance and a diagonal pair.** The CoM sits **17.2 mm behind both
diagonal support lines**, so every swing is ~0.97 N·m of pitch moment no pair
of feet can make, in the same direction for both diagonals. Nothing moves the
feet or the trunk to fix it; the four-foot windows and the settle re-level, as
in DOG5's demo. A nose-up that grows cycle by cycle is this.

Offline, over two settle blocks with the swing tracked perfectly
(`selftest` §12): no trip, peak 3.30 N·m (rear knee), Fz exactly the weight.
The law costs p50 ~610 µs on the Pi against ~490 µs for the stand.

## What "level" means: the setpoint is latched, not assumed

DOG5's `SETPOINT_DYNAMIC`, ported whole — the IMU is the same board in the
same orientation on both robots, so the convention ports with it. The rule is
**world := body here**, on all three axes:

| axis | latched | where | re-latched? |
|---|---|---|---|
| roll, pitch | `law.latch_setpoint` | **`limp`** — the only phase at zero torque | no, once per run |
| yaw | `law.arm` | the lift handover, the first sweep torque is live | yes, on every re-arm |

The asymmetry is the point. Heading has no truth to return to, so yaw re-locks
whenever torque re-arms. **Level does**, so re-latching roll/pitch mid-run
would redefine level as whatever tilt the robot was limping at — and drag
`TILT_STOP_DEG`'s reference along with it.

What it absorbs, per run, with nothing to measure or transcribe: the IMU mount
tilt, the floor's slope, and the resting pose's lean, all three at once. What
it costs: start the robot on a slope and it will hold that slope. For a stand
on one patch of floor that is the right trade; the day the robot has to stay
upright *across* a slope it is the wrong one.

Two details that are not cosmetic:

- **The setpoint is built into `R_des`, not subtracted from roll/pitch.** DOG5
  subtracted, which is exact for its RPY wrench and would *not* be exact here:
  this law forms `e_R` from the SO(3) log map of `R_dᵀ R`, and two Euler
  triples subtracted componentwise are not the rotation between them.
  `controller.latched_attitude` builds it in, so the error stays one log map
  with nothing subtracted anywhere. At the latch attitude `e_R` is **exactly**
  zero, so arming can never step the wrench.
- **The tilt trip measures from the setpoint** (`law.tilt_from_setpoint_deg`),
  because the trip has to mean "the robot has left the attitude the law is
  holding it at". `state.tilt_deg` still measures from true level and is what
  the log and the status line report; on a sloped floor the two differ by the
  slope.

`SETPOINT_DYNAMIC = False` falls back to the config statics and reproduces
exactly what DOG5 flew before 2026-08-28.

## What is still open

- **The setpoint statics are DOG5's, unverified on DOG6.**
  `SETPOINT_ROLL_DEG` / `SETPOINT_PITCH_DEG` are that rig's measured pair.
  They are only the *fallback* and the sanity reference the latch warns
  against, but the moment DOG6's board is bolted on they are the number to
  re-measure.
- **Every gain is `[UNTUNED]`.** The attitude pair is what the whole change
  exists to make tunable, so sweep it first, in sim.
- **Nothing measures whether a foot stayed put.** The cone bounds what the
  allocator *asks* the floor for. The rear-foot slide that ended the
  2026-09-15 runs is still invisible to a law reading joint encoders only.
- **The handover is still one sweep.** The torque cap ramps over 1.0 s, which
  bounds how hard the new law can push; it does not blend the two laws.
