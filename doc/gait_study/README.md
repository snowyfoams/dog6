# Gait-parameter study: what tips the WASD walk, and whether it accumulates

*Experimental design, 2026-10-21, branch `cloud`.  No code has been written
for it and no run has been made; this document is the protocol the runs
will follow and the analysis they will be read by.  Everything it cites is
in the tree: the law (`hw/balance`), the two harnesses
(`doc/crouch_trot/hwsim.py`, `doc/walk/walksim.py`, `doc/walk/longrun.py`),
the robot logs' columns (`hw/stand.py`, `StandLog.FIELDS`) and the
touchdown-detection study that was reverted at `1c0a10a` (its design and
MuJoCo numbers are at `6f51e56`, `doc/contact/README.md`).*

**TL;DR.**  The robot trots in place indefinitely and tips sideways after
some seconds to minutes of WASD, worst under A/D.  Two explanations were
proposed: (H1) an error that *accumulates* through the walk until the PD can
no longer recover it, and (H2) the *time-driven* gait clock loading feet
that have not landed and unloading feet that have not lifted, so the force
allocation is wrong at every mistimed touchdown.  This document turns both
into measurable claims, adds a third that the same runs will test — (H3) a
lateral command crosses the trot's support diagonal 3.4× faster than a
forward one, and the foothold's velocity term is a quarter of the
capture-point gain, so A/D has far less margin than W/S for reasons that
are geometric, not temporal — and lays out five experiments (E0–E4) plus a
control set (E5) in which the gait parameters (duty, period, contact ramp,
apex, torque slew, swing damping, settle, the foothold gain) are varied
one factor at a time around the flown baseline, in MuJoCo first and then
on the robot.  Every metric is computable from the log `hw.stand --log`
already writes.  Section 7 is the decision table: which pattern of results
supports which hypothesis, and what each would mean for the controller
that comes next.

Two things found while writing it, both to settle before the first run:

- **The recorded "best flight" command line is not reproducible as
  written.**  `hw/fold2_walk.py`'s docstring gives `--duty 0.8
  --contact-ramp 0.2`; `gait.TrotGait` refuses a ramp above
  `(duty − 0.5)/(2 duty)` = 0.1875 at duty 0.80 (the freeze-point guard,
  in the tree since `a779397`, 2026-09-16; `hw.stand` turns the refusal
  into a parser error).  Whatever flew on 2026-10-07 had a ramp of 0.1875
  or less.  The baseline of this study (section 3.1) must therefore be
  read off a run's banner or log, not off the docstring.
- **"PD error accumulation" has to be located in a state before it can be
  tested.**  The SRB PD has no integrator and no memory; its error is
  whatever the sensors read this sweep.  What *can* accumulate is listed in
  section 1.2, and H1 is stated against those states.
- **The baseline's silent defaults.**  The line flown on 2026-10-21
  (`--duty 0.8 --contact-ramp 0.15 --tau-slew 300 --kp-joint 4
  --kp-joint-abd 5`, T = 0.6 s) runs with the governor **on** (the law
  pauses the walk at 5° of lean) and, because the joint spring is on, with
  `--com-auto` **on** (the CoM read in every HOLD and applied at T).
  "Fairly stable" was observed with the governor pressing SPACE; section
  3.1 writes B0 out in full.

## 1. The observations, and three mechanisms that would produce them

### 1.1 What the operator sees

| | observation | where it is already in the record |
|---|---|---|
| O1 | the trot in place does not fail | 413 s in place on 2026-10-06 (`fold2_walk_2.npz`, doc/walk §11.9), tilt 3.5–7.7° per 15 s window, no fall |
| O2 | under WASD the robot tips after "a while"; A/D tips in roll sooner and harder than W/S | the 2026-10-06 fall: 4 s into A at 0.03 m/s, roll +0.9 → +7.6° in 80 ms across one landing, the fall hold at 43° 0.8 s later |
| O3 | as the walk goes on the legs stop moving smoothly | not yet quantified; the swing lands 15 mm (median) / 27 mm (p90) off its arc on the robot (governor.py) |
| O4 | a leg the clock has in *swing* stays on the floor and scuffs; the stance leg "cannot lift" | in MuJoCo every swing of the robot-like plant reaches 7–9 mm of a 20 mm arc (doc/contact §3); on the robot the apex is reported good in place and not measured walking |

And one observation that constrains the hypotheses more than any of the
four: **after every SPACE in the 2026-10-06 flight the trot in place had the
trunk back under 3° within 1–2 s** (doc/walk §11.9).  Whatever makes the
walk fail is undone in two cycles of trotting in place.

### 1.2 What can and cannot accumulate in this architecture

The controller is memoryless: `controller.balance_wrench` is a pure
function of the sweep's state, the swing impedance has no integral term,
the QP has no warm start (`QP_BETA = 0`), the joint layer's targets are
recomputed from the reference each sweep.  A "PD error" cannot grow on its
own.  The states that *do* carry the past from one step to the next, and so
are the only places H1 can live:

| state | how it moves | what resets it |
|---|---|---|
| **the stance geometry** — where the four feet are under the trunk | each landing is placed by eq (33) + Raibert about the *estimated* trunk, re-aimed every sweep from a `v_hat` that is rms 0.05–0.08 m/s standing (config `WALK_V_FILTER_HZ`); on a rolled trunk the trunk-frame arc lands the low-side foot early and inboard (doc/walk §11.3: the fold's stance narrowed 124 → 110 mm in 30 s with the trot's 10 N/m swing) | the at-rest swing: with `v_ref = r = 0` every swing flies from the foot back to its own site (footstep.py, `_plan_at_rest`), so a SPACE restores the stance within one cycle |
| **the estimator's x/y** (leg odometry) | under-reads distance ~10 %, slips invisible to it, `p_est` wanders ~10 % of distance walked (doc/walk §6.7); the x/y rows pull the CoM toward `xy_des` with 9 /s² up to 1 m/s² | each HOLD and each T re-latch `xy_des` (law `est_xy`); the leash bounds the reference within 30 mm of the filter; the walk's anchors are relative to the same filter, so the drift mostly cancels out of the step |
| **the world anchors of the planted feet** (`walk.anchor_w`) | latched at each touchdown from `p_est + R x_b`; a foot that slides after latching pulls the joint layer against a stale anchor, clamped to 20 mm (`HOLD_XY_ERR_MAX`), Kp 0 by default (`WALK_KP_JOINT`) so only the damper acts | the next touchdown of that foot |
| **the attitude setpoint** | on the fold stands it is **fixed**, not latched (`fold2_trot.stand_options`: `dynamic_setpoint=False`, so `R_des` carries the config statics `SETPOINT_ROLL_DEG` / `SETPOINT_PITCH_DEG`, DOG5's measured pair, "unverified on DOG6"); the IMU mount, the floor's slope and that pair's error are one constant attitude error the integrator-less loop parks against at `Kp⁻¹ M` — the ~1° of roll and the 26–28 N against 20–23 N of the 2026-10-06 in-place trot look like it; walking onto a differently sloped patch adds to it | never, within a run |
| **the CoM pin** | `--com-auto` moves it at T from the HOLD's median; stiction in the HOLD biases the reading by millimetres (doc/walk §11.7) | the next HOLD + T |
| **the drivers' torque per amp** | falls with winding temperature; a 10 % loss costs under a degree in place (doc/walk §11.3, `kt 0.9`) but nothing has measured it over a 10-minute walk | cooling |
| **foot slip on the floor** | the allocator asks inside μ = 0.5; nothing measures whether a foot stayed put (hw/balance/README, "what is still open") | the next swing |

The first row is the strongest candidate: it is the one accumulation
MuJoCo has already shown, it is walking-only (the at-rest arc undoes it),
and it is undone by a SPACE — which is exactly O1–O2 and the recovery
observation.  H1 is therefore stated as **"a stance-geometry drift that
grows with walking and is reset by trotting in place"**, with the other
rows as alternatives to be excluded by the controls of E5.

### 1.3 H2 — the clock loads feet that are not down

`gait.TrotGait` is a timetable: a foot is in stance because the clock says
so, early and late touchdown are not detected ("a pure timetable, as both of
them were").  The allocator receives the clock's weights
(`law.update`, stage 4), ramped in over `ramp · T_st` after the clock's
touchdown.  What follows when the foot is still airborne at that instant:

1. the foot is handed a share of the weight it cannot react; the leg
   extends into the air and the delivered wrench is short by that share;
2. at the same time the *other* diagonal — the only real support — is being
   ramped *out* by the same clock; between the two the trunk is unsupported
   on the side of the late foot;
3. the trunk rolls toward that side; a rolled trunk puts the next
   low-side swing arc below the floor (early landing, scuff — O4) and the
   next high-side arc above it (late landing — step 1 again), **so the
   mistiming is self-reinforcing through the attitude** (doc/walk §11.9's
   "loop that closes through the landings").

Quantitatively: the weight the clock asks of a foot that lands τ after
the clock's touchdown is `smoothstep(τ / (ramp · T_st))`, so for a 40 ms
lag at T = 0.6 s:

| duty | swing | four-foot window | ramp (fraction of stance) | ramp (ms) | weight asked at τ = 40 ms |
|---|---|---|---|---|---|
| 0.70 | 180 ms | 120 ms | 0.129 (the walk's `0.9 (d−0.5)/(2d)`) | 54 | **0.83** |
| 0.75 | 150 ms | 150 ms | 0.150 | 68 | 0.64 |
| 0.80 | 120 ms | 180 ms | 0.150 (the trot's `CONTACT_RAMP`) | 72 | 0.58 |
| 0.80 | 120 ms | 180 ms | 0.1875 (the ceiling) | 90 | 0.41 |

(`gait.smoothstep` is DOG5's 3u²−2u³.)  The lag itself is set by the swing's
torque-rate bound (doc/walk §4: the foot is 5.8 kg in z through the
reflected rotor; at 120 ms of swing the 20 mm arc's feedforward alone asks
~1 000 N·m/s of a knee the gate slews at 120, ~300 at 180 ms), so the
parameters that move the lag are the swing duration, the apex, the gate's
slew and the swing law's damping, and the parameters that move the load
asked at a given lag are the ramp and the duty.  **H2 predicts that falls
track the product — the "phantom impulse" of section 2 — whichever knob
sets it.**  In MuJoCo the lag was 39–62 ms (doc/contact §5, `late`); on the
robot it has never been measured, and measuring it is E0.

### 1.4 H3 — a lateral command crosses the support diagonal

During a swing the trunk stands on one diagonal pair.  Forces at two points
on a line make no moment about that line, so about its own support line
the diagonal has **no authority at all** and the trunk is an inverted
pendulum with τ = √(h/g) = √(0.160/9.81) = **0.128 s** (doc/walk §4).  A
disturbance across the line grows by e^{T_sw/τ}: ×2.6 over a 120 ms swing,
×4.1 over 180 ms.  What the command does across that line:

FOLD2's feet sit at (±219.5, ±65) mm (hw/fold2_trot.py), so the FL–RR
diagonal's unit direction is (0.959, 0.284) and its normal (−0.284, 0.959).
A trunk velocity `v` has the component across the diagonal

| command | v across the diagonal | over a 180 ms swing |
|---|---|---|
| W/S at 0.06 m/s | 0.284 × 0.06 = 0.017 m/s | 3.1 mm |
| W/S at 0.12 m/s | 0.034 m/s | 6.1 mm |
| A/D at 0.03 m/s | 0.959 × 0.03 = **0.029 m/s** | 5.2 mm |
| A/D at 0.06 m/s | **0.058 m/s** | 10.4 mm |

A lateral command moves the CoM across the only line the trunk cannot push
about **3.4× faster** than a forward one of the same speed.  What is
supposed to catch it is the foothold: eq (33)'s `v̂ T_st/2` sustains the
velocity and Raibert's `k_v (v̂ − v_ref)` corrects it, with `STEP_KV` =
0.03 s against a capture-point gain of √(h/g) ≈ 0.13–0.14 s
(config.py: "the capture point would be sqrt(h/g) = 0.14 s") and the step
bounded to ±40 mm of the site in y (`STEP_MAX_XY`).  H3 says: A/D fails
because the lateral foothold under-corrects a crossing that forward walking
barely makes, and the failure rate is a function of the crossing speed
`v_⊥`, not of the direction key.  It predicts, in particular, that
**W at 0.10 m/s and A at 0.03 m/s (v_⊥ 0.028 and 0.029 m/s) should fail
alike**, and that raising `k_v` toward 0.13 s helps A/D and does nothing
for the timing errors H2 is about.

### 1.5 How the three are told apart

They are not exclusive — the most likely picture is H2 as the trigger, H3
as why A/D is where it bites, and H1 (stance drift) as a slow modulator
that lowers the margin with each step walked.  What distinguishes them is
*what the failure depends on*:

| | H1 (stance drift) | H2 (clock–contact mismatch) | H3 (lateral geometry) |
|---|---|---|---|
| depends on | distance walked since the last in-place trot | the touchdown lag and the load ramp | the velocity across the diagonal |
| hazard of a fall over time at constant command | **rising** (Weibull shape > 1) | constant per step (memoryless, shape ≈ 1) | constant per step |
| a SPACE pause of a few cycles | resets it | changes nothing about the next landing | changes nothing |
| the gait parameter that helps most | none; the foothold's noise and the joint layer | ramp ↑, slew ↑, swing Kd_z ↑, apex ↓ | duty ↑ (shorter two-foot phase), `k_v` ↑ |
| the pre-fall signature in the log | stance width / site offsets trending for seconds | an asymmetric landing (low side early, high side late) 1–3 cycles before | the first landing after the command, at the phase where the CoM is farthest across |
| direction dependence | weak | via the roll coupling of §1.3 step 3 only | strong, and collapses onto `v_⊥` |

## 2. The metrics, all from the existing log

### 2.0 The tracked set, in priority

Twelve quantities, three tiers.  Tier 1 is read on every run before
anything else; tier 2 is the mechanism, one number per landing or per
swing; tier 3 is the slow part, one number per cycle or per window.
Every one is computed offline from `hw.stand --log`'s columns by the
analysis script of §9; nothing on the robot changes.

| # | tier | quantity | one line | unit | per | tests |
|---|---|---|---|---|---|---|
| 1 | 1 | **attitude** | roll and pitch from the setpoint; tilt = max(\|roll\|, \|pitch\|); its rms per 10 s, its peak, the governor's 1.5 Hz lean replayed | deg | sweep → window | all |
| 2 | 1 | **outcome** | fell (tilt from the setpoint > 15° before the fall hold), the stopwatch time and the side; the number of landings before it | — | run | all |
| 3 | 1 | **real swing height** | `z_clear`: each foot's height above the plane of the planted feet, IMU-levelled; per swing its apex against the 20 mm asked, and whether it ever cleared 4 mm (a **scuffed** swing) | mm | sweep → swing | O3, O4, H2 |
| 4 | 1 | **swing tracking error** | `p_swing − x_b` per axis; its rms over the swing and its lateral value at the end of the arc (the fall's 24–27 mm) | mm | swing | O3, H2/H3 |
| 5 | 2 | **touchdown lag `τ_td`** | the floor's touchdown minus the clock's, per landing; + late, − early | ms | landing | **H2** |
| 6 | 2 | **phantom impulse `Φ`** | the normal force the allocator asked of a foot while it was still above 4 mm, integrated | N·s | landing | **H2** |
| 7 | 2 | **roll step `Δφ`** | the roll change across the 80 ms after each landing, signed toward the landing foot's side | deg | landing | H2, H3 |
| 8 | 2 | **liftoff lag `τ_lo`** | the clock's liftoff to the foot clearing 4 mm; ∞ if it never does | ms | swing | O4, H2 |
| 9 | 3 | **stance geometry** | each foot's mid-stance x/y against its site, and the stance width left–right, per cycle | mm | cycle | **H1** |
| 10 | 3 | **attitude bias** | the cycle mean of `e_R` and of the commanded moment: what "PD error" would be if it grew | deg, N·m | cycle | H1 |
| 11 | 3 | **estimator** | `xy_des − (p_est + R c)`, the fraction of sweeps at the 1 m/s² clamp, `p_est`'s drift per leg of the shuttle | mm | cycle | H1 |
| 12 | 3 | **torque** | the swing joints' request the slew did not pass; sweeps at the 9 N·m clip; `Σ\|tau_meas\| / Σ\|tau_cmd\|` on the planted legs as the thermal proxy | N·m·s, count, — | cycle | H2, E5 |

Three definitions have to be fixed before the first log is read, and are
fixed here: the **attitude reference** is the setpoint the law holds
(FOLD2 flies the config statics, §1.2), with roll and pitch from true level
kept beside it; the **floor plane** for `z_clear` is the two lowest planted
feet's (the reverted detector's rule, so a planted foot lifted by a roll
cannot raise it); and the **touchdown instant** without a sensor is
§2.1's two-witness rule — the foot within 4 mm of the plane and stopped
while the arc still descends, or the swing impedance commanding 8 N or
more of push — calibrated against MuJoCo's contact truth (§6.3) before any
robot number is quoted.

`hw.stand --log` writes one row per sweep (`StandLog.FIELDS`): `t`,
`phase`, `roll/pitch/yaw`, `omega_b`, `b_d`, `e_R`, `fz`, `f_w`, `q`,
`q_ref`, `tau_cmd/tau_meas/tau_req`, `contact_w`, `swing_s`, `p_swing`,
`x_b`, `p_est/v_est/est_age`, `xy_des/xy_on`, `com_meas/com_pin`.  Three
known defects shape what can be computed:

- **`qd` is one row repeated** (commit `9a4e591`, "known, not fixed"):
  every velocity below is a differentiated `q` or `x_b`, smoothed the way
  the law smooths (`state.read`'s 40 Hz filter), never the column.
- **the loop runs ~5.5 ms a sweep in every trot**, not 4 (doc/walk §11.9);
  every duration is from the log's `t`, never from a row count.
- **the walk's reference is not logged** (no `v_ref`, no `leashed`, no
  governor state; the sim's trace has them as `w_ref_*`).  `v_ref` is
  recoverable by differentiating `xy_des` (the CoM setpoint, which is the
  reference); the key presses and the gait phase at each press are
  recoverable from where `xy_des` starts to move against `swing_s`.
  Section 9 lists this as the first logging addition.

### 2.1 Per landing (the clock's touchdown of leg *i*, cycle *n*)

| symbol | definition | columns | tests |
|---|---|---|---|
| `z_clear,i` | the foot's clearance above the plane of the two lowest *planted* feet, IMU-levelled: `z_origin − r_foot + (R x_b,i)_z` with `z_origin` from the planted feet (`hw.stand.SwingHeight`'s number, the detector's `z_clear`) | `x_b`, `roll`, `pitch`, `contact_w` | — |
| **`τ_td`** (touchdown lag) | `t_td,actual − t_td,clock`; + late, − early.  `t_td,clock` is where `contact_w` leaves 0; `t_td,actual` is the first sweep after `s ≥ 0.3` on which **either** `z_clear < 4 mm` and the foot has stopped descending (`ż_clear ≥ −0.03 m/s`) while the arc still descends, **or** the swing impedance's commanded push `f_push = −[Kp (p_swing − x_b) + Kd (ṗ_swing − ẋ_b)]_z ≥ 8 N` — the reverted detector's three-condition test (`6f51e56`, contact.py), run offline with hindsight.  Thresholds are `config.TD_*`'s of that commit; E0 and the sim calibrate them (§6.3) | `x_b`, `p_swing`, `swing_s`, `contact_w`, `roll`, `pitch` | H2 |
| **`Φ_i,n`** (phantom impulse) | `∫ fz_i dt` over the sweeps where `contact_w,i > 0` and `z_clear,i > 4 mm` — the normal force the allocator asked of a foot that was in the air, N·s | `fz`, `contact_w`, `x_b`, `roll`, `pitch` | H2 |
| `Δφ_n` (roll step) | `roll(t_td + 80 ms) − roll(t_td − 10 ms)` across the landing, signed toward the landing foot's side (the 2026-10-06 signature was +6.7° in 80 ms) | `roll`, `contact_w` | H2, H3 |
| `e_land,i` | the landing's trunk-frame x/y against the plan: `x_b,i` at `t_td,actual` minus `p_swing,i` on the arc's last sweep; its lateral part is the "24–27 mm to the right" of the fall | `x_b`, `p_swing` | H2, H3 |
| `s_land,i` | the foot's trunk-frame x/y at mid-stance against the HOLD's site (`sites_b`, from `q` in the HOLD): the stance-geometry error, per foot | `q`, `x_b`, `phase` | **H1** |
| `w_stance,n` | the stance width `y_FL − y_FR` and `y_RL − y_RR` at the four-foot window, mm | `x_b` | **H1** |

### 2.2 Per swing (the clock's liftoff of leg *i*)

| symbol | definition | tests |
|---|---|---|
| `τ_lo` (liftoff lag) | time from the clock's liftoff (`contact_w` reaches 0) to `z_clear > 4 mm`; ∞ if never | H2 (O4) |
| **scuffed swing** | a swing whose `z_clear` never exceeds 4 mm — "on the floor but swinging", O4 quantified | H2 |
| `H_reached` | max `z_clear` over the swing (the status line's "apex reached") | H2 |
| `e_swing` | `‖p_swing − x_b‖` rms over the swing, and its lateral part at `s = 0.9` | O3 |
| `∫ (|τ_req| − |τ_cmd|)⁺ dt` | the swing joints' request the gate's slew did not pass (the triangle under the peak of doc/walk fig. 1) | H2 |

### 2.3 Per cycle and per window (10 s, 30 s)

| symbol | definition | tests |
|---|---|---|
| `tilt_rms` | rms of max(|roll|, |pitch|) from the setpoint per window (`longrun.py`'s `tilt_rms_per_10s`) | all |
| `lean` | the governor's 1.5 Hz low-passed roll/pitch from the setpoint (replayed offline; the on-robot governor is off in the test runs, §5) | all |
| `ē_R,n`, `M̄_n` | the cycle mean of `e_R` and of the commanded moment `b_d[3:6]`: the attitude loop's **bias**, which is what "PD error" would mean if it grew | H1 |
| `xy_err_n` | `‖xy_des − (p_est + R c)‖` per cycle, and the fraction of sweeps at the 1 m/s² clamp | H1 (estimator row) |
| `kt_n` | `Σ|tau_meas| / Σ|tau_cmd|` over the planted legs per cycle: the current loop's follow, a thermal proxy until temperature is logged | E5 |
| `drift` | `p_est` at the end of a burst against its start, and against the distance the command integrates | H1 (estimator row) |

### 2.4 Per run

`fell` (tilt from the setpoint past 15°, `hwsim.trot_metrics`' rule, before
the fall hold at 45°), `t_fall` (s of command before it), `t_tilt15`, the
number of landings before the fall, peak torque, the fraction of swings
scuffed, and the per-landing distributions above as median / p90.

## 3. The factors

### 3.1 The baseline, B0

The configuration the operator flew on 2026-10-21 and judged "fairly
stable", by eye and without a log to say so:

```
hw.fold2_walk --duty 0.8 --contact-ramp 0.15 --tau-slew 300 --kp-joint 4 --kp-joint-abd 5
```

Everything that line does not name is a default, and two of those
defaults are not neutral.  Written out, B0 is:

| | value | where it comes from |
|---|---|---|
| clock | T 0.6 s, duty 0.80 (swing 120 ms, four-foot window 180 ms), ramp 0.15 (72 ms), **no settle** | the line; `fold2_trot.PERIOD_S`; `SETTLE_S` 0 since 2026-10-20 |
| gate | slew 300 N·m/s, clip 9 N·m | the line; `TAU_HARD_NM` |
| swing | the Cartesian impedance 150/150/400, 6/6/40, apex 20 mm, no feedforward | `WALK_SWING_LAW`, `KP/KD_SWING_WALK`, `fold2_trot.SWING_HEIGHT` |
| planted legs | the joint spring: Kp 4 on pitch and knee, 5 on abduction, Kd 0.2, targets world-anchored (walk.py) | the line |
| the CoM | **measured in every HOLD and applied at T** (`--com-auto` is on by default once `--kp-joint` > 0): the pin and the stance sites move by measured − pinned, the trunk slides onto them for 0.5 s | `hw/stand.py`'s `com_able`, `sequence.py` |
| the governor | **on**: the keys are paused at 5° of lean and resumed after 1 s under 3° | `WALK_PAUSE_LEAN_DEG` |
| keys | 0.06 m/s a press in x/y (the box 0.12 / 0.06), 6 °/s a press | `WALK_V_STEP`, 2026-10-20 |
| the rest | FOLD2 at 160 mm, QP, the x/y loop on the filter (9/6, 1 m/s²), the fixed attitude setpoint, the fall hold at 45° | `fold2_trot.stand_options` |

Two consequences for the study.  **The governor is part of what was
observed**: a walk that leans past 5° is paused by the law before the
operator sees it fall, so "fairly stable" is B0 *with* the governor
pressing SPACE, and its exit report (`governor N pauses, … s paused, lean
peak …`) is the one number from today's runs that says how often it did.
The hypothesis runs therefore fly `--pause-lean 0` and the governor's
contribution is measured as its own cell (§5.4).  **The CoM reading is a
covariate** (§3.3): the notice at T (read x/y, spread, applied or NOT
APPLIED) goes on every run sheet, and the log's `com_meas` / `com_pin`
columns keep it.

B1 is the tree's defaults since 2026-10-20 — duty 0.70, ramp 0.129, slew
120, Kp 0 (damping only, and so no `--com-auto`) — the configuration B0
was reached from by hand, and the one cell of E2 that moves four factors
at once.  The 2026-10-07 line (the same spring, settle 0.2, a ramp
≤ 0.1875) is B0 + settle, which E2's settle cell covers.

### 3.2 The gait factors and what each one moves

| factor | B0 | levels | moves | predicted sign under H1 / H2 / H3 |
|---|---|---|---|---|
| duty `d` | 0.80 | 0.70, 0.75, 0.80 | swing 180/150/120 ms (torque-rate lag ↑ with d), four-foot window 120/150/180 ms (re-level time ↑), two-foot phase ↓ with d | 0 / **worse** (lag ↑) / **better** (e^{T_sw/τ}: 4.1 → 2.6) |
| period `T` | 0.6 s | 0.5, 0.6, 0.8 s | at fixed d: swing and window scale together; step length per cycle ↑ with T at a given speed | worse (longer steps) / better (lag ↓) / worse (two-foot phase ↑) |
| contact ramp `r` | 0.15 (72 ms) | 0.075, 0.15, 0.1875 (the ceiling `(d−0.5)/(2d)` at duty 0.80) | the load asked of a late foot (§1.3 table); the unloading of a foot still carrying | 0 / **better** with r ↑ / 0 |
| apex `H` | 20 mm | 15, 20, 30 mm | clearance against the roll-induced early landing (65 mm × sin 5° = 5.7 mm; sin 10° = 11.3 mm) ↑; torque rate ∝ H ↑ (lag ↑) | 0 / U-shaped / 0 |
| gate slew `S` | 300 N·m/s | 120, 300 | the swing's lag (doc/walk §4: 120 passes triangles of a 12–15 N·m request) | 0 / **better** / 0 |
| swing damping `Kd_z` (`Kp_z`) | 40 (400) | 40, 60 (400, 600) | `Kd_z v_ref` is the velocity feedforward that carries the arc (swing.py, "what flew on the bench") | 0 / better / 0 |
| settle | 0 | 0, 0.2 s every 2 cycles | a four-foot re-level every 1.2 s; the reference keeps moving over planted feet (6–12 mm at 0.03–0.06 m/s) | better (a reset) / mixed / mixed |
| the joint spring (+ `--com-auto`) | Kp 4/5, the CoM measured | Kp 4/5 with the CoM; Kp 0 without it (the tree's default stance) | the stance's stiffness, and whether the CoM is corrected | worse without (a softer stance drifts more; doc/walk §11.7: the damping-only stance fell on the robot-like plant) / 0 / 0 |
| the governor | on at 5° | off, on | whether the law pauses the walk before it falls | — (a measurement of today's observation, not a hypothesis) |
| foothold gain `k_v` | 0.03 s | 0.03, 0.08, 0.13 | the lateral capture term | 0 / 0 / **better** for A/D |
| command | — | W/S 0.06, 0.12; A/D 0.03, 0.06; Q/E 6, 12 °/s; W 0.10 + A 0.03 (matched `v_⊥`) | `v_⊥` and the step length | — |
| burst structure | — | shuttle without SPACE; shuttle with 10 s of SPACE between legs (§5.5) | the distance walked, and whether the at-rest trot resets the state | **the H1 lever** |

Factors held fixed across the study and recorded in every run's name: the
posture (FOLD2), height 160 mm, the SRB gains (kp_att 90/17, roll 290/23),
the x/y loop (9/6, 1 m/s²), the leash 30 mm, `STEP_MAX_XY`,
`--v-filter-hz 1`, the allocator (QP), the fall hold 45°.

### 3.3 Confounds, and the control for each (E5)

| confound | why it matters | control |
|---|---|---|
| motor temperature | torque per amp falls as the windings warm; a long session is a warm robot; "after a while" could be thermal | log the drivers' temperature (0x9C/0xA1 reply, `MotorBus.temps()`, not yet in the log — §9); counterbalance run order; ≥ 3 min cool-down between cells; `kt_n` as the proxy meanwhile |
| battery voltage | the current loop's headroom | log the 0x9A voltage at the start and end of each run; a fresh pack per session or a supply |
| the CoM read in HOLD | a few mm moves the in-place tilt from 2° to 10° in MuJoCo (doc/walk §11.3) | `--com-auto` on in every run; the reading (`com_meas` median, spread, applied or not) is a covariate; a run whose HOLD read is NOT APPLIED is repeated |
| the gait phase at the key press | one run is one phase; in MuJoCo the outcome flipped with it (doc/walk §6.3) | recover the phase from the log (§2) and bin by it (E4); on the robot the press is random, which is a randomisation, not a bias |
| the operator | key timing, burst length, the SPACE | a fixed schedule read aloud from a stopwatch for now; the scripted schedule of §9 as soon as it exists |
| the floor patch | the latched setpoint holds the slope of the patch the robot limped on | the same start mark and heading every run; W/S and A/D bursts in both senses (S after W, D after A) so the robot stays on the patch |
| the IMU's age and the sweep time | an attitude 40 ms old is the roll gains' stability limit (hw/balance/README) | `imu_age` and the sweep interval from `t` are per-run covariates; a run with `imu_age` p95 > 30 ms or sweeps > 7 ms is excluded |
| the detector thresholds (offline) | an offline touchdown estimate that is wrong makes H2 untestable | calibrated against MuJoCo's `foot_fn` truth on the same plant family (§6.3) before any robot log is read |

## 4. The experiments

### E0 — Instrument the logs that exist (no robot time)

*Purpose.*  Define and check every metric of section 2 on the runs already
recorded before any new run is made, so the predicted signatures are
written down first: `fold2_walk_2.npz` (17 min, the fall at 629 s), the
2026-10-20 runs (impedance, no settle) and the 2026-10-21 trot that would
not run with the detector.

*What it produces.*  For the 2026-10-06 fall: `τ_td` for every landing of
the last 10 cycles, `Φ`, `Δφ`, `e_land` and `w_stance` against time, the
scuffed-swing fraction per 30 s of the whole walk, `ē_R` and `M̄` per cycle
over the 413 s in place and over each commanded burst, the hazard of the
five SPACE moments.  For the in-place trot: the same, as the null
distribution every walking metric is compared against.

*Reads.*  H2 predicts `τ_td` asymmetric by side in the three cycles before
the fall and `Φ` on the FR/RL pair; H1 predicts `w_stance` or `s_land`
trending over the 4 s of A, and the 413 s in place flat; H3 predicts the
first large `Δφ` at the landing where the CoM is farthest across the
diagonal.  If `τ_td`'s median in place is already past the ramp duration,
the robot's clock is loading every foot in the air and H2 is live before
any walking is commanded.

### E1 — Time structure: does anything accumulate, and does a pause reset it

*Design.*  B0, the lateral command at 0.06 m/s (one press), governor off.
A lab floor does not allow 180 s of one-way lateral walking (10 m), so the
command is a **shuttle**: 8 s of A, then D pressed twice in quick
succession (the keys sum through zero to −0.06 m/s; pressed within 0.1 s of
each other the reference never lands on exactly zero, so no swing is flown
"at rest"), 8 s of D, and so on, ±0.5 m about the start mark.  Three
schedules:

- **E1-C (continuous shuttle):** [A 8 s, D·D 8 s, A·A 8 s, …] until the
  fall or 180 s, N = 8.  The response is `t_fall` (censored at 180 s).
  Every reversal is a transient of its own (E4); they are the same in every
  schedule.
- **E1-B (shuttle with pauses):** [A 8 s, SPACE 10 s, D 8 s, SPACE 10 s, …]
  until the fall or 300 s, N = 8: the same legs, with ~16 cycles of the
  at-rest trot (the arcs back onto the sites) between them.  The responses
  are `t_fall` counted in *commanded* seconds, and every per-cycle metric
  at the *start* of leg k against the *end* of leg k−1.
- **E1-S (speed):** the continuous shuttle at 0.03 m/s (`--v-step 0.03`,
  16 s legs: the same distance per leg in twice the time), N = 6.
- **E1-P (in place):** the trot in place for 600 s, N = 2: the trend null.

*Reads.*  Fit a Weibull to `t_fall`: shape k > 1.5 is a rising hazard
(something accumulates), k ≈ 1 is memoryless (every step rolls the same
die).  If E1-B's bursts start each time where the previous ended (stance
width, `s_land`, `ē_R`), the pause does not reset the state and H1 is in a
row of §1.2 other than the stance; if they start fresh and `t_fall` in
commanded seconds equals E1-C's, nothing accumulates across a pause.  If
E1-S falls at the same *distance* as E1-C, the failure is set by what was
walked (H1); at the same *time*, by something that is not the walk
(thermal, E5); at a shorter distance and time, by the per-step hazard
rising with speed (H2/H3).  E1-P's trends calibrate the thermal and
estimator rows over 10 minutes with nothing walked.

### E2 — The clock–contact mismatch: one factor at a time around B0

*Design.*  Each factor of §3.2 at its levels, every other factor at B0;
two commands per cell — the lateral shuttle and the forward shuttle (W 8 s,
S·S 8 s, …), both at 0.06 m/s — each a fixed 60 s (or the fall), governor
off.  N = 6 per cell per command; the key cells (ramp, slew, duty) N = 8.
Order: the slew first (it is the one knob the operator has already found
decisive and it moves only the lag), then the ramp (it moves only the load
at a given lag), then the duty (it moves both, in opposite directions, and
is the cell where H2 and H3 disagree), then Kd_z, apex, period, settle,
the spring.

| cell | levels | runs (lateral + forward) |
|---|---|---|
| slew | 120, **300** | 2 × 8 × 2 = 32 |
| ramp | 0.075, **0.15**, 0.1875 | 3 × 8 × 2 = 48 (B0 shared) |
| duty | 0.70 (ramp 0.129), 0.75 (0.15), **0.80** | 48 (B0 shared) |
| Kd_z / Kp_z | **40/400**, 60/400, 60/600 | 3 × 6 × 2 = 36 |
| apex | 15, **20**, 30 | 36 |
| period | 0.5, **0.6**, 0.8 (d = 0.80) | 36 |
| settle | **0**, 0.2 | 24 |
| spring + CoM | **Kp 4/5 with `--com-auto`**, Kp 0 without | 24 |
| B1 | the 2026-10-20 defaults (duty 0.70, ramp 0.129, slew 120, Kp 0) | 16 |

About 200 robot runs at 2–3 min each (B0 is shared across the cells);
§5.3 says how to cut it to the first hundred without losing the decision.

*Responses.*  Per landing `τ_td`, `Φ`, `Δφ`; per run `fell`, `t_fall`, the
scuffed fraction, `H_reached`.

*Reads.*  Under H2 the roll step per landing and the fall rate are a
function of `Φ` across *all* cells (Spearman ρ between cell-median `Φ` and
fall rate, and within-run between `Φ_n` and `Δφ_n`): the slew and the Kd_z
cells lower `Φ` by lowering the lag, the ramp cells by lowering the load,
and all three lower the falls in proportion.  The duty cell is the
discriminator: if 0.80 wins *despite* a larger `τ_td` and `Φ`, the
two-foot phase length matters more than the mistiming (H3); if 0.70 wins
with the smaller `Φ`, H2.  The apex cell tests §1.3 step 3 directly: 30 mm
should cut the early landings on the low side and add lag on the high
side.

### E3 — Direction and speed: is it the crossing velocity

*Design.*  B0, the gait fixed, 30 s bursts, N = 6 each: W 0.06, W 0.12,
S 0.06, A 0.03, A 0.06, D 0.03, Q 6 °/s, E 12 °/s, W 0.06 + A 0.03,
and the matched pair **W 0.10 vs A 0.03** (`v_⊥` 0.028 vs 0.029 m/s).
Then the lateral cell with `k_v` ∈ {0.03, 0.08, 0.13} at A 0.03 and A 0.06,
N = 6 each, and one cell with `STEP_MAX_XY`'s y at 60 mm.

*Reads.*  Plot fall rate and `Δφ` p90 against `v_⊥` with the command as
the label.  Under H3 the points collapse onto one curve and the matched
pair agrees; `k_v` 0.13 moves the A/D points down and the W/S points not
at all.  Under H2 the A/D points sit above the curve at the same `v_⊥`
*because* their `τ_td` is more asymmetric (the roll coupling), and `k_v`
does nothing to `τ_td`.  Under H1 the fall rate follows the distance,
i.e. the speed, with no direction structure.

### E4 — The gait phase at the command

*Design.*  No extra runs: every E2/E3 run's press phase is recovered from
the log (§2), and the outcome binned into six 100 ms bins of the 0.6 s
clock, as doc/walk §6.3 did in MuJoCo.  In MuJoCo (§6) the six phases are
run explicitly.

*Reads.*  A phase dependence of the first landing's `Δφ` and of `fell` is
what H2 and H3 predict (the first swing after the step carries the whole
command's acceleration); H1 predicts none.  It also decides whether the
scripted schedule of §9 should synchronise the command to the four-foot
window — which would be the cheapest intervention the study can propose.

### E5 — Controls

The confound table of §3.3, each as a covariate in every model and, where
it needs runs of its own: the thermal row (E1-P, and the temperature column
once logged), the CoM row (a run whose HOLD read spread > 15 mm or was not
applied is repeated), the floor row (one cell re-run on a second patch).

## 5. Protocol on the robot

### 5.1 One run

1. `hw.fold2_walk --log gs_<cell>_<cmd>_<rep>.npz <the cell's flags>
   --pause-lean 0`, from the same start mark, the same heading.  The
   banner is photographed or copied into the run sheet (the npz carries no
   arguments — §9).
2. Limp, settle, crouch, rise; **HOLD still for ≥ 2 s** so `--com-auto`
   reads the CoM (its notice at T says what it read and whether it applied
   it; NOT APPLIED → park and repeat; the no-spring cell has no reading and
   starts from the same stillness).
3. T; 5 s in place (the at-rest arcs, the null segment of every metric).
4. The scheduled command, read from a stopwatch: the burst, then SPACE.
   The operator presses nothing else.  Past 45° the fall hold takes it;
   ENTER parks.
5. After the burst: 5 s in place, T, ENTER (park).  The exit report's
   lines (`walk … touchdowns anchored, leash acted on … sweeps`, the swing
   apex, `xy refused`) go on the run sheet.
6. ≥ 3 min before the next run of the same cell; the temperature noted at
   step 1 and step 5 while it is not in the log.

### 5.2 Order and blocking

Cells are run in blocks of a session (~20 runs, an hour of floor time);
within a block the cells alternate A/W and level so that no level is
always warmer than another; B0 is re-run at the start and the end of every
block (a drift check across the session).  A run is excluded, and
repeated, on: `imu_age` p95 > 30 ms, a sweep interval p95 > 7 ms, a CoM
read not applied, a bus gap, or a press outside its window by > 1 s.

### 5.3 Sample size, and the first hundred

With a fall rate of 5/8 against 1/8 Fisher's exact test gives a one-sided
p ≈ 0.06 (two-sided ≈ 0.12), so binary outcomes need N = 8 per cell for
the key comparisons and a directional hypothesis stated in advance; the
continuous responses (`τ_td`, `Φ`, `Δφ`, ~50 landings a run) separate
cells at N = 4–6.  The decision does not need all 250 runs of E2.  The
first hundred, in order:

| # | runs | what it settles |
|---|---|---|
| 1 | E0 | the metrics, and whether `τ_td` is already past the ramp in place |
| 2 | E1-C and E1-B (16) | accumulation vs memoryless; the pause |
| 3 | E2 slew (32) | the lag knob alone |
| 4 | E2 ramp (32) | the load knob alone |
| 5 | E3 matched pair + `k_v` (24) | H3 |

Everything after is refinement of the parameter map (§8's table), not of
the hypothesis.

### 5.4 Session 1: the first recorded session, about 25 runs

What today's observation lacks is a log.  Session 1 records B0 with the
failure observable, measures what the governor was contributing, and takes
the first three E2 cells — the ones that move the touchdown lag and the
load separately.  Every line below is a complete command; `<cell>` and
`<rep>` go into the file name because the npz does not keep the arguments.

```
# the common part of every run (the 2026-10-21 line; T = 0.6 s is fold2_trot's default)
B0="--duty 0.8 --contact-ramp 0.15 --tau-slew 300 --kp-joint 4 --kp-joint-abd 5"
RUN="python -m hw.fold2_walk $B0 --pause-lean 0"
```

| # | cell, runs | command line | schedule after T (stopwatch) | reads |
|---|---|---|---|---|
| 1 | `inplace`, 1 | `$RUN --log gs1_inplace_1.npz` | 120 s in place, T, ENTER | the null distribution of every metric; `τ_td` in place |
| 2–5 | `B0_lat`, 4 | `$RUN --log gs1_B0_lat_<rep>.npz` | 5 s in place; the lateral shuttle (A 8 s, D·D 8 s, A·A …) to the fall or 120 s; SPACE; 5 s; T; ENTER | E1-C: `t_fall`, the fall anatomy, `τ_td`, `Φ`, `Δφ`, `w_stance` |
| 6–8 | `B0_latpause`, 3 | `$RUN --log gs1_B0_latpause_<rep>.npz` | 5 s in place; A 8 s, SPACE 10 s, D 8 s, SPACE 10 s, … to the fall or 180 s | E1-B: does the pause reset `w_stance`, `s_land`, `ē_R` |
| 9–10 | `B0_fwd`, 2 | `$RUN --log gs1_B0_fwd_<rep>.npz` | the forward shuttle (W 8 s, S·S 8 s, …), 120 s | E3: the same metrics at a quarter of the `v_⊥` |
| 11–12 | `B0gov_lat`, 2 | `python -m hw.fold2_walk $B0 --log gs1_B0gov_lat_<rep>.npz` | the lateral shuttle, 120 s | today's configuration as flown: the governor's pauses and paused seconds from the exit report, against runs 2–5 |
| 13–15 | `slew120_lat`, 3 | `$RUN --tau-slew 120 --log gs1_slew120_lat_<rep>.npz` | the lateral shuttle, 120 s | E2: the lag knob alone |
| 16–18 | `ramp1875_lat`, 3 | `$RUN --contact-ramp 0.1875 --log gs1_ramp1875_lat_<rep>.npz` | the lateral shuttle, 120 s | E2: the load knob, the ceiling |
| 19–21 | `ramp075_lat`, 3 | `$RUN --contact-ramp 0.075 --log gs1_ramp075_lat_<rep>.npz` | the lateral shuttle, 120 s | E2: the load knob, half |
| 22–24 | `duty70_lat`, 3 | `$RUN --duty 0.7 --contact-ramp 0.129 --log gs1_duty70_lat_<rep>.npz` | the lateral shuttle, 120 s | E2: the H2/H3 discriminator |
| 25–26 | `B0_lat`, 2 more | `$RUN --log gs1_B0_lat_<rep>.npz` | the lateral shuttle, 120 s | the session's drift check against runs 2–5 |

(Flags later in a line override `$B0`'s; `--contact-ramp 0.1875` is the
ceiling at duty 0.80 and 0.19 is refused.)  Interleave the cells rather
than running each block whole, so that no cell is always the warmer
robot; three minutes between runs.

**The run sheet**, one row per run, written by hand until §9's sidecar
exists:

| column | from |
|---|---|
| run name, the full command line | typed |
| the banner's gait line (`gait … s at duty … (… ms of swing)`) and the swing line | the terminal at start |
| battery voltage at start and end | the pack |
| session time since power-on, and the motors' temperature if read (`hw.bringup` prints `temps()`; nothing in `hw.stand` does yet) | the clock |
| the CoM notice at T, if `--com-auto` ran (`read x/y, spread, applied / NOT APPLIED`) | the terminal at T |
| each key press and SPACE with its stopwatch time; the ENTER | the stopwatch |
| what happened: fell (which side, stopwatch time), fall hold taken, a bus gap, anything the operator saw (a scuff, a leg "not lifting", a lean) | the operator |
| the exit report: `law timing`, `walk … touchdowns anchored, leash acted on … sweeps, … landings clamped`, `swing foot apex … reached`, `xy … refused`, the governor's line | the terminal at exit |

**What Session 1 yields once the analysis script (§9, item 1) exists**:
the fall-anatomy timeline of every fall (roll, `contact_w`, `z_clear` and
`fz` of the four legs over the last 1.5 s); the `τ_td` distribution per
leg for B0 in place, lateral and forward, and for the slew, ramp and duty
cells beside it; `Φ` against `Δφ` within every run; `w_stance` and the
site offsets against time with the pause boundaries marked; the count of
scuffed swings per cell; the governor's pause count per 120 s of lateral
walking under B0.  Those six figures are the thesis chapter's results
section for the baseline, and the first test of H1 (the pause rows), H2
(the lag and load cells) and H3 (lateral against forward).

### 5.5 Safety

Governor off is a decision to *observe* the failure; the fall hold at 45°
(the joint PD at 5/0.2 + gravity, every fall in MuJoCo settled within
1.8° of the stand, hw/balance/README) is what protects the robot and it
stays on.  Bursts are bounded (30 s) in E2–E3; only E1 runs to the fall.
A session ends on any bus gap, any driver over-temperature flag (0x9A), or
two falls in a row with the fall hold whirling.

## 6. MuJoCo first

### 6.1 What the harness already does

`doc/walk/walksim.py` runs `hw.fold2_walk` as `hw.stand.main` builds it
(`entry(posture="fold2", …)`), with the estimator one sweep late, the
operator's keys from a schedule, and reports `fell`, tilt, torque, slip and
**`late`** — ms from the law's contact to the floor's first newton, the
clock's touchdown error against truth — per run; `longrun.py` adds the
tilt rms per 10 s and the drift over 30 s on the robot-like plant
(`HWLIKE`: rotor 0.6×, 0.1 N·m friction, IMU 15 ms, μ 0.7, CoM 3/5 mm off,
kt 0.9); `HwParams` takes a trunk push and the mass.  `walksim.entry` takes
every factor of §3.2 but `k_v`'s companion `STEP_MAX_XY` and the swing's
Kd (both are one keyword away).

### 6.2 The sim study, `gaitsim.py` (§9)

The E1–E4 grid on FOLD2, the impedance law, `--com-auto`, the robot-like
plant and the ideal one, six gait phases a cell, 60 s bursts (E1: 180 s),
about 15–25 s of wall time a run, four at a time: ~1 500 runs, an
afternoon.  Two things it answers that the robot cannot cheaply:

- the full factor map with the phase controlled, so the robot's hundred
  runs are spent on the cells the sim ranks as decisive;
- `τ_td` and `Φ` against the floor's truth (`foot_fn`), which is §6.3.

### 6.3 Calibrating the offline touchdown estimate

On every sim run the offline estimator of §2.1 is computed from the same
columns the robot log has (`x_b`, `p_swing`, `swing_s`, `contact_w`, roll,
pitch — the sim logs them) and compared with `foot_fn`'s first newton: its
median and p90 error, its false-early rate, per plant and per cell.  The
thresholds (4 mm, 8 N, 0.03 m/s, s ≥ 0.3) are moved until the false-early
rate is under 5 % on both plants and the p90 error under 15 ms; those are
the thresholds the robot logs are then read with, and their sim error is
quoted beside every robot `τ_td` in the thesis.  Without this step H2 is
tested with an instrument of unknown accuracy.

### 6.4 What the sim will not settle

The robot-like plant's swing is *worse* than the robot's (doc/contact §3:
7–9 mm of a 20 mm arc, 33–40 ms late at DOG5's rotor; the operator calls
the robot's apex good), so the sim's `τ_td` is an upper bound, not an
estimate; the sim loop runs at 4 ms where the robot's runs at 5.5; there
is no thermal drift, no backlash, no cable loom; and the fall of
2026-10-06 was *not* reproduced by the model (doc/walk §11.9: "the model
does not reproduce the robot's lagging swing").  The sim ranks the cells
and calibrates the instrument; the robot decides.

## 7. Analysis plan and the decision table

### 7.1 Models

- Per-landing responses (`τ_td`, `Φ`, `Δφ`, `e_land`) are nested in runs:
  run medians and p90s are the unit for cell comparisons (Kruskal–Wallis
  across levels, Mann–Whitney pairwise, bootstrap CIs on the medians);
  within-run Spearman ρ between `Φ_n` and `Δφ_n` for H2's coupling.
- `fell`: Fisher's exact test per cell pair; a logistic regression of
  `fell` on cell-median `Φ`, `v_⊥`, `T_sw` and the covariates of §3.3,
  with the run as the unit.
- `t_fall`: Kaplan–Meier per cell (censoring at the burst's end), a
  Weibull fit for the shape (E1), a Cox model on the same predictors.
- Trends (H1): Mann–Kendall per run on `w_stance,n`, `s_land,n`, `ē_R,n`,
  `xy_err_n`, `kt_n`, and the burst-boundary comparison of E1-B as a paired
  test (end of burst k−1 vs start of burst k).
- Phase (E4): `fell` and the first `Δφ` by phase bin, χ² and
  Kruskal–Wallis.

### 7.2 The decision table

| pattern of results | supports | what the controller needs next |
|---|---|---|
| Weibull k ≈ 1; E1-B bursts start fresh; no trend in `w_stance`, `ē_R`; falls track `Φ` across the slew/ramp/Kd_z cells; the pre-fall landings are side-asymmetric in `τ_td` | **H2** | contact-driven loading: the clock's touchdown replaced or gated by a detected one (the reverted scheme A re-tried from E0's robot numbers, or a measured-torque witness), and the ramp at its ceiling meanwhile |
| falls collapse onto `v_⊥`; the matched pair agrees; `k_v` 0.13 helps A/D only; duty 0.80 wins even at larger `Φ` | **H3** | a capture-point lateral foothold (`k_v` → √(h/g)), the step bound widened in y, and a shorter two-foot phase |
| k > 1.5; bursts resume where they left off; `w_stance` or `s_land` trends with distance; E1-S fails at the same distance; the in-place null is flat | **H1** (stance drift) | the foothold's noise (the placement's low-pass, `v_hat`) and a stance that is re-centred on the sites while walking, not only at rest |
| k > 1.5; bursts resume; `kt_n` or temperature trends; E1-S fails at the same *time*; E1-P drifts too | H1's thermal row | a torque-constant compensation, and shorter sessions |
| k ≈ 1 but `τ_td` is small (< ramp) and `Φ` ≈ 0; falls do not track `Φ`; no `v_⊥` structure | none of the three | the SRB gains and the x/y loop, next (doc/walk §11.3's CoM path first) |

Mixed patterns are read in the order H2, H3, H1: a rising hazard with
asymmetric landings is H2 triggered more often as H1 lowers the margin,
and the fix is still the contact-driven loading first.

## 8. What goes into the thesis

A chapter, *Experimental validation of the time-driven gait on DOG6*:

1. **Problem statement** — O1–O4 and the recovery observation, with the
   2026-10-06 timeline as the motivating figure (roll, `contact_w`,
   `z_clear` and `fz` of the four legs over the last 1.5 s).
2. **Hypotheses and mechanisms** — §1, with two figures: the diagonal
   geometry (the FOLD2 stance, the support line, `v_⊥` for W and A) and the
   phantom-load curve (weight asked vs lag, one line per ramp).
3. **Metrics** — §2's tables, and the calibration of the offline touchdown
   estimate against MuJoCo truth (§6.3): error histograms per plant.
4. **Method** — §3–6: the factors, the confounds and their controls, the
   protocol, the sim pre-study.
5. **Results** — per experiment: E1's survival curves and Weibull shapes
   with the burst-boundary paired plot; E2's `Φ` vs fall-rate scatter with
   the cells labelled, and `Δφ` vs `Φ` within runs; E3's fall rate vs
   `v_⊥` with the matched pair; E4's phase histogram.
6. **The parameter map** — one table: each gait parameter, its tested
   range, its effect on `τ_td`, `Φ`, `Δφ` p90 and the fall rate, with CIs;
   the best cell found and its margin to B0.
7. **Discussion** — the decision table's row that the data landed in, what
   it says about the SRB-PD architecture's limit (a memoryless balance law
   with a memoryless clock has no state in which to be wrong, and no state
   in which to be right about a foot that is not down), and the next
   controller: contact-driven phase and a lateral capture term, in the
   order the data chose.

## 9. The code to write next (not written here)

In the order the study needs it:

1. `doc/gait_study/analyse.py` — the offline metrics of §2 from an
   `hw.stand` npz (and from `hwsim`'s result dict): the touchdown estimate,
   `Φ`, `Δφ`, scuffs, stance width, the per-cycle biases, the trends, the
   hazard; a table per run and the figures of §8.  E0 is this script on
   the existing logs.
2. Logging additions in `hw/stand.py`: the drivers' temperature per sweep
   (`MotorBus.temps()` reads the 0xA1 reply's byte), the walk's `v_ref`,
   `leashed` and the governor's lean, and a sidecar JSON of the parsed
   arguments beside every npz (the banner, kept).
3. A scripted command schedule for the walk hook (`--script "5:A,25:SPACE"`
   or a file), so a burst is the same burst in every run and the press can
   be synchronised to the four-foot window for E4.
4. `doc/gait_study/gaitsim.py` on `walksim.entry` — the §6.2 grid, the
   `late` and `Φ` truth columns, the §6.3 calibration.
5. The fix for the `qd` log column, so the velocities need not be
   differentiated.

None of it touches the law.

## References

- Di Carlo, Wensing, Katz, Bledt, Kim, *Dynamic Locomotion in the MIT
  Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018,
  [doi:10.1109/IROS.2018.8594448](https://doi.org/10.1109/IROS.2018.8594448)
  — the reference generator, eq (33), the gait scheduler this one follows.
- Bledt, Powell, Katz, Di Carlo, Wensing, Kim, *MIT Cheetah 3: Design and
  Control of a Robust, Dynamic Quadruped Robot*, IROS 2018,
  [doi:10.1109/IROS.2018.8593885](https://doi.org/10.1109/IROS.2018.8593885)
  — the SRB balance controller and its force distribution.
- Bledt, Wensing, Ingersoll, Kim, *Contact Model Fusion for Event-Based
  Locomotion in Unstructured Terrains*, ICRA 2018,
  [doi:10.1109/ICRA.2018.8460904](https://doi.org/10.1109/ICRA.2018.8460904)
  — probabilistic contact estimation from the gait phase, foot height and
  force, without a foot sensor: the event-driven alternative H2 points to.
- Raibert, *Legged Robots That Balance*, MIT Press 1986 — the foothold's
  velocity feedback term (`STEP_KV`).
- Pratt, Carff, Drakunov, Goswami, *Capture Point: A Step toward Humanoid
  Push Recovery*, IEEE-RAS Humanoids 2006, pp. 200–207
  ([record](https://portfolio.erau.edu/en/publications/capture-point-a-step-toward-humanoid-push-recovery/))
  — the √(h/g) gain H3 compares `STEP_KV` against.
- Hutter et al., *ANYmal — a highly mobile and dynamic quadrupedal robot*,
  IROS 2016, [doi:10.1109/IROS.2016.7758092](https://doi.org/10.1109/IROS.2016.7758092)
  — a time-driven gait with contact-triggered phase transitions, the
  design the decision table's first row leads to.
- Kalbfleisch, Prentice, *The Statistical Analysis of Failure Time Data*,
  2nd ed., Wiley 2002, ISBN 978-0-471-36357-6
  ([publisher](https://www.wiley-vch.de/en/areas-interest/mathematics-statistics/the-statistical-analysis-of-failure-time-data-978-0-471-36357-6))
  — Weibull hazard shapes and censored survival analysis (E1).
- `doc/walk/README.md` §4, §6.3–6.4, §11 and `doc/contact/README.md` at
  `6f51e56` — the numbers quoted above.
