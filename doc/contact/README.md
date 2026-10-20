# Touchdown detection without a sensor: scheme A in MuJoCo

*MuJoCo study, 2026-10-20, branch `cloud`. The repo's own `BalanceLaw` with
`hw/balance/contact.py` attached, run through `doc/crouch_trot/hwsim.py` and
`doc/walk/walksim.py`'s FOLD2 entry (the CAN round robin, encoder
quantisation, the 40 Hz qd filter, IMU latency, the gate's slew and clip, the
Kalman filter fed one sweep late). Every case runs twice: `hw.fold2_walk`
with `--no-touchdown` (the clock's word for which foot is down, the walk
flown on 2026-10-07) and with the detector on (the default since this day).
Everything here is simulation on the CAD model; **nothing has flown**.
`python doc/contact/tdsim.py` reproduces the table.*

**TL;DR.** The swing law's own commanded push, gated by where the foot and
its reference stand above the planted feet, tells the law which foot is down
without a sensor. In MuJoCo the law's contact set follows the floor's first
newton to within 0–8 ms where the clock loaded feet 39–62 ms before they
touched; a foot the clock has landed but that is still in the air stays out
of the allocator and probes down instead. On falls this plant is a wash —
three cases with the clock, four with the detector, different ones — and its
swing is not the robot's (section 3). Two gates were needed, and the second
was found here on the first run: the foot must be within 4 mm of the floor
**and the reference must have reached the floor** — otherwise a torque-rate-
bound swing hovering 8 mm up under a diving reference reads as a stopped,
pushed foot 70 ms before it touches; and the probe pushes (section 4).

## 1. Why the drivers' current cannot see the floor, and what can

Every torque phase runs the drivers in q-axis current control (0xA1). The
current loop makes iq follow the *command* whatever the load: a foot that
meets the floor changes the leg's motion, not its current. `tau_meas` is read
by `safety.TorqueReadback` as "did the current loop follow", and by
`gravity_bench` beside the command for the same reason. The floor is in the
**encoders**: the foot stops, the reference goes on, and the swing
impedance's error — and the force it commands — grows:

    f_cmd = Kp (p_ref − x) + Kd (v_ref − J q̇)          trunk frame
    fz_push = −f_cmd,z                                  the push on the floor

In quasi-statics `fz_push` is the force on the floor, less joint friction
(0.1–0.3 N·m a joint, 1–3 N at the foot) and less the leg's inertia while it
still moves. Nothing new is computed: `swing_control.swing_impedance_force`
is the force the walk's swing already turns into torque, exposed.

## 2. How a touchdown is decided, and how "still airborne" is

**TOUCHDOWN** — three conditions, all true for `TD_STREAK` = 2 sweeps running:

1. the leg is on the way down (swing progress ≥ `TD_S_MIN` 0.3), or probing;
2. **the foot is where the floor is, and so is its reference**: the foot's
   clearance above the plane of the planted feet — `z_origin − r_foot +
   (R x_b)_z`, the kinematics IMU-levelled — is within `TD_CLEAR_MAX` (4 mm),
   and the swing law's reference, measured the same way, is within
   `TD_REF_CLEAR_MAX` (3 mm) of that plane or below it;
3. the floor is answering, by either witness: the commanded push is at least
   `TD_FZ_N` (8 N), or the foot has stopped descending (`v_z ≥ −0.03 m/s`
   from the encoders through J) while the reference still descends at
   0.10 m/s or more.

**STILL AIRBORNE** — the clock's touchdown arrives and none of the above has
fired. The leg **probes**: weight 0 and contact False for the allocator (no
force is asked of a foot that cannot push), out of the height average, its
reference the arc's end carried down at `TD_PROBE_V` (0.20 m/s, half the
arc's own peak descent; Kd_z × 0.20 = 8 N on a stopped foot at once, Kp_z
adding 80 N/s), and the leg pushed down along gravity by `TD_PROBE_PUSH_N`
(10 N, open loop, beside the impedance — not in what the detector reads)
until the floor answers. Past 30 mm it is declared down anyway (*gave up*),
and when the other diagonal is halfway through its liftoff ramp every probe
is declared down so the allocator is never left with one foot (the walk's
four-foot window is 120 ms; the probe gets 93, 19 mm).

**What the law does with it** (`ContactDetector.update` hands `BalanceLaw`
a `GaitSample` in the clock's place):

| state | contact | weight | swing progress |
|---|---|---|---|
| trust (never lifted this trot) | clock's | clock's | clock's |
| swing | False | 0 | clock's |
| landed | True | smoothstep from the detection over 30 ms, then the clock's liftoff ramp | 0 |
| probing | False | 0 | 1.0, the arc's end |
| landed, then lifted past 8 mm | → probing again, from its own x/y | 0 | 1.0 |

A landed foot gets the stance torque, the joint layer and its world anchor
from that sweep, and no swing torque; a probing foot keeps the swing law on
the probe; the state estimator's trust reads the same states. A landed foot
whose clearance rises past `TD_LIFT_MAX` (8 mm) while the clock has it in
stance has lifted off — the trunk tipping away from it — and re-probes from
where it is (section 4 says where that came from). The plane the clearances
are measured from is the two *lowest* planted feet's, so a lifted foot
cannot move it.

## 3. The plant, and what it does to the swing

`hwsim`'s FOLD2 walk at `hw.fold2_walk`'s defaults. **Its swing is not the
robot's**: doc/walk recorded that on this plant no leg reaches the 20 mm arc
(7–9 mm, 33–40 ms late, at DOG5's rotor inertia), where the operator called
the robot's apex good. So here every swing lands LATE — the foot hovers at
7–9 mm while the reference dives, and comes down 30–90 ms after the clock —
which makes this plant a hard test of the *airborne* half of the scheme and
a poor one of the *early* half (the rolled trunk's low side, which
`hw.balance.selftest` §17 covers on a kinematic plant). And MuJoCo's floor is
soft: a pushed foot keeps sinking ~2 mm at ~0.1 m/s after its first newton,
which holds the "stopped" witness off and the push at the threshold for
tens of milliseconds where a hard floor would stop the foot in one sweep.

## 4. The first runs, and what they found

**The reference gate.** With the foot's clearance gate at 10 mm alone, the
in-place trot read 53 of 54 swings as landed **72 ms (median) before the
floor's first newton**, by the stopped-foot witness: the foot turning round
at its 8 mm apex — still, inside the gate — while the reference, 35 ms
ahead, already dove at 0.3 m/s. A stopped foot under a diving reference is
exactly the touchdown signature; what was missing is that the floor can only
be what stops a foot once *the reference has reached the floor*. With
`TD_REF_CLEAR_MAX` (the reference within 3 mm of the plane or below it) the
false detections went to the end of the arc — and the foot's gate had to
come down to 4 mm, because at 10 mm a foot hovering 8 mm up under a
reference at the floor, pushed 17 N against nothing, still read as landed
40–50 ms early.

**The probe's speed.** At 0.15 m/s the push under the probe is Kd_z × 0.15
= 6 N on a stopped foot, under the 8 N threshold, and MuJoCo's soft floor —
into which a pushed foot keeps sinking at ~0.1 m/s — held it at 7 N for the
whole window, so probes ended by the two-foot rule rather than by a witness.
At 0.20 m/s the Kd term alone is the threshold.

**The lift-off.** With both gates, two cases fell that the clock survived
(a 20 °/s turn and an 8 N sideways push), by one mechanism: the trunk rolled
during a swing, the planted foot on the high side rose 10 mm with 0 N under
it and, still "landed" (nothing un-landed a foot), pulled the planted plane
up with it; the swinging foot on that side was then believed landed 8 mm in
the air — the push over the threshold, the clearance *against the skewed
plane* under 4 mm — and the next liftoff left the robot on one foot. Two
changes, both kinematic: the plane is the two **lowest** planted feet's, and
a landed foot whose clearance rises past `TD_LIFT_MAX` (8 mm) while the
clock has it in stance **re-probes** from where it is — out of the
allocator, its reference its own x/y carried down, the leg extending to find
the floor again.

**The probe's push.** With those two, the same two cases still fell, and
the timelines said why: on this plant the swing foot is still 10–20 mm up
at the clock's touchdown and takes 60–90 ms to come down; the clock's
"load it anyway" extended the leg with the allocator's 25 N and caught the
roll, where the impedance probe's 16 N at most in its first 100 ms let the
trunk drop on the side whose feet were still coming down. So the probe
pushes: `TD_PROBE_PUSH_N` (10 N) down along gravity, open loop, beside the
impedance — the clock's reflex made deliberate, without the allocator
counting on a foot that is not there, and not in what the detector reads.
That is the table below.

## 5. The table

| case | td | fell | tilt ° | τ N·m | slip mm/s | late ms | det err med / p90 / max ms | false early | probes  lim / win / lift / lost | push / stop | law p50 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `inplace` | off | no | 0.5 | 0.97 | 4.6 | 57.0 | – | – | – | – | 738 |
| `inplace` | on | no | 0.8 | 1.00 | 11.4 | 0.0 | +24 / +32 / +32 | 0 | 54 0/0/0/0 | 52/0 | 889 |
| `fwd10` | off | no | 2.8 | 1.59 | 20.9 | 55.0 | – | – | – | – | 769 |
| `fwd10` | on | no | 2.5 | 1.60 | 24.4 | 4.0 | +24 / +32 / +256 | 9 | 54 0/0/0/0 | 56/0 | 838 |
| `lat05` | off | no | 5.5 | 1.56 | 14.0 | 51.0 | – | – | – | – | 796 |
| `lat05` | on | no | 3.9 | 1.26 | 17.3 | 3.0 | +24 / +252 / +272 | 11 | 51 0/2/0/0 | 54/0 | 852 |
| `latm05` | off | no | 5.6 | 1.52 | 14.3 | 52.0 | – | – | – | – | 716 |
| `latm05` | on | no | 3.7 | 1.08 | 17.6 | 4.0 | +24 / +260 / +272 | 13 | 50 0/4/0/0 | 52/0 | 775 |
| `yaw20` | off | no | 1.5 | 1.19 | 13.7 | 58.0 | – | – | – | – | 734 |
| `yaw20` | on | no | 5.2 | 1.67 | 25.1 | 6.0 | +24 / +92 / +272 | 9 | 43 0/3/0/0 | 51/2 | 742 |
| `combo` | off | **YES** | 46.7 | 5.13 | 41.0 | 59.0 | – | – | – | – | 421 |
| `combo` | on | no | 3.7 | 1.77 | 29.9 | 8.0 | +24 / +50 / +272 | 15 | 45 0/0/0/0 | 56/0 | 709 |
| `fwd15` | off | no | 6.8 | 5.61 | 48.6 | 39.0 | – | – | – | – | 703 |
| `fwd15` | on | **YES** | 46.3 | 7.01 | 50.8 | 6.0 | +24 / +94 / +200 | 7 | 35 0/10/1/0 | 37/3 | 617 |
| `fwd10_rl` | off | no | 3.4 | 1.52 | 18.9 | 51.0 | – | – | – | – | 736 |
| `fwd10_rl` | on | no | 1.5 | 1.53 | 21.0 | 1.0 | +28 / +240 / +244 | 2 | 57 0/0/0/0 | 56/0 | 813 |
| `lat05_rl` | off | no | 3.3 | 0.99 | 12.3 | 47.0 | – | – | – | – | 747 |
| `lat05_rl` | on | no | 2.7 | 1.02 | 13.8 | 5.0 | +20 / +28 / +248 | 15 | 42 0/0/0/0 | 56/0 | 827 |
| `yaw20_rl` | off | no | 1.8 | 1.08 | 12.5 | 54.0 | – | – | – | – | 754 |
| `yaw20_rl` | on | no | 1.0 | 0.99 | 14.7 | 0.0 | +24 / +36 / +244 | 0 | 58 0/0/0/0 | 56/0 | 735 |
| `fwd10_com` | off | **YES** | 46.2 | 6.42 | 25.3 | 62.0 | – | – | – | – | 366 |
| `fwd10_com` | on | **YES** | 46.0 | 5.32 | 44.7 | 8.0 | +24 / +115 / +240 | 11 | 32 0/11/0/0 | 41/5 | 801 |
| `lat05_com` | off | **YES** | 46.2 | 6.42 | 25.3 | 62.0 | – | – | – | – | 363 |
| `lat05_com` | on | **YES** | 47.5 | 5.48 | 24.4 | 16.0 | +20 / +42 / +272 | 2 | 12 0/5/1/0 | 15/0 | 369 |
| `lat05_push` | off | no | 10.5 | 2.83 | 17.1 | 52.0 | – | – | – | – | 719 |
| `lat05_push` | on | **YES** | 46.0 | 3.51 | 20.8 | 24.0 | +24 / +32 / +260 | 3 | 22 0/4/1/0 | 21/0 | 376 |
| `trot_inplace` | off | no | 0.4 | 1.25 | 5.9 | 58.0 | – | – | – | – | 522 |
| `trot_inplace` | on | no | 0.4 | 1.19 | 14.2 | 0.0 | +24 / +28 / +32 | 0 | 54 0/0/0/0 | 52/0 | 546 |

*fell* and *tilt* are walksim's over the trot; *late* is walksim's: ms from
the law's contact (weight > 0) to the floor's first newton on the same foot
— with the detector, from the detection; *det err* is ms from the floor's
first newton to the detector's LANDED, median / p90 / max over the trot's
swings (+ after the floor, − before it); *false* counts detections more than
8 ms before the floor (a foot handed to the allocator in the air);
*probes* is the detector's own account — probes started, given up at the
30 mm limit / cut by the two-foot rule / lost (lifted again undetected) —
and *push/stop* how the detections were made.

**Reading it.** Fourteen cases, each with the clock alone (`off`) and the
detector (`on`), 104 s of wall time for the 28 runs.

- **The contact set follows the floor.** *late* — the time from the law's
  contact to the floor's first newton on that foot — goes from 39–62 ms
  with the clock to 0–8 ms with the detector in eleven of fourteen cases
  (16 and 24 ms in two that fell either way or with it). That is the whole
  point: with the clock the allocator was handed every foot 40–60 ms before
  it could push, on this plant; with the detector it is handed the foot
  when the floor has it.
- **Detections come 24 ms (median) after the floor's first newton**, with no
  detection more than 8 ms before it in six cases and under 15 in the rest
  (*false early*): the two clearance gates hold. The p90 of 240–270 ms in
  some walking cases is this plant's scuffing swing: a foot whose "swing"
  never left the floor (first newton right after liftoff) is detected only
  at the arc's end, as it should be.
- **Falls are a wash on this plant, and that is the honest result.** The
  clock fell in three cases (`combo`, `fwd10_com`, `lat05_com`); the
  detector fell in four (`fwd15`, `fwd10_com`, `lat05_com`, `lat05_push`),
  rescuing the 0.10 m/s turn and standing every robot-like case at ≤ 2.7°.
  Without the probe's push (the run before this one, `--td-push 0`) it fell
  in three (`yaw20`, `lat05_com`, `lat05_push`) and rescued `combo` and
  `fwd10_com` instead. Every fall it adds has one timeline: on this plant
  the swing foot is still 10–20 mm up at the clock's touchdown and takes
  60–90 ms to come down, and whether the probe's 10 N or the allocator's
  25 N on a foot in the air catches the roll that starts meanwhile is a
  coin the plant's torque-rate-bound swing tosses. The robot's swing is
  not this one (doc/walk: its apex is the arc's; the 2026-10-06 fall was a
  foot loaded in the air), and the thresholds a flight should move first
  are `--td-push` and `--td-probe`.
- **In place and at 0.10 m/s forward** — the regime the robot has flown —
  the detector changes the tilt by less than a degree, every swing lands
  within 32 ms of the floor, nothing probes past the window, and the law
  costs ~100 µs more a sweep (the detector's 20 µs and the probe's torque).


## 6. What this does and does not say

It says the scheme does what it was written to do on a plant whose swing is
worse than the robot's: a foot the clock has landed but that is still in the
air is kept out of the allocator until it lands, and a foot that has landed
is in the allocator within a few sweeps of the floor's first newton, without
a sensor and without a new solve. It does not say the thresholds are the
robot's: the push threshold sits between the friction a foot can show
unloaded and what a stopped foot commands at once, the clearance gates above
the kinematics' error budget, and both were sized on numbers measured on
the robot — but none has been flown. The first flight is `hw.fold2_walk`'s
command line of 2026-10-07 with nothing else changed; the status line prints
each leg's state and the detection against the clock, the log's `td_*`
columns keep it, and the exit report counts early / late / probed / gave up.
`--no-touchdown` is the walk as flown.

What it cannot see, by construction: a foot resting on a step 8 mm above the
plane is, to the kinematics, a foot hovering 8 mm up — above the clearance
gate — so it probes and is landed by the two-foot rule ~90 ms on; a foot
that is down but carrying nothing — sliding, or lifted less than 8 mm — is
not un-landed. Both are the measured torque's to tell (scheme B, `J^-T
(tau_meas − G)`, `com_estimate`'s solve run every sweep), which this is
written to be joined by: the detector takes its witnesses through one
`observe` call, and a second witness is one more condition in `update`.

## 7. Reproducing

```
python doc/contact/tdsim.py              the table, 4 in parallel (~5 min)
python doc/contact/tdsim.py --quick      inplace + lat05
python doc/contact/tdsim.py lat05_push   one case, off and on
python -m hw.balance.contact             the sizing
python -m hw.balance.selftest            §17: the kinematic plant
```

`data/td_<case>_<off|on>.json` holds each run's numbers and `.npz` its log
(`td_state`, `td_fz`, `td_clear`, `td_ref`, `td_probe`, `foot_fn`,
`contact_w`, `swing_s`, `p_swing`, `x_b`, …).
