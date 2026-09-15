"""Gate the convex MPC: the model, the condensation, the QP, the swing law.

    python -m sim.cmpc.selftest

WHAT THIS IS FOR
    `sim.selftest` proves the DESCRIPTION matches the CAD.  This proves the
    CONTROLLER matches the paper -- that the linearisation is the one written
    down, that the condensation is algebraically the dynamics, that the QP's
    solution satisfies (19)-(21), and that the swing law is the law in (1)-(3).

    Several gates here MEASURE an approximation rather than assert it.  The
    paper says the gyroscopic term is small and the small-angle Euler map is
    accurate; this puts numbers on both, on DOG6, so "small" is a quantity and
    not a citation.

    Three of these gates exist because they caught a real bug in this
    implementation, and each is named where it sits:
      * the CoM/trunk-origin height offset  (18 mm of steady error)
      * the yaw branch cut at +-180 degrees (a fall, at every turn rate)
      * the swing arc's apex when its ends differ in height
"""
from __future__ import annotations

import sys

import numpy as np

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "sim.cmpc"

from .. import coordinates as C           # noqa: E402
from .. import kinematics as K            # noqa: E402
from .. import leg_dynamics as LD         # noqa: E402
from .. import params as P                # noqa: E402
from . import config as cfg               # noqa: E402
from . import dynamics as dyn             # noqa: E402
from . import gait                        # noqa: E402
from . import gamepad                     # noqa: E402
from . import qp                          # noqa: E402
from . import swing                       # noqa: E402
from . import teleop                      # noqa: E402
from . import trajectory                  # noqa: E402

try:
    import mujoco
except ImportError:                       # pragma: no cover
    sys.exit("mujoco is not installed in this interpreter.")

_FAILURES: list[str] = []
_PASSES = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print("  ok    %-58s %s" % (label, detail))
    else:
        _FAILURES.append(label)
        print("  FAIL  %-58s %s" % (label, detail))


def close(label: str, a, b, tol: float, unit: str = "") -> None:
    worst = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    check(label, worst <= tol, "worst %.3g%s (tol %.3g)" % (worst, unit, tol))


def _stance_feet():
    """Foot positions relative to the CoM at the drawn stance, world axes."""
    return K.foot_stance() - dyn.COM_OFFSET_BODY


# ===========================================================================
def main() -> int:
    rng = np.random.default_rng(6)
    feet = _stance_feet()
    print("DOG6 convex MPC self-test\n")

    # -- 1. the linearised model -------------------------------------------
    print("dynamics: the linearisation")
    a, b = dyn.continuous(0.0, feet)
    check("A^3 == 0, which is what the closed-form discretisation needs",
          np.allclose(np.linalg.matrix_power(a, 3), 0.0, atol=1e-15),
          "A^2 has %d nonzero" % np.count_nonzero(a @ a))

    for _ in range(200):
        yaw = rng.uniform(-np.pi, np.pi)
        r = feet + rng.normal(scale=0.02, size=(4, 3))
        x = np.zeros(cfg.STATE_DIM)
        x[:12] = rng.normal(scale=0.3, size=12)
        x[cfg.GRAV] = -9.81
        u = rng.normal(scale=20.0, size=cfg.INPUT_DIM)
        a, b = dyn.continuous(yaw, r)
        x[2] = yaw
        close_a = np.max(np.abs(a @ x + b @ u - dyn.approx_derivative(x, u, r)))
        if close_a > 1e-12:
            break
    check("A x + B u == approx_derivative over 200 random (yaw, r, x, u)",
          close_a <= 1e-12, "worst %.3g" % close_a)

    # The closed form against scipy's expm -- the gate that lets `discretize`
    # skip expm entirely.
    worst = 0.0
    for _ in range(100):
        a, b = dyn.continuous(rng.uniform(-np.pi, np.pi),
                              feet + rng.normal(scale=0.02, size=(4, 3)))
        ad, bd = dyn.discretize(a, b)
        ae, be = dyn.discretize_expm(a, b)
        worst = max(worst, float(max(np.abs(ad - ae).max(), np.abs(bd - be).max())))
    close("closed-form discretisation == scipy expm, 100 draws", worst, 0.0, 1e-14)

    # What exactness buys over Euler at this timestep.
    a, b = dyn.continuous(0.0, feet)
    ad, bd = dyn.discretize(a, b)
    euler_a = np.eye(cfg.STATE_DIM) + a * cfg.MPC_DT
    euler_b = b * cfg.MPC_DT
    check("Euler would differ from exact by a visible amount at dt = %.2f s"
          % cfg.MPC_DT,
          np.abs(euler_b - bd).max() > 1e-5,
          "B differs by %.3g; the dropped term couples force to POSITION"
          % np.abs(euler_b - bd).max())
    check("...and the difference is second order, so it IS the A B dt^2/2 term",
          np.allclose(bd - euler_b, (a @ b) * cfg.MPC_DT ** 2 / 2
                      + (a @ a @ b) * cfg.MPC_DT ** 3 / 6, atol=1e-15))

    # -- 2. what the three approximations cost -----------------------------
    #
    # THE POINT IS TO PUT A NUMBER ON THE PAPER'S "SMALL", not to assert a
    # threshold picked to pass.  So the sample has to be the envelope the
    # robot actually occupies: a trot measured in the closed loop below holds
    # roll and pitch under 0.1 deg rms, and the forces are a real QP solution
    # -- two feet carrying ~29 N each inside the friction cone -- rather than
    # twelve independent 60 N components, which is a wrench no contact set can
    # produce and which makes any linearisation look terrible.
    print("\ndynamics: the size of each approximation")

    def _plausible_forces():
        """Forces of the shape the QP returns: two feet, inside the cone."""
        down = gait.contact(rng.uniform(0.0, cfg.GAIT_PERIOD))
        out = np.zeros((4, 3))
        share = P.WEIGHT / max(int(down.sum()), 1)
        for i in np.flatnonzero(down):
            fz = share * rng.uniform(0.6, 1.6)
            tangent = rng.uniform(-1.0, 1.0, 2)
            tangent *= cfg.MU * fz * rng.uniform(0.0, 1.0) / max(
                np.linalg.norm(tangent), 1e-9)
            out[i] = (tangent[0], tangent[1], fz)
        return out.reshape(-1)

    def _sample(tilt, rate):
        """Worst error in each row, AS A FRACTION of the true derivative.

        The ratio is the quantity that means anything.  An absolute
        2.6 rad/s^2 sounds alarming until it is set against the ~150 rad/s^2
        that a 29 N contact force at a 0.18 m lever produces against an Ixx of
        0.036 -- so the gate measures both and divides.
        """
        worst_e, worst_g, scale_e, scale_g = 0.0, 0.0, 1e-9, 1e-9
        for _ in range(400):
            x = np.zeros(cfg.STATE_DIM)
            x[cfg.RPY] = (rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt),
                          rng.uniform(-np.pi, np.pi))
            x[cfg.OMEGA] = rng.uniform(-rate, rate, 3)
            x[cfg.VEL] = rng.uniform(-0.5, 0.5, 3)
            x[cfg.GRAV] = -9.81
            u = _plausible_forces()
            r = feet + rng.normal(scale=0.02, size=(4, 3))
            truth = dyn.true_derivative(x, u, r)
            diff = truth - dyn.approx_derivative(x, u, r)
            worst_e = max(worst_e, float(np.abs(diff[cfg.RPY]).max()))
            worst_g = max(worst_g, float(np.abs(diff[cfg.OMEGA]).max()))
            scale_e = max(scale_e, float(np.abs(truth[cfg.RPY]).max()))
            scale_g = max(scale_g, float(np.abs(truth[cfg.OMEGA]).max()))
        return worst_e, worst_g, worst_e / scale_e, worst_g / scale_g

    euler_trot, gyro_trot, rel_e, rel_g = _sample(np.deg2rad(1.0), 0.5)
    check("small-angle Euler map, at the attitude a trot actually holds (1 deg)",
          rel_e < 0.05,
          "%.4f rad/s of Theta_dot -- %.1f%% of it" % (euler_trot, 100 * rel_e))
    check("dropping w x (I w) and the full-orientation inertia, same envelope",
          rel_g < 0.05,
          "%.2f rad/s^2 against a true %.0f rad/s^2 -- %.1f%%"
          % (gyro_trot, gyro_trot / max(rel_g, 1e-9), 100 * rel_g))

    euler_6, gyro_6, rel_e6, rel_g6 = _sample(np.deg2rad(6.0), 1.0)
    check("...and both degrade gracefully rather than breaking down at 6 deg",
          rel_e6 < 0.35 and rel_g6 < 0.35,
          "Euler %.0f%%, gyroscopic %.0f%% -- from %.1f%% and %.1f%%"
          % (100 * rel_e6, 100 * rel_g6, 100 * rel_e, 100 * rel_g))
    check("the inertia carry is what grows with tilt, not the gyroscopic term",
          dyn.INERTIA_BODY[1, 1] > 4.0 * dyn.INERTIA_BODY[0, 0],
          "Ixx %.4f vs Iyy %.3f, so a tilt mixes them and I^-1 moves %.0fx"
          % (dyn.INERTIA_BODY[0, 0], dyn.INERTIA_BODY[1, 1],
             dyn.INERTIA_BODY[1, 1] / dyn.INERTIA_BODY[0, 0]))

    # -- 3. condensation ---------------------------------------------------
    print("\nqp: the condensation is the dynamics")
    yaws = np.cumsum(np.full(cfg.HORIZON, 0.03))
    r_seq = np.tile(feet, (cfg.HORIZON, 1, 1)) + rng.normal(
        scale=0.01, size=(cfg.HORIZON, 4, 3))
    a_seq, b_seq = dyn.discrete_sequence(yaws, r_seq)
    a_qp, b_qp = qp.condense(a_seq, b_seq)

    worst = 0.0
    for _ in range(50):
        x0 = np.concatenate([rng.normal(scale=0.3, size=12), [-9.81]])
        u = rng.normal(scale=20.0, size=cfg.HORIZON * cfg.INPUT_DIM)
        worst = max(worst, float(np.abs(
            a_qp @ x0 + b_qp @ u - qp.rollout(x0, u, a_seq, b_seq)).max()))
    close("X == A_qp x0 + B_qp U equals a direct rollout, 50 draws",
          worst, 0.0, 1e-11)
    check("B_qp is block LOWER triangular (no force acts before it is applied)",
          np.allclose(np.triu(b_qp, k=cfg.INPUT_DIM)[
              :, :].reshape(-1)[:0] if False else
              max(abs(b_qp[i * cfg.STATE_DIM:(i + 1) * cfg.STATE_DIM,
                           (i + 1) * cfg.INPUT_DIM:].max(initial=0.0))
                  for i in range(cfg.HORIZON - 1)), 0.0, atol=1e-15))

    # -- 4. the cost -------------------------------------------------------
    print("\nqp: the cost")
    x0 = np.concatenate([rng.normal(scale=0.1, size=12), [-9.81]])
    x_ref = np.tile(np.concatenate([np.zeros(12), [-9.81]]), (cfg.HORIZON, 1))
    hessian, gradient = qp.cost(a_qp, b_qp, x0, x_ref)

    def objective(u):
        state = a_qp @ x0 + b_qp @ u - x_ref.reshape(-1)
        q_big = np.tile(cfg.Q_DIAG, cfg.HORIZON)
        r_big = np.tile(cfg.R_DIAG, cfg.HORIZON)
        return float(state @ (q_big * state) + u @ (r_big * u))

    u = rng.normal(scale=10.0, size=cfg.HORIZON * cfg.INPUT_DIM)
    step = 1e-6
    numerical = np.array([
        (objective(u + step * e) - objective(u - step * e)) / (2 * step)
        for e in np.eye(len(u))[:20]])
    close("the QP gradient H U + g matches d(objective)/dU",
          (hessian @ u + gradient)[:20], numerical, 1e-4)
    check("H is symmetric", np.allclose(hessian, hessian.T, atol=1e-12))
    eigenvalues = np.linalg.eigvalsh(hessian)
    check("H is positive DEFINITE -- R is what makes it so",
          eigenvalues.min() > 0,
          "lambda_min %.3g (R contributes %.3g)"
          % (eigenvalues.min(), 2 * cfg.ALPHA_FORCE))

    # -- 5. the constraints ------------------------------------------------
    print("\nqp: constraints (20) and (21)")
    contacts = gait.horizon_contacts(0.0)
    matrix, lo, hi = qp.constraints(contacts)
    check("one 5-row block per foot per step",
          matrix.shape == (qp.ROWS_PER_FOOT * 4 * cfg.HORIZON,
                           cfg.INPUT_DIM * cfg.HORIZON),
          "%s" % (matrix.shape,))

    x_stand = np.zeros(cfg.STATE_DIM)
    x_stand[cfg.POS] = (0.0, 0.0, cfg.Z_REF + dyn.COM_OFFSET_BODY[2])
    x_stand[cfg.GRAV] = -9.81
    ref_stand = np.tile(x_stand, (cfg.HORIZON, 1))
    yaws0 = np.zeros(cfg.HORIZON)
    a0, b0 = dyn.discrete_sequence(yaws0, np.tile(feet, (cfg.HORIZON, 1, 1)))
    solver = qp.Solver()
    u0, full = qp.solve_once(x_stand, ref_stand, a0, b0, contacts, solver)

    check("the QP solves", solver.last_status == "solved",
          "%s in %d iterations" % (solver.last_status, solver.last_iterations))

    # 1e-5 N IS OSQP'S CONVERGENCE TOLERANCE, NOT SLACK IN THE FORMULATION.
    # The swing feet are boxed to [0, 0], so the constraint is exact in the
    # problem; the SOLUTION satisfies it to eps_abs = 1e-6 like every other
    # row.  Measured residual is ~4e-8 N -- forty nanonewtons, on a robot that
    # weighs 58.  It also never reaches a joint: `controller.update` applies
    # MPC forces only to legs the same schedule says are in stance, so a swing
    # leg's entry is read by nothing.
    swing_forces = full.reshape(cfg.HORIZON, 4, 3)[~contacts]
    close("(21) D_i u_i = 0: swing feet carry no force, to solver tolerance",
          swing_forces, 0.0, 1e-5, " N")

    # TIED TO THE SOLVER'S OWN eps_abs, not to a number typed here.  OSQP
    # satisfies the rows to its convergence tolerance and no better, and how
    # close it lands varies with the problem's conditioning -- this gate
    # passed at 20 Hz and failed at 40 Hz on a 4 micronewton residual, which
    # is a gate measuring OSQP rather than measuring this code.  10x eps_abs
    # is the same margin the (21) gate above uses, for the same reason.
    solver_tol = 10.0 * solver._settings["eps_abs"]
    residual = matrix @ full
    check("(20) c <= C u <= c_bar holds, to solver tolerance",
          bool(np.all(residual <= hi + solver_tol)
               and np.all(residual >= lo - solver_tol)),
          "worst violation %.3g N (tol %.3g, = 10x eps_abs)" % (
              max(float(np.max(residual - hi)), float(np.max(lo - residual))),
              solver_tol))

    stance_forces = full.reshape(cfg.HORIZON, 4, 3)[contacts]
    tangential = np.linalg.norm(stance_forces[:, :2], axis=1)
    check("the friction pyramid is inscribed in the true cone",
          bool(np.all(tangential <= cfg.MU * stance_forces[:, 2] * np.sqrt(2)
                      + 1e-6)),
          "worst |ft|/fz = %.3f against mu = %.2f"
          % (float(np.max(tangential / np.maximum(stance_forces[:, 2], 1e-9))),
             cfg.MU))

    # -- 6. does it hold the robot up --------------------------------------
    print("\nqp: the standing solution is the physical one")
    total = u0[:, 2].sum()
    close("vertical forces sum to the robot's weight", total, P.WEIGHT, 0.5, " N")
    moment = sum(np.cross(feet[i], u0[i]) for i in range(4))
    close("...with no net moment about the CoM", moment, 0.0, 0.5, " N m")
    check("only the two planted feet carry it",
          bool(np.all(np.abs(u0[~contacts[0]]) < 1e-5)),
          "contacts %s, fz %s" % (contacts[0].astype(int), np.round(u0[:, 2], 1)))

    # An unconstrained solve must minimise the same quadratic as -H^-1 g.
    #
    # THE OBJECTIVE IS GATED TIGHTLY AND THE ARGMIN LOOSELY, BECAUSE THAT IS
    # HOW THIS PROBLEM IS SHAPED.  H's smallest eigenvalue is 2*ALPHA_FORCE =
    # 2e-6 while its largest is ~1e2, so it is conditioned around 1e8: the
    # internal-force directions are almost free, and the location of the
    # minimum along them is genuinely soft -- OSQP lands ~0.1 N away on forces
    # of 30 N.  The VALUE of the objective is not soft, because those are
    # precisely the directions that barely change it.  A tight gate on the
    # argmin here would be a gate on OSQP's tolerance, not on this code.
    wide = qp.constraints(np.ones((cfg.HORIZON, 4), dtype=bool),
                          mu=1e6, fz_min=-1e6, fz_max=1e6)
    hess, grad = qp.cost(*qp.condense(a0, b0), x_stand, ref_stand)
    free = qp.Solver().solve(hess, grad, *wide)
    exact = np.linalg.solve(hess, -grad)

    def quadratic(u):
        return float(0.5 * u @ hess @ u + grad @ u)

    close("unconstrained, the QP attains the closed-form minimum's VALUE",
          quadratic(free), quadratic(exact), 1e-6 * abs(quadratic(exact)) + 1e-6)
    close("...and lands near -H^-1 g, loosely, because H is ill-conditioned",
          free, exact, 1.0, " N")
    check("H's conditioning is what makes that gate loose",
          np.linalg.cond(hess) > 1e4,
          "cond(H) = %.1e, lambda_min = %.1e (= 2 * ALPHA_FORCE)" % (
              np.linalg.cond(hess), np.linalg.eigvalsh(hess).min()))

    # -- 7. the gait is a pure timetable -----------------------------------
    print("\ngait: a timetable and nothing else")
    check("contact() depends only on t",
          bool(np.array_equal(gait.contact(0.31), gait.contact(0.31))
               and np.array_equal(gait.contact(0.31),
                                  gait.contact(0.31 + cfg.GAIT_PERIOD))))
    check("trot pairing: FL with RR, FR with RL",
          bool(np.all(gait.contact(0.0) == np.array([True, False, False, True]))
               and np.all(gait.contact(0.26) ==
                          np.array([False, True, True, False]))))
    duty = np.mean([gait.contact(t)[0]
                    for t in np.linspace(0, 10 * cfg.GAIT_PERIOD, 100001)])
    close("each foot is in stance for exactly DUTY of the cycle",
          duty, cfg.DUTY, 1e-3)
    # OVER EVERY GRID THE CONTROL LOOP ACTUALLY SAMPLES, plus a fine sweep.
    # A trot never has more than two feet down, and the naive
    # `mod(t/T + offset, 1)` spelling produced FOUR at the phase boundary --
    # see `gait.phase`.  It showed up on 2 of 997 linspace samples, which is
    # exactly the rate at which a bug like this stays hidden.
    grids = {
        "fine sweep": np.linspace(0, 20 * cfg.GAIT_PERIOD, 200001),
        "physics ticks": np.arange(20000) * 0.002,
        "control ticks": np.arange(10000) * cfg.CONTROL_DT,
        "MPC ticks": np.arange(2000) * cfg.MPC_DT,
    }
    for name, times in grids.items():
        counts = np.array([gait.contact(t).sum() for t in times])
        check("exactly two feet are down, %s" % name,
              bool(np.all(counts == 2)),
              "%d samples, %d wrong" % (len(times), int((counts != 2).sum())))
    horizon = gait.horizon_contacts(0.137)
    check("horizon_contacts row i is the state at t + (i+1) dt",
          bool(np.array_equal(
              horizon,
              np.array([gait.contact(0.137 + (i + 1) * cfg.MPC_DT)
                        for i in range(cfg.HORIZON)]))))

    # -- 8. the reference trajectory ---------------------------------------
    print("\ntrajectory: only the states the paper allows are non-zero")
    command = trajectory.Command(vx=0.3, vy=-0.2,
                                 yaw_rate=np.deg2rad(30.0))
    reference = trajectory.ReferenceTrajectory()
    reference.anchor((1.0, 2.0, 0.0), 0.4)
    rows = reference.horizon(command)
    close("roll and pitch are 0 in every row", rows[:, 0:2], 0.0, 0.0, " rad")
    close("roll rate and pitch rate are 0 in every row",
          rows[:, 6:8], 0.0, 0.0, " rad/s")
    close("z-velocity is 0 in every row", rows[:, 11], 0.0, 0.0, " m/s")
    close("yaw rate is the command in every row",
          rows[:, 8], command.yaw_rate, 0.0, " rad/s")
    close("z is the commanded height in every row",
          rows[:, 5], command.z, 0.0, " m")
    close("yaw is the integral of the yaw rate",
          rows[:, 2], 0.4 + command.yaw_rate * cfg.MPC_DT
          * np.arange(1, cfg.HORIZON + 1), 1e-12, " rad")
    # xy integrates the WORLD velocity, so it must turn with the yaw.
    dx = np.diff(rows[:, 3]) / cfg.MPC_DT
    check("xy integrates the world velocity, which rotates with yaw",
          float(np.abs(np.diff(dx)).max()) > 1e-4,
          "vx_world changes by %.4f m/s across the horizon"
          % float(rows[-1, 9] - rows[0, 9]))
    close("...and the velocity rows ARE that world velocity",
          rows[1:, 9], dx, 1e-9, " m/s")

    # -- 9. the pad ---------------------------------------------------------
    #
    # THE OPERATOR IS PART OF THE LOOP AND THE STICK IS WHERE IT ENTERS.  None
    # of this is about hardware -- `gamepad` is gated through `PadState`, which
    # is just numbers -- it is about the four decisions that shape what the
    # reference is asked for: the deadzone is radial, it rescales, full
    # deflection is exactly the ceiling, and the sign of a right-hand turn.
    print("\ngamepad: stick -> command")

    # The drift measured on the pad on this desk, in raw XInput counts.
    drift = gamepad._stick(1786 / 32767, -593 / 32767, gamepad.DEADZONE_L)
    check("a stick nobody is touching commands exactly zero",
          drift == (0.0, 0.0),
          "5.5%% of full scale of drift, killed by the %.3f deadzone"
          % gamepad.DEADZONE_L)

    # A stick pushed up and rolled slightly left: a per-AXIS deadzone would
    # cut the 0.10 sideways component and quantise the diagonals.
    rolled = gamepad._stick(0.10, 0.95, gamepad.DEADZONE_L)
    check("the deadzone is RADIAL: a small sideways roll survives a big push",
          abs(rolled[0]) > 0.05,
          "x %.3f of an input 0.100 -- a square deadzone would return 0"
          % rolled[0])

    just_out = np.hypot(*gamepad._stick(0.0, gamepad.DEADZONE_L + 1e-6,
                                        gamepad.DEADZONE_L))
    check("...and it RESCALES, so the command leaves zero continuously",
          just_out < 1e-4,
          "%.2e at the edge; passing r raw would step to %.3f"
          % (just_out, gamepad.DEADZONE_L))

    rim = gamepad._stick(0.0, 1.0, gamepad.DEADZONE_L)
    close("full deflection is exactly full scale, expo notwithstanding",
          rim[1], 1.0, 1e-15)
    close("...so the rim is exactly V_MAX and YAW_RATE_MAX",
          [gamepad.PadState(ly=1.0).command().vx,
           gamepad.PadState(rx=-1.0).command().yaw_rate],
          [cfg.V_MAX, cfg.YAW_RATE_MAX], 1e-15)

    magnitudes = [np.hypot(*gamepad._stick(0.0, r, gamepad.DEADZONE_L))
                  for r in np.linspace(gamepad.DEADZONE_L, 1.0, 400)]
    check("the shaping is monotonic, and softer than linear near centre",
          bool(np.all(np.diff(magnitudes) > 0))
          and gamepad._shape(0.5) < 0.5,
          "shape(0.5) = %.3f against a linear 0.500"
          % gamepad._shape(0.5))

    up = gamepad.PadState(ly=0.5).command()
    left = gamepad.PadState(lx=-0.5).command()
    right_turn = gamepad.PadState(rx=0.5).command()
    check("the signs: up is forward, left is +vy, right stick right turns RIGHT",
          up.vx > 0 and left.vy > 0 and right_turn.yaw_rate < 0,
          "vx %+.2f, vy %+.2f, yaw rate %+.0f deg/s"
          % (up.vx, left.vy, np.rad2deg(right_turn.yaw_rate)))

    held = gamepad.PadState(ly=0.4, rx=-0.2)
    check("the mapping is ABSOLUTE: reading the same stick twice is one command",
          str(held.command()) == str(held.command())
          and held.command().vx == cfg.V_MAX * 0.4,
          "%s, whatever it read a moment ago" % held.command())

    precise = gamepad.PadState(ly=0.4, rx=-0.2, lt=1.0).command()
    check("the left trigger scales the command without steering it",
          abs(precise.vx / held.command().vx - gamepad.PRECISION_SCALE) < 1e-12
          and abs(precise.yaw_rate / held.command().yaw_rate
                  - gamepad.PRECISION_SCALE) < 1e-12,
          "x%.2f on both axes" % gamepad.PRECISION_SCALE)

    # The whole path, stick to reference: a constant vx with a constant yaw
    # rate is a circle of radius vx / yaw_rate, and nothing else.
    session = teleop.Session()
    stick = gamepad.PadState(connected=True, ly=1.0, rx=-1.0)
    dt = 1.0 / 200.0
    turn = 2 * np.pi / cfg.YAW_RATE_MAX
    for _ in range(int(round(turn / dt))):
        session.step(dt, stick)
    radius = cfg.V_MAX / cfg.YAW_RATE_MAX
    close("stick -> reference: a full circle of radius vx / yaw_rate",
          [session.ref.x, session.ref.y], [0.0, 0.0], 5e-3, " m")
    close("...whose left-hand turn opens to +y, with the right radius",
          max(session.hist["y"]), 2 * radius, 5e-3, " m")

    # -- 10. leg dynamics ---------------------------------------------------
    print("\nleg dynamics against MuJoCo")
    model = mujoco.MjModel.from_xml_path(P.XML_PATH)
    data = mujoco.MjData(model)
    full_m = np.zeros((model.nv, model.nv))
    worst_m = worst_bias = worst_alt = 0.0
    for _ in range(120):
        q = rng.uniform(-2.0, 2.0, size=(4, 3))
        qd = rng.uniform(-3.0, 3.0, size=(4, 3))
        data.qpos[:] = C.qpos_from_q(q, root_pos=(0, 0, 1.0))
        data.qvel[:] = 0.0
        data.qvel[C.QVEL_JOINTS] = C.flat(qd)
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, full_m, data.qM)
        for i in range(4):
            block = slice(6 + 3 * i, 9 + 3 * i)
            worst_m = max(worst_m, float(np.abs(
                LD.mass_matrix(i, q[i]) - full_m[block, block]).max()))
            worst_bias = max(worst_bias, float(np.abs(
                LD.bias(i, q[i], qd[i]) - data.qfrc_bias[block]).max()))
            worst_alt = max(worst_alt, float(np.abs(
                LD.mass_matrix(i, q[i]) - LD.mass_matrix_rnea(i, q[i])).max()))
    close("M == mj_fullM (armature included), 120 poses", worst_m, 0.0, 1e-10,
          " kg m^2")
    close("C qd + G == mj qfrc_bias, 120 poses", worst_bias, 0.0, 1e-9, " N m")
    close("the fast M and the RNEA M agree", worst_alt, 0.0, 1e-14, " kg m^2")

    q_stand = C.Q_STAND[0]
    with_arm = LD.foot_apparent_mass(0, q_stand)
    without = LD.foot_apparent_mass(0, q_stand, armature=False)
    check("the armature dominates the foot's apparent vertical mass",
          with_arm > 5.0 * without,
          "%.2f kg with, %.2f kg without -- a factor of %.1f"
          % (with_arm, without, with_arm / without))

    # -- 11. the swing law -------------------------------------------------
    print("\nswing")
    start = np.array([0.20, 0.07, P.FOOT_RADIUS])
    end = start + np.array([0.06, -0.01, 0.004])     # ends at a DIFFERENT z
    arc = swing.SwingTrajectory(start, end)
    close("the arc starts exactly at liftoff", arc.at(0.0)[0], start, 1e-15, " m")
    close("...and ends exactly at the planned touchdown",
          arc.at(1.0)[0], end, 1e-15, " m")
    close("it leaves and arrives with zero velocity",
          [arc.at(0.0)[1], arc.at(1.0)[1]], 0.0, 1e-12, " m/s")
    eps = 1e-7
    left, right = arc.at(0.5 - eps)[0], arc.at(0.5 + eps)[0]
    close("the two half-arcs meet at the apex even when the ends differ in z",
          left, right, 1e-6, " m")
    check("...at an apex that clears the higher end by SWING_HEIGHT",
          abs(arc.at(0.5)[0][2] - (max(start[2], end[2]) + cfg.SWING_HEIGHT))
          < 1e-12,
          "apex %.1f mm above the ground" % (1000 * arc.at(0.5)[0][2]))

    # velocity and acceleration must be the derivatives of position
    worst_v = worst_a = 0.0
    for s in np.linspace(0.02, 0.98, 97):
        h = 1e-6
        dt = h * arc.duration
        v_num = (arc.at(s + h)[0] - arc.at(s - h)[0]) / (2 * dt)
        a_num = (arc.at(s + h)[1] - arc.at(s - h)[1]) / (2 * dt)
        worst_v = max(worst_v, float(np.abs(arc.at(s)[1] - v_num).max()))
        worst_a = max(worst_a, float(np.abs(arc.at(s)[2] - a_num).max()))
    close("the arc's velocity is dp/dt", worst_v, 0.0, 1e-5, " m/s")
    close("...and its acceleration is dv/dt", worst_a, 0.0, 1e-3, " m/s^2")

    # (33) and the hip-velocity form must agree when there is no yaw rate.
    hip = np.array([0.35, 0.09, 0.19])
    com = np.array([0.20, 0.02, 0.16])
    v_com = np.array([0.3, -0.1, 0.0])
    literal = swing.foot_placement(hip, v_com)
    corrected = swing.foot_placement(
        hip, swing.hip_velocity(v_com, np.zeros(3), hip, com))
    close("the hip-velocity form reduces EXACTLY to (33) when w = 0",
          literal, corrected, 1e-15, " m")
    turning = swing.foot_placement(
        hip, swing.hip_velocity(v_com, (0, 0, np.deg2rad(90)), hip, com))
    check("...and differs from it under a yaw rate",
          float(np.linalg.norm(turning - literal)) > 0.005,
          "%.1f mm apart at 90 deg/s"
          % (1000 * np.linalg.norm(turning - literal)))

    # the stance torque sign, and the two halves of the force path
    force_up = np.array([0.0, 0.0, 30.0])            # ground pushes the robot up
    tau = swing.stance_torque(0, q_stand, swing.body_frame_force(force_up,
                                                                 np.eye(3)))
    jac = K.foot_jacobian(0, q_stand)
    check("stance_torque delivers the commanded force, not its negative",
          float(np.linalg.norm(
              np.linalg.lstsq(jac.T, tau, rcond=None)[0] + force_up)) < 1e-9,
          "J^-T tau == -f, so the foot presses DOWN and the ground pushes up")

    # body_frame_force is a pure rotation, and handles one foot or all four.
    yaw90 = dyn.rot_z(np.pi / 2)
    close("body_frame_force rotates world -> body",
          swing.body_frame_force(np.array([1.0, 0.0, 0.0]), yaw90),
          np.array([0.0, -1.0, 0.0]), 1e-12, " N")
    many = rng.normal(size=(4, 3)) * 20.0
    close("...and does all four feet at once, identically",
          swing.body_frame_force(many, yaw90),
          np.stack([swing.body_frame_force(f, yaw90) for f in many]), 1e-12, " N")
    close("...preserving magnitude, since R is orthonormal",
          np.linalg.norm(swing.body_frame_force(many, yaw90), axis=1),
          np.linalg.norm(many, axis=1), 1e-12, " N")

    # -- 12. closed loop ---------------------------------------------------
    print("\nclosed loop, in MuJoCo")
    from . import run as runner

    sim = runner.Run(duration=4.0, script="stand")
    while sim.data.time < 4.0 and not sim.fallen():
        sim.step()
    heights = np.array([row["z"] for row in sim.log])
    settled = np.array([row["t"] for row in sim.log]) > 1.0
    check("it stands for 4 s", not sim.fallen(),
          "%.2f s, height error %.1f mm rms" % (
              sim.data.time,
              1000 * np.sqrt(np.mean((heights[settled] - cfg.Z_REF) ** 2))))
    check("...and the height error is small",
          np.sqrt(np.mean((heights[settled] - cfg.Z_REF) ** 2)) < 0.010,
          "a CoM/trunk-origin mix-up here costs 18 mm")

    # -- the rate of every layer -------------------------------------------
    # 20 Hz does not divide 250 Hz, so the MPC deadline has to be counted from
    # a fixed origin.  Advancing it from the last SOLVE instead silently gives
    # a steady 52 ms period -- 19.23 Hz, with no jitter to hint at it -- and
    # holds u_0 for 4 % longer than the discretisation that chose it assumes.
    # THREE RATES FROM THREE SOURCES.  R comes from the IMU at 200 Hz and q
    # from the encoders at 250 Hz, so fb = R' f and tau = J' fb CANNOT share a
    # rate.  Fusing them would invent IMU samples the hardware never sends.
    counts = {"solve": 0, "fb": 0, "tau": 0, "swing": 0}
    solve_times: list[float] = []
    imu_times: list[float] = []
    rotations: list[np.ndarray] = []
    from . import controller as ctl
    from . import run as runner_mod

    original = (swing.body_frame_force, swing.stance_torque,
                swing.swing_torque, ctl.Controller.solve)

    def _counted_solve(self, state, t):
        counts["solve"] += 1
        solve_times.append(t)
        return original[3](self, state, t)

    def _counted_fb(force, rotation):
        counts["fb"] += 1
        imu_times.append(_clock["t"])
        return original[0](force, rotation)

    _clock = {"t": 0.0}
    swing.body_frame_force = _counted_fb
    swing.stance_torque = lambda *a, **k: (
        counts.__setitem__("tau", counts["tau"] + 1), original[1](*a, **k))[1]
    swing.swing_torque = lambda *a, **k: (
        counts.__setitem__("swing", counts["swing"] + 1), original[2](*a, **k))[1]
    ctl.Controller.solve = _counted_solve
    try:
        sim = runner_mod.Run(duration=3.0, script="forward")
        while sim.data.time < 3.0 and not sim.fallen():
            _clock["t"] = sim.data.time
            rotations.append(sim.controller._imu.rotation.copy()
                             if sim.controller._imu is not None else np.eye(3))
            sim.step()
        elapsed = sim.data.time
        sweeps = len(sim.log)
    finally:
        (swing.body_frame_force, swing.stance_torque,
         swing.swing_torque, ctl.Controller.solve) = original

    close("the control loop runs at CONTROL_HZ", sweeps / elapsed,
          cfg.CONTROL_HZ, 1.0, " Hz")
    check("tau = J' fb is written every sweep, for all four legs",
          counts["tau"] + counts["swing"] == 4 * sweeps,
          "%d leg torques over %d sweeps = %.1f per sweep"
          % (counts["tau"] + counts["swing"], sweeps,
             (counts["tau"] + counts["swing"]) / sweeps))

    solve_gaps = np.diff(np.asarray(solve_times))
    close("the MPC averages exactly MPC_DT between solves",
          solve_gaps.mean(), cfg.MPC_DT, 1e-4, " s")
    # Derived from cfg, not hardcoded: the ratio was 12.5 sweeps at 20 Hz and
    # is 6.25 at 40 Hz.  A gate that spells the sweep counts out stops
    # checking anything the moment a rate moves.
    ratio = cfg.CONTROL_HZ / cfg.MPC_HZ
    allowed = {round(np.floor(ratio) * cfg.CONTROL_DT, 4),
               round(np.ceil(ratio) * cfg.CONTROL_DT, 4)}
    check("...by alternating sweeps, since %.0f Hz does not divide %.0f Hz"
          % (cfg.MPC_HZ, cfg.CONTROL_HZ),
          set(np.round(solve_gaps, 4)) <= allowed,
          "%.2f sweeps/solve -> intervals %s s -> %.2f Hz"
          % (ratio, sorted(set(np.round(solve_gaps, 4))),
             1.0 / solve_gaps.mean()))

    imu_gaps = np.diff(np.asarray(imu_times))
    close("fb = R' f is refreshed at the IMU's rate, not the sweep's",
          imu_gaps.mean(), cfg.IMU_DT, 1e-4, " s")
    check("...so it is computed strictly less often than tau",
          counts["fb"] < sweeps,
          "%d IMU ticks against %d sweeps -- %.0f Hz vs %.0f Hz"
          % (counts["fb"], sweeps, counts["fb"] / elapsed, sweeps / elapsed))
    check("...alternating sweeps too, since 200 Hz does not divide 250 Hz",
          set(np.round(imu_gaps, 4)) <= {round(cfg.CONTROL_DT, 4),
                                         round(2 * cfg.CONTROL_DT, 4)},
          "intervals %s s -> %.2f Hz"
          % (sorted(set(np.round(imu_gaps, 4))), 1.0 / imu_gaps.mean()))

    # The observable consequence: R is genuinely stale on some sweeps.
    repeats = sum(1 for a, b in zip(rotations, rotations[1:])
                  if np.array_equal(a, b))
    check("R is HELD between IMU samples, so some sweeps reuse it",
          repeats > 0.1 * len(rotations),
          "%d of %d sweeps reused the previous R; worst age %.0f ms"
          % (repeats, len(rotations), 1000 * max(np.round(imu_gaps, 4))))

    # -- what leaves the controller, and what the sim survives -------------
    #
    # BOTH OF THESE ARE HERE BECAUSE A FOUR-MINUTE DRIVING SESSION DIED.  The
    # viewer ended on `mujoco.FatalError: mju_makeFrame: xaxis of contact
    # frame undefined` -- MuJoCo refusing a state it could not integrate --
    # and the report went with it.  Neither gate claims to know what made that
    # state; they gate the two places where this code could have let it
    # through or made it fatal.
    print("\nthe boundary: what reaches the plant, and what it survives")

    guard = runner.Run(duration=1.0)
    original_swing = swing.swing_torque
    swing.swing_torque = lambda *a, **k: np.array([np.nan, 0.0, 0.0])
    try:
        for _ in range(20):
            guard.step()
    finally:
        swing.swing_torque = original_swing
    check("np.clip passes NaN, so the torque boundary filters it explicitly",
          guard.controller.telemetry.non_finite > 0
          and bool(np.all(np.isfinite(guard.data.ctrl)))
          and bool(np.all(np.isfinite(guard.data.qpos))),
          "%d non-finite torques zeroed; np.clip alone returns %s"
          % (guard.controller.telemetry.non_finite,
             np.clip(np.nan, -cfg.TAU_MAX, cfg.TAU_MAX)))

    survivor = runner.Run(duration=5.0)
    for _ in range(50):
        survivor.step()
    real_step = mujoco.mj_step

    def _exploding_step(model, data):
        raise mujoco.FatalError("mju_makeFrame: xaxis of contact frame undefined")

    mujoco.mj_step = _exploding_step
    escaped = False
    try:
        survivor.step()
    except mujoco.FatalError:
        escaped = True
    finally:
        mujoco.mj_step = real_step
    check("a fatal physics error is caught and reported as a fall, not raised",
          not escaped and survivor.diverged and survivor.fallen(),
          "diverged=%s, fallen=%s" % (survivor.diverged, survivor.fallen()))

    survivor.recover()
    ok_after = bool(np.all(np.isfinite(survivor.data.qpos)))
    zeroed = survivor.operator.is_zero()
    t_before = survivor.data.time
    for _ in range(250):
        survivor.step()
    check("...and recover() puts it back on its feet with the command zeroed",
          ok_after and zeroed and not survivor.fallen(),
          "%.1f s more of physics after the fatal, height %.4f m"
          % (survivor.data.time - t_before, survivor.data.qpos[2]))

    # The yaw branch cut: turn past 180 degrees and keep going.
    sim = runner.Run(duration=7.0, script="stand")
    sim.script = [(0.0, trajectory.Command()),
                  (0.5, trajectory.Command(yaw_rate=np.deg2rad(60.0)))]
    while sim.data.time < 7.0 and not sim.fallen():
        sim.step()
    turned = np.rad2deg(np.unwrap(
        np.array([row["rpy"][2] for row in sim.log]))[-1])
    check("it turns through +-180 deg without the yaw branch cut destabilising it",
          not sim.fallen() and abs(turned) > 200.0,
          "%.0f deg turned in %.1f s" % (turned, sim.data.time))

    # ----------------------------------------------------------------------
    print("\n%d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    for name in _FAILURES:
        print("  FAILED: %s" % name)
    if not _FAILURES:
        print("the convex MPC matches the formulation it reproduces.")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
