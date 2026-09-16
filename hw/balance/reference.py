"""The height reference: a C2 S-curve in h, and its conversion to a CoM command.

    t -> (h_cmd, hdot_cmd, hddot_cmd) -> (p_c,z,d, pdot_c,z,d)

WHY IT IS AUTHORED IN h AND NOT IN THE CoM
    `h` is floor to trunk BOTTOM -- the height an operator can put a ruler on,
    and the one every number in doc/dog6_stand_control is quoted in.  T is
    tuned against what the robot visibly does, so the profile is shaped in the
    quantity a person watches.  The PD consumes a CoM height, so the CoM
    conversion is the generator's LAST step, not its first.

WHY QUINTIC AND NOT `sim.stand.smoothstep`
    The raised cosine s -> (1 - cos pi s)/2 is C1: its velocity is zero at
    both ends but its SLOPE is not, so the acceleration the reference implies
    STEPS at s = 0 and s = 1.  With no feedforward that step does not enter
    b_d directly -- it arrives as a corner in the velocity error, and so in
    the commanded wrench, at each end of the ramp.  That is at the handover,
    and again exactly where the 2026-09-15 runs failed: late, near the top.

    The quintic has zero velocity AND zero acceleration at both ends:

        sigma(s)    = 10 s^3 - 15 s^4 + 6 s^5
        sigma'(s)   = 30 s^2 (1 - s)^2
        sigma''(s)  = 60 s (1 - s)(1 - 2 s)

    IT IS NOT A GENTLER RAMP.  Peak velocity is 1.875 dh/T against the cosine's
    1.571, and peak acceleration 5.77 dh/T^2 against 4.935.  The S-curve buys
    continuity at the ENDS and pays for it in the middle.  At dh = 115 mm and
    T = 3.0 s that is 72 mm/s and 0.074 m/s^2.

ALL THREE ARE CLOSED FORM, AND THAT IS THE POINT
    The commanded rate must be the analytic derivative of the commanded
    position -- never a finite difference taken inside the loop, or k_d,z is
    fed the derivative of the reference's own quantisation.  Nothing here is
    differentiated at run time.

    `hddot` costs nothing and is returned, but the balance PD DOES NOT USE IT.
    It is there for the feedforward that is not written yet.

THREE THINGS THE GENERATOR OWNS, AND THEY ARE NOT THE LAW'S BUSINESS
    h0            latched from the MEASURED height at the instant torque arms,
                  not from CROUCH_HEIGHT.  Starting a ramp from a height the
                  robot is not at is a step input, and it is a step input into
                  the one term (k_d,z) that differentiates it.
    re-planning   if the stand is paused, retimed or aborted, re-plan from the
                  current (h, hdot, hddot) rather than restarting s.  A
                  quintic through arbitrary boundary conditions is the same
                  five coefficients -- `Quintic.between` is that, and
                  `Quintic.ramp` is the special case with everything zero.
    the rate limit  T is the only knob, and it sets both peaks above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/reference.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from . import config as cfg          # noqa: E402

__all__ = ["Quintic", "HeightCommand", "ComCommand", "com_command",
           "PEAK_VELOCITY_FACTOR", "PEAK_ACCELERATION_FACTOR"]

#: sigma' and sigma'' at their extrema, for sizing arguments.  30/16 and
#: 10/sqrt(3), exactly.
PEAK_VELOCITY_FACTOR = 1.875
PEAK_ACCELERATION_FACTOR = 10.0 / np.sqrt(3.0)          # 5.7735


class HeightCommand(NamedTuple):
    """The reference in h -- floor to trunk BOTTOM.  All three, always."""
    h: float            # m
    hdot: float         # m/s
    hddot: float        # m/s^2   returned, not used by the PD.  See above.


class ComCommand(NamedTuple):
    """The same reference as the PD consumes it: a CoM height, WORLD frame."""
    p_cz: float         # m
    p_cz_dot: float     # m/s


@dataclass(frozen=True)
class Quintic:
    """A fifth-order polynomial in t, clamped and held outside [0, T].

    HELD, NOT EXTRAPOLATED.  Past T the trajectory returns its end point with
    zero velocity and zero acceleration -- which is the hold, and is exactly
    what the ramp's own boundary conditions say.  A quintic evaluated past its
    horizon runs away as t^5.
    """

    c: np.ndarray       # (6,) coefficients, c[0] + c[1] t + ... + c[5] t^5
    T: float

    @staticmethod
    def between(h0: float, v0: float, a0: float,
                h1: float, v1: float, a1: float, T: float) -> "Quintic":
        """The unique quintic meeting six boundary conditions over [0, T].

        This is the re-planner.  Pausing, retiming or aborting a stand all
        reduce to calling it with the CURRENT (h, hdot, hddot) as the start --
        which keeps the reference C2 across the re-plan, where restarting s
        would put a velocity step into the middle of a lift.
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError("a ramp needs a positive duration, got %r" % T)
        # The first three coefficients are the start conditions outright; the
        # last three solve a fixed 3x3 that depends only on T.
        c0, c1, c2 = h0, v0, 0.5 * a0
        dh = h1 - (c0 + c1 * T + c2 * T * T)
        dv = v1 - (c1 + 2.0 * c2 * T)
        da = a1 - 2.0 * c2
        matrix = np.array([[T ** 3, T ** 4, T ** 5],
                           [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                           [6 * T, 12 * T ** 2, 20 * T ** 3]])
        c3, c4, c5 = np.linalg.solve(matrix, np.array([dh, dv, da]))
        return Quintic(np.array([c0, c1, c2, c3, c4, c5]), T)

    @staticmethod
    def ramp(h0: float, h1: float, T: float) -> "Quintic":
        """Rest to rest: the sigma of the module docstring, scaled to (h0, h1).

        Written through `between` rather than by tabulating (0, 0, 0, 10, -15,
        6) so there is ONE quintic in this file and the special case cannot
        drift from the general one.  `selftest` checks the coefficients come
        out at exactly that tabulation.
        """
        return Quintic.between(h0, 0.0, 0.0, h1, 0.0, 0.0, T)

    def at(self, t: float) -> HeightCommand:
        """(h, hdot, hddot) at `t`, clamped to [0, T] and HELD outside it."""
        t = min(max(float(t), 0.0), self.T)
        c = self.c
        powers = np.array([1.0, t, t * t, t ** 3, t ** 4, t ** 5])
        h = float(c @ powers)
        hdot = float(c[1] + 2 * c[2] * t + 3 * c[3] * t * t
                     + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4)
        hddot = float(2 * c[2] + 6 * c[3] * t + 12 * c[4] * t * t
                      + 20 * c[5] * t ** 3)
        return HeightCommand(h, hdot, hddot)

    @property
    def peak_velocity(self) -> float:
        """|hdot| at its largest over the ramp.  Sampled, because `between`
        admits profiles whose extremum is not at the midpoint."""
        ts = np.linspace(0.0, self.T, 401)
        return float(max(abs(self.at(t).hdot) for t in ts))

    @property
    def peak_acceleration(self) -> float:
        ts = np.linspace(0.0, self.T, 401)
        return float(max(abs(self.at(t).hddot) for t in ts))


def com_command(command: HeightCommand, R) -> ComCommand:
    """The h reference as the PD consumes it: a CoM height in the WORLD frame.

        p_c,z,d    = h_cmd + TRUNK_BOTTOM_OFFSET + (R c^b)_z
        pdot_c,z,d = hdot_cmd                    + (omega x R c^b)_z

    THE SECOND TERM OF THE RATE IS ZERO HERE AND THAT IS NOT AN OMISSION.
    `config.COM_BODY` is a CONSTANT (see its docstring), so the only way the
    commanded CoM height can move relative to the commanded trunk height is by
    the trunk ROTATING -- and the reference commands a level trunk, so the
    reference's own omega is zero.  The measured side carries the matching
    term in `state.read`, where omega is whatever the robot is doing.

    WITH A CONSTANT c^b THE CONVERSION COEFFICIENT IS EXACTLY 1.  Were c^b
    evaluated at the pose it would be 0.677 over this ramp -- the legs unfold
    downward, so a 115 mm trunk lift raises the CoM by only 78 mm -- and
    dropping it would make the commanded CoM rate 48 % too fast at every
    instant.  Here both the reference and the measurement use the same
    constant, so the whole offset cancels out of the error and the PD's z
    channel sees precisely the trunk-height error.
    """
    offset_z = float((np.asarray(R, dtype=float).reshape(3, 3)
                      @ cfg.COM_BODY)[2])
    return ComCommand(
        p_cz=command.h + cfg.TRUNK_BOTTOM_OFFSET + offset_z,
        p_cz_dot=command.hdot,
    )


def describe() -> str:
    ramp = Quintic.ramp(cfg.H_CROUCH, cfg.H_LIFT, cfg.T_RISE)
    delta = cfg.H_LIFT - cfg.H_CROUCH
    return "\n".join([
        "DOG6 stand height reference: quintic S-curve, C2 at both ends",
        "  h %.1f -> %.1f mm over %.1f s   (h is FLOOR TO TRUNK BOTTOM)"
        % (1e3 * cfg.H_CROUCH, 1e3 * cfg.H_LIFT, cfg.T_RISE),
        "  peak hdot   %.1f mm/s   = %.3f * dh/T" % (
            1e3 * ramp.peak_velocity, ramp.peak_velocity * cfg.T_RISE / delta),
        "  peak hddot  %.3f m/s^2  = %.3f * dh/T^2" % (
            ramp.peak_acceleration,
            ramp.peak_acceleration * cfg.T_RISE ** 2 / delta),
        "  against the raised cosine's 1.571 and 4.935: the S-curve buys",
        "  continuity at the ENDS, not a gentler middle.",
        "  coefficients  %s" % np.array2string(ramp.c, precision=4),
    ])


if __name__ == "__main__":
    print(describe())
