"""The gait schedule.  A pure function of the clock -- nothing is observed.

    contact(t) -> (4,) bool

THIS IS THE WHOLE MODULE'S CONTRACT AND IT IS DELIBERATELY NARROW.  There is
no contact sensor, no early-touchdown detection, no late-liftoff extension, no
schedule adaptation, no phase variable driven by anything but time.  A foot is
in stance because the table says it is, and if the foot is not actually on the
ground then the MPC is planning against a contact that does not exist and will
find out through the tracking error.

That is a real limitation and it is the requested one.  It also makes the MPC
side interpretable: the contact schedule over the whole horizon is known
exactly and in advance, which is the assumption the convex formulation rests
on -- the QP's constraint structure at step i depends on which feet are down
at step i, and if that were being estimated the problem would change shape
every solve.

    t = 0 is the start of FL's stance.

WHY THE HORIZON QUERY IS ITS OWN FUNCTION
    `horizon_contacts(t)` is not a loop over `contact(t + i*dt)` for the
    convenience of the caller -- it is the thing the QP actually needs, and
    having it in one place means the schedule the constraints are built from
    and the schedule the swing legs follow cannot drift apart.  They are the
    same table read twice.
"""
from __future__ import annotations

import numpy as np

from .. import coordinates as C
from . import config as cfg


def phase(t, offset=None) -> np.ndarray:
    """Each leg's position in its own cycle, in [0, 1).

    0 is the instant that leg touches down; `cfg.DUTY` is the instant it lifts.

    THE OFFSET IS APPLIED IN SECONDS, BEFORE THE MODULO, AND THAT IS NOT
    COSMETIC.  The obvious spelling -- `mod(t / T + offset, 1)` -- forms a sum
    close to 1.0, where doubles are spaced 2^-53 and the operands are spaced
    2^-54.  A phase of 0.49999999999999994 plus an offset of 0.5 is therefore
    ROUNDED UP to exactly 1.0, and comes back out of the modulo as 0.0.

    The consequence is not a rounding artifact, it is a wrong contact state: a
    foot a hair before liftoff and its diagonal partner a hair before
    touchdown both read "in stance", and `contact()` returns all four feet
    down for a gait that never has more than two.  The MPC would then plan
    forces for a foot in the air, and the controller would deliver them.

    Adding the offset in TIME keeps both operands at the period's scale, where
    the modulo is exact, and the failure does not arise.
    """
    offset = cfg.PHASE_OFFSET if offset is None else np.asarray(offset, float)
    shifted = np.asarray(t, float)[..., None] + offset * cfg.GAIT_PERIOD
    return np.mod(shifted, cfg.GAIT_PERIOD) / cfg.GAIT_PERIOD


def contact(t) -> np.ndarray:
    """(4,) bool -- which feet the schedule says are on the ground at time t."""
    return phase(t) < cfg.DUTY


def swing_phase(t) -> np.ndarray:
    """(4,) progress through the CURRENT swing, in [0, 1).

    Zero for a foot in stance, so it is only meaningful where `contact` is
    False.  A swing that has just begun reads ~0; one about to touch down
    reads ~1.
    """
    ph = phase(t)
    prog = (ph - cfg.DUTY) / (1.0 - cfg.DUTY)
    return np.where(ph < cfg.DUTY, 0.0, prog)


def stance_phase(t) -> np.ndarray:
    """(4,) progress through the current stance, in [0, 1).  Zero while swinging."""
    ph = phase(t)
    return np.where(ph < cfg.DUTY, ph / cfg.DUTY, 0.0)


def time_to_touchdown(t) -> np.ndarray:
    """(4,) seconds until each foot next touches down.  Zero if already down.

    What the foot-placement law needs: it is choosing where a foot will land,
    and the reference velocity it should integrate over is the time between
    now and that landing.
    """
    ph = phase(t)
    return np.where(ph < cfg.DUTY, 0.0, (1.0 - ph) * cfg.GAIT_PERIOD)


def time_to_liftoff(t) -> np.ndarray:
    """(4,) seconds until each foot next lifts off.  Zero if already swinging."""
    ph = phase(t)
    return np.where(ph < cfg.DUTY, (cfg.DUTY - ph) * cfg.GAIT_PERIOD, 0.0)


def horizon_contacts(t, horizon: int | None = None, dt: float | None = None):
    """(horizon, 4) bool -- the contact schedule the QP builds its constraints from.

    Row i is the contact state at ``t + (i + 1) * dt``, which is the state at
    the END of the interval whose force u_i is being solved for.  That offset
    is the one easy place to be wrong by one step: u_i acts over
    [t + i*dt, t + (i+1)*dt], and a foot that lifts inside that interval
    cannot be asked for force across it.  Taking the LATER of the two ends is
    the conservative reading -- a foot is only allowed force if it is still
    down when the interval closes.
    """
    horizon = cfg.HORIZON if horizon is None else int(horizon)
    dt = cfg.MPC_DT if dt is None else float(dt)
    times = float(t) + dt * np.arange(1, horizon + 1)
    return contact(times)


def describe(t: float = 0.0) -> str:
    rows = ["DOG6 trot schedule, T %.2f s, duty %.2f" % (cfg.GAIT_PERIOD, cfg.DUTY),
            "  legs            %s" % ", ".join(C.LEGS),
            "  phase offset    %s" % cfg.PHASE_OFFSET,
            "  pairs           FL+RR, then FR+RL",
            "",
            "  t/T    " + "  ".join("%-4s" % leg for leg in C.LEGS)]
    for frac in np.linspace(0.0, 1.0, 11)[:-1]:
        c = contact(t + frac * cfg.GAIT_PERIOD)
        rows.append("  %.2f   " % frac
                    + "  ".join("%-4s" % ("down" if x else "  up") for x in c))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
