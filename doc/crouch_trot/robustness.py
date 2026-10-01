"""The committed trots under the non-idealities a real DOG6 has and the model
does not: IMU delay, CoM offset, joint friction, foot friction, torque
constant, unmodelled mass.  One perturbation at a time, then combinations.

    python robustness.py [wide|wide_W|fold] ...
Writes data/robust_<trot>.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hwsim as H  # noqa: E402
from hwsim import trot_metrics as metrics  # noqa: E402
import fixes as FX  # noqa: E402

PERTURB = {
    "ideal": {},
    "imu_lat_15ms": {"imu_latency": 15e-3},
    "imu_lat_30ms": {"imu_latency": 30e-3},
    "com_y_+5mm": {"com_err_mm": (0.0, 5.0)},
    "com_y_+10mm": {"com_err_mm": (0.0, 10.0)},
    "com_x_+10mm": {"com_err_mm": (10.0, 0.0)},
    "joint_fric_0.1": {"frictionloss": 0.1},
    "joint_fric_0.3": {"frictionloss": 0.3},
    "foot_mu_0.5": {"friction": 0.5},
    "kt_0.85": {"kt_scale": 0.85},
    "mass_+0.4kg": {"trunk_mass_add": 0.4},
    "compute_1ms": {"t_compute": 1.0e-3},
    "combo_mild": {"imu_latency": 15e-3, "com_err_mm": (3.0, 5.0),
                   "frictionloss": 0.1, "friction": 0.7, "kt_scale": 0.9},
}

TROTS = {
    "wide": (H.entry_trot, dict(hold_s=2.0, trot_s=12.0, park=False), {}),
    "wide_W": (H.entry_trot, dict(hold_s=2.0, trot_s=12.0, park=False,
                                  step_first=True), {}),
    "fold": (H.entry_fold_trot, dict(hold_s=2.0, trot_s=12.0, park=False),
             {"prime_gate": True}),
    # candidate fix: hw.fold_stand's roll gains on hw.trot
    "wide_r290": (lambda: H.entry_trot().replace(roll_gains=(290.0, 23.0)),
                  dict(hold_s=2.0, trot_s=12.0, park=False), {}),
    "wide_W_r290": (lambda: H.entry_trot().replace(roll_gains=(290.0, 23.0)),
                    dict(hold_s=2.0, trot_s=12.0, park=False, step_first=True), {}),
    # fold stance, hw.trot's Cartesian swing at 0.8 s (roll gains stay 290/23)
    "fold_cart08": (lambda: H.entry_fold_trot().replace(gait_period=0.8, swing="cartesian"),
                    dict(hold_s=2.0, trot_s=12.0, park=False), {"prime_gate": True}),
    "wide_r290_place": (lambda: H.entry_trot().replace(roll_gains=(290.0, 23.0)),
                        dict(hold_s=2.0, trot_s=12.0, park=False), {}, {"place": 0.15}),
    "fold_cart08_place": (lambda: H.entry_fold_trot().replace(gait_period=0.8, swing="cartesian"),
                          dict(hold_s=2.0, trot_s=12.0, park=False), {"prime_gate": True},
                          {"place": 0.15}),
    "fold_hybrid08": (lambda: H.entry_fold_trot().replace(gait_period=0.8, swing="cartesian"),
                      dict(hold_s=2.0, trot_s=12.0, park=False), {"prime_gate": True},
                      {"hybrid": True}),
}


def main(argv):
    names = [a for a in argv if a in TROTS] or list(TROTS)
    only = [a[3:] for a in argv if a.startswith("-p=")]
    for trot_name in names:
        factory, opkw, base_hw = TROTS[trot_name][:3]
        variant = TROTS[trot_name][3] if len(TROTS[trot_name]) > 3 else {}
        table = {}
        for pname, pkw in PERTURB.items():
            if only and pname not in only:
                continue
            t0 = time.time()
            entry = factory()
            res = FX.run_variant(entry, opkw, {**base_hw, **pkw}, **variant)
            m = metrics(res)
            table[pname] = m
            print("%-7s %-15s fell=%-5s trot %5.2f s  max tilt %6.2f  tilt>15 after %s s  stop=%s  (%.0f s)"
                  % (trot_name, pname, m["fell"], m.get("trot_s", 0),
                     m.get("max_tilt_trot", np.nan), m.get("t_tilt15_after_T"),
                     str(m["stop"])[:60], time.time() - t0))
            sys.stdout.flush()
        json.dump(table, open(os.path.join(HERE, "data", "robust_%s.json" % trot_name), "w"),
                  indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:])
