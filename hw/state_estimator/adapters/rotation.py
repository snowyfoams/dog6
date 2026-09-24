"""rpy -> R in the ZYX convention `sim.coordinates.rot_zyx` fixes.  numpy only.

Free of any `sim` or `hw` import so the model layer could use it too.  The
price is a second copy of `rot_zyx`, and a copy that agrees by inspection is
exactly what this project does not trust: `tests/test_layers.py` gates the two
against each other over random triples.
"""
from __future__ import annotations

import numpy as np

__all__ = ("rpy_zyx_to_R",)


def rpy_zyx_to_R(rpy: np.ndarray) -> np.ndarray:
    """R_{world <- trunk} = Rz(yaw) Ry(pitch) Rx(roll), (3,) -> (3, 3).

    `rpy` is (roll, pitch, yaw) in RADIANS -- the TRUNK's ZYX triple, on FLU
    axes, so the right-hand rule fixes the signs:

        roll  > 0   right side down
        pitch > 0   nose DOWN
        yaw   > 0   nose left

    The same matrix as `sim.coordinates.rot_zyx(roll, pitch, yaw)`.
    """
    rpy = np.asarray(rpy, dtype=float)
    assert rpy.shape == (3,), f"rpy must be (3,), got {rpy.shape}"
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
