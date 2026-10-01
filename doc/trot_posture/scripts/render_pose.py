"""Screenshots of a candidate posture: the crouch and the stand, side and iso.

    python doc/trot_posture/scripts/render_pose.py NAME FAMILY XF XR Y H [--crouch-h MM] [--views side iso front]

Static poses from evalpose.build (no simulation).  Overlays on the floor: a
red disc at the whole-body CoM's ground projection, yellow lines for the two
support diagonals (the trot's two-foot support lines) -- the disc sits on
both lines only when the stance is centred.  One mujoco.Renderer per process
(a second one on Windows renders black), so several postures go through one
call.  Output: doc/trot_posture/fig/pose_<NAME>_<crouch|stand>_<view>.png/.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import evalpose as EP                     # noqa: E402  (sets sys.path for the repo)
import mujoco                             # noqa: E402
from PIL import Image                     # noqa: E402
from sim import coordinates as C, params as P, kinematics as SK   # noqa: E402
from hw.balance import config as BCFG     # noqa: E402

FIG = os.path.join(HERE, "..", "fig")
os.makedirs(FIG, exist_ok=True)
VIEWS = {
    "side": dict(azimuth=90.0, elevation=-8.0),
    "front": dict(azimuth=180.0, elevation=-10.0),
    "iso": dict(azimuth=135.0, elevation=-24.0),
}
_R = {}


def _renderer(width=960, height=720):
    if "r" not in _R:
        m = mujoco.MjModel.from_xml_path(P.XML_PATH)
        m.vis.global_.offwidth = max(width, 1600)
        m.vis.global_.offheight = max(height, 1200)
        m.vis.quality.shadowsize = 4096
        m.vis.headlight.ambient[:] = 0.38
        m.vis.headlight.diffuse[:] = 0.55
        m.vis.headlight.specular[:] = 0.05
        grid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "grid")
        m.mat_reflectance[grid] = 0.0
        _R["m"], _R["d"] = m, mujoco.MjData(m)
        _R["r"] = mujoco.Renderer(m, height=height, width=width)
        _R["trunk"] = m.body("trunk").id
        _R["feet"] = [m.site("foot_" + leg).id for leg in C.LEGS]
    return _R["m"], _R["d"], _R["r"]


def _segment(scene, a, b, radius=0.0035, rgba=(1.0, 0.85, 0.1, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def _disc(scene, pos, radius=0.012, rgba=(0.9, 0.1, 0.1, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([radius, 0.0015, 0.0]), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def shot(q, z_origin, view, path, distance=0.72, lookat_z=0.07):
    m, d, r = _renderer()
    d.qpos[:] = C.qpos_from_q(q, root_pos=(0.0, 0.0, float(z_origin)))
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, lookat_z]
    cam.distance = distance
    cam.azimuth = VIEWS[view]["azimuth"]
    cam.elevation = VIEWS[view]["elevation"]
    r.update_scene(d, camera=cam)
    com = d.subtree_com[_R["trunk"]]
    _disc(r.scene, (com[0], com[1], 0.0016))
    feet = np.array([d.site_xpos[s] for s in _R["feet"]])
    for i, j in ((0, 3), (1, 2)):
        _segment(r.scene, [feet[i, 0], feet[i, 1], 0.0012], [feet[j, 0], feet[j, 1], 0.0012])
    img = r.render()
    Image.fromarray(img).save(path)
    Image.fromarray(img).convert("RGB").save(path[:-4] + ".jpg", quality=88, optimize=True)
    return path


def render_spec(spec: EP.Spec, views=("side", "iso")) -> list:
    pose, info = EP.build(spec)
    out = []
    q_stand, _, _ = EP.solve(EP.sites_for(spec.xf, spec.xr, spec.y, spec.h), EP.seed_for(spec.family))
    poses = [("stand", q_stand, 1e-3 * spec.h + BCFG.TRUNK_BOTTOM_OFFSET)]
    if pose is not None:
        poses.insert(0, ("crouch", pose.q, pose.z_origin))
    for tag, q, z in poses:
        for view in views:
            path = os.path.join(FIG, "pose_%s_%s_%s.png" % (spec.name, tag, view))
            out.append(shot(q, z, view, path))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("family", choices=EP.FAMILIES)
    ap.add_argument("xf", type=float); ap.add_argument("xr", type=float)
    ap.add_argument("y", type=float); ap.add_argument("h", type=float)
    ap.add_argument("--crouch-h", type=float, default=None)
    ap.add_argument("--views", nargs="+", default=["side", "iso"])
    a = ap.parse_args(argv)
    for p in render_spec(EP.Spec(a.name, a.family, a.xf, a.xr, a.y, a.h, a.crouch_h), a.views):
        print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
