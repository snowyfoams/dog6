"""The layer rules, checked mechanically instead of by review.

    python -m pytest hw/state_estimator/tests

    estimator/   imports numpy and the standard library only -- a WHITELIST,
                 so a new stdlib module is a decision made here, on purpose.
                 No file, clock, print or eval; no module-level state; no
                 mutable default; every signature takes arrays, floats and
                 LKFParams and returns arrays or a dataclass.
    adapters/    rpy_zyx_to_R is the controller's rotation; run_once is
                 update() with four feet down.
    config/      lkf.yaml has exactly LKFParams' keys.
"""
from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hw.state_estimator.adapters import (ImuSource, LegSource, rpy_zyx_to_R,
                                         run_once)
from hw.state_estimator.estimator import (LinearKFPosVelEstimator, LKFOutput,
                                          LKFParams)

PKG = Path(__file__).resolve().parents[1]          # hw/state_estimator
REPO = PKG.parents[1]
MODEL_FILES = sorted((PKG / "estimator").glob("*.py"))

ALLOWED_IMPORTS = {"__future__", "dataclasses", "numpy"}
FORBIDDEN_CALLS = {"print", "open", "input", "breakpoint", "exec", "eval",
                   "compile", "__import__"}
FILE_IO_ATTRS = {"load", "save", "savez", "savez_compressed", "loadtxt",
                 "savetxt", "genfromtxt", "fromfile", "tofile", "memmap",
                 "fromregex"}
ARG_TYPES = {"np.ndarray", "float", "LKFParams", "LKFParams | None"}
RETURN_TYPES = {"np.ndarray", "tuple[np.ndarray, np.ndarray]", "LKFOutput", "None"}


def _model_trees() -> list[tuple[str, ast.Module]]:
    assert len(MODEL_FILES) == 5, [f.name for f in MODEL_FILES]
    return [(f.name, ast.parse(f.read_text(encoding="utf-8"))) for f in MODEL_FILES]


def _is_immutable_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.Tuple):
        return all(_is_immutable_literal(e) for e in node.elts)
    return False


# -- estimator/ ---------------------------------------------------------------
def test_model_layer_imports_only_numpy_and_the_standard_library():
    for name, tree in _model_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in ALLOWED_IMPORTS, \
                        f"{name}:{node.lineno}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    assert node.module.split(".")[0] in ALLOWED_IMPORTS, \
                        f"{name}:{node.lineno}: from {node.module} import ..."
                else:
                    assert node.level == 1, \
                        f"{name}:{node.lineno}: relative import leaves estimator/"


def test_model_layer_has_no_io_clock_print_or_module_state():
    for name, tree in _model_trees():
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.Global, ast.Nonlocal)), \
                f"{name}:{node.lineno}: global/nonlocal"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in FORBIDDEN_CALLS, \
                        f"{name}:{node.lineno}: {func.id}()"
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in FILE_IO_ATTRS, \
                        f"{name}:{node.lineno}: .{func.attr}()"

        # At module level: a docstring, imports, defs and __all__.  Nothing
        # that could hold state between two calls.
        for i, stmt in enumerate(tree.body):
            if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            if (isinstance(stmt, ast.Assign)
                    and [ast.unparse(t) for t in stmt.targets] == ["__all__"]
                    and _is_immutable_literal(stmt.value)):
                continue
            pytest.fail(f"{name}:{stmt.lineno}: module-level `{ast.unparse(stmt)[:60]}`")


def test_model_layer_has_no_mutable_default():
    for name, tree in _model_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Lambda)):
                defaults = node.args.defaults + [d for d in node.args.kw_defaults if d]
                for d in defaults:
                    assert _is_immutable_literal(d), \
                        f"{name}:{d.lineno}: default `{ast.unparse(d)}`"
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
                    if value is not None:
                        assert _is_immutable_literal(value), \
                            f"{name}:{stmt.lineno}: class attribute `{ast.unparse(stmt)}`"


def test_model_layer_signatures_carry_no_hardware_type():
    for name, tree in _model_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            where = f"{name}:{node.lineno} {node.name}"
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == "self":
                    continue
                assert arg.annotation is not None, f"{where}({arg.arg}) is unannotated"
                assert ast.unparse(arg.annotation) in ARG_TYPES, \
                    f"{where}({arg.arg}: {ast.unparse(arg.annotation)})"
            assert node.args.vararg is None and node.args.kwarg is None, where
            assert node.returns is not None, f"{where} has no return annotation"
            assert ast.unparse(node.returns) in RETURN_TYPES, \
                f"{where} -> {ast.unparse(node.returns)}"


def test_importing_the_model_layer_imports_nothing_else_from_hw_or_sim():
    """The source checks above see estimator/; this sees the package chain too.

    `import hw.state_estimator.estimator` also runs hw/__init__.py and
    hw/state_estimator/__init__.py.  If either grows an import of the IMU,
    the bus or the kinematics, the model layer drags it in with it.
    """
    code = ("import sys\n"
            "before = set(sys.modules)\n"
            "import hw.state_estimator.estimator\n"
            "print(' '.join(sorted(m for m in set(sys.modules) - before\n"
            "                      if m.split('.')[0] in ('hw', 'sim'))))\n")
    loaded = subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.split()
    model = "hw.state_estimator.estimator"
    assert set(loaded) == {"hw", "hw.state_estimator", model,
                           f"{model}.params", f"{model}.matrices",
                           f"{model}.measurement", f"{model}.lkf"}


# -- adapters/ ----------------------------------------------------------------
def test_rpy_zyx_to_R_is_the_rotation_the_controller_uses():
    """A second copy of `rot_zyx`, so it is gated against the first.

    And the route `robot_io` documents for rpy -- the trunk's triple, read back
    out of the controller's own R -- has to rebuild that R exactly.
    """
    from hw import imu as IMU
    from sim import coordinates as C

    rng = np.random.default_rng(3)
    for _ in range(200):
        rpy = rng.uniform(-1.0, 1.0, size=3) * [np.pi, 0.49 * np.pi, np.pi]
        R = rpy_zyx_to_R(rpy)
        np.testing.assert_allclose(R, C.rot_zyx(*rpy), rtol=0, atol=1e-14)
        np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

        R_ctrl = IMU.trunk_rotation(*rpy)
        rpy_trunk = np.array(C.zyx_from_rot(R_ctrl))
        np.testing.assert_allclose(rpy_zyx_to_R(rpy_trunk), R_ctrl, rtol=0, atol=1e-12)


def test_rpy_signs_are_the_ones_a_tilt_test_reproduces():
    left, nose = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert (rpy_zyx_to_R(np.array([0.1, 0.0, 0.0])) @ left)[2] > 0.0    # roll: left up
    assert (rpy_zyx_to_R(np.array([0.0, 0.1, 0.0])) @ nose)[2] < 0.0    # pitch: nose down
    assert (rpy_zyx_to_R(np.array([0.0, 0.0, 0.1])) @ nose)[1] > 0.0    # yaw: nose left


class _StillImu(ImuSource):
    def __init__(self, rpy, omega_b, acc_b):
        self.sample = (rpy, omega_b, acc_b)

    def read(self):
        return self.sample


class _StillLegs(LegSource):
    def __init__(self, r, rd):
        self.sample = (r, rd)

    def read(self):
        return self.sample


def test_run_once_is_update_with_four_feet_down():
    rpy = np.array([0.05, -0.08, 0.3])
    omega_b = np.array([0.02, -0.01, 0.05])
    R_wb = rpy_zyx_to_R(rpy)
    acc_b = R_wb.T @ np.array([0.0, 0.0, 9.81])
    r = np.array([[+0.182, +0.065, -0.177535], [+0.182, -0.065, -0.177535],
                  [-0.182, +0.065, -0.177535], [-0.182, -0.065, -0.177535]])
    rd = np.full((4, 3), 0.01)
    imu, legs = _StillImu(rpy, omega_b, acc_b), _StillLegs(r, rd)

    via_adapter, direct = LinearKFPosVelEstimator(), LinearKFPosVelEstimator()
    via_adapter.reset(R_wb, r)
    direct.reset(R_wb, r)
    for _ in range(20):
        a = run_once(imu, legs, via_adapter, 0.002)
        b = direct.update(R_wb, omega_b, acc_b, r, rd, np.full(4, 0.5), 0.002)
        for field in dataclasses.fields(LKFOutput):
            np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))


def test_sources_cannot_be_constructed_without_a_driver():
    with pytest.raises(TypeError):
        ImuSource()
    with pytest.raises(TypeError):
        LegSource()


# -- config/ ------------------------------------------------------------------
def test_lkf_yaml_has_exactly_the_LKFParams_keys():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((PKG / "config" / "lkf.yaml").read_text(encoding="utf-8"))
    assert set(cfg) == {f.name for f in dataclasses.fields(LKFParams)}
    for key, value in cfg.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), \
            f"lkf.yaml: {key} = {value!r} is not a number"
    LKFParams(**cfg)
