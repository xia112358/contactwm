"""Playback must not import the retired editing/IK dependency stack."""
from __future__ import annotations

import ast
from pathlib import Path


def test_motion_edit_has_no_jax_or_retired_solver_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "motion_edit"
    retired = {"pyroki_fullbody_ik", "pyroki_taskspace", "pyroki_trajectory_optimizer",
               "robot_mirror_ik", "whole_trajectory_projector", "torch_newton_compare"}
    forbidden = {"jax", "jaxlib", "jaxlie", "jaxls", "pyroki"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args
                  and isinstance(node.args[0], ast.Constant)):
                modules.append(node.args[0].value)
        assert not [module for module in modules
                    if module.split(".")[0] in forbidden
                    or module.split(".")[-1] in retired], str(path)
