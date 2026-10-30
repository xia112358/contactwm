"""Explicit runtime resources for the packaged SuperMimic workflow.

Source assets are package-relative. Data, checkpoints, logs and caches remain in
an explicitly registered external runtime; they are never silently redirected
into the source checkout.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    from somaforge_core.robot_assets import somaforge_root
    return somaforge_root().resolve()


def runtime_root() -> Path:
    override = os.environ.get("SOMAFORGE_SUPERMIMIC_RUNTIME")
    if override:
        path = Path(override).expanduser()
    else:
        root = workspace_root()
        config = Path(os.environ.get("SOMAFORGE_RESOURCES", str(root / "configs/workflow_resources.json"))).expanduser()
        if not config.is_absolute():
            config = root / config
        try:
            resources = json.loads(config.read_text())
            path = Path(resources["supermimic"]["runtime_root"]).expanduser()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Configure supermimic.runtime_root in configs/workflow_resources.json "
                               "or set SOMAFORGE_SUPERMIMIC_RUNTIME to the external runtime directory") from exc
        if not path.is_absolute():
            path = root / path
    if not path.is_absolute():
        raise ValueError("SOMAFORGE_SUPERMIMIC_RUNTIME must be an absolute path")
    return path.resolve()


def runtime_path(*parts: str) -> Path:
    return runtime_root().joinpath(*parts)


def package_path(*parts: str) -> Path:
    if parts and parts[0].split("/")[0] == "experiments":
        return Path(__file__).resolve().parent.joinpath("resources", *parts)
    return PACKAGE_ROOT.joinpath(*parts)


def module_root(name: str) -> Path:
    """Locate installed package source without importing its simulation modules."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Required SomaForge package is not available: {name}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()
