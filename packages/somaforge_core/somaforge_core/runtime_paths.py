"""Locations shared by Lab launchers, native Newton workers, and installers."""
from __future__ import annotations

import json
import os
from pathlib import Path

ISAACLAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"


def project_root() -> Path:
    return Path(os.environ.get("SOMAFORGE_ROOT", Path(__file__).resolve().parents[3])).resolve()


def isaaclab_path() -> Path:
    # Resolve the source snapshot from the same catalog/lock fingerprint as
    # installation, even when called from an unactivated shell.
    import importlib.util
    source = project_root() / "scripts/manage_environment.py"
    spec = importlib.util.spec_from_file_location("somaforge_environment_manager", source)
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    plan = manager.make_plan(project_root(), "main", catalog=os.environ.get("SOMAFORGE_ENV_CATALOG"))
    return Path(plan["lab"]["snapshot"])


def newton_environment() -> Path:
    if os.environ.get("SOMAFORGE_ENV_PROFILE") == "main":
        value = os.environ.get("SOMAFORGE_ENV_PREFIX", "")
        if not value or not Path(value).is_absolute():
            raise RuntimeError("Managed main profile requires an absolute SOMAFORGE_ENV_PREFIX")
        prefix = Path(value).resolve()
        managed = project_root() / "runtime/environments/conda"
        if prefix == managed or not prefix.is_relative_to(managed):
            raise RuntimeError(f"Managed Newton environment must be inside {managed}: {prefix}")
        return prefix
    return Path(default_ik_conda_environment())


def newton_python() -> Path:
    executable = newton_environment() / "bin/python"
    if not executable.is_file():
        setup = "python3 scripts/manage_environment.py create --profile main"
        raise RuntimeError(f"Missing SomaForge main runtime: {executable}. Run {setup}.")
    return executable


def default_ik_conda_environment() -> str:
    """Shared Newton workers use the validated main runtime by default."""
    root = project_root()
    catalog_path = Path(os.environ.get("SOMAFORGE_ENV_CATALOG", root / "configs/environments/profiles.json"))
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema") != "somaforge_environment_profiles_v1":
        raise RuntimeError("Unsupported SomaForge environment profile catalog")
    relative = Path(catalog["profiles"]["main"]["prefix"])
    prefix = (root / relative).resolve()
    managed = (root / "runtime/environments/conda").resolve()
    if relative.is_absolute() or prefix == managed or not prefix.is_relative_to(managed):
        raise RuntimeError(f"Managed IK environment must be inside {managed}: {prefix}")
    return str(prefix)


def newton_subprocess_environment() -> dict[str, str]:
    """Select the same release for native workers and Kit subprocesses."""
    root, lab = project_root(), isaaclab_path()
    env = dict(os.environ)
    paths = [root, root / "src/holosoma"]
    paths.extend(root / "packages" / package for package in (
        "somaforge_core", "contact_solver", "generator", "motion_edit", "supermimic",
    ))
    paths.extend(sorted(path for path in (lab / "source").iterdir() if path.is_dir()))
    env.update(
        SOMAFORGE_ROOT=str(root), ISAACLAB_PATH=str(lab),
        SOMAFORGE_NEWTON_ENV=str(newton_environment()),
        SOMAFORGE_NEWTON_PROFILE="main", PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(map(str, paths)),
    )
    for name in ("PYTHONHOME", "HOLOSOMA_ISAACLAB3_NEWTON_HEADLESS_EXPERIENCE",
                 "HOLOSOMA_ISAACLAB3_NEWTON_KIT_EXPERIENCE"):
        env.pop(name, None)
    prefix = newton_environment()
    env.update(SOMAFORGE_ENV_PROFILE="main", SOMAFORGE_ENV_PREFIX=str(prefix),
               SOMAFORGE_ISAACLAB_PATH=str(lab), CONDA_PREFIX=str(prefix), CONDA_DEFAULT_ENV=str(prefix))
    env["PATH"] = os.pathsep.join([str(prefix / "bin"), "/usr/local/bin", "/usr/bin", "/bin"])
    for name in ("VIRTUAL_ENV", "PYTHONUSERBASE"):
        env.pop(name, None)
    return env
