#!/usr/bin/env python3
"""Select an existing SomaForge workflow entrypoint without changing its algorithm.

Arguments after ``--`` belong to the selected backend, including AppLauncher
arguments. Importing this module or using --dry-run never starts a simulator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = (
    "packages/somaforge_core", "packages/contact_solver", "packages/generator",
    "packages/motion_edit", "packages/supermimic", "src/holosoma",
)
CACHE_DIRS = {
    "TMPDIR": "tmp", "WARP_CACHE_PATH": "cache/warp",
    "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TRITON_CACHE_DIR": "cache/triton",
    "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
    "CUDA_CACHE_PATH": "cache/cuda", "XDG_CACHE_HOME": "cache/xdg",
    "MPLCONFIGDIR": "cache/matplotlib", "WANDB_DIR": "wandb",
    "WANDB_CACHE_DIR": "cache/wandb",
}


@dataclass(frozen=True)
class Launch:
    stage: str
    backend: str
    profile: str
    argv: tuple[str, ...]
    cwd: Path
    runtime_root: Path
    resources: Path
    environment: dict[str, str]

    def as_dict(self) -> dict:
        return {
            "schema": "somaforge_workflow_launch_v1",
            "stage": self.stage, "backend": self.backend, "profile": self.profile,
            "argv": list(self.argv), "cwd": str(self.cwd),
            "runtime_root": str(self.runtime_root), "resources": str(self.resources),
            "environment": self.environment,
            "interpreter_available": Path(self.argv[0]).is_file() and os.access(self.argv[0], os.X_OK),
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("tracking", "collect", "initialize", "contact", "predictor"))
    result.add_argument("--backend", choices=("supermimic",),
                        help="tracking algorithm (supermimic); valid only for tracking")
    result.add_argument("--python", dest="python", help="explicit interpreter for the selected runtime profile")
    result.add_argument("--resources", type=Path,
                        help="resource registry (default: SOMAFORGE_RESOURCES or configs/workflow_resources.local.json, then configs/workflow_resources.json)")
    result.add_argument("--dry-run", action="store_true", help="print the launch plan without creating files or running it")
    return result


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    split = values.index("--") if "--" in values else len(values)
    own = parser().parse_args(values[:split])
    return own, values[split + 1:] if split < len(values) else []


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (root / path if not path.is_absolute() else path).resolve()


def _python(value: str, root: Path, env: Mapping[str, str]) -> str:
    if os.sep not in value and not value.startswith("."):
        found = shutil.which(value, path=env.get("PATH", os.defpath))
        if found:
            return str(Path(found).absolute())
    # Do not resolve an interpreter symlink: venv selection uses its invoked path.
    candidate = Path(value).expanduser()
    return str(candidate.absolute() if candidate.is_absolute() else (root / candidate).absolute())


def default_resources(root: Path) -> Path:
    local = root / "configs/workflow_resources.local.json"
    return local if local.is_file() else root / "configs/workflow_resources.json"


def build_launch(options: argparse.Namespace, backend_args: list[str], *,
                 root: Path = ROOT, environ: Mapping[str, str] | None = None) -> Launch:
    inherited = os.environ if environ is None else environ
    root = root.resolve()
    if options.stage != "tracking" and options.backend is not None:
        raise ValueError("--backend applies only to tracking; PPO downstream integration is deferred")
    backend = options.backend or ("supermimic" if options.stage != "predictor" else "predictor")
    profile = "supermimic" if backend == "supermimic" else "ppo"
    registry = _path(options.resources or inherited.get("SOMAFORGE_RESOURCES")
                     or default_resources(root), root)
    resources = json.loads(registry.read_text())
    section = resources[profile]
    interpreter = options.python or inherited.get(f"SOMAFORGE_{profile.upper()}_PYTHON") or section["python"]
    executable = _python(str(interpreter), root, inherited)
    runtime_value = inherited.get(f"SOMAFORGE_{profile.upper()}_RUNTIME") or section["runtime_root"]
    runtime = _path(runtime_value, root)
    if runtime.is_relative_to(root):
        raise ValueError(f"Runtime must resolve outside the source checkout: {runtime}; configure an external data directory")

    module = {"tracking": "supermimic.train", "collect": "supermimic.contact_collect",
              "initialize": "supermimic.contact_initialize", "contact": "supermimic.contact_train",
              "predictor": "generator.train_full1000_position"}[options.stage]
    entrypoint = ("-m", module)

    python_paths = [str(root / relative) for relative in LOCAL_PACKAGES] + [str(root)]
    environment = {
        "SOMAFORGE_ROOT": str(root), "SOMAFORGE_RESOURCES": str(registry),
        f"SOMAFORGE_{profile.upper()}_RUNTIME": str(runtime),
        "SOMAFORGE_ASSET_MANIFEST": str(root / "configs/assets_manifest.json"),
        "SOMAFORGE_TRAINING_MANIFEST": str(root / "configs/training_pipeline_manifest.json"),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    }
    lab_value = inherited.get(f"SOMAFORGE_{profile.upper()}_ISAACLAB") or section.get("isaaclab_root")
    if lab_value:
        lab = _path(lab_value, root)
        environment.update(ISAACLAB_PATH=str(lab), SOMAFORGE_ISAACLAB_PATH=str(lab))
        python_paths.extend(str(path) for path in sorted((lab / "source").glob("*"))
                            if path.is_dir() and ((path / "pyproject.toml").is_file() or (path / path.name).is_dir()))
    # Project-local contact fixes precede shared packages without mutating them.
    if section.get("newton_pythonpath"):
        overlay = _path(section["newton_pythonpath"], root)
        if not (overlay / "newton/__init__.py").is_file():
            raise ValueError(f"Configured Newton runtime is missing: {overlay}")
        python_paths.insert(0, str(overlay))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.update({name: str(runtime / relative) for name, relative in CACHE_DIRS.items()})
    prefix = Path(executable).parent.parent
    old_prefixes = [Path(inherited[name]).resolve() for name in ("CONDA_PREFIX", "VIRTUAL_ENV") if inherited.get(name)]
    def scrub(value: str) -> str:
        return os.pathsep.join(item for item in value.split(os.pathsep) if item and
            not any(Path(item).resolve().is_relative_to(old) and old != prefix.resolve() for old in old_prefixes))
    environment["PATH"] = str(Path(executable).parent) + os.pathsep + scrub(inherited.get("PATH", os.defpath))
    if "LD_LIBRARY_PATH" in inherited:
        environment["LD_LIBRARY_PATH"] = scrub(inherited["LD_LIBRARY_PATH"])
    environment["SOMAFORGE_NEWTON_ENV"] = str(prefix)
    if profile == "ppo" and prefix.resolve().is_relative_to((root / "runtime/environments/conda").resolve()):
        environment.update(SOMAFORGE_ENV_PROFILE="main", SOMAFORGE_ENV_PREFIX=str(prefix))
    if (prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(prefix)
    if profile == "supermimic":
        # Preserve the native launcher's CPU compilation defaults, not a new training recipe.
        environment["OMP_NUM_THREADS"] = inherited.get("OMP_NUM_THREADS", "16")
        environment["TORCHINDUCTOR_COMPILE_THREADS"] = inherited.get("TORCHINDUCTOR_COMPILE_THREADS", "8")
        visible = inherited.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            environment["CUDA_VISIBLE_DEVICES"] = visible
        if visible and "MUJOCO_EGL_DEVICE_ID" not in inherited and all(item.isdigit() for item in visible.split(",")):
            environment["MUJOCO_EGL_DEVICE_ID"] = visible.split(",")[0]
    return Launch(options.stage, backend, profile, (executable, *entrypoint, *backend_args),
                  root, runtime, registry, environment)


def execute(launch: Launch, *, environ: Mapping[str, str] | None = None) -> None:
    if not (Path(launch.argv[0]).is_file() and os.access(launch.argv[0], os.X_OK)):
        raise ValueError(f"Python interpreter is not executable: {launch.argv[0]}")
    env = dict(os.environ if environ is None else environ)
    # Old activation must not inject another Python/IsaacLab source tree.
    for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONPATH", "VIRTUAL_ENV",
                 "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "ISAACLAB_PATH", "SOMAFORGE_ISAACLAB_PATH",
                 "SOMAFORGE_ENV_PROFILE", "SOMAFORGE_ENV_PREFIX", "SOMAFORGE_NEWTON_ENV", "SOMAFORGE_NEWTON_PROFILE"):
        env.pop(name, None)
    env.update(launch.environment)
    # Native help must work even when external data/caches are read-only.
    if not any(arg in ("--help", "-h") for arg in launch.argv[1:]):
        for name in CACHE_DIRS:
            Path(env[name]).mkdir(parents=True, exist_ok=True)
    os.chdir(launch.cwd)
    os.execve(launch.argv[0], list(launch.argv), env)


def main(argv: list[str] | None = None) -> int:
    options, forwarded = parse_args(argv)
    try:
        launch = build_launch(options, forwarded)
        if options.dry_run:
            print(json.dumps(launch.as_dict(), indent=2))
        else:
            execute(launch)
    except (OSError, ValueError) as error:
        parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
