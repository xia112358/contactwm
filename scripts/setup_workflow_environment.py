#!/usr/bin/env python3
"""Install the default ContactWM workflow from pinned Isaac Lab sources.

The default creates a fresh Python 3.12 environment. --base-python is an explicit
compatibility option for an existing simulation environment, never a prerequisite.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_workflow import CACHE_DIRS, ROOT, _path, default_resources
from bootstrap_runtime import bootstrap

LAB_COMMIT = "46d6cc3d7b115906fad12efd372fbf69f4e3d3ed"
LAB_URL = "https://github.com/isaac-sim/IsaacLab.git"
PACKAGES = ("somaforge_core", "contact_solver", "generator", "motion_edit", "supermimic")


def make_plan(args, *, root=ROOT, environ=None):
    env = os.environ if environ is None else environ
    registry = _path(args.resources or env.get("SOMAFORGE_RESOURCES") or default_resources(root), root)
    resources = json.loads(registry.read_text())
    section = resources["supermimic"]
    runtime = _path(env.get("SOMAFORGE_SUPERMIMIC_RUNTIME") or section["runtime_root"], root)
    native = _path(env.get("SOMAFORGE_PPO_RUNTIME") or resources["ppo"]["runtime_root"], root)
    for path in (runtime, native):
        if path.is_relative_to(root.resolve()):
            raise ValueError(f"Runtime must be outside the checkout: {path}")
    configured_python = env.get("SOMAFORGE_SUPERMIMIC_PYTHON") or section["python"]
    prefix = _path(args.prefix, root) if args.prefix else (root / configured_python).expanduser().absolute().parent.parent
    lab = _path(args.isaaclab or env.get("SOMAFORGE_SUPERMIMIC_ISAACLAB") or section["isaaclab_root"], root)
    base = args.base_python
    if base:
        # Do not resolve symlinks: a venv interpreter's invoked path defines it.
        base = base.expanduser().absolute()
        if not base.is_file():
            raise ValueError(f"Base interpreter is missing: {base}")
        if prefix.resolve() == base.parent.parent.resolve():
            raise ValueError("Environment prefix must differ from the base interpreter prefix")
    python = prefix / "bin/python"
    commands = []
    if not lab.exists():
        commands.extend([
            ["git", "clone", "--no-checkout", LAB_URL, str(lab)],
            ["git", "-C", str(lab), "checkout", "--detach", LAB_COMMIT],
        ])
    if base:
        commands.append(["uv", "venv", "--system-site-packages", "--python", str(base), str(prefix)])
    else:
        commands.append(["uv", "sync", "--frozen", "--project", str(lab), "--python", "3.12",
                         "--extra", "all", "--extra", "importers", "--extra", "test"])
    commands.append(["uv", "pip", "install", "--python", str(python), "--no-deps", "--link-mode", "copy",
                     "-r", str(root / "configs/environments/workflow-extras.lock.txt")])
    editable = [item for name in PACKAGES for item in ("-e", str(root / "packages" / name))]
    commands.append(["uv", "pip", "install", "--python", str(python), "--no-deps", "--link-mode", "copy",
                     *editable, "-e", str(root / "src/holosoma")])
    return dict(resources=str(registry), prefix=str(prefix), python=str(python), runtime=str(runtime),
                native_runtime=str(native), isaaclab=str(lab), isaaclab_commit=LAB_COMMIT,
                base_python=str(base) if base else None, commands=commands)


def install(plan, *, root=ROOT):
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("Install uv first (python3 -m pip install uv==0.13.1), then rerun this command")
    prefix, lab = Path(plan["prefix"]), Path(plan["isaaclab"])
    state_path = root / "runtime/environments/workflow-install.json"
    identity = {key: plan[key] for key in ("prefix", "isaaclab", "isaaclab_commit", "base_python")}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError(f"Existing installation uses different settings; use a new checkout: {state_path}")
    elif prefix.exists() or prefix.is_symlink():
        raise ValueError(f"Refusing an existing unowned environment: {prefix}; choose a new --prefix")
    if lab.exists():
        actual = subprocess.check_output(["git", "-C", str(lab), "rev-parse", "HEAD"], text=True).strip()
        if actual != LAB_COMMIT:
            raise ValueError(f"Isaac Lab must be at {LAB_COMMIT}, found {actual}: {lab}")
        changes = subprocess.check_output(["git", "-C", str(lab), "diff", "HEAD", "--"], text=True)
        if changes:
            raise ValueError(f"Isaac Lab has modified tracked files; supply a clean pinned checkout: {lab}")
    bootstrap(root, Path(plan["native_runtime"]))
    runtime = Path(plan["runtime"])
    for name in ("data", "logs", "models", "results", "checkpoints", "outputs", *CACHE_DIRS.values()):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    alias = root / "runtime/supermimic"
    if (alias.exists() or alias.is_symlink()) and (not alias.is_symlink() or alias.resolve() != runtime):
        raise ValueError(f"Existing runtime alias differs from configured runtime: {alias}")
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(runtime, target_is_directory=True)
    env = dict(os.environ)
    # Installation caches belong to the external runtime; do not inherit another venv.
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)
    env.update({name: str(runtime / relative) for name, relative in CACHE_DIRS.items()})
    env.update(UV_PROJECT_ENVIRONMENT=str(prefix), UV_CACHE_DIR=str(runtime / "cache/uv-workflow"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    def save(status):
        state_path.write_text(json.dumps(dict(identity=identity, status=status, plan=plan), indent=2) + "\n")
    save("installing")
    try:
        for command in plan["commands"]:
            command = [uv if item == "uv" and index == 0 else item for index, item in enumerate(command)]
            # A retry resumes the owned overlay instead of recreating it.
            if plan["base_python"] and command[1] == "venv" and (prefix / "pyvenv.cfg").is_file():
                continue
            if command[:2] == ["git", "clone"]:
                lab.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=root, env=env, check=True)
        save("installed")
    except BaseException:
        save("interrupted")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, help="Explicit resource registry")
    parser.add_argument("--isaaclab", type=Path, help="Clean Isaac Lab checkout at the pinned revision")
    parser.add_argument("--base-python", type=Path, help="Optional existing compatible simulation interpreter")
    parser.add_argument("--prefix", type=Path, help="Custom target environment; configure its interpreter in your resource file")
    parser.add_argument("--dry-run", action="store_true", help="Print installation commands without changing files")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            install(plan)
            print(f"Workflow interpreter: {plan['python']}")
            print("Verify: python3 scripts/check_workflow.py --check-cli")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
