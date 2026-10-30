#!/usr/bin/env python3
"""Prepare a checkout's local runtime and restore its pinned canonical G1 USD.

Only Python's standard library is needed. Existing paths and assets are checked,
never replaced; no simulator, package installation or network access is involved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Iterator


USD_SUFFIXES = {".usd", ".usda", ".usdc"}
RUNTIME_DIRECTORIES = (
    "motions", "models", "terrains", "manifests", "generated", "cache", "archives",
    "holosoma/data", "holosoma/logs", "holosoma/tmp", "motion_edit/data",
)
LOGICAL_LINKS = {
    "data": "runtime/current/holosoma/data",
    "logs": "runtime/current/holosoma/logs",
    "tmp": "runtime/current/holosoma/tmp",
    "packages/motion_edit/data": "../../runtime/current/motion_edit/data",
}


@contextmanager
def _checkout_environment(root: Path) -> Iterator[None]:
    previous = os.environ.get("SOMAFORGE_ROOT")
    os.environ["SOMAFORGE_ROOT"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SOMAFORGE_ROOT", None)
        else:
            os.environ["SOMAFORGE_ROOT"] = previous


def _asset_contract(root: Path):
    # Loading this stdlib-only file directly avoids importing core.__init__,
    # which intentionally imports NumPy-dependent public utilities.
    path = root / "packages/somaforge_core/somaforge_core/robot_assets.py"
    spec = importlib.util.spec_from_file_location("_somaforge_bootstrap_robot_assets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load canonical robot asset contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _check_directory(path: Path) -> None:
    if _present(path) and not path.is_dir():
        raise FileExistsError(f"Expected a directory; refusing to replace existing path: {path}")
    if not _present(path):
        parent = path.parent
        while not _present(parent) and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise FileExistsError(f"Directory parent conflicts with an existing path: {parent}")


def _check_link(path: Path, target: str) -> None:
    if not _present(path):
        return
    if not path.is_symlink() or path.resolve() != (path.parent / target).resolve():
        raise FileExistsError(f"Refusing to replace {path}; expected a symlink to {target}")


def _bundle_files(directory: Path) -> dict[Path, Path]:
    return {path.relative_to(directory): path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in USD_SUFFIXES}


def _check_sidecar(path: Path, expected: dict[str, str]) -> None:
    if not _present(path):
        return
    if not path.is_file():
        raise FileExistsError(f"Robot asset sidecar is not a file: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for key in ("asset_id", "urdf_sha256", "asset_bundle_sha256", "usd_bundle_sha256"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Existing robot asset sidecar has incompatible {key}: {path}")


def bootstrap(root: Path, data_root: Path | None = None) -> dict:
    root = root.expanduser().resolve()
    contract = _asset_contract(root)
    with _checkout_environment(root):
        source_metadata = contract.canonical_g1_source_metadata()
        source = root / "assets/g1_spherehand_usd"
        entry_name = contract.G1_SPHEREHAND_USD_RELATIVE.name
        expected_hash = contract.G1_SPHEREHAND_USD_BUNDLE_SHA256
        source_hash = contract._usd_bundle_sha256(source / entry_name)
        if source_hash != expected_hash:
            raise ValueError("Pinned G1 USD bundle fingerprint mismatch; fetch complete repository assets "
                             f"(git lfs pull if applicable). Expected {expected_hash}, got {source_hash}")
        destination = root / contract.G1_SPHEREHAND_USD_RELATIVE.parent
        files = _bundle_files(source)
        _check_directory(destination)
        existing = _bundle_files(destination)
        extras = set(existing) - set(files)
        if extras:
            raise ValueError(f"Canonical G1 USD directory has unexpected USD files: {sorted(map(str, extras))}")
        for relative, original in files.items():
            output = destination / relative
            _check_directory(output.parent)
            if _present(output) and (not output.is_file() or contract._sha256(output) != contract._sha256(original)):
                raise ValueError(f"Existing canonical G1 USD differs from the pinned bundle: {output}")
        metadata = {**source_metadata, "usd_path": str(destination / entry_name),
                    "usd_bundle_sha256": source_hash}
        sidecar = destination / "somaforge_robot_asset.json"
        _check_sidecar(sidecar, metadata)

        # Preflight every existing entry before creating anything, including
        # command-window output paths and the optional external runtime link.
        current = root / "runtime/current"
        if data_root is not None:
            resolved_data = data_root.expanduser().resolve()
            if resolved_data == current or current in resolved_data.parents:
                raise ValueError("--data-root must be outside runtime/current to avoid a recursive symlink")
            target = os.path.relpath(resolved_data, current.parent)
            _check_link(current, target)
        else:
            if current.is_symlink() and not current.is_dir():
                raise FileNotFoundError(f"Existing runtime/current symlink has no usable directory: {current}")
            _check_directory(current)
            resolved_data = current.resolve()
            target = None
        _check_directory(current.parent)
        _check_directory(resolved_data)
        for name in RUNTIME_DIRECTORIES:
            _check_directory(resolved_data / name)
        for name, link_target in LOGICAL_LINKS.items():
            _check_directory((root / name).parent)
            _check_link(root / name, link_target)

        restored = []
        for relative, original in sorted(files.items()):
            output = destination / relative
            if not _present(output):
                output.parent.mkdir(parents=True, exist_ok=True)
                with original.open("rb") as source_stream, output.open("xb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)
                restored.append(relative.as_posix())
        # Verify the actual destination, including all composed payload files.
        verified = contract.canonical_g1_asset_metadata()
        if not _present(sidecar):
            with sidecar.open("x", encoding="utf-8") as stream:
                json.dump(verified, stream, indent=2, sort_keys=True)
                stream.write("\n")
        resolved_data.mkdir(parents=True, exist_ok=True)
        current.parent.mkdir(parents=True, exist_ok=True)
        if target is not None and not _present(current):
            current.symlink_to(target, target_is_directory=True)
        for name in RUNTIME_DIRECTORIES:
            (resolved_data / name).mkdir(parents=True, exist_ok=True)
        for name, link_target in LOGICAL_LINKS.items():
            path = root / name
            if not _present(path):
                path.symlink_to(link_target, target_is_directory=True)
        return {"root": str(root), "data_root": str(resolved_data),
                "restored_usd_files": restored, "robot_asset": verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="SomaForge checkout (defaults to this script's checkout)")
    parser.add_argument("--data-root", type=Path,
                        help="Create runtime/current as a link to this directory; never replaces existing entries")
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.root, args.data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
