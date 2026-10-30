#!/usr/bin/env python3
"""Inventory the existing canonical PHP corpus and explicitly selected experts.

This reads manifests/status and stats files only. It never downloads, copies,
rewrites or selects checkpoints by scanning for the newest filename.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPONENTS = (
    "motions-locomotion", "motions-low-step", "motions-high-step",
    "motions-low-climb-76", "motions-high-climb-76",
)
DATASET = Path("runtime/current/motions/php_wbt_v1_20261016")
TEACHERS = Path("runtime/current/models/php_wbt_baseline_20261017")


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _entry(root: Path, source: Path, destination: Path | None = None) -> dict:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required PHP release input is missing: {source}")
    if _within(source, (root / "tmp").resolve()):
        raise ValueError(f"Temporary diagnostic files cannot be required release inputs: {source}")
    if destination is None:
        runtime = (root / "runtime/current").resolve()
        if _within(source, runtime):
            destination = Path("runtime/current") / source.relative_to(runtime)
        elif _within(source, root):
            destination = source.relative_to(root)
        else:
            raise ValueError(f"External input requires an explicit portable destination: {source}")
    if destination.is_absolute() or ".." in destination.parts:
        raise ValueError(f"Release destination must be a safe repository-relative path: {destination}")
    if "archives" in destination.parts:
        raise ValueError(f"Archive duplicates are not part of the PHP release: {destination}")
    return {"source": str(source), "destination": destination.as_posix(), "size": source.stat().st_size}


def build_inventory(root: Path, php_checkout: Path, *, expected_motion_count: int = 296) -> dict:
    root = root.expanduser().resolve()
    php_checkout = php_checkout.expanduser().resolve()
    dataset = root / DATASET
    files = {}

    def add(path: Path, destination: Path | None = None) -> None:
        entry = _entry(root, path, destination)
        prior = files.get(entry["destination"])
        if prior is not None and prior["source"] != entry["source"]:
            raise ValueError(f"Two PHP inputs have the same destination: {entry['destination']}")
        files[entry["destination"]] = entry

    motion_count = 0
    canonical_paths, raw_paths, terrain_paths = set(), set(), set()
    add(dataset / "collection.json")
    for component in COMPONENTS:
        manifest = dataset / component / "manifest.json"
        payload = json.loads(manifest.read_text())
        if payload.get("php_component") != component:
            raise ValueError(f"PHP component/manifest mismatch: {manifest}")
        add(manifest)
        add(dataset / "raw" / component / "README.md")

        def input_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else manifest.parent / path

        for motion in payload["motion_files"]:
            canonical = input_path(motion["motion_file"])
            original = input_path(motion["source_file"])
            add(canonical)
            add(original)
            canonical_paths.add(canonical.resolve())
            raw_paths.add(original.resolve())
            motion_count += 1
            if motion.get("reference_motion_file"):
                add(input_path(motion["reference_motion_file"]))
        for terrain in payload["terrains"]:
            add(input_path(terrain["terrain_file"]))
            source = input_path(terrain["source_file"])
            add(source)
            terrain_paths.add(source.resolve())
    if (motion_count != expected_motion_count or len(canonical_paths) != motion_count
            or len(raw_paths) != motion_count or len(terrain_paths) != motion_count):
        raise ValueError(f"Expected {expected_motion_count} distinct canonical/source/terrain pairs, got "
                         f"{motion_count}/{len(canonical_paths)}/{len(raw_paths)}/{len(terrain_paths)}")
    for name in ("README.md", "LICENSE", "NOTICE", "release-assets.json"):
        add(php_checkout / name, DATASET / "raw/upstream" / name)
    dataset_id = "php-wbt-296-data"
    result = [{"id": dataset_id,
        "purpose": "296 canonical G1 Newton-FK motions, paired raw PHP inputs and terrains; no observed contact/force labels",
        "status": "existing_canonical_kinematics", "dependencies": [],
        "files": sorted(files.values(), key=lambda item: item["destination"])}]
    status_path = root / TEACHERS / "status.json"
    status = json.loads(status_path.read_text())
    runs = {run["component"]: run for run in status["runs"]}
    for component in COMPONENTS:
        run = runs[component]
        selected = run.get("selected_checkpoint") or run.get("last_saved_checkpoint")
        if not selected:
            raise ValueError(f"No explicit saved/selected expert checkpoint in status: {component}")
        checkpoint = Path(selected).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = status_path.parent / checkpoint
        config = checkpoint.parent / "holosoma_config.yaml"
        entries = [_entry(root, checkpoint), _entry(root, config)]
        result.append({"id": "php-wbt-teacher-" + component.removeprefix("motions-"),
            "purpose": "Existing dense WBT expert checkpoint with its native training configuration: " + component,
            "status": "explicit_existing_checkpoint", "dependencies": [dataset_id], "files": entries})
    return {"schema": "somaforge_data_release_inventory_v1", "components": result,
            "summary": {"canonical_motion_count": motion_count, "raw_motion_count": len(raw_paths),
                        "raw_terrain_count": len(terrain_paths), "teacher_count": len(COMPONENTS),
                        "file_count": sum(len(component["files"]) for component in result),
                        "total_size": sum(entry["size"] for component in result for entry in component["files"])},
            "source_roots": {"checkout": str(root), "runtime": str((root / "runtime/current").resolve())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--php-checkout", type=Path, default=Path.home() / "php_parkour")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = build_inventory(args.root, args.php_checkout)
    output = args.output or args.root / "tmp/data_release_20261022/php_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(inventory, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), **inventory["summary"]}, indent=2))


if __name__ == "__main__":
    main()
