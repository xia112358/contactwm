#!/usr/bin/env python3
"""Combine explicitly supplied motion manifests; never choose teacher checkpoints."""
from dataclasses import dataclass
import json
from pathlib import Path

import tyro


@dataclass(frozen=True)
class Options:
    manifests: tuple[Path, ...]
    output: Path


def main():
    options = tyro.cli(Options)
    from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
    loaded = [load_motion_terrain_manifest(str(path)) for path in options.manifests]
    if not loaded:
        raise ValueError("Supply at least one manifest")
    result = {key: value for key, value in loaded[0].items() if key not in {"path", "base_dir", "motion_files", "terrains"}}
    result.update(motion_files=[], terrains=[], terrain_spacing=[0., 0.])
    seen = set()
    for manifest in loaded:
        for key in ("robot_asset_sha256", "robot_asset_bundle_sha256", "kinematics_backend"):
            if manifest.get(key) != result.get(key):
                raise ValueError(f"Cannot combine manifests with different {key}")
        terrain_ids = {}
        for terrain in manifest["terrains"]:
            terrain_ids[terrain["terrain_id"]] = len(result["terrains"])
            result["terrains"].append({**terrain, "terrain_id": terrain_ids[terrain["terrain_id"]]})
        for motion in manifest["motion_files"]:
            path = str(Path(motion["motion_file"]).resolve())
            if path in seen:
                raise ValueError(f"Duplicate motion: {path}")
            seen.add(path)
            result["motion_files"].append({**motion, "motion_id": len(result["motion_files"]),
                "terrain_id": terrain_ids[motion["terrain_id"]]})
        result["terrain_spacing"] = [max(a, b) for a, b in zip(result["terrain_spacing"], manifest.get("terrain_spacing", (0., 0.)))]
    output = options.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    load_motion_terrain_manifest(str(output))
    print(output)


if __name__ == "__main__":
    main()
