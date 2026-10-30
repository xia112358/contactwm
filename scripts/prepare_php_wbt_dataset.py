#!/usr/bin/env python3
"""Verify PHP releases and build independent canonical WBT datasets."""

# Deferred project imports keep verified downloads independent of Newton/Isaac Lab.
# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    "motions-locomotion",
    "motions-low-step",
    "motions-high-step",
    "motions-low-climb-76",
    "motions-high-climb-76",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_downloader(checkout: Path):
    spec = importlib.util.spec_from_file_location("php_release_assets", checkout / "scripts/download_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_component(module, checkout: Path, component: str, raw_root: Path) -> Path:
    metadata = module.read_component(checkout / "release-assets.json", component)
    destination = raw_root / component
    if not module.verify(metadata, destination):
        return destination
    cached = Path.home() / ".cache/php-parkour" / component
    if not module.verify(metadata, cached):
        destination.mkdir(parents=True, exist_ok=True)
        for entry in metadata["files"]:
            target = destination / module.relative_path(entry["path"])
            if target.exists():
                raise FileExistsError(f"Refusing to replace unverified source: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached / entry["path"], target)
    else:
        archive = metadata["archive"]
        archive_path = raw_root / "archives" / Path(archive["url"]).name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            if archive_path.stat().st_size != archive["size"] or module.sha256(archive_path) != archive["sha256"]:
                raise ValueError(f"Existing archive failed verification; preserved at {archive_path}")
        else:
            print(f"Downloading {component}: {archive['url']}", flush=True)
            if not archive["url"].startswith("https://github.com/amazon-far/php_parkour/releases/download/"):
                raise ValueError("Expected an official PHP HTTPS release URL")
            with urlopen(archive["url"], timeout=60) as response, archive_path.open("xb") as stream:  # noqa: S310
                shutil.copyfileobj(response, stream)
            if archive_path.stat().st_size != archive["size"] or module.sha256(archive_path) != archive["sha256"]:
                raise ValueError(f"Download failed verification; preserved at {archive_path}")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to replace unverified extraction: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        module.extract_verified(archive_path, metadata, destination)
    problems = module.verify(metadata, destination)
    if problems:
        raise ValueError(problems)
    print(f"Verified {component}: {len(metadata['files'])} files", flush=True)
    return destination


def terrain_mesh(boxes: np.ndarray, variant: int, body_positions: np.ndarray):
    """PHP rows are [xyz, quaternion WXYZ, full box extents]."""
    import trimesh

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[-1] != 10 or not 0 <= variant < boxes.shape[1]:
        raise ValueError(f"Invalid PHP terrain shape/variant: {boxes.shape}, {variant}")
    parts = []
    for row in boxes[:, variant]:
        if not np.isfinite(row).all() or np.any(row[7:] <= 0):
            raise ValueError("Nonfinite terrain row or nonpositive box extents")
        quaternion = row[3:7]
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4):
            raise ValueError("PHP terrain quaternion is not normalized")
        transform = trimesh.transformations.quaternion_matrix(quaternion)
        transform[:3, 3] = row[:3]
        parts.append(trimesh.creation.box(extents=row[7:], transform=transform))
    xy = np.asarray(body_positions)[..., :2].reshape(-1, 2)
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    if parts:
        lower = np.minimum(lower, np.stack([p.bounds[0, :2] for p in parts]).min(axis=0))
        upper = np.maximum(upper, np.stack([p.bounds[1, :2] for p in parts]).max(axis=0))
    lower -= 2.0
    upper += 2.0
    ground = trimesh.creation.box(extents=[*(upper - lower), 0.04])
    ground.apply_translation([*((lower + upper) / 2), -0.02])
    return trimesh.util.concatenate([ground, *parts])


def prepare_component(source: Path, output: Path, component: str, variant: int) -> dict:
    from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
    from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk
    from somaforge_core import (
        DIRECT_NEWTON_KINEMATICS_BACKEND,
        canonical_g1_asset_metadata,
        decode_kinematics_provenance,
        sha256_file,
    )

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = load_motion_terrain_manifest(str(manifest_path))
        if previous.get("php_terrain_variant") != variant:
            raise ValueError("Existing dataset uses a different terrain variant")
        return {
            "component": component,
            "manifest": str(manifest_path),
            "motion_count": len(previous["motion_files"]),
            "terrain_spacing": previous["terrain_spacing"],
        }
    output.mkdir(parents=True, exist_ok=True)
    motions, terrains, durations = [], [], []
    all_bounds = []
    for index, original in enumerate(sorted(source.glob("*_motion.npz"))):
        terrain_source = original.with_name(original.name.replace("_motion.npz", "_terrain.npy"))
        if not terrain_source.is_file():
            raise FileNotFoundError(terrain_source)
        motion_path = output / "motions" / f"motion_{index:04d}.npz"
        terrain_path = output / "terrains" / f"terrain_{index:04d}.obj"
        if not motion_path.exists():
            canonicalize_motion_with_direct_newton_fk(original, motion_path, device="cpu")
        with np.load(motion_path, allow_pickle=False) as data:
            provenance = decode_kinematics_provenance(data["kinematics_provenance_json"], context=str(motion_path))
            if provenance["source_sha256"] != sha256_file(original):
                raise ValueError(f"Canonical source changed: {original}")
            positions = data["body_pos_w"]
            fps = float(data["fps"])
            durations.append(len(data["joint_pos"]) / fps)
        mesh = terrain_mesh(np.load(terrain_source, allow_pickle=False), variant, positions)
        terrain_path.parent.mkdir(parents=True, exist_ok=True)
        if not terrain_path.exists():
            mesh.export(terrain_path)
        all_bounds.append(mesh.bounds)
        motions.append(
            {
                "motion_id": f"{component}_{index:04d}",
                "motion_file": str(motion_path.relative_to(output)),
                "terrain_id": index,
                "weight": 1.0,
                "source_file": str(original),
                "source_sha256": sha256_file(original),
                "motion_sha256": sha256_file(motion_path),
                "kinematics_backend": DIRECT_NEWTON_KINEMATICS_BACKEND,
            }
        )
        terrains.append(
            {
                "terrain_id": index,
                "terrain_file": str(terrain_path.relative_to(output)),
                "includes_ground": True,
                "terrain_sha256": sha256_file(terrain_path),
                "source_file": str(terrain_source),
                "source_sha256": sha256_file(terrain_source),
                "php_variant": variant,
            }
        )
        print(f"Canonicalized {component} {index + 1}: {len(positions)} frames", flush=True)
    if not motions:
        raise ValueError(f"No PHP motions in {source}")
    bounds = np.stack(all_bounds)
    # Existing terrain placement adds a grid origin without recentering clips.
    # Global extrema therefore determine a safe spacing even for asymmetric clips.
    spacing = bounds[:, 1, :2].max(axis=0) - bounds[:, 0, :2].min(axis=0) + 2.0
    asset = canonical_g1_asset_metadata()
    payload = {
        "schema": "somaforge_motion_terrain_manifest_v1",
        "schema_version": 1,
        "source_kind": "php_release_direct_newton_fk",
        "robot_asset_id": "robot.g1.spherehand",
        "robot_asset_sha256": asset["urdf_sha256"],
        "robot_asset_bundle_sha256": asset["asset_bundle_sha256"],
        "robot_asset_usd_bundle_sha256": asset["usd_bundle_sha256"],
        "kinematics_backend": DIRECT_NEWTON_KINEMATICS_BACKEND,
        "php_component": component,
        "php_terrain_variant": variant,
        "terrain_randomization": "fixed released variant; no injected obstacle noise",
        "terrain_spacing": spacing.tolist(),
        "duration_seconds": sum(durations),
        "max_duration_seconds": max(durations),
        "motion_files": motions,
        "terrains": terrains,
    }
    write_json(manifest_path, payload)
    load_motion_terrain_manifest(str(manifest_path))
    return {
        "component": component,
        "manifest": str(manifest_path),
        "motion_count": len(motions),
        "terrain_spacing": spacing.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--php-checkout", type=Path, default=Path.home() / "php_parkour")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--terrain-variant", type=int, default=0)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    module = load_downloader(args.php_checkout)
    sources = {name: fetch_component(module, args.php_checkout, name, output / "raw") for name in args.components}
    if args.download_only:
        return
    entries = [prepare_component(sources[name], output / name, name, args.terrain_variant) for name in args.components]
    write_json(output / "collection.json", {"schema": "php_independent_wbt_collection_v1", "datasets": entries})
    print(json.dumps(entries, indent=2), flush=True)


if __name__ == "__main__":
    main()
