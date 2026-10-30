"""Utilities for motion-to-terrain binding manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from somaforge_core.robot_assets import canonical_g1_asset_metadata
from somaforge_core.motion_schema import NEWTON_KINEMATICS_BACKENDS
from somaforge_core.published_data import load_published_json, relocate_published_values, resolve_published_path

from holosoma.utils.path import resolve_data_file_path


def _load_manifest_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on optional local install
            raise ImportError(f"PyYAML is required to read motion-terrain manifest: {path}") from exc
        data = relocate_published_values(yaml.safe_load(text))
    else:
        data = load_published_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Motion-terrain manifest must contain a mapping: {path}")
    return data


def load_motion_terrain_manifest(manifest_path: str) -> dict[str, Any]:
    """Load a motion-terrain manifest and resolve relative paths.

    Supported shapes:
    - InstinctLab-style:
      {"motion_files": [{"motion_file": "...", "terrain_id": 0}],
       "terrains": [{"terrain_file": "...", "terrain_id": 0}]}
    - Holosoma scene-style:
      {"scenes": [{"motion_file": "...", "terrain_file": "...", "terrain_id": 0}]}
    """
    path = Path(resolve_data_file_path(manifest_path)).expanduser().resolve()
    data = _load_manifest_file(path)
    base_dir = path.parent

    terrains = _normalize_terrains(data, base_dir)
    motion_files = _normalize_motion_files(data, base_dir, terrains)
    if not terrains:
        raise ValueError(f"Manifest contains no terrains: {path}")
    if not motion_files:
        raise ValueError(f"Manifest contains no motion files: {path}")

    if data.get("schema") == "somaforge_motion_terrain_manifest_v1":
        _validate_canonical_manifest(data, motion_files, terrains, path)

    result = dict(data)
    result.update({
        "path": str(path),
        "base_dir": str(base_dir),
        "terrains": terrains,
        "motion_files": motion_files,
    })
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(path: str, expected: Any, *, context: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{context} requires a SHA256 digest")
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} file does not exist: {resolved}")
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(f"{context} SHA256 mismatch: expected {expected}, got {actual}")


def _validate_canonical_manifest(
    data: dict[str, Any],
    motion_files: list[dict[str, Any]],
    terrains: list[dict[str, Any]],
    path: Path,
) -> None:
    if data.get("robot_asset_id") != "robot.g1.spherehand":
        raise ValueError(f"Canonical manifest must use robot.g1.spherehand: {path}")
    robot_asset = canonical_g1_asset_metadata()
    expected_fingerprints = {
        "robot_asset_sha256": robot_asset["urdf_sha256"],
        "robot_asset_bundle_sha256": robot_asset["asset_bundle_sha256"],
        "robot_asset_usd_bundle_sha256": robot_asset["usd_bundle_sha256"],
    }
    for key, expected in expected_fingerprints.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Canonical manifest {key} mismatch: expected {expected}, got {data.get(key)!r}: {path}"
            )
    manifest_backend = data.get("kinematics_backend")
    if manifest_backend not in NEWTON_KINEMATICS_BACKENDS:
        raise ValueError(f"Canonical manifest must use a supported Newton kinematics backend: {path}")
    for entry in motion_files:
        if entry.get("kinematics_backend") != manifest_backend:
            raise ValueError(
                f"motion_id={entry.get('motion_id')} kinematics backend does not match manifest: {path}"
            )
        _validate_digest(
            entry["motion_file"], entry.get("motion_sha256"), context=f"motion_id={entry.get('motion_id')}"
        )
        source_file = entry.get("source_file")
        if not source_file:
            raise ValueError(f"motion_id={entry.get('motion_id')} requires source_file")
        _validate_digest(
            str(source_file), entry.get("source_sha256"), context=f"motion_id={entry.get('motion_id')} source"
        )
        reference_motion_file = entry.get("reference_motion_file")
        if reference_motion_file is not None:
            _validate_digest(
                str(reference_motion_file),
                entry.get("reference_motion_sha256"),
                context=f"motion_id={entry.get('motion_id')} reference motion",
            )
    for entry in terrains:
        _validate_digest(
            entry["terrain_file"], entry.get("terrain_sha256"), context=f"terrain_id={entry.get('terrain_id')}"
        )


def _resolve_manifest_path(value: str, base_dir: Path) -> str:
    return str(resolve_published_path(value, base_dir=base_dir))


def _normalize_terrains(data: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    if "terrains" in data:
        terrains = data["terrains"]
    elif "scenes" in data:
        seen: dict[Any, dict[str, Any]] = {}
        for idx, scene in enumerate(data["scenes"]):
            terrain_file = scene.get("terrain_file") or scene.get("terrain_obj") or scene.get("obstacle_urdf")
            if not terrain_file:
                continue
            terrain_id = scene.get("terrain_id", scene.get("seq", idx))
            seen.setdefault(
                terrain_id,
                {
                    "terrain_id": terrain_id,
                    "terrain_file": terrain_file,
                },
            )
        terrains = list(seen.values())
    else:
        terrains = []

    normalized = []
    for idx, terrain in enumerate(terrains):
        terrain_id = terrain.get("terrain_id", idx)
        terrain_file = terrain.get("terrain_file")
        if not terrain_file:
            raise ValueError(f"Terrain entry is missing terrain_file: {terrain}")
        entry = dict(terrain)
        entry.update(
            {
                "terrain_id": int(terrain_id),
                "terrain_file": _resolve_manifest_path(str(terrain_file), base_dir),
            }
        )
        normalized.append(entry)
    return normalized


def _normalize_motion_files(
    data: dict[str, Any],
    base_dir: Path,
    terrains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "motion_files" in data:
        motion_files = data["motion_files"]
    elif "scenes" in data:
        terrain_by_file = {terrain["terrain_file"]: terrain["terrain_id"] for terrain in terrains}
        motion_files = []
        for idx, scene in enumerate(data["scenes"]):
            motion_file = scene.get("motion_file")
            if not motion_file:
                continue
            terrain_id = scene.get("terrain_id")
            if terrain_id is None:
                terrain_file = scene.get("terrain_file") or scene.get("terrain_obj") or scene.get("obstacle_urdf")
                terrain_id = terrain_by_file.get(_resolve_manifest_path(str(terrain_file), base_dir), scene.get("seq", idx))
            motion_files.append(
                {
                    "motion_file": motion_file,
                    "terrain_id": terrain_id,
                    "weight": scene.get("weight", 1.0),
                }
            )
    else:
        motion_files = []

    terrain_ids = {terrain["terrain_id"] for terrain in terrains}
    normalized = []
    for motion in motion_files:
        terrain_id = int(motion["terrain_id"])
        if terrain_id not in terrain_ids:
            raise ValueError(f"Motion references unknown terrain_id={terrain_id}: {motion}")
        entry = dict(motion)
        entry.update(
            {
                "motion_file": _resolve_manifest_path(str(motion["motion_file"]), base_dir),
                "terrain_id": terrain_id,
                "weight": float(motion.get("weight", 1.0)),
            }
        )
        if entry.get("source_file"):
            entry["source_file"] = _resolve_manifest_path(str(entry["source_file"]), base_dir)
        if entry.get("reference_motion_file"):
            entry["reference_motion_file"] = _resolve_manifest_path(str(entry["reference_motion_file"]), base_dir)
        normalized.append(entry)
    return normalized
