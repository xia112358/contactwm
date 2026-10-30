#!/usr/bin/env python3
"""Build a Holosoma motion-matched terrain manifest.

This script does not modify source motion or terrain files. It writes a manifest
and, when terrain URDFs are used, cached concatenated OBJ files under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _infer_climb_and_scale(path: Path) -> tuple[str, str]:
    match = re.search(r"(climb_\d+)_z_scale_([0-9.]+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer climb id and z scale from motion filename: {path.name}")
    return match.group(1), match.group(2)


def _parse_xyz(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in value.split()], dtype=np.float64)


def _load_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            face = []
            for token in line.split()[1:]:
                face.append(int(token.split("/")[0]) - 1)
            if len(face) >= 3:
                faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float64), faces


def _write_obj(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated cache for Holosoma motion-matched terrain manifest\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")


def _urdf_to_obj(urdf_path: Path, output_obj: Path) -> None:
    root = ET.parse(urdf_path).getroot()
    all_vertices = []
    all_faces: list[list[int]] = []
    vertex_offset = 0
    mesh_elems = root.findall(".//collision//mesh")
    if not mesh_elems:
        mesh_elems = root.findall(".//visual//mesh")
    for mesh_elem in mesh_elems:
        filename = mesh_elem.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        if mesh_path.suffix.lower() != ".obj":
            raise ValueError(f"Only OBJ mesh entries are supported in terrain URDF cache conversion: {mesh_path}")

        vertices, faces = _load_obj(mesh_path)
        scale = _parse_xyz(mesh_elem.attrib.get("scale"), (1.0, 1.0, 1.0))
        vertices = vertices * scale
        all_vertices.append(vertices)
        for face in faces:
            all_faces.append([idx + 1 + vertex_offset for idx in face])
        vertex_offset += len(vertices)

    if not all_vertices:
        raise ValueError(f"No OBJ mesh entries found in terrain URDF: {urdf_path}")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(output_obj, np.concatenate(all_vertices, axis=0), all_faces)


def build_manifest(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    terrain_cache_dir = output_dir / "terrain_obj_cache"
    motion_entries = []
    terrain_by_key: dict[tuple[str, str], dict] = {}

    motion_files = sorted(args.motion_dir.resolve().glob(args.motion_glob))
    if args.limit is not None:
        motion_files = motion_files[: args.limit]
    if not motion_files:
        raise FileNotFoundError(f"No motion files found: {args.motion_dir / args.motion_glob}")

    for motion_file in motion_files:
        climb_id, z_scale = _infer_climb_and_scale(motion_file)
        terrain_dir = args.terrain_root.resolve() / climb_id
        terrain_urdf = terrain_dir / f"multi_boxes_z_scale_{z_scale}.urdf"
        if not terrain_urdf.exists():
            print(f"[skip] missing terrain for {motion_file.name}: {terrain_urdf}")
            continue

        key = (climb_id, z_scale)
        if key not in terrain_by_key:
            terrain_id = len(terrain_by_key)
            terrain_obj = terrain_cache_dir / climb_id / f"multi_boxes_z_scale_{z_scale}.obj"
            if not terrain_obj.exists() or args.overwrite_cache:
                _urdf_to_obj(terrain_urdf, terrain_obj)
            terrain_by_key[key] = {
                "terrain_id": terrain_id,
                "terrain_file": str(terrain_obj.relative_to(output_dir)),
                "source_urdf": str(terrain_urdf),
            }
        motion_entries.append(
            {
                "motion_file": str(motion_file),
                "terrain_id": terrain_by_key[key]["terrain_id"],
                "weight": 1.0,
            }
        )

    return {
        "schema_version": 1,
        "description": "Motion files bound to matching terrain ids for Holosoma WBT.",
        "terrains": list(terrain_by_key.values()),
        "motion_files": motion_entries,
    }


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True, help="Directory containing converted Holosoma .npz motions.")
    parser.add_argument("--motion-glob", default="*.npz")
    parser.add_argument("--terrain-root", type=Path, default=repo_root / "OmniRetarget_Dataset/models/terrain")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "configs/motion_matched")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (args.output_dir / "manifest.json")
    manifest = build_manifest(args)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['motion_files'])} motion(s), {len(manifest['terrains'])} terrain(s) to {output}")


if __name__ == "__main__":
    main()
