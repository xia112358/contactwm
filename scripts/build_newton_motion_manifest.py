#!/usr/bin/env python3
"""Fingerprint Newton-canonical motions and their terrain/source assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from somaforge_core import (
    NEWTON_KINEMATICS_BACKEND,
    POSE_FINITE_DIFFERENCE,
    canonical_g1_asset_metadata,
    decode_kinematics_provenance,
    sha256_file,
    validate_pose_velocity_consistency,
    validate_root_body_consistency,
)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    base = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema"] = "somaforge_motion_terrain_manifest_v1"
    payload["schema_version"] = 1
    payload["description"] = (
        f"Canonical {len(payload['motion_files'])}-motion raw OmniRetarget qpos baseline computed with "
        "Isaac Lab 3/Newton FK. No force channels and no Motion Edit."
    )
    payload["source_kind"] = "omniretarget_qpos_newton_fk"
    payload["robot_asset_id"] = "robot.g1.spherehand"
    robot_asset = canonical_g1_asset_metadata()
    payload["robot_asset_sha256"] = robot_asset["urdf_sha256"]
    payload["robot_asset_bundle_sha256"] = robot_asset["asset_bundle_sha256"]
    payload["robot_asset_usd_bundle_sha256"] = robot_asset["usd_bundle_sha256"]
    payload["kinematics_backend"] = NEWTON_KINEMATICS_BACKEND

    for entry in payload["motion_files"]:
        motion_path = (base / entry["motion_file"]).resolve()
        source_path = (args.raw_root.expanduser().resolve() / motion_path.name).resolve()
        with np.load(motion_path, allow_pickle=False) as data:
            provenance = decode_kinematics_provenance(
                data.get("kinematics_provenance_json", None),
                context=str(motion_path),
            )
            body_names = [str(name) for name in data["body_names"].tolist()]
            validate_root_body_consistency(
                data["joint_pos"],
                data["body_pos_w"],
                data["body_quat_w"],
                root_body_index=body_names.index("pelvis"),
            )
            validate_pose_velocity_consistency(
                data["body_pos_w"],
                data["body_quat_w"],
                data["body_lin_vel_w"],
                data["body_ang_vel_w"],
                float(np.asarray(data["fps"]).reshape(-1)[0]),
            )
        source_sha256 = sha256_file(source_path)
        if provenance.get("source_sha256") != source_sha256:
            raise ValueError(f"Source SHA256 in provenance does not match {source_path}")
        if provenance.get("velocity_derivation") != POSE_FINITE_DIFFERENCE:
            raise ValueError(f"Motion does not use pose-derived velocities: {motion_path}")
        entry["source_file"] = _relative(source_path, base)
        entry["source_sha256"] = source_sha256
        entry["motion_sha256"] = sha256_file(motion_path)
        entry["kinematics_schema"] = provenance["schema"]
        entry["kinematics_backend"] = provenance["kinematics_backend"]
        entry["velocity_derivation"] = provenance["velocity_derivation"]

    for entry in payload["terrains"]:
        terrain_path = (base / entry["terrain_file"]).resolve()
        entry["terrain_sha256"] = sha256_file(terrain_path)

    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote canonical manifest: {manifest_path} ({len(payload['motion_files'])} motions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
