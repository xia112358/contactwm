from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from somaforge_core import (
    canonical_g1_asset_metadata,
    encode_contact_force_provenance,
    encode_kinematics_provenance,
    newton_contact_provenance,
    newton_rollout_kinematics_provenance,
    sha256_file,
)
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
from scripts.extract_all_rollout_ref_contact_force_demos import _build_output_manifest, _single_motion_manifest


def _write_base_manifest(tmp_path: Path) -> tuple[Path, dict]:
    motion = tmp_path / "climb_01.npz"
    np.savez_compressed(motion, joint_pos=np.zeros((2, 36), dtype=np.float32))
    source = tmp_path / "climb_01_source.npz"
    np.savez_compressed(source, qpos=np.zeros((2, 36), dtype=np.float32))
    terrain = tmp_path / "terrain_01.obj"
    terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    asset = canonical_g1_asset_metadata()
    payload = {
        "schema": "somaforge_motion_terrain_manifest_v1",
        "schema_version": 1,
        "robot_asset_id": "robot.g1.spherehand",
        "robot_asset_sha256": asset["urdf_sha256"],
        "robot_asset_bundle_sha256": asset["asset_bundle_sha256"],
        "robot_asset_usd_bundle_sha256": asset["usd_bundle_sha256"],
        "kinematics_backend": "isaaclab3_newton_fk",
        "terrains": [
            {"terrain_id": 1, "terrain_file": terrain.name, "terrain_sha256": sha256_file(terrain)}
        ],
        "motion_files": [
            {
                "motion_id": "01",
                "motion_file": motion.name,
                "motion_sha256": sha256_file(motion),
                "source_file": source.name,
                "source_sha256": sha256_file(source),
                "terrain_id": 1,
                "kinematics_backend": "isaaclab3_newton_fk",
            }
        ],
    }
    path = tmp_path / "base_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, load_motion_terrain_manifest(str(path))


def test_single_motion_manifest_preserves_canonical_robot_fingerprints(tmp_path: Path) -> None:
    base_path, base = _write_base_manifest(tmp_path)
    motion_entry = base["motion_files"][0]

    single = _single_motion_manifest(base, base_path, motion_entry, climb_id=1)
    output = tmp_path / "climb_01_manifest.json"
    output.write_text(json.dumps(single), encoding="utf-8")
    loaded = load_motion_terrain_manifest(str(output))

    assert loaded["robot_asset_sha256"] == base["robot_asset_sha256"]
    assert loaded["robot_asset_bundle_sha256"] == base["robot_asset_bundle_sha256"]
    assert loaded["robot_asset_usd_bundle_sha256"] == base["robot_asset_usd_bundle_sha256"]
    assert len(loaded["motion_files"]) == 1
    assert len(loaded["terrains"]) == 1


def test_output_manifest_preserves_canonical_robot_fingerprints(tmp_path: Path) -> None:
    base_path, base = _write_base_manifest(tmp_path)
    output_dir = tmp_path / "motions"
    output_dir.mkdir()
    recording = tmp_path / "test_recording.npz"
    recording.write_bytes(b"recording")
    provenance = newton_contact_provenance(
        solver_config={"solver": "mjwarp", "collision_pipeline": "newton"},
        source_recording="test_recording.npz",
        force_reduce="sum",
        threshold_n=10.0,
    )
    kinematics_provenance = newton_rollout_kinematics_provenance(
        source_path=str(recording),
        source_sha256=sha256_file(recording),
        output_fps=50.0,
        body_names=["pelvis"],
    )
    np.savez_compressed(
        output_dir / "climb_01_rollout_ref_contact_force.npz",
        contact_force_provenance_json=np.asarray(encode_contact_force_provenance(provenance)),
        kinematics_provenance_json=np.asarray(encode_kinematics_provenance(kinematics_provenance)),
        rollout_ref_source_recording=np.asarray(str(recording)),
    )
    output_manifest = tmp_path / "newton_contact_force_8part.json"

    _build_output_manifest(base, base_path, output_dir, output_manifest, allow_missing=True)
    loaded = load_motion_terrain_manifest(str(output_manifest))

    assert loaded["robot_asset_sha256"] == base["robot_asset_sha256"]
    assert loaded["robot_asset_bundle_sha256"] == base["robot_asset_bundle_sha256"]
    assert loaded["robot_asset_usd_bundle_sha256"] == base["robot_asset_usd_bundle_sha256"]
    assert loaded["kinematics_backend"] == "isaaclab3_newton_rollout"
    assert loaded["motion_files"][0]["source_file"] == str(recording.resolve())
    assert len(loaded["motion_files"]) == 1
