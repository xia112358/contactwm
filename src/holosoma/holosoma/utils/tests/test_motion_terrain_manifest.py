import hashlib
import json

import pytest

from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
from somaforge_core.robot_assets import canonical_g1_asset_metadata


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp_path, *, backend="isaaclab3_newton_fk"):
    motion = tmp_path / "motion.npz"
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    terrain = tmp_path / "terrain.obj"
    motion.write_bytes(b"motion")
    source.write_bytes(b"source")
    reference.write_bytes(b"reference")
    terrain.write_bytes(b"terrain")
    robot_asset = canonical_g1_asset_metadata()
    payload = {
        "schema": "somaforge_motion_terrain_manifest_v1",
        "robot_asset_id": "robot.g1.spherehand",
        "robot_asset_sha256": robot_asset["urdf_sha256"],
        "robot_asset_bundle_sha256": robot_asset["asset_bundle_sha256"],
        "robot_asset_usd_bundle_sha256": robot_asset["usd_bundle_sha256"],
        "kinematics_backend": backend,
        "motion_files": [
            {
                "motion_id": 7,
                "motion_file": motion.name,
                "motion_sha256": _sha256(b"motion"),
                "source_file": source.name,
                "source_sha256": _sha256(b"source"),
                "reference_motion_file": reference.name,
                "reference_motion_sha256": _sha256(b"reference"),
                "terrain_id": 3,
                "kinematics_schema": "somaforge_canonical_motion_v1",
                "kinematics_backend": backend,
            }
        ],
        "terrains": [
            {
                "terrain_id": 3,
                "terrain_file": terrain.name,
                "terrain_sha256": _sha256(b"terrain"),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_manifest_validates_hashes_and_preserves_metadata(tmp_path) -> None:
    payload = load_motion_terrain_manifest(str(_manifest(tmp_path)))

    assert payload["motion_files"][0]["motion_id"] == 7
    assert payload["motion_files"][0]["kinematics_schema"] == "somaforge_canonical_motion_v1"
    assert payload["motion_files"][0]["reference_motion_file"] == str((tmp_path / "reference.npz").resolve())
    assert payload["terrains"][0]["terrain_sha256"] == _sha256(b"terrain")


def test_canonical_manifest_rejects_changed_motion(tmp_path) -> None:
    path = _manifest(tmp_path)
    (tmp_path / "motion.npz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_motion_terrain_manifest(str(path))


def test_canonical_manifest_rejects_changed_robot_asset(tmp_path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["robot_asset_usd_bundle_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="robot_asset_usd_bundle_sha256 mismatch"):
        load_motion_terrain_manifest(str(path))


def test_canonical_manifest_accepts_newton_rollout_backend(tmp_path) -> None:
    payload = load_motion_terrain_manifest(str(_manifest(tmp_path, backend="isaaclab3_newton_rollout")))

    assert payload["kinematics_backend"] == "isaaclab3_newton_rollout"
