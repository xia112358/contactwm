from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from somaforge_core.contact_schema import encode_contact_force_provenance, newton_contact_provenance
from somaforge_core.robot_assets import encode_robot_asset_json

from motion_edit.adapters.asset_manifest import load_asset_manifest
from motion_edit.contact.surface_catalog import surfaces_from_obj_mesh_faces


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssetManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        force = root / "force.npz"
        reference = root / "reference.npz"
        source = root / "source.npz"
        terrain = root / "terrain.obj"
        provenance = newton_contact_provenance(solver_config={"nconmax_per_env": 64})
        mask = np.asarray([[True] + [False] * 7, [False, True] + [False] * 6], dtype=bool)
        joint_names = np.asarray([f"joint_{index}" for index in range(29)])
        joint_pos = np.zeros((2, 36), dtype=np.float32)
        robot_asset_json = np.asarray(encode_robot_asset_json())
        np.savez(
            reference,
            fps=np.asarray(50.0),
            joint_names=joint_names,
            joint_pos=joint_pos,
            robot_asset_json=robot_asset_json,
        )
        np.savez(
            force,
            fps=np.asarray(50.0),
            joint_names=joint_names,
            joint_pos=joint_pos,
            robot_asset_json=robot_asset_json,
            contact_force_provenance_json=encode_contact_force_provenance(provenance),
            contact_force_part_order=np.asarray(["LHEE", "LTOE", "RHEE", "RTOE", "LH", "RH", "LK", "RK"]),
            contact_force_part_mask=mask,
            contact_force_part_w=np.zeros((2, 8, 3), dtype=float),
            contact_force_part_position_w=np.zeros((2, 8, 3), dtype=float),
            proto_start_idx=np.asarray([0], dtype=np.int64),
            proto_end_idx=np.asarray([2], dtype=np.int64),
        )
        np.savez(source, fps=np.asarray(50.0))
        terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "somaforge_motion_terrain_manifest_v1",
                    "source_kind": "newton_policy_rollout_contact_force_8part",
                    "contact_force_schema": "somaforge_contact_force_8part_v1",
                    "contact_force_backend": "isaaclab3_newton_mjwarp",
                    "terrains": [
                        {"terrain_id": 0, "terrain_file": terrain.name, "terrain_sha256": _sha256(terrain)}
                    ],
                    "motion_files": [
                        {
                            "motion_id": 0,
                            "motion_file": force.name,
                            "motion_sha256": _sha256(force),
                            "reference_motion_file": reference.name,
                            "reference_motion_sha256": _sha256(reference),
                            "source_file": source.name,
                            "source_sha256": _sha256(source),
                            "terrain_id": 0,
                            "contact_solver_sha256": provenance["solver_config_sha256"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_loads_and_validates_manifest_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._fixture(Path(tmp))
            assets = load_asset_manifest(manifest)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].motion_id, "climb_00")
        self.assertEqual(assets[0].motion_asset_id, "climb_00_newton_8part")
        self.assertEqual(assets[0].reference_motion_path.name, "reference.npz")
        self.assertTrue(assets[0].provenance["training_eligible"])

    def test_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            (root / "terrain.obj").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terrain_sha256 mismatch"):
                load_asset_manifest(manifest)

    def test_direct_obj_surface_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            terrain = Path(tmp) / "terrain.obj"
            terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            surfaces = surfaces_from_obj_mesh_faces(
                motion_id="climb_00", obj_path=terrain, include_sides=False, include_ground=False
            )
        self.assertEqual(len(surfaces), 1)
        self.assertGreater(surfaces[0].normal[2], 0.99)
        self.assertEqual(surfaces[0].metadata["mesh_path"], str(terrain.resolve()))


if __name__ == "__main__":
    unittest.main()
