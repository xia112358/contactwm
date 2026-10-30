from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from motion_edit.contact.jitter import generate_contact_jitter_plans


class ContactJitterPlanTests(unittest.TestCase):
    def _write_empty_contact_layer(self, contact: Path, motion_id: str) -> None:
        for name in ("anchors", "events", "patches", "transitions"):
            (contact / name).mkdir(parents=True, exist_ok=True)
            (contact / name / f"{motion_id}.jsonl").write_text("", encoding="utf-8")

    def test_generate_contact_jitter_plans_writes_surface_constrained_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "data" / "layers"
            contact = layers / "contact" / "test_ready"
            motion_id = "motion_a"
            self._write_empty_contact_layer(contact, motion_id)

            anchor = {
                "motion_id": motion_id,
                "anchor_id": "anchor_left_heel_0000_0010",
                "body": "left_heel",
                "start_frame": 0,
                "end_frame": 10,
                "editable": True,
                "world_position": [0.5, 0.5, 0.0],
                "surface_id": "box_top",
                "surface_type": "mesh_face",
                "surface_normal": [0.0, 0.0, 1.0],
                "surface_origin": [0.0, 0.0, 0.0],
                "surface_tangent_u": [1.0, 0.0, 0.0],
                "surface_tangent_v": [0.0, 1.0, 0.0],
                "surface_bounds": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
                "surface_coordinates": {"u": 0.5, "v": 0.5},
                "metadata": {
                    "surface_bindings": [
                        {
                            "polygon_surface_coordinates": [
                                {"u": 0.0, "v": 0.0},
                                {"u": 1.0, "v": 0.0},
                                {"u": 1.0, "v": 1.0},
                                {"u": 0.0, "v": 1.0},
                            ]
                        }
                    ]
                },
            }
            (contact / "anchors" / f"{motion_id}.jsonl").write_text(
                json.dumps(anchor) + "\n",
                encoding="utf-8",
            )
            cut_summary = root / "cut_summary.json"
            cut_summary.write_text(
                json.dumps(
                    [
                        {
                            "status": "ok",
                            "motion_id": motion_id,
                            "motion_asset_id": "asset_a",
                            "motion_path": str(root / "motion.npz"),
                            "ready_layer": "contact/test_ready",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch("motion_edit.contact.jitter.LAYERS_ROOT", layers):
                results, stats = generate_contact_jitter_plans(
                    cut_summary,
                    root / "plans",
                    augmentations_per_motion=1,
                    offset_radius=0.05,
                    edit_probability=1.0,
                    seed=7,
                )

            self.assertEqual(stats["plan_count"], 1)
            self.assertEqual(len(results), 1)
            plan = json.loads(results[0].plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "validated")
            self.assertEqual(len(plan["edits"]), 1)
            edit = plan["edits"][0]
            self.assertEqual(edit["constraint_mode"], "reject")
            self.assertFalse(edit["clamped"])
            self.assertEqual(edit["surface_id"], "box_top")
            self.assertAlmostEqual(edit["delta_world"][2], 0.0)
            after = edit["surface_coordinates_after"]
            self.assertGreaterEqual(after["u"], 0.0)
            self.assertLessEqual(after["u"], 1.0)
            self.assertGreaterEqual(after["v"], 0.0)
            self.assertLessEqual(after["v"], 1.0)

    def test_same_foot_adjacent_anchors_share_jitter_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "data" / "layers"
            contact = layers / "contact" / "test_ready"
            motion_id = "motion_b"
            self._write_empty_contact_layer(contact, motion_id)

            base_anchor = {
                "motion_id": motion_id,
                "body": "right_heel",
                "editable": True,
                "surface_id": "box_top",
                "surface_type": "mesh_face",
                "surface_normal": [0.0, 0.0, 1.0],
                "surface_origin": [0.0, 0.0, 0.0],
                "surface_tangent_u": [1.0, 0.0, 0.0],
                "surface_tangent_v": [0.0, 1.0, 0.0],
                "surface_bounds": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
                "metadata": {
                    "surface_bindings": [
                        {
                            "polygon_surface_coordinates": [
                                {"u": 0.0, "v": 0.0},
                                {"u": 1.0, "v": 0.0},
                                {"u": 1.0, "v": 1.0},
                                {"u": 0.0, "v": 1.0},
                            ]
                        }
                    ]
                },
            }
            anchors = [
                {
                    **base_anchor,
                    "anchor_id": "anchor_right_foot_sole_0000_0003",
                    "start_frame": 0,
                    "end_frame": 3,
                    "world_position": [0.50, 0.50, 0.0],
                    "surface_coordinates": {"u": 0.50, "v": 0.50},
                },
                {
                    **base_anchor,
                    "anchor_id": "anchor_right_foot_heel_0003_0005",
                    "start_frame": 3,
                    "end_frame": 5,
                    "world_position": [0.55, 0.50, 0.0],
                    "surface_coordinates": {"u": 0.55, "v": 0.50},
                },
            ]
            (contact / "anchors" / f"{motion_id}.jsonl").write_text(
                "".join(json.dumps(anchor) + "\n" for anchor in anchors),
                encoding="utf-8",
            )
            cut_summary = root / "cut_summary.json"
            cut_summary.write_text(
                json.dumps(
                    [
                        {
                            "status": "ok",
                            "motion_id": motion_id,
                            "motion_asset_id": "asset_b",
                            "motion_path": str(root / "motion.npz"),
                            "ready_layer": "contact/test_ready",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch("motion_edit.contact.jitter.LAYERS_ROOT", layers):
                results, stats = generate_contact_jitter_plans(
                    cut_summary,
                    root / "plans",
                    augmentations_per_motion=1,
                    offset_radius=0.05,
                    edit_probability=1.0,
                    max_edits=2,
                    seed=7,
                    sampler="local_disk",
                )

            self.assertEqual(stats["plan_count"], 1)
            self.assertEqual(len(results), 1)
            plan = json.loads(results[0].plan_path.read_text(encoding="utf-8"))
            self.assertEqual(len(plan["edits"]), 2)
            deltas = [edit["delta_world"] for edit in plan["edits"]]
            tangents = [edit["tangent_delta"] for edit in plan["edits"]]
            self.assertEqual(deltas[0], deltas[1])
            self.assertEqual(tangents[0], tangents[1])
            for edit in plan["edits"]:
                metadata = edit["metadata"]
                self.assertTrue(metadata["foot_group_jitter"])
                self.assertEqual(metadata["foot_group_body"], "right_heel")
                self.assertEqual(
                    metadata["foot_group_anchor_ids"],
                    ["anchor_right_foot_sole_0000_0003", "anchor_right_foot_heel_0003_0005"],
                )


if __name__ == "__main__":
    unittest.main()
