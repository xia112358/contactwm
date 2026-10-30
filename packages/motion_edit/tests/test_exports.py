from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motion_edit.contact import (
    ContactAnchorRecord,
    ContactGraph,
    ContactSurfaceRecord,
    contact_graph_from_masks,
)
from motion_edit.export import (
    export_contact_overlay,
    export_cutter_segments,
    export_motion_manifest,
    export_split_npz,
    export_surface_binding_overlay,
    export_surface_binding_report,
)
from motion_edit.io import read_jsonl
from motion_edit.schema import SegmentRecord


def _contact_segment(motion_path: str) -> SegmentRecord:
    return SegmentRecord(
        motion_id="motion_a",
        segment_id="motion_a_force_0000",
        start_frame=1,
        end_frame=4,
        source="force_contact",
        status="candidate",
        motion_path=motion_path,
        clip_npz=motion_path,
        contact_start="11",
        contact_end="00",
        active="10",
        support="01",
        metadata={
            "contact_transition": {
                "transition_id": "motion_a_force_transition_0000",
                "start_frame": 1,
                "end_frame": 4,
                "transition_type": "support_transfer",
            },
            "active_body": "LF",
            "support_bodies": ["RF"],
            "source_anchor_id": "anchor_src",
            "target_anchor_id": "anchor_dst",
            "transition_type": "support_transfer",
            "contact_events": [{"event_id": "event_0"}],
            "contact_anchors": [{"anchor_id": "anchor_src"}, {"anchor_id": "anchor_dst"}],
            "contact_patches": [{"patch_id": "patch_src", "patch_type": "foot"}],
            "old_anchor_world": [0.0, 0.0, 0.0],
            "new_anchor_world": [0.1, 0.0, 0.0],
            "delta_world": [0.1, 0.0, 0.0],
            "affected_frames": [1, 4],
            "contact_anchor_edit": {
                "edit_type": "move_contact_anchor",
                "anchor_id": "anchor_src",
                "requested_delta_world": [0.1, 0.0, 0.2],
                "tangent_delta": [0.1, 0.0],
                "surface_id": "platform_top",
                "surface_normal": [0.0, 0.0, 1.0],
                "surface_coordinates_before": {"u": 0.0, "v": 0.0},
                "surface_coordinates_after": {"u": 0.1, "v": 0.0},
                "constraint_mode": "reject",
                "clamped": False,
            },
        },
    )


class ExportContactMetadataTests(unittest.TestCase):
    def test_surface_binding_report_writes_summary_and_anchor_details(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="box_0_top",
            object_id="box_0",
            surface_type="box_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
        )
        graph = ContactGraph(
            motion_id="motion_a",
            anchors=[
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="bound",
                    body="LF",
                    start_frame=0,
                    end_frame=2,
                    world_position=[0.1, 0.2, 0.0],
                    object_id="box_0",
                    surface_id="box_0_top",
                    surface_type="box_face",
                    surface_normal=[0.0, 0.0, 1.0],
                    surface_bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
                    surface_coordinates={"u": 0.1, "v": 0.2},
                    metadata={
                        "surface_bindings": [
                            {
                                "original_world_position": [0.1, 0.2, 0.02],
                                "projected_world_position": [0.1, 0.2, 0.0],
                                "bound_world_position": [0.1, 0.2, 0.0],
                                "signed_surface_distance": 0.02,
                                "raw_surface_coordinates": {"u": 0.1, "v": 0.2},
                                "surface_coordinates": {"u": 0.1, "v": 0.2},
                                "clamped": False,
                                "surface_binding_source": "test",
                            }
                        ]
                    },
                ),
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="failed",
                    body="LF",
                    start_frame=0,
                    end_frame=2,
                    world_position=[2.0, 0.0, 0.0],
                    metadata={"surface_binding_failed": True, "surface_binding_failure_reason": "no compatible surface"},
                ),
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="unbound",
                    body="body_0",
                    start_frame=0,
                    end_frame=2,
                    world_position=[0.0, 0.0, 0.0],
                ),
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="edge",
                    body="LF",
                    start_frame=0,
                    end_frame=2,
                    world_position=[1.1, 0.0, 0.0],
                    metadata={
                        "raw_contact_position_refinement": {
                            "binding_candidate_class": "edge_candidate",
                            "accepted_raw_contact_sample_count": 0,
                        }
                    },
                ),
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="clamped",
                    body="LF",
                    start_frame=0,
                    end_frame=2,
                    world_position=[1.0, 0.0, 0.0],
                    surface_id="box_0_top",
                    surface_coordinates={"u": 1.0, "v": 0.0},
                    metadata={
                        "surface_bindings": [
                            {
                                "original_world_position": [2.0, 0.0, 0.0],
                                "projected_world_position": [2.0, 0.0, 0.0],
                                "bound_world_position": [1.0, 0.0, 0.0],
                                "signed_surface_distance": 0.0,
                                "raw_surface_coordinates": {"u": 2.0, "v": 0.0},
                                "surface_coordinates": {"u": 1.0, "v": 0.0},
                                "clamped": True,
                                "surface_binding_source": "test",
                            }
                        ]
                    },
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = export_surface_binding_report(Path(tmp) / "report.json", graph=graph, surfaces=[surface])
            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["summary"]["anchor_count"], 5)
        self.assertEqual(report["summary"]["bound_count"], 3)
        self.assertEqual(report["summary"]["failed_count"], 1)
        self.assertEqual(report["summary"]["unbound_count"], 1)
        self.assertEqual(report["summary"]["clamped_count"], 1)
        self.assertEqual(report["summary"]["low_confidence_count"], 1)
        by_id = {item["anchor_id"]: item for item in report["anchors"]}
        self.assertEqual(by_id["bound"]["surface_id"], "box_0_top")
        self.assertEqual(by_id["bound"]["surface_coordinates"], {"u": 0.1, "v": 0.2})
        self.assertEqual(by_id["bound"]["binding"]["projected_world_position"], [0.1, 0.2, 0.0])
        self.assertEqual(by_id["failed"]["status"], "failed")
        self.assertEqual(by_id["unbound"]["status"], "unbound")
        self.assertEqual(by_id["edge"]["status"], "suspicious")
        self.assertEqual(by_id["edge"]["binding_candidate_class"], "edge_candidate")
        self.assertEqual(by_id["clamped"]["status"], "clamped")

    def test_surface_binding_overlay_exports_quads_points_and_projection_lines(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="box_0_top",
            object_id="box_0",
            surface_type="box_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-1.0, 1.0], "v": [-0.5, 0.5]},
        )
        graph = ContactGraph(
            motion_id="motion_a",
            anchors=[
                ContactAnchorRecord(
                    motion_id="motion_a",
                    anchor_id="anchor_lf",
                    body="LF",
                    start_frame=0,
                    end_frame=2,
                    world_position=[0.1, 0.2, 0.0],
                    surface_id="box_0_top",
                    metadata={
                        "patch_role": "toe",
                        "surface_bindings": [
                            {
                                "original_world_position": [0.1, 0.2, 0.02],
                                "bound_world_position": [0.1, 0.2, 0.0],
                                "clamped": False,
                            }
                        ]
                    },
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = export_surface_binding_overlay(Path(tmp) / "overlay.json", graph=graph, surfaces=[surface])
            overlay = json.loads(out.read_text(encoding="utf-8"))

        objects = overlay["objects"]
        surface_quad = next(item for item in objects if item["type"] == "surface_quad")
        anchor_point = next(item for item in objects if item["type"] == "anchor_point")
        projection_line = next(item for item in objects if item["type"] == "projection_line")
        self.assertEqual(surface_quad["corners"][0], [-1.0, -0.5, 0.0])
        self.assertEqual(surface_quad["corners"][2], [1.0, 0.5, 0.0])
        self.assertEqual(anchor_point["status"], "bound")
        self.assertEqual(anchor_point["color"], [60, 140, 255])
        self.assertEqual(anchor_point["patch_role"], "toe")
        self.assertEqual(anchor_point["start_frame"], 0)
        self.assertEqual(anchor_point["end_frame"], 2)
        self.assertEqual(projection_line["from"], [0.1, 0.2, 0.02])
        self.assertEqual(projection_line["to"], [0.1, 0.2, 0.0])

    def test_surface_binding_overlay_prefers_mesh_polygon_corners(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="mesh_side",
            object_id="box",
            surface_type="mesh_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-10.0, 10.0], "v": [-10.0, 10.0]},
            metadata={"polygon_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]},
        )
        graph = ContactGraph(motion_id="motion_a")
        with tempfile.TemporaryDirectory() as tmp:
            out = export_surface_binding_overlay(Path(tmp) / "overlay.json", graph=graph, surfaces=[surface])
            overlay = json.loads(out.read_text(encoding="utf-8"))

        surface_quad = next(item for item in overlay["objects"] if item["type"] == "surface_quad")
        self.assertEqual(surface_quad["corners"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        self.assertEqual(surface_quad["surface_shape"], "polygon")


    def test_contact_overlay_export_writes_contact_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = contact_graph_from_masks(
                motion_id="motion_a",
                contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
                body_names=["LF", "RF"],
            )
            out = export_contact_overlay(root / "overlay.json", graph)
            overlay = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(overlay["schema_version"], 1)
        self.assertEqual(overlay["contact_graph"]["motion_id"], "motion_a")
        self.assertGreaterEqual(len(overlay["contact_graph"]["anchors"]), 1)

    def test_cutter_export_preserves_contact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            np.savez(motion, qpos=np.zeros((6, 2)))
            export_cutter_segments(root / "cutter", [_contact_segment(str(motion))])

            records = read_jsonl(root / "cutter" / "motion_a.segments.jsonl")

        self.assertEqual(records[0]["metadata"]["active_body"], "LF")
        self.assertEqual(records[0]["metadata"]["contact_transition"]["transition_id"], "motion_a_force_transition_0000")

    def test_split_npz_includes_contact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            np.savez(motion, qpos=np.zeros((6, 2)))
            written = export_split_npz(root / "split", [_contact_segment(str(motion))])

            data = np.load(written[0], allow_pickle=True)

        self.assertEqual(str(data["motion_edit_active_body"]), "LF")
        self.assertEqual(json.loads(str(data["motion_edit_support_bodies"])), ["RF"])
        self.assertEqual(str(data["motion_edit_source_anchor_id"]), "anchor_src")
        self.assertEqual(str(data["motion_edit_target_anchor_id"]), "anchor_dst")
        self.assertEqual(json.loads(str(data["motion_edit_contact_transition"]))["transition_id"], "motion_a_force_transition_0000")
        self.assertEqual(json.loads(str(data["motion_edit_old_anchor_world"])), [0.0, 0.0, 0.0])
        self.assertEqual(json.loads(str(data["motion_edit_new_anchor_world"])), [0.1, 0.0, 0.0])
        self.assertEqual(json.loads(str(data["motion_edit_delta_world"])), [0.1, 0.0, 0.0])
        self.assertEqual(json.loads(str(data["motion_edit_requested_delta_world"])), [0.1, 0.0, 0.2])
        self.assertEqual(json.loads(str(data["motion_edit_tangent_delta"])), [0.1, 0.0])
        self.assertEqual(str(data["motion_edit_surface_id"]), "platform_top")
        self.assertEqual(json.loads(str(data["motion_edit_surface_normal"])), [0.0, 0.0, 1.0])
        self.assertEqual(json.loads(str(data["motion_edit_surface_coordinates_after"])), {"u": 0.1, "v": 0.0})
        self.assertEqual(str(data["motion_edit_constraint_mode"]), "reject")
        self.assertFalse(bool(data["motion_edit_clamped"]))
        self.assertEqual(json.loads(str(data["motion_edit_affected_frames"])), [1, 4])
        self.assertEqual(json.loads(str(data["motion_edit_contact_patches"]))[0]["patch_type"], "foot")


    def test_manifest_includes_contact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            np.savez(motion, qpos=np.zeros((6, 2)))
            out = export_motion_manifest(root / "manifest.json", [_contact_segment(str(motion))])
            manifest = json.loads(out.read_text(encoding="utf-8"))

        item = manifest["motions"][0]["segments"][0]
        self.assertEqual(item["active_body"], "LF")
        self.assertEqual(item["support_bodies"], ["RF"])
        self.assertEqual(item["source_anchor_id"], "anchor_src")
        self.assertEqual(item["target_anchor_id"], "anchor_dst")
        self.assertEqual(item["old_anchor_world"], [0.0, 0.0, 0.0])
        self.assertEqual(item["new_anchor_world"], [0.1, 0.0, 0.0])
        self.assertEqual(item["delta_world"], [0.1, 0.0, 0.0])
        self.assertEqual(item["requested_delta_world"], [0.1, 0.0, 0.2])
        self.assertEqual(item["tangent_delta"], [0.1, 0.0])
        self.assertEqual(item["affected_frames"], [1, 4])
        self.assertEqual(item["surface_id"], "platform_top")
        self.assertEqual(item["surface_normal"], [0.0, 0.0, 1.0])
        self.assertEqual(item["surface_coordinates_after"], {"u": 0.1, "v": 0.0})
        self.assertEqual(item["constraint_mode"], "reject")
        self.assertFalse(item["clamped"])
        self.assertEqual(item["transition_type"], "support_transfer")
        self.assertEqual(item["contact_metadata"]["event_count"], 1)
        self.assertEqual(item["contact_metadata"]["anchor_count"], 2)
        self.assertEqual(item["contact_metadata"]["patch_count"], 1)
        self.assertEqual(item["contact_metadata"]["patches"][0]["patch_type"], "foot")
        self.assertEqual(item["contact_metadata"]["anchor_edit"]["edit_type"], "move_contact_anchor")


if __name__ == "__main__":
    unittest.main()
