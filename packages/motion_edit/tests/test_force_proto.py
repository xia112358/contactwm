from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motion_edit import cli
from motion_edit.contact import read_contact_graph
from motion_edit.force_proto import contact_graph_from_masked_motion, segments_from_masked_motion
from motion_edit.layers import read_layer


CONTACT_PART_ORDER_8 = np.asarray(
    ["LHEE", "LTOE", "RHEE", "RTOE", "LH", "RH", "LK", "RK"]
)


def _write_proto_motion(path: Path) -> None:
    np.savez(
        path,
        proto_start_idx=np.asarray([1]),
        proto_end_idx=np.asarray([4]),
        contact_part_mask=np.asarray(
            [
                [False, False, True, False, False, False, False, False],
                [True, False, True, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
            ]
        ),
        active_part_mask=np.asarray(
            [
                [False, False, True, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
            ]
        ),
        support_part_mask=np.asarray(
            [
                [False, False, True, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
            ]
        ),
        part_order=CONTACT_PART_ORDER_8,
    )


def _write_proto_motion_with_eight_parts(path: Path) -> None:
    contact = np.asarray(
        [
            [True, False, False, False, False, False, True, True],
            [True, True, False, False, False, False, True, True],
            [False, True, False, False, True, False, True, True],
            [False, False, False, False, True, True, True, True],
        ]
    )
    body_names = np.asarray(
        [
            "world",
            "pelvis",
            "left_ankle_roll_sphere_1_link", "left_ankle_roll_sphere_2_link",
            "left_ankle_roll_sphere_3_link", "left_ankle_roll_sphere_4_link", "left_ankle_roll_sphere_5_link",
            "right_ankle_roll_sphere_1_link", "right_ankle_roll_sphere_2_link",
            "right_ankle_roll_sphere_3_link", "right_ankle_roll_sphere_4_link", "right_ankle_roll_sphere_5_link",
            "left_sphere_hand_link", "right_sphere_hand_link",
            "left_knee_link",
            "right_knee_link",
        ]
    )
    body_pos_w = np.zeros((4, len(body_names), 3), dtype=float)
    body_pos_w[:, 0, :] = [99.0, 0.0, 0.0]
    body_pos_w[:, 1, :] = [88.0, 0.0, 0.0]
    for index in range(2, len(body_names)):
        body_pos_w[:, index, :] = [0.1 * (index - 1), 0.0, 0.0]
    np.savez(
        path,
        proto_start_idx=np.asarray([0]),
        proto_end_idx=np.asarray([4]),
        contact_part_mask=contact,
        active_part_mask=contact,
        support_part_mask=contact,
        part_order=CONTACT_PART_ORDER_8,
        body_names=body_names,
        body_pos_w=body_pos_w,
    )


def _write_proto_motion_with_positions(path: Path) -> None:
    _write_proto_motion(path)
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["body_pos_w"] = np.zeros((5, 8, 3), dtype=float)
    payload["body_pos_w"][:, 0, 0] = np.linspace(0.0, 0.8, 5)
    payload["body_pos_w"][:, 2, 0] = 1.0
    np.savez(path, **payload)


class ForceProtoContactTests(unittest.TestCase):
    def test_contact_graph_accepts_multi_proto_numpy_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_multi.npz"
            np.savez(
                path,
                proto_start_idx=np.asarray([0, 2]),
                proto_end_idx=np.asarray([2, 5]),
                contact_part_mask=np.asarray(
                    [
                        [True, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False, False],
                        [False, True, False, False, False, False, False, False],
                    ]
                ),
                part_order=CONTACT_PART_ORDER_8,
            )

            graph = contact_graph_from_masked_motion(path)

        self.assertEqual(len(graph.transitions), 2)

    def test_force_proto_without_proto_indices_derives_segments_from_event_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_events.npz"
            np.savez(
                path,
                contact_part_mask=np.asarray(
                    [
                        [True, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                    ]
                ),
                part_order=CONTACT_PART_ORDER_8,
            )

            segments = segments_from_masked_motion(path)
            graph = contact_graph_from_masked_motion(path)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (1, 2))
        self.assertEqual(len(graph.transitions), 1)

    def test_force_proto_segments_include_structured_contact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_a.npz"
            _write_proto_motion(path)

            segments = segments_from_masked_motion(path)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual((segment.start_frame, segment.end_frame), (1, 4))
        self.assertEqual(segment.contact_start, "10100000")
        self.assertEqual(segment.contact_end, "00000000")
        self.assertEqual(segment.metadata["active_body"], "left_heel")
        self.assertEqual(segment.metadata["support_bodies"], ["right_heel"])
        self.assertIn("contact_transition", segment.metadata)
        self.assertIn("contact_events", segment.metadata)
        self.assertIn("contact_anchors", segment.metadata)
        self.assertIn("contact_patches", segment.metadata)
        self.assertGreaterEqual(len(segment.metadata["contact_patches"]), 1)

    def test_force_proto_anchor_positions_are_estimated_when_body_pos_w_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_a.npz"
            _write_proto_motion_with_positions(path)

            graph = contact_graph_from_masked_motion(path)

        lf_anchor = next(anchor for anchor in graph.anchors if anchor.body == "left_heel")
        self.assertEqual(lf_anchor.world_position, [0.30000000000000004, 0.0, 0.0])
        self.assertEqual(lf_anchor.position_source, "body_pos_w_mean")
        self.assertIn("mean_drift_xy", lf_anchor.metadata)

    def test_force_proto_ignores_invalid_contact_part_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_valid_positions.npz"
            contact = np.zeros((3, 8), dtype=bool)
            contact[:, 0] = True
            positions = np.zeros((3, 8, 3), dtype=np.float32)
            positions[:, 0, 0] = [1.0, 0.0, 3.0]
            valid = np.zeros((3, 8), dtype=bool)
            valid[[0, 2], 0] = True
            np.savez(
                path,
                contact_force_part_order=CONTACT_PART_ORDER_8,
                contact_force_part_mask=contact,
                contact_force_part_position_w=positions,
                contact_force_part_position_valid=valid,
            )

            graph = contact_graph_from_masked_motion(path)

        left_heel = next(anchor for anchor in graph.anchors if anchor.body == "left_heel")
        self.assertEqual(left_heel.world_position, [2.0, 0.0, 0.0])

    def test_contact_graph_from_masked_motion_uses_same_contact_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_a.npz"
            _write_proto_motion(path)

            graph = contact_graph_from_masked_motion(path)

        self.assertEqual(graph.motion_id, "motion_a")
        self.assertEqual(len(graph.transitions), 1)
        self.assertGreaterEqual(len(graph.events), 1)
        self.assertGreaterEqual(len(graph.anchors), 1)

    def test_force_proto_preserves_canonical_eight_contact_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_parts.npz"
            _write_proto_motion_with_eight_parts(path)

            graph = contact_graph_from_masked_motion(path)
            segments = segments_from_masked_motion(path)

        bodies = {anchor.body for anchor in graph.anchors}
        self.assertIn("left_heel", bodies)
        self.assertIn("left_toe", bodies)
        self.assertIn("left_knee", bodies)
        self.assertNotIn("world", bodies)
        self.assertNotIn("pelvis", bodies)
        self.assertEqual(segments[0].contact_start, "10000011")
        self.assertEqual(segments[0].contact_end, "00001111")
        self.assertEqual(segments[0].metadata["contact_transition"]["metadata"]["body_names"], [
            "left_heel",
            "left_toe",
            "right_heel",
            "right_toe",
            "left_hand",
            "right_hand",
            "left_knee",
            "right_knee",
        ])

    def test_force_proto_part_positions_use_body_link_mapping_not_mask_column_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_parts.npz"
            _write_proto_motion_with_eight_parts(path)

            graph = contact_graph_from_masked_motion(path)

        left_heel = next(anchor for anchor in graph.anchors if anchor.body == "left_heel")
        left_toe = next(anchor for anchor in graph.anchors if anchor.body == "left_toe")
        self.assertEqual(left_heel.world_position, [0.15000000000000002, 0.0, 0.0])
        self.assertEqual(left_toe.world_position, [0.4000000000000001, 0.0, 0.0])

    def test_import_force_proto_writes_contact_layer_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            motion_path = motion_dir / "motion_a.npz"
            _write_proto_motion_with_positions(motion_path)
            original_motion_bytes = motion_path.read_bytes()
            layers_root = root / "layers"
            args = argparse.Namespace(
                motion_dir=str(motion_dir),
                pattern="*.npz",
                layer_name="force_contact",
                source="force_contact",
            )
            with (
                mock.patch.object(cli, "ensure_data_dirs"),
                mock.patch.object(cli, "LAYERS_ROOT", layers_root),
                mock.patch.object(cli, "layer_dir", lambda status, name: layers_root / "candidates" / name),
            ):
                cli._cmd_import_force_proto(args)
                list_args = argparse.Namespace(source="contact/force_contact", motion_id="motion_a", limit=1)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    cli._cmd_list_contact_layer(list_args)
                overlay_path = root / "overlay.json"
                export_args = argparse.Namespace(
                    source="contact/force_contact",
                    motion_id="motion_a",
                    output=str(overlay_path),
                )
                cli._cmd_export_contact_overlay(export_args)
                overlay_text = overlay_path.read_text(encoding="utf-8")
            segments = read_layer(
                layers_root / "candidates" / "force_contact" / "motion_a.jsonl",
                default_source="force_contact", default_status="candidate",
            )
            graph = read_contact_graph(layers_root / "contact" / "force_contact", "motion_a")
            self.assertEqual(motion_path.read_bytes(), original_motion_bytes)
            self.assertEqual(len(segments), 1)
            self.assertEqual(len(graph.transitions), 1)
            self.assertGreaterEqual(len(graph.anchors), 1)
            self.assertIn("total_anchors=", out.getvalue())
            self.assertIn("motion_a", overlay_text)


if __name__ == "__main__":
    unittest.main()
