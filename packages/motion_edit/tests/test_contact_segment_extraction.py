from __future__ import annotations

import unittest

import numpy as np

from motion_edit.contact.graph import contact_graph_from_masks
from motion_edit.storage.canonical import segments_from_contact_transitions


class ContactSegmentExtractionTests(unittest.TestCase):
    def test_stable_contact_segments_use_global_contact_anchor_clusters(self) -> None:
        contact = np.zeros((12, 2), dtype=bool)
        contact[0:3, 0] = True
        contact[7:10, 0] = True
        contact[:, 1] = True
        body_names = ["left_foot", "right_hand"]

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            body_names=body_names,
        )

        self.assertEqual(len(graph.transitions), 1)
        transition = graph.transitions[0]
        self.assertEqual(transition.start_frame, 0)
        self.assertEqual(transition.end_frame, 7)
        self.assertEqual(transition.active_body, "left_foot")
        self.assertEqual(transition.metadata["segmentation_kind"], "stable_contact_anchor")
        self.assertEqual(transition.metadata["endpoint_policy"], "global_stable_contact_cluster")
        self.assertEqual(transition.metadata["stable_anchor_frames"], [0, 7, 12])
        self.assertEqual(transition.metadata["touchdown_part"], "10")
        self.assertEqual(transition.metadata["outside_policy"], "copy_original")
        self.assertEqual(transition.metadata["owned_bodies"], ["left_foot"])

    def test_canonical_segment_metadata_keeps_stable_endpoint_semantics(self) -> None:
        contact = np.zeros((12, 2), dtype=bool)
        contact[0:3, 0] = True
        contact[7:10, 0] = True
        contact[:, 1] = True

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            body_names=["left_foot", "right_hand"],
        )
        segments = segments_from_contact_transitions(
            graph,
            motion_version_id="motion_v1",
            motion_path="motion_a.npz",
        )
        foot_segments = [segment for segment in segments if segment.active == "left_foot"]

        self.assertEqual(len(foot_segments), 1)
        segment = foot_segments[0]
        self.assertEqual(segment.start_frame, 0)
        self.assertEqual(segment.end_frame, 7)
        self.assertEqual(segment.metadata["segmentation_kind"], "stable_contact_anchor")
        self.assertEqual(segment.metadata["endpoint_policy"], "global_stable_contact_cluster")
        self.assertEqual(segment.metadata["outside_policy"], "copy_original")
        self.assertEqual(segment.metadata["owned_bodies"], ["left_foot"])
        self.assertIsNotNone(segment.metadata["source_anchor_id"])
        self.assertIsNotNone(segment.metadata["target_anchor_id"])

    def test_proto_indices_take_precedence_over_auto_stable_proto(self) -> None:
        contact = np.zeros((12, 1), dtype=bool)
        contact[0:3, 0] = True
        contact[7:10, 0] = True

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            proto_starts=[1],
            proto_ends=[5],
            body_names=["left_foot"],
        )

        self.assertEqual(len(graph.transitions), 1)
        transition = graph.transitions[0]
        self.assertEqual(transition.start_frame, 1)
        self.assertEqual(transition.end_frame, 5)
        self.assertEqual(transition.metadata["segmentation_kind"], "proto_index")

    def test_no_internal_stable_touchdown_does_not_create_stable_proto(self) -> None:
        contact = np.zeros((8, 1), dtype=bool)
        contact[2:5, 0] = True

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            body_names=["left_foot"],
        )

        self.assertEqual(len(graph.anchors), 1)
        self.assertEqual(graph.transitions, [])


if __name__ == "__main__":
    unittest.main()
