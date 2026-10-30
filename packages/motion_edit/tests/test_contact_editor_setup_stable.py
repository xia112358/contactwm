from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from motion_edit.contact import ContactSurfaceRecord, contact_graph_from_masks, read_contact_graph, write_contact_layer, write_contact_surfaces
from motion_edit.contact.schema import ContactAnchorRecord, ContactTransitionRecord


class ContactEditorStableTransitionTests(unittest.TestCase):

    def test_foot_patch_subcontacts_do_not_create_extra_cut_frames(self) -> None:
        from motion_edit.contact.graph import ContactGraph
        from motion_edit.contact.segmentation.editor_cuts import stable_proto_transitions_for_editor

        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_toe_000000_000004",
                body="right_foot",
                start_frame=0,
                end_frame=4,
                metadata={"patch_role": "toe"},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_heel_000004_000008",
                body="right_foot",
                start_frame=4,
                end_frame=8,
                metadata={"patch_role": "heel"},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_sole_000008_000012",
                body="right_foot",
                start_frame=8,
                end_frame=12,
                metadata={"patch_role": "sole"},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="left_hand_000020_000030",
                body="left_hand",
                start_frame=20,
                end_frame=30,
            ),
        ]
        transitions = stable_proto_transitions_for_editor(
            graph=ContactGraph(motion_id="motion_a", anchors=anchors),
            motion="unused.npz",
            fps=50,
            fallback=[],
        )

        self.assertEqual(len(transitions), 1)
        self.assertEqual((transitions[0].start_frame, transitions[0].end_frame), (0, 20))
        self.assertEqual(transitions[0].active_body, "left_hand")
        self.assertEqual(transitions[0].metadata["foot_patch_policy"], "heel_toe_sole_do_not_cut")
        self.assertEqual(transitions[0].metadata["contact_phase_scope"], "parent_limb_union")

    def test_close_limb_cut_candidates_merge_by_min_proto_segment_frames(self) -> None:
        from motion_edit.contact.graph import ContactGraph
        from motion_edit.contact.segmentation.editor_cuts import EditorCutConfig, stable_proto_transitions_for_editor

        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="left_foot_000000_000010",
                body="left_foot",
                start_frame=0,
                end_frame=10,
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_000012_000030",
                body="right_foot",
                start_frame=12,
                end_frame=30,
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="left_hand_000026_000040",
                body="left_hand",
                start_frame=26,
                end_frame=40,
            ),
        ]
        transitions = stable_proto_transitions_for_editor(
            graph=ContactGraph(motion_id="motion_a", anchors=anchors),
            motion="unused.npz",
            fps=50,
            fallback=[],
            cut_config=EditorCutConfig(min_proto_segment_frames=20, cluster_window=0, same_parent_body_merge_gap=0),
        )

        self.assertEqual(len(transitions), 1)
        self.assertEqual((transitions[0].start_frame, transitions[0].end_frame), (0, 26))
        self.assertEqual(transitions[0].active_body, "right_foot")
        self.assertEqual(transitions[0].metadata["active_bodies"], ["right_foot", "left_hand"])
        self.assertEqual(
            transitions[0].metadata["cluster_anchor_ids"],
            ["right_foot_000012_000030", "left_hand_000026_000040"],
        )
        self.assertEqual(transitions[0].metadata["config"]["min_proto_segment_frames"], 20)

    def test_fallback_transitions_get_contact_source_metadata(self) -> None:
        from motion_edit.contact.graph import ContactGraph
        from motion_edit.contact.segmentation.editor_cuts import stable_proto_transitions_for_editor

        fallback = [
            ContactTransitionRecord(
                motion_id="motion_a",
                transition_id="fallback_0",
                start_frame=0,
                end_frame=10,
                active_body="left_foot",
                source="stable_proto",
                metadata={},
            )
        ]
        transitions = stable_proto_transitions_for_editor(
            graph=ContactGraph(motion_id="motion_a", anchors=[]),
            motion="unused.npz",
            fps=50,
            fallback=fallback,
        )

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].metadata["contact_source"], "fallback_existing_transitions")
        self.assertEqual(transitions[0].metadata["contact_phase_scope"], "legacy_transition")
        self.assertTrue(transitions[0].metadata["cut_rebuild_failed"])
        self.assertEqual(transitions[0].metadata["cut_rebuild_reason"], "no_parent_limb_contact_starts")

    def test_same_parent_body_gap_defaults_to_stable_touchdown_cluster_window(self) -> None:
        from motion_edit.contact.graph import ContactGraph
        from motion_edit.contact.segmentation.editor_cuts import stable_proto_transitions_for_editor

        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_toe_000000_000004",
                body="right_foot_toe",
                start_frame=0,
                end_frame=4,
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="right_foot_heel_000009_000014",
                body="right_foot_heel",
                start_frame=9,
                end_frame=14,
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="left_hand_000030_000040",
                body="left_hand",
                start_frame=30,
                end_frame=40,
            ),
        ]
        transitions = stable_proto_transitions_for_editor(
            graph=ContactGraph(motion_id="motion_a", anchors=anchors),
            motion="unused.npz",
            fps=50,
            fallback=[],
        )

        self.assertEqual(len(transitions), 1)
        self.assertEqual((transitions[0].start_frame, transitions[0].end_frame), (0, 30))
        self.assertEqual(transitions[0].metadata["cluster_anchor_ids"], ["left_hand_000030_000040"])
        self.assertEqual(transitions[0].metadata["config"]["same_parent_body_merge_gap"], 6)


if __name__ == "__main__":
    unittest.main()
