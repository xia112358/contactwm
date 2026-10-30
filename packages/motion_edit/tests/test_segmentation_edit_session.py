from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from motion_edit.schema import SegmentRecord
from motion_edit.segmentation import (
    add_draft_segment,
    create_segmentation_edit_session,
    delete_draft_segment,
    discard_segmentation_edit_session,
    list_draft_segments,
    read_segmentation_edit_session,
    relabel_draft_segment,
    save_segmentation_edit_session,
    trim_draft_segment,
)
from motion_edit.storage import io as storage_io
from motion_edit.storage.schema import MotionVersionRecord
from motion_edit.storage.io import read_canonical_segments, write_canonical_segments, write_motion_version


class SegmentationEditSessionTests(unittest.TestCase):
    def _patch_storage(self, root: Path):
        return (
            mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
            mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            mock.patch.object(storage_io, "TOKENS_ROOT", root / "tokens"),
        )

    def _seed_motion_version(self, root: Path) -> list[SegmentRecord]:
        motion = root / "motion_a.npz"
        motion.write_bytes(b"placeholder")
        write_motion_version(MotionVersionRecord(motion_version_id="motion_a_raw", motion_path=str(motion)))
        segments = [
            SegmentRecord(
                motion_id="motion_a",
                segment_id="seg_0",
                start_frame=0,
                end_frame=10,
                source="auto_contact",
                status="candidate",
                motion_path=str(motion),
                metadata={"active_body": "left_foot", "support_bodies": ["right_foot"], "transition_type": "liftoff"},
            ),
            SegmentRecord(
                motion_id="motion_a",
                segment_id="seg_1",
                start_frame=10,
                end_frame=20,
                source="auto_contact",
                status="candidate",
                motion_path=str(motion),
                metadata={"active_body": "right_foot", "support_bodies": ["left_foot"], "transition_type": "touchdown"},
            ),
        ]
        write_canonical_segments("motion_a_raw", segments)
        return segments

    def test_session_edits_draft_until_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patch_storage(root)[0], self._patch_storage(root)[1], self._patch_storage(root)[2]:
                original = self._seed_motion_version(root)
                session = create_segmentation_edit_session(
                    "motion_a_raw",
                    session_id="manual_session",
                    workbench_root=root / "workbench",
                )

                edited = trim_draft_segment(
                    session,
                    segment_id="seg_0",
                    start_frame=1,
                    end_frame=9,
                    reason="tighten boundary",
                    workbench_root=root / "workbench",
                )

                self.assertEqual((edited.start_frame, edited.end_frame), (1, 9))
                self.assertEqual(read_canonical_segments("motion_a_raw")[0].start_frame, original[0].start_frame)
                self.assertEqual(list_draft_segments(session)[0].status, "manual")

                out = save_segmentation_edit_session(
                    session,
                    reason="accept manually revised segmentation",
                    workbench_root=root / "workbench",
                )
                saved = read_canonical_segments("motion_a_raw")
                saved_session = read_segmentation_edit_session("manual_session", workbench_root=root / "workbench")

                self.assertEqual(out, root / "segments" / "motion_a_raw.jsonl")
                self.assertEqual((saved[0].start_frame, saved[0].end_frame), (1, 9))
                self.assertEqual(saved[0].status, "manual")
                self.assertEqual(saved_session.state, "saved")

    def test_add_relabel_delete_and_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patch_storage(root)[0], self._patch_storage(root)[1], self._patch_storage(root)[2]:
                self._seed_motion_version(root)
                session = create_segmentation_edit_session(
                    "motion_a_raw",
                    session_id="edit_add_delete",
                    workbench_root=root / "workbench",
                )

                added = add_draft_segment(
                    session,
                    motion_id="motion_a",
                    start_frame=20,
                    end_frame=25,
                    active_body="left_hand",
                    support_bodies=["left_foot", "right_foot"],
                    transition_type="manual_reach",
                    reason="missing hand reach segment",
                    workbench_root=root / "workbench",
                )
                relabeled = relabel_draft_segment(
                    session,
                    segment_id="seg_1",
                    active_body="right_hand",
                    support_bodies=["left_foot"],
                    transition_type="support_transfer",
                    status="accepted",
                    reason="semantic correction",
                    workbench_root=root / "workbench",
                )
                deleted = delete_draft_segment(
                    session,
                    segment_id="seg_0",
                    reason="drop unstable lead-in",
                    workbench_root=root / "workbench",
                )
                discarded = discard_segmentation_edit_session(
                    session,
                    reason="test discard",
                    workbench_root=root / "workbench",
                )

                draft_ids = [segment.segment_id for segment in list_draft_segments(session)]
                canonical_ids = [segment.segment_id for segment in read_canonical_segments("motion_a_raw")]

                self.assertIn(added.segment_id, draft_ids)
                self.assertEqual(relabeled.metadata["transition_type"], "support_transfer")
                self.assertEqual(relabeled.status, "accepted")
                self.assertEqual(deleted.segment_id, "seg_0")
                self.assertNotIn("seg_0", draft_ids)
                self.assertEqual(canonical_ids, ["seg_0", "seg_1"])
                self.assertEqual(discarded.state, "discarded")


if __name__ == "__main__":
    unittest.main()
