from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from motion_edit import curation
from motion_edit.curation import load_layer_segments, write_status_layer
from motion_edit.io import write_jsonl
from motion_edit.layers import write_layer
from motion_edit.schema import SegmentRecord


def _segment(segment_id: str, start: int, end: int) -> SegmentRecord:
    return SegmentRecord(
        motion_id="motion_a",
        segment_id=segment_id,
        start_frame=start,
        end_frame=end,
        source="force_contact",
        status="candidate",
        motion_path="/tmp/motion_a.npz",
    )


class CurationUpsertTests(unittest.TestCase):
    def test_status_layer_upsert_preserves_existing_segments_for_same_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layers_root = Path(tmp) / "layers"
            with mock.patch.object(curation, "LAYERS_ROOT", layers_root):
                write_status_layer("accepted", "probe", [_segment("seg_a", 0, 10)])
                write_status_layer("accepted", "probe", [_segment("seg_b", 10, 20)])

                accepted = load_layer_segments("accepted/probe")

        self.assertEqual([segment.segment_id for segment in accepted], ["seg_a", "seg_b"])
        self.assertTrue(all(segment.status == "accepted" for segment in accepted))
        self.assertEqual(accepted[0].metadata["motion_edit_edits"][-1]["kind"], "curate")
        self.assertEqual(accepted[0].metadata["motion_edit_edits"][-1]["source"], "cli")
        self.assertEqual(accepted[0].metadata["motion_edit_edits"][-1]["params"]["new_status"], "accepted")

    def test_load_layer_segments_reads_candidate_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layers_root = Path(tmp) / "layers"
            path = layers_root / "candidates" / "source"
            path.mkdir(parents=True)
            write_layer(path / "motion_a.jsonl", [_segment("seg_a", 0, 10)])

            with mock.patch.object(curation, "LAYERS_ROOT", layers_root):
                loaded = load_layer_segments("candidates/source")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].segment_id, "seg_a")

    def test_load_layer_segments_infers_status_from_status_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layers_root = Path(tmp) / "layers"
            path = layers_root / "accepted" / "probe"
            write_jsonl(
                path / "motion_a.jsonl",
                [
                    {
                        "motion_id": "motion_a",
                        "segment_id": "seg_a",
                        "start_frame": 0,
                        "end_frame": 10,
                        "source": "force_contact",
                    }
                ],
            )

            with mock.patch.object(curation, "LAYERS_ROOT", layers_root):
                loaded = load_layer_segments("accepted/probe")

        self.assertEqual(loaded[0].status, "accepted")


if __name__ == "__main__":
    unittest.main()
