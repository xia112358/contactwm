from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_edit.storage import MotionAssetRecord, read_motion_asset, write_motion_asset
from motion_edit.workbench.recent import (
    RecentMotionEntry,
    clear_recent_motions,
    read_recent_motions,
    upsert_recent_motion,
)


class MotionAssetBundleTests(unittest.TestCase):
    def test_motion_asset_promotes_legacy_derived_bundle_fields(self) -> None:
        record = MotionAssetRecord(
            motion_asset_id="climb00",
            motion_path="/motions/climb00.npz",
            motion_id="climb00_motion",
            terrain_urdf="/terrain/climb00.urdf",
            surface_catalog_path="data/surfaces/climb00.jsonl",
            derived={
                "contact_layer": "contact/climb00_raw",
                "bound_contact_layer": "contact/climb00_ready",
                "edit_plan_path": "data/workbench/climb00/plan.json",
                "output_contact_layer": "contact/climb00_edited",
                "output_segment_layer": "candidates/climb00_edited",
            },
        )

        self.assertEqual(record.contact_layer, "contact/climb00_raw")
        self.assertEqual(record.bound_contact_layer, "contact/climb00_ready")
        self.assertEqual(record.source_contact_layer, "contact/climb00_ready")
        self.assertEqual(record.edit_plan_path, "data/workbench/climb00/plan.json")
        self.assertEqual(record.output_contact_layer, "contact/climb00_edited")
        self.assertEqual(record.output_segment_layer, "candidates/climb00_edited")
        self.assertEqual(record.missing_contact_editor_fields(), [])

    def test_motion_asset_writes_top_level_bundle_fields_and_backfills_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motions" / "climb00.json"
            record = MotionAssetRecord(
                motion_asset_id="climb00",
                motion_path="/motions/climb00.npz",
                motion_id="climb00_motion",
                surface_catalog_path="data/surfaces/climb00.jsonl",
                bound_contact_layer="contact/climb00_ready",
                edit_plan_path="data/workbench/climb00/plan.json",
                output_contact_layer="contact/climb00_edited",
                output_segment_layer="candidates/climb00_edited",
            )
            write_motion_asset(record, path)
            loaded = read_motion_asset("climb00", path)
            payload = loaded.to_dict()

        self.assertEqual(loaded.source_contact_layer, "contact/climb00_ready")
        self.assertEqual(payload["bound_contact_layer"], "contact/climb00_ready")
        self.assertEqual(payload["edit_plan_path"], "data/workbench/climb00/plan.json")
        self.assertEqual(payload["output_contact_layer"], "contact/climb00_edited")
        self.assertEqual(payload["output_segment_layer"], "candidates/climb00_edited")
        self.assertEqual(payload["derived"]["bound_contact_layer"], "contact/climb00_ready")
        self.assertEqual(payload["derived"]["output_segment_layer"], "candidates/climb00_edited")

    def test_recent_motion_tracks_motion_asset_identity_and_editor_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion.npz"
            motion.write_bytes(b"npz")
            asset = root / "motion_asset.json"
            asset.write_text("{}", encoding="utf-8")
            cache = root / "recent.json"
            entry = RecentMotionEntry(
                label="motion",
                motion_path=str(motion),
                motion_id="motion",
                motion_ref_id="motion_asset",
                motion_asset_id="motion_asset",
                motion_asset_path=str(asset),
                terrain_urdf="terrain.urdf",
                surface_catalog="surfaces.jsonl",
                contact_layer="contact/ready",
                edit_plan_path="plan.json",
                output_contact_layer="contact/out",
                output_segment_layer="candidates/out",
            )
            upsert_recent_motion(entry, cache)
            loaded = read_recent_motions(cache)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].key(), "motion_asset")
        self.assertEqual(loaded[0].motion_asset_id, "motion_asset")
        self.assertEqual(loaded[0].surface_catalog, "surfaces.jsonl")
        self.assertEqual(loaded[0].output_segment_layer, "candidates/out")

    def test_recent_motion_keeps_source_and_generated_version_as_peers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_motion = root / "source.npz"
            generated_motion = root / "generated.npz"
            source_motion.touch()
            generated_motion.touch()
            cache = root / "recent.json"
            source = RecentMotionEntry(
                label="climb",
                motion_path=str(source_motion),
                motion_id="climb",
                motion_ref_id="climb-asset",
                motion_asset_id="climb-asset",
            )
            generated = RecentMotionEntry(
                label="climb · Edited",
                motion_path=str(generated_motion),
                motion_id="climb",
                motion_ref_id="climb-asset-edited",
                motion_asset_id="climb-asset",
                motion_version_id="climb-asset-edited",
                metadata={"kind": "generated"},
            )
            upsert_recent_motion(source, cache)
            upsert_recent_motion(generated, cache)
            loaded = read_recent_motions(cache)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].key(), "climb-asset-edited")
        self.assertEqual(loaded[1].motion_asset_id, "climb-asset")

    def test_new_editor_session_clears_previous_recent_motions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "source.npz"
            motion.touch()
            cache = root / "recent.json"
            upsert_recent_motion(
                RecentMotionEntry(
                    label="old",
                    motion_path=str(motion),
                    motion_id="old",
                    motion_asset_id="old-asset",
                ),
                cache,
            )

            clear_recent_motions(cache)

            self.assertEqual(read_recent_motions(cache), [])
            self.assertTrue(cache.exists())


if __name__ == "__main__":
    unittest.main()
