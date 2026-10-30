from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motion_edit import cli, paths
from motion_edit.contact import ContactSurfaceRecord, contact_graph_from_masks, read_contact_graph, write_contact_layer, write_contact_surfaces
from motion_edit.io import read_jsonl, write_jsonl
from motion_edit.layers import write_layer
from motion_edit.schema import SegmentRecord
from motion_edit.storage import io as storage_io
from motion_edit.storage import (
    MotionAssetRecord,
    MotionVersionRecord,
    TokenRecord,
    canonical_history_event_path,
    canonical_segmentation_exists,
    list_motion_assets,
    read_canonical_segments,
    read_motion_asset,
    read_motion_version,
    read_token_catalog,
    replace_canonical_segments,
    update_canonical_segments,
    upsert_motion_version_canonical_path,
    write_canonical_segments,
    write_motion_asset,
    write_motion_version,
    write_token_catalog,
)
from motion_edit.storage.segments import canonical_segment_id, get_segment_motion_version_id, with_segment_motion_version_id
from motion_edit.storage.tokens import build_tokens_from_segments


class StorageSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        # Catalog commands initialize their data roots. Keep every such write
        # inside this test's temporary directory, including newly added roots.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in ("DATA_ROOT", "CATALOGS_ROOT", "LAYERS_ROOT", "MOTIONS_ROOT",
                     "MOTION_ASSETS_ROOT", "MOTION_VERSIONS_ROOT", "SEGMENTS_ROOT",
                     "TOKENS_ROOT", "SURFACES_ROOT", "EXPORTS_ROOT", "WORKBENCH_ROOT", "BACKUPS_ROOT"):
            self.enterContext(mock.patch.object(paths, name, root / name.lower()))

    def test_motion_version_record_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "versions" / "climb00_raw.json"
            record = MotionVersionRecord(
                motion_version_id="climb00_raw",
                base_motion_id="climb00",
                kind="raw",
                motion_path="/motions/climb00.npz",
                contact_layer="contact/force_contact",
                canonical_segment_path="data/segments/climb00_raw.jsonl",
            )
            write_motion_version(record, path)
            loaded = read_motion_version("climb00_raw", path)

        self.assertEqual(loaded.motion_version_id, "climb00_raw")
        self.assertEqual(loaded.motion_path, "/motions/climb00.npz")
        self.assertEqual(loaded.contact_layer, "contact/force_contact")

    def test_upsert_motion_version_canonical_path_preserves_existing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = MotionVersionRecord(
                motion_version_id="motion_a_aug",
                motion_path="/motions/aug.npz",
                kind="augmented",
                motion_asset_id="motion_a",
                parent_motion_version_id="motion_a_raw",
                edit_plan_id="plan_a",
                token_catalog_path="data/tokens/motion_a_aug.jsonl",
                metadata={"quality": "draft"},
            )
            with mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"):
                write_motion_version(record)
                updated = upsert_motion_version_canonical_path(
                    "motion_a_aug",
                    "data/segments/motion_a_aug.jsonl",
                    contact_layer="contact/aug",
                )

        self.assertEqual(updated.kind, "augmented")
        self.assertEqual(updated.motion_asset_id, "motion_a")
        self.assertEqual(updated.parent_motion_version_id, "motion_a_raw")
        self.assertEqual(updated.edit_plan_id, "plan_a")
        self.assertEqual(updated.token_catalog_path, "data/tokens/motion_a_aug.jsonl")
        self.assertEqual(updated.metadata, {"quality": "draft"})
        self.assertEqual(updated.canonical_segment_path, "data/segments/motion_a_aug.jsonl")
        self.assertEqual(updated.contact_layer, "contact/aug")

    def test_motion_asset_record_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assets" / "climb00.json"
            record = MotionAssetRecord(
                motion_asset_id="climb00",
                motion_path="/motions/climb00.npz",
                motion_id="climb_00_z_scale_1.0",
                terrain_urdf="/terrain/climb_00.urdf",
                surface_catalog_path="data/surfaces/climb_00.jsonl",
                derived={"contact_layer": "contact/climb00_raw"},
                source="local",
                fps=50.0,
            )
            write_motion_asset(record, path)
            loaded = read_motion_asset("climb00", path)
            listed = list_motion_assets(path.parent)

        self.assertEqual(loaded.motion_asset_id, "climb00")
        self.assertEqual(loaded.motion_path, "/motions/climb00.npz")
        self.assertEqual(loaded.motion_id, "climb_00_z_scale_1.0")
        self.assertEqual(loaded.terrain_urdf, "/terrain/climb_00.urdf")
        self.assertEqual(loaded.derived["contact_layer"], "contact/climb00_raw")
        self.assertEqual(loaded.fps, 50.0)
        self.assertEqual([item.motion_asset_id for item in listed], ["climb00"])

    def test_canonical_segments_write_read_adds_motion_version_id(self) -> None:
        segment = SegmentRecord(
            motion_id="climb00",
            segment_id="seg_0",
            start_frame=1,
            end_frame=5,
            source="force_contact",
            motion_path="/motions/climb00.npz",
            metadata={"cut_source": "contact_auto"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segments" / "climb00_raw.jsonl"
            written = write_canonical_segments("climb00_raw", [segment], path)
            loaded = read_canonical_segments("climb00_raw", path)

        self.assertEqual(written.name, "climb00_raw.jsonl")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].segment_id, "seg_0")
        self.assertEqual(loaded[0].metadata["motion_version_id"], "climb00_raw")

    def test_replace_and_update_canonical_segments_use_single_active_file_and_history(self) -> None:
        original = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
        )
        replacement = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_1",
            start_frame=2,
            end_frame=4,
            source="canonical",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"):
                self.assertFalse(canonical_segmentation_exists("motion_a_raw"))
                write_canonical_segments("motion_a_raw", [original])
                self.assertTrue(canonical_segmentation_exists("motion_a_raw"))
                replace_canonical_segments(
                    "motion_a_raw",
                    [replacement],
                    reason="reset for test",
                    source="test",
                    kind="reset",
                )
                loaded = read_canonical_segments("motion_a_raw")
                update_canonical_segments(
                    "motion_a_raw",
                    lambda segments: [SegmentRecord(**{**segments[0].__dict__, "status": "manual"})],
                    reason="manual update",
                    source="test",
                    kind="manual_update",
                )
                updated = read_canonical_segments("motion_a_raw")
                active_files = sorted(path.name for path in (root / "segments").glob("motion_a_raw*.jsonl"))
                backup_files = sorted(path.name for path in (root / "segments" / "history" / "motion_a_raw").glob("*.jsonl"))
                history_events = read_jsonl(canonical_history_event_path("motion_a_raw"))

        self.assertEqual([segment.segment_id for segment in loaded], ["seg_1"])
        self.assertEqual(updated[0].status, "manual")
        self.assertEqual(active_files, ["motion_a_raw.jsonl"])
        self.assertEqual(backup_files, ["000000.jsonl", "000002.jsonl"])
        self.assertEqual([event["kind"] for event in history_events], ["backup_canonical_segmentation", "reset", "backup_canonical_segmentation", "manual_update"])

    def test_segment_motion_version_helpers(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
        )

        updated = with_segment_motion_version_id(segment, "motion_a_raw")

        self.assertIsNone(get_segment_motion_version_id(segment))
        self.assertEqual(get_segment_motion_version_id(updated), "motion_a_raw")
        self.assertEqual(canonical_segment_id("motion_a_raw", index=3), "motion_a_raw_seg_0003")
        self.assertEqual(canonical_segment_id("motion_a_raw", transition_id="transition_0"), "motion_a_raw_transition_0")

    def test_token_catalog_references_segments_without_motion_arrays(self) -> None:
        token = TokenRecord(
            token_id="tok_0",
            motion_version_id="climb00_raw",
            segment_id="seg_0",
            token_family="LF_to_RF_support_transfer",
            active_body="LF",
            support_bodies=["RF"],
            continuous_params={"duration": 4},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens" / "climb00_raw.jsonl"
            write_token_catalog("climb00_raw", [token], path)
            loaded = read_token_catalog("climb00_raw", path)

        self.assertEqual(loaded[0].motion_version_id, "climb00_raw")
        self.assertEqual(loaded[0].segment_id, "seg_0")
        self.assertNotIn("qpos", loaded[0].metadata)
        self.assertNotIn("motion", loaded[0].continuous_params)


    def test_canonical_change_marks_token_catalog_stale(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
            metadata={"motion_version_id": "motion_a_raw"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = MotionVersionRecord(
                motion_version_id="motion_a_raw",
                motion_path="/motions/motion_a.npz",
                token_catalog_path=str(root / "tokens" / "motion_a_raw.jsonl"),
                metadata={"token_catalog_status": "current"},
            )
            with (
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            ):
                write_motion_version(version)
                replace_canonical_segments(
                    "motion_a_raw",
                    [segment],
                    reason="manual edit",
                    source="test",
                    backup_existing=False,
                )
                loaded = read_motion_version("motion_a_raw")

        self.assertEqual(loaded.metadata["token_catalog_status"], "stale")
        self.assertEqual(loaded.metadata["token_catalog_stale_reason"], "canonical_segmentation_updated")

    def test_build_tokens_from_segments_does_not_copy_motion_arrays(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
            motion_path="/motions/motion_a.npz",
            metadata={"active_body": "LF", "support_bodies": ["RF"]},
        )

        tokens = build_tokens_from_segments("motion_a_raw", [segment])

        self.assertEqual(tokens[0].segment_id, "seg_0")
        self.assertNotIn("motion_path", tokens[0].to_dict())

    def test_ensure_data_dirs_includes_canonical_storage_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "SURFACES_ROOT", root / "surfaces"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
            ):
                paths.ensure_data_dirs()

            self.assertTrue((root / "motions" / "raw").is_dir())
            self.assertTrue((root / "motions" / "generated").is_dir())
            self.assertTrue((root / "motion_assets").is_dir())
            self.assertTrue((root / "motion_versions").is_dir())
            self.assertTrue((root / "segments").is_dir())
            self.assertTrue((root / "tokens").is_dir())
            self.assertTrue((root / "exports" / "split_npz").is_dir())

    def test_build_token_catalog_from_canonical_segments(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=2,
            end_frame=8,
            source="canonical",
            status="accepted",
            metadata={
                "motion_version_id": "motion_a_raw",
                "active_body": "LF",
                "support_bodies": ["RF"],
                "transition_type": "support_transfer",
                "source_anchor_id": "anchor_src",
                "target_anchor_id": "anchor_dst",
                "parent_transition_id": "transition_0",
                "contact_anchor_edit": {"tangent_delta": [0.1, 0.0]},
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(storage_io, "TOKENS_ROOT", root / "tokens"),
            ):
                write_canonical_segments("motion_a_raw", [segment])
                cli._cmd_build_token_catalog(type("Args", (), {"motion_version_id": "motion_a_raw", "output": None})())
                tokens = read_token_catalog("motion_a_raw")

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].motion_version_id, "motion_a_raw")
        self.assertEqual(tokens[0].segment_id, "seg_0")
        self.assertEqual(tokens[0].token_family, "LF__RF__support_transfer")
        self.assertEqual(tokens[0].continuous_params["duration"], 6)
        self.assertEqual(tokens[0].continuous_params["tangent_delta"], [0.1, 0.0])
        self.assertNotIn("qpos", tokens[0].metadata)

    def test_build_token_catalog_updates_motion_version_record(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
            metadata={"motion_version_id": "motion_a_raw"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = MotionVersionRecord(motion_version_id="motion_a_raw", motion_path="/motions/motion_a.npz")
            with (
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(storage_io, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            ):
                write_motion_version(version)
                write_canonical_segments("motion_a_raw", [segment])
                cli._cmd_build_token_catalog(type("Args", (), {"motion_version_id": "motion_a_raw", "output": None})())
                loaded = read_motion_version("motion_a_raw")

        self.assertEqual(loaded.token_catalog_path, str(root / "tokens" / "motion_a_raw.jsonl"))
        self.assertEqual(loaded.metadata["token_catalog_status"], "current")
        self.assertNotIn("token_catalog_stale_reason", loaded.metadata)

    def test_register_motion_asset_cli_writes_reference_without_copying_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            motion.write_bytes(b"source-motion")
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(storage_io, "MOTIONS_ROOT", root / "motions"),
            ):
                cli._cmd_register_motion_asset(
                    type(
                        "Args",
                        (),
                        {
                            "motion_asset_id": "motion_a",
                            "motion": str(motion),
                            "fps": 50.0,
                            "source": "local",
                        },
                    )()
                )
                loaded = read_motion_asset("motion_a")

            self.assertEqual(motion.read_bytes(), b"source-motion")
            self.assertTrue((root / "motions" / "motion_a.json").exists())
            self.assertFalse((root / "motions" / "raw" / "motion_a.npz").exists())

    def test_register_motion_cli_writes_rollout_motion_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            terrain = root / "terrain.urdf"
            surface = root / "surface.jsonl"
            motion.write_bytes(b"source-motion")
            terrain.write_text("<robot/>", encoding="utf-8")
            surface.write_text("", encoding="utf-8")
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTIONS_ROOT", root / "motions"),
            ):
                args = cli.build_parser().parse_args(
                    [
                        "register-motion",
                        "--motion-id",
                        "motion_a",
                        "--motion",
                        str(motion),
                        "--rollout-motion-id",
                        "climb_00_z_scale_1.0",
                        "--terrain-urdf",
                        str(terrain),
                        "--surface-catalog",
                        str(surface),
                        "--contact-layer",
                        "contact/motion_a_raw",
                        "--bound-contact-layer",
                        "contact/motion_a_bound",
                        "--edit-plan",
                        "data/workbench/motion_a_edits.json",
                        "--output-contact-layer",
                        "contact/motion_a_edited",
                        "--raw-contact",
                        "--raw-contact-source",
                        "newton_raw_rigid_contacts",
                    ]
                )
                args.func(args)
                loaded = read_motion_asset("motion_a")

        self.assertEqual(loaded.motion_path, str(motion))
        self.assertEqual(loaded.motion_id, "climb_00_z_scale_1.0")
        self.assertEqual(loaded.terrain_urdf, str(terrain))
        self.assertEqual(loaded.surface_catalog_path, str(surface))
        self.assertTrue(loaded.raw_contact["available"])
        self.assertEqual(loaded.derived["contact_layer"], "contact/motion_a_raw")
        self.assertEqual(loaded.derived["bound_contact_layer"], "contact/motion_a_bound")

    def test_import_force_proto_uses_registered_motion_id_and_updates_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "motions_src"
            motion_dir.mkdir()
            motion = motion_dir / "climb_01_rollout_ref_contact_force.npz"
            np.savez(
                motion,
                proto_start_idx=np.asarray([0], dtype=np.int64),
                proto_end_idx=np.asarray([3], dtype=np.int64),
                contact_part_mask=np.asarray(
                    [
                        [True, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                        [False, True, False, False, False, False, False, False],
                    ],
                    dtype=bool,
                ),
                active_part_mask=np.asarray(
                    [
                        [True, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                        [False, True, False, False, False, False, False, False],
                    ],
                    dtype=bool,
                ),
                support_part_mask=np.asarray(
                    [
                        [False, True, False, False, False, False, False, False],
                        [False, True, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                    ],
                    dtype=bool,
                ),
            )
            record = MotionAssetRecord(
                motion_asset_id="climb_01_raw_contact_29",
                motion_id="climb_01_z_scale_1.0",
                motion_path=str(motion),
                raw_contact={"available": True, "source": "newton_raw_rigid_contacts"},
            )
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(cli, "LAYERS_ROOT", root / "layers"),
            ):
                write_motion_asset(record)
                cli._cmd_import_force_proto(
                    type(
                        "Args",
                        (),
                        {
                            "motion_dir": str(motion_dir),
                            "pattern": "*.npz",
                            "layer_name": "raw_contact_29",
                            "source": "force_contact",
                            "use_registered_motion_ids": True,
                            "update_motions": True,
                        },
                    )()
                )
                graph = read_contact_graph(root / "layers" / "contact" / "raw_contact_29", "climb_01_z_scale_1.0")
                updated = read_motion_asset("climb_01_raw_contact_29")

        self.assertEqual(graph.motion_id, "climb_01_z_scale_1.0")
        self.assertEqual(updated.derived["contact_layer"], "contact/raw_contact_29")

    def test_import_force_proto_reads_contact_force_part_mask_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "motions_src"
            motion_dir.mkdir()
            motion = motion_dir / "climb_02_rollout_ref_contact_force.npz"
            np.savez(
                motion,
                contact_force_part_order=np.asarray(["LHEE", "LTOE", "RHEE", "RTOE", "LH", "RH", "LK", "RK"]),
                contact_force_part_mask=np.asarray(
                    [
                        [True, False, False, False, False, False, False, False],
                        [True, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False],
                    ],
                    dtype=bool,
                ),
                contact_force_part_position_w=np.asarray(
                    [
                        [[0.0, 0.0, 0.0]] * 8,
                        [[0.1, 0.0, 0.0]] * 8,
                        [[0.0, 0.0, 0.0]] * 8,
                    ],
                    dtype=np.float32,
                ),
            )
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(cli, "LAYERS_ROOT", root / "layers"),
            ):
                cli._cmd_import_force_proto(
                    type(
                        "Args",
                        (),
                        {
                            "motion_dir": str(motion_dir),
                            "pattern": "*.npz",
                            "layer_name": "raw_contact_alias",
                            "source": "force_contact",
                            "use_registered_motion_ids": False,
                            "update_motions": False,
                        },
                    )()
                )
                graph = read_contact_graph(root / "layers" / "contact" / "raw_contact_alias", "climb_02_rollout_ref_contact_force")

        self.assertGreater(len(graph.anchors), 0)
        self.assertEqual(graph.anchors[0].body, "left_heel")

    def test_register_motion_version_cli_writes_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "motion_a.npz"
            motion.write_bytes(b"source-motion")
            with (
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "LAYERS_ROOT", root / "layers"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "MOTION_ASSETS_ROOT", root / "motion_assets"),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            ):
                cli._cmd_register_motion_version(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "motion": str(motion),
                            "kind": "raw",
                            "base_motion_id": "motion_a",
                            "motion_asset_id": "motion_a",
                            "parent_motion_version_id": None,
                            "edit_plan_id": None,
                            "contact_layer": "contact/force_contact",
                        },
                    )()
                )
                loaded = read_motion_version("motion_a_raw")

            self.assertEqual(motion.read_bytes(), b"source-motion")
            self.assertFalse((root / "motions" / "generated" / "motion_a_raw.npz").exists())

        self.assertEqual(loaded.motion_version_id, "motion_a_raw")
        self.assertEqual(loaded.motion_asset_id, "motion_a")
        self.assertEqual(loaded.contact_layer, "contact/force_contact")

    def test_build_canonical_segmentation_from_contact_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers_root = root / "layers"
            graph = contact_graph_from_masks(
                motion_id="motion_a",
                contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
                body_names=["LF", "RF"],
            )
            write_contact_layer(layers_root / "contact" / "force_contact", graph)
            motion = root / "motion_a.npz"
            motion.write_bytes(b"npz")
            args = type(
                "Args",
                (),
                {
                    "motion_version_id": "motion_a_raw",
                    "motion": str(motion),
                    "motion_id": "motion_a",
                    "contact_layer": "contact/force_contact",
                    "source": None,
                    "cut_source": "contact_auto",
                    "reset_canonical": False,
                    "overwrite": False,
                    "reason": None,
                },
            )()
            with (
                mock.patch.object(cli, "LAYERS_ROOT", layers_root),
                mock.patch.object(paths, "LAYERS_ROOT", layers_root),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
            ):
                cli._cmd_build_canonical_segmentation(args)
                segments = read_canonical_segments("motion_a_raw")
                version = read_motion_version("motion_a_raw")

            self.assertEqual(version.motion_version_id, "motion_a_raw")
            self.assertEqual(version.contact_layer, "contact/force_contact")
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].metadata["motion_version_id"], "motion_a_raw")
            self.assertEqual(segments[0].metadata["cut_source"], "contact_auto")
            self.assertIn("parent_transition_id", segments[0].metadata)
            self.assertEqual(len(list((root / "segments").glob("motion_a_raw*.jsonl"))), 1)

    def test_build_canonical_segmentation_refuses_existing_without_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers_root = root / "layers"
            graph = contact_graph_from_masks(
                motion_id="motion_a",
                contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
                body_names=["LF", "RF"],
            )
            write_contact_layer(layers_root / "contact" / "force_contact", graph)
            motion = root / "motion_a.npz"
            motion.write_bytes(b"npz")
            base_args = {
                "motion_version_id": "motion_a_raw",
                "motion": str(motion),
                "motion_id": "motion_a",
                "contact_layer": "contact/force_contact",
                "source": None,
                "cut_source": "contact_auto",
                "reset_canonical": False,
                "overwrite": False,
                "reason": None,
            }
            with (
                mock.patch.object(cli, "LAYERS_ROOT", layers_root),
                mock.patch.object(paths, "LAYERS_ROOT", layers_root),
                mock.patch.object(paths, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(paths, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(paths, "MOTIONS_ROOT", root / "motions"),
                mock.patch.object(paths, "TOKENS_ROOT", root / "tokens"),
                mock.patch.object(paths, "EXPORTS_ROOT", root / "exports"),
                mock.patch.object(paths, "CATALOGS_ROOT", root / "catalogs"),
                mock.patch.object(paths, "WORKBENCH_ROOT", root / "workbench"),
                mock.patch.object(paths, "BACKUPS_ROOT", root / "backups"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
            ):
                cli._cmd_build_canonical_segmentation(type("Args", (), base_args)())
                with self.assertRaises(ValueError):
                    cli._cmd_build_canonical_segmentation(type("Args", (), base_args)())
                reset_args = {**base_args, "reset_canonical": True, "reason": "reset test"}
                cli._cmd_build_canonical_segmentation(type("Args", (), reset_args)())
                active_files = sorted(path.name for path in (root / "segments").glob("motion_a_raw*.jsonl"))
                backup_files = sorted(path.name for path in (root / "segments" / "history" / "motion_a_raw").glob("*.jsonl"))
                history_events = read_jsonl(canonical_history_event_path("motion_a_raw"))

        self.assertEqual(active_files, ["motion_a_raw.jsonl"])
        self.assertGreaterEqual(len(backup_files), 1)
        self.assertIn("reset_canonical_segmentation", [event["kind"] for event in history_events])

    def test_mark_segment_status_updates_canonical_segmentation(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
            status="candidate",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = MotionVersionRecord(
                motion_version_id="motion_a_raw",
                motion_path="/motions/motion_a.npz",
                token_catalog_path=str(root / "tokens" / "motion_a_raw.jsonl"),
                metadata={"token_catalog_status": "current"},
            )
            with (
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            ):
                write_motion_version(version)
                write_canonical_segments("motion_a_raw", [segment])
                cli._cmd_mark_segment_status(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "segment_id": ["seg_0"],
                            "status": "accepted",
                            "reason": "good_contact",
                            "source": "test",
                            "status_by_file": None,
                        },
                    )()
                )
                loaded = read_canonical_segments("motion_a_raw")
                active_files = sorted(path.name for path in (root / "segments").glob("motion_a_raw*.jsonl"))
                history_events = read_jsonl(canonical_history_event_path("motion_a_raw"))
                loaded_version = read_motion_version("motion_a_raw")

            self.assertFalse((root / "layers" / "accepted").exists())

        self.assertEqual(active_files, ["motion_a_raw.jsonl"])
        self.assertIn("mark_status", [event["kind"] for event in history_events])
        self.assertEqual(loaded_version.metadata["token_catalog_status"], "stale")
        self.assertEqual(loaded[0].status, "accepted")
        self.assertEqual(loaded[0].metadata["status_history"][0]["old_status"], "candidate")
        self.assertEqual(loaded[0].metadata["status_history"][0]["new_status"], "accepted")
        self.assertEqual(loaded[0].metadata["status_history"][0]["reason"], "good_contact")
        self.assertEqual(loaded[0].metadata["status_history"][0]["source"], "test")

    def test_mark_segment_status_updates_multiple_segments_from_file(self) -> None:
        segments = [
            SegmentRecord(motion_id="motion_a", segment_id="seg_0", start_frame=0, end_frame=2, source="canonical"),
            SegmentRecord(motion_id="motion_a", segment_id="seg_1", start_frame=2, end_frame=4, source="canonical"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            updates = root / "updates.jsonl"
            write_jsonl(
                updates,
                [
                    {"segment_id": "seg_0", "status": "accepted", "reason": "clean", "source": "batch"},
                    {"segment_id": "seg_1", "status": "rejected", "reason": "bad", "source": "batch"},
                ],
            )
            with mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"):
                write_canonical_segments("motion_a_raw", segments)
                cli._cmd_mark_segment_status(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "segment_id": None,
                            "status": None,
                            "reason": None,
                            "source": "mark_segment_status",
                            "status_by_file": str(updates),
                        },
                    )()
                )
                loaded = read_canonical_segments("motion_a_raw")

        self.assertEqual([segment.status for segment in loaded], ["accepted", "rejected"])
        self.assertEqual(loaded[0].metadata["status_history"][0]["source"], "batch")
        self.assertEqual(loaded[1].metadata["status_history"][0]["reason"], "bad")

    def test_canonical_action_trim_split_delete_updates_same_canonical_path(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=10,
            source="canonical",
            status="candidate",
            metadata={"motion_version_id": "motion_a_raw"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"):
                write_canonical_segments("motion_a_raw", [segment])
                cli._cmd_canonical_action(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "segment_id": "seg_0",
                            "motion_id": None,
                            "index": None,
                            "action": "trim",
                            "start_frame": 2,
                            "end_frame": 8,
                            "frame": None,
                            "reason": "tighten bounds",
                            "source": "test",
                        },
                    )()
                )
                trimmed = read_canonical_segments("motion_a_raw")
                cli._cmd_canonical_action(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "segment_id": "seg_0",
                            "motion_id": None,
                            "index": None,
                            "action": "split",
                            "start_frame": None,
                            "end_frame": None,
                            "frame": 5,
                            "reason": "split transition",
                            "source": "test",
                        },
                    )()
                )
                split = read_canonical_segments("motion_a_raw")
                cli._cmd_canonical_action(
                    type(
                        "Args",
                        (),
                        {
                            "motion_version_id": "motion_a_raw",
                            "segment_id": split[0].segment_id,
                            "motion_id": None,
                            "index": None,
                            "action": "delete",
                            "start_frame": None,
                            "end_frame": None,
                            "frame": None,
                            "reason": "remove bad child",
                            "source": "test",
                        },
                    )()
                )
                deleted = read_canonical_segments("motion_a_raw")
                active_files = sorted(path.name for path in (root / "segments").glob("motion_a_raw*.jsonl"))
                history_events = read_jsonl(canonical_history_event_path("motion_a_raw"))

        self.assertEqual((trimmed[0].start_frame, trimmed[0].end_frame), (2, 8))
        self.assertEqual(len(split), 2)
        self.assertEqual(len(deleted), 1)
        self.assertEqual(active_files, ["motion_a_raw.jsonl"])
        self.assertIn("trim", [event["kind"] for event in history_events])
        self.assertIn("split", [event["kind"] for event in history_events])
        self.assertIn("delete", [event["kind"] for event in history_events])

    def test_bind_contact_surfaces_can_rebind_canonical_and_mark_tokens_stale(self) -> None:
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="seg_0",
            start_frame=0,
            end_frame=2,
            source="canonical",
            status="candidate",
            metadata={"motion_version_id": "motion_a_raw"},
        )
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers_root = root / "layers"
            graph = contact_graph_from_masks(
                motion_id="motion_a",
                contact_mask=np.asarray([[True], [False], [True]]),
                body_pos_w=np.asarray([[[0.1, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]]]),
                body_names=["LF"],
            )
            write_contact_layer(layers_root / "contact" / "force_contact", graph)
            surface_catalog = root / "surfaces.jsonl"
            write_contact_surfaces(surface_catalog, [surface])
            version = MotionVersionRecord(
                motion_version_id="motion_a_raw",
                motion_path="/motions/motion_a.npz",
                contact_layer="contact/force_contact",
                token_catalog_path=str(root / "tokens" / "motion_a_raw.jsonl"),
                metadata={"token_catalog_status": "current"},
            )
            with (
                mock.patch.object(cli, "LAYERS_ROOT", layers_root),
                mock.patch.object(storage_io, "SEGMENTS_ROOT", root / "segments"),
                mock.patch.object(storage_io, "MOTION_VERSIONS_ROOT", root / "motion_versions"),
            ):
                write_motion_version(version)
                write_canonical_segments("motion_a_raw", [segment])
                cli._cmd_bind_contact_surfaces(
                    type(
                        "Args",
                        (),
                        {
                            "contact_layer": None,
                            "motion_id": "motion_a",
                            "surface_catalog": str(surface_catalog),
                            "output_contact_layer": "contact/force_contact_bound",
                            "max_distance": 0.05,
                            "mode": "reject",
                            "motion_version_id": "motion_a_raw",
                            "update_motion_version": True,
                            "rebind_canonical_segments": True,
                        },
                    )()
                )
                loaded = read_canonical_segments("motion_a_raw")
                loaded_version = read_motion_version("motion_a_raw")
                history_events = read_jsonl(canonical_history_event_path("motion_a_raw"))

        self.assertEqual(loaded_version.contact_layer, "contact/force_contact_bound")
        self.assertEqual(loaded_version.metadata["token_catalog_status"], "stale")
        self.assertIn("surface_binding_rebind", [event["kind"] for event in history_events])
        self.assertIn("contact_transition", loaded[0].metadata)


if __name__ == "__main__":
    unittest.main()
