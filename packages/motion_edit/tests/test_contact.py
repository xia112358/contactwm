from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motion_edit import cli
from motion_edit.contact import (
    anchors_from_contact_mask,
    bind_anchor_to_plane,
    bind_anchor_to_surface,
    bind_anchors_to_surfaces,
    bind_segment_to_contact_graph,
    contact_graph_from_masks,
    detect_contact_events,
    make_anchor_move_edit,
    move_contact_anchor,
    move_contact_anchor_on_surface,
    move_anchor_in_graph,
    move_anchor_in_contact_layer,
    merge_nearby_contact_anchors,
    filter_short_raw_missing_anchors,
    read_contact_anchors,
    read_contact_events,
    read_contact_graph,
    read_contact_jsonl,
    read_contact_patches,
    read_contact_surfaces,
    read_contact_transitions,
    refine_contact_graph_anchor_positions_from_raw_contacts,
    segment_from_contact_transition,
    split_foot_contact_anchors,
    surface_compatible_with_body,
    transitions_from_proto_indices,
    write_contact_surfaces,
    write_contact_jsonl,
    write_contact_layer,
)
from motion_edit.contact.schema import ContactAnchorEditRecord, ContactAnchorRecord, ContactSurfaceRecord
from motion_edit.contact.raw_contacts import RawContactMotion
from motion_edit.schema import SegmentRecord


class ContactEventTests(unittest.TestCase):
    def test_anchor_world_position_roundtrips_jsonl(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.1, 0.2, 0.3],
            object_position=[0.0, 0.2, 0.3],
            object_id="terrain",
            normal=[0.0, 0.0, 1.0],
            surface_id="platform_top",
            surface_type="box_face",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[0.0, 0.0, 0.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-1.0, 1.0], "v": [-0.5, 0.5]},
            surface_coordinates={"u": 0.1, "v": 0.2},
            surface_binding_source="manual",
            editable=True,
            position_source="manual",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anchors.jsonl"
            write_contact_jsonl(path, [anchor])
            loaded = read_contact_anchors(path)[0]

        self.assertEqual(loaded.world_position, [0.1, 0.2, 0.3])
        self.assertEqual(loaded.object_position, [0.0, 0.2, 0.3])
        self.assertEqual(loaded.normal, [0.0, 0.0, 1.0])
        self.assertEqual(loaded.surface_id, "platform_top")
        self.assertEqual(loaded.surface_type, "box_face")
        self.assertEqual(loaded.surface_normal, [0.0, 0.0, 1.0])
        self.assertEqual(loaded.surface_bounds, {"u": [-1.0, 1.0], "v": [-0.5, 0.5]})
        self.assertEqual(loaded.surface_coordinates, {"u": 0.1, "v": 0.2})
        self.assertEqual(loaded.surface_binding_source, "manual")
        self.assertTrue(loaded.editable)
        self.assertEqual(loaded.position_source, "manual")

    def test_contact_surface_record_roundtrips_jsonl(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="box_0_top",
            object_id="box_0",
            surface_type="box_face",
            origin=[1.0, 0.0, 0.8],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-0.25, 0.25], "v": [-0.25, 0.25]},
            source="manual_surface_catalog",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surfaces.jsonl"
            write_contact_surfaces(path, [surface])
            loaded = read_contact_surfaces(path)[0]

        self.assertEqual(loaded.surface_id, "box_0_top")
        self.assertEqual(loaded.object_id, "box_0")
        self.assertEqual(loaded.normal, [0.0, 0.0, 1.0])
        self.assertEqual(loaded.bounds, {"u": [-0.25, 0.25], "v": [-0.25, 0.25]})

    def test_contact_surface_record_rejects_non_normalized_axes(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="bad_top",
            object_id=None,
            surface_type="plane",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 2.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
        )

        with self.assertRaisesRegex(ValueError, "normal"):
            surface.validate()

    def test_anchor_edit_surface_constraint_fields_validate(self) -> None:
        edit = ContactAnchorEditRecord(
            edit_id="edit_a",
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            old_world_position=[0.0, 0.0, 0.0],
            new_world_position=[0.1, 0.0, 0.0],
            requested_delta_world=[0.1, 0.0, 0.2],
            delta_world=[0.1, 0.0, 0.0],
            tangent_delta=[0.1, 0.0],
            surface_id="platform_top",
            surface_normal=[0.0, 0.0, 1.0],
            surface_coordinates_before={"u": 0.0, "v": 0.0},
            surface_coordinates_after={"u": 0.1, "v": 0.0},
            constraint_mode="reject",
            clamped=False,
        )

        data = edit.to_dict()

        self.assertEqual(data["requested_delta_world"], [0.1, 0.0, 0.2])
        self.assertEqual(data["delta_world"], [0.1, 0.0, 0.0])
        self.assertEqual(data["tangent_delta"], [0.1, 0.0])
        self.assertEqual(data["surface_id"], "platform_top")

    def test_move_contact_anchor_records_old_new_and_delta(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[1.0, 2.0, 0.0],
            position_source="body_pos_w_mean",
        )

        moved = move_contact_anchor(anchor, delta_world=[0.1, 0.0, 0.0])
        edit = make_anchor_move_edit(anchor, new_world_position=[1.1, 2.0, 0.0], affected_frames=[0, 10], source="lte")

        self.assertEqual(moved.world_position, [1.1, 2.0, 0.0])
        self.assertEqual(moved.position_source, "manual")
        self.assertEqual(moved.metadata["contact_anchor_edits"][-1]["old_world_position"], [1.0, 2.0, 0.0])
        self.assertEqual(edit.edit_type, "move_contact_anchor")
        self.assertEqual(edit.old_world_position, [1.0, 2.0, 0.0])
        self.assertEqual(edit.new_world_position, [1.1, 2.0, 0.0])
        self.assertEqual(edit.delta_world, [0.10000000000000009, 0.0, 0.0])

    def test_move_contact_anchor_on_surface_projects_normal_delta(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.0],
            object_id="box",
            surface_id="top",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[0.0, 0.0, 0.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
            surface_coordinates={"u": 0.0, "v": 0.0},
        )

        moved, edit = move_contact_anchor_on_surface(anchor, requested_world_delta=[0.1, 0.0, 0.2])

        self.assertEqual(moved.world_position, [0.1, 0.0, 0.0])
        self.assertEqual(edit.requested_delta_world, [0.1, 0.0, 0.2])
        self.assertEqual(edit.delta_world, [0.1, 0.0, 0.0])
        self.assertEqual(edit.tangent_delta, [0.1, 0.0])
        self.assertEqual(edit.surface_id, "top")
        self.assertEqual(edit.constraint_mode, "reject")
        self.assertFalse(edit.clamped)

    def test_move_contact_anchor_on_surface_uses_tangent_basis(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[1.0, 2.0, 0.0],
            object_id="box",
            surface_id="top",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[1.0, 2.0, 0.0],
            surface_tangent_u=[0.0, 1.0, 0.0],
            surface_tangent_v=[1.0, 0.0, 0.0],
            surface_bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
            surface_coordinates={"u": 0.0, "v": 0.0},
        )

        moved, edit = move_contact_anchor_on_surface(anchor, tangent_delta=[0.2, 0.3])

        self.assertEqual(moved.world_position, [1.3, 2.2, 0.0])
        self.assertEqual(moved.surface_coordinates, {"u": 0.2, "v": 0.3})
        self.assertEqual(edit.delta_world, [0.3, 0.2, 0.0])

    def test_move_contact_anchor_on_surface_rejects_outside_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.0],
            surface_id="top",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[0.0, 0.0, 0.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-0.1, 0.1], "v": [-0.1, 0.1]},
            surface_coordinates={"u": 0.0, "v": 0.0},
        )

        with self.assertRaises(ValueError):
            move_contact_anchor_on_surface(anchor, tangent_delta=[0.2, 0.0])

    def test_move_contact_anchor_on_surface_clamps_outside_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.0],
            object_id="box",
            surface_id="top",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[0.0, 0.0, 0.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-0.1, 0.1], "v": [-0.1, 0.1]},
            surface_coordinates={"u": 0.0, "v": 0.0},
        )

        moved, edit = move_contact_anchor_on_surface(anchor, tangent_delta=[0.2, 0.0], mode="clamp")

        self.assertEqual(moved.world_position, [0.1, 0.0, 0.0])
        self.assertEqual(moved.object_id, "box")
        self.assertEqual(moved.surface_id, "top")
        self.assertEqual(edit.delta_world, [0.1, 0.0, 0.0])
        self.assertEqual(edit.tangent_delta, [0.1, 0.0])
        self.assertEqual(edit.surface_id, "top")
        self.assertEqual(edit.constraint_mode, "clamp")
        self.assertTrue(edit.clamped)

    def test_move_contact_anchor_on_mesh_surface_uses_polygon_not_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.25, 0.25, 0.0],
            surface_id="mesh_tri",
            surface_type="mesh_face",
            surface_normal=[0.0, 0.0, 1.0],
            surface_origin=[0.0, 0.0, 0.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
            surface_coordinates={"u": 0.25, "v": 0.25},
            metadata={
                "surface_bindings": [
                    {
                        "polygon_surface_coordinates": [
                            {"u": 0.0, "v": 0.0},
                            {"u": 1.0, "v": 0.0},
                            {"u": 0.0, "v": 1.0},
                        ]
                    }
                ]
            },
        )

        with self.assertRaisesRegex(ValueError, "polygon"):
            move_contact_anchor_on_surface(anchor, tangent_delta=[0.5, 0.5])
        moved, edit = move_contact_anchor_on_surface(anchor, tangent_delta=[0.5, 0.5], mode="clamp")
        self.assertAlmostEqual(sum(moved.surface_coordinates.values()), 1.0)
        self.assertTrue(edit.clamped)

    def test_bind_anchor_to_plane_creates_surface_binding_for_safe_moves(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.2, 0.3, 0.0],
        )

        bound = bind_anchor_to_plane(
            anchor,
            surface_id="platform_top",
            normal=[0.0, 0.0, 1.0],
            origin=[0.0, 0.0, 0.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [0.0, 1.0], "v": [0.0, 1.0]},
            surface_type="box_face",
            source="terrain_binding",
        )
        moved, edit = move_contact_anchor_on_surface(bound, tangent_delta=[0.1, 0.0])

        self.assertEqual(bound.surface_coordinates, {"u": 0.2, "v": 0.3})
        self.assertEqual(bound.surface_binding_source, "terrain_binding")
        self.assertEqual(moved.world_position, [0.30000000000000004, 0.3, 0.0])
        self.assertEqual(edit.surface_id, "platform_top")

    def test_bind_anchor_to_surface_computes_surface_coordinates(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.2, 0.3, 0.02],
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
            source="manual_surface_catalog",
        )

        bound = bind_anchor_to_surface(anchor, surface, max_distance=0.05)

        self.assertEqual(bound.object_id, "box_0")
        self.assertEqual(bound.surface_id, "box_0_top")
        self.assertEqual(bound.world_position, [0.2, 0.3, 0.0])
        self.assertEqual(bound.surface_coordinates, {"u": 0.2, "v": 0.3})
        self.assertEqual(bound.metadata["surface_bindings"][-1]["signed_surface_distance"], 0.02)
        self.assertEqual(bound.metadata["surface_bindings"][-1]["projected_world_position"], [0.2, 0.3, 0.0])

    def test_bind_anchor_to_surface_rejects_distance_and_bounds(self) -> None:
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="top",
            object_id=None,
            surface_type="plane",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-0.1, 0.1], "v": [-0.1, 0.1]},
        )
        far = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_far",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.2],
        )
        outside = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_outside",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.2, 0.0, 0.0],
        )

        with self.assertRaisesRegex(ValueError, "max_distance"):
            bind_anchor_to_surface(far, surface, max_distance=0.05)
        with self.assertRaisesRegex(ValueError, "bounds"):
            bind_anchor_to_surface(outside, surface, max_distance=0.05)

    def test_bind_anchor_to_surface_clamps_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.2, 0.0, 0.0],
        )
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="top",
            object_id="box",
            surface_type="box_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-0.1, 0.1], "v": [-0.1, 0.1]},
        )

        bound = bind_anchor_to_surface(anchor, surface, mode="clamp")

        self.assertEqual(bound.world_position, [0.1, 0.0, 0.0])
        self.assertEqual(bound.surface_coordinates, {"u": 0.1, "v": 0.0})
        self.assertTrue(bound.metadata["surface_bindings"][-1]["clamped"])

    def test_mesh_surface_binding_requires_polygon_not_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.75, 0.75, 0.0],
        )
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="mesh_top",
            object_id="box",
            surface_type="mesh_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
        )

        with self.assertRaisesRegex(ValueError, "polygon_world"):
            bind_anchor_to_surface(anchor, surface)

    def test_mesh_surface_binding_uses_polygon_not_outer_bounds(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=10,
            world_position=[0.75, 0.75, 0.0],
        )
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="mesh_tri",
            object_id="box",
            surface_type="mesh_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
            metadata={"polygon_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
        )

        with self.assertRaisesRegex(ValueError, "polygon"):
            bind_anchor_to_surface(anchor, surface)
        clamped = bind_anchor_to_surface(anchor, surface, mode="clamp")
        self.assertAlmostEqual(sum(clamped.surface_coordinates.values()), 1.0)
        self.assertTrue(clamped.metadata["surface_bindings"][-1]["clamped"])

    def test_bind_anchors_to_surfaces_applies_body_policy_and_failure_metadata(self) -> None:
        foot = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.0],
        )
        hand = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lh",
            body="LH",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.0],
        )
        vertical = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="wall",
            object_id="wall",
            surface_type="box_face",
            origin=[0.0, 0.0, 0.0],
            normal=[1.0, 0.0, 0.0],
            tangent_u=[0.0, 1.0, 0.0],
            tangent_v=[0.0, 0.0, 1.0],
            bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
        )

        bound = bind_anchors_to_surfaces([foot, hand], [vertical])

        self.assertTrue(surface_compatible_with_body("LF", vertical))
        self.assertTrue(surface_compatible_with_body("LH", vertical))
        self.assertEqual(bound[0].surface_id, "wall")
        self.assertEqual(bound[1].surface_id, "wall")

    def test_bound_anchor_can_move_on_surface(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="LF",
            start_frame=0,
            end_frame=10,
            world_position=[0.0, 0.0, 0.01],
        )
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="top",
            object_id="box",
            surface_type="box_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
        )

        bound = bind_anchor_to_surface(anchor, surface)
        moved, edit = move_contact_anchor_on_surface(bound, tangent_delta=[0.1, 0.0])

        self.assertEqual(moved.world_position, [0.1, 0.0, 0.0])
        self.assertEqual(edit.delta_world, [0.1, 0.0, 0.0])


    def test_refine_anchor_position_from_raw_contact_surface_points(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True], [True], [False]]),
            body_pos_w=np.asarray([[[9.0, 9.0, 9.0]], [[9.0, 9.0, 9.0]], [[0.0, 0.0, 0.0]]]),
            body_names=["left_foot"],
        )
        surface = ContactSurfaceRecord(
            motion_id="motion_a",
            surface_id="box_0_top",
            object_id="box_0",
            surface_type="mesh_face",
            origin=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            tangent_u=[1.0, 0.0, 0.0],
            tangent_v=[0.0, 1.0, 0.0],
            bounds={"u": [0.0, 1.0], "v": [0.0, 1.0]},
            metadata={"polygon_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            np.savez(
                motion,
                raw_contact_count=np.asarray([1, 1, 0], dtype=np.int32),
                raw_contact_point0_w=np.asarray(
                    [
                        [[0.2, 0.2, 0.0]],
                        [[0.4, 0.2, 0.0]],
                        [[0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                raw_contact_point1_w=np.asarray(
                    [
                        [[0.21, 0.2, 0.01]],
                        [[0.41, 0.2, 0.01]],
                        [[0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                contact_force_part_position_w=np.asarray(
                    [
                        [[0.21, 0.2, 0.01], [3.0, 3.0, 3.0]],
                        [[0.41, 0.2, 0.01], [3.0, 3.0, 3.0]],
                        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
                    ],
                    dtype=np.float32,
                ),
                contact_force_part_order=np.asarray(["LF", "RF"]),
                raw_contact_source=np.asarray("newton_raw_rigid_contacts"),
            )
            refined = refine_contact_graph_anchor_positions_from_raw_contacts(graph, motion, surfaces=[surface])

        self.assertEqual(refined.anchors[0].position_source, "raw_contact_point0_w_polygon_median")
        self.assertEqual(refined.anchors[0].world_position, [0.30000000447034836, 0.20000000298023224, 0.0])
        self.assertEqual(refined.patches[0].patch_center_world, refined.anchors[0].world_position)
        self.assertEqual(
            refined.anchors[0].metadata["raw_contact_position_refinement"]["accepted_raw_contact_sample_count"],
            2,
        )
        self.assertEqual(refined.anchors[0].metadata["raw_contact_position_refinement"]["binding_candidate_class"], "top")

    def test_refine_anchor_derives_part_positions_from_canonical_body_fk(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True], [True], [False]]),
            body_pos_w=np.asarray(
                [[[0.2, 0.2, 0.02]], [[0.4, 0.2, 0.02]], [[0.0, 0.0, 0.0]]]
            ),
            body_names=["left_hand"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            np.savez(
                motion,
                raw_contact_count=np.asarray([1, 1, 0], dtype=np.int32),
                raw_contact_point0_w=np.asarray(
                    [[[0.2, 0.2, 0.0]], [[0.4, 0.2, 0.0]], [[0.0, 0.0, 0.0]]],
                    dtype=np.float32,
                ),
                raw_contact_point1_w=np.asarray(
                    [[[0.2, 0.2, 0.02]], [[0.4, 0.2, 0.02]], [[0.0, 0.0, 0.0]]],
                    dtype=np.float32,
                ),
                contact_force_part_order=np.asarray(["LH"]),
                body_pos_w=np.asarray(
                    [
                        [[0.2, 0.2, 0.02], [0.2, 0.2, 0.03]],
                        [[0.4, 0.2, 0.02], [0.4, 0.2, 0.03]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]],
                    ],
                    dtype=np.float32,
                ),
                body_names=np.asarray(
                    ["left_sphere_hand_link", "left_sphere_hand_tip_link"]
                ),
            )
            refined = refine_contact_graph_anchor_positions_from_raw_contacts(
                graph,
                motion,
            )

        self.assertFalse(
            refined.anchors[0].metadata["raw_contact_position_refinement_failed"]
        )
        self.assertEqual(
            refined.anchors[0].metadata["raw_contact_position_refinement"][
                "accepted_raw_contact_sample_count"
            ],
            2,
        )

    def test_multi_rollout_raw_contact_uses_body0_point0_shape0_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            np.savez(
                motion,
                raw_contact_count=np.asarray([1], dtype=np.int32),
                raw_contact_point0_w=np.asarray([[[1.0, 2.0, 3.0]]]),
                raw_contact_point1_w=np.asarray([[[4.0, 5.0, 6.0]]]),
                raw_contact_shape0=np.asarray([[51]], dtype=np.int32),
                raw_contact_shape1=np.asarray([[0]], dtype=np.int32),
                raw_contact_source=np.asarray(
                    "multi_rollout_stable_label_median_external_contacts"
                ),
                contact_force_part_position_w=np.asarray(
                    [[[1.0, 2.0, 3.0]]]
                ),
                contact_force_part_order=np.asarray(["LHEE"]),
            )
            raw = RawContactMotion(motion)

        np.testing.assert_allclose(raw.robot_contact_point_w[0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(raw.surface_contact_point_w[0, 0], [4.0, 5.0, 6.0])
        assert int(raw.robot_contact_shape[0, 0]) == 51

    def test_raw_contact_position_refinement_summarizes_foot_as_heel_toe(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True], [True], [True], [True], [False]]),
            body_pos_w=np.zeros((5, 1, 3), dtype=float),
            body_names=["left_foot"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            np.savez(
                motion,
                raw_contact_count=np.asarray([2, 2, 2, 2, 0], dtype=np.int32),
                raw_contact_point0_w=np.asarray(
                    [
                        [[-0.12, 0.0, 0.0], [0.12, 0.0, 0.0]],
                        [[-0.10, 0.0, 0.0], [0.14, 0.0, 0.0]],
                        [[-0.11, 0.0, 0.0], [0.13, 0.0, 0.0]],
                        [[-0.13, 0.0, 0.0], [0.11, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                raw_contact_point1_w=np.asarray(
                    [
                        [[-0.12, 0.0, 0.0], [0.12, 0.0, 0.0]],
                        [[-0.10, 0.0, 0.0], [0.14, 0.0, 0.0]],
                        [[-0.11, 0.0, 0.0], [0.13, 0.0, 0.0]],
                        [[-0.13, 0.0, 0.0], [0.11, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                raw_contact_force_w=np.ones((5, 2, 3), dtype=np.float32),
                raw_contact_shape1=np.asarray([[39, 45], [39, 45], [40, 44], [40, 44], [0, 0]], dtype=np.int32),
                contact_force_part_position_w=np.zeros((5, 1, 3), dtype=np.float32),
                contact_force_part_order=np.asarray(["LF"]),
                raw_contact_source=np.asarray("newton_raw_rigid_contacts"),
            )
            refined = refine_contact_graph_anchor_positions_from_raw_contacts(graph, motion)

        summary = refined.anchors[0].metadata["raw_contact_position_refinement"]["foot_contact_summary"]
        self.assertEqual(summary["method"], "raw_contact_pca_heel_toe")
        self.assertIn("heel", summary["contacts"])
        self.assertIn("toe", summary["contacts"])
        self.assertLess(summary["contacts"]["heel"]["world_position"][0], 0.0)
        self.assertGreater(summary["contacts"]["toe"]["world_position"][0], 0.0)
        self.assertEqual(summary["contacts"]["heel"]["raw_shape_ids"], [39, 40])
        self.assertEqual(summary["contacts"]["toe"]["raw_shape_ids"], [44, 45])

    def test_split_foot_contact_anchors_from_raw_heel_toe_summary(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True], [True], [True], [True], [False]]),
            body_pos_w=np.zeros((5, 1, 3), dtype=float),
            body_names=["left_foot"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            np.savez(
                motion,
                raw_contact_count=np.asarray([1, 1, 2, 2, 0], dtype=np.int32),
                raw_contact_point0_w=np.asarray(
                    [
                        [[0.12, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.13, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[-0.10, 0.0, 0.0], [0.14, 0.0, 0.0]],
                        [[-0.11, 0.0, 0.0], [0.13, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                raw_contact_point1_w=np.asarray(
                    [
                        [[0.12, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.13, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[-0.10, 0.0, 0.0], [0.14, 0.0, 0.0]],
                        [[-0.11, 0.0, 0.0], [0.13, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                raw_contact_force_w=np.ones((5, 2, 3), dtype=np.float32),
                raw_contact_shape1=np.asarray([[45, 0], [45, 0], [39, 45], [39, 45], [0, 0]], dtype=np.int32),
                contact_force_part_position_w=np.zeros((5, 1, 3), dtype=np.float32),
                contact_force_part_order=np.asarray(["LF"]),
                raw_contact_source=np.asarray("newton_raw_rigid_contacts"),
            )
            refined = refine_contact_graph_anchor_positions_from_raw_contacts(graph, motion)
        split = split_foot_contact_anchors(refined)

        self.assertEqual(len(split.anchors), 2)
        by_role = {anchor.metadata["patch_role"]: anchor for anchor in split.anchors}
        self.assertEqual(by_role["toe"].start_frame, 0)
        self.assertEqual(by_role["toe"].end_frame, 2)
        self.assertEqual(by_role["sole"].start_frame, 2)
        self.assertEqual(by_role["sole"].end_frame, 4)
        self.assertGreater(by_role["toe"].world_position[0], by_role["sole"].world_position[0])
        self.assertEqual(by_role["toe"].metadata["parent_anchor_id"], refined.anchors[0].anchor_id)

    def test_split_foot_contact_preserves_short_role_transition_coverage(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="climb_00",
            anchor_id="climb_00_anchor_left_toe_000147_000171",
            body="left_toe",
            start_frame=147,
            end_frame=171,
            world_position=[0.0, 0.0, 0.0],
            surface_id="box_top",
            metadata={
                "raw_contact_position_refinement": {
                    "foot_contact_summary": {
                        "intervals": [
                            {"patch_role": "heel", "frame_start": 147, "frame_end": 150},
                            {"patch_role": "sole", "frame_start": 150, "frame_end": 152},
                            {"patch_role": "heel", "frame_start": 152, "frame_end": 157},
                            {"patch_role": "sole", "frame_start": 157, "frame_end": 160},
                            {"patch_role": "heel", "frame_start": 160, "frame_end": 161},
                            {"patch_role": "sole", "frame_start": 161, "frame_end": 163},
                            {"patch_role": "heel", "frame_start": 163, "frame_end": 164},
                            {"patch_role": "sole", "frame_start": 164, "frame_end": 166},
                            {"patch_role": "heel", "frame_start": 166, "frame_end": 171},
                        ],
                        "contacts": {
                            "heel": {
                                "world_position": [0.0, 0.0, 0.0],
                                "sample_count": 20,
                                "confidence": 1.0,
                            },
                            "toe": {
                                "world_position": [0.02, 0.0, 0.0],
                                "sample_count": 20,
                                "confidence": 1.0,
                            },
                        },
                    }
                }
            },
        )
        graph_type = type(
            contact_graph_from_masks(
                motion_id="climb_00",
                contact_mask=None,
            )
        )
        graph = graph_type(motion_id="climb_00", anchors=[anchor])

        split = split_foot_contact_anchors(graph)

        bounds = [
            (item.metadata["patch_role"], item.start_frame, item.end_frame)
            for item in split.anchors
        ]
        self.assertEqual(
            bounds,
            [
                ("heel", 147, 150),
                ("sole", 150, 166),
                ("heel", 166, 171),
            ],
        )
        covered = {
            frame
            for item in split.anchors
            for frame in range(item.start_frame, item.end_frame)
        }
        self.assertEqual(covered, set(range(anchor.start_frame, anchor.end_frame)))

    def test_split_foot_contact_uses_parent_episode_for_missing_shape_frames(self) -> None:
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="left_toe_parent",
            body="left_toe",
            start_frame=10,
            end_frame=20,
            world_position=[0.0, 0.0, 0.0],
            surface_id="box_top",
            metadata={
                "raw_contact_position_refinement": {
                    "foot_contact_summary": {
                        "intervals": [
                            {"patch_role": "heel", "frame_start": 12, "frame_end": 15},
                            {"patch_role": "sole", "frame_start": 17, "frame_end": 19},
                        ],
                        "contacts": {
                            "heel": {
                                "world_position": [0.0, 0.0, 0.0],
                                "sample_count": 3,
                                "confidence": 1.0,
                            },
                            "toe": {
                                "world_position": [0.02, 0.0, 0.0],
                                "sample_count": 2,
                                "confidence": 1.0,
                            },
                        },
                    }
                }
            },
        )
        graph_type = type(
            contact_graph_from_masks(motion_id="motion_a", contact_mask=None)
        )

        split = split_foot_contact_anchors(
            graph_type(motion_id="motion_a", anchors=[anchor])
        )

        self.assertEqual(
            [
                (item.metadata["patch_role"], item.start_frame, item.end_frame)
                for item in split.anchors
            ],
            [("heel", 10, 16), ("sole", 16, 20)],
        )


    def test_merge_nearby_contact_anchors_merges_short_same_class_gaps(self) -> None:
        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="a0",
                body="left_foot",
                start_frame=0,
                end_frame=10,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "ground"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="a1",
                body="left_foot",
                start_frame=12,
                end_frame=20,
                world_position=[0.02, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "ground"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="a2",
                body="left_foot",
                start_frame=22,
                end_frame=24,
                world_position=[0.03, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "edge_candidate"}},
            ),
        ]
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[False]]),
            body_names=["left_foot"],
        )
        graph = graph.__class__(motion_id="motion_a", anchors=anchors)

        merged, events = merge_nearby_contact_anchors(graph, max_gap=3, max_distance=0.06)

        self.assertEqual(len(merged.anchors), 2)
        self.assertEqual(merged.anchors[0].start_frame, 0)
        self.assertEqual(merged.anchors[0].end_frame, 20)
        self.assertEqual(merged.anchors[0].metadata["merged_anchor_ids"], ["a0", "a1"])
        self.assertEqual(len(events), 1)
        self.assertEqual(merged.anchors[1].anchor_id, "a2")

    def test_merge_nearby_contact_anchors_does_not_merge_different_patch_roles(self) -> None:
        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="toe",
                body="left_foot",
                start_frame=0,
                end_frame=10,
                world_position=[0.0, 0.0, 0.0],
                metadata={"patch_role": "toe", "raw_contact_position_refinement": {"binding_candidate_class": "top"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="heel",
                body="left_foot",
                start_frame=11,
                end_frame=20,
                world_position=[0.01, 0.0, 0.0],
                metadata={"patch_role": "heel", "raw_contact_position_refinement": {"binding_candidate_class": "top"}},
            ),
        ]
        graph = contact_graph_from_masks(motion_id="motion_a", contact_mask=np.asarray([[False]]), body_names=["left_foot"])
        graph = graph.__class__(motion_id="motion_a", anchors=anchors)

        merged, events = merge_nearby_contact_anchors(graph, max_gap=3, max_distance=0.06)

        self.assertEqual([anchor.anchor_id for anchor in merged.anchors], ["toe", "heel"])
        self.assertEqual(events, [])


    def test_filter_short_raw_missing_anchors_removes_adjacent_noise(self) -> None:
        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="noise",
                body="right_hand",
                start_frame=0,
                end_frame=3,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "raw_missing"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="top",
                body="right_hand",
                start_frame=4,
                end_frame=20,
                world_position=[0.03, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "top"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="long_raw_missing",
                body="right_hand",
                start_frame=30,
                end_frame=40,
                world_position=[0.03, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "raw_missing"}},
            ),
        ]
        graph = contact_graph_from_masks(motion_id="motion_a", contact_mask=None)
        graph = graph.__class__(motion_id="motion_a", anchors=anchors)

        filtered, events = filter_short_raw_missing_anchors(graph)

        self.assertEqual([anchor.anchor_id for anchor in filtered.anchors], ["top", "long_raw_missing"])
        self.assertEqual(events[0]["anchor_id"], "noise")

    def test_filter_contact_anchors_can_drop_binding_classes(self) -> None:
        anchors = [
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="raw",
                body="left_foot",
                start_frame=0,
                end_frame=20,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "raw_missing"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="edge",
                body="left_foot",
                start_frame=30,
                end_frame=35,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "edge_candidate"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="top",
                body="left_foot",
                start_frame=40,
                end_frame=80,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "top"}},
            ),
            ContactAnchorRecord(
                motion_id="motion_a",
                anchor_id="outside",
                body="left_foot",
                start_frame=90,
                end_frame=100,
                world_position=[0.0, 0.0, 0.0],
                metadata={"raw_contact_position_refinement": {"binding_candidate_class": "outside_known_surfaces"}},
            ),
        ]
        graph = contact_graph_from_masks(motion_id="motion_a", contact_mask=None)
        graph = graph.__class__(motion_id="motion_a", anchors=anchors)

        filtered, events = filter_short_raw_missing_anchors(graph, drop_classes={"raw_missing", "edge_candidate"})

        self.assertEqual([anchor.anchor_id for anchor in filtered.anchors], ["top", "outside"])
        self.assertEqual([event["binding_candidate_class"] for event in events], ["raw_missing", "edge_candidate"])


    def test_move_anchor_in_graph_updates_anchor_patch_and_returns_edit(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True, False], [True, False], [False, False]]),
            body_pos_w=np.asarray(
                [
                    [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                    [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]],
                ]
            ),
            body_names=["left_foot", "right_foot"],
        )

        with self.assertRaises(ValueError):
            move_anchor_in_graph(graph, anchor_id=graph.anchors[0].anchor_id, delta_world=[0.1, 0.0, 0.0])

        moved_graph, edit = move_anchor_in_graph(
            graph,
            anchor_id=graph.anchors[0].anchor_id,
            delta_world=[0.1, 0.0, 0.0],
            allow_free_3d=True,
        )

        self.assertEqual(moved_graph.anchors[0].world_position, [1.1, 2.0, 0.0])
        self.assertEqual(moved_graph.patches[0].patch_center_world, [1.1, 2.0, 0.0])
        self.assertEqual(edit.edit_type, "move_contact_anchor")

    def test_move_anchor_in_contact_layer_writes_graph_and_edit_record(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True, False], [True, False], [False, False]]),
            body_pos_w=np.asarray(
                [
                    [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                    [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]],
                ]
            ),
            body_names=["left_foot", "right_foot"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_contact_layer(root / "source", graph)
            moved_graph, edit = move_anchor_in_contact_layer(
                root / "source",
                root / "moved",
                motion_id="motion_a",
                anchor_id=graph.anchors[0].anchor_id,
                delta_world=[0.1, 0.0, 0.0],
                allow_free_3d=True,
            )
            loaded = read_contact_graph(root / "moved", "motion_a")
            edit_lines = (root / "moved" / "edits" / "motion_a.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(loaded.anchors[0].world_position, [1.1, 2.0, 0.0])
        self.assertEqual(moved_graph.anchors[0].world_position, [1.1, 2.0, 0.0])
        self.assertEqual(edit.edit_type, "move_contact_anchor")
        self.assertEqual(len(edit_lines), 1)

    def test_detects_touchdown_liftoff_active_and_support_changes(self) -> None:
        contact = np.asarray(
            [
                [False, True],
                [True, True],
                [True, False],
                [False, False],
                [False, True],
            ]
        )
        active = np.asarray(
            [
                [False, True],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
            ]
        )
        support = np.asarray(
            [
                [False, True],
                [False, True],
                [True, False],
                [True, False],
                [False, True],
            ]
        )

        events = detect_contact_events(
            motion_id="motion_a",
            contact_mask=contact,
            active_mask=active,
            support_mask=support,
            body_names=["LF", "RF"],
            source="test",
        )

        event_types = [(event.frame, event.body, event.event_type) for event in events]
        self.assertIn((1, "LF", "touchdown"), event_types)
        self.assertIn((2, "RF", "liftoff"), event_types)
        self.assertIn((1, "active", "active_change"), event_types)
        self.assertIn((2, "support", "support_switch"), event_types)

    def test_anchor_interval_grouping(self) -> None:
        contact = np.asarray(
            [
                [False, True],
                [True, True],
                [True, False],
                [False, False],
                [False, True],
            ]
        )
        support = np.asarray(
            [
                [False, True],
                [False, True],
                [True, False],
                [True, False],
                [False, True],
            ]
        )

        anchors = anchors_from_contact_mask(
            motion_id="motion_a",
            contact_mask=contact,
            support_mask=support,
            body_names=["LF", "RF"],
        )

        spans = [(anchor.body, anchor.start_frame, anchor.end_frame, anchor.role) for anchor in anchors]
        self.assertIn(("LF", 1, 3, "support"), spans)
        self.assertIn(("RF", 0, 2, "support"), spans)
        self.assertIn(("RF", 4, 5, "support"), spans)

    def test_anchor_position_estimated_from_body_pos_w_interval(self) -> None:
        contact = np.asarray([[True, False], [True, False], [False, False]])
        body_pos_w = np.asarray(
            [
                [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.2, 2.0, 0.0], [0.0, 0.0, 0.0]],
                [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]],
            ]
        )

        anchors = anchors_from_contact_mask(
            motion_id="motion_a",
            contact_mask=contact,
            body_pos_w=body_pos_w,
            body_names=["LF", "RF"],
        )

        self.assertEqual(anchors[0].world_position, [1.1, 2.0, 0.0])
        self.assertEqual(anchors[0].position_source, "body_pos_w_mean")
        self.assertEqual(anchors[0].metadata["first_world_position"], [1.0, 2.0, 0.0])
        self.assertEqual(anchors[0].metadata["last_world_position"], [1.2, 2.0, 0.0])
        self.assertAlmostEqual(anchors[0].metadata["max_drift_xy"], 0.2)

    def test_transition_converts_to_segment_with_contact_metadata(self) -> None:
        contact = np.asarray(
            [
                [False, True],
                [True, True],
                [True, False],
                [False, False],
                [False, True],
            ]
        )
        active = np.asarray(
            [
                [False, True],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
            ]
        )
        support = np.asarray(
            [
                [False, True],
                [False, True],
                [True, False],
                [True, False],
                [False, True],
            ]
        )

        events, anchors, transitions = transitions_from_proto_indices(
            motion_id="motion_a",
            starts=[1],
            ends=[4],
            contact_mask=contact,
            active_mask=active,
            support_mask=support,
            body_names=["LF", "RF"],
        )
        segment = segment_from_contact_transition(
            transition=transitions[0],
            segment_id="motion_a_force_0000",
            source="force_contact",
            status="candidate",
            track="proto",
            motion_path="/tmp/motion_a.npz",
            clip_npz="/tmp/motion_a.npz",
            clip_output_dir=None,
            clip_file_name="motion_a.npz",
            atom_label="force_contact_00",
            contact_start="11",
            contact_end="00",
            active="10",
            support="01",
            events=events,
            anchors=anchors,
        )

        self.assertEqual(segment.metadata["active_body"], "LF")
        self.assertEqual(segment.metadata["support_bodies"], ["RF"])
        self.assertEqual(segment.metadata["contact_transition"]["start_frame"], 1)
        self.assertGreaterEqual(len(segment.metadata["contact_events"]), 1)
        self.assertGreaterEqual(len(segment.metadata["contact_anchors"]), 1)

    def test_contact_records_roundtrip_as_typed_jsonl(self) -> None:
        contact = np.asarray([[False, True], [True, True], [True, False]])
        events, anchors, transitions = transitions_from_proto_indices(
            motion_id="motion_a",
            starts=[0],
            ends=[3],
            contact_mask=contact,
            body_names=["LF", "RF"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_contact_jsonl(root / "events.jsonl", events)
            write_contact_jsonl(root / "anchors.jsonl", anchors)
            write_contact_jsonl(root / "patches.jsonl", contact_graph_from_masks(motion_id="motion_a", contact_mask=contact, body_names=["LF", "RF"]).patches)
            write_contact_jsonl(root / "transitions.jsonl", transitions)

            loaded_events = read_contact_events(root / "events.jsonl")
            loaded_anchors = read_contact_anchors(root / "anchors.jsonl")
            loaded_patches = read_contact_patches(root / "patches.jsonl")
            loaded_transitions = read_contact_transitions(root / "transitions.jsonl")

        self.assertEqual(loaded_events[0].event_id, events[0].event_id)
        self.assertEqual(loaded_anchors[0].anchor_id, anchors[0].anchor_id)
        self.assertEqual(loaded_patches[0].anchor_id, loaded_anchors[0].anchor_id)
        self.assertEqual(loaded_transitions[0].transition_id, transitions[0].transition_id)

    def test_contact_graph_groups_events_anchors_and_transitions(self) -> None:
        contact = np.asarray([[True, False], [False, False], [True, False]])

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            body_names=["LF", "RF"],
            source="test",
        )

        self.assertEqual(graph.motion_id, "motion_a")
        self.assertTrue(any(event.event_type == "liftoff" for event in graph.events))
        self.assertTrue(any(event.event_type == "touchdown" for event in graph.events))
        self.assertEqual(len(graph.anchors), 2)
        self.assertEqual(len(graph.patches), 2)
        self.assertEqual(graph.patches[0].patch_type, "foot")
        self.assertEqual(graph.transitions[0].active_body, "LF")
        self.assertEqual(graph.to_dict()["motion_id"], "motion_a")
        self.assertEqual(len(graph.to_dict()["patches"]), 2)

    def test_contact_patch_uses_anchor_position_and_patch_preset(self) -> None:
        contact = np.asarray([[True, False], [True, False], [False, False]])
        body_pos_w = np.asarray(
            [
                [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.2, 2.0, 0.0], [0.0, 0.0, 0.0]],
                [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]],
            ]
        )

        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=contact,
            body_pos_w=body_pos_w,
            body_names=["left_foot", "right_foot"],
        )

        patch = graph.patches[0]
        self.assertEqual(patch.patch_type, "foot")
        self.assertEqual(patch.patch_center_world, [1.1, 2.0, 0.0])
        self.assertEqual(patch.link_names, ["left_foot", "left_foot_sole"])
        self.assertAlmostEqual(patch.slip_score or 0.0, 0.1)

    def test_contact_layer_roundtrips_graph_components(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
            body_names=["LF", "RF"],
            source="test",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "contact_layer"
            write_contact_layer(root, graph)
            loaded = read_contact_graph(root, "motion_a")

        self.assertEqual([event.event_id for event in loaded.events], [event.event_id for event in graph.events])
        self.assertEqual([anchor.anchor_id for anchor in loaded.anchors], [anchor.anchor_id for anchor in graph.anchors])
        self.assertEqual([patch.patch_id for patch in loaded.patches], [patch.patch_id for patch in graph.patches])
        self.assertEqual(
            [transition.transition_id for transition in loaded.transitions],
            [transition.transition_id for transition in graph.transitions],
        )

    def test_contact_layer_read_derives_missing_patches_for_legacy_layers(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
            body_names=["LF", "RF"],
            source="test",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "contact_layer"
            write_contact_jsonl(root / "events" / "motion_a.jsonl", graph.events)
            write_contact_jsonl(root / "anchors" / "motion_a.jsonl", graph.anchors)
            write_contact_jsonl(root / "transitions" / "motion_a.jsonl", graph.transitions)
            loaded = read_contact_graph(root, "motion_a")

        self.assertEqual(len(loaded.patches), len(graph.anchors))

    def test_bind_segment_to_contact_graph_refreshes_metadata_for_bounds(self) -> None:
        graph = contact_graph_from_masks(
            motion_id="motion_a",
            contact_mask=np.asarray([[True, False], [False, False], [True, False]]),
            body_names=["LF", "RF"],
            source="test",
        )
        segment = SegmentRecord(
            motion_id="motion_a",
            segment_id="segment_a",
            start_frame=1,
            end_frame=2,
            source="manual",
            metadata={"contact_transition": {"stale": True}},
        )

        bound = bind_segment_to_contact_graph(segment, graph)

        self.assertEqual(bound.metadata["contact_transition"]["transition_id"], graph.transitions[0].transition_id)
        self.assertEqual(bound.metadata["contact_binding"]["transition_id"], graph.transitions[0].transition_id)
        self.assertEqual(bound.metadata["contact_binding"]["patch_count"], 2)
        self.assertEqual(len(bound.metadata["contact_patches"]), 2)
        self.assertEqual(bound.metadata["active_body"], "LF")


if __name__ == "__main__":
    unittest.main()
