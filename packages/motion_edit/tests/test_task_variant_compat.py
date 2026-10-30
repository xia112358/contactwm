from __future__ import annotations

import json

import numpy as np

from motion_edit.contact.plans import ContactEditPlan, read_contact_edit_plan, validate_contact_edit_plan
from motion_edit.contact.schema import (
    ContactAnchorEditRecord,
    ContactAnchorRecord,
    ContactSurfaceRecord,
)


def _surface(surface_id: str, z: float) -> dict:
    return {
        "surface_id": surface_id,
        "object_id": surface_id.removesuffix("_top"),
        "surface_type": "plane",
        "origin": [0.5, -0.3, z],
        "normal": [0.0, 0.0, 1.0],
        "tangent_u": [1.0, 0.0, 0.0],
        "tangent_v": [0.0, 1.0, 0.0],
        "bounds": {"u": [-1.0, 1.0], "v": [-1.0, 1.0]},
    }


def test_reader_accepts_task_variant_fields(tmp_path) -> None:
    payload = {
        "plan_id": "height_110",
        "source_motion_path": "/tmp/source.npz",
        "source_motion_id": "climb_00",
        "source_contact_layer": "contact/source",
        "edits": [],
        "pose_edits": [],
        "surface_transforms": [
            {
                "transform_id": "height_follow",
                "kind": "surface_follow",
                "height_scale": 1.1,
                "translation_world": [0.0, 0.0, 0.07],
                "source_surface": _surface("box_1p0_top", 0.70),
                "target_surface": _surface("box_1p1_top", 0.77),
            }
        ],
        "status": "validated",
        "metadata": {"schema": "motion_edit_height_task_variant_v1"},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    plan = read_contact_edit_plan(path)

    assert plan.pose_edits == []
    assert len(plan.surface_transforms) == 1
    assert validate_contact_edit_plan(plan) == []
