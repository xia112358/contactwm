from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np

from somaforge_core.published_data import load_published_json

from motion_edit.contact.schema import ContactAnchorEditRecord, PoseEditRecord
from motion_edit.contact.surface_frame import map_points_between_surface_frames

ContactEditPlanStatus = Literal["draft", "validated", "locked", "generated"]


@dataclass(frozen=True)
class ContactEditPlan:
    plan_id: str
    source_motion_path: str
    source_motion_id: str
    source_contact_layer: str
    source_segment_layer: str | None = None
    edits: list[dict[str, Any]] = field(default_factory=list)
    pose_edits: list[dict[str, Any]] = field(default_factory=list)
    surface_transforms: list[dict[str, Any]] = field(default_factory=list)
    status: ContactEditPlanStatus = "draft"
    output_motion_path: str | None = None
    output_contact_layer: str | None = None
    output_segment_layer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, allow_free: bool = False) -> None:
        validate_contact_edit_plan(self, allow_free=allow_free)

    def to_dict(self) -> dict[str, Any]:
        validate_contact_edit_plan(self, allow_free=True)
        return asdict(self)


def _edit_dict(edit: ContactAnchorEditRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(edit, ContactAnchorEditRecord):
        return edit.to_dict()
    return dict(edit)


def write_contact_edit_plan(path: str | Path, plan: ContactEditPlan) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_contact_edit_plan(path: str | Path) -> ContactEditPlan:
    data = load_published_json(path)
    return ContactEditPlan(**data)


def append_anchor_edit_to_plan(
    path: str | Path,
    edit: ContactAnchorEditRecord | dict[str, Any],
    *,
    plan_id: str,
    source_motion_path: str,
    source_motion_id: str,
    source_contact_layer: str,
    source_segment_layer: str | None = None,
) -> ContactEditPlan:
    plan_path = Path(path).expanduser()
    if plan_path.exists():
        plan = read_contact_edit_plan(plan_path)
    else:
        plan = ContactEditPlan(
            plan_id=plan_id,
            source_motion_path=source_motion_path,
            source_motion_id=source_motion_id,
            source_contact_layer=source_contact_layer,
            source_segment_layer=source_segment_layer,
        )
    edits = list(plan.edits)
    edits.append(_edit_dict(edit))
    updated = ContactEditPlan(
        plan_id=plan.plan_id,
        source_motion_path=plan.source_motion_path,
        source_motion_id=plan.source_motion_id,
        source_contact_layer=plan.source_contact_layer,
        source_segment_layer=plan.source_segment_layer or source_segment_layer,
        edits=edits,
        pose_edits=list(plan.pose_edits),
        surface_transforms=list(plan.surface_transforms),
        status="draft" if plan.status == "validated" else plan.status,
        output_motion_path=plan.output_motion_path,
        output_contact_layer=plan.output_contact_layer,
        output_segment_layer=plan.output_segment_layer,
        metadata=dict(plan.metadata),
    )
    write_contact_edit_plan(plan_path, updated)
    return updated


def validate_contact_edit_plan(plan: ContactEditPlan, *, allow_free: bool = False) -> list[str]:
    if plan.status not in {"draft", "validated", "locked", "generated"}:
        raise ValueError(f"{plan.plan_id}: unsupported status {plan.status}")
    if not plan.source_motion_id:
        raise ValueError(f"{plan.plan_id}: source_motion_id is required")
    if not plan.source_contact_layer:
        raise ValueError(f"{plan.plan_id}: source_contact_layer is required")
    warnings: list[str] = []
    for index, transform in enumerate(plan.surface_transforms):
        transform_id = str(transform.get("transform_id") or f"surface_transform_{index}")
        if transform.get("kind") != "surface_follow":
            raise ValueError(f"{transform_id}: unsupported surface transform kind")
        source_surface = transform.get("source_surface")
        target_surface = transform.get("target_surface")
        if not isinstance(source_surface, dict) or not source_surface.get("surface_id"):
            raise ValueError(f"{transform_id}: source_surface is required")
        if not isinstance(target_surface, dict) or not target_surface.get("surface_id"):
            raise ValueError(f"{transform_id}: target_surface is required")
        translation = transform.get("translation_world")
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError(f"{transform_id}: translation_world must be xyz")
        if not all(isfinite(float(value)) for value in translation):
            raise ValueError(f"{transform_id}: translation_world must be finite")
    for raw_pose_edit in plan.pose_edits:
        PoseEditRecord(**raw_pose_edit).validate()
    for index, raw_edit in enumerate(plan.edits):
        edit = ContactAnchorEditRecord(**raw_edit)
        edit.validate()
        if edit.affected_frames is not None:
            if len(edit.affected_frames) != 2:
                raise ValueError(f"{edit.edit_id}: affected_frames must be [start, end]")
            if edit.affected_frames[0] < 0 or edit.affected_frames[1] <= edit.affected_frames[0]:
                raise ValueError(f"{edit.edit_id}: affected_frames must be a valid positive interval")
        free_edit = edit.constraint_mode in {None, "free_3d"} or not edit.surface_id
        if free_edit and not allow_free:
            raise ValueError(f"{edit.edit_id}: edit {index} is not surface-constrained")
        if edit.surface_normal is not None and edit.delta_world is not None:
            normal_component = sum(float(edit.surface_normal[i]) * float(edit.delta_world[i]) for i in range(3))
            if abs(normal_component) > 1e-6 and edit.constraint_mode != "surface_transform":
                raise ValueError(f"{edit.edit_id}: effective delta has surface-normal displacement")
        if edit.constraint_mode == "surface_transform":
            transform = edit.metadata.get("surface_transform") if isinstance(edit.metadata, dict) else None
            if not isinstance(transform, dict) or transform.get("kind") != "surface_follow":
                raise ValueError(f"{edit.edit_id}: surface_transform requires surface_follow metadata")
            translation = transform.get("translation_world")
            if not isinstance(translation, list) or len(translation) != 3:
                raise ValueError(f"{edit.edit_id}: surface_transform requires xyz translation_world")
            if edit.delta_world is None:
                raise ValueError(f"{edit.edit_id}: surface_transform requires delta_world")
            source_surface = transform.get("source_surface")
            target_surface = transform.get("target_surface")
            if (
                isinstance(source_surface, dict)
                and isinstance(target_surface, dict)
                and edit.old_world_position is not None
            ):
                expected_new = map_points_between_surface_frames(
                    edit.old_world_position,
                    source_surface,
                    target_surface,
                )
                expected_delta = expected_new - np.asarray(
                    edit.old_world_position, dtype=np.float64
                )
            else:
                expected_delta = np.asarray(translation, dtype=np.float64)
            residual = sum((float(edit.delta_world[i]) - float(expected_delta[i])) ** 2 for i in range(3)) ** 0.5
            if residual > 1.0e-6:
                raise ValueError(f"{edit.edit_id}: contact must follow the transformed surface exactly")
            if edit.surface_coordinates_before != edit.surface_coordinates_after:
                raise ValueError(f"{edit.edit_id}: surface-follow edit must preserve surface coordinates")
        if edit.constraint_mode == "reject" and edit.clamped:
            raise ValueError(f"{edit.edit_id}: reject-mode edit cannot be clamped")
        if edit.clamped:
            warnings.append(f"{edit.edit_id}: clamped to surface bounds")

    for index, raw_edit in enumerate(plan.pose_edits):
        _validate_pose_edit(plan.plan_id, index, raw_edit)
    for index, raw_transform in enumerate(plan.surface_transforms):
        _validate_surface_transform(plan.plan_id, index, raw_transform)
    return warnings


def _validate_pose_edit(plan_id: str, index: int, raw: dict[str, Any]) -> None:
    edit_id = str(raw.get("edit_id") or f"{plan_id}:pose_edits[{index}]")
    if raw.get("edit_type") != "translate_pose":
        raise ValueError(f"{edit_id}: unsupported pose edit type {raw.get('edit_type')!r}")
    translation = raw.get("translation_world")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError(f"{edit_id}: translation_world must have length 3")
    frames = raw.get("affected_frames")
    if not isinstance(frames, list) or len(frames) != 2 or int(frames[1]) <= int(frames[0]):
        raise ValueError(f"{edit_id}: affected_frames must be a non-empty [start, end) interval")
    semantic_names = raw.get("semantic_names")
    if not isinstance(semantic_names, list) or not semantic_names:
        raise ValueError(f"{edit_id}: semantic_names must be a non-empty list")


def _validate_surface_transform(plan_id: str, index: int, raw: dict[str, Any]) -> None:
    transform_id = str(raw.get("transform_id") or f"{plan_id}:surface_transforms[{index}]")
    if raw.get("kind") != "surface_follow":
        raise ValueError(f"{transform_id}: unsupported surface transform kind {raw.get('kind')!r}")
    for field_name in ("source_surface", "target_surface"):
        surface = raw.get(field_name)
        if not isinstance(surface, dict):
            raise ValueError(f"{transform_id}: {field_name} must be an object")
        for required in ("surface_id", "origin", "normal", "tangent_u", "tangent_v"):
            if required not in surface:
                raise ValueError(f"{transform_id}: {field_name}.{required} is required")
        for vector_name in ("origin", "normal", "tangent_u", "tangent_v"):
            vector = surface.get(vector_name)
            if not isinstance(vector, list) or len(vector) != 3:
                raise ValueError(f"{transform_id}: {field_name}.{vector_name} must have length 3")
    translation = raw.get("translation_world")
    if translation is not None and (not isinstance(translation, list) or len(translation) != 3):
        raise ValueError(f"{transform_id}: translation_world must have length 3")
    height_scale = raw.get("height_scale")
    if height_scale is not None and float(height_scale) <= 0.0:
        raise ValueError(f"{transform_id}: height_scale must be positive")
