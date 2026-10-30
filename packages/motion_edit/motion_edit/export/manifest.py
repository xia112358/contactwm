from __future__ import annotations

import json
from pathlib import Path

from motion_edit.layers import group_by_motion
from motion_edit.schema import SegmentRecord
from motion_edit.storage.schema import MotionVersionRecord


def _contact_edit_value(segment: SegmentRecord, key: str):
    if key in segment.metadata:
        return segment.metadata.get(key)
    edit = segment.metadata.get("contact_anchor_edit")
    if isinstance(edit, dict):
        return edit.get(key)
    return None


def export_motion_manifest(path: str | Path, segments: list[SegmentRecord]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    motions = []
    for motion_id, items in group_by_motion(segments).items():
        motion_path = next((item.motion_path or item.clip_npz for item in items if item.motion_path or item.clip_npz), "")
        motions.append(
            {
                "motion_id": motion_id,
                "motion_file": motion_path,
                "segments": [
                    {
                        "segment_id": item.segment_id,
                        "start_frame": item.start_frame,
                        "end_frame": item.end_frame,
                        "source": item.source,
                        "status": item.status,
                        "active": item.active,
                        "support": item.support,
                        "contact_start": item.contact_start,
                        "contact_end": item.contact_end,
                        "transition_type": item.metadata.get("transition_type"),
                        "source_anchor_id": item.metadata.get("source_anchor_id"),
                        "target_anchor_id": item.metadata.get("target_anchor_id"),
                        "old_anchor_world": _contact_edit_value(item, "old_world_position") or item.metadata.get("old_anchor_world"),
                        "new_anchor_world": _contact_edit_value(item, "new_world_position") or item.metadata.get("new_anchor_world"),
                        "delta_world": _contact_edit_value(item, "delta_world"),
                        "requested_delta_world": _contact_edit_value(item, "requested_delta_world"),
                        "tangent_delta": _contact_edit_value(item, "tangent_delta"),
                        "affected_frames": _contact_edit_value(item, "affected_frames"),
                        "surface_id": _contact_edit_value(item, "surface_id"),
                        "surface_normal": _contact_edit_value(item, "surface_normal"),
                        "surface_coordinates_before": _contact_edit_value(item, "surface_coordinates_before"),
                        "surface_coordinates_after": _contact_edit_value(item, "surface_coordinates_after"),
                        "constraint_mode": _contact_edit_value(item, "constraint_mode"),
                        "clamped": _contact_edit_value(item, "clamped"),
                        "active_body": item.metadata.get("active_body"),
                        "support_bodies": item.metadata.get("support_bodies"),
                        "contact_metadata": {
                            "transition": item.metadata.get("contact_transition"),
                            "event_count": len(item.metadata.get("contact_events") or []),
                            "anchor_count": len(item.metadata.get("contact_anchors") or []),
                            "patch_count": len(item.metadata.get("contact_patches") or []),
                            "patches": item.metadata.get("contact_patches") or [],
                            "anchor_edit": item.metadata.get("contact_anchor_edit"),
                        },
                    }
                    for item in items
                ],
            }
        )
    out.write_text(json.dumps({"schema_version": 1, "motions": motions}, indent=2), encoding="utf-8")
    return out


def export_motion_version_manifest(
    path: str | Path,
    *,
    version: MotionVersionRecord,
    segments: list[SegmentRecord],
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "motion_version_id": version.motion_version_id,
        "motion_path": version.motion_path,
        "contact_layer": version.contact_layer,
        "canonical_segment_path": version.canonical_segment_path,
        "token_catalog_path": version.token_catalog_path,
        "segments": [
            {
                "segment_id": item.segment_id,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "source": item.source,
                "status": item.status,
                "cut_source": item.metadata.get("cut_source"),
                "motion_version_id": item.metadata.get("motion_version_id"),
                "parent_transition_id": item.metadata.get("parent_transition_id"),
                "source_anchor_id": item.metadata.get("source_anchor_id"),
                "target_anchor_id": item.metadata.get("target_anchor_id"),
                "active_body": item.metadata.get("active_body"),
                "support_bodies": item.metadata.get("support_bodies"),
                "transition_type": item.metadata.get("transition_type"),
                "contact_metadata": {
                    "transition": item.metadata.get("contact_transition"),
                    "event_count": len(item.metadata.get("contact_events") or []),
                    "anchor_count": len(item.metadata.get("contact_anchors") or []),
                    "patch_count": len(item.metadata.get("contact_patches") or []),
                },
            }
            for item in segments
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
