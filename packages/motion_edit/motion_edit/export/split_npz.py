from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from motion_edit.adapters.holosoma_npz import load_motion_npz, subset_arrays
from motion_edit.schema import SegmentRecord


def _contact_edit_value(segment: SegmentRecord, key: str):
    if key in segment.metadata:
        return segment.metadata.get(key)
    edit = segment.metadata.get("contact_anchor_edit")
    if isinstance(edit, dict):
        return edit.get(key)
    return None


def export_split_npz(output_dir: str | Path, segments: list[SegmentRecord]) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for segment in segments:
        source = segment.motion_path or segment.clip_npz
        if not source:
            continue
        arrays = load_motion_npz(source)
        clipped = subset_arrays(arrays, segment.start_frame, segment.end_frame)
        clipped["motion_edit_segment_id"] = np.asarray(segment.segment_id)
        clipped["motion_edit_motion_version_id"] = np.asarray(str(segment.metadata.get("motion_version_id") or ""))
        clipped["motion_edit_token_id"] = np.asarray(str(segment.metadata.get("token_id") or ""))
        clipped["motion_edit_source_motion"] = np.asarray(str(source))
        clipped["motion_edit_contact_transition"] = np.asarray(
            json.dumps(segment.metadata.get("contact_transition"), sort_keys=True)
            if segment.metadata.get("contact_transition") is not None
            else ""
        )
        clipped["motion_edit_active_body"] = np.asarray(str(segment.metadata.get("active_body") or ""))
        clipped["motion_edit_support_bodies"] = np.asarray(
            json.dumps(segment.metadata.get("support_bodies") or [], sort_keys=True)
        )
        clipped["motion_edit_source_anchor_id"] = np.asarray(str(segment.metadata.get("source_anchor_id") or ""))
        clipped["motion_edit_target_anchor_id"] = np.asarray(str(segment.metadata.get("target_anchor_id") or ""))
        clipped["motion_edit_old_anchor_world"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "old_world_position") or _contact_edit_value(segment, "old_anchor_world") or [], sort_keys=True)
        )
        clipped["motion_edit_new_anchor_world"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "new_world_position") or _contact_edit_value(segment, "new_anchor_world") or [], sort_keys=True)
        )
        clipped["motion_edit_delta_world"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "delta_world") or [], sort_keys=True)
        )
        clipped["motion_edit_requested_delta_world"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "requested_delta_world") or [], sort_keys=True)
        )
        clipped["motion_edit_tangent_delta"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "tangent_delta") or [], sort_keys=True)
        )
        clipped["motion_edit_surface_id"] = np.asarray(str(_contact_edit_value(segment, "surface_id") or ""))
        clipped["motion_edit_surface_normal"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "surface_normal") or [], sort_keys=True)
        )
        clipped["motion_edit_surface_coordinates_before"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "surface_coordinates_before") or {}, sort_keys=True)
        )
        clipped["motion_edit_surface_coordinates_after"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "surface_coordinates_after") or {}, sort_keys=True)
        )
        clipped["motion_edit_constraint_mode"] = np.asarray(str(_contact_edit_value(segment, "constraint_mode") or ""))
        clipped["motion_edit_clamped"] = np.asarray(bool(_contact_edit_value(segment, "clamped") or False))
        clipped["motion_edit_affected_frames"] = np.asarray(
            json.dumps(_contact_edit_value(segment, "affected_frames") or [], sort_keys=True)
        )
        clipped["motion_edit_contact_patches"] = np.asarray(
            json.dumps(segment.metadata.get("contact_patches") or [], sort_keys=True)
        )
        out = root / f"{segment.segment_id}_frames_{segment.start_frame:04d}_{segment.end_frame:04d}.npz"
        np.savez(out, **clipped)
        written.append(out)
    return written
