from __future__ import annotations

from motion_edit.schema import SegmentRecord
from motion_edit.storage.segments import get_segment_motion_version_id
from motion_edit.storage.schema import TokenRecord


def _safe_part(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        return "+".join(str(item) for item in value) if value else "none"
    text = str(value)
    return text if text else "unknown"


def token_family_for_segment(segment: SegmentRecord) -> str:
    active_body = segment.metadata.get("active_body") or segment.active
    support_bodies = segment.metadata.get("support_bodies") or segment.support
    transition_type = segment.metadata.get("transition_type") or "unknown"
    return f"{_safe_part(active_body)}__{_safe_part(support_bodies)}__{_safe_part(transition_type)}"


def continuous_params_for_segment(segment: SegmentRecord) -> dict:
    edit = segment.metadata.get("contact_anchor_edit")
    if not isinstance(edit, dict):
        edit = {}
    params = {
        "duration": int(segment.end_frame - segment.start_frame),
    }
    for key in [
        "delta_world",
        "requested_delta_world",
        "tangent_delta",
        "surface_coordinates_before",
        "surface_coordinates_after",
    ]:
        value = segment.metadata.get(key, edit.get(key))
        if value is not None:
            params[key] = value
    source_anchor = segment.metadata.get("source_anchor_id")
    target_anchor = segment.metadata.get("target_anchor_id")
    if source_anchor and target_anchor:
        params["anchor_pair"] = [source_anchor, target_anchor]
    return params


def token_from_segment(segment: SegmentRecord, *, motion_version_id: str) -> TokenRecord:
    segment_motion_version_id = get_segment_motion_version_id(segment) or motion_version_id
    transition = segment.metadata.get("contact_transition")
    parent_transition_id = segment.metadata.get("parent_transition_id")
    if parent_transition_id is None and isinstance(transition, dict):
        parent_transition_id = transition.get("transition_id")
    token = TokenRecord(
        token_id=f"{segment_motion_version_id}_{segment.segment_id}",
        motion_version_id=segment_motion_version_id,
        segment_id=segment.segment_id,
        token_family=token_family_for_segment(segment),
        active_body=segment.metadata.get("active_body") or segment.active,
        support_bodies=list(segment.metadata.get("support_bodies") or ([] if segment.support is None else [segment.support])),
        source_anchor_id=segment.metadata.get("source_anchor_id"),
        target_anchor_id=segment.metadata.get("target_anchor_id"),
        parent_transition_id=parent_transition_id,
        continuous_params=continuous_params_for_segment(segment),
        status=segment.status,
        metadata={
            "cut_source": segment.metadata.get("cut_source"),
        },
    )
    token.validate()
    return token


def build_tokens_from_segments(motion_version_id: str, segments: list[SegmentRecord]) -> list[TokenRecord]:
    return [token_from_segment(segment, motion_version_id=motion_version_id) for segment in segments]
