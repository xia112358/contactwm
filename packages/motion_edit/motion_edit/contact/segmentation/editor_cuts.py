from __future__ import annotations

from dataclasses import dataclass, replace

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.schema import ContactAnchorRecord, ContactTransitionRecord
from motion_edit.contact.stable_proto import STABLE_PROTO_ENDPOINT_POLICY, STABLE_PROTO_KIND, StableContactProtoConfig


_INITIAL_FRAME = 0
_PARENT_BODY_ORDER = ("left_foot", "right_foot", "left_hand", "right_hand", "left_knee", "right_knee")


@dataclass(frozen=True)
class EditorCutConfig:
    min_proto_segment_frames: int = 20
    same_parent_body_merge_gap: int | None = None
    cluster_window: int | None = None


def _resolve_editor_cut_config(stable_cfg: StableContactProtoConfig, cut_config: EditorCutConfig | None) -> EditorCutConfig:
    cfg = cut_config or EditorCutConfig()
    cluster_window = int(cfg.cluster_window) if cfg.cluster_window is not None else int(stable_cfg.stable_touchdown_cluster_window)
    same_parent_body_merge_gap = (
        int(cfg.same_parent_body_merge_gap)
        if cfg.same_parent_body_merge_gap is not None
        else max(int(stable_cfg.merge_transition_window), int(stable_cfg.stable_touchdown_cluster_window))
    )
    return EditorCutConfig(
        min_proto_segment_frames=int(cfg.min_proto_segment_frames),
        same_parent_body_merge_gap=same_parent_body_merge_gap,
        cluster_window=cluster_window,
    )


def _parent_body(body: str) -> str:
    lowered = str(body).lower()
    foot_channel_parents = {
        "left_heel": "left_foot",
        "left_toe": "left_foot",
        "right_heel": "right_foot",
        "right_toe": "right_foot",
        "lhee": "left_foot",
        "ltoe": "left_foot",
        "rhee": "right_foot",
        "rtoe": "right_foot",
    }
    if lowered in foot_channel_parents:
        return foot_channel_parents[lowered]
    for foot in ("left_foot", "right_foot"):
        if lowered == foot or lowered.startswith(f"{foot}_") or foot in lowered:
            return foot
    for hand in ("left_hand", "right_hand"):
        if lowered == hand or lowered.startswith(f"{hand}_") or hand in lowered:
            return hand
    for knee in ("left_knee", "right_knee"):
        if lowered == knee or lowered.startswith(f"{knee}_") or knee in lowered:
            return knee
    aliases = {
        "lf": "left_foot",
        "rf": "right_foot",
        "lh": "left_hand",
        "rh": "right_hand",
        "lk": "left_knee",
        "rk": "right_knee",
    }
    return aliases.get(lowered, lowered)


def _is_active_motion_body(body: str, active_parts: set[str]) -> bool:
    return _parent_body(body) in active_parts


def _body_sort_key(body: str) -> tuple[int, str]:
    parent = _parent_body(body)
    try:
        return (_PARENT_BODY_ORDER.index(parent), parent)
    except ValueError:
        return (len(_PARENT_BODY_ORDER), parent)


def _contact_bodies_at(anchors: list[ContactAnchorRecord], frame: int) -> list[str]:
    bodies = {
        _parent_body(anchor.body)
        for anchor in anchors
        if int(anchor.start_frame) <= int(frame) < int(anchor.end_frame)
    }
    return sorted(bodies, key=_body_sort_key)


def _anchor_for_body_at(
    anchors: list[ContactAnchorRecord],
    *,
    body: str | None,
    frame: int,
) -> ContactAnchorRecord | None:
    if body is None:
        return None
    parent = _parent_body(body)
    return next(
        (
            anchor
            for anchor in anchors
            if _parent_body(anchor.body) == parent and int(anchor.start_frame) <= int(frame) < int(anchor.end_frame)
        ),
        None,
    )


def _last_anchor_before(
    anchors: list[ContactAnchorRecord],
    *,
    body: str | None,
    frame: int,
) -> ContactAnchorRecord | None:
    if body is None:
        return None
    parent = _parent_body(body)
    candidates = [
        anchor
        for anchor in anchors
        if _parent_body(anchor.body) == parent and int(anchor.end_frame) <= int(frame)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda anchor: (int(anchor.end_frame), int(anchor.start_frame), anchor.anchor_id))


def _merge_intervals_for_parent(
    anchors: list[ContactAnchorRecord],
    *,
    max_gap: int,
) -> list[tuple[int, int, list[ContactAnchorRecord]]]:
    ordered = sorted(anchors, key=lambda anchor: (int(anchor.start_frame), int(anchor.end_frame), anchor.anchor_id))
    merged: list[tuple[int, int, list[ContactAnchorRecord]]] = []
    for anchor in ordered:
        start = int(anchor.start_frame)
        end = int(anchor.end_frame)
        if end <= start:
            continue
        if not merged or start - merged[-1][1] > int(max_gap):
            merged.append((start, end, [anchor]))
            continue
        prev_start, prev_end, prev_anchors = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), [*prev_anchors, anchor])
    return merged


def _merge_short_cut_clusters(
    clusters: dict[int, list[ContactAnchorRecord]],
    cluster_bodies: dict[int, list[str]],
    *,
    min_segment_frames: int,
) -> tuple[dict[int, list[ContactAnchorRecord]], dict[int, list[str]]]:
    if int(min_segment_frames) <= 0 or len(clusters) <= 1:
        return clusters, cluster_bodies
    merged_clusters: dict[int, list[ContactAnchorRecord]] = {}
    merged_bodies: dict[int, list[str]] = {}
    active_frame: int | None = None
    active_items: list[ContactAnchorRecord] = []
    active_bodies: set[str] = set()
    for frame in sorted(clusters):
        if active_frame is None:
            active_frame = frame
            active_items = list(clusters[frame])
            active_bodies = set(cluster_bodies.get(frame, []))
            continue
        if frame - active_frame < int(min_segment_frames):
            active_frame = frame
            active_items.extend(clusters[frame])
            active_bodies.update(cluster_bodies.get(frame, []))
            continue
        merged_clusters[active_frame] = list(active_items)
        merged_bodies[active_frame] = sorted(active_bodies, key=_body_sort_key)
        active_frame = frame
        active_items = list(clusters[frame])
        active_bodies = set(cluster_bodies.get(frame, []))
    if active_frame is not None:
        merged_clusters[active_frame] = list(active_items)
        merged_bodies[active_frame] = sorted(active_bodies, key=_body_sort_key)
    return merged_clusters, merged_bodies


def _cluster_contact_starts(
    anchors: list[ContactAnchorRecord],
    *,
    active_parts: set[str],
    cluster_window: int,
    same_body_gap: int,
    min_segment_frames: int,
) -> tuple[list[int], dict[int, list[ContactAnchorRecord]], dict[int, list[str]]]:
    by_parent: dict[str, list[ContactAnchorRecord]] = {}
    for anchor in anchors:
        parent = _parent_body(anchor.body)
        if parent not in active_parts:
            continue
        by_parent.setdefault(parent, []).append(anchor)

    parent_events: list[tuple[int, str, list[ContactAnchorRecord]]] = []
    for parent, parent_anchors in by_parent.items():
        for start, _end, group in _merge_intervals_for_parent(parent_anchors, max_gap=same_body_gap):
            if start <= _INITIAL_FRAME:
                continue
            parent_events.append((int(start), parent, group))
    parent_events.sort(key=lambda item: (item[0], _body_sort_key(item[1])))

    clusters: dict[int, list[ContactAnchorRecord]] = {}
    cluster_bodies: dict[int, list[str]] = {}
    cluster_frame: int | None = None
    cluster_items: list[ContactAnchorRecord] = []
    cluster_parents: set[str] = set()
    for frame, parent, group in parent_events:
        if cluster_frame is None:
            cluster_frame = frame
            cluster_items = list(group)
            cluster_parents = {parent}
            continue
        if frame - cluster_frame <= int(cluster_window):
            cluster_frame = frame
            cluster_items.extend(group)
            cluster_parents.add(parent)
            continue
        clusters[cluster_frame] = list(cluster_items)
        cluster_bodies[cluster_frame] = sorted(cluster_parents, key=_body_sort_key)
        cluster_frame = frame
        cluster_items = list(group)
        cluster_parents = {parent}
    if cluster_frame is not None:
        clusters[cluster_frame] = list(cluster_items)
        cluster_bodies[cluster_frame] = sorted(cluster_parents, key=_body_sort_key)
    clusters, cluster_bodies = _merge_short_cut_clusters(
        clusters,
        cluster_bodies,
        min_segment_frames=int(min_segment_frames),
    )
    return sorted(clusters), clusters, cluster_bodies


def _fallback_transitions(
    fallback: list[ContactTransitionRecord],
    *,
    reason: str,
) -> list[ContactTransitionRecord]:
    out: list[ContactTransitionRecord] = []
    for transition in fallback:
        metadata = dict(transition.metadata or {})
        metadata.setdefault("contact_source", "fallback_existing_transitions")
        metadata.setdefault("contact_phase_scope", "legacy_transition")
        metadata.setdefault("cut_rebuild_failed", True)
        metadata.setdefault("cut_rebuild_reason", reason)
        out.append(replace(transition, metadata=metadata))
    return out


def stable_proto_transitions_for_editor(
    *,
    graph: ContactGraph,
    motion: str,
    fps: int,
    fallback: list[ContactTransitionRecord],
    cut_config: EditorCutConfig | None = None,
) -> list[ContactTransitionRecord]:
    """Rebuild editor timeline cuts from cleaned body-level contact phases.

    Contact-editor cleanup has already refined, split, filtered, and surface-bound
    contact points. Foot heel/toe/sole records are editable patch handles for the
    same limb contact phase, so they are unioned at the parent foot body before
    timeline cut frames are derived. The raw force mask is not re-parsed here.
    """

    _ = motion
    cfg = StableContactProtoConfig(fps=float(fps))
    editor_cfg = _resolve_editor_cut_config(cfg, cut_config)
    anchors = sorted(graph.anchors, key=lambda anchor: (int(anchor.start_frame), int(anchor.end_frame), anchor.body, anchor.anchor_id))
    active_parts = set(cfg.active_motion_parts)
    cut_frames, anchors_by_cut, bodies_by_cut = _cluster_contact_starts(
        anchors,
        active_parts=active_parts,
        cluster_window=int(editor_cfg.cluster_window or 0),
        same_body_gap=int(editor_cfg.same_parent_body_merge_gap or 0),
        min_segment_frames=int(editor_cfg.min_proto_segment_frames),
    )
    if not cut_frames:
        return _fallback_transitions(fallback, reason="no_parent_limb_contact_starts")

    n_frames = max((int(anchor.end_frame) for anchor in anchors), default=0)
    stable_anchor_frames = [_INITIAL_FRAME, *cut_frames]
    if n_frames > 0:
        stable_anchor_frames.append(int(n_frames))
    stable_anchor_frames = sorted(dict.fromkeys(stable_anchor_frames))

    all_parent_bodies = {_parent_body(anchor.body) for anchor in anchors}
    transitions: list[ContactTransitionRecord] = []
    previous_frame = _INITIAL_FRAME
    for proto_index, cut_frame in enumerate(cut_frames):
        cluster_anchors = sorted(anchors_by_cut[cut_frame], key=lambda anchor: (_body_sort_key(anchor.body), int(anchor.start_frame), anchor.anchor_id))
        active_bodies = list(bodies_by_cut.get(cut_frame, []))
        active_body = active_bodies[0] if active_bodies else None
        support_bodies = [body for body in _contact_bodies_at(anchors, previous_frame) if body not in set(active_bodies)]
        free_bodies = sorted(
            all_parent_bodies - set(active_bodies) - set(support_bodies),
            key=_body_sort_key,
        )
        target_anchor = cluster_anchors[0] if cluster_anchors else None
        source_anchor = _anchor_for_body_at(anchors, body=active_body, frame=previous_frame)
        if source_anchor is None:
            source_anchor = _last_anchor_before(anchors, body=active_body, frame=cut_frame)
        transition = ContactTransitionRecord(
            motion_id=graph.motion_id,
            transition_id=f"{graph.motion_id}_contact_editor_anchor_proto_{proto_index:04d}",
            start_frame=int(previous_frame),
            end_frame=int(cut_frame),
            active_body=active_body,
            support_bodies=support_bodies,
            start_event_id=None,
            end_event_id=None,
            source_anchor_id=source_anchor.anchor_id if source_anchor else None,
            target_anchor_id=target_anchor.anchor_id if target_anchor else None,
            transition_type="support_transfer" if active_bodies else "unknown",
            source="contact_editor_anchor_proto",
            metadata={
                "segmentation_kind": STABLE_PROTO_KIND,
                "endpoint_policy": STABLE_PROTO_ENDPOINT_POLICY,
                "contact_source": "cleaned_contact_points",
                "contact_phase_scope": "parent_limb_union",
                "foot_patch_policy": "heel_toe_sole_do_not_cut",
                "outside_policy": "copy_original",
                "proto_index": int(proto_index),
                "anchor_start": int(previous_frame),
                "anchor_end": int(cut_frame),
                "active_bodies": active_bodies,
                "support_bodies": support_bodies,
                "free_bodies": free_bodies,
                "owned_bodies": active_bodies,
                "cluster_anchor_ids": [anchor.anchor_id for anchor in cluster_anchors],
                "stable_anchor_frames": stable_anchor_frames,
                "config": {
                    "stable_touchdown_cluster_window": int(cfg.stable_touchdown_cluster_window),
                    "cluster_window": int(editor_cfg.cluster_window or 0),
                    "same_parent_body_merge_gap": int(editor_cfg.same_parent_body_merge_gap or 0),
                    "min_proto_segment_frames": int(editor_cfg.min_proto_segment_frames),
                    "active_motion_parts": list(cfg.active_motion_parts),
                    "source": "cleaned_contact_points",
                },
            },
        )
        transition.validate()
        transitions.append(transition)
        previous_frame = int(cut_frame)
    return transitions or _fallback_transitions(fallback, reason="empty_generated_transitions")
