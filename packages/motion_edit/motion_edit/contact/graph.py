from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from motion_edit.contact.patches import patches_from_anchors
from motion_edit.contact.schema import ContactAnchorRecord, ContactEventRecord, ContactPatchRecord, ContactTransitionRecord
from motion_edit.contact.stable_proto import transitions_from_stable_contact_anchors
from motion_edit.contact.transitions import transitions_from_anchor_pairs, transitions_from_event_pairs, transitions_from_proto_indices


@dataclass(frozen=True)
class ContactGraph:
    motion_id: str
    events: list[ContactEventRecord] = field(default_factory=list)
    anchors: list[ContactAnchorRecord] = field(default_factory=list)
    patches: list[ContactPatchRecord] = field(default_factory=list)
    transitions: list[ContactTransitionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "motion_id": self.motion_id,
            "events": [event.to_dict() for event in self.events],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "patches": [patch.to_dict() for patch in self.patches],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


def contact_graph_from_masks(
    *,
    motion_id: str,
    contact_mask: np.ndarray | None,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_pos_w: np.ndarray | None = None,
    proto_starts: Iterable[int] | None = None,
    proto_ends: Iterable[int] | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "contact_mask",
) -> ContactGraph:
    starts = list(proto_starts) if proto_starts is not None else []
    ends = list(proto_ends) if proto_ends is not None else []
    events, anchors, transitions = transitions_from_proto_indices(
        motion_id=motion_id,
        starts=starts,
        ends=ends,
        contact_mask=contact_mask,
        active_mask=active_mask,
        support_mask=support_mask,
        body_pos_w=body_pos_w,
        body_names=body_names,
        source=source,
    )
    if not transitions:
        transitions = transitions_from_stable_contact_anchors(
            motion_id=motion_id,
            anchors=anchors,
            events=events,
            contact_mask=contact_mask,
            active_mask=active_mask,
            support_mask=support_mask,
            body_pos_w=body_pos_w,
            body_names=body_names,
            source=source,
        )
    if not transitions:
        transitions = transitions_from_event_pairs(
            motion_id=motion_id,
            events=events,
            anchors=anchors,
            source=source,
        )
    if not transitions:
        transitions = transitions_from_anchor_pairs(
            motion_id=motion_id,
            anchors=anchors,
            events=events,
            contact_mask=contact_mask,
            active_mask=active_mask,
            support_mask=support_mask,
            body_names=body_names,
            source=source,
        )
    from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
    # Graph roles are editing/planning intents. Legacy automatic keep proposals
    # never establish measured load or a stationary physical support point.
    patches = patches_from_anchors(anchors)
    for record in (*events, *anchors, *patches, *transitions):
        record.metadata['support_assessment_schema'] = SUPPORT_ASSESSMENT_SCHEMA
        record.metadata['support_role_semantics'] = 'contact_keep_intent'
        record.metadata['actual_support_status'] = 'unknown_without_solver_loads'
    return ContactGraph(motion_id=motion_id, events=events, anchors=anchors, patches=patches, transitions=transitions)
