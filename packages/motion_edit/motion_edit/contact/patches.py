from __future__ import annotations

from motion_edit.contact.schema import ContactAnchorRecord, ContactPatchRecord

FOOT_KEYWORDS = ("foot", "ankle", "toe", "lf", "rf", "left_foot", "right_foot")
HAND_KEYWORDS = ("hand", "wrist", "palm", "lh", "rh", "left_hand", "right_hand")


def patch_type_for_body(body: str) -> str:
    normalized = body.lower()
    if any(token in normalized for token in FOOT_KEYWORDS):
        return "foot"
    if any(token in normalized for token in HAND_KEYWORDS):
        return "hand"
    return "unknown"


def link_names_for_patch_type(patch_type: str, body: str) -> list[str] | None:
    if patch_type == "foot":
        return [body, f"{body}_sole"]
    if patch_type == "hand":
        return [body, f"{body}_palm"]
    return [body]


def patches_from_anchors(anchors: list[ContactAnchorRecord]) -> list[ContactPatchRecord]:
    patches: list[ContactPatchRecord] = []
    for anchor in anchors:
        patch_type = patch_type_for_body(anchor.body)
        patch = ContactPatchRecord(
            motion_id=anchor.motion_id,
            patch_id=f"{anchor.anchor_id}_patch",
            body=anchor.body,
            start_frame=anchor.start_frame,
            end_frame=anchor.end_frame,
            patch_type=patch_type,
            patch_center_world=anchor.world_position,
            link_names=link_names_for_patch_type(patch_type, anchor.body),
            anchor_id=anchor.anchor_id,
            slip_score=anchor.metadata.get("mean_drift_xy"),
            metadata={"role": anchor.role, "source_anchor_id": anchor.anchor_id},
        )
        patch.validate()
        patches.append(patch)
    return sorted(patches, key=lambda patch: (patch.start_frame, patch.end_frame, patch.body, patch.patch_id))
