from __future__ import annotations

from typing import Iterable

import numpy as np

from motion_edit.contact.schema import ContactEventRecord


def body_names_for_mask(mask: np.ndarray, body_names: Iterable[str] | None = None) -> list[str]:
    arr = np.asarray(mask)
    if arr.ndim == 1:
        count = arr.shape[0]
    elif arr.ndim >= 2:
        count = arr.shape[1]
    else:
        count = 0
    names = list(body_names or [])
    if len(names) < count:
        names.extend(f"body_{index}" for index in range(len(names), count))
    return names[:count]


def bodies_from_mask(row: np.ndarray, body_names: list[str]) -> list[str]:
    return [body_names[index] for index, value in enumerate(np.asarray(row).reshape(-1)) if bool(value)]


def _as_2d_bool(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim != 2:
        raise ValueError(f"contact mask must be 1D or 2D, got shape={arr.shape}")
    return arr


def detect_contact_events(
    *,
    motion_id: str,
    contact_mask: np.ndarray | None,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "contact_mask",
) -> list[ContactEventRecord]:
    contact = _as_2d_bool(contact_mask)
    active = _as_2d_bool(active_mask)
    support = _as_2d_bool(support_mask)
    reference = next(mask for mask in (contact, active, support) if mask is not None)
    names = body_names_for_mask(reference, body_names)
    events: list[ContactEventRecord] = []

    if contact is not None:
        for frame in range(1, contact.shape[0]):
            before = contact[frame - 1]
            after = contact[frame]
            before_bodies = bodies_from_mask(before, names)
            after_bodies = bodies_from_mask(after, names)
            for body_index, body in enumerate(names):
                if not before[body_index] and after[body_index]:
                    events.append(
                        ContactEventRecord(
                            motion_id=motion_id,
                            event_id=f"{motion_id}_touchdown_{body}_{frame:06d}",
                            frame=frame,
                            body=body,
                            event_type="touchdown",
                            contact_before=before_bodies,
                            contact_after=after_bodies,
                            source=source,
                            confidence=1.0,
                        )
                    )
                elif before[body_index] and not after[body_index]:
                    events.append(
                        ContactEventRecord(
                            motion_id=motion_id,
                            event_id=f"{motion_id}_liftoff_{body}_{frame:06d}",
                            frame=frame,
                            body=body,
                            event_type="liftoff",
                            contact_before=before_bodies,
                            contact_after=after_bodies,
                            source=source,
                            confidence=1.0,
                        )
                    )

    for frame in range(1, active.shape[0] if active is not None else 0):
        before = active[frame - 1]
        after = active[frame]
        if np.array_equal(before, after):
            continue
        events.append(
            ContactEventRecord(
                motion_id=motion_id,
                event_id=f"{motion_id}_active_change_{frame:06d}",
                frame=frame,
                body="active",
                event_type="active_change",
                contact_before=bodies_from_mask(before, names),
                contact_after=bodies_from_mask(after, names),
                source=source,
                confidence=1.0,
            )
        )

    for frame in range(1, support.shape[0] if support is not None else 0):
        before = support[frame - 1]
        after = support[frame]
        if np.array_equal(before, after):
            continue
        events.append(
            ContactEventRecord(
                motion_id=motion_id,
                event_id=f"{motion_id}_support_switch_{frame:06d}",
                frame=frame,
                body="support",
                event_type="support_switch",
                contact_before=bodies_from_mask(before, names),
                contact_after=bodies_from_mask(after, names),
                source=source,
                confidence=1.0,
            )
        )

    return sorted(events, key=lambda event: (event.frame, event.event_type, event.body, event.event_id))

