#!/usr/bin/env python3
"""Export 50 Hz contact-rule diagnostics for OmniRetarget terrain motions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from somaforge_core.robot_assets import canonical_g1_urdf_path, validate_canonical_g1_urdf

DATASET_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from extract_omniretarget_cram_proto import (  # noqa: E402
    CONTACT_PROBE_FRAMES,
    DEFAULT_CONTACT_PARTS,
    PART_ORDER,
    _clean_contact,
    _contact_masks,
    _debounced_contact,
    _holosoma_motion_info,
    _load_context,
    _nearest_surface_distance,
    _point_box_signed_distance,
    _resample_context,
    _top_gap,
)

ACTIVE_ROLE_PARTS = ("left_foot", "right_foot", "left_hand", "right_hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("robot-terrain"))
    parser.add_argument("--holosoma-motion-dir", type=Path, default=Path("data/holosoma_motions_50hz"))
    parser.add_argument("--robot-urdf", type=Path, default=None)
    parser.add_argument("--motions", nargs="*", default=["climb_00_z_scale_1.0.npz"])
    parser.add_argument("--contact-parts", nargs="+", default=DEFAULT_CONTACT_PARTS, choices=PART_ORDER)
    parser.add_argument("--merge-transition-window", type=int, default=2)
    parser.add_argument("--hand-merge-transition-window", type=int, default=2)
    parser.add_argument("--body-merge-transition-window", type=int, default=2)
    parser.add_argument("--support-contact-count", type=int, default=3)
    parser.add_argument("--support-window", type=int, default=3)
    parser.add_argument("--support-vz-thresh", type=float, default=0.16)
    parser.add_argument("--support-xy-speed-thresh", type=float, default=0.12)
    parser.add_argument("--motion-start-vz-thresh", type=float, default=0.05)
    parser.add_argument("--motion-start-lookback", type=int, default=20)
    parser.add_argument("--anchor-min-xy-displacement", type=float, default=0.05)
    parser.add_argument("--event-free-window", type=int, default=3)
    parser.add_argument("--event-contact-window", type=int, default=3)
    parser.add_argument("--event-stable-speed-thresh", type=float, default=0.12)
    parser.add_argument("--event-cluster-window", type=int, default=6)
    parser.add_argument("--event-cluster-all-limbs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-free-state-window", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_contact_diagnostics_50hz"))
    parser.add_argument("--holosoma-motion-suffix", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _motion_paths(data_dir: Path, motions: list[str]) -> list[Path]:
    paths = [Path(m) if Path(m).is_absolute() else data_dir / m for m in motions]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing motion file(s): {missing}")
    return paths


def _nearest_surface_with_box(point: np.ndarray, boxes) -> tuple[float, int]:
    distances = np.asarray([_point_box_signed_distance(point, box) for box in boxes], dtype=np.float64)
    box_i = int(np.argmin(np.abs(distances)))
    return float(distances[box_i]), box_i


def _top_gap_with_box(point: np.ndarray, boxes) -> tuple[float, int, float]:
    candidates: list[tuple[float, int, float]] = []
    for box_i, box in enumerate(boxes):
        dxy = point[:2] - box.center_xy
        inside_x = abs(float(dxy @ box.axis_x)) <= box.half_x + 0.04
        inside_y = abs(float(dxy @ box.axis_y)) <= box.half_y + 0.04
        if inside_x and inside_y:
            candidates.append((float(point[2]) - box.z_max, box_i, float(box.z_max)))
    if not candidates:
        candidates.append((float(point[2]), -1, 0.0))
    return min(candidates, key=lambda item: abs(item[0]))


def _probe_name(part: str, probe_i: int) -> str:
    names = CONTACT_PROBE_FRAMES.get(part, ())
    if probe_i < len(names):
        return names[probe_i]
    return f"{part}_point_{probe_i}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sustained(mask: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.zeros(len(mask), dtype=bool)
    for frame in range(0, len(mask) - window + 1):
        out[frame] = bool(mask[frame : frame + window].all())
    return out


def _motion_start_frame(
    support_break: int,
    stable_support: np.ndarray,
    center_vz: np.ndarray,
    contact_count: np.ndarray,
    vz_thresh: float,
    lookback: int,
) -> int:
    start = max(0, support_break - max(0, int(lookback)))
    candidate = support_break
    for frame in range(support_break - 1, start - 1, -1):
        if not stable_support[frame]:
            break
        if abs(float(center_vz[frame])) >= vz_thresh or contact_count[frame] < contact_count.max():
            candidate = frame
    return candidate


def _foot_support_diagnostics(
    motion_name: str,
    part: str,
    part_i: int,
    fps: float,
    probes: np.ndarray,
    probe_rule: np.ndarray,
    min_contact_count: int,
    support_window: int,
    vz_thresh: float,
    xy_speed_thresh: float,
    motion_start_vz_thresh: float,
    motion_start_lookback: int,
) -> tuple[list[dict], list[dict]]:
    center = probes.mean(axis=1)
    center_vel = np.zeros_like(center)
    center_vel[1:] = np.diff(center, axis=0) * fps
    center_xy_speed = np.linalg.norm(center_vel[:, :2], axis=1)
    center_vz = center_vel[:, 2]
    contact_count = probe_rule.sum(axis=1).astype(np.int64)
    geometric_contact = contact_count > 0
    stable_instant = (
        (contact_count >= min_contact_count)
        & (np.abs(center_vz) <= vz_thresh)
        & (center_xy_speed <= xy_speed_thresh)
    )
    stable_support = _sustained(stable_instant, support_window)

    frame_rows: list[dict] = []
    for frame in range(len(probes)):
        frame_rows.append(
            {
                "motion": motion_name,
                "frame": frame,
                "time_s": frame / fps,
                "fps": fps,
                "part": part,
                "part_index": part_i,
                "contact_count": int(contact_count[frame]),
                "geometric_contact": int(geometric_contact[frame]),
                "stable_support_instant": int(stable_instant[frame]),
                "stable_support": int(stable_support[frame]),
                "center_x": float(center[frame, 0]),
                "center_y": float(center[frame, 1]),
                "center_z": float(center[frame, 2]),
                "center_xy_speed": float(center_xy_speed[frame]),
                "center_vz": float(center_vz[frame]),
            }
        )

    transition_rows: list[dict] = []
    in_support = False
    frame = 0
    while frame < len(probes):
        if not in_support:
            if stable_support[frame]:
                in_support = True
            frame += 1
            continue

        if stable_support[frame]:
            frame += 1
            continue

        support_break = frame
        motion_start = _motion_start_frame(
            support_break,
            stable_instant,
            center_vz,
            contact_count,
            motion_start_vz_thresh,
            motion_start_lookback,
        )

        air_start = ""
        first_touch = ""
        cursor = support_break
        while cursor < len(probes) and not stable_support[cursor]:
            if air_start == "" and not geometric_contact[cursor]:
                air_start = cursor
            if air_start != "" and first_touch == "" and geometric_contact[cursor]:
                first_touch = cursor
            cursor += 1
        if cursor >= len(probes):
            break

        new_support = cursor
        if air_start != "" and first_touch == "" and geometric_contact[new_support]:
            first_touch = new_support
        start_center = center[motion_start]
        end_center = center[new_support]
        span = center[motion_start : new_support + 1]
        transition_rows.append(
            {
                "motion": motion_name,
                "part": part,
                "part_index": part_i,
                "motion_start": motion_start,
                "support_break": support_break,
                "air_start": air_start,
                "first_touch": first_touch,
                "new_support": new_support,
                "motion_start_time_s": motion_start / fps,
                "support_break_time_s": support_break / fps,
                "air_start_time_s": "" if air_start == "" else air_start / fps,
                "first_touch_time_s": "" if first_touch == "" else first_touch / fps,
                "new_support_time_s": new_support / fps,
                "transition_frames": new_support - motion_start + 1,
                "air_frames": "" if air_start == "" or first_touch == "" else first_touch - air_start,
                "xy_displacement_m": float(np.linalg.norm(end_center[:2] - start_center[:2])),
                "z_lift_m": float(span[:, 2].max() - start_center[2]),
                "start_contact_count": int(contact_count[motion_start]),
                "break_contact_count": int(contact_count[support_break]),
                "new_support_contact_count": int(contact_count[new_support]),
                "max_xy_speed_mps": float(center_xy_speed[motion_start : new_support + 1].max()),
                "max_abs_vz_mps": float(np.abs(center_vz[motion_start : new_support + 1]).max()),
            }
        )
        frame = new_support + support_window
        in_support = True

    return frame_rows, transition_rows


def _support_transition_anchor_rows(
    transition_rows: list[dict],
    min_xy_displacement: float,
) -> list[dict]:
    anchor_rows: list[dict] = []
    kept_transition_id = 0
    for row in transition_rows:
        has_air = row["air_start"] != "" and row["first_touch"] != ""
        has_motion = float(row["xy_displacement_m"]) >= min_xy_displacement
        if not has_air and not has_motion:
            continue

        for anchor_type, anchor_key in (
            ("motion_start", "motion_start"),
            ("new_support", "new_support"),
        ):
            anchor_frame = int(row[anchor_key])
            anchor_rows.append(
                {
                    "motion": row["motion"],
                    "part": row["part"],
                    "part_index": row["part_index"],
                    "transition_id": kept_transition_id,
                    "anchor_type": anchor_type,
                    "anchor_frame": anchor_frame,
                    "anchor_time_s": row[f"{anchor_key}_time_s"],
                    "segment_start": row["motion_start"],
                    "segment_end": row["new_support"],
                    "motion_start": row["motion_start"],
                    "support_break": row["support_break"],
                    "air_start": row["air_start"],
                    "first_touch": row["first_touch"],
                    "new_support": row["new_support"],
                    "air_frames": row["air_frames"],
                    "transition_frames": row["transition_frames"],
                    "xy_displacement_m": row["xy_displacement_m"],
                    "z_lift_m": row["z_lift_m"],
                    "max_xy_speed_mps": row["max_xy_speed_mps"],
                    "max_abs_vz_mps": row["max_abs_vz_mps"],
                    "has_air": int(has_air),
                    "passes_xy_displacement": int(has_motion),
                }
            )
        kept_transition_id += 1
    return anchor_rows


def _support_transition_proto_rows(transition_rows: list[dict]) -> list[dict]:
    events: list[tuple[int, str, int, dict]] = []
    for row in transition_rows:
        has_air = row["air_start"] != "" and row["first_touch"] != ""
        has_motion = float(row["xy_displacement_m"]) >= 0.05
        if not has_air and not has_motion:
            continue
        part_i = int(row["part_index"])
        events.append((int(row["motion_start"]), "start", part_i, row))
        events.append((int(row["new_support"]), "end", part_i, row))

    priority = {"end": 0, "start": 1}
    events.sort(key=lambda item: (item[0], priority[item[1]], item[2]))

    rows: list[dict] = []
    active: dict[int, dict] = {}
    pending: dict[int, dict] = {}
    segment_start: int | None = None
    proto_id = 0

    def begin_next_segment(frame: int) -> None:
        nonlocal active, pending, segment_start
        if active or not pending:
            return
        active = dict(pending)
        pending = {}
        segment_start = frame

    i = 0
    while i < len(events):
        frame = events[i][0]
        same_frame = []
        while i < len(events) and events[i][0] == frame:
            same_frame.append(events[i])
            i += 1

        begin_next_segment(frame)

        ending = [item for item in same_frame if item[1] == "end" and item[2] in active]
        if ending and segment_start is not None and active:
            active_mask = np.zeros(len(PART_ORDER), dtype=bool)
            for part_i in active:
                active_mask[part_i] = True
            ending_mask = np.zeros(len(PART_ORDER), dtype=bool)
            for _, _, part_i, _ in ending:
                ending_mask[part_i] = True
            rows.append(
                {
                    "proto_id": proto_id,
                    "motion": ending[0][3]["motion"],
                    "start": segment_start,
                    "end": frame,
                    "duration_frames": frame - segment_start,
                    "duration_s": (frame - segment_start) / float(ending[0][3]["fps"] if "fps" in ending[0][3] else 50.0),
                    "active_part": _names_from_mask_local(active_mask),
                    "ending_part": _names_from_mask_local(ending_mask),
                    "num_active": int(active_mask.sum()),
                    "source_transition_ids": "|".join(str(active[p]["transition_id"]) for p in sorted(active)),
                }
            )
            proto_id += 1
            for _, _, part_i, _ in ending:
                active.pop(part_i, None)
            if active:
                segment_start = frame
            else:
                segment_start = None
                begin_next_segment(frame)

        for _, kind, part_i, row in same_frame:
            if kind != "start":
                continue
            if active:
                pending[part_i] = row
            else:
                pending[part_i] = row

        begin_next_segment(frame)

    return rows


def _new_contact_proto_rows(
    motion_name: str,
    fps: float,
    num_frames: int,
    contact: np.ndarray,
    part_positions: dict[str, np.ndarray],
    stable_free_window: int,
    stable_contact_window: int,
    stable_speed_thresh: float,
    cluster_window: int,
    cluster_all_limbs: bool,
) -> list[dict]:
    events: list[tuple[int, int, str, str]] = []
    for part in ("left_foot", "right_foot", "left_hand", "right_hand"):
        part_i = PART_ORDER.index(part)
        part_contact = contact[:, part_i].astype(bool)
        center_vel = np.zeros_like(part_positions[part])
        center_vel[1:] = np.diff(part_positions[part], axis=0) * fps
        center_speed = np.linalg.norm(center_vel, axis=1)
        for frame in range(stable_free_window, num_frames - stable_contact_window + 1):
            if part_contact[frame - 1] or not part_contact[frame]:
                continue
            if part_contact[frame - stable_free_window : frame].any():
                continue
            stable_frame = ""
            for candidate in range(frame, num_frames - stable_contact_window + 1):
                if not part_contact[candidate]:
                    break
                contact_window = part_contact[candidate : candidate + stable_contact_window]
                speed_window = center_speed[candidate : candidate + stable_contact_window]
                if contact_window.all() and np.all(speed_window <= stable_speed_thresh):
                    stable_frame = candidate
                    break
            if stable_frame != "":
                kind = "foot_stable_contact" if "foot" in part else "hand_stable_contact"
                events.append((int(stable_frame), part_i, kind, ""))

    events.sort(key=lambda item: (item[0], item[1], item[2]))
    clustered: list[tuple[int, list[tuple[int, str, str]]]] = []
    for frame, part_i, kind, source_id in events:
        same_cluster_type = cluster_all_limbs or kind == clustered[-1][1][-1][1] if clustered else False
        if clustered and same_cluster_type and frame - clustered[-1][0] <= cluster_window:
            cluster_frame, cluster_events = clustered[-1]
            cluster_events.append((part_i, kind, source_id))
            clustered[-1] = (frame, cluster_events)
        else:
            clustered.append((frame, [(part_i, kind, source_id)]))

    grouped: list[tuple[int, list[tuple[int, str, str]]]] = []
    for frame, frame_events in clustered:
        if grouped and grouped[-1][0] == frame:
            grouped[-1][1].extend(frame_events)
        else:
            grouped.append((frame, frame_events))

    rows: list[dict] = []
    start = 0
    proto_id = 0
    for frame, frame_events in grouped:
        if frame <= start:
            continue
        active_mask = np.zeros(len(PART_ORDER), dtype=bool)
        kinds = []
        source_ids = []
        for part_i, kind, source_id in frame_events:
            active_mask[part_i] = True
            kinds.append(kind)
            if source_id:
                source_ids.append(source_id)
        rows.append(
            {
                "proto_id": proto_id,
                "motion": motion_name,
                "start": start,
                "end": frame,
                "duration_frames": frame - start,
                "duration_s": (frame - start) / fps,
                "active_part": _names_from_mask_local(active_mask),
                "established_part": _names_from_mask_local(active_mask),
                "num_active": int(active_mask.sum()),
                "event_kind": "|".join(sorted(set(kinds))),
                "source_transition_ids": "|".join(source_ids),
            }
        )
        proto_id += 1
        start = frame

    return rows


def _intervals_bool(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.r_[mask.astype(bool), False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _event_role_proto_rows(
    proto_rows: list[dict],
    contact: np.ndarray,
    part_positions: dict[str, np.ndarray],
    fps: float,
    stable_speed_thresh: float,
    stable_window: int,
    free_state_window: int,
    min_active_displacement: float = 0.03,
) -> list[dict]:
    role_masks = _global_role_masks(
        contact,
        part_positions,
        fps,
        stable_speed_thresh,
        stable_window,
        free_state_window,
        min_active_displacement,
    )

    role_rows: list[dict] = []
    for row in proto_rows:
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        acquire_mask = np.zeros(len(PART_ORDER), dtype=bool)
        for name in row["established_part"].split("|"):
            if name:
                acquire_mask[PART_ORDER.index(name)] = True

        release_mask = role_masks["release_event"][start:end].any(axis=0)
        support_mask = role_masks["support"][start:end].any(axis=0)
        active_mask = role_masks["active"][start:end].any(axis=0)
        free_mask = role_masks["free"][start:end].all(axis=0)
        ref_valid_mask = role_masks["ref_valid"][start:end].any(axis=0)

        role = dict(row)
        role["ending_part"] = row["established_part"]
        role["acquire_event"] = _names_from_mask_local(acquire_mask)
        role["release_event"] = _names_from_mask_local(release_mask)
        role["support_state"] = _names_from_mask_local(support_mask)
        role["active_state"] = _names_from_mask_local(active_mask)
        role["free_state"] = _names_from_mask_local(free_mask)
        role["ref_valid_state"] = _names_from_mask_local(ref_valid_mask)
        role_rows.append(role)

    return role_rows


def _event_role_outputs(
    proto_rows: list[dict],
    contact: np.ndarray,
    part_positions: dict[str, np.ndarray],
    fps: float,
    stable_speed_thresh: float,
    stable_window: int,
    free_state_window: int,
    min_active_displacement: float = 0.03,
) -> tuple[list[dict], list[dict]]:
    role_masks = _global_role_masks(
        contact,
        part_positions,
        fps,
        stable_speed_thresh,
        stable_window,
        free_state_window,
        min_active_displacement,
    )
    _assign_active_owner_proto(role_masks, proto_rows, stable_window)
    return (
        _event_role_proto_rows_from_masks(proto_rows, role_masks),
        _event_role_interval_rows_from_masks(proto_rows, role_masks, fps),
    )


def _event_role_proto_rows_from_masks(
    proto_rows: list[dict],
    role_masks: dict[str, np.ndarray],
) -> list[dict]:
    role_rows: list[dict] = []
    for row in proto_rows:
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        acquire_mask = np.zeros(len(PART_ORDER), dtype=bool)
        for name in row["established_part"].split("|"):
            if name:
                acquire_mask[PART_ORDER.index(name)] = True

        owned_active = role_masks["active"][start:end] & (role_masks["active_owner"][start:end] == int(row["proto_id"]))
        support_counts = role_masks["support"][start:end].sum(axis=0)
        active_counts = owned_active.sum(axis=0)
        free_counts = role_masks["free"][start:end].sum(axis=0)

        release_mask = role_masks["release_event"][start:end].any(axis=0)
        support_mask = (support_counts > 0) & (support_counts >= active_counts) & (support_counts >= free_counts)
        active_mask = (active_counts > 0) & (active_counts > support_counts) & (active_counts >= free_counts)
        free_mask = (free_counts > 0) & (free_counts > support_counts) & (free_counts > active_counts)
        if not active_mask.any():
            active_mask = acquire_mask.copy()
            support_mask[active_mask] = False
            free_mask[active_mask] = False
        ref_valid_mask = active_mask

        role = dict(row)
        role["ending_part"] = row["established_part"]
        role["acquire_event"] = _names_from_mask_local(acquire_mask)
        role["release_event"] = _names_from_mask_local(release_mask)
        role["support_state"] = _names_from_mask_local(support_mask)
        role["active_state"] = _names_from_mask_local(active_mask)
        role["free_state"] = _names_from_mask_local(free_mask)
        role["ref_valid_state"] = _names_from_mask_local(ref_valid_mask)
        role_rows.append(role)

    return role_rows


def _event_role_interval_rows_from_masks(
    proto_rows: list[dict],
    role_masks: dict[str, np.ndarray],
    fps: float,
) -> list[dict]:
    rows: list[dict] = []
    for proto in proto_rows:
        proto_id = int(proto["proto_id"])
        start = int(proto["start"])
        end = int(proto["end"])
        if end <= start:
            continue
        interval_start = start
        prev_key = None
        interval_id = 0
        for frame in range(start, end + 1):
            if frame < end:
                owned_active = role_masks["active"][frame] & (role_masks["active_owner"][frame] == proto_id)
                key = (
                    tuple(role_masks["support"][frame]),
                    tuple(owned_active),
                    tuple(role_masks["free"][frame]),
                    tuple(role_masks["ref_valid"][frame] & (role_masks["active_owner"][frame] == proto_id)),
                    tuple(role_masks["acquire_event"][frame]),
                    tuple(role_masks["release_event"][frame]),
                )
            else:
                key = None
            if prev_key is None:
                prev_key = key
                continue
            if key == prev_key:
                continue

            rows.append(
                {
                    "proto_id": proto_id,
                    "interval_id": interval_id,
                    "motion": proto["motion"],
                    "start": interval_start,
                    "end": frame,
                    "duration_frames": frame - interval_start,
                    "duration_s": (frame - interval_start) / fps,
                    "support_state": _names_from_mask_local(role_masks["support"][interval_start]),
                    "active_state": _names_from_mask_local(
                        role_masks["active"][interval_start]
                        & (role_masks["active_owner"][interval_start] == proto_id)
                    ),
                    "free_state": _names_from_mask_local(role_masks["free"][interval_start]),
                    "ref_valid_state": _names_from_mask_local(
                        role_masks["ref_valid"][interval_start]
                        & (role_masks["active_owner"][interval_start] == proto_id)
                    ),
                    "acquire_event": _names_from_mask_local(role_masks["acquire_event"][interval_start]),
                    "release_event": _names_from_mask_local(role_masks["release_event"][interval_start]),
                }
            )
            interval_start = frame
            interval_id += 1
            prev_key = key

    return rows


def _global_role_masks(
    contact: np.ndarray,
    part_positions: dict[str, np.ndarray],
    fps: float,
    stable_speed_thresh: float,
    stable_window: int,
    free_state_window: int,
    min_active_displacement: float,
) -> dict[str, np.ndarray]:
    num_frames = contact.shape[0]
    stable_window = max(1, int(stable_window))
    free_state_window = max(1, int(free_state_window))
    role_part_indices = [PART_ORDER.index(part) for part in ACTIVE_ROLE_PARTS]
    support = np.zeros((num_frames, len(PART_ORDER)), dtype=bool)
    active = np.zeros_like(support)
    free = np.zeros_like(support)
    ref_valid = np.zeros_like(support)
    release_event = np.zeros_like(support)
    acquire_event = np.zeros_like(support)
    active_owner = np.full((num_frames, len(PART_ORDER)), -1, dtype=np.int32)

    for part_i in role_part_indices:
        part = PART_ORDER[part_i]
        c = contact[:, part_i].astype(bool)
        vel = np.zeros_like(part_positions[part])
        vel[1:] = np.diff(part_positions[part], axis=0) * fps
        speed = np.linalg.norm(vel, axis=1)
        stable_contact = c & (speed <= stable_speed_thresh)

        for frame in range(1, num_frames):
            if not c[frame - 1] and c[frame]:
                acquire_event[frame, part_i] = True
            if c[frame - 1] and not c[frame]:
                release_event[frame, part_i] = True

        support[:, part_i] = stable_contact
        active[:, part_i] = c & ~stable_contact

        for frame in np.flatnonzero(acquire_event[:, part_i]):
            active[max(0, frame - stable_window) : frame + 1, part_i] = True
        for frame in np.flatnonzero(release_event[:, part_i]):
            active[frame : min(num_frames, frame + stable_window), part_i] = True

        for frame in range(num_frames):
            history_start = max(0, frame + 1 - free_state_window)
            if not c[history_start : frame + 1].any() and not active[frame, part_i]:
                free[frame, part_i] = True

        support[active[:, part_i], part_i] = False
        free[active[:, part_i] | support[:, part_i], part_i] = False
        ref_valid[:, part_i] = active[:, part_i]

    return {
        "support": support,
        "active": active,
        "free": free,
        "ref_valid": ref_valid,
        "acquire_event": acquire_event,
        "release_event": release_event,
        "active_owner": active_owner,
    }


def _assign_active_owner_proto(
    role_masks: dict[str, np.ndarray],
    proto_rows: list[dict],
    stable_window: int,
) -> None:
    active_owner = role_masks["active_owner"]
    active = role_masks["active"]
    acquire_event = role_masks["acquire_event"]
    release_event = role_masks["release_event"]
    num_frames = active.shape[0]
    stable_window = max(1, int(stable_window))
    for row in proto_rows:
        proto_id = int(row["proto_id"])
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        proto_part_mask = np.zeros(len(PART_ORDER), dtype=bool)
        for name in row["established_part"].split("|"):
            if name:
                proto_part_mask[PART_ORDER.index(name)] = True

        for part_i in range(len(PART_ORDER)):
            if proto_part_mask[part_i]:
                event_end = min(num_frames, end + 1)
                event_frames = np.flatnonzero(acquire_event[start:event_end, part_i]) + start
                for frame in event_frames:
                    cursor = frame
                    while cursor >= 0 and active[cursor, part_i] and active_owner[cursor, part_i] < 0:
                        active_owner[cursor, part_i] = proto_id
                        cursor -= 1

            release_frames = np.flatnonzero(release_event[start:end, part_i]) + start
            for frame in release_frames:
                cursor = frame
                while cursor < min(num_frames, end) and active[cursor, part_i] and active_owner[cursor, part_i] < 0:
                    active_owner[cursor, part_i] = proto_id
                    cursor += 1

    for row in proto_rows:
        proto_id = int(row["proto_id"])
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        owned_in_proto = active[start:end] & (active_owner[start:end] < 0)
        active_owner[start:end][owned_in_proto] = proto_id


def _names_from_mask_local(mask: np.ndarray) -> str:
    names = [part for part, value in zip(PART_ORDER, mask) if value]
    return "|".join(names)


def _diagnose_motion(
    ctx,
    enabled_parts: set[str],
    foot_gap: int,
    hand_gap: int,
    body_gap: int,
    support_contact_count: int,
    support_window: int,
    support_vz_thresh: float,
    support_xy_speed_thresh: float,
    motion_start_vz_thresh: float,
    motion_start_lookback: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    clean_contact = _contact_masks(ctx, enabled_parts)
    debounced = _debounced_contact(clean_contact, foot_gap, hand_gap, body_gap)
    part_rows: list[dict] = []
    probe_rows: list[dict] = []
    foot_support_rows: list[dict] = []
    foot_transition_rows: list[dict] = []

    for part_i, part in enumerate(PART_ORDER):
        if part not in enabled_parts:
            continue
        pos = ctx.part_pos[part]

        if "foot" in part:
            probes = ctx.part_probe_pos[part]
            gap = np.asarray(
                [[_top_gap(point, ctx.boxes) for point in frame_points] for frame_points in probes],
                dtype=np.float64,
            )
            gap_vel = np.zeros_like(gap)
            gap_vel[1:] = np.diff(gap, axis=0) * ctx.fps
            probe_rule = (np.abs(gap) < 0.035) & (np.abs(gap_vel) < 0.35)
            raw_rule_any = probe_rule.any(axis=1)
            support_rows, transition_rows = _foot_support_diagnostics(
                ctx.path.name,
                part,
                part_i,
                ctx.fps,
                probes,
                probe_rule,
                support_contact_count,
                support_window,
                support_vz_thresh,
                support_xy_speed_thresh,
                motion_start_vz_thresh,
                motion_start_lookback,
            )
            foot_support_rows.extend(support_rows)
            foot_transition_rows.extend(transition_rows)

            for frame in range(len(ctx.qpos)):
                best_probe = int(np.argmin(np.abs(gap[frame])))
                best_gap, best_box, best_surface_z = _top_gap_with_box(probes[frame, best_probe], ctx.boxes)
                part_rows.append(
                    {
                        "motion": ctx.path.name,
                        "frame": frame,
                        "time_s": frame / ctx.fps,
                        "fps": ctx.fps,
                        "part": part,
                        "part_index": part_i,
                        "rule_kind": "foot_top_gap_velocity",
                        "raw_rule_contact": int(raw_rule_any[frame]),
                        "clean_contact": int(clean_contact[frame, part_i]),
                        "debounced_contact": int(debounced[frame, part_i]),
                        "best_probe": _probe_name(part, best_probe),
                        "best_gap_m": best_gap,
                        "best_gap_vel_mps": float(gap_vel[frame, best_probe]),
                        "nearest_dist_m": "",
                        "matched_box_index": best_box,
                        "matched_surface_z": best_surface_z,
                    }
                )
                for probe_i, point in enumerate(probes[frame]):
                    probe_gap, box_i, surface_z = _top_gap_with_box(point, ctx.boxes)
                    probe_rows.append(
                        {
                            "motion": ctx.path.name,
                            "frame": frame,
                            "time_s": frame / ctx.fps,
                            "fps": ctx.fps,
                            "part": part,
                            "part_index": part_i,
                            "probe": _probe_name(part, probe_i),
                            "probe_index": probe_i,
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "z": float(point[2]),
                            "top_gap_m": probe_gap,
                            "gap_vel_mps": float(gap_vel[frame, probe_i]),
                            "nearest_dist_m": "",
                            "rule_contact": int(probe_rule[frame, probe_i]),
                            "matched_box_index": box_i,
                            "matched_surface_z": surface_z,
                        }
                    )
            continue

        if "hand" in part:
            probe_center = ctx.part_probe_pos[part].mean(axis=1)
            dist = np.asarray([_nearest_surface_distance(p, ctx.boxes) for p in probe_center], dtype=np.float64)
            strict = np.abs(dist) < 0.06
            keep = np.abs(dist) < 0.065
            raw_rule = np.zeros(len(pos), dtype=bool)
            for frame in range(len(pos)):
                raw_rule[frame] = bool(strict[frame] or (frame > 0 and raw_rule[frame - 1] and keep[frame]))
            points = probe_center[:, None, :]
            rule_kind = "hand_nearest_surface_hysteresis"
        else:
            dist = np.asarray([_nearest_surface_distance(p, ctx.boxes) for p in pos], dtype=np.float64)
            raw_rule = np.abs(dist) < 0.035
            points = pos[:, None, :]
            rule_kind = "body_nearest_surface_distance"

        for frame in range(len(ctx.qpos)):
            point = points[frame, 0]
            nearest_dist, box_i = _nearest_surface_with_box(point, ctx.boxes)
            part_rows.append(
                {
                    "motion": ctx.path.name,
                    "frame": frame,
                    "time_s": frame / ctx.fps,
                    "fps": ctx.fps,
                    "part": part,
                    "part_index": part_i,
                    "rule_kind": rule_kind,
                    "raw_rule_contact": int(raw_rule[frame]),
                    "clean_contact": int(clean_contact[frame, part_i]),
                    "debounced_contact": int(debounced[frame, part_i]),
                    "best_probe": "probe_center" if "hand" in part else "part_center",
                    "best_gap_m": "",
                    "best_gap_vel_mps": "",
                    "nearest_dist_m": nearest_dist,
                    "matched_box_index": box_i,
                    "matched_surface_z": "",
                }
            )
            probe_rows.append(
                {
                    "motion": ctx.path.name,
                    "frame": frame,
                    "time_s": frame / ctx.fps,
                    "fps": ctx.fps,
                    "part": part,
                    "part_index": part_i,
                    "probe": "probe_center" if "hand" in part else "part_center",
                    "probe_index": 0,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "top_gap_m": "",
                    "gap_vel_mps": "",
                    "nearest_dist_m": nearest_dist,
                    "rule_contact": int(raw_rule[frame]),
                    "matched_box_index": box_i,
                    "matched_surface_z": "",
                }
            )

    return part_rows, probe_rows, foot_support_rows, foot_transition_rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    data_dir = _resolve(dataset_root, args.data_dir)
    holosoma_motion_dir = _resolve(dataset_root, args.holosoma_motion_dir)
    output_dir = _resolve(dataset_root, args.output_dir)
    robot_urdf = (
        args.robot_urdf.expanduser().resolve()
        if args.robot_urdf
        else canonical_g1_urdf_path()
    )
    validate_canonical_g1_urdf(robot_urdf)
    enabled_parts = set(args.contact_parts)

    part_fieldnames = [
        "motion",
        "frame",
        "time_s",
        "fps",
        "part",
        "part_index",
        "rule_kind",
        "raw_rule_contact",
        "clean_contact",
        "debounced_contact",
        "best_probe",
        "best_gap_m",
        "best_gap_vel_mps",
        "nearest_dist_m",
        "matched_box_index",
        "matched_surface_z",
    ]
    probe_fieldnames = [
        "motion",
        "frame",
        "time_s",
        "fps",
        "part",
        "part_index",
        "probe",
        "probe_index",
        "x",
        "y",
        "z",
        "top_gap_m",
        "gap_vel_mps",
        "nearest_dist_m",
        "rule_contact",
        "matched_box_index",
        "matched_surface_z",
    ]
    foot_support_fieldnames = [
        "motion",
        "frame",
        "time_s",
        "fps",
        "part",
        "part_index",
        "contact_count",
        "geometric_contact",
        "stable_support_instant",
        "stable_support",
        "center_x",
        "center_y",
        "center_z",
        "center_xy_speed",
        "center_vz",
    ]
    foot_transition_fieldnames = [
        "motion",
        "part",
        "part_index",
        "motion_start",
        "support_break",
        "air_start",
        "first_touch",
        "new_support",
        "motion_start_time_s",
        "support_break_time_s",
        "air_start_time_s",
        "first_touch_time_s",
        "new_support_time_s",
        "transition_frames",
        "air_frames",
        "xy_displacement_m",
        "z_lift_m",
        "start_contact_count",
        "break_contact_count",
        "new_support_contact_count",
        "max_xy_speed_mps",
        "max_abs_vz_mps",
    ]
    anchor_fieldnames = [
        "motion",
        "part",
        "part_index",
        "transition_id",
        "anchor_type",
        "anchor_frame",
        "anchor_time_s",
        "segment_start",
        "segment_end",
        "motion_start",
        "support_break",
        "air_start",
        "first_touch",
        "new_support",
        "air_frames",
        "transition_frames",
        "xy_displacement_m",
        "z_lift_m",
        "max_xy_speed_mps",
        "max_abs_vz_mps",
        "has_air",
        "passes_xy_displacement",
    ]
    proto_segment_fieldnames = [
        "proto_id",
        "motion",
        "start",
        "end",
        "duration_frames",
        "duration_s",
        "active_part",
        "ending_part",
        "num_active",
        "source_transition_ids",
    ]
    new_contact_proto_fieldnames = [
        "proto_id",
        "motion",
        "start",
        "end",
        "duration_frames",
        "duration_s",
        "active_part",
        "established_part",
        "num_active",
        "event_kind",
        "source_transition_ids",
    ]
    event_role_proto_fieldnames = new_contact_proto_fieldnames + [
        "ending_part",
        "acquire_event",
        "release_event",
        "support_state",
        "active_state",
        "free_state",
        "ref_valid_state",
    ]
    event_role_interval_fieldnames = [
        "proto_id",
        "interval_id",
        "motion",
        "start",
        "end",
        "duration_frames",
        "duration_s",
        "support_state",
        "active_state",
        "free_state",
        "ref_valid_state",
        "acquire_event",
        "release_event",
    ]

    for motion in _motion_paths(data_dir, args.motions):
        ctx = _load_context(motion, dataset_root, robot_urdf)
        _, target_frames, target_fps = _holosoma_motion_info(
            motion, holosoma_motion_dir, args.holosoma_motion_suffix
        )
        ctx_50hz = _resample_context(ctx, target_frames, target_fps)
        part_rows, probe_rows, foot_support_rows, foot_transition_rows = _diagnose_motion(
            ctx_50hz,
            enabled_parts,
            args.merge_transition_window,
            args.hand_merge_transition_window,
            args.body_merge_transition_window,
            args.support_contact_count,
            args.support_window,
            args.support_vz_thresh,
            args.support_xy_speed_thresh,
            args.motion_start_vz_thresh,
            args.motion_start_lookback,
        )
        anchor_rows = _support_transition_anchor_rows(
            foot_transition_rows,
            args.anchor_min_xy_displacement,
        )
        proto_rows = _support_transition_proto_rows(anchor_rows[::2])
        clean_contact = _contact_masks(ctx_50hz, enabled_parts)
        new_contact_proto_rows = _new_contact_proto_rows(
            motion.name,
            ctx_50hz.fps,
            len(ctx_50hz.qpos),
            clean_contact,
            ctx_50hz.part_pos,
            args.event_free_window,
            args.event_contact_window,
            args.event_stable_speed_thresh,
            args.event_cluster_window,
            args.event_cluster_all_limbs,
        )
        role_proto_rows, role_interval_rows = _event_role_outputs(
            new_contact_proto_rows,
            clean_contact,
            ctx_50hz.part_pos,
            ctx_50hz.fps,
            args.event_stable_speed_thresh,
            args.event_contact_window,
            args.event_free_state_window,
        )
        part_path = output_dir / f"{motion.stem}_part_contact_diagnostics.csv"
        probe_path = output_dir / f"{motion.stem}_probe_contact_diagnostics.csv"
        support_path = output_dir / f"{motion.stem}_foot_support_diagnostics.csv"
        transition_path = output_dir / f"{motion.stem}_foot_support_transitions.csv"
        anchor_path = output_dir / f"{motion.stem}_support_transition_anchors.csv"
        proto_segment_path = output_dir / f"{motion.stem}_support_transition_proto_segments.csv"
        new_contact_proto_path = output_dir / f"{motion.stem}_new_contact_proto_segments.csv"
        event_role_proto_path = output_dir / f"{motion.stem}_event_role_proto_segments.csv"
        event_role_interval_path = output_dir / f"{motion.stem}_event_role_intervals.csv"
        _write_csv(part_path, part_rows, part_fieldnames, args.overwrite)
        _write_csv(probe_path, probe_rows, probe_fieldnames, args.overwrite)
        _write_csv(support_path, foot_support_rows, foot_support_fieldnames, args.overwrite)
        _write_csv(transition_path, foot_transition_rows, foot_transition_fieldnames, args.overwrite)
        _write_csv(anchor_path, anchor_rows, anchor_fieldnames, args.overwrite)
        _write_csv(proto_segment_path, proto_rows, proto_segment_fieldnames, args.overwrite)
        _write_csv(new_contact_proto_path, new_contact_proto_rows, new_contact_proto_fieldnames, args.overwrite)
        _write_csv(event_role_proto_path, role_proto_rows, event_role_proto_fieldnames, args.overwrite)
        _write_csv(event_role_interval_path, role_interval_rows, event_role_interval_fieldnames, args.overwrite)
        print(
            f"{motion.name}: raw {len(ctx.qpos)} @ {ctx.fps:g} Hz -> "
            f"{len(ctx_50hz.qpos)} @ {ctx_50hz.fps:g} Hz"
        )
        print(f"  part diagnostics: {part_path}")
        print(f"  probe diagnostics: {probe_path}")
        print(f"  foot support diagnostics: {support_path}")
        print(f"  foot support transitions: {transition_path}")
        print(f"  support transition anchors: {anchor_path}")
        print(f"  support transition proto segments: {proto_segment_path}")
        print(f"  new contact proto segments: {new_contact_proto_path}")
        print(f"  event role proto segments: {event_role_proto_path}")
        print(f"  event role intervals: {event_role_interval_path}")


if __name__ == "__main__":
    main()
