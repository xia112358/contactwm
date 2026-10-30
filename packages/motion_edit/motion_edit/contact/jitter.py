from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from motion_edit.contact.edits import move_contact_anchor_on_surface
from motion_edit.contact.layers import read_contact_graph
from motion_edit.contact.plans import ContactEditPlan, validate_contact_edit_plan, write_contact_edit_plan
from motion_edit.contact.schema import ContactAnchorRecord
from motion_edit.contact.surface_geometry import point_in_polygon_uv
from motion_edit.paths import LAYERS_ROOT
from somaforge_core.contact_schema import CONTACT_FORCE_PART_NAMES


DEFAULT_JITTER_BODIES = CONTACT_FORCE_PART_NAMES
FOOT_JITTER_BODIES = {"left_heel", "left_toe", "right_heel", "right_toe"}
FOOT_GROUP_MAX_GAP_FRAMES = 3
JitterSampler = str


@dataclass(frozen=True)
class ContactJitterPlanResult:
    plan_path: Path
    plan_id: str
    motion_id: str
    source_motion_path: str
    source_contact_layer: str
    edit_count: int
    attempted_anchor_count: int
    seed: int


def _load_cut_summary(path: str | Path) -> list[dict]:
    summary_path = Path(path).expanduser()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"cut summary must be a list: {summary_path}")
    return [item for item in payload if isinstance(item, dict) and item.get("status", "ok") == "ok"]


def _anchor_can_jitter(anchor: ContactAnchorRecord, bodies: set[str]) -> bool:
    return (
        anchor.editable
        and anchor.body in bodies
        and anchor.world_position is not None
        and anchor.surface_id is not None
        and anchor.surface_coordinates is not None
        and anchor.surface_normal is not None
        and anchor.surface_tangent_u is not None
        and anchor.surface_tangent_v is not None
    )


def _sample_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    # Rejection sampling keeps the u/v metric isotropic because the stored tangent basis is orthonormal.
    while True:
        du = rng.uniform(-radius, radius)
        dv = rng.uniform(-radius, radius)
        if du * du + dv * dv <= radius * radius:
            return du, dv


def _sample_annulus(rng: random.Random, radius: float, min_radius_fraction: float) -> tuple[float, float]:
    lo = max(0.0, min(1.0, float(min_radius_fraction)))
    while True:
        du, dv = _sample_disk(rng, radius)
        distance = (du * du + dv * dv) ** 0.5
        if distance >= radius * lo:
            return du, dv


def _latest_surface_polygon(anchor: ContactAnchorRecord) -> list[tuple[float, float]]:
    bindings = anchor.metadata.get("surface_bindings")
    if not isinstance(bindings, list) or not bindings:
        return []
    latest = bindings[-1]
    raw = latest.get("polygon_surface_coordinates") if isinstance(latest, dict) else None
    if not isinstance(raw, list) or len(raw) < 3:
        return []
    polygon: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "u" not in item or "v" not in item:
            return []
        polygon.append((float(item["u"]), float(item["v"])))
    return polygon


def _surface_bounds(anchor: ContactAnchorRecord) -> tuple[float, float, float, float] | None:
    polygon = _latest_surface_polygon(anchor)
    if polygon:
        us = [point[0] for point in polygon]
        vs = [point[1] for point in polygon]
        return min(us), max(us), min(vs), max(vs)
    bounds = anchor.surface_bounds
    if not isinstance(bounds, dict):
        return None
    raw_u = bounds.get("u") or bounds.get("u_bounds")
    raw_v = bounds.get("v") or bounds.get("v_bounds")
    if raw_u is None or raw_v is None:
        return None
    u0, u1 = sorted(float(item) for item in raw_u)
    v0, v1 = sorted(float(item) for item in raw_v)
    return u0, u1, v0, v1


def _sample_surface_target(anchor: ContactAnchorRecord, rng: random.Random, radius: float) -> dict[str, float]:
    bounds = _surface_bounds(anchor)
    if bounds is None:
        raise ValueError("anchor has no surface polygon or bounds for uniform surface sampling")
    u0, u1, v0, v1 = bounds
    polygon = _latest_surface_polygon(anchor)
    start = anchor.surface_coordinates or {}
    start_u = float(start.get("u", 0.0))
    start_v = float(start.get("v", 0.0))
    u0 = max(u0, start_u - radius)
    u1 = min(u1, start_u + radius)
    v0 = max(v0, start_v - radius)
    v1 = min(v1, start_v + radius)
    if u1 < u0 or v1 < v0:
        raise ValueError("surface target radius does not intersect surface bounds")
    for _ in range(128):
        u = rng.uniform(u0, u1)
        v = rng.uniform(v0, v1)
        if (u - start_u) * (u - start_u) + (v - start_v) * (v - start_v) > radius * radius:
            continue
        if not polygon or point_in_polygon_uv((u, v), polygon):
            return {"u": u, "v": v}
    raise ValueError("could not sample a point inside the surface polygon")


def _radius_for_anchor(anchor: ContactAnchorRecord, radius: float, knee_radius_scale: float) -> float:
    if anchor.body in {"left_knee", "right_knee"}:
        return radius * float(knee_radius_scale)
    return radius


def _sample_move(
    anchor: ContactAnchorRecord,
    *,
    rng: random.Random,
    radius: float,
    sampler: JitterSampler,
    min_radius_fraction: float,
) -> tuple[tuple[float, float] | None, dict[str, float] | None, str]:
    chosen = sampler
    if sampler == "mixed":
        chosen = rng.choice(("local_disk", "local_annulus", "surface_uniform"))
    if chosen == "local_disk":
        return _sample_disk(rng, radius), None, chosen
    if chosen == "local_annulus":
        return _sample_annulus(rng, radius, min_radius_fraction), None, chosen
    if chosen == "surface_uniform":
        return None, _sample_surface_target(anchor, rng, radius), chosen
    raise ValueError(f"unsupported contact jitter sampler: {sampler}")


def _iter_selected_anchors(
    anchors: Iterable[ContactAnchorRecord],
    *,
    rng: random.Random,
    bodies: set[str],
    edit_probability: float,
    max_edits: int,
) -> list[ContactAnchorRecord]:
    candidates = [anchor for anchor in anchors if _anchor_can_jitter(anchor, bodies)]
    rng.shuffle(candidates)
    selected = [anchor for anchor in candidates if rng.random() < edit_probability]
    if not selected and candidates:
        selected = [candidates[0]]
    if max_edits > 0:
        selected = selected[:max_edits]
    return selected


def _group_selected_anchors(
    selected: list[ContactAnchorRecord],
    *,
    max_gap_frames: int = FOOT_GROUP_MAX_GAP_FRAMES,
) -> list[list[ContactAnchorRecord]]:
    selected_order = {anchor.anchor_id: index for index, anchor in enumerate(selected)}
    groups: list[list[ContactAnchorRecord]] = [[anchor] for anchor in selected if anchor.body not in FOOT_JITTER_BODIES]

    for body in sorted(FOOT_JITTER_BODIES):
        body_anchors = [anchor for anchor in selected if anchor.body == body]
        body_anchors.sort(key=lambda anchor: (anchor.start_frame, anchor.end_frame, anchor.anchor_id))
        current: list[ContactAnchorRecord] = []
        current_end = -1
        for anchor in body_anchors:
            if not current or anchor.start_frame > current_end + max_gap_frames:
                if current:
                    groups.append(current)
                current = [anchor]
                current_end = anchor.end_frame
                continue
            current.append(anchor)
            current_end = max(current_end, anchor.end_frame)
        if current:
            groups.append(current)

    groups.sort(key=lambda group: min(selected_order[anchor.anchor_id] for anchor in group))
    return groups


def _add_jitter_metadata(
    edit: dict,
    *,
    anchor: ContactAnchorRecord,
    group: list[ContactAnchorRecord],
    augmentation_index: int,
    seed: int,
    offset_radius: float,
    effective_radius: float,
    sampler: JitterSampler,
    sampled_mode: str,
    min_radius_fraction: float,
    knee_radius_scale: float,
    shared_world_delta: Iterable[float] | None,
) -> dict:
    edit.setdefault("metadata", {})
    metadata = edit["metadata"]
    metadata.update(
        {
            "augmentation_index": augmentation_index,
            "augmentation_seed": seed,
            "offset_radius": offset_radius,
            "effective_offset_radius": effective_radius,
            "jitter_sampler": sampler,
            "sampled_jitter_mode": sampled_mode,
            "min_radius_fraction": min_radius_fraction,
            "knee_radius_scale": knee_radius_scale,
        }
    )
    if anchor.body in FOOT_JITTER_BODIES and len(group) > 1:
        metadata.update(
            {
                "foot_group_jitter": True,
                "foot_group_body": anchor.body,
                "foot_group_anchor_ids": [item.anchor_id for item in group],
                "foot_group_start_frame": min(item.start_frame for item in group),
                "foot_group_end_frame": max(item.end_frame for item in group),
                "foot_group_shared_delta_world": [float(item) for item in (shared_world_delta or [])],
            }
        )
    return edit


def _sample_group_edits(
    group: list[ContactAnchorRecord],
    *,
    augmentation_index: int,
    rng: random.Random,
    seed: int,
    offset_radius: float,
    mode: str,
    sampler: JitterSampler,
    min_radius_fraction: float,
    knee_radius_scale: float,
) -> list[dict]:
    representative = group[0]
    effective_radius = _radius_for_anchor(representative, offset_radius, knee_radius_scale)
    tangent_delta, surface_target, sampled_mode = _sample_move(
        representative,
        rng=rng,
        radius=effective_radius,
        sampler=sampler,
        min_radius_fraction=min_radius_fraction,
    )
    _moved, edit_record = move_contact_anchor_on_surface(
        representative,
        tangent_delta=tangent_delta,
        new_surface_coordinates=surface_target,
        mode=mode,
        source="random_surface_jitter",
        edit_id=f"{representative.anchor_id}_jitter_{augmentation_index:04d}",
    )
    shared_world_delta = edit_record.delta_world

    edits: list[dict] = []
    for anchor in group:
        _moved, anchor_edit_record = move_contact_anchor_on_surface(
            anchor,
            requested_world_delta=shared_world_delta,
            mode=mode,
            source="random_surface_jitter",
            edit_id=f"{anchor.anchor_id}_jitter_{augmentation_index:04d}",
        )
        edit = anchor_edit_record.to_dict()
        edits.append(
            _add_jitter_metadata(
                edit,
                anchor=anchor,
                group=group,
                augmentation_index=augmentation_index,
                seed=seed,
                offset_radius=offset_radius,
                effective_radius=effective_radius,
                sampler=sampler,
                sampled_mode=sampled_mode,
                min_radius_fraction=min_radius_fraction,
                knee_radius_scale=knee_radius_scale,
                shared_world_delta=shared_world_delta,
            )
        )
    return edits


def _make_jitter_plan_for_motion(
    item: dict,
    *,
    augmentation_index: int,
    rng: random.Random,
    seed: int,
    output_dir: Path,
    offset_radius: float,
    edit_probability: float,
    max_edits: int,
    max_attempts: int,
    bodies: set[str],
    mode: str,
    sampler: JitterSampler,
    min_radius_fraction: float,
    knee_radius_scale: float,
) -> ContactJitterPlanResult | None:
    motion_id = str(item.get("motion_id") or item.get("motion_asset_id") or "")
    motion_path = str(item.get("motion_path") or "")
    contact_layer = str(item.get("ready_layer") or "")
    if not motion_id or not motion_path or not contact_layer:
        return None

    graph = read_contact_graph(LAYERS_ROOT / contact_layer, motion_id)
    selected = _iter_selected_anchors(
        graph.anchors,
        rng=rng,
        bodies=bodies,
        edit_probability=edit_probability,
        max_edits=max_edits,
    )

    edits: list[dict] = []
    attempted = 0
    for group in _group_selected_anchors(selected):
        attempted += len(group)
        group_edits = None
        for _ in range(max_attempts):
            try:
                group_edits = _sample_group_edits(
                    group,
                    augmentation_index=augmentation_index,
                    rng=rng,
                    seed=seed,
                    offset_radius=offset_radius,
                    mode=mode,
                    sampler=sampler,
                    min_radius_fraction=min_radius_fraction,
                    knee_radius_scale=knee_radius_scale,
                )
            except ValueError:
                continue
            break
        if group_edits is not None:
            edits.extend(group_edits)

    if not edits:
        return None

    stem = motion_id.replace("/", "_")
    plan_id = f"{stem}_surface_jitter_{augmentation_index:04d}"
    plan = ContactEditPlan(
        plan_id=plan_id,
        source_motion_path=motion_path,
        source_motion_id=motion_id,
        source_contact_layer=contact_layer,
        source_segment_layer=None,
        edits=edits,
        status="validated",
        output_motion_path=f"data/motions/generated/{plan_id}.npz",
        output_contact_layer=f"contact/{plan_id}",
        output_segment_layer=f"candidates/{plan_id}",
        metadata={
            "augmentation": "random_surface_jitter",
            "augmentation_index": augmentation_index,
            "augmentation_seed": seed,
            "offset_radius": offset_radius,
            "edit_probability": edit_probability,
            "max_edits": max_edits,
            "constraint_mode": mode,
            "jitter_sampler": sampler,
            "min_radius_fraction": min_radius_fraction,
            "knee_radius_scale": knee_radius_scale,
            "source_cut_summary_motion_asset_id": item.get("motion_asset_id"),
        },
    )
    validate_contact_edit_plan(plan)
    plan_path = output_dir / f"{plan_id}.json"
    write_contact_edit_plan(plan_path, plan)
    return ContactJitterPlanResult(
        plan_path=plan_path,
        plan_id=plan_id,
        motion_id=motion_id,
        source_motion_path=motion_path,
        source_contact_layer=contact_layer,
        edit_count=len(edits),
        attempted_anchor_count=attempted,
        seed=seed,
    )


def generate_contact_jitter_plans(
    cut_summary: str | Path,
    output_dir: str | Path,
    *,
    augmentations_per_motion: int = 8,
    offset_radius: float = 0.05,
    edit_probability: float = 0.35,
    max_edits: int = 12,
    max_attempts: int = 32,
    seed: int = 0,
    bodies: Iterable[str] = DEFAULT_JITTER_BODIES,
    mode: str = "reject",
    sampler: JitterSampler = "local_disk",
    min_radius_fraction: float = 0.5,
    knee_radius_scale: float = 0.5,
    limit_motions: int | None = None,
) -> tuple[list[ContactJitterPlanResult], dict[str, int | float | str]]:
    if augmentations_per_motion <= 0:
        raise ValueError("augmentations_per_motion must be positive")
    if offset_radius <= 0.0:
        raise ValueError("offset_radius must be positive")
    if not 0.0 < edit_probability <= 1.0:
        raise ValueError("edit_probability must be in (0, 1]")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if mode not in {"reject", "clamp"}:
        raise ValueError("mode must be reject or clamp")
    if sampler not in {"local_disk", "local_annulus", "surface_uniform", "mixed"}:
        raise ValueError("sampler must be local_disk, local_annulus, surface_uniform, or mixed")
    if not 0.0 <= min_radius_fraction <= 1.0:
        raise ValueError("min_radius_fraction must be in [0, 1]")
    if knee_radius_scale <= 0.0:
        raise ValueError("knee_radius_scale must be positive")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    body_set = {str(body) for body in bodies}
    items = _load_cut_summary(cut_summary)
    if limit_motions is not None:
        items = items[: max(0, int(limit_motions))]

    results: list[ContactJitterPlanResult] = []
    skipped_no_edits = 0
    for item in items:
        for aug_index in range(augmentations_per_motion):
            aug_seed = seed + len(results) + skipped_no_edits
            rng = random.Random(aug_seed)
            result = _make_jitter_plan_for_motion(
                item,
                augmentation_index=aug_index,
                rng=rng,
                seed=aug_seed,
                output_dir=out_dir,
                offset_radius=offset_radius,
                edit_probability=edit_probability,
                max_edits=max_edits,
                max_attempts=max_attempts,
                bodies=body_set,
                mode=mode,
                sampler=sampler,
                min_radius_fraction=min_radius_fraction,
                knee_radius_scale=knee_radius_scale,
            )
            if result is None:
                skipped_no_edits += 1
                continue
            results.append(result)

    manifest = {
        "kind": "motion_edit_contact_jitter_plan_manifest",
        "cut_summary": str(Path(cut_summary).expanduser()),
        "output_dir": str(out_dir),
        "augmentations_per_motion": augmentations_per_motion,
        "offset_radius": offset_radius,
        "edit_probability": edit_probability,
        "max_edits": max_edits,
        "max_attempts": max_attempts,
        "seed": seed,
        "bodies": sorted(body_set),
        "mode": mode,
        "sampler": sampler,
        "min_radius_fraction": min_radius_fraction,
        "knee_radius_scale": knee_radius_scale,
        "source_motion_count": len(items),
        "plan_count": len(results),
        "skipped_no_edits": skipped_no_edits,
        "plans": [
            {
                "plan_path": str(result.plan_path),
                "plan_id": result.plan_id,
                "motion_id": result.motion_id,
                "source_motion_path": result.source_motion_path,
                "source_contact_layer": result.source_contact_layer,
                "edit_count": result.edit_count,
                "attempted_anchor_count": result.attempted_anchor_count,
                "seed": result.seed,
            }
            for result in results
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    stats: dict[str, int | float | str] = {
        "source_motion_count": len(items),
        "plan_count": len(results),
        "skipped_no_edits": skipped_no_edits,
        "offset_radius": offset_radius,
        "edit_probability": edit_probability,
        "mode": mode,
        "sampler": sampler,
        "min_radius_fraction": min_radius_fraction,
        "knee_radius_scale": knee_radius_scale,
    }
    return results, stats
