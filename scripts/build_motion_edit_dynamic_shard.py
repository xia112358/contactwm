#!/usr/bin/env python3
"""Build the next motion-edit finetune shard from unified eval failures."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runtime/current/manifests/motion_edit_ref_v1.json"
STANDARD_ID = "motion_edit_full_start0_v1"


@dataclass
class MotionScore:
    motion_id: int
    terrain_id: int
    motion_file: str
    reps: int
    pass_count: int
    fail_count: int
    fail_rate: float
    mean_progress: float
    mean_fail_progress: float
    median_fail_done_step: float
    early_fail_bonus: float
    unstable_bonus: float
    score: float


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _normalize_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    path = path.expanduser().resolve()
    manifest = _load_json(path)
    base_dir = path.parent
    entries = manifest.get("motion_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no motion_files: {path}")

    normalized: list[dict[str, Any]] = []
    path_to_id: dict[str, int] = {}
    for motion_id, entry in enumerate(entries):
        if not isinstance(entry, dict) or "motion_file" not in entry:
            raise ValueError(f"Bad motion entry at index {motion_id}: {entry!r}")
        out = dict(entry)
        out["motion_file"] = _normalize_path(str(out["motion_file"]), base_dir)
        out["terrain_id"] = int(out["terrain_id"])
        out["weight"] = float(out.get("weight", 1.0))
        normalized.append(out)
        path_to_id[out["motion_file"]] = motion_id

    terrain_entries = manifest.get("terrains")
    if not isinstance(terrain_entries, list) or not terrain_entries:
        raise ValueError(f"Manifest has no terrains: {path}")
    terrains: list[dict[str, Any]] = []
    for entry in terrain_entries:
        if not isinstance(entry, dict) or "terrain_file" not in entry:
            raise ValueError(f"Bad terrain entry: {entry!r}")
        out = dict(entry)
        out["terrain_file"] = _normalize_path(str(out["terrain_file"]), base_dir)
        out["terrain_id"] = int(out["terrain_id"])
        terrains.append(out)

    manifest = dict(manifest)
    manifest["terrains"] = terrains
    manifest["motion_files"] = normalized
    return manifest, normalized, path_to_id


def _summary_path_for_acceptance_csv(path: Path) -> Path:
    return path.with_name(f"{path.stem}_summary.json")


def _eval_standard_for_csv(path: Path, *, allow_nonstandard: bool) -> dict[str, Any] | None:
    path = path.expanduser().resolve()
    if "legacy_unstandardized" in path.parts and not allow_nonstandard:
        raise ValueError(
            f"Refusing legacy unstandardized eval CSV: {path}. "
            "Use --allow-nonstandard-eval-csv only for diagnostics."
        )
    summary_path = _summary_path_for_acceptance_csv(path)
    if not summary_path.exists():
        if allow_nonstandard:
            return None
        raise ValueError(
            f"Refusing eval CSV without formal summary metadata: {path}. "
            f"Expected {summary_path} with eval_standard.formal=true."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unreadable eval summary metadata: {summary_path}: {exc}") from exc
    standard = summary.get("eval_standard")
    if not isinstance(standard, dict):
        if allow_nonstandard:
            return None
        raise ValueError(f"Refusing eval CSV without eval_standard metadata: {path}")
    if not allow_nonstandard and (standard.get("standard_id") != STANDARD_ID or standard.get("formal") is not True):
        raise ValueError(
            f"Refusing non-standard eval CSV: {path}. "
            f"Expected eval_standard.standard_id={STANDARD_ID!r} and formal=true in {summary_path}."
        )
    return standard


def _expand_eval_csvs(paths: list[Path], patterns: list[str], *, allow_nonstandard: bool) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        out.append(path.expanduser().resolve())
    for pattern in patterns:
        for value in glob.glob(str(Path(pattern).expanduser())):
            out.append(Path(value).resolve())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("No eval CSVs were provided.")
    for path in unique:
        _eval_standard_for_csv(path, allow_nonstandard=allow_nonstandard)
    return unique


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _row_motion_id(
    row: dict[str, str],
    *,
    manifest_entries: list[dict[str, Any]],
    path_to_id: dict[str, int],
    manifest_dir: Path,
) -> int | None:
    row_motion_file = row.get("motion_file", "")
    normalized_row_path = _normalize_path(row_motion_file, manifest_dir) if row_motion_file else ""
    raw_motion_id = row.get("motion_id", "")
    if raw_motion_id != "":
        try:
            motion_id = int(raw_motion_id)
        except ValueError:
            motion_id = -1
        if 0 <= motion_id < len(manifest_entries):
            expected = str(manifest_entries[motion_id]["motion_file"])
            if not normalized_row_path or normalized_row_path == expected:
                return motion_id
    if normalized_row_path:
        return path_to_id.get(normalized_row_path)
    return None


def _load_eval_rows(
    csv_paths: list[Path],
    *,
    manifest_entries: list[dict[str, Any]],
    path_to_id: dict[str, int],
    manifest_dir: Path,
) -> tuple[dict[int, list[dict[str, str]]], Counter[str]]:
    rows_by_motion: dict[int, list[dict[str, str]]] = {idx: [] for idx in range(len(manifest_entries))}
    counters: Counter[str] = Counter()
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            for row in reader:
                motion_id = _row_motion_id(
                    row,
                    manifest_entries=manifest_entries,
                    path_to_id=path_to_id,
                    manifest_dir=manifest_dir,
                )
                if motion_id is None:
                    counters["unmatched_rows"] += 1
                    continue
                rows_by_motion[motion_id].append(row)
                counters["matched_rows"] += 1
        counters["csv_count"] += 1
    return rows_by_motion, counters


def _score_motion(
    motion_id: int,
    entry: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    early_fail_horizon: int,
    fail_weight: float,
    progress_weight: float,
    early_weight: float,
    unstable_weight: float,
) -> MotionScore:
    reps = len(rows)
    statuses = [row.get("status", "") for row in rows]
    pass_count = sum(1 for status in statuses if status == "pass")
    fail_count = reps - pass_count
    fail_rate = fail_count / reps if reps else 0.0
    progresses = [_float_value(row, "progress", 0.0) for row in rows]
    mean_progress = sum(progresses) / len(progresses) if progresses else 0.0
    fail_rows = [row for row in rows if row.get("status", "") != "pass"]
    fail_progresses = [_float_value(row, "progress", 0.0) for row in fail_rows]
    mean_fail_progress = sum(fail_progresses) / len(fail_progresses) if fail_progresses else 1.0
    fail_done_steps = [_float_value(row, "done_step", math.nan) for row in fail_rows]
    fail_done_steps = [value for value in fail_done_steps if math.isfinite(value)]
    median_fail_done_step = statistics.median(fail_done_steps) if fail_done_steps else math.nan
    early_fail_bonus = 0.0
    if math.isfinite(median_fail_done_step) and early_fail_horizon > 0:
        early_fail_bonus = 1.0 - min(max(median_fail_done_step, 0.0), early_fail_horizon) / early_fail_horizon
    unstable_bonus = 1.0 if reps > 0 and 0 < pass_count < reps else 0.0
    score = (
        fail_weight * fail_rate
        + progress_weight * (1.0 - mean_progress)
        + early_weight * early_fail_bonus
        + unstable_weight * unstable_bonus
    )
    return MotionScore(
        motion_id=motion_id,
        terrain_id=int(entry["terrain_id"]),
        motion_file=str(entry["motion_file"]),
        reps=reps,
        pass_count=pass_count,
        fail_count=fail_count,
        fail_rate=fail_rate,
        mean_progress=mean_progress,
        mean_fail_progress=mean_fail_progress,
        median_fail_done_step=median_fail_done_step,
        early_fail_bonus=early_fail_bonus,
        unstable_bonus=unstable_bonus,
        score=score,
    )


def _bucket_counts(target_count: int, hard_ratio: float, anchor_ratio: float, random_ratio: float) -> tuple[int, int, int]:
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    total = hard_ratio + anchor_ratio + random_ratio
    if total <= 0:
        raise ValueError("At least one bucket ratio must be positive.")
    hard_count = int(round(target_count * hard_ratio / total))
    anchor_count = int(round(target_count * anchor_ratio / total))
    random_count = target_count - hard_count - anchor_count
    if random_count < 0:
        anchor_count += random_count
        random_count = 0
    return hard_count, anchor_count, random_count


def _take_candidates(
    candidates: list[MotionScore],
    count: int,
    *,
    bucket: str,
    selected: dict[int, str],
    terrain_counts: Counter[int],
    max_per_terrain: int,
) -> None:
    if count <= 0:
        return
    for motion in candidates:
        if len([value for value in selected.values() if value == bucket]) >= count:
            return
        if motion.motion_id in selected:
            continue
        if max_per_terrain > 0 and terrain_counts[motion.terrain_id] >= max_per_terrain:
            continue
        selected[motion.motion_id] = bucket
        terrain_counts[motion.terrain_id] += 1


def _fill_to_target(
    candidates: list[MotionScore],
    target_count: int,
    *,
    selected: dict[int, str],
    terrain_counts: Counter[int],
    max_per_terrain: int,
    bucket: str,
) -> None:
    for motion in candidates:
        if len(selected) >= target_count:
            return
        if motion.motion_id in selected:
            continue
        if max_per_terrain > 0 and terrain_counts[motion.terrain_id] >= max_per_terrain:
            continue
        selected[motion.motion_id] = bucket
        terrain_counts[motion.terrain_id] += 1


def _select_motions(
    scores: list[MotionScore],
    *,
    target_count: int,
    hard_ratio: float,
    anchor_ratio: float,
    random_ratio: float,
    max_per_terrain: int,
    seed: int,
) -> dict[int, str]:
    rng = random.Random(seed)
    hard_count, anchor_count, random_count = _bucket_counts(target_count, hard_ratio, anchor_ratio, random_ratio)
    selected: dict[int, str] = {}
    terrain_counts: Counter[int] = Counter()

    hard_pool = [motion for motion in scores if motion.reps > 0 and motion.fail_count > 0]
    hard_pool.sort(key=lambda motion: (-motion.score, -motion.fail_rate, motion.mean_progress, motion.motion_id))
    anchor_pool = [motion for motion in scores if motion.reps > 0 and motion.fail_count == 0]
    rng.shuffle(anchor_pool)
    random_pool = list(scores)
    rng.shuffle(random_pool)

    _take_candidates(
        hard_pool,
        hard_count,
        bucket="hard",
        selected=selected,
        terrain_counts=terrain_counts,
        max_per_terrain=max_per_terrain,
    )
    _take_candidates(
        anchor_pool,
        anchor_count,
        bucket="anchor",
        selected=selected,
        terrain_counts=terrain_counts,
        max_per_terrain=max_per_terrain,
    )
    _take_candidates(
        random_pool,
        random_count,
        bucket="random",
        selected=selected,
        terrain_counts=terrain_counts,
        max_per_terrain=max_per_terrain,
    )

    fill_pool = sorted(scores, key=lambda motion: (-motion.score, motion.motion_id))
    _fill_to_target(
        fill_pool,
        target_count,
        selected=selected,
        terrain_counts=terrain_counts,
        max_per_terrain=max_per_terrain,
        bucket="fill",
    )
    _fill_to_target(
        fill_pool,
        target_count,
        selected=selected,
        terrain_counts=terrain_counts,
        max_per_terrain=0,
        bucket="fill_relaxed",
    )
    if len(selected) < min(target_count, len(scores)):
        raise RuntimeError(f"Only selected {len(selected)} motions for target_count={target_count}")
    return selected


def _write_scores_csv(path: Path, scores: list[MotionScore], selected: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "motion_id",
        "terrain_id",
        "motion_file",
        "reps",
        "pass_count",
        "fail_count",
        "fail_rate",
        "mean_progress",
        "mean_fail_progress",
        "median_fail_done_step",
        "early_fail_bonus",
        "unstable_bonus",
        "score",
        "selected_bucket",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for motion in sorted(scores, key=lambda value: (-value.score, value.motion_id)):
            median_done = "" if not math.isfinite(motion.median_fail_done_step) else f"{motion.median_fail_done_step:.6g}"
            writer.writerow(
                {
                    "motion_id": motion.motion_id,
                    "terrain_id": motion.terrain_id,
                    "motion_file": motion.motion_file,
                    "reps": motion.reps,
                    "pass_count": motion.pass_count,
                    "fail_count": motion.fail_count,
                    "fail_rate": f"{motion.fail_rate:.6g}",
                    "mean_progress": f"{motion.mean_progress:.6g}",
                    "mean_fail_progress": f"{motion.mean_fail_progress:.6g}",
                    "median_fail_done_step": median_done,
                    "early_fail_bonus": f"{motion.early_fail_bonus:.6g}",
                    "unstable_bonus": f"{motion.unstable_bonus:.6g}",
                    "score": f"{motion.score:.6g}",
                    "selected_bucket": selected.get(motion.motion_id, ""),
                }
            )


def _write_manifest(
    output_path: Path,
    source_manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    selected: dict[int, str],
    *,
    overwrite: bool,
    description: str,
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_entries = [entries[idx] for idx in sorted(selected)]
    terrains = [dict(entry) for entry in source_manifest.get("terrains", [])]
    manifest = {
        "schema_version": source_manifest.get("schema_version", 1),
        "description": description,
        "terrains": terrains,
        "motion_files": selected_entries,
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    source_manifest: Path,
    eval_csvs: list[Path],
    output_manifest: Path,
    scores_csv: Path | None,
    scores: list[MotionScore],
    selected: dict[int, str],
    csv_counters: Counter[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_scores = [score for score in scores if score.motion_id in selected]
    terrain_counts = Counter(score.terrain_id for score in selected_scores)
    bucket_counts = Counter(selected.values())
    payload = {
        "source_manifest": str(source_manifest),
        "eval_csvs": [str(path) for path in eval_csvs],
        "output_manifest": str(output_manifest),
        "scores_csv": str(scores_csv) if scores_csv else None,
        "target_count": args.target_count,
        "selected_count": len(selected),
        "bucket_counts": dict(bucket_counts),
        "terrain_counts": dict(sorted(terrain_counts.items())),
        "score_weights": {
            "fail": args.fail_weight,
            "progress": args.progress_weight,
            "early": args.early_weight,
            "unstable": args.unstable_weight,
        },
        "ratios": {
            "hard": args.hard_ratio,
            "anchor": args.anchor_ratio,
            "random": args.random_ratio,
        },
        "max_per_terrain": args.max_per_terrain,
        "early_fail_horizon": args.early_fail_horizon,
        "csv_counters": dict(csv_counters),
        "eval_standard_id": STANDARD_ID,
        "allow_nonstandard_eval_csv": bool(args.allow_nonstandard_eval_csv),
        "eval_csv_metadata": [
            _eval_standard_for_csv(path, allow_nonstandard=True)
            for path in eval_csvs
        ],
        "selected_motion_ids": sorted(selected),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_dynamic_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    source_manifest, entries, path_to_id = _load_manifest(manifest_path)
    eval_csvs = _expand_eval_csvs(
        args.eval_csv,
        args.eval_glob,
        allow_nonstandard=args.allow_nonstandard_eval_csv,
    )
    rows_by_motion, csv_counters = _load_eval_rows(
        eval_csvs,
        manifest_entries=entries,
        path_to_id=path_to_id,
        manifest_dir=manifest_path.parent,
    )
    scores = [
        _score_motion(
            motion_id,
            entry,
            rows_by_motion[motion_id],
            early_fail_horizon=args.early_fail_horizon,
            fail_weight=args.fail_weight,
            progress_weight=args.progress_weight,
            early_weight=args.early_weight,
            unstable_weight=args.unstable_weight,
        )
        for motion_id, entry in enumerate(entries)
    ]
    selected = _select_motions(
        scores,
        target_count=args.target_count,
        hard_ratio=args.hard_ratio,
        anchor_ratio=args.anchor_ratio,
        random_ratio=args.random_ratio,
        max_per_terrain=args.max_per_terrain,
        seed=args.seed,
    )
    selected_scores = [score for score in scores if score.motion_id in selected]
    description = (
        f"Dynamic motion-edit hard shard from {manifest_path.name}; "
        f"{len(selected)} selected from {len(entries)} motions."
    )
    _write_manifest(
        args.output_manifest,
        source_manifest,
        entries,
        selected,
        overwrite=args.overwrite,
        description=description,
    )
    if args.scores_csv is not None:
        _write_scores_csv(args.scores_csv.expanduser().resolve(), scores, selected)
    if args.summary_json is not None:
        _write_summary(
            args.summary_json.expanduser().resolve(),
            source_manifest=manifest_path,
            eval_csvs=eval_csvs,
            output_manifest=args.output_manifest.expanduser().resolve(),
            scores_csv=args.scores_csv.expanduser().resolve() if args.scores_csv else None,
            scores=scores,
            selected=selected,
            csv_counters=csv_counters,
            args=args,
        )
    return {
        "output_manifest": str(args.output_manifest.expanduser().resolve()),
        "selected_count": len(selected),
        "bucket_counts": dict(Counter(selected.values())),
        "terrain_counts": dict(sorted(Counter(score.terrain_id for score in selected_scores).items())),
        "csv_counters": dict(csv_counters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-csv", action="append", type=Path, default=[])
    parser.add_argument("--eval-glob", action="append", default=[])
    parser.add_argument(
        "--allow-nonstandard-eval-csv",
        action="store_true",
        help="Allow legacy or diagnostic eval CSVs. Formal shard building rejects them by default.",
    )
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--target-count", type=int, default=116)
    parser.add_argument("--hard-ratio", type=float, default=0.70)
    parser.add_argument("--anchor-ratio", type=float, default=0.20)
    parser.add_argument("--random-ratio", type=float, default=0.10)
    parser.add_argument("--max-per-terrain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-fail-horizon", type=int, default=200)
    parser.add_argument("--fail-weight", type=float, default=3.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--early-weight", type=float, default=0.5)
    parser.add_argument("--unstable-weight", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = build_dynamic_shard(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
