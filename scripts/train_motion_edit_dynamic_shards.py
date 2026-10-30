#!/usr/bin/env python3
"""Stage motion-edit finetuning on dynamic hard shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from somaforge_core import stage_spec
from train_motion_edit_force import DEFAULT_PROJECT, REPO_ROOT, _build_command

DEFAULT_MANIFEST = REPO_ROOT / "runtime/current/manifests/motion_edit_ref_v1.json"
DEFAULT_SHARD00 = (
    REPO_ROOT
    / "runtime/current/manifests/motion_edit_ref_v1_shards"
    / "motion_edit_raw29_large_mixed_n64_force_ref_manifest_shard_00.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs/motion_matched/motion_edit_dynamic_hard_shards"
DEFAULT_EVAL_OUTPUT_DIR = REPO_ROOT / "logs" / DEFAULT_PROJECT / "eval_csv"
DEFAULT_RUN_NAME = "motion_edit_dynamic_hard_shard"
FORMAL_EVAL_REPS = 3


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _latest_checkpoint_for_run(run_name: str, project: str) -> Path:
    log_root = REPO_ROOT / "logs" / project
    run_dirs = sorted(
        log_root.glob(f"*-{run_name}-locomotion"),
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No log directory found for project {project!r}, training name {run_name!r}")
    checkpoints = sorted(run_dirs[-1].glob("model_*.pt"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dirs[-1]}")
    return checkpoints[-1]


def _load_motion_count(manifest_path: Path) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("motion_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no motion_files: {manifest_path}")
    return len(entries)


def _resolve_initial_manifest(args: argparse.Namespace) -> Path:
    if args.initial_shard_manifest is not None:
        return args.initial_shard_manifest.expanduser().resolve()
    motion_manifest = args.motion_manifest.expanduser().resolve()
    if motion_manifest == DEFAULT_MANIFEST.resolve() and DEFAULT_SHARD00.exists():
        return DEFAULT_SHARD00.resolve()
    return motion_manifest


def _extra_algo_args(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    generated: list[str] = []
    if args.min_learning_rate is not None:
        generated.extend(
            [
                "--algo.config.min-actor-learning-rate",
                str(args.min_learning_rate),
                "--algo.config.min-critic-learning-rate",
                str(args.min_learning_rate),
            ]
        )
    if args.max_learning_rate is not None:
        generated.extend(
            [
                "--algo.config.max-actor-learning-rate",
                str(args.max_learning_rate),
                "--algo.config.max-critic-learning-rate",
                str(args.max_learning_rate),
            ]
        )
    if args.desired_kl is not None:
        generated.extend(["--algo.config.desired-kl", str(args.desired_kl)])
    if args.schedule is not None:
        generated.extend(["--algo.config.schedule", args.schedule])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return generated + extra_args


def _build_train_command(
    *,
    checkpoint: Path,
    motion_manifest: Path,
    name: str,
    args: argparse.Namespace,
    extra_args: list[str],
) -> list[str]:
    train_args = SimpleNamespace(
        checkpoint=checkpoint,
        motion_manifest=motion_manifest,
        num_envs=args.num_envs,
        iterations=args.stage_iterations,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        reset_sampler=args.reset_sampler,
        start_at_timestep_zero_prob=args.start_at_timestep_zero_prob,
        load_optimizer=args.load_optimizer,
        anchor_kl_checkpoint=args.anchor_kl_checkpoint,
        anchor_kl_coef=args.anchor_kl_coef,
        init_at_random_ep_len=args.init_at_random_ep_len,
        use_start_probe_envs=args.use_start_probe_envs,
        probe_env_per_motion=args.probe_env_per_motion,
        group_probe=args.group_probe,
        group_probe_by=args.group_probe_by,
        probe_env_per_group=args.probe_env_per_group,
        group_variant_sample_count=args.group_variant_sample_count,
        completion_learning_sampler=args.completion_learning_sampler,
        completion_success_streak_threshold=args.completion_success_streak_threshold,
        completion_learned_replay_weight=args.completion_learned_replay_weight,
        completion_weight_beta=args.completion_weight_beta,
        canonicalize_motion_order_on_load=args.canonicalize_motion_order_on_load,
        project=args.project,
        name=name,
        gui=args.gui,
    )
    return _build_command(train_args, _extra_algo_args(args, extra_args))


def _build_eval_command(
    *,
    checkpoint: Path,
    name: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python,
        str(REPO_ROOT / "scripts/eval_motion_edit_acceptance.py"),
        "--checkpoint",
        str(checkpoint),
        "--motion-manifest",
        str(args.motion_manifest.expanduser().resolve()),
        "--name",
        name,
        "--project",
        args.project,
        "--output-dir",
        str(args.eval_output_dir.expanduser().resolve()),
        "--python",
        args.python,
        "--device",
        args.device,
        "--eval-step-margin",
        str(args.eval_step_margin),
        "--repeats",
        str(args.eval_reps),
    ]
    if args.allow_diagnostic_eval:
        cmd.append("--allow-diagnostic")
    if args.max_eval_steps is not None:
        cmd.extend(["--max-eval-steps", str(args.max_eval_steps)])
    if args.compare_against is not None:
        cmd.extend(["--compare-against", str(args.compare_against.expanduser().resolve())])
    return cmd


def _with_repeat_suffix(path: Path, rep_index: int) -> Path:
    stem = path.stem
    suffix = path.suffix
    replacements = (
        ("_acceptance_summary", f"_rep{rep_index}_acceptance_summary"),
        ("_acceptance", f"_rep{rep_index}_acceptance"),
        ("_fail", f"_rep{rep_index}_fail"),
    )
    for old, new in replacements:
        if stem.endswith(old):
            return path.with_name(f"{stem[: -len(old)]}{new}{suffix}")
    return path.with_name(f"{stem}_rep{rep_index}{suffix}")


def _eval_csv_paths(args: argparse.Namespace, name: str) -> list[Path]:
    base = args.eval_output_dir.expanduser().resolve() / f"{name}_acceptance.csv"
    if args.eval_reps <= 1:
        return [base]
    return [_with_repeat_suffix(base, rep_index) for rep_index in range(1, args.eval_reps + 1)]


def _summarize_eval(csv_paths: list[Path], total_motions: int) -> dict[str, object]:
    rows_by_motion: dict[int, list[str]] = {idx: [] for idx in range(total_motions)}
    per_rep_pass_counts: list[int] = []
    for csv_path in csv_paths:
        pass_count = 0
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                motion_id = int(row["motion_id"])
                status = row.get("status", "")
                if 0 <= motion_id < total_motions:
                    rows_by_motion[motion_id].append(status)
                if status == "pass":
                    pass_count += 1
        per_rep_pass_counts.append(pass_count)
    reps = len(csv_paths)
    majority_threshold = reps // 2 + 1
    majority_pass = 0
    pass_all = 0
    pass_any = 0
    for statuses in rows_by_motion.values():
        count = sum(1 for status in statuses if status == "pass")
        if count >= majority_threshold:
            majority_pass += 1
        if count == reps:
            pass_all += 1
        if count > 0:
            pass_any += 1
    return {
        "csv_paths": [str(path) for path in csv_paths],
        "total_motions": total_motions,
        "reps": reps,
        "per_rep_pass_counts": per_rep_pass_counts,
        "majority_pass": majority_pass,
        "majority_rate": majority_pass / total_motions if total_motions else math.nan,
        "pass_all": pass_all,
        "pass_any": pass_any,
    }


def _build_dynamic_manifest_command(
    *,
    eval_csvs: list[Path],
    output_manifest: Path,
    scores_csv: Path,
    summary_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_motion_edit_dynamic_shard.py"),
        "--manifest",
        str(args.motion_manifest.expanduser().resolve()),
        "--output-manifest",
        str(output_manifest),
        "--scores-csv",
        str(scores_csv),
        "--summary-json",
        str(summary_json),
        "--target-count",
        str(args.target_count),
        "--hard-ratio",
        str(args.hard_ratio),
        "--anchor-ratio",
        str(args.anchor_ratio),
        "--random-ratio",
        str(args.random_ratio),
        "--max-per-terrain",
        str(args.max_per_terrain),
        "--seed",
        str(args.seed),
        "--early-fail-horizon",
        str(args.early_fail_horizon),
        "--fail-weight",
        str(args.fail_weight),
        "--progress-weight",
        str(args.progress_weight),
        "--early-weight",
        str(args.early_weight),
        "--unstable-weight",
        str(args.unstable_weight),
        "--overwrite",
    ]
    for csv_path in eval_csvs:
        cmd.extend(["--eval-csv", str(csv_path)])
    if args.allow_diagnostic_eval:
        cmd.append("--allow-nonstandard-eval-csv")
    return cmd


def _print_command(cmd: list[str]) -> None:
    print(shlex.join(cmd), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--initial-shard-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Local log project under logs/.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--stage-iterations", type=int, default=400)
    parser.add_argument("--target-count", type=int, default=116)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--max-learning-rate", type=float)
    parser.add_argument("--desired-kl", type=float)
    parser.add_argument("--schedule", choices=("adaptive", "fixed"))
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--reset-sampler", default="failure_window")
    parser.add_argument("--start-at-timestep-zero-prob", type=float, default=0.2)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--anchor-kl-checkpoint",
        type=Path,
        default=None,
        help="Frozen reference checkpoint for KL(current || reference) actor regularization.",
    )
    parser.add_argument("--anchor-kl-coef", type=float, default=0.0)
    parser.add_argument("--init-at-random-ep-len", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-start-probe-envs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--probe-env-per-motion", type=int, default=0)
    parser.add_argument("--group-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-probe-by", choices=("terrain_id", "climb_id"), default="terrain_id")
    parser.add_argument("--probe-env-per-group", type=int, default=8)
    parser.add_argument(
        "--group-variant-sample-count",
        type=int,
        default=0,
        help="Limit each group to this many random motion variants per reset batch; 0 uses all variants.",
    )
    parser.add_argument("--completion-learning-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--completion-success-streak-threshold", type=int, default=3)
    parser.add_argument("--completion-learned-replay-weight", type=float, default=0.1)
    parser.add_argument("--completion-weight-beta", type=float, default=0.05)
    parser.add_argument("--canonicalize-motion-order-on-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-reps", type=int, default=3)
    parser.add_argument(
        "--allow-diagnostic-eval",
        action="store_true",
        help="Allow non-standard eval settings and pass diagnostic CSVs into shard building.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--eval-step-margin", type=int, default=8)
    parser.add_argument("--compare-against", type=Path)
    parser.add_argument("--baseline-majority-rate", type=float)
    parser.add_argument("--accept-min-improvement", type=float, default=0.0)
    parser.add_argument("--rollback-on-regression", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    stage_spec("motion_edit")

    if args.stages <= 0:
        raise ValueError(f"stages must be positive, got {args.stages}")
    if args.eval_reps <= 0:
        raise ValueError(f"eval_reps must be positive, got {args.eval_reps}")
    if not args.allow_diagnostic_eval and args.eval_reps != FORMAL_EVAL_REPS:
        raise ValueError(
            f"Formal motion-edit eval requires --eval-reps {FORMAL_EVAL_REPS}; "
            "use --allow-diagnostic-eval only for temporary diagnostics."
        )
    if not args.allow_diagnostic_eval and args.max_eval_steps is not None:
        raise ValueError(
            "Formal motion-edit eval uses automatic max_eval_steps=max_motion_len+eval_step_margin; "
            "use --allow-diagnostic-eval only for temporary diagnostics."
        )

    args.output_dir = args.output_dir.expanduser().resolve()
    args.eval_output_dir = args.eval_output_dir.expanduser().resolve()
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.eval_output_dir.mkdir(parents=True, exist_ok=True)

    total_motions = _load_motion_count(args.motion_manifest.expanduser().resolve())
    current_checkpoint = args.checkpoint.expanduser().resolve()
    current_manifest = _resolve_initial_manifest(args)
    best_checkpoint = current_checkpoint
    best_rate = args.baseline_majority_rate if args.baseline_majority_rate is not None else -1.0
    history: list[dict[str, object]] = []

    print(
        f"Dynamic hard-shard training: stages={args.stages}, stage_iterations={args.stage_iterations}, "
        f"initial_manifest={current_manifest}, start_checkpoint={current_checkpoint}",
        flush=True,
    )

    for stage in range(args.stages):
        run_name = f"{args.name}_stage{stage:02d}"
        print(f"\nStage {stage + 1}/{args.stages}: train manifest={current_manifest}", flush=True)
        train_cmd = _build_train_command(
            checkpoint=current_checkpoint,
            motion_manifest=current_manifest,
            name=run_name,
            args=args,
            extra_args=extra_args,
        )
        print("Train command:", flush=True)
        _print_command(train_cmd)
        if not args.dry_run:
            subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
            stage_checkpoint = _latest_checkpoint_for_run(run_name, args.project)
        else:
            stage_checkpoint = current_checkpoint

        eval_name = f"{args.name}_stage{stage:02d}"
        eval_cmd = _build_eval_command(checkpoint=stage_checkpoint, name=eval_name, args=args)
        eval_csvs = _eval_csv_paths(args, eval_name)
        print("Eval command:", flush=True)
        _print_command(eval_cmd)
        if not args.dry_run:
            subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)

        if args.dry_run:
            eval_summary = {
                "csv_paths": [str(path) for path in eval_csvs],
                "total_motions": total_motions,
                "reps": args.eval_reps,
                "majority_pass": None,
                "majority_rate": None,
            }
            accepted = True
        else:
            eval_summary = _summarize_eval(eval_csvs, total_motions)
            rate = float(eval_summary["majority_rate"])
            accepted = rate >= best_rate + args.accept_min_improvement
            if accepted:
                best_rate = rate
                best_checkpoint = stage_checkpoint
            elif args.rollback_on_regression:
                current_checkpoint = best_checkpoint

        if accepted or not args.rollback_on_regression:
            current_checkpoint = stage_checkpoint

        stage_record = {
            "stage": stage,
            "run_name": run_name,
            "train_manifest": str(current_manifest),
            "checkpoint": str(stage_checkpoint),
            "accepted": accepted,
            "best_checkpoint": str(best_checkpoint),
            "best_majority_rate": best_rate,
            "eval_summary": eval_summary,
        }
        history.append(stage_record)
        print(f"Stage summary: {json.dumps(stage_record, indent=2)}", flush=True)

        if stage == args.stages - 1:
            continue

        next_manifest = args.output_dir / f"{args.name}_stage{stage + 1:02d}_manifest.json"
        scores_csv = args.output_dir / f"{args.name}_stage{stage:02d}_scores.csv"
        shard_summary_json = args.output_dir / f"{args.name}_stage{stage:02d}_shard_summary.json"
        build_cmd = _build_dynamic_manifest_command(
            eval_csvs=eval_csvs,
            output_manifest=next_manifest,
            scores_csv=scores_csv,
            summary_json=shard_summary_json,
            args=args,
        )
        print("Build next-shard command:", flush=True)
        _print_command(build_cmd)
        if not args.dry_run:
            subprocess.run(build_cmd, cwd=REPO_ROOT, check=True)
        current_manifest = next_manifest

    history_path = args.output_dir / f"{args.name}_history.json"
    if not args.dry_run:
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\nBest checkpoint: {best_checkpoint}", flush=True)
    print(f"Best majority rate: {best_rate}", flush=True)
    print(f"History: {history_path}", flush=True)


if __name__ == "__main__":
    main()
