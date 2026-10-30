#!/usr/bin/env python3
"""Run motion-edit force-ref finetuning one manifest shard at a time."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from train_motion_edit_force import DEFAULT_PROJECT, REPO_ROOT, _build_command

DEFAULT_RUN_NAME = "motion_edit_force_ref_sharded"
INDEX_KIND = "motion_edit_force_ref_shard_index"


def _load_shard_index(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != INDEX_KIND:
        raise ValueError(f"Unsupported shard index: {path}")
    shards = data.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Shard index has no shards: {path}")
    return data


def _resolve_index_path(value: str, index_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = index_dir / path
    return path.resolve()


def _selected_shards(index: dict[str, Any], start_shard: int, num_shards: int | None) -> list[dict[str, Any]]:
    shards = list(index["shards"])
    if start_shard < 0 or start_shard >= len(shards):
        raise ValueError(f"start_shard must be in [0, {len(shards) - 1}], got {start_shard}")
    end = len(shards) if num_shards is None else min(len(shards), start_shard + max(num_shards, 0))
    if end <= start_shard:
        raise ValueError("Selected shard range is empty")
    return shards[start_shard:end]


def _iteration_plan(num_shards: int, total_iterations: int, iterations_per_shard: int | None) -> list[int]:
    if iterations_per_shard is not None:
        if iterations_per_shard <= 0:
            raise ValueError(f"iterations_per_shard must be positive, got {iterations_per_shard}")
        return [iterations_per_shard] * num_shards
    if total_iterations <= 0:
        raise ValueError(f"total_iterations must be positive, got {total_iterations}")
    base = total_iterations // num_shards
    rem = total_iterations % num_shards
    return [base + (1 if i < rem else 0) for i in range(num_shards)]


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _latest_checkpoint_for_run(run_name: str, project: str) -> Path:
    log_root = REPO_ROOT / "logs" / project
    run_dirs = sorted(
        log_root.glob(f"*-{run_name}-locomotion"),
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No log directory found for project {project!r}, training name {run_name!r}")
    checkpoints = sorted(run_dirs[-1].glob("model_*.pt"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dirs[-1]}")
    return checkpoints[-1]


def _build_shard_command(
    *,
    checkpoint: Path,
    motion_manifest: Path,
    name: str,
    args: argparse.Namespace,
    iterations: int,
    extra_args: list[str],
) -> list[str]:
    train_args = SimpleNamespace(
        checkpoint=checkpoint,
        motion_manifest=motion_manifest,
        num_envs=args.num_envs,
        iterations=iterations,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        reset_sampler=args.reset_sampler,
        start_at_timestep_zero_prob=args.start_at_timestep_zero_prob,
        load_optimizer=args.load_optimizer,
        init_at_random_ep_len=args.init_at_random_ep_len,
        use_start_probe_envs=args.use_start_probe_envs,
        probe_env_per_motion=args.probe_env_per_motion,
        group_probe=args.group_probe,
        group_probe_by=args.group_probe_by,
        probe_env_per_group=args.probe_env_per_group,
        group_variant_sample_count=args.group_variant_sample_count,
        canonicalize_motion_order_on_load=args.canonicalize_motion_order_on_load,
        project=args.project,
        name=name,
        gui=args.gui,
    )
    return _build_command(train_args, extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Base policy checkpoint for the first shard.")
    parser.add_argument(
        "--shard-index",
        required=True,
        type=Path,
        help="shard_index.json from build_motion_manifest_shards.py.",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--total-iterations", type=int, default=2000)
    parser.add_argument("--iterations-per-shard", type=int)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--reset-sampler", default="completion_recent30_failure_window")
    parser.add_argument("--start-at-timestep-zero-prob", type=float, default=0.2)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-at-random-ep-len", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-start-probe-envs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--probe-env-per-motion", type=int, default=0)
    parser.add_argument(
        "--group-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable group-balanced probe envs for motion-edit generated variants.",
    )
    parser.add_argument("--group-probe-by", choices=("terrain_id", "climb_id"), default="terrain_id")
    parser.add_argument("--probe-env-per-group", type=int, default=8)
    parser.add_argument(
        "--group-variant-sample-count",
        type=int,
        default=0,
        help="Limit each group to this many random motion variants per reset batch; 0 uses all variants.",
    )
    parser.add_argument(
        "--canonicalize-motion-order-on-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preorder loaded motion tensors for this force-ref finetune experiment.",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Local log project under logs/.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--gui", action="store_true", help="Run with Isaac Sim GUI enabled.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    args, extra_args = parser.parse_known_args()

    index_path = args.shard_index.expanduser().resolve()
    index = _load_shard_index(index_path)
    shards = _selected_shards(index, args.start_shard, args.num_shards)
    iterations = _iteration_plan(len(shards), args.total_iterations, args.iterations_per_shard)

    current_checkpoint = args.checkpoint.expanduser().resolve()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print(
        f"Running {len(shards)} shard(s), total scheduled iterations={sum(iterations)}, "
        f"start checkpoint={current_checkpoint}",
        flush=True,
    )
    for local_idx, (shard, shard_iterations) in enumerate(zip(shards, iterations)):
        shard_id = int(shard["shard_id"])
        manifest_path = _resolve_index_path(str(shard["manifest_path"]), index_path.parent)
        run_name = f"{args.name}_shard{shard_id:02d}of{int(index['num_shards']):02d}"
        group_keys = shard.get("group_keys", [])
        print(
            f"\nShard {local_idx + 1}/{len(shards)}: id={shard_id}, "
            f"groups={group_keys}, motions={shard.get('motion_count')}, iterations={shard_iterations}",
            flush=True,
        )
        cmd = _build_shard_command(
            checkpoint=current_checkpoint,
            motion_manifest=manifest_path,
            name=run_name,
            args=args,
            iterations=shard_iterations,
            extra_args=extra_args,
        )
        print("Command:")
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        current_checkpoint = _latest_checkpoint_for_run(run_name, args.project)
        print(f"Next checkpoint: {current_checkpoint}", flush=True)

    if not args.dry_run:
        print(f"\nFinal checkpoint: {current_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
