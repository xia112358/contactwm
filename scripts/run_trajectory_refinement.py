#!/usr/bin/env python3
"""Prepare or explicitly execute a manifest-driven joint policy/rollout workflow.

This is a CPU controller, not a simulator entrypoint. Simulator options are
forwarded to existing entrypoints where AppLauncher parses them before tyro.
"""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path

from somaforge_core import trajectory_refinement as pipeline


def dispatch(args, parser):
    launcher = dict(device=args.device, visualizer=args.visualizer, headless=args.headless)
    if args.stage == "prepare":
        plan = pipeline.prepare(args.config, execute=args.execute)
        return dict(stage="prepare", written=args.execute, reference_count=len(plan["cases"]),
                    terrain_count=len(plan["manifest"]["terrains"]), output_dir=plan["output_dir"],
                    gpu_started=False, checkpoint_metadata_check="deferred_to_train_or_evaluation")
    plan = pipeline.load_plan(args.config)
    if args.stage in ("evaluate", "collect", "export") and not args.checkpoint:
        parser.error(f"--stage {args.stage} requires --checkpoint")
    if args.stage == "status":
        return pipeline.status(plan)
    if args.stage == "train":
        return pipeline.train(plan, checkpoint=args.checkpoint, execute=args.execute, **launcher)
    if args.stage == "export":
        return pipeline.export_dataset(plan, args.checkpoint, execute=args.execute)
    return pipeline.evaluate_or_collect(plan, args.stage, args.checkpoint, execute=args.execute, **launcher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "collect", "export", "status"), default="prepare")
    parser.add_argument("--execute", action="store_true", help="Execute this stage. Otherwise only inspect inputs/print commands.")
    parser.add_argument("--checkpoint", type=Path, help="Resume checkpoint for train; fixed refined checkpoint for later stages.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualizer", choices=("kit",))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    config, _ = pipeline.load_config(args.config)
    directory = pipeline.resolve(config["output_dir"], pipeline.somaforge_root())
    lock = nullcontext()
    if args.execute:
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / "controller.lock").open("a")
    with lock as stream:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = dispatch(args, parser)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
