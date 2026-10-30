"""CPU controller: one complete balanced inference batch and aligned consensus.

Simulator flags are forwarded to eval_agent's official AppLauncher parser.
"""
import argparse
import fcntl
import json
from pathlib import Path

from somaforge_core.parallel_rollout_collection import collect
from somaforge_core.trajectory_refinement import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envs-per-motion", type=int, default=16)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualizer", choices=["kit"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.training_plan)
    kwargs = dict(execute=args.execute, output=args.output_dir, envs_per_motion=args.envs_per_motion,
        chunk_frames=args.chunk_frames, device=args.device, visualizer=args.visualizer,
        headless=args.headless or not args.visualizer)
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir/"controller.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = collect(plan, args.checkpoint, **kwargs)
    else:
        result = collect(plan, args.checkpoint, **kwargs)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
