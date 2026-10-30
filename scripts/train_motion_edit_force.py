#!/usr/bin/env python3
"""Launch contact-force finetuning for a Motion Edit reference manifest."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = "exp:g1-29dof-wbt-contact-force"
DEFAULT_RUN_NAME = "motion_edit_force_ref_finetune"
DEFAULT_PROJECT = "MotionEditFinetune"
MOTION_EDIT_ROBOT_CONFIG = "robot:g1-29dof"


def _build_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    train_script = REPO_ROOT / "src/holosoma/holosoma/train_agent.py"
    headless = not args.gui
    completion_learning_sampler = bool(getattr(args, "completion_learning_sampler", False))
    completion_success_streak_threshold = int(getattr(args, "completion_success_streak_threshold", 3))
    completion_learned_replay_weight = float(getattr(args, "completion_learned_replay_weight", 0.1))
    completion_weight_beta = float(getattr(args, "completion_weight_beta", 0.05))

    cmd = [
        sys.executable,
        str(train_script),
        EXP_NAME,
        "simulator:isaaclab3-newton",
        MOTION_EDIT_ROBOT_CONFIG,
        "--training.checkpoint",
        str(args.checkpoint.expanduser()),
        "--training.project",
        args.project,
        "--training.name",
        args.name,
        "--training.num-envs",
        str(args.num_envs),
        "--training.headless",
        str(headless),
        "--training.export-onnx",
        "False",
        "--algo.config.num-learning-iterations",
        str(args.iterations),
        "--algo.config.actor-learning-rate",
        str(args.learning_rate),
        "--algo.config.critic-learning-rate",
        str(args.learning_rate),
        "--algo.config.load-optimizer",
        str(args.load_optimizer),
        "--algo.config.init-at-random-ep-len",
        str(args.init_at_random_ep_len),
        "--algo.config.export-onnx",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.motion-manifest",
        str(args.motion_manifest.expanduser()),
        "--command.setup-terms.motion-command.params.motion-config.canonicalize-motion-order-on-load",
        str(args.canonicalize_motion_order_on_load),
        "--command.setup-terms.motion-command.params.motion-config.reset-sampler",
        args.reset_sampler,
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob",
        str(args.start_at_timestep_zero_prob),
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.use-start-probe-envs",
        str(args.use_start_probe_envs),
        "--command.setup-terms.motion-command.params.motion-config.probe-env-per-motion",
        str(args.probe_env_per_motion),
        "--command.setup-terms.motion-command.params.motion-config.use-group-probe-envs",
        str(args.group_probe),
        "--command.setup-terms.motion-command.params.motion-config.group-probe-by",
        args.group_probe_by,
        "--command.setup-terms.motion-command.params.motion-config.probe-env-per-group",
        str(args.probe_env_per_group),
        "--command.setup-terms.motion-command.params.motion-config.group-variant-sample-count",
        str(args.group_variant_sample_count),
        "--command.setup-terms.motion-command.params.motion-config.use-completion-learning-sampler",
        str(completion_learning_sampler),
        "--command.setup-terms.motion-command.params.motion-config.completion-success-streak-threshold",
        str(completion_success_streak_threshold),
        "--command.setup-terms.motion-command.params.motion-config.completion-learned-replay-weight",
        str(completion_learned_replay_weight),
        "--command.setup-terms.motion-command.params.motion-config.completion-weight-beta",
        str(completion_weight_beta),
        "--terrain.terrain-term.motion-matched-manifest",
        str(args.motion_manifest.expanduser()),
        "--logger.video.enabled",
        "False",
    ]
    anchor_kl_checkpoint = getattr(args, "anchor_kl_checkpoint", None)
    anchor_kl_coef = float(getattr(args, "anchor_kl_coef", 0.0))
    if anchor_kl_checkpoint is not None and anchor_kl_coef > 0.0:
        cmd.extend(
            [
                "--algo.config.anchor-kl-checkpoint",
                str(anchor_kl_checkpoint.expanduser()),
                "--algo.config.anchor-kl-coef",
                str(anchor_kl_coef),
            ]
        )
    if args.save_interval is not None:
        cmd.extend(["--algo.config.save-interval", str(args.save_interval)])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Base policy checkpoint to finetune from.")
    parser.add_argument("--motion-manifest", required=True, type=Path, help="Motion-matched force-ref manifest.")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--reset-sampler", default="completion_recent30_failure_window")
    parser.add_argument("--start-at-timestep-zero-prob", type=float, default=0.2)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=True)
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
        "--completion-learning-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ordinary start0 episode completion streaks to downweight learned motions without probe envs.",
    )
    parser.add_argument("--completion-success-streak-threshold", type=int, default=3)
    parser.add_argument("--completion-learned-replay-weight", type=float, default=0.1)
    parser.add_argument("--completion-weight-beta", type=float, default=0.05)
    parser.add_argument(
        "--canonicalize-motion-order-on-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preorder loaded motion tensors for this force-ref finetune experiment.",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Local log project under logs/.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--gui", action="store_true", help="Run with Isaac Sim GUI enabled.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    args, extra_args = parser.parse_known_args()

    cmd = _build_command(args, extra_args)
    print("Command:")
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    from somaforge_core.runtime_paths import newton_subprocess_environment
    env = newton_subprocess_environment()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
