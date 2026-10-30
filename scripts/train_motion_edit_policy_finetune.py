#!/usr/bin/env python3
"""Fine-tune an existing WBT policy on one generated motion reference."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import tyro
from holosoma import train_agent
from holosoma.config_values.wbt.g1.command import motion_config as default_motion_config
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.sim_utils import (
    parse_isaaclab_launcher_args,
    sync_launcher_headless_config,
)


@dataclass(frozen=True)
class FinetuneOptions:
    checkpoint: Path
    motion_manifest: Path
    output_dir: Path
    project: str = "MotionEditPolicyFinetune"
    name: str = "motion_edit_single"
    num_envs: int = 4096
    iterations: int = 10000
    save_interval: int = 500
    max_episode_length_s: float = 22.0
    full_checkpoints: bool = False
    """Deprecated compatibility option; ordinary checkpoints are always used."""
    warm_start: bool = False


def _replace_motion_config(command, manifest: Path):
    setup_terms = dict(command.setup_terms)
    motion_term = setup_terms["motion_command"]
    params = dict(motion_term.params)
    motion = params["motion_config"]
    params["motion_config"] = dataclasses.replace(
        motion,
        motion_file="",
        motion_dir="",
        motion_manifest=str(manifest),
        canonicalize_motion_order_on_load=True,
        reset_sampler="completion_recent30_failure_window",
        start_at_timestep_zero_prob=0.0,
        freeze_at_timestep_zero_prob=0.0,
        use_start_probe_envs=True,
        probe_env_per_motion=3,
        probe_uniform_mix=0.4,
        # Use current defaults even when the checkpoint contains older windows.
        failure_window_pre_frames=default_motion_config.failure_window_pre_frames,
        failure_window_post_frames=20,
        failure_window_before_prob=default_motion_config.failure_window_before_prob,
        failure_window_success_horizon_frames=50,
        hotspot_failure_uniform_mix=0.3,
        hotspot_failure_decay=0.995,
        hotspot_failure_min_count=1.0,
        use_group_probe_envs=False,
        chain_motion_segments=False,
        hold_at_motion_end_in_eval=False,
        local_motion_segment_reference=False,
    )
    setup_terms["motion_command"] = dataclasses.replace(
        motion_term,
        params=params,
    )
    return dataclasses.replace(command, setup_terms=setup_terms)


def main() -> None:
    launcher_args = parse_isaaclab_launcher_args(
        "Fine-tune a WBT policy on one Motion Edit reference."
    )
    options = tyro.cli(FinetuneOptions)
    checkpoint = options.checkpoint.expanduser().resolve()
    manifest = options.motion_manifest.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if options.num_envs <= 0 or options.iterations <= 0:
        raise ValueError("num_envs and iterations must be positive")
    if options.save_interval <= 0:
        raise ValueError("save_interval must be positive")

    config, _ = load_saved_experiment_config(
        CheckpointConfig(checkpoint=str(checkpoint))
    )
    if not bool(config.robot.asset.enable_self_collisions):
        raise ValueError(
            "Motion Edit fine-tuning requires a checkpoint trained with "
            "robot.asset.enable_self_collisions=True"
        )
    mujoco_warp = config.simulator.config.mujoco_warp
    if (
        int(mujoco_warp.nconmax_per_env) < 160
        or int(mujoco_warp.njmax_per_env) < 1024
    ):
        raise ValueError(
            "Motion Edit fine-tuning requires the retained self-collision "
            "capacity: nconmax_per_env>=160 and njmax_per_env>=1024"
        )
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            checkpoint=str(checkpoint),
            num_envs=int(options.num_envs),
            project=str(options.project),
            name=str(options.name),
        ),
        algo=dataclasses.replace(
            config.algo,
            config=dataclasses.replace(
                config.algo.config,
                num_learning_iterations=int(options.iterations),
                save_interval=int(options.save_interval),
                full_checkpoints=False,
                warm_start=bool(options.warm_start),
            ),
        ),
        command=_replace_motion_config(config.command, manifest),
        terrain=dataclasses.replace(
            config.terrain,
            terrain_term=dataclasses.replace(
                config.terrain.terrain_term,
                motion_matched_manifest=str(manifest),
                spawn=dataclasses.replace(
                    config.terrain.terrain_term.spawn,
                    randomize_tiles=False,
                    xy_offset_range=0.0,
                ),
            ),
        ),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(
                    config.simulator.config.sim,
                    max_episode_length_s=float(options.max_episode_length_s),
                ),
            ),
        ),
        logger=dataclasses.replace(
            config.logger,
            base_dir=str(output_dir),
            headless_recording=False,
            video=dataclasses.replace(config.logger.video, enabled=False),
        ),
    )
    config = sync_launcher_headless_config(config, launcher_args)
    print(f"checkpoint={checkpoint}")
    print(f"motion_manifest={manifest}")
    print(f"output_dir={output_dir}")
    print(f"num_envs={options.num_envs}")
    print(f"iterations={options.iterations}")
    train_agent.train(config, launcher_args=launcher_args)


if __name__ == "__main__":
    main()
