#!/usr/bin/env python3
"""Train one PHP skill collection with SomaForge's standard Newton WBT recipe."""

# Simulator-dependent imports follow official AppLauncher argument parsing.
# ruff: noqa: PLC0415
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config


@dataclass(frozen=True)
class Options:
    manifest: Path
    output_dir: Path
    checkpoint: Path | None = None


def main() -> None:
    launcher = parse_isaaclab_launcher_args("Train one PHP dataset with canonical SomaForge WBT.")
    options = tyro.cli(Options)
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_baseline_29
    from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest

    from holosoma import train_agent

    manifest = options.manifest.expanduser().resolve()
    dataset = load_motion_terrain_manifest(str(manifest))
    if dataset.get("source_kind") != "php_release_direct_newton_fk":
        raise ValueError("Expected a verified PHP canonical dataset")
    base = g1_29dof_wbt_baseline_29
    if base.training.num_envs <= 3 * len(dataset["motion_files"]):
        raise ValueError("Need normal training environments in addition to three probes per motion")
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None if options.checkpoint is None else str(options.checkpoint.expanduser().resolve())
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    spacing = dataset["terrain_spacing"]
    setup_terms = dict(base.command.setup_terms)
    motion_term = setup_terms["motion_command"]
    motion_params = dict(motion_term.params)
    motion_params["motion_config"] = dataclasses.replace(
        motion_params["motion_config"], motion_manifest=str(manifest)
    )
    setup_terms["motion_command"] = dataclasses.replace(motion_term, params=motion_params)
    termination_terms = dict(base.termination.terms)
    bad_tracking = termination_terms["bad_tracking"]
    termination_terms["bad_tracking"] = dataclasses.replace(
        bad_tracking,
        params={
            **bad_tracking.params,
            "bad_ref_pos_threshold": 0.5,
            "bad_motion_body_pos_threshold": 0.25,
            "bad_motion_body_pos_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_knee_link",
                "right_knee_link",
            ],
        },
    )
    config = dataclasses.replace(
        base,
        training=dataclasses.replace(
            base.training,
            project="PHP_WBT",
            name=dataset["php_component"],
            checkpoint=checkpoint,
        ),
        command=dataclasses.replace(base.command, setup_terms=setup_terms),
        termination=dataclasses.replace(base.termination, terms=termination_terms),
        terrain=dataclasses.replace(
            base.terrain,
            terrain_term=dataclasses.replace(
                base.terrain.terrain_term,
                motion_matched_manifest=str(manifest),
                terrain_length=float(spacing[0]),
                terrain_width=float(spacing[1]),
                spawn=dataclasses.replace(base.terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.0),
            ),
        ),
        logger=dataclasses.replace(base.logger, base_dir=str(output)),
    )
    config = sync_launcher_headless_config(config, launcher)
    record = {
        "schema": "php_wbt_launch_v1",
        "manifest": str(manifest),
        "component": dataset["php_component"],
        "motion_count": len(dataset["motion_files"]),
        "recipe": "g1_29dof_wbt_baseline_29",
        "bad_tracking": termination_terms["bad_tracking"].params,
        "options": dataclasses.asdict(options),
        "num_envs": config.training.num_envs,
        "iterations": config.algo.config.num_learning_iterations,
        "save_interval": config.algo.config.save_interval,
        "seed": config.training.seed,
        "terrain_variant": dataset["php_terrain_variant"],
        "terrain_spacing": spacing,
    }
    # Each invocation gets a separate record and log directory; historical runs stay intact.
    import time

    (output / f"launch_{time.time_ns()}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(record, indent=2, default=str), flush=True)
    train_agent.train(config, launcher_args=launcher)


if __name__ == "__main__":
    main()
