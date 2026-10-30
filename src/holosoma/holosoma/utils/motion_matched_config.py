from __future__ import annotations

import dataclasses
from typing import Any

from loguru import logger

from holosoma.config_types.command import CommandManagerCfg, MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.terrain import MeshType


def normalize_motion_matched_config(config: ExperimentConfig) -> ExperimentConfig:
    """Make motion-matched training use one manifest for both motion and terrain."""
    manifest = _get_motion_matched_manifest(config)
    if not manifest:
        return config

    terrain = config.terrain
    terrain_term = getattr(config.terrain, "terrain_term", None)
    if terrain_term is not None:
        terrain_term = dataclasses.replace(
            terrain_term,
            mesh_type=MeshType.LOAD_OBJ,
            motion_matched_manifest=manifest,
        )
        terrain = dataclasses.replace(terrain, terrain_term=terrain_term)

    command = _apply_motion_matched_manifest_to_command(config.command, manifest)

    logger.info("Using motion-matched manifest for motion and terrain: {}", manifest)
    return dataclasses.replace(config, terrain=terrain, command=command)


def _get_motion_matched_manifest(config: ExperimentConfig) -> str:
    terrain_term = getattr(config.terrain, "terrain_term", None)
    terrain_manifest = getattr(terrain_term, "motion_matched_manifest", "") or ""
    command_manifest = _get_command_motion_manifest(config.command)

    if command_manifest and terrain_manifest and command_manifest != terrain_manifest:
        raise ValueError(
            "motion_manifest and motion_matched_manifest must be identical when both are set: "
            f"command={command_manifest!r}, terrain={terrain_manifest!r}"
        )
    return command_manifest or terrain_manifest


def _get_command_motion_manifest(command: CommandManagerCfg | None) -> str:
    if command is None:
        return ""

    motion_term = command.setup_terms.get("motion_command")
    if motion_term is None:
        return ""

    motion_config = motion_term.params.get("motion_config")
    if motion_config is None:
        return ""

    if isinstance(motion_config, MotionConfig):
        return motion_config.motion_manifest
    if isinstance(motion_config, dict):
        return str(motion_config.get("motion_manifest") or "")
    return ""


def _apply_motion_matched_manifest_to_command(
    command: CommandManagerCfg | None,
    manifest: str,
) -> CommandManagerCfg | None:
    if command is None:
        return None

    motion_term = command.setup_terms.get("motion_command")
    if motion_term is None:
        return command

    motion_config = motion_term.params.get("motion_config")
    if isinstance(motion_config, MotionConfig):
        normalized_motion_config: MotionConfig | dict[str, Any] = dataclasses.replace(
            motion_config,
            motion_file="",
            motion_dir="",
            motion_manifest=manifest,
        )
    elif isinstance(motion_config, dict):
        normalized_motion_config = {
            **motion_config,
            "motion_file": "",
            "motion_dir": "",
            "motion_manifest": manifest,
        }
    else:
        return command

    normalized_params = {**motion_term.params, "motion_config": normalized_motion_config}
    normalized_term = dataclasses.replace(motion_term, params=normalized_params)
    return dataclasses.replace(command, setup_terms={**command.setup_terms, "motion_command": normalized_term})
