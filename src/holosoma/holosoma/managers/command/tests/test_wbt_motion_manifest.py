from dataclasses import dataclass
from typing import get_args, get_type_hints

import pytest

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg
from holosoma.utils.motion_matched_config import normalize_motion_matched_config


@dataclass(frozen=True)
class DummyExperimentConfig:
    command: CommandManagerCfg | None
    terrain: TerrainManagerCfg


def _motion_config(*, motion_manifest: str = "") -> MotionConfig:
    return MotionConfig(
        motion_file="default_motion.npz",
        motion_dir="default_motion_dir",
        body_name_ref=["torso_link"],
        body_names_to_track=["torso_link"],
        motion_manifest=motion_manifest,
        noise_to_initial_pose=NoiseToInitialPoseConfig(),
    )


def _command(*, motion_manifest: str = "") -> CommandManagerCfg:
    return CommandManagerCfg(
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MotionCommand",
                params={"motion_config": _motion_config(motion_manifest=motion_manifest)},
            )
        }
    )


def _terrain(*, motion_matched_manifest: str = "", mesh_type: MeshType = MeshType.PLANE) -> TerrainManagerCfg:
    return TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            mesh_type=mesh_type,
            motion_matched_manifest=motion_matched_manifest,
        )
    )


def _config(
    *,
    command_manifest: str = "",
    terrain_manifest: str = "",
    mesh_type: MeshType = MeshType.PLANE,
) -> DummyExperimentConfig:
    return DummyExperimentConfig(
        command=_command(motion_manifest=command_manifest),
        terrain=_terrain(motion_matched_manifest=terrain_manifest, mesh_type=mesh_type),
    )


def _normalized_motion_config(config: DummyExperimentConfig) -> MotionConfig:
    term = config.command.setup_terms["motion_command"]
    motion_config = term.params["motion_config"]
    assert isinstance(motion_config, MotionConfig)
    return motion_config


def test_motion_reset_sampler_modes_are_current() -> None:
    sampler_type = get_type_hints(MotionConfig)["reset_sampler"]
    assert set(get_args(sampler_type)) == {
        "uniform",
        "adaptive",
        "failure_window",
        "completion_recent30_failure_window",
        "completion_ema_failure_window",
        "hotspot_failure_window",
    }


def test_command_manifest_drives_motion_and_terrain_from_one_manifest() -> None:
    config = _config(command_manifest="runtime/current/manifests/motion_edit_ref_v1.json")

    normalized = normalize_motion_matched_config(config)

    motion_config = _normalized_motion_config(normalized)
    assert motion_config.motion_manifest == "runtime/current/manifests/motion_edit_ref_v1.json"
    assert motion_config.motion_file == ""
    assert motion_config.motion_dir == ""
    assert normalized.terrain.terrain_term.motion_matched_manifest == motion_config.motion_manifest
    assert normalized.terrain.terrain_term.mesh_type == MeshType.LOAD_OBJ


def test_terrain_manifest_drives_motion_and_terrain_from_one_manifest() -> None:
    config = _config(terrain_manifest="runtime/current/manifests/motion_edit_ref_v1.json")

    normalized = normalize_motion_matched_config(config)

    motion_config = _normalized_motion_config(normalized)
    assert motion_config.motion_manifest == "runtime/current/manifests/motion_edit_ref_v1.json"
    assert motion_config.motion_file == ""
    assert motion_config.motion_dir == ""
    assert normalized.terrain.terrain_term.motion_matched_manifest == motion_config.motion_manifest
    assert normalized.terrain.terrain_term.mesh_type == MeshType.LOAD_OBJ


def test_matching_manifests_are_kept_as_one_source() -> None:
    manifest = "runtime/current/manifests/motion_edit_ref_v1.json"
    config = _config(command_manifest=manifest, terrain_manifest=manifest, mesh_type=MeshType.LOAD_OBJ)

    normalized = normalize_motion_matched_config(config)

    motion_config = _normalized_motion_config(normalized)
    assert motion_config.motion_manifest == manifest
    assert normalized.terrain.terrain_term.motion_matched_manifest == manifest
    assert normalized.terrain.terrain_term.mesh_type == MeshType.LOAD_OBJ


def test_conflicting_manifests_fail_before_training() -> None:
    config = _config(
        command_manifest="runtime/current/manifests/motion_edit_ref_v1.json",
        terrain_manifest="runtime/current/manifests/climb00_motion_edit_ref.json",
    )

    with pytest.raises(ValueError, match="must be identical"):
        normalize_motion_matched_config(config)


def test_no_manifest_leaves_config_unchanged() -> None:
    config = _config()

    normalized = normalize_motion_matched_config(config)

    assert normalized is config
