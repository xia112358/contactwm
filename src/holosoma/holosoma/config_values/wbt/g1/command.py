"""Whole-body tracking command presets for the canonical G1 pipeline."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

BASELINE_SINGLE_MANIFEST = "runtime/current/manifests/omniretarget_baseline.json"
BASELINE_29_MANIFEST = "runtime/current/manifests/omniretarget_baseline_29.json"
CONTACT_FORCE_MANIFEST = "runtime/current/manifests/newton_contact_force_8part.json"
CLIMB00_MANIFEST = "runtime/current/manifests/wbt_single_climb_00.json"
SPARSE_CLIMB00_MANIFEST = "runtime/current/motions/wbt/climb00_sparse_command_stable_v3/manifest.json"

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="",
    motion_manifest=BASELINE_29_MANIFEST,
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    reset_sampler="completion_recent30_failure_window",
    noise_to_initial_pose=init_pose_config,
)


def make_wbt_command(
    manifest: str,
    *,
    reset_sampler: str = "completion_recent30_failure_window",
    use_start_probe_envs: bool = True,
    start_at_timestep_zero_prob: float = 0.0,
) -> CommandManagerCfg:
    config = replace(
        motion_config,
        motion_manifest=manifest,
        reset_sampler=reset_sampler,
        use_start_probe_envs=use_start_probe_envs,
        start_at_timestep_zero_prob=start_at_timestep_zero_prob,
        freeze_at_timestep_zero_prob=0.0,
    )
    return CommandManagerCfg(
        params={},
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MotionCommand",
                params={"motion_config": config},
            )
        },
        reset_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
        step_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
    )




g1_29dof_wbt_baseline_single_command = make_wbt_command(
    BASELINE_SINGLE_MANIFEST,
    use_start_probe_envs=True,
)
g1_29dof_wbt_baseline_29_command = make_wbt_command(BASELINE_29_MANIFEST)
g1_29dof_wbt_contact_force_command = make_wbt_command(CONTACT_FORCE_MANIFEST)
__all__ = [
    "BASELINE_29_MANIFEST",
    "BASELINE_SINGLE_MANIFEST",
    "CONTACT_FORCE_MANIFEST",
    "CLIMB00_MANIFEST",
    "SPARSE_CLIMB00_MANIFEST",
    "g1_29dof_wbt_baseline_29_command",
    "g1_29dof_wbt_baseline_single_command",
    "g1_29dof_wbt_contact_force_command",
    "make_wbt_command",
]
