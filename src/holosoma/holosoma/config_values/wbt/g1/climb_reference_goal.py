"""Independent climb00 experiment based on arXiv:2602.20375v1."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.config_types.algo import LayerConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values import randomization, terrain
from holosoma.config_values.wbt.g1 import command, observation
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_baseline_29

CLIMB00_TRIMMED_207_MANIFEST = "runtime/current/motions/wbt/climb00_continuous_coverage_trimmed/training_manifest_207_trimmed.json"
CLIMB00_SINGLE_FULL_MANIFEST = "runtime/current/manifests/wbt_single_climb_00.json"
_COMMAND_PATH = "holosoma.managers.command.terms.climb_reference_goal:ClimbReferenceGoalCommand"

_motion_config = replace(
    command.motion_config,
    motion_manifest=CLIMB00_TRIMMED_207_MANIFEST,
    motion_file="",
    motion_dir="",
    reset_sampler="uniform",
    canonicalize_motion_order_on_load=True,
    use_start_probe_envs=False,
    use_group_probe_envs=False,
    start_at_timestep_zero_prob=0.0,
    freeze_at_timestep_zero_prob=0.0,
)

_command_params = {
    "motion_config": _motion_config,
    "expected_event_count": 24,
    "imitation_probability_start": 1.0,
    "imitation_probability_final": 0.5,
    "initial_difficulty": 0.0,
    "generalization_root_xy": 0.4,
    "generalization_root_roll_pitch": 0.15,
    "generalization_root_yaw": 0.8,
    "generalization_goal_xy": 0.20,
    "generalization_endpoint_jitter": 0.08,
    "goal_stability_tail_steps": 3,
    "assistive_force_max": 350.0,
    "assistive_beta_max": 0.75,
}

g1_29dof_climb00_reference_goal_command = CommandManagerCfg(
    setup_terms={"motion_command": CommandTermCfg(func=_COMMAND_PATH, params=_command_params)},
    reset_terms={"motion_command": CommandTermCfg(func=_COMMAND_PATH)},
    step_terms={"motion_command": CommandTermCfg(func=_COMMAND_PATH)},
)

_goal_term = {
    "00_root_pose_goal": ObsTermCfg(
        func="holosoma.managers.observation.terms.climb_reference_goal:climb_root_pose_goal"
    )
}
_actor_geometry_terms = {
    name: observation.actor_obs_shared.terms[name]
    for name in ("robot_body_pos_b", "robot_body_ori_b", "terrain_height_scan")
}
_actor_proprio_terms = {
    "projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", noise=0.015
    ),
    **{
        name: replace(observation.actor_obs_shared.terms[name], noise=noise)
        for name, noise in (
            ("base_ang_vel", 0.10),
            ("dof_pos", 0.005),
            ("dof_vel", 0.25),
            ("actions", 0.0),
        )
    },
}
_actor_terms = {**_goal_term, **_actor_proprio_terms}
_critic_terms = {
    **_actor_terms,
    **_actor_geometry_terms,
    "base_lin_vel": observation.actor_obs_shared.terms["base_lin_vel"],
    "task_indicator": ObsTermCfg(
        func="holosoma.managers.observation.terms.climb_reference_goal:climb_task_indicator"
    ),
    "reference_joint_state": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:motion_command"),
    "reference_torso_position": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b"),
    "reference_torso_orientation": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b"),
    "actual_contact_force": ObsTermCfg(
        func="holosoma.managers.observation.terms.climb_reference_goal:climb_actual_contact_force"
    ),
    "assistive_wrench": ObsTermCfg(
        func="holosoma.managers.observation.terms.climb_reference_goal:climb_assistive_wrench"
    ),
}

g1_29dof_climb00_reference_goal_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=_actor_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=_critic_terms,
        ),
    }
)

g1_29dof_climb00_reference_goal_reward = RewardManagerCfg(
    terms={
        "imitation_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:climb_imitation_tracking",
            weight=6.0,
        ),
        "imitation_base_height_error": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:climb_imitation_base_height_error",
            weight=-10.0,
        ),
        "generalization_goal_position": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:climb_generalization_position_error",
            weight=-5.0,
        ),
        "generalization_goal_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:climb_generalization_orientation_error",
            weight=-1.0,
        ),
        "generalization_reach": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:ClimbSparseGoalSuccess",
            params={
                "torso_radius": 0.20,
                "orientation_threshold": 0.50,
                "endpoint_radius": 0.12,
                "root_speed_threshold": 0.35,
                "contact_threshold": 10.0,
                "stable_frames_required": 3,
            },
            weight=10.0,
        ),
        "survival": RewardTermCfg(
            func="holosoma.managers.reward.terms.climb_reference_goal:climb_survival",
            weight=30.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "applied_torque": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_torque",
            weight=-5.0e-4,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-5.0,
        ),
    }
)

g1_29dof_climb00_reference_goal_termination = TerminationManagerCfg(
    terms={
        "contact_goal_deadline": TerminationTermCfg(
            func="holosoma.managers.termination.terms.climb_reference_goal:climb_contact_goal_deadline",
            is_timeout=True,
        ),
        "base_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:base_height_below_threshold",
            params={"min_height": 0.25},
        ),
        "severe_state_invalid": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:severe_state_invalid",
            params={
                "min_height": 0.05,
                "max_height": 3.0,
                "max_ref_pos_error": 5.0,
                "max_root_lin_vel": 20.0,
                "max_root_ang_vel": 40.0,
                "max_body_lin_vel": 40.0,
                "max_joint_abs_vel": 80.0,
            },
        ),
    }
)

_CURRICULUM_PATH = (
    "holosoma.managers.curriculum.terms.climb_reference_goal:ClimbReferenceGoalCurriculum"
)
_curriculum_term = CurriculumTermCfg(
    func=_CURRICULUM_PATH,
    params={
        "update_interval_steps": 24,
        "performance_ema_alpha": 0.10,
        "promote_threshold": 0.80,
        "demote_threshold": 0.60,
        "difficulty_step_up": 0.001,
        "difficulty_step_down": 0.01,
        "min_completed_episodes": 128,
    },
)
g1_29dof_climb00_reference_goal_curriculum = CurriculumManagerCfg(
    params={"num_compute_average_epl": 1000},
    setup_terms={"reference_goal_curriculum": _curriculum_term},
    step_terms={"reference_goal_curriculum": _curriculum_term},
)

_single_full_motion_config = replace(
    _motion_config,
    motion_manifest=CLIMB00_SINGLE_FULL_MANIFEST,
)
_single_full_command_params = {
    **_command_params,
    "motion_config": _single_full_motion_config,
    "goal_mode": "motion_end",
    "expected_event_count": 1,
    "imitation_probability_start": 1.0,
    "imitation_probability_final": 1.0,
}
g1_29dof_climb00_single_full_noref_command = CommandManagerCfg(
    setup_terms={
        "motion_command": CommandTermCfg(func=_COMMAND_PATH, params=_single_full_command_params)
    },
    reset_terms={"motion_command": CommandTermCfg(func=_COMMAND_PATH)},
    step_terms={"motion_command": CommandTermCfg(func=_COMMAND_PATH)},
)

_single_full_reward_terms = {
    name: term
    for name, term in g1_29dof_climb00_reference_goal_reward.terms.items()
    if not name.startswith("generalization_")
}
_single_full_reward_terms["imitation_tracking"] = replace(
    _single_full_reward_terms["imitation_tracking"],
    params={"joint_position_sigma": 0.50, "projected_gravity_sigma": 0.40},
    weight=8.0,
)
g1_29dof_climb00_single_full_noref_reward = RewardManagerCfg(
    terms=_single_full_reward_terms
)

_IMITATION_CURRICULUM_PATH = (
    "holosoma.managers.curriculum.terms.climb_reference_goal:"
    "ClimbImitationTrackingCurriculum"
)
_imitation_curriculum_term = CurriculumTermCfg(
    func=_IMITATION_CURRICULUM_PATH,
    params={
        "update_interval_steps": 24,
        "performance_ema_alpha": 0.05,
        "promote_threshold": 0.72,
        "demote_threshold": 0.55,
        "difficulty_step_up": 0.002,
        "difficulty_step_down": 0.005,
        "warmup_updates": 20,
    },
)
g1_29dof_climb00_single_full_noref_curriculum = CurriculumManagerCfg(
    params={"num_compute_average_epl": 1000},
    setup_terms={"imitation_tracking_curriculum": _imitation_curriculum_term},
    step_terms={"imitation_tracking_curriculum": _imitation_curriculum_term},
)

g1_29dof_climb00_reference_goal_24event = replace(
    g1_29dof_wbt_baseline_29,
    training=replace(
        g1_29dof_wbt_baseline_29.training,
        name="climb00_reference_goal_24event_10k",
        num_envs=4096,
    ),
    algo=replace(
        g1_29dof_wbt_baseline_29.algo,
        config=replace(
            g1_29dof_wbt_baseline_29.algo.config,
            num_learning_iterations=10_000,
            num_steps_per_env=24,
            num_learning_epochs=5,
            num_mini_batches=4,
            actor_learning_rate=1.0e-4,
            critic_learning_rate=1.0e-4,
            entropy_coef=0.001,
            value_loss_coef=0.5,
            desired_kl=0.01,
            save_interval=250,
            module_dict=replace(
                g1_29dof_wbt_baseline_29.algo.config.module_dict,
                actor=replace(
                    g1_29dof_wbt_baseline_29.algo.config.module_dict.actor,
                    input_dim=["actor_obs"],
                    layer_config=LayerConfig(hidden_dims=[1024, 512, 256], activation="ELU"),
                ),
                critic=replace(
                    g1_29dof_wbt_baseline_29.algo.config.module_dict.critic,
                    layer_config=LayerConfig(hidden_dims=[1024, 512, 256], activation="ELU"),
                ),
            ),
        ),
    ),
    terrain=replace(
        terrain.terrain_motion_matched,
        terrain_term=replace(
            terrain.terrain_motion_matched.terrain_term,
            motion_matched_manifest=CLIMB00_TRIMMED_207_MANIFEST,
        ),
    ),
    observation=g1_29dof_climb00_reference_goal_observation,
    reward=g1_29dof_climb00_reference_goal_reward,
    termination=g1_29dof_climb00_reference_goal_termination,
    command=g1_29dof_climb00_reference_goal_command,
    curriculum=g1_29dof_climb00_reference_goal_curriculum,
    randomization=randomization.g1_29dof_wbt_randomization,
    simulator=replace(
        g1_29dof_wbt_baseline_29.simulator,
        config=replace(
            g1_29dof_wbt_baseline_29.simulator.config,
            sim=replace(g1_29dof_wbt_baseline_29.simulator.config.sim, max_episode_length_s=17.0),
            scene=replace(g1_29dof_wbt_baseline_29.simulator.config.scene, env_spacing=0.0),
        ),
    ),
)

g1_29dof_climb00_single_full_noref_imitation_2k = replace(
    g1_29dof_climb00_reference_goal_24event,
    training=replace(
        g1_29dof_climb00_reference_goal_24event.training,
        name="climb00_single_full_noref_imitation_2k",
        num_envs=4096,
    ),
    algo=replace(
        g1_29dof_climb00_reference_goal_24event.algo,
        config=replace(
            g1_29dof_climb00_reference_goal_24event.algo.config,
            num_learning_iterations=2_000,
        ),
    ),
    terrain=replace(
        terrain.terrain_motion_matched,
        terrain_term=replace(
            terrain.terrain_motion_matched.terrain_term,
            motion_matched_manifest=CLIMB00_SINGLE_FULL_MANIFEST,
        ),
    ),
    reward=g1_29dof_climb00_single_full_noref_reward,
    command=g1_29dof_climb00_single_full_noref_command,
    curriculum=g1_29dof_climb00_single_full_noref_curriculum,
    simulator=replace(
        g1_29dof_climb00_reference_goal_24event.simulator,
        config=replace(
            g1_29dof_climb00_reference_goal_24event.simulator.config,
            sim=replace(
                g1_29dof_climb00_reference_goal_24event.simulator.config.sim,
                max_episode_length_s=21.0,
            ),
        ),
    ),
)


__all__ = [
    "CLIMB00_SINGLE_FULL_MANIFEST",
    "CLIMB00_TRIMMED_207_MANIFEST",
    "g1_29dof_climb00_reference_goal_24event",
    "g1_29dof_climb00_single_full_noref_imitation_2k",
]
