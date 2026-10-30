"""Canonical G1 whole-body tracking experiment presets."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_types.algo import LayerConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)
from holosoma.config_values.wbt.g1 import command


def _terrain_for(manifest: str):
    return replace(
        terrain.terrain_motion_matched,
        terrain_term=replace(
            terrain.terrain_motion_matched.terrain_term,
            motion_matched_manifest=manifest,
        ),
    )


_ppo = replace(
    algo.ppo,
    config=replace(
        algo.ppo.config,
        num_learning_iterations=20000,
        num_learning_epochs=5,
        save_interval=1000,
        entropy_coef=0.005,
        init_noise_std=0.2,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        min_actor_learning_rate=None,
        min_critic_learning_rate=None,
        max_actor_learning_rate=None,
        max_critic_learning_rate=None,
        init_at_random_ep_len=True,
        empirical_normalization=True,
        use_symmetry=False,
        export_onnx=False,
        actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.0),
        critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.0),
    ),
)

_robot = replace(
    robot.g1_29dof,
    control=replace(
        robot.g1_29dof.control,
        action_scale=0.25,
        action_scales_by_effort_limit_over_p_gain=True,
    ),
    asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
    init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
)

_nightly = NightlyConfig(
    iterations=8000,
    metrics={
        "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
        "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
        "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
        "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
        "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
        "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
    },
)

g1_29dof_wbt_baseline_29 = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_baseline_29_hotspot",
        num_envs=4096,
        export_onnx=False,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=_ppo,
    simulator=replace(
        simulator.isaaclab3_newton,
        config=replace(
            simulator.isaaclab3_newton.config,
            sim=replace(simulator.isaaclab3_newton.config.sim, max_episode_length_s=10.0),
            # Canonical self-collision G1 WBT scenes exceed the generic 512
            # constraint budget (climb00 reset peak: 652). Keep the established
            # 1536 allocation used by stable 4096-env climb00 training; it also
            # covers later contact-rich states rather than only startup peaks.
            # Keep this on the shared WBT base so train/eval and every derived
            # preset use the same non-truncating Newton allocation.
            mujoco_warp=replace(
                simulator.isaaclab3_newton.config.mujoco_warp,
                njmax_per_env=1536,
            ),
        ),
    ),
    robot=_robot,
    terrain=_terrain_for(command.BASELINE_29_MANIFEST),
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_baseline_29_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=_nightly,
)

g1_29dof_wbt_baseline_single = replace(
    g1_29dof_wbt_baseline_29,
    training=replace(
        g1_29dof_wbt_baseline_29.training,
        name="g1_29dof_wbt_baseline_single_hotspot",
    ),
    terrain=_terrain_for(command.BASELINE_SINGLE_MANIFEST),
    command=command.g1_29dof_wbt_baseline_single_command,
    simulator=replace(
        g1_29dof_wbt_baseline_29.simulator,
        config=replace(
            g1_29dof_wbt_baseline_29.simulator.config,
            sim=replace(g1_29dof_wbt_baseline_29.simulator.config.sim, max_episode_length_s=22.0),
        ),
    ),
)

g1_29dof_wbt_baseline_29_future_ref = replace(
    g1_29dof_wbt_baseline_29,
    training=replace(
        g1_29dof_wbt_baseline_29.training,
        name="g1_29dof_wbt_baseline_29_future_ref_hotspot",
    ),
    observation=observation.g1_29dof_wbt_future_ref_observation,
)

g1_29dof_wbt_baseline_29_future_ref_no_qdot = replace(
    g1_29dof_wbt_baseline_29_future_ref,
    training=replace(
        g1_29dof_wbt_baseline_29_future_ref.training,
        name="climb00_future_ref_no_qdot_3k_scratch",
    ),
    terrain=_terrain_for(command.CLIMB00_MANIFEST),
    observation=observation.g1_29dof_wbt_future_ref_no_qdot_observation,
    command=command.make_wbt_command(command.CLIMB00_MANIFEST, use_start_probe_envs=True),
    termination=replace(
        termination.g1_29dof_wbt_termination,
        terms={
            **termination.g1_29dof_wbt_termination.terms,
            "bad_tracking": replace(
                termination.g1_29dof_wbt_termination.terms["bad_tracking"],
                params={
                    **termination.g1_29dof_wbt_termination.terms["bad_tracking"].params,
                    "bad_ref_pos_threshold": 0.5,
                    "bad_ref_ori_threshold": 0.8,
                    "bad_motion_body_pos_threshold": 0.25,
                    "bad_motion_body_pos_body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                },
            ),
        },
    ),
)

g1_29dof_wbt_sparse_climb00_3k = replace(
    g1_29dof_wbt_baseline_29_future_ref_no_qdot,
    training=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.training,
        name="climb00_sparse_ref_3k_scratch",
    ),
    algo=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo,
        config=replace(
            g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo.config,
            num_learning_iterations=3000,
            save_interval=250,
            module_dict=replace(
                g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo.config.module_dict,
                actor=replace(
                    g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo.config.module_dict.actor,
                    input_dim=["actor_obs", "actor_proprio_history"],
                ),
            ),
        ),
    ),
    terrain=_terrain_for(command.SPARSE_CLIMB00_MANIFEST),
    observation=observation.g1_29dof_wbt_sparse_climb_observation,
    reward=reward.g1_29dof_wbt_sparse_climb_reward,
    command=command.make_wbt_command(command.SPARSE_CLIMB00_MANIFEST, use_start_probe_envs=True),
    termination=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination,
        terms={
            **g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms,
            "bad_tracking": replace(
                g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms["bad_tracking"],
                params={
                    **g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms[
                        "bad_tracking"
                    ].params,
                    "bad_motion_body_pos_body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                        "left_knee_link",
                        "right_knee_link",
                    ],
                },
            ),
        },
    ),
)

_SPARSE_CLIMB_ROOT_CONTACT_REWARD_TERMS = {
    name: term
    for name, term in g1_29dof_wbt_sparse_climb00_3k.reward.terms.items()
    if name
    in {
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "action_rate_l2",
        "limits_dof_pos",
        "undesired_contacts",
        "sparse_climb_contact_match",
    }
}
_SPARSE_CLIMB_ROOT_CONTACT_REWARD_TERMS["sparse_climb_contact_match"] = replace(
    _SPARSE_CLIMB_ROOT_CONTACT_REWARD_TERMS["sparse_climb_contact_match"],
    weight=10.0,
)

g1_29dof_wbt_sparse_climb00_root_contact_resume_3250 = replace(
    g1_29dof_wbt_sparse_climb00_3k,
    training=replace(
        g1_29dof_wbt_sparse_climb00_3k.training,
        name="climb00_sparse_root_contact_resume_3250",
    ),
    algo=replace(
        g1_29dof_wbt_sparse_climb00_3k.algo,
        config=replace(
            g1_29dof_wbt_sparse_climb00_3k.algo.config,
            num_learning_iterations=3250,
            actor_learning_rate=1.0e-4,
            critic_learning_rate=1.0e-4,
            load_optimizer=False,
            save_interval=50,
        ),
    ),
    reward=replace(
        g1_29dof_wbt_sparse_climb00_3k.reward,
        terms=_SPARSE_CLIMB_ROOT_CONTACT_REWARD_TERMS,
    ),
    termination=replace(
        g1_29dof_wbt_sparse_climb00_3k.termination,
        terms={
            **g1_29dof_wbt_sparse_climb00_3k.termination.terms,
            "bad_tracking": replace(
                g1_29dof_wbt_sparse_climb00_3k.termination.terms["bad_tracking"],
                params={
                    **g1_29dof_wbt_sparse_climb00_3k.termination.terms["bad_tracking"].params,
                    "bad_motion_body_pos_body_names": [],
                },
            ),
        },
    ),
)

_all_tracking_body_names = list(
    termination.g1_29dof_wbt_termination.terms["bad_tracking"].params["body_names_to_track"]
)

g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k = replace(
    g1_29dof_wbt_baseline_29_future_ref_no_qdot,
    training=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.training,
        name="climb00_noqdot_rolloutref_tracking_curriculum_10k",
    ),
    algo=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo,
        config=replace(
            g1_29dof_wbt_baseline_29_future_ref_no_qdot.algo.config,
            num_learning_iterations=10_000,
            actor_learning_rate=1.0e-4,
            critic_learning_rate=1.0e-4,
            load_optimizer=False,
            save_interval=100,
        ),
    ),
    reward=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.reward,
        terms={
            **g1_29dof_wbt_baseline_29_future_ref_no_qdot.reward.terms,
            "motion_global_ref_position_error_exp": replace(
                g1_29dof_wbt_baseline_29_future_ref_no_qdot.reward.terms[
                    "motion_global_ref_position_error_exp"
                ],
                params={"sigma": 0.15},
            ),
            "motion_relative_body_position_error_exp": replace(
                g1_29dof_wbt_baseline_29_future_ref_no_qdot.reward.terms[
                    "motion_relative_body_position_error_exp"
                ],
                params={"sigma": 0.15, "worst_k": 3},
            ),
        },
    ),
    termination=replace(
        g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination,
        terms={
            **g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms,
            "bad_tracking": replace(
                g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms["bad_tracking"],
                params={
                    **g1_29dof_wbt_baseline_29_future_ref_no_qdot.termination.terms[
                        "bad_tracking"
                    ].params,
                    "bad_ref_pos_threshold": 0.50,
                    "bad_ref_ori_threshold": 0.50,
                    "bad_motion_body_pos_threshold": 0.30,
                    "bad_motion_body_pos_body_names": _all_tracking_body_names,
                    "probe_fixed_qualification_boundary": True,
                },
            ),
        },
    ),
    curriculum=curriculum.g1_29dof_wbt_tracking_precision_curriculum_10k,
)

_php_tracking_termination = replace(
    termination.g1_29dof_wbt_termination,
    terms={
        **termination.g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            termination.g1_29dof_wbt_termination.terms["bad_tracking"],
            params={
                **termination.g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    },
)

g1_29dof_wbt_climb00_php_expert = replace(
    g1_29dof_wbt_baseline_29,
    training=replace(
        g1_29dof_wbt_baseline_29.training,
        name="g1_29dof_wbt_climb00_php_expert",
        num_envs=4096,
    ),
    algo=replace(
        g1_29dof_wbt_baseline_29.algo,
        config=replace(g1_29dof_wbt_baseline_29.algo.config, init_noise_std=1.0),
    ),
    robot=replace(
        g1_29dof_wbt_baseline_29.robot,
        control=replace(
            g1_29dof_wbt_baseline_29.robot.control,
            action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
        ),
    ),
    terrain=_terrain_for(command.CLIMB00_MANIFEST),
    observation=observation.g1_29dof_wbt_php_expert_observation,
    command=command.make_wbt_command(command.CLIMB00_MANIFEST),
    termination=_php_tracking_termination,
)

_php_student_algo = replace(
    algo.distill_ppo,
    config=replace(
        algo.distill_ppo.config,
        num_learning_iterations=20000,
        num_steps_per_env=24,
        num_learning_epochs=2,
        num_mini_batches=96,
        entropy_coef=0.001,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        min_actor_learning_rate=None,
        min_critic_learning_rate=None,
        max_actor_learning_rate=3e-4,
        max_critic_learning_rate=3e-4,
        desired_kl=0.01,
        adaptive_schedule_start_iter=1000,
        init_noise_std=0.01,
        use_symmetry=False,
        save_interval=1000,
        export_onnx=False,
        module_dict=replace(
            algo.distill_ppo.config.module_dict,
            actor=replace(
                algo.distill_ppo.config.module_dict.actor,
                type="CNNEncoder",
                input_dim=["actor_obs", "depth_obs"],
                layer_config=LayerConfig(
                    hidden_dims=[2048, 1024, 512, 256, 128],
                    activation="ELU",
                    encoder_activation="ELU",
                    encoder_input_name="depth_obs",
                    module_input_name=("actor_obs",),
                    input_channels=1,
                    input_height=58,
                    input_width=87,
                    hidden_channels=(16, 32, 32),
                    kernel_size=(5, 3, 3),
                    stride=(2, 2, 1),
                    padding=(2, 1, 1),
                    global_average_pool=True,
                ),
            ),
        ),
        distill=replace(
            algo.distill_ppo.config.distill,
            enable_kl=True,
            teacher_checkpoint_path=None,
            teacher_obs_keys=("teacher_actor_obs",),
            teacher_hidden_dims=(512, 256, 128),
            teacher_module_type="MLP",
            teacher_init_noise_std=1.0,
            distill_type="php_dagger_ppo",
            dagger_coef=10.0,
            lambda_dagger_init=1.0,
            lambda_dagger_final=0.1,
            lambda_ppo_init=None,
            lambda_ppo_final=None,
            lambda_kl_init=1.0,
            lambda_kl_final=0.1,
            lambda_kl_anneal_iters=10000,
            lambda_kl_anneal_schedule="php_parkour",
            use_mean_mse_fallback=True,
            dagger_valid_use_bad_tracking=True,
            dagger_bad_ref_pos_threshold=0.5,
            dagger_bad_ref_ori_threshold=0.8,
            dagger_bad_motion_body_pos_threshold=0.25,
            dagger_bad_motion_body_pos_body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
            php_student_termination_anneal_iters=10000,
        ),
    ),
)

g1_29dof_wbt_climb00_php_student = replace(
    g1_29dof_wbt_climb00_php_expert,
    training=replace(
        g1_29dof_wbt_climb00_php_expert.training,
        name="g1_29dof_wbt_climb00_php_student",
        num_envs=4096,
    ),
    algo=_php_student_algo,
    observation=observation.g1_29dof_wbt_php_student_observation,
    termination=termination.g1_29dof_wbt_php_student_termination,
    randomization=randomization.g1_29dof_wbt_php_student_randomization,
    command=command.make_wbt_command(
        command.CLIMB00_MANIFEST,
        reset_sampler="uniform",
        use_start_probe_envs=False,
    ),
    simulator=replace(
        g1_29dof_wbt_climb00_php_expert.simulator,
        config=replace(
            g1_29dof_wbt_climb00_php_expert.simulator.config,
            depth_ray_camera=replace(
                g1_29dof_wbt_climb00_php_expert.simulator.config.depth_ray_camera,
                enabled=True,
            ),
            scene=replace(g1_29dof_wbt_climb00_php_expert.simulator.config.scene, env_spacing=0.0),
        ),
    ),
)

g1_29dof_wbt_contact_force = replace(
    g1_29dof_wbt_baseline_29,
    training=replace(
        g1_29dof_wbt_baseline_29.training,
        name="g1_29dof_wbt_contact_force_8part_hotspot",
    ),
    terrain=_terrain_for(command.CONTACT_FORCE_MANIFEST),
    observation=observation.g1_29dof_wbt_contact_force_observation,
    reward=reward.g1_29dof_wbt_contact_force_reward,
    command=command.g1_29dof_wbt_contact_force_command,
    simulator=replace(
        g1_29dof_wbt_baseline_29.simulator,
        config=replace(
            g1_29dof_wbt_baseline_29.simulator.config,
            sim=replace(g1_29dof_wbt_baseline_29.simulator.config.sim, max_episode_length_s=20.0),
            scene=replace(g1_29dof_wbt_baseline_29.simulator.config.scene, env_spacing=0.0),
        ),
    ),
)

__all__ = [
    "g1_29dof_wbt_baseline_29",
    "g1_29dof_wbt_baseline_29_future_ref",
    "g1_29dof_wbt_baseline_29_future_ref_no_qdot",
    "g1_29dof_wbt_sparse_climb00_3k",
    "g1_29dof_wbt_sparse_climb00_root_contact_resume_3250",
    "g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k",
    "g1_29dof_wbt_baseline_single",
    "g1_29dof_wbt_contact_force",
    "g1_29dof_wbt_climb00_php_expert",
    "g1_29dof_wbt_climb00_php_student",
]
