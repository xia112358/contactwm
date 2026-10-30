from holosoma.config_values.experiment import DEFAULTS
from holosoma.config_values.wbt.g1.command import (
    BASELINE_29_MANIFEST,
    BASELINE_SINGLE_MANIFEST,
    CLIMB00_MANIFEST,
    CONTACT_FORCE_MANIFEST,
    SPARSE_CLIMB00_MANIFEST,
    motion_config,
)
from holosoma.config_values.wbt.g1.climb_reference_goal import CLIMB00_TRIMMED_207_MANIFEST, CLIMB00_SINGLE_FULL_MANIFEST

EXPECTED_MANIFESTS = {
    "wbt_expert_dense": BASELINE_29_MANIFEST,
    "wbt_expert_sparse": BASELINE_29_MANIFEST,
    "wbt_expert_force": BASELINE_29_MANIFEST,
    "wbt_student": BASELINE_29_MANIFEST,
    "g1_29dof_wbt_baseline_single": BASELINE_SINGLE_MANIFEST,
    "g1_29dof_wbt_baseline_29": BASELINE_29_MANIFEST,
    "g1_29dof_wbt_baseline_29_future_ref": BASELINE_29_MANIFEST,
    "g1_29dof_wbt_baseline_29_future_ref_no_qdot": CLIMB00_MANIFEST,
    "g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k": CLIMB00_MANIFEST,
    "g1_29dof_wbt_sparse_climb00_3k": SPARSE_CLIMB00_MANIFEST,
    "g1_29dof_wbt_sparse_climb00_root_contact_resume_3250": SPARSE_CLIMB00_MANIFEST,
    "g1_29dof_wbt_contact_force": CONTACT_FORCE_MANIFEST,
    "g1_29dof_wbt_climb00_php_expert": CLIMB00_MANIFEST,
    "g1_29dof_wbt_climb00_php_student": CLIMB00_MANIFEST,
    "g1_29dof_climb00_reference_goal_24event": CLIMB00_TRIMMED_207_MANIFEST,
    "g1_29dof_climb00_single_full_noref_imitation_2k": CLIMB00_SINGLE_FULL_MANIFEST,
}


def _motion_config(experiment):
    return experiment.command.setup_terms["motion_command"].params["motion_config"]


def _observation_terms(experiment, group: str):
    return experiment.observation.groups[group].terms


def test_only_canonical_pipeline_experiments_are_registered() -> None:
    assert set(DEFAULTS) == set(EXPECTED_MANIFESTS)
    assert len({experiment.training.name for experiment in DEFAULTS.values()}) == len(DEFAULTS)


def test_experiments_bind_one_manifest_to_motion_and_terrain() -> None:
    for name, experiment in DEFAULTS.items():
        expected_manifest = EXPECTED_MANIFESTS[name]
        assert _motion_config(experiment).motion_manifest == expected_manifest
        assert experiment.terrain.terrain_term.motion_matched_manifest == expected_manifest


def test_wbt_reference_observation_has_one_joint_target_source() -> None:
    """The aggregate q/qd command must not be duplicated by explicit terms."""
    for name, experiment in DEFAULTS.items():
        for group in ("actor_obs", "critic_obs"):
            terms = _observation_terms(experiment, group)
            if name.startswith("g1_29dof_wbt_sparse_climb00") or name == "wbt_expert_sparse":
                expected = "00_sparse_climb_reference_window" if group == "actor_obs" else "00_reference_window"
                assert expected in terms
            elif name in ("g1_29dof_climb00_reference_goal_24event", "g1_29dof_climb00_single_full_noref_imitation_2k"):
                assert "00_root_pose_goal" in terms
                assert "motion_command" not in terms
                assert ("reference_joint_state" in terms) == (group == "critic_obs")
            elif "future_ref" in name:
                assert "00_reference_window" in terms
            else:
                if (
                    experiment in (DEFAULTS["g1_29dof_wbt_climb00_php_student"], DEFAULTS["wbt_student"])
                    and group == "actor_obs"
                ):
                    assert "velocity_command" in terms
                    assert "motion_command" not in terms
                else:
                    assert "motion_command" in terms
            assert "motion_ref_joint_pos" not in terms
            assert "motion_ref_joint_vel" not in terms


def test_future_reference_frames_repeat_the_current_reference_contract() -> None:
    experiment = DEFAULTS["g1_29dof_wbt_baseline_29_future_ref"]
    expected_state_names = [
        "pelvis_global_pos",
        "pelvis_global_lin_vel",
        "robot_body_pos_b",
        "robot_body_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "terrain_height_scan",
        "dof_pos",
        "dof_vel",
        "actions",
    ]
    for group, include_current, add_noise in (
        ("actor_obs", False, True),
        ("critic_obs", True, False),
    ):
        terms = _observation_terms(experiment, group)
        assert list(terms) == ["00_reference_window", *expected_state_names]
        assert terms["00_reference_window"].params == {
            "include_current": include_current,
            "add_noise": add_noise,
        }


def test_future_reference_no_qdot_omits_joint_velocity_only() -> None:
    for preset in (
        "g1_29dof_wbt_baseline_29_future_ref_no_qdot",
        "g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k",
    ):
        experiment = DEFAULTS[preset]
        for group, include_current, add_noise in (
            ("actor_obs", False, True),
            ("critic_obs", True, False),
        ):
            terms = _observation_terms(experiment, group)
            assert terms["00_reference_window"].params == {
                "include_current": include_current,
                "add_noise": add_noise,
                "include_joint_velocity": False,
            }


def test_sparse_climb_actor_and_privileged_critic_contract() -> None:
    experiment = DEFAULTS["g1_29dof_wbt_sparse_climb00_3k"]
    actor = _observation_terms(experiment, "actor_obs")
    actor_history = _observation_terms(experiment, "actor_proprio_history")
    critic = _observation_terms(experiment, "critic_obs")
    assert actor["00_sparse_climb_reference_window"].params == {"add_noise": True}
    assert "00_reference_window" not in actor
    assert "motion_command" not in actor
    assert critic["00_reference_window"].params == {
        "include_current": True,
        "add_noise": False,
        "include_joint_velocity": False,
    }
    assert experiment.observation.groups["actor_proprio_history"].history_length == 5
    assert set(actor_history) == {
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "actions",
    }
    assert experiment.algo.config.module_dict.actor.input_dim == ["actor_obs", "actor_proprio_history"]
    reward_terms = experiment.reward.terms
    assert reward_terms["sparse_climb_contact_match"].weight == 1.0
    expected_endpoints = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_knee_link",
        "right_knee_link",
    }
    position_terms = [reward_terms[f"sparse_climb_position_{part}"] for part in ("lf", "rf", "lh", "rh", "lk", "rk")]
    orientation_terms = [
        reward_terms[f"sparse_climb_orientation_{part}"] for part in ("lf", "rf", "lh", "rh", "lk", "rk")
    ]
    assert {term.params["body_names"][0] for term in position_terms} == expected_endpoints
    assert {term.params["body_names"][0] for term in orientation_terms} == expected_endpoints
    assert all(term.weight == 0.25 for term in (*position_terms, *orientation_terms))
    undesired_pattern = reward_terms["undesired_contacts"].params["undesired_contacts_body_names"]
    assert "left_knee_link" in undesired_pattern
    assert "right_knee_link" in undesired_pattern
    assert set(experiment.termination.terms["bad_tracking"].params["bad_motion_body_pos_body_names"]) == expected_endpoints


def test_sparse_root_contact_resume_reward_contract() -> None:
    experiment = DEFAULTS["g1_29dof_wbt_sparse_climb00_root_contact_resume_3250"]
    assert set(experiment.reward.terms) == {
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "action_rate_l2",
        "limits_dof_pos",
        "undesired_contacts",
        "sparse_climb_contact_match",
    }
    assert experiment.reward.terms["sparse_climb_contact_match"].weight == 10.0
    assert experiment.termination.terms["bad_tracking"].params["bad_motion_body_pos_body_names"] == []
    assert experiment.algo.config.load_optimizer is False




def test_experiments_use_canonical_robot_newton_and_compact_outputs() -> None:
    curriculum = DEFAULTS["g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k"]
    sparse = DEFAULTS["g1_29dof_wbt_sparse_climb00_3k"]
    sparse_resume = DEFAULTS["g1_29dof_wbt_sparse_climb00_root_contact_resume_3250"]
    reference_goal = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    for experiment in DEFAULTS.values():
        assert experiment.robot.asset.urdf_file == "g1/g1_29dof_spherehand.urdf"
        assert experiment.simulator.config.name == "isaaclab3_newton"
        assert experiment.simulator.config.mujoco_warp.njmax_per_env == 1536
        assert experiment.training.num_envs == (2048 if experiment is DEFAULTS["wbt_student"] else 4096)
        assert experiment.algo.config.export_onnx is False
        expected_iterations = (
            10000
            if experiment in (curriculum, reference_goal)
            else 3250
            if experiment is sparse_resume
            else 3000
            if experiment is sparse
            else 2000
            if experiment is DEFAULTS["g1_29dof_climb00_single_full_noref_imitation_2k"]
            else 20000
        )
        assert experiment.algo.config.num_learning_iterations == expected_iterations
        if experiment in (DEFAULTS["g1_29dof_wbt_climb00_php_student"], DEFAULTS["wbt_student"]):
            assert experiment.algo.config.num_learning_epochs == 2
            assert experiment.algo.config.actor_learning_rate == 3e-4
            assert experiment.algo.config.critic_learning_rate == 3e-4
        elif experiment in (curriculum, reference_goal, sparse_resume,
                            DEFAULTS["g1_29dof_climb00_single_full_noref_imitation_2k"]):
            assert experiment.algo.config.num_learning_epochs == 5
            assert experiment.algo.config.actor_learning_rate == 1e-4
            assert experiment.algo.config.critic_learning_rate == 1e-4
        else:
            assert experiment.algo.config.num_learning_epochs == 5
            assert experiment.algo.config.actor_learning_rate == 1e-3
            assert experiment.algo.config.critic_learning_rate == 1e-3
        assert experiment.algo.config.min_actor_learning_rate is None
        assert experiment.algo.config.min_critic_learning_rate is None
        if experiment in (DEFAULTS["g1_29dof_wbt_climb00_php_student"], DEFAULTS["wbt_student"]):
            assert experiment.algo.config.max_actor_learning_rate == 3e-4
            assert experiment.algo.config.max_critic_learning_rate == 3e-4
        else:
            assert experiment.algo.config.max_actor_learning_rate is None
            assert experiment.algo.config.max_critic_learning_rate is None
        expected_save_interval = (
            200
            if experiment is DEFAULTS["wbt_student"]
            else 100
            if experiment is curriculum
            else 50
            if experiment is sparse_resume
            else 250
            if experiment in (sparse, reference_goal, DEFAULTS["g1_29dof_climb00_single_full_noref_imitation_2k"])
            else 1000
        )
        assert experiment.algo.config.save_interval == expected_save_interval


def test_tracking_precision_curriculum_contract() -> None:
    experiment = DEFAULTS["g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k"]
    bad_tracking = experiment.termination.terms["bad_tracking"].params
    reward_terms = experiment.reward.terms
    curriculum = experiment.curriculum.step_terms["tracking_precision"].params

    assert experiment.algo.config.load_optimizer is False
    assert bad_tracking["bad_ref_pos_threshold"] == 0.50
    assert bad_tracking["bad_ref_ori_threshold"] == 0.50
    assert bad_tracking["bad_motion_body_pos_threshold"] == 0.30
    assert bad_tracking["bad_motion_body_pos_body_names"] == bad_tracking["body_names_to_track"]
    assert bad_tracking["probe_fixed_qualification_boundary"] is True
    assert bad_tracking.get("exclude_probe_envs", False) is False
    assert reward_terms["motion_global_ref_position_error_exp"].params == {"sigma": 0.15}
    assert reward_terms["motion_relative_body_position_error_exp"].params == {
        "sigma": 0.15,
        "worst_k": 3,
    }
    assert curriculum == {
        "probe_quantile": 0.90,
        "probe_history_size": 100,
        "min_probe_episodes": 10,
        "base_body_threshold": 0.30,
        "body_threshold_bounds": (0.05, 0.30),
        "base_root_pos_threshold": 0.50,
        "root_pos_threshold_bounds": (0.10, 0.50),
        "base_root_ori_threshold": 0.50,
        "root_ori_threshold_bounds": (0.15, 0.50),
        "base_position_sigma": 0.15,
        "position_sigma_bounds": (0.05, 0.15),
        "base_reset_noise_scale": 1.0,
        "reset_noise_scale_bounds": (1.0 / 6.0, 1.0),
    }


def test_experiment_sampler_roles_are_explicit() -> None:
    for experiment in DEFAULTS.values():
        motion = _motion_config(experiment)
        expected_sampler = (
            "uniform"
            if experiment in (
                DEFAULTS["wbt_student"],
                DEFAULTS["g1_29dof_wbt_climb00_php_student"],
                DEFAULTS["g1_29dof_climb00_reference_goal_24event"],
                DEFAULTS["g1_29dof_climb00_single_full_noref_imitation_2k"],
            )
            else "completion_recent30_failure_window"
        )
        assert motion.reset_sampler == expected_sampler
        assert motion.start_at_timestep_zero_prob == 0.0
    single_motion = _motion_config(DEFAULTS["g1_29dof_wbt_baseline_single"])
    assert single_motion.use_start_probe_envs is True
    assert single_motion.probe_env_per_motion == 3


def test_php_student_matches_paper_training_contract_at_4096_envs() -> None:
    expert = DEFAULTS["g1_29dof_wbt_climb00_php_expert"]
    student = DEFAULTS["g1_29dof_wbt_climb00_php_student"]
    actor = student.algo.config.module_dict.actor
    distill = student.algo.config.distill

    assert expert.robot.control.action_scale == 1.0
    assert expert.algo.config.init_noise_std == 1.0
    assert student.training.num_envs == 4096
    assert student.algo.config.init_noise_std == 0.01
    assert student.algo.config.num_steps_per_env == 24
    assert student.algo.config.num_mini_batches == 96
    assert actor.input_dim == ["actor_obs", "depth_obs"]
    assert actor.layer_config.input_height == 58
    assert actor.layer_config.input_width == 87
    assert actor.layer_config.global_average_pool is True
    assert actor.layer_config.hidden_dims == [2048, 1024, 512, 256, 128]
    assert distill.dagger_coef == 10.0
    assert distill.lambda_dagger_init == 1.0
    assert distill.lambda_dagger_final == 0.1
    assert distill.lambda_kl_anneal_iters == 10000
    assert distill.teacher_checkpoint_path is None


def test_experiment_episode_horizons_match_training_stage() -> None:
    assert DEFAULTS["g1_29dof_wbt_baseline_single"].simulator.config.sim.max_episode_length_s == 22.0
    assert DEFAULTS["g1_29dof_wbt_baseline_29"].simulator.config.sim.max_episode_length_s == 10.0
    assert DEFAULTS["g1_29dof_wbt_contact_force"].simulator.config.sim.max_episode_length_s == 20.0
