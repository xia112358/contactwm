from __future__ import annotations

from holosoma.config_values.experiment import DEFAULTS


def test_climb_reference_goal_is_independent_registered_experiment() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    command = experiment.command.setup_terms["motion_command"]
    assert command.func.endswith(":ClimbReferenceGoalCommand")
    assert command.params["expected_event_count"] == 24
    assert command.params["goal_stability_tail_steps"] == 3
    assert command.params["motion_config"].motion_manifest.endswith(
        "training_manifest_207_trimmed.json"
    )
    assert experiment.algo.config.module_dict.actor.input_dim == ["actor_obs"]


def test_actor_has_goal_but_no_reference_phase_or_task_mode() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    actor_terms = experiment.observation.groups["actor_obs"].terms
    actor_func_names = {term.func for term in actor_terms.values()}
    assert any(name.endswith(":climb_root_pose_goal") for name in actor_func_names)
    assert not any("motion_ref" in name or "reference_window" in name for name in actor_func_names)
    assert not any("task_indicator" in name or "phase" in name for name in actor_func_names)
    assert not any("terrain_height_scan" in name or "robot_body" in name for name in actor_func_names)
    critic_funcs = {term.func for term in experiment.observation.groups["critic_obs"].terms.values()}
    assert any(name.endswith(":climb_task_indicator") for name in critic_funcs)
    assert any(name.endswith(":climb_assistive_wrench") for name in critic_funcs)


def test_climb_reference_goal_uses_paper_training_hyperparameters() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    algo = experiment.algo.config
    assert algo.num_steps_per_env == 24
    assert algo.num_learning_epochs == 5
    assert algo.num_mini_batches == 4
    assert algo.actor_learning_rate == 1.0e-4
    assert algo.critic_learning_rate == 1.0e-4
    assert algo.entropy_coef == 0.001
    assert algo.value_loss_coef == 0.5
    assert algo.module_dict.actor.layer_config.hidden_dims == [1024, 512, 256]
    assert algo.module_dict.critic.layer_config.hidden_dims == [1024, 512, 256]


def test_climb_reference_goal_uses_task_success_curriculum_and_goal_reward() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    params = experiment.curriculum.step_terms["reference_goal_curriculum"].params
    assert "anneal_iterations" not in params
    assert params["difficulty_step_down"] > params["difficulty_step_up"]
    assert {
        "generalization_goal_position",
        "generalization_goal_orientation",
        "generalization_reach",
    } <= experiment.reward.terms.keys()
    assert "generalization_sparse_goal" not in experiment.reward.terms
    assert experiment.reward.terms["generalization_reach"].func.endswith(":ClimbSparseGoalSuccess")
    assert "relaxed_tracking_threshold" not in params
    assert "severe_state_invalid" in experiment.termination.terms


def test_climb_contacts_do_not_trigger_locomotion_contact_termination() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_reference_goal_24event"]
    assert "undesired_contacts" not in experiment.reward.terms
    assert "undesired_contacts" not in experiment.termination.terms


def test_single_full_noref_experiment_is_pure_imitation_without_event_splits() -> None:
    experiment = DEFAULTS["g1_29dof_climb00_single_full_noref_imitation_2k"]
    command = experiment.command.setup_terms["motion_command"]
    assert command.params["goal_mode"] == "motion_end"
    assert command.params["expected_event_count"] == 1
    assert command.params["imitation_probability_start"] == 1.0
    assert command.params["imitation_probability_final"] == 1.0
    assert command.params["motion_config"].motion_manifest.endswith("wbt_single_climb_00.json")
    assert experiment.algo.config.num_learning_iterations == 2_000
    assert "imitation_tracking_curriculum" in experiment.curriculum.step_terms
    assert not any(name.startswith("generalization_") for name in experiment.reward.terms)

    actor_funcs = {
        term.func for term in experiment.observation.groups["actor_obs"].terms.values()
    }
    assert any(name.endswith(":climb_root_pose_goal") for name in actor_funcs)
    assert not any("motion_ref" in name or "reference_window" in name for name in actor_funcs)
