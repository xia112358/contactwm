from types import SimpleNamespace

import pytest
import torch
import holosoma.managers.reward.terms.climb_reference_goal as climb_rewards
from holosoma.managers.reward.terms.climb_reference_goal import (
    _sparse_climb_contact_force_magnitude,
)
from holosoma.managers.reward.terms.wbt import (
    DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
    SPARSE_CLIMB_CONTACT_BODY_NAMES,
    _part_contact_force_magnitude,
    _sparse_contact_balanced_score,
    motion_sparse_contact_match_exp,
)


def test_sparse_climb_contact_groups_use_canonical_sphere_hands() -> None:
    assert SPARSE_CLIMB_CONTACT_BODY_NAMES[2] == (
        "left_sphere_hand_link",
        "left_sphere_hand_tip_link",
    )
    assert SPARSE_CLIMB_CONTACT_BODY_NAMES[3] == (
        "right_sphere_hand_link",
        "right_sphere_hand_tip_link",
    )


def test_legacy_sparse_contact_reward_remains_importable() -> None:
    assert callable(motion_sparse_contact_match_exp)


def test_balanced_contact_score_rejects_all_false_shortcut() -> None:
    expected = torch.tensor([[True, True, False, True, False, False]])
    actual_none = torch.zeros_like(expected)
    score, recall, specificity = _sparse_contact_balanced_score(actual_none, expected)
    torch.testing.assert_close(score, torch.tensor([0.0]))
    torch.testing.assert_close(recall, torch.tensor([0.0]))
    torch.testing.assert_close(specificity, torch.tensor([1.0]))


def test_balanced_contact_score_is_one_only_for_exact_match() -> None:
    expected = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, False, False, False, False, False],
        ]
    )
    score, recall, specificity = _sparse_contact_balanced_score(expected, expected)
    torch.testing.assert_close(score, torch.ones(2))
    torch.testing.assert_close(recall, torch.ones(2))
    torch.testing.assert_close(specificity, torch.ones(2))


def test_part_contact_force_uses_full_link_view_and_sums_per_history_sample() -> None:
    link_names = ["heel_a", "heel_b", "hand"]
    link_history = torch.zeros(1, 2, 3, 3)
    link_history[0, 0, 0, 2] = 6.0
    link_history[0, 0, 1, 2] = 7.0
    link_history[0, 1, 2, 2] = 12.0
    simulator = SimpleNamespace(
        contact_link_body_names=link_names,
        contact_link_forces_history=link_history,
        # The articulation view deliberately omits the hand and one heel.
        body_names=["heel_a"],
        contact_forces_history=torch.zeros(1, 2, 1, 3),
    )
    env = SimpleNamespace(simulator=simulator)

    magnitude = _part_contact_force_magnitude(
        env,
        (("heel_a", "heel_b"), ("hand",)),
        force_reduce="sum",
        history_reduce="max",
    )

    torch.testing.assert_close(magnitude, torch.tensor([[13.0, 12.0]]))


def test_part_contact_force_rejects_partial_collision_group() -> None:
    simulator = SimpleNamespace(
        contact_link_body_names=["heel_a"],
        contact_link_forces_history=torch.zeros(1, 2, 1, 3),
        body_names=["heel_a"],
        contact_forces_history=torch.zeros(1, 2, 1, 3),
    )
    env = SimpleNamespace(simulator=simulator)

    with pytest.raises(RuntimeError, match="heel_b"):
        _part_contact_force_magnitude(
            env,
            (("heel_a", "heel_b"),),
            force_reduce="sum",
            history_reduce="max",
        )


def test_contact_force_indexes_and_values_are_cached_within_one_step() -> None:
    link_history = torch.zeros(1, 2, 2, 3)
    simulator = SimpleNamespace(
        contact_link_body_names=["a", "b"],
        contact_link_forces_history=link_history,
    )
    env = SimpleNamespace(simulator=simulator)

    first = _part_contact_force_magnitude(env, (("a",), ("b",)), history_reduce="max")
    link_history[0, 0, 0, 2] = 9.0
    cached = _part_contact_force_magnitude(env, (("a",), ("b",)), history_reduce="max")
    torch.testing.assert_close(cached, first)

    env._holosoma_step_cache.clear()
    refreshed = _part_contact_force_magnitude(env, (("a",), ("b",)), history_reduce="max")
    torch.testing.assert_close(refreshed, torch.tensor([[9.0, 0.0]]))
    assert len(env._holosoma_contact_group_index_cache) == 1


def test_sparse_six_part_force_matches_direct_grouping() -> None:
    canonical_names = [name for group in DEFAULT_CONTACT_FORCE_PART_BODY_NAMES for name in group]
    history = torch.randn(3, 4, len(canonical_names), 3)
    simulator = SimpleNamespace(
        contact_link_body_names=canonical_names,
        contact_link_forces_history=history,
    )
    direct_env = SimpleNamespace(simulator=simulator)
    cached_env = SimpleNamespace(simulator=simulator)

    direct = _part_contact_force_magnitude(
        direct_env,
        SPARSE_CLIMB_CONTACT_BODY_NAMES,
        force_reduce="sum",
        history_reduce="max",
    )
    derived = _sparse_climb_contact_force_magnitude(cached_env)
    torch.testing.assert_close(derived, direct)


def test_generalization_goal_terms_are_dense_and_mode_masked(monkeypatch) -> None:
    command = SimpleNamespace(
        is_imitation=torch.tensor([True, False]),
        robot_ref_pos_w=torch.tensor([[0.0, 0.0, 0.8], [3.0, 4.0, 0.8]]),
        goal_ref_pos_w=torch.zeros(2, 3),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    env = SimpleNamespace(log_dict={})
    monkeypatch.setattr(climb_rewards, "_command", lambda _: command)

    position = climb_rewards.climb_generalization_position_error(env)
    orientation = climb_rewards.climb_generalization_orientation_error(env)
    reached = climb_rewards.climb_generalization_reach(env)

    torch.testing.assert_close(position, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(orientation, torch.zeros(2))
    torch.testing.assert_close(reached, torch.zeros(2))


def test_sparse_goal_success_reports_all_task_completions_but_rewards_generalization(monkeypatch) -> None:
    expected = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, False, False, True, False, False],
        ]
    )
    command = SimpleNamespace(
        is_imitation=torch.tensor([True, False]),
        robot_ref_pos_w=torch.zeros(2, 3),
        goal_ref_pos_w=torch.zeros(2, 3),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        robot_body_pos_w=torch.zeros(2, 6, 3),
        goal_endpoint_pos_w=torch.zeros(2, 6, 3),
        goal_endpoint_track_indexes=torch.arange(6),
        goal_contact_mask=expected,
        robot_root_lin_vel_w=torch.zeros(2, 3),
    )
    observed = []
    curriculum = SimpleNamespace(observe_task_success=lambda value: observed.append(value.clone()))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        log_dict={},
        curriculum_manager=SimpleNamespace(get_term=lambda _: curriculum),
    )
    force = expected.to(torch.float32) * 20.0
    monkeypatch.setattr(climb_rewards, "_command", lambda _: command)
    monkeypatch.setattr(climb_rewards, "_sparse_climb_contact_force_magnitude", lambda _: force)

    term = climb_rewards.ClimbSparseGoalSuccess(SimpleNamespace(), env)
    for _ in range(2):
        torch.testing.assert_close(term(env), torch.zeros(2))
    reward = term(env)

    torch.testing.assert_close(reward, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(observed[-1], torch.tensor([True, True]))
    torch.testing.assert_close(env.log_dict["climb_reference_goal/task_success"], torch.ones(2))
