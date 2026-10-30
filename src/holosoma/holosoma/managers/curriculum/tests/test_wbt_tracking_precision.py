from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.curriculum.terms.wbt import (
    TrackingPrecisionCurriculum,
    derive_tracking_settings,
    quantile_or_none,
)
from holosoma.managers.reward.terms import wbt as wbt_reward
from holosoma.managers.reward.terms.wbt import motion_relative_body_position_error_exp


def test_probe_quantile_uses_complete_trajectory_maxima() -> None:
    episode_maxima = torch.tensor([0.10, 0.20, 0.30, 0.40, float("nan")])
    assert quantile_or_none(episode_maxima, 0.80) == pytest.approx(0.34)
    assert quantile_or_none(torch.tensor([]), 0.80) is None


def test_probe_p80_derives_every_tracking_setting_without_stages() -> None:
    settings = derive_tracking_settings(
        0.18,
        base_body_threshold=0.30,
        body_threshold_bounds=(0.05, 0.30),
        base_root_pos_threshold=0.50,
        root_pos_threshold_bounds=(0.10, 0.50),
        base_root_ori_threshold=0.50,
        root_ori_threshold_bounds=(0.15, 0.50),
        base_position_sigma=0.15,
        position_sigma_bounds=(0.05, 0.15),
        base_reset_noise_scale=1.0,
        reset_noise_scale_bounds=(1.0 / 6.0, 1.0),
    )
    assert settings == pytest.approx(
        {
            "body_pos_threshold": 0.18,
            "root_pos_threshold": 0.30,
            "root_ori_threshold": 0.30,
            "position_sigma": 0.09,
            "reset_noise_scale": 0.60,
            "derived_scale": 0.60,
        }
    )


def test_probe_p80_respects_absolute_safety_bounds() -> None:
    settings = derive_tracking_settings(
        0.01,
        base_body_threshold=0.30,
        body_threshold_bounds=(0.05, 0.30),
        base_root_pos_threshold=0.50,
        root_pos_threshold_bounds=(0.10, 0.50),
        base_root_ori_threshold=0.50,
        root_ori_threshold_bounds=(0.15, 0.50),
        base_position_sigma=0.15,
        position_sigma_bounds=(0.05, 0.15),
        base_reset_noise_scale=1.0,
        reset_noise_scale_bounds=(1.0 / 6.0, 1.0),
    )
    assert settings["body_pos_threshold"] == pytest.approx(0.05)
    assert settings["root_pos_threshold"] == pytest.approx(0.10)
    assert settings["root_ori_threshold"] == pytest.approx(0.15)
    assert settings["position_sigma"] == pytest.approx(0.05)
    assert settings["reset_noise_scale"] == pytest.approx(1.0 / 6.0)


def _make_probe_accumulator() -> tuple[TrackingPrecisionCurriculum, list[float]]:
    term = TrackingPrecisionCurriculum.__new__(TrackingPrecisionCurriculum)
    term.env = SimpleNamespace(device="cpu")
    term.command = SimpleNamespace(
        _probe_env_mask=torch.tensor([True]),
        _probe_episode_valid=torch.tensor([True]),
        motion_ids=torch.tensor([0]),
        time_steps=torch.tensor([0]),
        motion=SimpleNamespace(
            motion_start_idx=torch.tensor([0]),
            motion_end_idx=torch.tensor([5]),
        ),
    )
    term._probe_episode_max = torch.zeros(1)
    term._probe_episode_seen = torch.zeros(1, dtype=torch.bool)
    term._probe_episode_completed = torch.zeros(1, dtype=torch.bool)
    term._probe_episode_qualified = torch.ones(1, dtype=torch.bool)
    term._completed_probe_episode_count = 0
    term._accepted_probe_episode_count = 0
    term._rejected_probe_episode_count = 0
    term._last_completed_probe_max_mean = 0.0
    term._last_completed_probe_max_max = 0.0
    term._last_completed_probe_qualified_fraction = 0.0
    term._last_probe_error_mean = 0.0
    term._last_probe_error_max = 0.0
    term.base_body_threshold = 0.30
    term.base_root_pos_threshold = 0.50
    term.base_root_ori_threshold = 0.50
    appended: list[float] = []
    term._append_history = lambda values: appended.extend(values.tolist())
    term._refresh_statistics = lambda: None
    return term, appended


def test_probe_accumulator_commits_only_at_motion_end() -> None:
    term, appended = _make_probe_accumulator()

    # The reset target frame is not an execution sample, and may contain a
    # stale rigid-body cache before the first simulator step.
    term.observe_probe_tracking_error(torch.tensor([9.0]))
    assert not term._probe_episode_seen.any()

    term.command.time_steps[:] = 1
    term.observe_probe_tracking_error(torch.tensor([0.25]))
    assert appended == []

    term.command.time_steps[:] = 3
    term.observe_probe_tracking_error(torch.tensor([0.20]))
    assert appended == pytest.approx([0.25])
    assert term._completed_probe_episode_count == 1
    assert term._accepted_probe_episode_count == 1
    assert term._rejected_probe_episode_count == 0
    assert term.completed_probe_qualification(torch.tensor([0])).tolist() == [True]

    # Re-evaluating the terminal frame must not duplicate the sample.
    term.observe_probe_tracking_error(torch.tensor([0.80]))
    assert appended == pytest.approx([0.25])


def test_probe_reset_rejects_failed_partial_trajectory() -> None:
    term, appended = _make_probe_accumulator()
    term.env = SimpleNamespace(device="cpu")

    term.command.time_steps[:] = 2
    term.observe_probe_tracking_error(torch.tensor([0.60]))
    term.reset(torch.tensor([0]))

    assert appended == []
    assert term._rejected_probe_episode_count == 1
    assert not term._probe_episode_seen.any()
    assert not term._probe_episode_completed.any()
    assert term._probe_episode_qualified.all()


def test_probe_reset_does_not_reject_qualified_partial_trajectory() -> None:
    term, appended = _make_probe_accumulator()

    term.command.time_steps[:] = 2
    term.observe_probe_tracking_error(torch.tensor([0.20]))
    term.reset(torch.tensor([0]))

    assert appended == []
    assert term._rejected_probe_episode_count == 0


def test_probe_accumulator_accepts_next_trajectory_after_timestep_wrap() -> None:
    term, appended = _make_probe_accumulator()

    term.command.time_steps[:] = 3
    term.observe_probe_tracking_error(torch.tensor([0.25]))
    term.command.time_steps[:] = 0
    term.observe_probe_tracking_error(torch.tensor([0.10]))
    term.command.time_steps[:] = 3
    term.observe_probe_tracking_error(torch.tensor([0.30]))

    assert appended == pytest.approx([0.25, 0.30])
    assert term._completed_probe_episode_count == 2


def test_probe_accumulator_rejects_complete_run_outside_fixed_boundary() -> None:
    term, appended = _make_probe_accumulator()

    term.command.time_steps[:] = 1
    term.observe_probe_tracking_error(
        torch.tensor([0.20]),
        ref_position_error=torch.tensor([0.60]),
        ref_orientation_error=torch.tensor([0.10]),
    )
    term.command.time_steps[:] = 3
    term.observe_probe_tracking_error(
        torch.tensor([0.25]),
        ref_position_error=torch.tensor([0.10]),
        ref_orientation_error=torch.tensor([0.10]),
    )

    assert appended == []
    assert term._completed_probe_episode_count == 1
    assert term._accepted_probe_episode_count == 0
    assert term._rejected_probe_episode_count == 1
    assert term._last_completed_probe_qualified_fraction == 0.0
    assert term.completed_probe_qualification(torch.tensor([0])).tolist() == [False]
    assert term._last_probe_error_mean == 0.0
    assert term._last_probe_error_max == 0.0


def test_fixed_probe_qualification_does_not_follow_adaptive_threshold() -> None:
    term, appended = _make_probe_accumulator()
    # 0.25 would exceed a tightened training boundary such as 0.15, but it
    # remains a valid complete execution under the fixed 0.30 qualification line.
    term.command.time_steps[:] = 1
    term.observe_probe_tracking_error(torch.tensor([0.25]))
    term.command.time_steps[:] = 3
    term.observe_probe_tracking_error(torch.tensor([0.20]))

    assert appended == pytest.approx([0.25])
    assert term._accepted_probe_episode_count == 1


def test_worst_k_body_reward_exposes_one_bad_limb(monkeypatch) -> None:
    body_count = 14
    target = torch.zeros(1, body_count, 3)
    actual = target.clone()
    actual[0, 0, 0] = 0.25
    motion_command = SimpleNamespace(body_pos_relative_w=target, robot_body_pos_w=actual)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _: motion_command))
    monkeypatch.setattr(wbt_reward, "_get_motion_command_and_assert_type", lambda _: motion_command)

    mean_reward = motion_relative_body_position_error_exp(env, sigma=0.15)
    worst_reward = motion_relative_body_position_error_exp(env, sigma=0.15, worst_k=3)

    assert float(worst_reward.item()) < float(mean_reward.item())
    torch.testing.assert_close(
        worst_reward,
        torch.exp(torch.tensor([-0.25**2 / (3 * 0.15**2)])),
    )
