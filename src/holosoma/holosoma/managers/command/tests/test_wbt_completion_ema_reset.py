from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand


def _command() -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.motion_cfg = SimpleNamespace(
        hotspot_failure_uniform_mix=0.0,
        failure_window_pre_frames=0,
        failure_window_post_frames=0,
        failure_window_before_prob=1.0,
    )
    command.motion = SimpleNamespace(
        motion_start_idx=torch.tensor([0, 1000]),
        motion_end_idx=torch.tensor([101, 1201]),
    )
    command.motion_ids = torch.tensor([0, 1])
    command._failure_window_retry_active = torch.zeros(2, dtype=torch.bool)
    command._failure_window_retry_motion_ids = torch.full((2,), -1, dtype=torch.long)
    command._failure_window_retry_target_steps = torch.zeros(2, dtype=torch.long)
    command._use_start_probe_envs = True
    command._use_group_probe_envs = False
    command._probe_count = torch.tensor([4, 4])
    command._probe_recent30_count = torch.tensor([4, 4])
    command._probe_completion_recent30 = torch.tensor([0.4, 0.6])
    return command


def test_completion_ema_branch_uses_each_motions_own_progress() -> None:
    command = _command()

    sampled = command._sample_completion_recent30_failure_window_time_steps(
        torch.tensor([0, 1])
    )

    assert sampled.tolist() == [40, 1120]
    assert command._failure_window_retry_active.tolist() == [True, True]
    assert command._failure_window_retry_motion_ids.tolist() == [0, 1]
    assert command._failure_window_retry_target_steps.tolist() == [40, 1120]


def test_completion_ema_branch_is_uniform_until_probe_has_reported() -> None:
    command = _command()
    command._probe_count.zero_()
    command._probe_recent30_count.zero_()

    torch.manual_seed(7)
    sampled = command._sample_completion_recent30_failure_window_time_steps(
        torch.tensor([0, 1])
    )

    assert 0 <= sampled[0].item() < 100
    assert 1000 <= sampled[1].item() < 1200
    assert command._failure_window_retry_active.tolist() == [False, False]


def test_completion_ema_branch_keeps_unreported_motion_uniform() -> None:
    command = _command()
    command._probe_count = torch.tensor([4, 0])
    command._probe_recent30_count = torch.tensor([4, 0])

    torch.manual_seed(7)
    sampled = command._sample_completion_recent30_failure_window_time_steps(
        torch.tensor([0, 1])
    )

    assert sampled[0].item() == 40
    assert 1000 <= sampled[1].item() < 1200
    assert command._failure_window_retry_active.tolist() == [True, False]
    assert command._failure_window_retry_motion_ids.tolist() == [0, -1]


def test_probe_success_logging_uses_fixed_tracking_qualification() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command._use_start_probe_envs = True
    command._use_group_probe_envs = False
    command._probe_env_mask = torch.tensor([True])
    command._probe_episode_valid = torch.tensor([True])
    command.motion_ids = torch.tensor([0])
    command.time_steps = torch.tensor([100])
    command.motion = SimpleNamespace(
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([101]),
    )
    command.motion_cfg = SimpleNamespace(probe_completion_alpha=0.02)
    command._probe_completion_recent30 = torch.zeros(1)
    command._probe_success_recent30 = torch.zeros(1)
    command._probe_fail_recent30 = torch.zeros(1)
    command._probe_timeout_recent30 = torch.zeros(1)
    command._probe_count = torch.zeros(1, dtype=torch.long)
    command._probe_fail_bin_count = 10
    command._probe_fail_bin_hist = torch.zeros(1, 10)
    qualification = SimpleNamespace(
        completed_probe_qualification=lambda env_ids: torch.zeros(env_ids.numel(), dtype=torch.bool)
    )
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False]),
            time_outs=torch.tensor([False]),
            term_dones={},
        ),
        curriculum_manager=SimpleNamespace(get_term=lambda name: qualification),
    )

    command._update_start_probe_stats(torch.tensor([0]))

    assert command._probe_completion_recent30.tolist() == pytest.approx([1.0])
    assert command._probe_success_recent30.tolist() == pytest.approx([0.0])
    assert command._probe_fail_recent30.tolist() == pytest.approx([1.0])
    assert command._probe_count.tolist() == [1]
