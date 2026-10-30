from types import SimpleNamespace

import pytest
import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand


def _command(motion_files: list[str]) -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.device = torch.device("cpu")
    command.motion = SimpleNamespace(motion_files=motion_files)
    count = len(motion_files)
    command._normal_motion_sampling_weights = torch.full((count,), 1.0 / count)
    command._probe_completion_recent30 = torch.zeros(count)
    command._probe_success_recent30 = torch.zeros(count)
    command._probe_fail_recent30 = torch.zeros(count)
    command._probe_timeout_recent30 = torch.zeros(count)
    command._probe_count = torch.zeros(count, dtype=torch.long)
    command._init_recent_probe_window()
    command._failure_window_failure_hist = torch.zeros(count, 3)
    command.adaptive_timesteps_sampler = SimpleNamespace(
        bin_failed_count=torch.zeros(count, 2),
        current_bin_failed_count=torch.zeros(count, 2),
    )
    return command


def test_sampler_checkpoint_roundtrip_remaps_reordered_manifest(tmp_path) -> None:
    first = str(tmp_path / "first.npz")
    second = str(tmp_path / "second.npz")
    source = _command([first, second])
    source._normal_motion_sampling_weights.copy_(torch.tensor([0.8, 0.2]))
    source._probe_completion_recent30.copy_(torch.tensor([0.3, 0.9]))
    source._probe_success_recent30.copy_(torch.tensor([0.2, 0.95]))
    source._probe_count.copy_(torch.tensor([4, 11]))
    source._failure_window_failure_hist.copy_(torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]))
    source.adaptive_timesteps_sampler.bin_failed_count.copy_(
        torch.tensor([[0.1, 0.2], [0.7, 0.8]])
    )

    source._probe_recent30_values[:, 0, 0] = source._probe_completion_recent30
    source._probe_recent30_values[:, 0, 1] = source._probe_success_recent30
    source._probe_recent30_count.fill_(1)
    source._probe_recent30_cursor.fill_(1)
    state = source.get_checkpoint_state()
    target = _command([second, first])
    target.load_checkpoint_state(state)

    assert target._normal_motion_sampling_weights.tolist() == pytest.approx([0.2, 0.8])
    assert target._probe_completion_recent30.tolist() == pytest.approx([0.9, 0.3])
    assert target._probe_success_recent30.tolist() == pytest.approx([0.95, 0.2])
    assert target._probe_count.tolist() == [11, 4]
    assert target._failure_window_failure_hist.tolist() == [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]
    assert torch.allclose(
        target.adaptive_timesteps_sampler.bin_failed_count,
        torch.tensor([[0.7, 0.8], [0.1, 0.2]]),
    )
    assert target._sampler_checkpoint_restored is True
    assert target._sampler_checkpoint_matched_motion_count == 2


def test_sampler_checkpoint_partially_restores_matching_motions(tmp_path) -> None:
    first = str(tmp_path / "first.npz")
    second = str(tmp_path / "second.npz")
    third = str(tmp_path / "third.npz")
    source = _command([first, second])
    source._probe_completion_recent30.copy_(torch.tensor([0.25, 0.75]))

    target = _command([second, third])
    source._probe_recent30_values[:, 0, 0] = source._probe_completion_recent30
    source._probe_recent30_count.fill_(1)
    source._probe_recent30_cursor.fill_(1)
    target.load_checkpoint_state(source.get_checkpoint_state())

    assert target._probe_completion_recent30.tolist() == pytest.approx([0.75, 0.0])
    assert target._sampler_checkpoint_restored is True
    assert target._sampler_checkpoint_matched_motion_count == 1
    assert target._sampler_checkpoint_saved_motion_count == 2


def test_sampler_checkpoint_legacy_fallback_is_recorded(tmp_path) -> None:
    command = _command([str(tmp_path / "motion.npz")])

    command.load_checkpoint_state(None)
    metrics = MotionCommand._append_checkpoint_restore_metrics(command, {})

    assert command._sampler_checkpoint_restored is False
    assert command._sampler_checkpoint_restore_reason == "legacy_checkpoint"
    assert metrics == {
        "sampler_checkpoint_restored": 0.0,
        "sampler_checkpoint_matched_motion_count": 0.0,
        "sampler_checkpoint_saved_motion_count": 0.0,
    }


def test_wbt_manager_wraps_motion_sampler_checkpoint_state() -> None:
    restored = []
    motion_command = SimpleNamespace(
        get_checkpoint_state=lambda: {"schema": "sampler"},
        load_checkpoint_state=restored.append,
    )
    manager = WholeBodyTrackingManager.__new__(WholeBodyTrackingManager)
    manager.command_manager = SimpleNamespace(get_state=lambda _name: motion_command)

    assert manager.get_checkpoint_state() == {
        "motion_sampler": {"schema": "sampler"},
    }

    manager.load_checkpoint_state({"motion_sampler": {"schema": "restored"}})

    assert restored == [{"schema": "restored"}]
