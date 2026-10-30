"""Recent probe outcomes must drive training and survive ordinary checkpoints."""
import csv
from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand


def command(files=("/a.npz", "/b.npz"), *, group=False):
    c = MotionCommand.__new__(MotionCommand)
    c.device = torch.device("cpu")
    c.motion = SimpleNamespace(motion_files=list(files), num_motions=len(files))
    c.motion_cfg = SimpleNamespace(probe_priority_temperature=0.2, probe_uniform_mix=0.4, probe_weight_beta=1.0)
    c._use_completion_learning_sampler = False
    c._use_start_probe_envs = not group
    c._use_group_probe_envs = group
    c._base_motion_sampling_weights = torch.full((len(files),), 1 / len(files))
    c._normal_motion_sampling_weights = c._base_motion_sampling_weights.clone()
    c._probe_env_per_motion = 3
    c._probe_fail_bin_count = 2
    c._probe_fail_bin_hist = torch.zeros(len(files), 2)
    for prefix, n in (("_probe", len(files)), ("_group_probe", 2)):
        setattr(c, prefix + "_count", torch.zeros(n, dtype=torch.long))
        for metric in ("completion", "success", "fail", "timeout"):
            setattr(c, prefix + "_" + metric + "_recent30", torch.zeros(n))
        c._init_recent_probe_window(group=prefix == "_group_probe")
    if group:
        c._num_motion_groups = 2
        c._motion_group_ids = torch.arange(len(files)) % 2
        c._group_motion_counts = torch.bincount(c._motion_group_ids, minlength=2)
        c._normal_group_sampling_weights = torch.tensor([0.5, 0.5])
        c._group_probe_fail_bin_hist = torch.zeros(2, 2)
    return c


def append(c, index, outcomes, *, group=False):
    success = torch.tensor(outcomes, dtype=torch.float32)
    completion = torch.where(success.bool(), 1.0, 0.25)
    c._append_recent_probe_results(index, completion, success, 1 - success, torch.zeros_like(success), group=group)
    getattr(c, "_group_probe_count" if group else "_probe_count")[index] += len(outcomes)


def test_simultaneous_finishes_count_as_individual_episodes():
    c = command()
    append(c, 0, [1, 0, 1])
    assert c._probe_recent30_count.tolist() == [3, 0]
    assert c._probe_success_recent30[0].item() == pytest.approx(2 / 3)
    assert c._probe_completion_recent30[0].item() == pytest.approx(0.75)
    assert c._probe_fail_recent30[0].item() == pytest.approx(1 / 3)


@pytest.mark.parametrize("chunks", [[3] * 20, [1] * 60, [17, 43], [60]])
def test_batching_and_wraparound_preserve_exact_last_30(chunks):
    c = command()
    outcomes = [0] * 25 + [1] * 20 + [0] * 15
    offset = 0
    for n in chunks:
        append(c, 0, outcomes[offset:offset + n])
        offset += n
    assert c._probe_count.tolist() == [60, 0]
    assert c._probe_recent30_count.tolist() == [30, 0]
    assert c._probe_success_recent30[0].item() == pytest.approx(0.5)
    assert c._probe_completion_recent30[0].item() == pytest.approx(0.625)


def test_old_failures_leave_window_and_training_priority_reverses():
    c = command()
    append(c, 0, [0] * 30)
    append(c, 1, [1] * 30)
    c.update_probe_motion_sampling_weights()
    assert c._normal_motion_sampling_weights[0] > c._normal_motion_sampling_weights[1]
    append(c, 0, [1] * 30)
    append(c, 1, [0] * 30)
    c.update_probe_motion_sampling_weights()
    assert c._probe_success_recent30.tolist() == [1.0, 0.0]
    assert c._normal_motion_sampling_weights[0] < c._normal_motion_sampling_weights[1]
    centers, valid = c._completion_recent30_reset_centers(torch.tensor([0, 1]))
    assert centers.tolist() == [1.0, 0.25]
    assert valid.all()


def test_checkpoint_restores_reordered_ring_and_next_eviction():
    source = command()
    append(source, 0, [0] * 23 + [1] * 14)
    append(source, 1, [1, 0, 1])
    source.update_probe_motion_sampling_weights()
    target = command(("/b.npz", "/a.npz"))
    target.load_checkpoint_state(source.get_checkpoint_state())
    append(source, 0, [1, 1, 0])
    append(target, 1, [1, 1, 0])
    assert torch.equal(target._probe_recent30_values[1], source._probe_recent30_values[0])
    assert torch.equal(target._probe_recent30_cursor[1], source._probe_recent30_cursor[0])
    assert target._probe_success_recent30[1] == source._probe_success_recent30[0]
    assert target._normal_motion_sampling_weights.tolist() == pytest.approx(
        source._normal_motion_sampling_weights.flip(0).tolist()
    )


def test_legacy_ema_is_not_fabricated_into_recent_observations():
    c = command()
    legacy = {"schema": "holosoma_wbt_motion_sampler_v1", "motion_keys": c.motion.motion_files,
              "tensors": {"_probe_count": torch.tensor([900, 800]),
                          "_probe_completion_ema": torch.tensor([0.8, 0.9]),
                          "_normal_motion_sampling_weights": torch.tensor([0.9, 0.1])}}
    c.load_checkpoint_state(legacy)
    assert c._probe_count.tolist() == [900, 800]
    assert c._probe_recent30_count.tolist() == [0, 0]
    assert c._normal_motion_sampling_weights.tolist() == [0.5, 0.5]
    centers, valid = c._completion_recent30_reset_centers(torch.tensor([0, 1]))
    assert not valid.any()
    append(c, 0, [1, 1, 1])
    assert c._probe_completion_recent30.tolist() == [1.0, 0.0]


def test_incomplete_window_checkpoint_is_not_partially_restored():
    source = command()
    append(source, 0, [1, 0, 1])
    state = source.get_checkpoint_state()
    state["tensors"].pop("_probe_recent30_cursor")
    target = command()
    target.load_checkpoint_state(state)
    assert not target._probe_recent30_count.any()
    assert not target._probe_recent30_values.any()


def test_group_training_and_checkpoint_use_recent_window():
    source = command(group=True)
    append(source, 0, [0] * 30, group=True)
    append(source, 1, [1] * 30, group=True)
    source.update_probe_motion_sampling_weights()
    assert source._normal_group_sampling_weights[0] > source._normal_group_sampling_weights[1]
    target = command(("/b.npz", "/a.npz"), group=True)
    target.load_checkpoint_state(source.get_checkpoint_state())
    assert target._group_probe_success_recent30.tolist() == [1.0, 0.0]
    append(target, 0, [0] * 30, group=True)
    append(target, 1, [1] * 30, group=True)
    target.update_probe_motion_sampling_weights()
    assert target._normal_group_sampling_weights[0] > target._normal_group_sampling_weights[1]


def test_csv_and_tensorboard_report_same_training_statistics(tmp_path):
    c = command()
    append(c, 0, [1, 0, 1])
    append(c, 1, [1] * 30)
    c.update_probe_motion_sampling_weights()
    c.write_start_probe_distribution(tmp_path, 100)
    rows = list(csv.DictReader((tmp_path / "start_probe_recent30_distribution.csv").open()))
    assert rows[0]["recent30_count"] == "3"
    assert float(rows[0]["start_to_end_success_recent30"]) == pytest.approx(2 / 3)
    metrics = c.get_motion_learning_progress_metrics()
    assert not any("ema" in name for name in metrics)
    assert metrics["start_to_end_success_recent30_min"] == pytest.approx(2 / 3)
    assert metrics["recent30_full_window_fraction"] == 0.5
