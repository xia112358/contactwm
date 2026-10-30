from collections import Counter

import pytest
import torch

from generator.spatial_replay import (
    spatial_replay_group, stratified_spatial_positions, trim_spatial_replay,
)


def entries(groups):
    return [(i, {}, {"group": group}) for i, group in enumerate(groups)]


def test_only_first_failure_is_frontier_even_if_recovery_later_succeeds():
    complete = [True, True, False, False, True, False]
    prefix_open = True
    groups = []
    for success in complete:
        groups.append(spatial_replay_group(prefix_open, success))
        prefix_open = prefix_open and success
    assert groups == ["success", "success", "frontier", "recovery", "recovery", "recovery"]


def test_long_failed_tail_cannot_displace_frontier_or_success_samples():
    replay = entries(["success"] * 2 + ["frontier"] + ["recovery"] * 200)
    chosen = stratified_spatial_positions(replay, 12)
    assert Counter(replay[i][2]["group"] for i in chosen) == {
        "frontier": 6, "success": 3, "recovery": 3,
    }


def test_missing_group_redistributes_quota_without_fabricating_success():
    replay = entries(["frontier"] + ["recovery"] * 3)
    chosen = stratified_spatial_positions(replay, 9)
    assert Counter(replay[i][2]["group"] for i in chosen) == {"frontier": 6, "recovery": 3}


def test_diversity_before_repetition_within_group():
    replay = entries(["frontier"] * 6)
    chosen = stratified_spatial_positions(replay, 12)
    assert len(set(chosen[:6])) == 6
    assert len(set(chosen[6:])) == 6


def test_sampling_is_reproducible_with_training_seed():
    replay = entries(["success", "frontier", "recovery"] * 10)
    torch.manual_seed(42)
    first = stratified_spatial_positions(replay, 12)
    torch.manual_seed(42)
    assert first == stratified_spatial_positions(replay, 12)


def test_trimming_retains_success_and_frontier_before_latest_recovery():
    replay = entries(["success"] * 4 + ["frontier"] * 4 + ["recovery"] * 200)
    retained = trim_spatial_replay(replay, 12)
    assert len(retained) == 12
    assert Counter(row[2]["group"] for row in retained) == {
        "success": 4, "frontier": 4, "recovery": 4,
    }
    assert retained[-1] == replay[-1]


def test_trimming_uses_spare_capacity_when_a_group_is_small():
    replay = entries(["success", "frontier"] + ["recovery"] * 20)
    assert len(trim_spatial_replay(replay, 12)) == 12


def test_unknown_group_is_not_silently_used():
    with pytest.raises(ValueError, match="unknown"):
        stratified_spatial_positions(entries(["unknown"]), 8)


def test_empty_replay_is_empty():
    assert stratified_spatial_positions([], 12) == []
