"""Separate successful prefix inputs, first failures, and subsequent recovery."""

import torch


GROUP_WEIGHTS = {"frontier": 2, "success": 1, "recovery": 1}


def spatial_replay_group(prefix_open: bool, complete: bool) -> str:
    if not prefix_open:
        return "recovery"
    return "success" if complete else "frontier"


def stratified_spatial_positions(entries, batch_size: int) -> list[int]:
    """Reserve half of a full three-group batch for the first failing input.

    Repeat within a small group rather than allowing long failed tails to
    displace it. Shuffle without replacement before repeating any member.
    """
    if batch_size < 1:
        raise ValueError("spatial replay batch size must be positive")
    groups = {name: [] for name in GROUP_WEIGHTS}
    for position, entry in enumerate(entries):
        group = entry[2]["group"]
        if group not in groups:
            raise ValueError(f"unknown spatial replay group: {group}")
        groups[group].append(position)
    available = [name for name, positions in groups.items() if positions]
    if not available:
        return []
    weight = sum(GROUP_WEIGHTS[name] for name in available)
    desired = {name: batch_size * GROUP_WEIGHTS[name] / weight for name in available}
    quotas = {name: int(desired[name]) for name in available}
    remainder = batch_size - sum(quotas.values())
    for name in sorted(available, key=lambda name: desired[name] - quotas[name], reverse=True)[:remainder]:
        quotas[name] += 1
    chosen = []
    for name in available:
        positions, remaining = groups[name], quotas[name]
        while remaining:
            take = min(remaining, len(positions))
            order = torch.randperm(len(positions))[:take].tolist()
            chosen.extend(positions[index] for index in order)
            remaining -= take
    return chosen


def trim_spatial_replay(entries, capacity: int):
    """Retain recent records within each group so recovery cannot evict all successes."""
    if capacity < len(GROUP_WEIGHTS):
        raise ValueError("stratified replay capacity must accommodate all three groups")
    if len(entries) <= capacity:
        return list(entries)
    per_group = capacity // len(GROUP_WEIGHTS)
    retained = set()
    for name in GROUP_WEIGHTS:
        positions = [i for i, entry in enumerate(entries) if entry[2]["group"] == name]
        retained.update(positions[-per_group:])
    for position in reversed(range(len(entries))):
        if len(retained) >= capacity:
            break
        retained.add(position)
    return [entries[position] for position in sorted(retained)]
