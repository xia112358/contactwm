"""Preserved experiment helpers for regression tests; no training CLI."""
def adjacent_pairs(samples, roles, surfaces, ids):
    """Pair exact event boundaries before the first two-feet-on-top endpoint."""
    groups = {}
    for i in ids:
        groups.setdefault(samples[i].motion_id, []).append(i)
    pairs = []
    for (motion, indices) in groups.items():
        indices.sort(key=lambda i: samples[i].current_frame)
        feet = [samples[i].target_frame for i in indices if all((roles[i][k] != 0 and surfaces[i][k] == 1 for k in (0, 1)))]
        if not feet:
            continue
        end = min(feet)
        for (a, b) in zip(indices, indices[1:]):
            if samples[a].target_frame != samples[b].current_frame:
                continue
            if samples[b].target_frame > end:
                continue
            if not any((roles[a][k] != 0 and surfaces[a][k] == 1 for k in (2, 3, 4, 5))):
                continue
            pairs.append((a, b))
    return pairs
