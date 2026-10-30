"""Canonical per-link physical extents; geometric features are not contact truth."""
import torch
from somaforge_core.heightmap import (HEIGHTMAP_ROWS, HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MIN_M,
    HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_RESOLUTION_M)


def body_geometry_features(model, current_q, heightmap, basis):
    """Per-link collision extent and observed terrain clearance features.

    All points come from the canonical collision asset. Height differences
    are observation features only, never a contact predicate. Out-of-view
    points are explicitly masked rather than assigned a border height.
    """
    points, _ = model.geometry(model.fk, current_q)
    local = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), points-current_q[:, None, :3])
    b, _, _ = local.shape
    n = len(model.body_geometry_names)
    idx = model.body_geometry_point_link[None, :, None].expand(b, -1, 3)
    low = local.new_full((b, n, 3), torch.inf).scatter_reduce(1, idx, local, reduce='amin', include_self=True)
    high = local.new_full((b, n, 3), -torch.inf).scatter_reduce(1, idx, local, reduce='amax', include_self=True)
    total = local.new_zeros(b, n, 3).scatter_add(1, idx, local)
    ids = model.body_geometry_point_link[None].expand(b, -1)
    count = local.new_zeros(b, n).scatter_add(1, ids, torch.ones_like(local[..., 0]))
    mean = total/count[..., None]
    x = (local[..., 0]-HEIGHTMAP_FORWARD_MIN_M)/HEIGHTMAP_RESOLUTION_M
    y = (local[..., 1]-HEIGHTMAP_LATERAL_MIN_M)/HEIGHTMAP_RESOLUTION_M
    visible = (x >= 0) & (x <= HEIGHTMAP_ROWS-1) & (y >= 0) & (y <= HEIGHTMAP_COLS-1)
    cell = x.round().long().clamp(0, HEIGHTMAP_ROWS-1)*HEIGHTMAP_COLS+y.round().long().clamp(0, HEIGHTMAP_COLS-1)
    gap = local[..., 2]-heightmap.flatten(1).gather(1, cell)
    observed = local.new_zeros(b, n).scatter_add(1, ids, visible.to(local))
    gap_mean = local.new_zeros(b, n).scatter_add(1, ids, torch.where(visible, gap, 0))/observed.clamp_min(1)
    gap_min = local.new_full((b, n), torch.inf).scatter_reduce(1, ids,
        torch.where(visible, gap, torch.inf), reduce='amin', include_self=True)
    gap_max = local.new_full((b, n), -torch.inf).scatter_reduce(1, ids,
        torch.where(visible, gap, -torch.inf), reduce='amax', include_self=True)
    gap_min = torch.where(observed > 0, gap_min, 0)
    gap_max = torch.where(observed > 0, gap_max, 0)
    return torch.cat((mean, low, high, gap_mean[..., None], gap_min[..., None],
                      gap_max[..., None], (observed/count)[..., None]), -1)

