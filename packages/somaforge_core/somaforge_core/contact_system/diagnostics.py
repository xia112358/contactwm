"""Fixed-trajectory diagnostics; no simulation or optimizer side effects."""
import torch


def variation(values, valid):
    """Adjacent variation only inside known, uninterrupted trajectory runs."""
    if values.shape != valid.shape or values.ndim < 1:
        raise ValueError("Diagnostic values and masks must share [time,...] shape")
    pair = valid[1:] & valid[:-1]
    change = (values[1:]-values[:-1]).abs()[pair]
    if not len(change):
        return dict(pairs=0, mean=None, p95=None, maximum=None)
    return dict(pairs=len(change), mean=float(change.mean()),
                p95=float(torch.quantile(change.float(), .95)), maximum=float(change.max()))


def witness_motion(points, positions, rotations, valid):
    """Separate rigid transport from witness reselection, without calling it slip."""
    if points.shape != positions.shape or points.shape[:-1] != valid.shape:
        raise ValueError("Witness diagnostic dimensions disagree")
    local = torch.einsum("...ji,...j->...i", rotations[:-1], points[:-1]-positions[:-1])
    transported = positions[1:]+torch.einsum("...ij,...j->...i", rotations[1:], local)
    pair = valid[1:] & valid[:-1]
    def describe(value):
        x = value.norm(dim=-1)[pair]
        return dict(mean_m=float(x.mean()), p95_m=float(torch.quantile(x.float(), .95))) if len(x) else None
    return dict(scope="geometric decomposition; not physical slip or causal attribution",
                pairs=int(pair.sum()), point_motion=describe(points[1:]-points[:-1]),
                rigid_transport=describe(transported-points[:-1]),
                reselection_residual=describe(points[1:]-transported))
