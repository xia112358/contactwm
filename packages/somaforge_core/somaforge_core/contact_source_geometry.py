"""Source-feature geometry for face attribution, never contact activation.

Newton's manifold perturbations can leave a witness a few floating units off
its emitting triangle. Projection identifies that triangle's geometric feature;
it must never replace a saved physical witness or change solver fields.
"""
import numpy as np

SOURCE_NORMAL_FAN_SCHEMA = 'newton_source_triangle_witness_normal_fan_v2'
FLOAT32_FACE_ROUNDOFF_FACTOR = 8*np.finfo(np.float32).eps


def project_source_triangle(point, triangle):
    point, triangle = np.asarray(point, float), np.asarray(triangle, float)
    if point.shape != (3,) or triangle.shape != (3, 3) or not np.isfinite(np.r_[point, triangle.ravel()]).all():
        raise ValueError('Invalid actual source-triangle witness')
    a, b, c = triangle
    normal = np.cross(b-a, c-a)
    length = np.linalg.norm(normal)
    if length == 0:
        raise ValueError('Degenerate actual source triangle')
    normal /= length
    projected = point-((point-a)@normal)*normal
    edges = np.roll(triangle, -1, axis=0)-triangle
    if np.all(np.cross(edges, projected-triangle)@normal >= 0):
        return projected
    parameter = ((projected-triangle)*edges).sum(-1)/np.square(edges).sum(-1)
    nearest = triangle+parameter.clip(0, 1)[:, None]*edges
    return nearest[np.square(nearest-projected).sum(-1).argmin()]


def incident_witness_faces(projected, normals, offsets, extents):
    """Plane membership at float32 arithmetic precision, in metres.

    This is a numerical predicate on an already identified source feature,
    not a gap/force threshold and not permission to activate a contact.
    """
    product = projected*normals
    bound = FLOAT32_FACE_ROUNDOFF_FACTOR*(np.abs(product).sum(-1)+np.abs(offsets)+extents)
    return np.abs(product.sum(-1)-offsets) <= bound


def project_source_triangles_tensor(points, triangles):
    """Batched equivalent of project_source_triangle; returns validity too."""
    import torch
    edge = triangles.roll(-1, -2)-triangles
    normal = torch.cross(edge[..., 0, :], -edge[..., 2, :], dim=-1)
    length = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    valid = (length.squeeze(-1) > 0) & torch.isfinite(triangles).all((-1, -2)) & torch.isfinite(points).all(-1)
    normal = normal/length.clamp_min(torch.finfo(points.dtype).tiny)
    projected = points-((points-triangles[..., 0, :])*normal).sum(-1, keepdim=True)*normal
    delta = projected[..., None, :]-triangles
    inside = (torch.cross(edge, delta, dim=-1)*normal[..., None, :]).sum(-1).ge(0).all(-1)
    parameter = (delta*edge).sum(-1)/edge.square().sum(-1).clamp_min(torch.finfo(points.dtype).tiny)
    nearest = triangles+parameter.clamp(0, 1)[..., None]*edge
    choice = (nearest-projected[..., None, :]).square().sum(-1).argmin(-1)
    nearest = nearest.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 3)).squeeze(-2)
    return torch.where(inside[..., None], projected, nearest), valid


def incident_witness_faces_tensor(projected, normals, offsets, extents):
    product = projected[..., None, :]*normals
    bound = FLOAT32_FACE_ROUNDOFF_FACTOR*(product.abs().sum(-1)+offsets.abs()+extents)
    return (product.sum(-1)-offsets).abs() <= bound
