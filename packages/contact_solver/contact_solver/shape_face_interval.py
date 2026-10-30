"""Shape support over a finite face, for loss geometry only.

The normal gap is measured on the original collision skin restricted to the
vertical prism of a finite convex face. Overhanging skin outside that prism
does not become penetration of an infinite plane. No function here establishes
Newton contact, projects a pose, or supplies a custom backward.
"""
import itertools

import torch


SHAPE_FACE_INTERVAL_SCHEMA = 'finite_face_clipped_collision_skin_interval_v1'


class NativeShapeFaceIntervals:
    """Immutable adapter for initialized robot skins and actual finite faces.

    Uses the same exact meshes/spheres and anatomical cuts as the native
    endpoint-position geometry. Empty horizontal overlap stays explicit; a
    caller must compute finite-face approach distance there, not treat it as
    achieved contact or invent a support face.
    """
    def __init__(self, metadata, fk, low, high):
        import numpy as np
        from scipy.spatial import ConvexHull
        from somaforge_core.contact_face_selection import upward_face_mask
        from contact_solver.native_contact_position import NativeContactRegionGeometry
        self.geometry = NativeContactRegionGeometry(metadata, fk, low, high)
        self.faces = {}
        self.polygons = {}
        catalog = metadata['surface_catalog']
        selected = upward_face_mask(np.array([f['normal_w'] for f in catalog]))
        for face, upward in zip(catalog, selected):
            if not upward:
                continue
            normal = np.asarray(face['normal_w'], dtype=float)
            triangles = np.asarray(face['triangles_w'], dtype=float)
            vertices = np.unique(triangles.reshape(-1, 3), axis=0)
            # Only the current horizontal task faces are accepted. Selection
            # itself belongs to the shared actual-primary-face authority.
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            u = np.cross(normal, axis); u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            hull = ConvexHull(np.stack((vertices@u, vertices@v), -1))
            polygon = vertices[hull.vertices]
            area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1).sum()/2
            if not np.isclose(area, hull.volume, rtol=1e-6, atol=1e-10):
                raise ValueError('A nonconvex or overlapping face needs its actual convex decomposition')
            edges = np.roll(polygon, -1, axis=0)-polygon
            outward = np.cross(edges, normal)
            outward /= np.linalg.norm(outward, axis=-1, keepdims=True)
            offsets = (outward*polygon).sum(-1)
            # Rounded catalog plane_offset is for attribution. The actual
            # triangle coordinates give the loss plane at full precision.
            plane = float(normal@polygon[0])
            if np.max(np.abs(triangles@normal-plane)) > 1e-7:
                raise ValueError('Nonplanar actual target face')
            self.faces[int(face['surface'])] = (torch.from_numpy(outward), torch.from_numpy(offsets),
                torch.from_numpy(normal), plane)
            self.polygons[int(face['surface'])] = torch.from_numpy(polygon)
        if not self.faces:
            raise ValueError('No actual selected upward finite faces')

    def gaps(self, fk, q, surfaces, *, world_frame=None):
        from somaforge_core.motion_contracts import BODY_NAMES
        from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d
        if surfaces.shape != (len(q), 6):
            raise ValueError('Shape/face routes must name six parts per pose')
        position, rotation = fk.link_poses(q, BODY_NAMES[1:7])
        rotation = _matrix_from_rotation6d(_rotation6d(rotation))
        if world_frame is not None:
            origin, basis = world_frame
            basis = _matrix_from_rotation6d(_rotation6d(basis.to(q)))
            position = origin[:, None].to(q)+torch.einsum('bij,bpj->bpi', basis, position)
            rotation = basis[:, None]@rotation
        result = q.new_full((len(q), 6, 4), torch.inf)
        for part in range(6):
            for sid in surfaces[:, part].unique().tolist():
                if sid < 0:
                    continue
                if sid not in self.faces:
                    raise ValueError('Intended surface is not an actual selected finite face')
                indices = (surfaces[:, part] == sid).nonzero().flatten()
                pos, rot = position[indices, part], rotation[indices, part]
                normals, offsets, direction, plane = self.faces[sid]
                normals = normals.to(q).expand(len(indices), -1, -1)
                offsets = offsets.to(q).expand(len(indices), -1)
                direction = direction.to(q).expand(len(indices), -1)
                for region in range((4, 4, 2, 2, 3, 3)[part]):
                    triangles, spheres, cut_n, cut_b = self.geometry.patch[part, region]
                    local = triangles.to(q)
                    world = pos[:, None, None]+torch.einsum('bij,tvj->btvi', rot, local)
                    support = triangle_prism_support(world, normals, offsets, direction)
                    if spheres:
                        cut_n = torch.einsum('bij,kj->bki', rot, cut_n.to(q))
                        cut_b = cut_b.to(q)[None]+(cut_n*pos[:, None]).sum(-1)
                        constraints = torch.cat((normals, cut_n), 1)
                        bounds = torch.cat((offsets, cut_b), 1)
                        for center, radius in spheres:
                            center = pos+torch.einsum('bij,j->bi', rot, center.to(q))
                            support = torch.minimum(support, sphere_prism_support(center, radius, constraints, bounds, direction))
                    result[indices, part, region] = support-plane
        return result


def interval_cost(gap, margin):
    """One signed coordinate, with lower zero and the actual margin as upper."""
    if not bool(torch.isfinite(gap).all() & torch.isfinite(margin).all() & (margin > 0).all()):
        raise ValueError('An interval needs finite gaps and actual positive margins')
    return (-gap).relu().square() + (gap-margin).relu().square()


def triangle_prism_support(triangles, normals, offsets, direction):
    """Minimum directional support of triangular skin inside a convex prism.

    All inputs are batched. Normals/offsets describe the face's tangent
    half-spaces. Enumerate the vertices of each clipped triangle: original
    vertices, edge/plane intersections, and triangle/two-plane intersections.
    Selection is piecewise smooth; coordinates retain their real derivatives.
    ``inf`` explicitly means empty skin, rather than a contact or zero gap.
    """
    batch, count = triangles.shape[:2]
    if triangles.shape != (batch, count, 3, 3) or normals.shape[:1] != (batch,):
        raise ValueError('Invalid batched triangle/prism geometry')
    if normals.shape[-1] != 3 or offsets.shape != normals.shape[:2] or direction.shape != (batch, 3):
        raise ValueError('Invalid prism planes or support direction')
    if not bool(torch.isfinite(triangles).all() & torch.isfinite(normals).all()
                & torch.isfinite(offsets).all() & torch.isfinite(direction).all()):
        raise ValueError('Nonfinite triangle/prism geometry')
    if not count:
        return triangles.new_full((batch,), torch.inf)
    # Numerical slack follows dtype and geometry scale; it is not a contact
    # distance threshold and does not alter any solver margin.
    eps = 128*torch.finfo(triangles.dtype).eps
    slack = eps*(1+triangles.detach().abs().amax((1, 2, 3))+offsets.abs().amax(-1))
    candidates, masks = [], []

    def add(points, valid=None):
        points = points.reshape(batch, count, -1, 3)
        inside = (torch.einsum('btvi,bki->btvk', points, normals)
                  <= offsets[:, None, None]+slack[:, None, None, None]).all(-1)
        if valid is not None:
            inside = inside & valid.reshape(batch, count, -1)
        candidates.append((points*direction[:, None, None]).sum(-1))
        masks.append(inside & torch.isfinite(points).all(-1))

    add(triangles)
    end = triangles.roll(-1, -2)
    d0 = torch.einsum('btvi,bki->btvk', triangles, normals)-offsets[:, None, None]
    d1 = torch.einsum('btvi,bki->btvk', end, normals)-offsets[:, None, None]
    denominator = d0-d1
    nonparallel = denominator.abs() > eps*(1+d0.abs()+d1.abs())
    fraction = d0/torch.where(nonparallel, denominator, torch.ones_like(denominator))
    points = triangles[..., None, :]+fraction[..., None]*(end-triangles)[..., None, :]
    add(points, nonparallel & (fraction >= 0) & (fraction <= 1))
    a, b, c = triangles.unbind(-2)
    e, f = b-a, c-a
    tri_normal = torch.linalg.cross(e, f)
    tri_offset = (tri_normal*a).sum(-1)
    ee, ff, ef = e.square().sum(-1), f.square().sum(-1), (e*f).sum(-1)
    determinant = ee*ff-ef.square()
    for i, j in itertools.combinations(range(normals.shape[1]), 2):
        matrix = torch.stack((tri_normal, normals[:, None, i].expand_as(tri_normal),
                              normals[:, None, j].expand_as(tri_normal)), -2)
        rhs = torch.stack((tri_offset, offsets[:, None, i].expand_as(tri_offset),
                           offsets[:, None, j].expand_as(tri_offset)), -1)
        good = torch.linalg.det(matrix).abs() > eps*tri_normal.norm(dim=-1)
        safe = torch.where(good[..., None, None], matrix, torch.eye(3).to(matrix))
        point = torch.linalg.solve(safe, rhs[..., None]).squeeze(-1)
        delta = point-a
        de, df = (delta*e).sum(-1), (delta*f).sum(-1)
        denominator = torch.where(determinant > 0, determinant, torch.ones_like(determinant))
        u, v = (de*ff-df*ef)/denominator, (df*ee-de*ef)/denominator
        add(point, good & (determinant > 0) & (u >= -eps) & (v >= -eps) & (u+v <= 1+eps))
    values, valid = torch.cat(candidates, -1), torch.cat(masks, -1)
    return values.masked_fill(~valid, torch.inf).amin((1, 2))


def sphere_prism_support(center, radius, normals, offsets, direction):
    """Exact sphere skin support restricted by face and anatomical planes.

    A linear objective on a clipped sphere reaches its minimum on the free
    sphere, a clipping circle, or two planes' sphere intersections. Enumerating
    these features keeps the scalar differentiable without a pose solver.
    Empty patches are returned as explicit ``inf``.
    """
    batch = len(center)
    if center.shape != (batch, 3) or direction.shape != center.shape or offsets.shape != normals.shape[:2]:
        raise ValueError('Invalid sphere/prism geometry')
    radius = torch.broadcast_to(torch.as_tensor(radius, device=center.device, dtype=center.dtype), (batch,))
    if not bool(torch.isfinite(center).all() & torch.isfinite(radius).all() & (radius > 0).all()
                & torch.isfinite(normals).all() & torch.isfinite(offsets).all()):
        raise ValueError('Nonfinite sphere/prism geometry')
    eps = 128*torch.finfo(center.dtype).eps
    slack = eps*(1+center.detach().abs().sum(-1)+radius+offsets.abs().sum(-1))
    candidates, masks = [], []

    def add(point, valid=None):
        ok = ((point[:, None]*normals).sum(-1) <= offsets+slack[:, None]).all(-1)
        if valid is not None:
            ok = ok & valid
        candidates.append((point*direction).sum(-1))
        masks.append(ok & torch.isfinite(point).all(-1))

    add(center-radius[:, None]*direction/direction.norm(dim=-1, keepdim=True).clamp_min(eps))
    for i in range(normals.shape[1]):
        n = normals[:, i]
        norm = n.norm(dim=-1)
        if bool((norm <= 0).any()):
            raise ValueError('Zero clipping normal')
        unit = n/norm[:, None]
        shift = (offsets[:, i]-(center*n).sum(-1))/norm
        extent2 = radius.square()-shift.square()
        circle_center = center+shift[:, None]*unit
        tangent = direction-(direction*unit).sum(-1, keepdim=True)*unit
        size = tangent.norm(dim=-1)
        extent = extent2.clamp_min(torch.finfo(center.dtype).tiny).sqrt()
        add(circle_center-extent[:, None]*tangent/size.clamp_min(eps)[:, None],
            (extent2 >= 0) & (size > eps))
        # Parallel objective is constant on the circle. A feasible arc either
        # includes a basis extremum or ends at another clipping plane.
        axis = torch.nn.functional.one_hot(unit.abs().argmin(-1), 3).to(center)
        first = torch.linalg.cross(unit, axis)
        first = first/first.norm(dim=-1, keepdim=True)
        second = torch.linalg.cross(unit, first)
        for vector in (first, -first, second, -second):
            add(circle_center+extent[:, None]*vector, (extent2 >= 0) & (size <= eps))
        add(circle_center, extent2 == 0)
    for i, j in itertools.combinations(range(normals.shape[1]), 2):
        pair = normals[:, [i, j]]
        line = torch.linalg.cross(pair[:, 0], pair[:, 1])
        length = line.norm(dim=-1)
        good = length > eps*pair[:, 0].norm(dim=-1)*pair[:, 1].norm(dim=-1)
        gram = pair@pair.transpose(-1, -2)
        safe = torch.where(good[:, None, None], gram, torch.eye(2).to(gram))
        rhs = offsets[:, [i, j]]-(pair*center[:, None]).sum(-1)
        shift = (pair.transpose(-1, -2)@torch.linalg.solve(safe, rhs[..., None])).squeeze(-1)
        extent2 = radius.square()-shift.square().sum(-1)
        extent = extent2.clamp_min(torch.finfo(center.dtype).tiny).sqrt()
        vector = line/length.clamp_min(eps)[:, None]
        for sign in (-1, 1):
            add(center+shift+sign*extent[:, None]*vector, good & (extent2 >= 0))
    return torch.stack(candidates, -1).masked_fill(~torch.stack(masks, -1), torch.inf).amin(-1)
