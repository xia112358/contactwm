"""Fixed material surface nodes, with area quadrature and wrench accounting."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn


class SurfaceAtlas(nn.Module):
    """Nodes are fixed in each part's link frame, never selected by a task goal."""

    def __init__(self, points, normals, areas, *, parts, provenance=None):
        super().__init__()
        points = torch.as_tensor(points)
        normals = torch.as_tensor(normals, dtype=points.dtype, device=points.device)
        areas = torch.as_tensor(areas, dtype=points.dtype, device=points.device)
        self.parts = tuple(parts)
        if points.ndim != 3 or points.shape[0] != len(self.parts) or points.shape[-1] != 3:
            raise ValueError("Expected [part,node,3] surface nodes")
        if normals.shape != points.shape or areas.shape != points.shape[:-1]:
            raise ValueError("Surface atlas dimensions disagree")
        if not torch.isfinite(points).all() or not torch.isfinite(normals).all() or not torch.isfinite(areas).all():
            raise ValueError("Nonfinite surface atlas")
        if (areas <= 0).any() or not torch.allclose(normals.norm(dim=-1), torch.ones_like(areas), atol=1e-5):
            raise ValueError("Surface nodes require positive area and unit normals")
        self.register_buffer("points_local", points.detach().clone())
        self.register_buffer("normals_local", normals.detach().clone())
        self.register_buffer("areas", areas.detach().clone())
        self.provenance = dict(provenance or {})
        digest = hashlib.sha256()
        for value in (points, normals, areas):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(repr(self.parts).encode())
        self.fingerprint = digest.hexdigest()

    def world(self, positions, rotations):
        return positions[..., None, :] + torch.einsum("...pij,pkj->...pki", rotations, self.points_local)

    @classmethod
    def from_surfaces(cls, triangles, spheres, *, parts, samples_per_part=128, provenance=None):
        """Deterministic area quadrature of supplied realized collision surfaces.

        Mesh nodes use area-stratified triangle sampling. Every primitive gets
        nodes even if small. This does not add or move physical colliders.
        """
        if samples_per_part < 8 or len(triangles) != len(parts) or len(spheres) != len(parts):
            raise ValueError("Invalid surface quadrature configuration")
        points, normals, areas = [], [], []
        for ts, ss in zip(triangles, spheres):
            ts = np.asarray(ts, dtype=np.float64).reshape(-1, 3, 3)
            ss = np.asarray(ss, dtype=np.float64).reshape(-1, 4)
            if not np.isfinite(ts).all() or not np.isfinite(ss).all() or np.any(ss[:, 3] <= 0):
                raise ValueError("Invalid actual collision surface geometry")
            cross = np.cross(ts[:, 1]-ts[:, 0], ts[:, 2]-ts[:, 0])
            triangle_area = np.linalg.norm(cross, axis=-1)*.5
            if np.any(triangle_area <= 0):
                raise ValueError("Degenerate collision triangle")
            component_area = ([triangle_area.sum()] if len(ts) else []) + list(4*np.pi*ss[:, 3]**2)
            component_area = np.asarray(component_area)
            if not len(component_area) or 4*len(component_area) > samples_per_part:
                raise ValueError("Insufficient nodes for collision surface components")
            allocation = np.full(len(component_area), 4, dtype=int)
            budget = samples_per_part-allocation.sum()
            desired = component_area/component_area.sum()*budget
            allocation += np.floor(desired).astype(int)
            remainder = samples_per_part-allocation.sum()
            allocation[np.argsort(-(desired-np.floor(desired)), kind="stable")[:remainder]] += 1
            pp, nn_, aa = [], [], []
            offset = 0
            if len(ts):
                size = allocation[0]
                cdf = np.cumsum(triangle_area)
                ids = np.searchsorted(cdf, (np.arange(size)+.5)/size*cdf[-1])
                i = np.arange(size)+.5
                u, v = np.mod(i*.7548776662466927, 1.), np.mod(i*.5698402909980532, 1.)
                root = np.sqrt(u)
                bary = np.stack((1-root, root*(1-v), root*v), -1)
                pp.extend(np.einsum("ni,nij->nj", bary, ts[ids]))
                nn_.extend(cross[ids]/(2*triangle_area[ids, None]))
                aa.extend(np.full(size, component_area[0]/size))
                offset = 1
            for sphere, size, area in zip(ss, allocation[offset:], component_area[offset:]):
                i = np.arange(size)+.5
                z = 1-2*i/size
                angle = i*np.pi*(3-np.sqrt(5))
                radius = np.sqrt(1-z*z)
                direction = np.stack((radius*np.cos(angle), radius*np.sin(angle), z), -1)
                pp.extend(sphere[:3]+sphere[3]*direction)
                nn_.extend(direction)
                aa.extend(np.full(size, area/size))
            points.append(pp); normals.append(nn_); areas.append(aa)
        return cls(torch.tensor(np.asarray(points), dtype=torch.float32),
                   torch.tensor(np.asarray(normals), dtype=torch.float32),
                   torch.tensor(np.asarray(areas), dtype=torch.float32), parts=parts, provenance=provenance)

    def project(self, view, positions, rotations, *, bandwidth):
        """Map real point-force pairs to nodes; report the moment residual.

        positions/rotations MUST be the part poses at constraint evaluation,
        never the post-integration trajectory pose. Unknown poses must be
        reported by the caller instead of projecting with a different state.
        Values are forces in N, not calibrated pressure or physical occupancy.
        """
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Surface projection requires positive bandwidth")
        nenv, npart = view.evidence.known.shape
        if tuple(positions.shape) != (nenv, npart, 3) or tuple(rotations.shape) != (nenv, npart, 3, 3):
            raise ValueError("Missing solve-geometry part poses")
        if tuple(self.parts) and len(self.parts) != npart:
            raise ValueError("Surface/part mapping mismatch")
        if not torch.isfinite(positions).all() or not torch.isfinite(rotations).all():
            raise ValueError("Nonfinite solve-geometry part poses")
        world = self.world(positions, rotations).reshape(nenv*npart, -1, 3)
        center = positions.reshape(-1, 3)
        candidates = world[view.groups]
        distance = (candidates-view.position_w[:, None]).square().sum(-1)
        prior = self.areas[view.groups % npart]
        weights = torch.softmax(-distance/(2*bandwidth**2)+prior.log(), -1)
        force = positions.new_zeros(nenv*npart, self.areas.shape[1], 3)
        evidence_mass = positions.new_zeros(nenv*npart, self.areas.shape[1])
        force.index_add_(0, view.groups, weights[..., None]*view.force_w[:, None])
        evidence_mass.index_add_(0, view.groups, weights)
        raw_moment = positions.new_zeros(nenv*npart, 3)
        point_moment = torch.linalg.cross(view.position_w-center[view.groups], view.force_w)
        if view.torque_w is not None:
            point_moment = point_moment+view.torque_w
        raw_moment.index_add_(0, view.groups, point_moment)
        grid_moment = torch.linalg.cross(world-center[:, None], force).sum(1)
        # Carry the residual explicitly: summing node forces alone cannot in
        # general preserve the contact wrench on a finite surface quadrature.
        return dict(force_w=force.reshape(nenv, npart, -1, 3),
                    evidence_mass=evidence_mass.reshape(nenv, npart, -1),
                    known=view.evidence.known, active=view.reduce()["active"],
                    moment_residual_w=(raw_moment-grid_moment).reshape(nenv, npart, 3),
                    torque_known=view.torque_w is not None)
