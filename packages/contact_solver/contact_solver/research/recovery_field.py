"""Differentiable box recovery geometry, never a contact classifier.

This deliberately small research module supports one verified closed OBB plus
floor. Mesh surface samples are area-uniform, with exact vertex support values
kept separately for whole-shape separation and independent audits.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

from somaforge_core.robot_assets import canonical_g1_urdf_path


def box_sdf(points, center, basis, half):
    """Exact Euclidean OBB SDF (positive outside), differentiable almost everywhere."""
    distance = ((points - center) @ basis).abs() - half
    return distance.clamp_min(0).norm(dim=-1) + distance.amax(-1).clamp_max(0)


def pose_increment(seed, x, scale):
    """35D local chart: world translation/rotation followed by canonical joints."""
    d = x * scale
    a = torch.cat((d.new_ones(1), d[3:6] * .5))
    a = a / a.norm()
    b = seed[3:7]
    quat = torch.cat(((a[0] * b[0] - a[1:] @ b[1:]).reshape(1),
                      a[0] * b[1:] + b[0] * a[1:] + torch.cross(a[1:], b[1:], dim=0)))
    return torch.cat((seed[:3] + d[:3], quat / quat.norm(), seed[7:] + d[6:]))


class RecoveryGeometry:
    def __init__(self, terrain, fk, *, device='cpu', samples=512):
        self.fk = fk
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
        mesh = trimesh.load(terrain, force='mesh')
        if not mesh.is_watertight or not mesh.is_convex or len(mesh.facets_normal) != 6:
            raise ValueError('Research fixture requires an actual closed six-face box')
        normals = np.asarray(mesh.facets_normal)
        centers = np.asarray([mesh.triangles_center[f].mean(0) for f in mesh.facets])
        offsets = (normals * centers).sum(1)
        top = int(np.argmax(normals[:, 2]))
        if not np.allclose(normals[top], [0, 0, 1], atol=1e-6):
            raise ValueError('Fixed fixture requires a horizontal top')
        side = normals[np.flatnonzero(abs(normals[:, 2]) < 1e-6)[0]]
        basis = np.stack((side, np.cross(normals[top], side), normals[top]), axis=1)
        local = np.asarray(mesh.vertices) @ basis
        low, high = local.min(0), local.max(0)
        self.center, self.half, self.basis = tensor(((low + high) / 2) @ basis.T), tensor((high - low) / 2), tensor(basis)
        if not np.allclose(np.max(abs(normals @ basis), axis=1), 1, atol=1e-6):
            raise ValueError('Terrain is not an orthogonal box')
        self.normal = tensor(np.concatenate((normals, [[0, 0, 1]])))
        self.offset = tensor(np.r_[offsets, 0.])
        self.top, self.ground = top, 6
        if float(box_sdf(tensor(mesh.vertices), self.center, self.basis, self.half).abs().max()) > 1e-6:
            raise ValueError('Analytic OBB does not reproduce actual terrain vertices')
        asset = canonical_g1_urdf_path()
        self.shapes, self.link_names = [], []
        rng = np.random.default_rng(14)
        for link in ET.parse(asset).getroot().findall('link'):
            name = link.get('name')
            for collision in link.findall('collision'):
                geom = list(collision.find('geometry'))[0]
                origin = collision.find('origin')
                transform = np.eye(4)
                if origin is not None:
                    transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
                    transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                shape = dict(name=name, kind=geom.tag, rotation=tensor(transform[:3, :3]), center=tensor(transform[:3, 3]))
                if geom.tag == 'mesh':
                    source = trimesh.load(asset.parent / geom.get('filename'), force='mesh')
                    vertices = np.asarray(source.vertices).copy() * np.fromstring(geom.get('scale', '1 1 1'), sep=' ')
                    triangles = vertices[np.asarray(source.faces)]
                    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
                    chosen = rng.choice(len(triangles), samples, p=area / area.sum())
                    uv = rng.random((samples, 2)); u = np.sqrt(uv[:, :1]); v = uv[:, 1:]
                    points = (1-u)*triangles[chosen, 0] + u*(1-v)*triangles[chosen, 1] + u*v*triangles[chosen, 2]
                    shape['vertices'] = tensor(vertices @ transform[:3, :3].T + transform[:3, 3])
                    shape['hull'] = shape['vertices'][ConvexHull(vertices).vertices]
                elif geom.tag == 'sphere':
                    radius = float(geom.get('radius')); shape['radius'] = radius
                    points = rng.normal(size=(samples, 3)); points = radius * points / np.linalg.norm(points, axis=1, keepdims=True)
                elif geom.tag == 'cylinder':
                    radius, half = float(geom.get('radius')), .5 * float(geom.get('length'))
                    shape.update(radius=radius, half=half)
                    angle = rng.uniform(0, 2*np.pi, samples)
                    caps = rng.random(samples) < radius / (radius + 2*half)
                    rho = np.where(caps, radius*np.sqrt(rng.random(samples)), radius)
                    z = np.where(caps, rng.choice([-half, half], samples), rng.uniform(-half, half, samples))
                    points = np.stack((rho*np.cos(angle), rho*np.sin(angle), z), axis=1)
                else:
                    raise ValueError(f'Unsupported canonical collision shape: {geom.tag}')
                shape['points'] = tensor(points @ transform[:3, :3].T + transform[:3, 3])
                if name not in self.link_names: self.link_names.append(name)
                shape['link'] = self.link_names.index(name)
                self.shapes.append(shape)
        self.link_names = tuple(self.link_names)
        self.local_points = torch.stack([s['points'] for s in self.shapes])
        self.shape_links = torch.tensor([s['link'] for s in self.shapes], device=device)
        self.foot_ids = [i for i, s in enumerate(self.shapes) if s['name'].startswith('left_ankle')]

    def evaluate(self, q):
        p, r = self.fk.link_poses(q[None], self.link_names)
        world = torch.einsum('sij,spj->spi', r[0, self.shape_links], self.local_points) + p[0, self.shape_links, None]
        sep = []
        for shape in self.shapes:
            i = shape['link']; local = self.normal @ r[0, i]
            if shape['kind'] == 'mesh':
                value = (shape['hull'] @ local.T).amin(0) + self.normal @ p[0, i] - self.offset
            else:
                center = p[0, i] + r[0, i] @ shape['center']
                direction = local @ shape['rotation']
                support = shape['radius'] if shape['kind'] == 'sphere' else shape['radius'] * direction[:, :2].norm(dim=1) + shape['half'] * direction[:, 2].abs()
                value = self.normal @ center - self.offset - support
            sep.append(value)
        return world, torch.stack(sep)

    def sdf(self, points):
        return box_sdf(points, self.center, self.basis, self.half)

    def foot_audit(self, q):
        p, r = self.fk.link_poses(q[None], self.link_names)
        vertices = [s['vertices'] @ r[0, s['link']].T + p[0, s['link']] for s in self.shapes
                    if s['name'] == 'left_ankle_roll_link' and s['kind'] == 'mesh']
        world = torch.cat(vertices)
        phi = self.sdf(world)
        return dict(inside_vertex_count=int((phi < 0).sum()), max_vertex_nearest_exit_cm=float((-phi).relu().max())*100,
                    center_world=world.mean(0).tolist())
