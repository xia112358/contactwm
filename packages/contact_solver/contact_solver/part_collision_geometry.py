"""Compact physical geometry observations for the six contact-part tokens."""
import itertools
import xml.etree.ElementTree as ET

import torch
from torch import nn

from somaforge_core.robot_assets import canonical_g1_urdf_path
from contact_solver.contact_constrained_projector import CanonicalContactCollisionGeometry
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_RESOLUTION_M


class PartCollisionGeometry(nn.Module):
    """Summarize complete canonical contact shapes; never decide contact.

    Meshes retain all source vertices for support extrema, and spheres use
    their analytical support points. Only 19 scalars per part reach the
    network. All grouped shapes must be rigidly attached to that part's FK
    reference link. Geometry of other body parts is not silently substituted.
    """
    feature_count = 19
    feature_schema = 'canonical_contact_part_support19_v1'

    def __init__(self, fk):
        super().__init__()
        geometry = CanonicalContactCollisionGeometry()
        root = ET.parse(canonical_g1_urdf_path()).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        for shape in geometry.shapes:
            if shape.kind not in ('mesh', 'sphere'):
                raise ValueError(f'Exact support extraction is not implemented for {shape.kind}')
            link, reference = shape.link_name, BODY_NAMES[shape.part+1]
            while link != reference:
                joint = by_child.get(link)
                if joint is None or joint.get('type') != 'fixed':
                    raise ValueError(f'Contact shape {shape.link_name} is not rigid relative to {reference}')
                link = joint.find('parent').get('link')
        axes = torch.cat((torch.eye(3), -torch.eye(3)))
        diagonal = torch.tensor(list(itertools.product((-1., 1.), repeat=3))) / 3**.5
        directions = torch.cat((axes, diagonal))
        q = fk.origin_xyz.new_zeros(1, 36); q[:, 3] = 1
        names = tuple(shape.link_name for shape in geometry.shapes)
        with torch.no_grad():
            position, rotation = fk.link_poses(q, names)
            ref_position, ref_rotation = fk.link_poses(q, BODY_NAMES[1:7])
            vertices, centers, radii = [[] for _ in range(6)], [[] for _ in range(6)], [[] for _ in range(6)]
            for i, shape in enumerate(geometry.shapes):
                part = shape.part
                transform = ref_rotation[0, part].T @ rotation[0, i]
                offset = ref_rotation[0, part].T @ (position[0, i]-ref_position[0, part])
                if shape.kind == 'sphere':
                    centers[part].append(transform @ shape.local_center.to(q)+offset)
                    radii[part].append(shape.radius)
                else:
                    vertices[part].append(shape.local_points.to(q) @ transform.T+offset)
            supports = []
            self.asset_counts = []
            for part in range(6):
                v = torch.cat(vertices[part]) if vertices[part] else q.new_empty(0, 3)
                c = torch.stack(centers[part]) if centers[part] else q.new_empty(0, 3)
                r = q.new_tensor(radii[part])
                candidates = []
                if len(v): candidates.append((v @ directions.to(q).T).amax(0))
                if len(c): candidates.append((c @ directions.to(q).T+r[:, None]).amax(0))
                if not candidates: raise ValueError(f'No canonical collision shapes for part {part}')
                supports.append(torch.stack(candidates).amax(0))
                self.register_buffer(f'vertices_{part}', v, persistent=False)
                self.register_buffer(f'centers_{part}', c, persistent=False)
                self.register_buffer(f'radii_{part}', r, persistent=False)
                self.asset_counts.append({'vertices': len(v), 'spheres': len(c)})
            self.register_buffer('local_support', torch.stack(supports), persistent=False)
        self.register_buffer('support_directions', directions, persistent=False)

    def lowest_points(self, local_rotation):
        """Exact downward support of each grouped shape in the root-yaw frame."""
        offsets = []
        for part in range(6):
            v, c, r = (getattr(self, f'{name}_{part}') for name in ('vertices', 'centers', 'radii'))
            normal = local_rotation[:, part, 2, :]
            candidates = []
            if len(v):
                lowest = (normal @ v.T).argmin(-1)
                candidates.append(v[lowest, None])
            if len(c):
                candidates.append(c[None]-r[None, :, None]*normal[:, None])
            points = torch.cat(candidates, 1)
            lowest = (points*normal[:, None]).sum(-1).argmin(-1)
            chosen = points.gather(1, lowest[:, None, None].expand(-1, 1, 3)).squeeze(1)
            offsets.append(torch.einsum('bij,bj->bi', local_rotation[:, part], chosen))
        return torch.stack(offsets, 1)

    def forward(self, local_position, local_rotation, heightmap):
        offset = self.lowest_points(local_rotation)
        point = local_position+offset
        x = (point[..., 0]-HEIGHTMAP_FORWARD_MIN_M)/HEIGHTMAP_RESOLUTION_M
        y = (point[..., 1]-HEIGHTMAP_LATERAL_MIN_M)/HEIGHTMAP_RESOLUTION_M
        visible = (x >= 0) & (x <= HEIGHTMAP_ROWS-1) & (y >= 0) & (y <= HEIGHTMAP_COLS-1)
        cell = x.round().long().clamp(0, HEIGHTMAP_ROWS-1)*HEIGHTMAP_COLS+y.round().long().clamp(0, HEIGHTMAP_COLS-1)
        gap = point[..., 2]-heightmap.flatten(1).gather(1, cell)
        gap = torch.where(visible, gap, 0)
        # 10 cm is a feature scale, not a collision/contact threshold.
        return torch.cat((self.local_support[None].expand(len(point), -1, -1)/.1,
                          offset/.1, gap[..., None]/.1, visible[..., None].to(point)), -1)
