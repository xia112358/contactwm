"""Device contact observations from the actual fixed-q Newton/MJWarp pass.

No distance threshold, pose projection, integration, or predicted intent is used.
Source-feature projection only attributes a witness to its actual incident face.
The caller must check ``errors`` before consuming observations. Tensor views
of solver fields are valid only until the next query; derived tensors own data.
"""
from __future__ import annotations

import numpy as np
import torch
from .contact_source_geometry import project_source_triangles_tensor, incident_witness_faces_tensor


def validate_observed(observed):
    """Verify compact actual solver decisions before consuming device pairs."""
    p = observed['pairs']
    keys = ('type', 'dist', 'includemargin', 'efc_address', 'constraint_rows', 'active',
            'constraint_allocated', 'eligible', 'primary_surface', 'part')
    if any(k not in p for k in keys):
        raise ValueError('Missing actual solver contact fields')
    dist, margin, kind, address, rows = (p[k] for k in
        ('dist', 'includemargin', 'type', 'efc_address', 'constraint_rows'))
    if dist.ndim != 1 or margin.shape != dist.shape or kind.shape != dist.shape:
        raise ValueError('Invalid Newton contact field shapes')
    if address.ndim != 2 or address.shape[0] != len(dist) or address.shape[1] < 1:
        raise ValueError('Invalid Newton constraint address shape')
    if rows.shape != dist.shape or p['active'].shape != dist.shape or p['constraint_allocated'].shape != dist.shape:
        raise ValueError('Invalid Newton contact decision/constraint row shapes')
    active = ((kind & 1) != 0) & (dist < margin)
    required = torch.arange(address.shape[1], device=address.device)[None] < rows[:, None]
    allocated = active & ((address >= 0) | ~required).all(-1)
    flags = torch.stack((~(torch.isfinite(dist).all() & torch.isfinite(margin).all()),
        ((rows < 1) | (rows > address.shape[1])).any(),
        ((active != p['active']) | (allocated != p['constraint_allocated'])).any(),
        (active & ~allocated).any())).detach().cpu().tolist()
    for invalid, message in zip(flags, ('Nonfinite Newton contact distance/margin',
        'Invalid expected contact constraint row count', 'Actual Newton activation/allocation disagreement',
        'Unallocated active Newton contact')):
        if invalid:
            raise ValueError(message)


class NewtonContactTensorReader:
    def __init__(self, model, solver, body_env, shape_surface):
        import warp as wp
        from .contact_schema import CONTACT_BODY_NAMES_BY_PART
        from .contact_face_selection import upward_face_mask
        from .newton_contacts import PARTS
        self.model, self.solver = model, solver
        self.device = str(model.device)
        self.worlds = max(1, int(model.world_count))
        self.body_env = torch.full((model.body_count,), -1, device=self.device, dtype=torch.long)
        self.body_part = torch.full_like(self.body_env, -1)
        self.link_names = tuple(sorted({str(model.body_label[b]).rsplit('/', 1)[-1] for b in body_env}))
        link_index = {name: i for i, name in enumerate(self.link_names)}
        self.body_link = torch.full_like(self.body_env, -1)
        for body, world in body_env.items():
            self.body_env[body] = world
            name = str(model.body_label[body]).rsplit('/', 1)[-1]
            self.body_link[body] = link_index[name]
            parts = [i for i, part in enumerate(PARTS) if name in CONTACT_BODY_NAMES_BY_PART[part]]
            if len(parts) > 1:
                raise ValueError('Ambiguous body/part mapping')
            if parts:
                self.body_part[body] = parts[0]
        if len(shape_surface) != 1:
            raise ValueError('Device normal-fan reader currently requires one actual shared terrain mesh')
        self.terrain_shape, faces = next(iter(shape_surface.items()))
        self.faces = sorted(faces, key=lambda face: int(face['surface']))
        self.surface_ids = torch.tensor([f['surface'] for f in self.faces], device=self.device)
        normals = np.asarray([f['normal_w'] for f in self.faces], dtype=np.float64)
        self.normals = torch.tensor(normals, device=self.device)
        vertices_by_face = [np.asarray(f['triangles_w']).reshape(-1, 3) for f in self.faces]
        self.face_offsets = torch.tensor([v[0]@n for v,n in zip(vertices_by_face,normals)], device=self.device)
        self.face_extents = torch.tensor([np.ptp(v,axis=0).max() for v in vertices_by_face], device=self.device)
        self.upward = torch.tensor(upward_face_mask(normals), device=self.device)
        ntri = max(i for f in self.faces for i in f['triangle_indices'])+1
        triangles = np.zeros((ntri, 3, 3), np.float64)
        present = np.zeros(ntri, bool)
        incident = np.zeros((ntri, 3, len(faces)), bool)
        source_face = np.zeros((ntri, len(faces)), bool)
        vertices = [{tuple(v) for t in f['triangles_w'] for v in t} for f in self.faces]
        for fi, face in enumerate(self.faces):
            for ti, triangle in zip(face['triangle_indices'], face['triangles_w']):
                if present[ti]:
                    raise ValueError('Duplicate source triangle in native surface binding')
                present[ti] = True
                triangles[ti] = triangle
                source_face[ti, fi] = True
                for vi, vertex in enumerate(triangle):
                    incident[ti, vi] = [tuple(vertex) in face_vertices for face_vertices in vertices]
        self.triangles = torch.tensor(triangles, device=self.device)
        self.present = torch.tensor(present, device=self.device)
        self.incident = torch.tensor(incident, device=self.device)
        self.source_face = torch.tensor(source_face, device=self.device)
        self.shape_types = wp.to_torch(model.shape_type)

    def read(self, state, contacts, capture):
        import warp as wp
        solver, model = self.solver, self.model
        contact = solver.mjw_data.contact
        get = lambda name: wp.to_torch(getattr(contact, name))
        dist, margin, kind = get('dist'), get('includemargin'), get('type')
        address, world, geom, dim = get('efc_address'), get('worldid').long(), get('geom').long(), get('dim')
        count = wp.to_torch(solver.mjw_data.nacon).reshape(-1)[0]
        row = torch.arange(len(dist), device=self.device)
        valid = row < count
        cone = int(solver.mjw_model.opt.cone)
        if cone not in (0, 1):
            raise ValueError('Unknown actual solver cone')
        required_rows = torch.where(dim == 1, 1, 2*(dim-1)) if cone == 0 else dim
        required = torch.arange(address.shape[1], device=self.device)[None] < required_rows[:, None]
        active = valid & ((kind & 1) != 0) & (dist < margin)
        allocated = active & ((address >= 0) | ~required).all(-1)
        mapping = wp.to_torch(solver.mjc_geom_to_newton_shape)
        mapped_valid = (world >= 0) & (world < mapping.shape[0]) & ((geom >= 0) & (geom < mapping.shape[1])).all(-1)
        shapes = mapping[world.clamp(0, mapping.shape[0]-1)[:, None], geom.clamp(0, mapping.shape[1]-1)].long()
        shape_valid = ((shapes >= 0) & (shapes < model.shape_count)).all(-1)
        shapes = shapes.clamp(0, model.shape_count-1)
        bodies = wp.to_torch(model.shape_body)[shapes].long()
        robot = (bodies >= 0) & (self.body_env[bodies.clamp_min(0)] >= 0)
        terrain_pair = valid & robot[:, 1] & ~robot[:, 0] & (shapes[:, 0] == self.terrain_shape)
        parts = self.body_part[bodies[:, 1].clamp_min(0)]
        same_world = self.body_env[bodies[:, 1].clamp_min(0)] == world
        task = terrain_pair & (parts >= 0) & same_world & ((kind & 1) != 0)
        raw_count = wp.to_torch(contacts.rigid_contact_count)[0]
        raw_capacity = min(len(capture.keys), len(solver._contact_tid_to_cid))
        raw_keys = wp.to_torch(capture.keys)[:raw_capacity]
        tid = torch.arange(len(raw_keys), device=self.device)
        raw_valid = tid < raw_count
        cid = wp.to_torch(solver._contact_tid_to_cid).long()[:len(raw_keys)]
        emitted = raw_valid & (cid >= 0)
        cid_valid = (cid >= 0) & (cid < len(dist)) & (cid < count)
        shape_bits, sub_bits = capture.shape_bits, capture.sub_key_bits
        if not (1 <= shape_bits <= 20 and 1 <= sub_bits <= 23):
            raise ValueError('Unsupported Newton contact source-key layout')
        mask = (1 << shape_bits)-1
        a = (raw_keys >> (sub_bits+shape_bits)) & mask
        b = (raw_keys >> sub_bits) & mask
        sub = raw_keys & ((1 << sub_bits)-1)
        analytic = torch.zeros_like(raw_valid)
        if capture.analytic_sphere_types is not None:
            analytic = (self.shape_types[a.clamp(0, model.shape_count-1)] == 8) & (self.shape_types[b.clamp(0, model.shape_count-1)] == 3)
        keys = (a << 43) | (b << 23) | torch.where(analytic, sub << 3, sub)
        raw_shapes = torch.stack((wp.to_torch(contacts.rigid_contact_shape0)[:raw_capacity],
                                  wp.to_torch(contacts.rigid_contact_shape1)[:raw_capacity]), -1).long()
        source_error = raw_valid & ((raw_keys < 0) | (a != raw_shapes[:, 0]) | (b != raw_shapes[:, 1]) |
            ((raw_keys >> (sub_bits+2*shape_bits)) != 0) | (analytic & (((sub & 1) != 1) | (sub >= (1 << 20)))))
        # Extra sink row absorbs unused raw capacity without aliasing real rows.
        destination = torch.where(emitted & cid_valid, cid, len(dist))
        source = torch.full((len(dist)+1,), -1, device=self.device, dtype=torch.long)
        source.scatter_reduce_(0, destination, keys, reduce='amax', include_self=True)
        coverage = torch.zeros(len(dist)+1, device=self.device, dtype=torch.long)
        coverage.scatter_add_(0, destination, emitted.long())
        source = source[:-1]
        geometry_points = []
        transforms = wp.to_torch(state.body_q).reshape(-1, 7) if state is not None else None
        if transforms is not None:
            shape_margin = wp.to_torch(model.shape_margin)
            for side in (0, 1):
                raw_point = wp.to_torch(getattr(contacts, f'rigid_contact_point{side}'))[:raw_capacity]
                raw_offset = wp.to_torch(getattr(contacts, f'rigid_contact_offset{side}'))[:raw_capacity]
                raw_margin = wp.to_torch(getattr(contacts, f'rigid_contact_margin{side}'))[:raw_capacity]
                raw_shape = raw_shapes[:, side].clamp(0, model.shape_count-1)
                raw_body = wp.to_torch(model.shape_body)[raw_shape].long()
                scale = torch.where(raw_margin != 0, (raw_margin-shape_margin[raw_shape]) /
                    torch.where(raw_margin != 0, raw_margin, 1), 0)
                local = (raw_point+raw_offset*scale[:, None]).double()
                transform = transforms[raw_body.clamp_min(0)].double()
                quat = torch.nn.functional.normalize(transform[:, 3:], dim=-1)
                # Same normalization and double precision rotation as the
                # audit reader's scipy Rotation.apply, then float32 storage.
                cross = 2*torch.linalg.cross(quat[:, :3], local)
                moved = local+quat[:, 3:]*cross+torch.linalg.cross(quat[:, :3], cross)+transform[:, :3]
                point = torch.where((raw_body >= 0)[:, None], moved, local).float()
                output = torch.zeros((len(dist)+1, 3), device=self.device)
                output.index_add_(0, destination, torch.where(emitted[:, None], point, 0))
                geometry_points.append(output[:-1])
        tri = (source & 0x7fffff) >> 4
        triangle_valid = (tri >= 0) & (tri < len(self.triangles)) & ((source & 8) != 0)
        ti = tri.clamp(0, len(self.triangles)-1)
        normal = get('frame')[:, 0].to(torch.float64)
        triangle = self.triangles[ti]
        score = ((triangle-triangle[:, :1])*normal[:, None]).sum(-1)
        support = score == score.amax(-1, keepdim=True)
        candidates = (self.incident[ti] & support[:, :, None]).any(1) | self.source_face[ti]
        if not geometry_points:
            raise ValueError('Actual terrain geometry witness required for source-feature face attribution')
        projected, feature_valid = project_source_triangles_tensor(geometry_points[0].double(), triangle)
        candidates &= incident_witness_faces_tensor(projected,self.normals,self.face_offsets,self.face_extents)
        alignment = normal @ self.normals.T
        face = alignment.masked_fill(~candidates, -torch.inf).argmax(-1)
        surface = self.surface_ids[face]
        eligible = task & active & allocated & triangle_valid & self.present[ti] & self.upward[face]
        errors = dict(
            capacity=(count < 0) | (count > len(dist)) | (raw_count > len(raw_keys)),
            fields=(valid & (~torch.isfinite(dist) | ~torch.isfinite(margin) | (required_rows < 1) | (required_rows > address.shape[1]))).any(),
            mapping=(valid & (~mapped_valid | ~shape_valid)).any(),
            unallocated=(active & ~allocated).any(),
            source=source_error.any() | (emitted & ~cid_valid).any() | (valid & (coverage[:-1] != 1)).any(),
            raw_mapping=(emitted & (raw_shapes != shapes[cid.clamp(0, len(dist)-1)]).any(-1)).any(),
            triangle=(task & (~triangle_valid | ~self.present[ti] | ~torch.isfinite(normal).all(-1) | (normal.norm(dim=-1) == 0))).any(),
            source_feature=(task & (~feature_valid | ~candidates.any(-1))).any(),
            cross_world=(terrain_pair & ~same_world).any(),
            unknown_external=(valid & ((kind & 1) != 0) & (robot.sum(-1) == 1) &
                ~((shapes[:, 0] == self.terrain_shape) & robot[:, 1])).any())
        group = (world*self.body_part.new_tensor(6)+parts).clamp(0, self.worlds*6-1)
        minimum = torch.full((self.worlds*6,), torch.inf, device=self.device)
        minimum.scatter_reduce_(0, group, torch.where(eligible, dist-margin, torch.inf), reduce='amin')
        winner = torch.full((self.worlds*6,), len(dist), device=self.device, dtype=torch.long)
        winner.scatter_reduce_(0, group, torch.where(eligible & (dist-margin == minimum[group]), row, len(dist)), reduce='amin')
        contact_mask = winner < len(dist)
        part_surface = torch.where(contact_mask, surface[winner.clamp_max(len(dist)-1)], -1)
        link = torch.where(bodies >= 0, self.body_link[bodies.clamp_min(0)], -1)
        both_same_world = robot.all(-1) & (self.body_env[bodies.clamp_min(0)] == world[:, None]).all(-1)
        full_kind = torch.where(valid & ((kind & 1) != 0) & terrain_pair & same_world, 0,
            torch.where(valid & ((kind & 1) != 0) & both_same_world, 1, -1))
        result = dict(count=count, dist=dist, includemargin=margin, type=kind, worldid=world,
            active=active, constraint_allocated=allocated, efc_address=address, constraint_rows=required_rows,
            shape0=shapes[:, 0], shape1=shapes[:, 1], body0=bodies[:, 0], body1=bodies[:, 1],
            source_key=source, primary_surface=surface, task_pair=task, eligible=eligible,
            part=parts, upward=self.upward[face], full_kind=full_kind,
            body_link0=link[:, 0], body_link1=link[:, 1], normal_w=get('frame')[:, 0],
            contact_part_mask=contact_mask.reshape(self.worlds, 6),
            contact_surface=part_surface.reshape(self.worlds, 6), errors=errors)
        if geometry_points:
            result.update(geometry_point0_w=geometry_points[0], geometry_point1_w=geometry_points[1])
            result['contact_position_w'] = torch.where(contact_mask[:, None],
                geometry_points[1][winner.clamp_max(len(dist)-1)], 0).reshape(self.worlds, 6, 3)
            errors['geometry'] = (valid & ~(torch.isfinite(geometry_points[0]).all(-1) &
                                           torch.isfinite(geometry_points[1]).all(-1))).any()
        return result

    @staticmethod
    def validate(result):
        """One synchronization checks all device error flags before consumption."""
        names = list(result['errors'])
        failed = torch.stack([result['errors'][name] for name in names]).cpu().tolist()
        if any(failed):
            raise ValueError('Invalid device Newton snapshot: '+', '.join(n for n, bad in zip(names, failed) if bad))
