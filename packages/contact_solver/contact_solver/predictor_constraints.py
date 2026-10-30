"""Batched canonical-G1 box/ground adapter for predictor training.

The optimization engine is robot independent. This adapter explicitly supports
this dataset's box/ground scenes and canonical mesh/sphere assets. Shape plane
separation is conservative, not a general mesh SDF or a contact label.
"""
import xml.etree.ElementTree as ET
import torch
from somaforge_core.robot_assets import canonical_g1_urdf_path
from somaforge_core.g1_kinematics import _rpy_matrix
from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.motion_contracts import BODY_NAMES
from .collision_geometry import _mesh_vertices, _cylinder_points
from .constraint_learning import Residual
from .constraint_training import ConstraintBatch
from .interaction_acceptance import DEFAULT_ACCEPTANCE


class CanonicalBoxGeometry:
    def __init__(self):
        path = canonical_g1_urdf_path()
        self.shapes = []
        parts = {link: i for i, part in enumerate(('left_foot', 'right_foot', 'left_hand', 'right_hand', 'left_knee', 'right_knee')) for link in CONTACT_BODY_NAMES_BY_PART[part]}
        for link in ET.parse(path).getroot().findall('link'):
            for index, collision in enumerate(link.findall('collision')):
                node = collision.find('geometry')
                if node is None or len(node) != 1: raise ValueError('Unknown collision geometry')
                shape = node[0]; origin = collision.find('origin')
                xyz = torch.tensor([float(v) for v in ('0 0 0' if origin is None else origin.get('xyz', '0 0 0')).split()])
                rotation = _rpy_matrix(tuple(float(v) for v in ('0 0 0' if origin is None else origin.get('rpy', '0 0 0')).split()))
                radius = 0.; cylinder = None
                if shape.tag == 'mesh':
                    points = _mesh_vertices(path.parent/shape.get('filename'), tuple(float(v) for v in shape.get('scale','1 1 1').split()), 2**31-1)
                    points = points@rotation.T+xyz
                elif shape.tag == 'sphere':
                    points = xyz[None]; radius = float(shape.get('radius'))
                elif shape.tag == 'cylinder':
                    cylinder = (xyz, rotation[:, 2], float(shape.get('radius')), float(shape.get('length')))
                    points = _cylinder_points(cylinder[2], cylinder[3], 128)@rotation.T+xyz
                else: raise ValueError(f'Unsupported full-shape geometry: {shape.tag}')
                self.shapes.append((link.get('name'), f'{link.get("name")}/{index}', parts.get(link.get('name'), -1), points, radius, cylinder))

    def evaluate(self, fk, q, scene):
        pos, rot = fk.link_poses(q, tuple(s[0] for s in self.shapes))
        center, basis, half = (scene[k].to(q) for k in ('box_center','box_rotation','box_half_extents'))
        ground = scene['ground_height'].to(q).reshape(-1)
        if not bool((half > 0).all()): raise ValueError('Box adapter requires positive box extents')
        clearances, top, floor = [], [], []
        for i, (_, _, _, points, radius, cylinder) in enumerate(self.shapes):
            world = torch.einsum('bij,pj->bpi', rot[:, i], points.to(q))+pos[:, i, None]
            local = torch.einsum('bpi,bij->bpj', world-center[:, None], basis)
            # Minimum support against each face, then maximum separating face.
            sep = torch.cat((local.amin(1)-half, -local.amax(1)-half), -1)-radius
            d = local.abs()-half[:, None]
            sdf = d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)-radius
            floor_gap = world[..., 2].amin(-1)-radius-ground
            if cylinder is not None:
                c, axis, r, length = cylinder
                cworld = torch.einsum('bij,j->bi', rot[:, i], c.to(q))+pos[:, i]
                aworld = torch.einsum('bij,j->bi', rot[:, i], axis.to(q))
                adot = torch.einsum('bi,bij->bj', aworld, basis).clamp(-1, 1)
                support = r*torch.linalg.cross(aworld[:, None].expand(-1, 3, -1), basis.transpose(1, 2)).norm(dim=-1)+.5*length*adot.abs()
                clocal = torch.einsum('bi,bij->bj', cworld-center, basis)
                sep = torch.cat((clocal-half-support, -clocal-half-support), -1)
                az = aworld[:, 2].clamp(-1, 1)
                floor_gap = cworld[:, 2]-ground-r*aworld[:, :2].norm(dim=-1)-.5*length*az.abs()
            clearances.append(torch.minimum(torch.minimum(sep.amax(-1), sdf.amin(-1)), floor_gap))
            top.append(sep[:, 2]); floor.append(floor_gap)
        return torch.stack(clearances, -1), torch.stack(top, -1), torch.stack(floor, -1)


class PredictorConstraintAdapter:
    def __init__(self, *, distance_scale=.01, root_relative_cost=100.):
        self.geometry = CanonicalBoxGeometry()
        self.scale = distance_scale
        self.root_relative_cost = root_relative_cost
        if distance_scale <= 0 or root_relative_cost < 0: raise ValueError('Invalid constraint scales')

    def __call__(self, model, rows, scene, *, active, surface, sample_ids,
                 context_ids, prior, nominal_q, payload=None, keep=None, keep_tolerance_m=.01):
        q, p = rows.q, rows.pair
        if len(sample_ids) != len(q) or len(context_ids) != len(q): raise ValueError('Missing sample identities')
        required = ('type','dist','includemargin','efc_address','constraint_rows','active','constraint_allocated','eligible','upward','task_pair')
        if any(k not in p for k in required): raise ValueError('Missing actual Newton contact fields')
        actual = ((p['type'].long() & 1) != 0) & (p['dist'] < p['includemargin'])
        address = p['efc_address'].reshape(len(p['dist']), -1)
        valid_rows = torch.arange(address.shape[1], device=q.device)[None] < p['constraint_rows'][:, None]
        if bool(((p['constraint_rows'] < 1) | (p['constraint_rows'] > address.shape[1])).any()):
            raise ValueError('Invalid Newton constraint row count')
        allocated = actual & ((address >= 0) | ~valid_rows).all(-1) & (p['constraint_rows'] > 0)
        if not torch.equal(actual, p['active']) or not torch.equal(allocated, p['constraint_allocated']):
            raise ValueError('Newton activation/allocation metadata mismatch')
        if bool((actual & ~allocated).any()): raise ValueError('Active Newton pair has no allocated constraints')
        if bool((p['eligible'] & ~(actual & allocated & p['task_pair'] & p['upward'])).any()):
            raise ValueError('Invalid primary-surface contact evidence')
        if bool(((p['full_kind'] >= 0) & ~rows.normal_valid).any()): raise ValueError('Unknown full-body normal')
        if keep is not None:
            with torch.no_grad():
                initial_pos, initial_rot = model.fk.link_poses(keep['q'].to(q), BODY_NAMES[1:7])
                material = torch.einsum('bpji,bpj->bpi', initial_rot, keep['anchor'].to(q)-initial_pos)
            pos, rot = model.fk.link_poses(q, BODY_NAMES[1:7])
            material_now = pos+torch.einsum('bpij,bpj->bpi', rot, material)
        distances, top, floor = self.geometry.evaluate(model.fk, q, scene)
        loss_rows, guards, evidence, diagnostics = [], [], {}, {}
        allowance = DEFAULT_ACCEPTANCE.shallow_penetration_m
        for i, (sample, context) in enumerate(zip(sample_ids, context_ids)):
            def row(name, value, scale=self.scale, ids=('value',)):
                return Residual.scaled(sample, context, name, 'le', value.reshape(-1), scale, component_ids=ids)
            full = (p['sample'] == i) & (p['full_kind'] >= 0)
            reported = rows.distances[full].amin().clamp_max(0) if bool(full.any()) else q[i].sum()*0
            worst = torch.minimum(reported, distances[i].amin())
            collision = row('collision/verified_fullbody', -allowance-worst)
            loss_rows.append(collision); guards.append(collision)
            shape_ids = tuple(s[1] for s in self.geometry.shapes)
            shape_row = row('collision/environment', -allowance-distances[i], ids=shape_ids)
            loss_rows.append(shape_row); guards.append(shape_row)
            own = (p['sample'] == i) & (p['full_kind'] == 1)
            # Raw shape IDs contain the query world's clone offset. Aggregate
            # by canonical link pair so shuffling batch slots preserves identity.
            if bool(own.any()):
                link0 = torch.minimum(p['body_link0'], p['body_link1'])
                link1 = torch.maximum(p['body_link0'], p['body_link1'])
                pairs = sorted(set(zip(link0[own].tolist(), link1[own].tolist())))
            else: pairs = []
            for a, b in pairs:
                selected = own & (link0 == a) & (link1 == b)
                self_row = row(f'collision/self/{a}/{b}', -allowance-rows.distances[selected].amin())
                loss_rows.append(self_row); guards.append(self_row)
            diagnostics[sample] = dict(verified_penetration_m=float((-worst).clamp_min(0).detach()),
                root_deviation_m=float((q[i, :3]-nominal_q[i, :3].to(q)).norm().detach()), contacts={})
            for part in active[i].nonzero().flatten().tolist():
                sid = int(surface[i, part])
                if sid not in (0, 1): raise ValueError('Box adapter only supports ground/top task surfaces')
                match = ((p['sample'] == i) & (p['part'] == part) & p['task_pair'] & p['upward'] & (p['primary_surface'] == sid))
                realized = bool((match & p['eligible']).any())
                key = (sample, context, f'contact/{part}/{sid}')
                evidence[key] = realized
                diagnostics[sample]['contacts'][f'{part}/{sid}'] = realized
                if bool(match.any()):
                    # Pair-specific margin; do not subtract a global guessed margin.
                    gap = (rows.distances[match]-p['includemargin'][match]+1e-6).amin()
                else:
                    shapes = [j for j,s in enumerate(self.geometry.shapes) if s[2] == part]
                    if not shapes: raise ValueError('Missing canonical part geometry')
                    # Guidance only: it never creates evidence of activation.
                    gap = (floor if sid == 0 else top)[i, shapes].amin()-rows.observed['configured_margin'][i]+1e-6
                name = f'contact/{part}/{sid}/upper'
                loss_rows.append(row(name, gap*0 if realized else gap))
                guards.append(row(name, gap))
                if keep is not None and bool(keep['mask'][i, part]):
                    faces = (rows.observed['face_surface'][i] == sid).nonzero().flatten()
                    if len(faces) != 1: raise ValueError('Missing or ambiguous kept-contact face')
                    normal = rows.observed['face_normal'][i, faces[0]].to(q)
                    if rows.world_frame is not None: normal = rows.world_frame[1][i].to(q).T@normal
                    delta = material_now[i, part]-keep['anchor'][i, part].to(q)
                    tangent = delta-normal*(delta@normal)
                    kept = row(f'keep/{part}/{sid}/tangent', tangent.norm()-keep_tolerance_m)
                    loss_rows.append(kept); guards.append(kept)
            # Extra activation has its own task acceptance tolerance. These
            # rows are objectives, not geometric declarations of contact.
            for part in range(active.shape[1]):
                extra = ((p['sample'] == i) & (p['part'] == part) & p['task_pair'] & p['upward'] &
                         (~active[i, part] | (p['primary_surface'] != surface[i, part])))
                value = ((1-DEFAULT_ACCEPTANCE.extra_activation_margin_fraction)*p['includemargin'][extra]-rows.distances[extra]).amax() if bool(extra.any()) else q[i].sum()*0
                loss_rows.append(row(f'release/{part}', value))
            lower = row('joint/lower', model.fk.joint_lower.to(q)-q[i, 7:], 1., tuple(str(j) for j in range(q.shape[-1]-7)))
            upper = row('joint/upper', q[i, 7:]-model.fk.joint_upper.to(q), 1., lower.component_ids)
            loss_rows.extend((lower, upper)); guards.extend((lower, upper))
        # Root translation and rotation deviation are explicit priors; quaternion
        # sign is aligned to avoid penalizing equivalent orientations.
        nominal = nominal_q.detach().to(q)
        sign = torch.where((q[:, 3:7]*nominal[:, 3:7]).sum(-1, keepdim=True) < 0, -1., 1.)
        root = ((q[:, :3]-nominal[:, :3])/.1).square().sum()+((q[:, 3:7]*sign-nominal[:, 3:7])/.5).square().sum()
        return ConstraintBatch(prior+self.root_relative_cost*.5*root, loss_rows, guards, evidence, len(q), payload, diagnostics)
