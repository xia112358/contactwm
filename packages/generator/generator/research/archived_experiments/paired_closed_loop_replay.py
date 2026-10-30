"""Preserved experiment helpers for regression tests; no training CLI."""
import numpy as np
import torch
from somaforge_core.newton_contact_query import query_contacts
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.g1_kinematics import _matrix_from_rotation6d
from generator.fullbody_dataset import _matrix_quaternion_wxyz
from generator.research.archived_experiments.contact_geometry import scene_faces
from generator.research.archived_experiments.paired_pose_transform import transform_q, yaw_basis
from generator.research.archived_experiments.contact_geometry import COLLISION_TOLERANCE_M

def observed_inputs(model, q, active, surface, anchor, scene):
    """Only current q, realized contacts and observed terrain enter forward."""
    q = np.asarray(q, np.float32)
    with torch.no_grad():
        (pos, rot) = model.fk(torch.tensor(q)[None, None])
    origin = pos[0, 0, 0].cpu().numpy()
    basis = yaw_basis(_matrix_quaternion_wxyz(_matrix_from_rotation6d(rot)[0, 0, 0].cpu().numpy()))
    local = dict(box_center=torch.tensor(basis.T @ (scene['box_center'][0].numpy() - origin), dtype=torch.float32)[None], box_rotation=torch.tensor(basis.T @ scene['box_rotation'][0].numpy(), dtype=torch.float32)[None], box_half_extents=scene['box_half_extents'], ground_height=scene['ground_height'] - float(origin[2]))
    local_anchor = torch.tensor((anchor - origin) @ basis, dtype=torch.float32)[None]
    x = dict(current_q=torch.tensor(transform_q(q, origin, basis, True))[None], current_contact=torch.tensor(active, dtype=torch.bool)[None], current_surface=torch.tensor(surface, dtype=torch.long)[None], current_anchor=torch.where(torch.tensor(active)[None, :, None], local_anchor, 0.0), faces=scene_faces(local))
    return (x, local, origin, basis)

def compatible_current(active, surface, bank_active, bank_surface):
    return (~active[None] | bank_active & (bank_surface == surface[None])).all(-1)

def successor_roles(active, surface, contract):
    end = contract['active_contact'].bool()
    end_surface = contract['contact_surface'].long()
    held = contract['persistent_support'].bool() & active & (surface == end_surface) & end
    touch = end & (contract['touchdown'].bool() | ~active | (surface != end_surface))
    role = torch.where(end, 3, 0)
    role = torch.where(held, 2, role)
    return torch.where(touch, 1, role)

def embedded_hold_depth(active, surface, anchor, scene, role):
    """Diagnostic inconsistency, NOT contact relabeling or a recovery action.

An embedded material point cannot be held fixed while its same collision
geometry escapes penetration. Reject this inherited hold target; do not invent
a touchdown or erase actual Newton activation to make it trainable.
"""
    anchor = np.asarray(anchor)
    surface = np.asarray(surface)
    active = np.asarray(active)
    top = float(scene['box_center'][0, 2] + scene['box_half_extents'][0, 2])
    ground = float(scene['ground_height'][0])
    if ((surface[active] < 0) | (surface[active] > 1)).any():
        raise ValueError('Unknown actual support face')
    height = np.where(surface == 1, top, ground)
    return np.maximum(height - anchor[:, 2], 0) * (np.asarray(role) == 2) * active

class DemonstrationSuccessorTeacher:

    def __init__(self, records, sequence, model):
        self.bank = [r for r in records if r['sequence'] == sequence and r['supervision_kind'] == 'demonstration']
        if not self.bank or any((not r['split'].startswith('train') for r in self.bank)):
            raise ValueError('Replay expert must contain TRAINING demonstrations only')
        self.model = model
        self.start = torch.stack([r['base'][0] for r in self.bank])
        self.active = torch.stack([r['contract']['start_contact'].bool() for r in self.bank])
        self.surface = torch.stack([r['contract']['start_surface'].long() for r in self.bank])
        endpoints = torch.stack([r['base'][-1] for r in self.bank]).numpy()
        raw = query_contacts(endpoints, None)
        self.query_fingerprint = torch.tensor(list(bytes.fromhex(raw['provenance']['model_fingerprint'])), dtype=torch.uint8)
        selected = select_contact_pairs(raw['pairs'], raw['surface_catalog'])
        self.valid = []
        self.audit = []
        for (i, r) in enumerate(self.bank):
            c = r['contract']
            a = np.asarray(selected['contact_part_mask'][i])
            s = np.asarray(selected['contact_surface'][i])
            ok = np.array_equal(a, c['active_contact'].numpy()) and np.array_equal(s[a], c['contact_surface'].numpy()[a])
            self.valid.append(ok)
            self.audit.append(dict(event=r['event'], first=r['first'], last=r['last'], newton_endpoint_verified=bool(ok)))

    def label(self, q, active, surface, anchor, scene, *, verify=True):
        (x, local, origin, basis) = observed_inputs(self.model, q, active, surface, anchor, scene)
        ok = compatible_current(torch.tensor(active), torch.tensor(surface), self.active, self.surface) & torch.tensor(self.valid)
        if not ok.any():
            return (None, dict(rejected='no_verified_compatible_training_start'))
        distance = ((self.start[:, :3] - torch.tensor(q[:3])) / 0.2).square().sum(-1)
        distance += ((self.start[:, 7:] - torch.tensor(q[7:])) / 0.3).square().mean(-1)
        distance += (self.active != torch.tensor(active)[None]).float().sum(-1)
        i = int(distance.masked_fill(~ok, float('inf')).argmin())
        r = self.bank[i]
        c = r['contract']
        role = successor_roles(x['current_contact'][0], x['current_surface'][0], c)[None]
        depth = embedded_hold_depth(active, surface, anchor, scene, role[0].numpy())
        if (depth > COLLISION_TOLERANCE_M).any():
            return (None, dict(rejected='embedded_persistent_target_requires_recovery_expert', expert_event=r['event'], recovery_required_parts=np.flatnonzero(depth > COLLISION_TOLERANCE_M).tolist(), embedded_hold_depth_cm=(100 * depth).tolist(), collision_budget_m=COLLISION_TOLERANCE_M, actual_contact_unchanged=True, synthetic_recovery_label=False, supervision_accepted=False))
        tq = torch.tensor(transform_q(r['base'][-1].numpy(), origin, basis, True))[None]
        ta = torch.tensor((c['end_anchor_world'].numpy() - origin) @ basis, dtype=torch.float32)[None]
        with torch.no_grad():
            (p, rot) = self.model.fk(tq[:, None])
            rotation = _matrix_from_rotation6d(rot[:, 0])[:, 1:7]
            (sp, sr) = self.model.fk(x['current_q'][:, None])
            sr = _matrix_from_rotation6d(sr[:, 0])[:, 1:7]
        target = dict(q=tq, role=role, surface=c['contact_surface'].long()[None], anchor=ta, newton_world_origin=torch.tensor(origin, dtype=tq.dtype)[None], newton_world_basis=torch.tensor(basis, dtype=tq.dtype)[None], newton_model_fingerprint=self.query_fingerprint[None], material_local=torch.einsum('bpji,bpj->bpi', rotation, ta - p[:, 0, 1:7]), duration=torch.tensor([(len(r['base']) - 1) / 50]), body_position=p[:, 0], observed_faces=x['faces'], solver_margin=r['solver_margin'][None], solver_margin_known=r['solver_margin_known'][None], start_anchor=x['current_anchor'], start_material_local=torch.einsum('bpji,bpj->bpi', sr, x['current_anchor'] - sp[:, 0, 1:7]))
        audit = dict(expert_event=r['event'], expert_first=r['first'], expert_last=r['last'], current_match_cost=float(distance[i]), newton_endpoint_verified=True, teacher='training_demonstration_successor', reachability_certified=False, duration_supervision='retrieved_demonstration_prior', learner_endpoint_is_target=False, supervision_accepted=False, actual_contact_unchanged=True)
        candidate = (x, target, local)
        if not verify:
            return (candidate, audit)
        from generator.research.archived_experiments.paired_replay_gate import verify_candidate
        gate = verify_candidate(self.model, candidate, origin, basis)
        audit['feasibility_gate'] = gate
        audit['supervision_accepted'] = gate['accepted']
        if not gate['accepted']:
            audit['rejected'] = 'no_verified_endpoint_witness'
            return (None, audit)
        return (candidate, audit)
