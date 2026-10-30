"""Preserved experiment helpers for regression tests; no training CLI."""
import json
import numpy as np
import torch
from somaforge_core.g1_kinematics import _quaternion_matrix_wxyz
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.contact_aggregation import release_supervision

class ContactRetention:

    def __init__(self, batch, labels, ids, n, links, device, upper_only=False, preserve_release=False, certified_release=None):
        if preserve_release and certified_release is None:
            raise ValueError('Release preservation requires independently certified action evidence; raw contact absence is insufficient')
        self.upper_only = upper_only
        self.preserve_release = preserve_release
        from somaforge_core.newton_contact_data import load_raw_contact_labels
        with np.load(labels, allow_pickle=False) as z:
            semantics = json.loads(z['contact_semantics_json'].item())
            pairs = select_contact_pairs(json.loads(z['contact_pairs_json'].item()), semantics['scene']['surface_catalog'])['contact_pairs']
        e = len(ids)
        if len(pairs) != e * n:
            raise ValueError('Contact retention alignment mismatch')
        names = list(batch['body_names'].astype(str))
        parents = ['left_ankle_roll_link', 'right_ankle_roll_link', 'left_sphere_hand_link', 'right_sphere_hand_link', 'left_knee_link', 'right_knee_link']
        indices = [names.index(x) for x in parents]
        p = batch['body_pos_w'].reshape(e, n, len(names), 3)[:, :, indices]
        r = batch['body_quat_w'].reshape(e, n, len(names), 4)[:, :, indices]

        def rotate(q, v):
            uv = np.cross(q[..., 1:], v)
            return v + 2 * (q[..., :1] * uv + np.cross(q[..., 1:], uv))
        groups = {}
        part_faces = {}
        for (f, row) in enumerate(pairs):
            (env, t) = divmod(f, n)
            for point in row:
                key = (point['part'], point['robot_shape'], point['surface'])
                groups.setdefault(key, {}).setdefault((env, t), []).append(point['position_w'])
                part_faces.setdefault((point['part'], point['surface']), np.zeros((e, n), bool))[env, t] = True
        normals = {s['surface']: np.asarray(s['normal_w']) for s in semantics['scene']['surface_catalog']}
        local_points = []
        targets = []
        masks = []
        body_indices = []
        normal_rows = []
        keys = []
        release_targets = []
        release_masks = []
        for (key, observations) in sorted(groups.items()):
            (part, shape, surface) = key
            mask = np.zeros((e, n), bool)
            local = []
            for ((env, t), points) in observations.items():
                mask[env, t] = True
                inv = r[env, t, part].copy()
                inv[1:] *= -1
                local.extend(rotate(inv, np.asarray(points) - p[env, t, part]))
            enabled = mask.sum(0) > e / 2
            if not enabled.any():
                continue
            fixed = np.median(local, axis=0)
            world = p[:, :, part] + rotate(r[:, :, part], fixed)
            normal = normals[surface]
            heights = world @ normal
            target = np.zeros(n)
            for t in np.flatnonzero(enabled):
                target[t] = np.median(heights[mask[:, t], t])
            active_part = part_faces[part, surface]
            evidence = None if certified_release is None else certified_release.get((part, surface))
            release = release_supervision(active_part, evidence)
            release_target = np.zeros(n)
            for t in np.flatnonzero(release):
                release_target[t] = np.median(heights[~active_part[:, t], t])
            release_targets.append(release_target)
            release_masks.append(release)
            local_points.append(fixed)
            targets.append(target)
            masks.append(enabled)
            body_indices.append(links.index(parents[part]))
            normal_rows.append(normal)
            keys.append(key)
        if not keys:
            raise ValueError('No majority source contact witnesses')
        as_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.local = as_tensor(local_points)
        self.normal = as_tensor(normal_rows)
        self.target = as_tensor(targets).T
        self.mask = as_tensor(masks).T
        self.release_target = as_tensor(release_targets).T
        self.release_mask = as_tensor(release_masks).T
        self.indices = body_indices
        self.report = dict(contract='majority_shape_face_fixed_material_normal_retention_v2', upper_only=upper_only, preserve_release=preserve_release, release_shape_frames=int(self.release_mask.sum()), release_evidence='independent_action_certificate_required', release_weight=0.1, keys=keys, active_shape_frames=int(self.mask.sum()), normal_scale_m=0.001, target='median source FK height of fixed material point among actual active source rollouts', contact_truth='independent Newton requery required; loss is not activation')

    def loss(self, p, r):
        world = p[:, self.indices] + torch.einsum('...ij,...j->...i',
            _quaternion_matrix_wxyz(r[:, self.indices]), self.local[None].expand(len(p), -1, -1))
        height = (world * self.normal).sum(-1)
        error = (height - self.target) / 0.001
        if self.upper_only:
            error = error.relu()
        loss = (error.square() * self.mask).sum() / self.mask.sum().clamp_min(1)
        if self.preserve_release:
            release_error = ((self.release_target - height) / 0.001).relu()
            loss = loss + 0.1 * (release_error.square() * self.release_mask).sum() / self.release_mask.sum().clamp_min(1)
        return loss
