"""Differentiable complete-solid distances, separate from native contact rows."""
import numpy as np
import torch

from somaforge_core.solid_distance import SolidDistanceScene, SOLID_WITNESS_SCHEMA
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d


class SolidWitnessRows:
    def __init__(self, q, pair, poses, *, world_frame=None):
        self.q, self.pair = q, pair
        self.sample = pair['sample']
        points = torch.stack((pair['point0_w'], pair['point1_w']), 1)
        normal = pair['normal_w']
        if world_frame is not None:
            origin, basis = world_frame
            points = torch.einsum('npi,nij->npj', points-origin[self.sample, None].double(), basis[self.sample].double())
            normal = torch.einsum('ni,nij->nj', normal, basis[self.sample].double())
        links = torch.stack((pair['body_link0'], pair['body_link1']), -1)
        position, rotation = poses
        pos = position[self.sample[:, None], links.clamp_min(0)]
        rot = rotation[self.sample[:, None], links.clamp_min(0)]
        material = (rot.detach().transpose(-1, -2) @ (points-pos.detach())[..., None]).detach()
        self.material_local = material.squeeze(-1).to(q)
        self.normal_local = normal.to(q)
        moved = pos+(rot @ material).squeeze(-1)
        moving = torch.where((links >= 0)[..., None], moved, points)
        displacement = moving-moving.detach()
        self.distances = (pair['dist']+(normal*(displacement[:, 1]-displacement[:, 0])).sum(-1)).to(q)
        self.points = points.to(q)

    def depths(self):
        depth = self.q.new_zeros(len(self.q), 2)
        if len(self.sample):
            group = self.sample*2+self.pair['full_kind']
            depth.flatten().scatter_reduce_(0, group, (-self.distances).relu(), reduce='amax', include_self=True)
        return depth


class SolidSceneRouter:
    """Use native scene metadata with the predictor's exact FK and frame."""
    def __init__(self, geometry_by_scene, link_names, *, backend="native_batch"):
        self.link_names = tuple(link_names)
        self.scenes = {int(key): SolidDistanceScene(value, self.link_names, backend=backend) for key, value in geometry_by_scene.items()}
        self._geometry_by_scene = geometry_by_scene
        self.recovery = None

    def recovery_distances(self, fk, rows, scene_ids, current_q_world, points_world, active, surface, *, audit=False):
        """Reproduce the rejected C1 blend for diagnosis; not a training distance."""
        if self.recovery is None:
            from contact_solver.solid_recovery_potential import SolidRecoveryPotential
            self.recovery = {int(key): SolidRecoveryPotential(value, self.link_names)
                             for key, value in self._geometry_by_scene.items()}
        distance = rows.solid.distances
        evidence = []
        for scene_id in scene_ids.unique().tolist():
            if int(scene_id) not in self.recovery:
                raise ValueError('Missing realized recovery scene route')
            selected = scene_ids == scene_id
            recovered, records = self.recovery[int(scene_id)](
                fk, rows, current_q_world, points_world, active, surface, sample_mask=selected, audit=audit)
            distance = torch.where(selected[rows.solid.sample], recovered, distance)
            evidence.extend(records)
        return distance, evidence

    def __call__(self, fk, q, scene_ids, *, world_frame=None):
        # Float64 rigid FK is shared by the geometry query and its material
        # derivative. The differentiable conversion returns gradients to q.
        position, rotation = fk.link_poses(q.double(), self.link_names)
        # FK constants and stored world charts originate in float32. Casting
        # their matrices to float64 retains small nonorthogonal errors; a
        # rigid GJK transform must instead be on SO(3). Use the same tensor
        # orthonormalization for the geometry and its material derivative.
        rotation = _matrix_from_rotation6d(_rotation6d(rotation))
        if world_frame is None:
            world_position, world_rotation = position, rotation
        else:
            origin, basis = world_frame
            basis = _matrix_from_rotation6d(_rotation6d(basis.double()))
            world_frame = (origin, basis)
            world_position = origin[:, None].double()+torch.einsum('bij,bnj->bni', basis, position)
            world_rotation = basis[:, None] @ rotation
        positions, rotations = world_position.detach().cpu().numpy(), world_rotation.detach().cpu().numpy()
        ids = scene_ids.detach().cpu().numpy()
        pieces = []
        for scene_id in np.unique(ids):
            if int(scene_id) not in self.scenes:
                raise ValueError('Missing realized complete-solid scene route')
            selected = np.flatnonzero(ids == scene_id)
            rows = self.scenes[int(scene_id)].query(positions[selected], rotations[selected])
            rows['sample'] = selected[rows['sample']]
            pieces.append(rows)
        if not pieces:
            raise ValueError('Empty solid geometry query')
        pair = {key: torch.as_tensor(np.concatenate([piece[key] for piece in pieces]), device=q.device) for key in pieces[0]}
        return SolidWitnessRows(q, pair, (position, rotation), world_frame=world_frame)

    def contract(self):
        return dict(schema=SOLID_WITNESS_SCHEMA,
                    geometry_backends=sorted({scene.backend for scene in self.scenes.values()}), method='float64 complete-shape GJK/EPA and differentiable canonical FK',
                    training_distance='raw complete-solid distances; C1 recovery blend excluded from training',
                    physical_depth_reporting='unchanged raw complete-solid distances',
                    scene_fingerprints={str(key): scene.fingerprint for key, scene in self.scenes.items()},
                    contact_truth='unchanged actual Newton/MJWarp activation, allocation and primary-face selection',
                    candidates='all enabled actual shape pairs; AABB rejection only; complete containment included')
