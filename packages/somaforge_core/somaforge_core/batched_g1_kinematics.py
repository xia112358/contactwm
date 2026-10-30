"""Canonical G1 FK restricted to the ancestors of requested links."""
import torch
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _quaternion_matrix_wxyz

class BatchedG1ForwardKinematics(CanonicalG1ForwardKinematics):

    def __init__(self):
        super().__init__()
        axis = torch.nn.functional.normalize(self.joint_axis, dim=-1)
        (x, y, z) = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
        self.register_buffer('_axis_skew', skew, persistent=False)
        self.register_buffer('_axis_outer', axis[:, :, None] * axis[:, None, :], persistent=False)
        self.register_buffer('_eye', torch.eye(3), persistent=False)
        self._plans = {}

    def link_poses(self, qpos, link_names):
        if qpos.shape[-1] != 36:
            raise ValueError('Expected canonical qpos with 36 coordinates')
        key = tuple(link_names)
        if key not in self._plans:
            parents = {record[1]: (i, record[0]) for (i, record) in enumerate(self.joint_records)}
            needed = set()
            for link in key:
                while link != self.root_link:
                    if link not in parents:
                        raise ValueError(f'Unknown canonical link: {link}')
                    (i, link) = parents[link]
                    needed.add(i)
            self._plans[key] = tuple(sorted(needed))
        rotations = {self.root_link: _quaternion_matrix_wxyz(qpos[..., 3:7])}
        positions = {self.root_link: qpos[..., :3]}
        for i in self._plans[key]:
            (parent, child, joint_type, q_index) = self.joint_records[i]
            pr = rotations[parent]
            positions[child] = positions[parent] + torch.einsum('...ij,j->...i', pr, self.origin_xyz[i].to(qpos))
            cr = pr @ self.origin_rotation[i].to(qpos)
            if q_index >= 0:
                if joint_type not in {'revolute', 'continuous'}:
                    raise ValueError('Unsupported joint type')
                angle = qpos[..., 7 + q_index]
                cosine = torch.cos(angle)[..., None, None]
                turn = cosine * self._eye.to(qpos) + (1.0 - cosine) * self._axis_outer[i].to(qpos) + torch.sin(angle)[..., None, None] * self._axis_skew[i].to(qpos)
                cr = cr @ turn
            rotations[child] = cr
        return (torch.stack([positions[name] for name in key], -2), torch.stack([rotations[name] for name in key], -3))
