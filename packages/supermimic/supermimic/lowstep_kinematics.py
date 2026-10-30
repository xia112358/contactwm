"""Differentiable canonical URDF FK, composing fixed links between actuated joints."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .robot import G1_JOINT_NAMES, TRACKED_BODIES
from .rotations import axis_angle_to_mat


class CanonicalKinematics:
    def __init__(self, device, tracked=TRACKED_BODIES):
        from somaforge_core.robot_assets import canonical_g1_urdf_path, validate_canonical_g1_urdf
        path = canonical_g1_urdf_path()
        validate_canonical_g1_urdf(path)
        root = ET.parse(path).getroot()
        children = {}
        joints = {}
        for joint in root.findall("joint"):
            children.setdefault(joint.find("parent").get("link"), []).append(joint)
            joints[joint.get("name")] = joint
        self.device = device
        self.body_names = ["pelvis"]
        self.parent, self.joint_idx = [-1], [-1]
        positions, rotations, axes = [np.zeros(3)], [np.eye(3)], [np.array([0., 0., 1.])]

        def visit(link, parent_index, fixed):
            for joint in children.get(link, []):
                origin = joint.find("origin")
                position = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                transform[:3, 3] = position
                transform = fixed @ transform
                child = joint.find("child").get("link")
                if joint.get("type") == "fixed":
                    visit(child, parent_index, transform)
                    continue
                if joint.get("type") not in {"revolute", "continuous"}:
                    raise ValueError(f"Unsupported canonical joint: {joint.attrib}")
                name = joint.get("name")
                self.body_names.append(child)
                self.parent.append(parent_index)
                self.joint_idx.append(G1_JOINT_NAMES.index(name))
                positions.append(transform[:3, 3])
                rotations.append(transform[:3, :3])
                axis = joint.find("axis")
                axes.append(np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1., 0., 0.]))
                visit(child, len(self.body_names) - 1, np.eye(4))

        visit("pelvis", 0, np.eye(4))
        if sorted(self.joint_idx[1:]) != list(range(29)):
            raise ValueError("Canonical URDF does not provide exactly 29 supported joints")
        self.num_bodies = len(self.body_names)
        self.tracked_idx = [self.body_names.index(name) for name in tracked]
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(positions), dtype=torch.float32, device=device)
        self.off_R = torch.tensor(np.array(rotations), dtype=torch.float32, device=device)
        self.q_lo = torch.tensor([float(joints[name].find("limit").get("lower")) for name in G1_JOINT_NAMES], device=device)
        self.q_hi = torch.tensor([float(joints[name].find("limit").get("upper")) for name in G1_JOINT_NAMES], device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        positions, rotations, velocities, angular = [p], [R], [v], [w]
        for index in range(1, self.num_bodies):
            parent, joint = self.parent[index], self.joint_idx[index]
            offset = (rotations[parent] @ self.off_p[index].unsqueeze(-1)).squeeze(-1)
            position = positions[parent] + offset
            base_rotation = rotations[parent] @ self.off_R[index]
            rotation = base_rotation @ axis_angle_to_mat(self.axes[index], q[..., joint])
            axis = (base_rotation @ self.axes[index].unsqueeze(-1)).squeeze(-1)
            positions.append(position)
            rotations.append(rotation)
            velocities.append(velocities[parent] + torch.cross(angular[parent], offset, dim=-1))
            angular.append(angular[parent] + axis * qd[..., joint:joint + 1])
        ids = range(self.num_bodies) if all_bodies else self.tracked_idx
        return tuple(torch.stack([values[i] for i in ids], -3 if values is rotations else -2)
                     for values in (positions, rotations, velocities, angular))
