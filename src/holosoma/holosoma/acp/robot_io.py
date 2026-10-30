"""Simulator-independent observation boundary for the canonical dense G1 tracker.

This is a reference-conditioned WBT interface, not a trained ACP planner. Its
state inputs can be supplied by encoders, IMU and an external odometry estimator.
Body poses are reconstructed from canonical URDF FK rather than simulator links.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics
from somaforge_core.g1_kinematics import _quaternion_matrix_wxyz


@dataclass
class RobotSample:
    """A synchronized packet; orientations xyzw, velocities in the stated frame."""
    timestamp: float
    base_position_w: torch.Tensor
    base_quaternion_xyzw: torch.Tensor
    base_velocity_w: torch.Tensor
    imu_angular_velocity_b: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


class G1TrackerInput:
    def __init__(self, joint_names, tracked_bodies, anchor_body, default_joint_position, term_config):
        self.joint_names = tuple(joint_names)
        if set(self.joint_names) != set(G1_29DOF_JOINT_ORDER) or len(self.joint_names) != 29:
            raise ValueError("Tracker joint order must describe the 29 canonical G1 joints exactly once")
        self.canonical_indices = [self.joint_names.index(name) for name in G1_29DOF_JOINT_ORDER]
        self.tracked_bodies = tuple(tracked_bodies)
        self.anchor_body = anchor_body
        self.default = default_joint_position
        self.terms = term_config
        self.fk = BatchedG1ForwardKinematics().to(default_joint_position.device)
        self.last_timestamp = None

    def reset(self):
        self.last_timestamp = None

    @torch.no_grad()
    def mapped_height_scan(self, sample, pipeline):
        """Reproduce WBT's torso-yaw 7x7 scan using only the accumulated map.

        The native ray caster's 20 m ray-start offset is not part of its
        reported body position. Unknown cells remain explicit in the mask.
        """
        from holosoma.acp.mapping import rotate_yaw

        qpos = torch.cat((sample.base_position_w, sample.base_quaternion_xyzw[:, [3, 0, 1, 2]],
                          sample.joint_position[:, self.canonical_indices]), -1)
        position, rotation = self.fk.link_poses(qpos, ("torso_link",))
        position, rotation = position[:, 0], rotation[:, 0]
        heading = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        axis = torch.arange(-3, 4, device=qpos.device, dtype=qpos.dtype)*0.1
        x, y = torch.meshgrid(axis, axis, indexing="xy")
        points = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x.flatten())), -1)
        world = rotate_yaw(points[None].expand(len(qpos), -1, -1), heading)+position[:, None]
        height, valid = pipeline.map.query(world[..., :2], sample.timestamp)
        return (position[:, 2:3]-height-0.5).masked_fill(~valid, 0), valid

    @torch.no_grad()
    def build(self, sample, *, reference_joints, reference_velocities, reference_anchor_position_w,
              reference_anchor_quaternion_xyzw, previous_action, terrain_height_scan):
        if not math.isfinite(sample.timestamp):
            raise ValueError("Robot state timestamp must be finite")
        if self.last_timestamp is not None and sample.timestamp < self.last_timestamp:
            raise ValueError("Robot state timestamp moved backwards; reset the episode explicitly")
        tensors = (sample.base_position_w, sample.base_quaternion_xyzw, sample.base_velocity_w,
                   sample.imu_angular_velocity_b, sample.joint_position, sample.joint_velocity,
                   reference_joints, reference_velocities, reference_anchor_position_w,
                   reference_anchor_quaternion_xyzw, previous_action, terrain_height_scan)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("Robot observations and reference commands must be finite")
        quaternion = sample.base_quaternion_xyzw[:, [3, 0, 1, 2]]
        root_rotation = _quaternion_matrix_wxyz(quaternion)
        qpos = torch.cat((sample.base_position_w, quaternion,
                          sample.joint_position[:, self.canonical_indices]), -1)
        names = (self.anchor_body, *self.tracked_bodies)
        positions, rotations = self.fk.link_poses(qpos, names)
        anchor_position, anchor_rotation = positions[:, 0], rotations[:, 0]
        inverse = anchor_rotation.transpose(-1, -2)
        reference_rotation = _quaternion_matrix_wxyz(reference_anchor_quaternion_xyzw[:, [3, 0, 1, 2]])
        body_positions = torch.einsum("bij,bnj->bni", inverse, positions[:, 1:]-anchor_position[:, None])
        body_rotations = inverse[:, None] @ rotations[:, 1:]
        terms = {
            "actions": previous_action,
            "base_ang_vel": sample.imu_angular_velocity_b,
            "base_lin_vel": torch.einsum("bij,bj->bi", root_rotation.transpose(-1, -2), sample.base_velocity_w),
            "dof_pos": sample.joint_position-self.default,
            "dof_vel": sample.joint_velocity,
            "motion_command": torch.cat((reference_joints, reference_velocities), -1),
            "motion_ref_pos_b": torch.einsum("bij,bj->bi", inverse, reference_anchor_position_w-anchor_position),
            "motion_ref_ori_b": (inverse @ reference_rotation)[..., :2].flatten(1),
            "pelvis_global_pos": sample.base_position_w,
            "pelvis_global_lin_vel": sample.base_velocity_w,
            "robot_body_pos_b": body_positions.flatten(1),
            "robot_body_ori_b": body_rotations[..., :2].flatten(1),
            "terrain_height_scan": terrain_height_scan,
        }
        if set(terms) != set(self.terms):
            raise ValueError(f"Unsupported tracker observation contract: {sorted(self.terms)}")
        values = []
        for name in sorted(self.terms):
            cfg = self.terms[name]
            value = terms[name]*torch.as_tensor(cfg.scale, device=qpos.device)
            if cfg.clip is not None:
                value = value.clamp(*cfg.clip)
            values.append(value)
        self.last_timestamp = sample.timestamp
        return torch.cat(values, -1), {"body_names": names, "body_position_w": positions,
                                      "body_rotation_w": rotations, "raw_terms": terms}
