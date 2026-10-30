from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger
from somaforge_core.contact_schema import CONTACT_FORCE_PART_ORDER, decode_contact_force_provenance
from somaforge_core.kinematics import validate_root_body_consistency
from somaforge_core.motion_schema import decode_kinematics_provenance
from somaforge_core.robot_assets import decode_robot_asset_json

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType

A2A_LIMB_REF_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

CHAIN_BOUNDARY_CONTACT_BODY_NAMES = (
    ("left_ankle_roll_link",),
    ("right_ankle_roll_link",),
    ("left_wrist_yaw_link",),
    ("right_wrist_yaw_link",),
    ("left_knee_link",),
    ("right_knee_link",),
    ("left_hip_roll_link",),
    ("right_hip_roll_link",),
)

CHAIN_BOUNDARY_PART_ORDER = ("LF", "RF", "LH", "RH", "LK", "RK", "LHIP", "RHIP")


def _part_names_to_chain_mask(value: Any, device: str | torch.device = "cpu") -> torch.Tensor:
    if value is None:
        names: list[str] = []
    elif isinstance(value, str):
        names = [part.strip() for part in re.split(r"[|, ]+", value) if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    aliases = {"LEFT_FOOT": "LF", "RIGHT_FOOT": "RF", "LEFT_HAND": "LH", "RIGHT_HAND": "RH"}
    normalized = {aliases.get(name.upper(), name.upper()) for name in names}
    mask = torch.zeros(len(CHAIN_BOUNDARY_PART_ORDER), dtype=torch.bool, device=device)
    for i, part in enumerate(CHAIN_BOUNDARY_PART_ORDER):
        if part in normalized:
            mask[i] = True
    return mask


def _get_bad_tracking_done_mask(term_dones: dict[str, torch.Tensor], env_ids: torch.Tensor) -> torch.Tensor | None:
    """Return the union of bad-tracking termination masks for the requested envs."""
    bad_tracking_names = [name for name in term_dones if "bad_tracking" in name]
    if not bad_tracking_names:
        return None

    mask = torch.zeros(env_ids.shape, dtype=torch.bool, device=env_ids.device)
    for name in bad_tracking_names:
        mask |= term_dones[name][env_ids].to(torch.bool)
    return mask


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        canonicalize_motion_order_on_load: bool = False,
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)
        self._motion_files = [str(motion_file)]

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._motion_order_canonicalized = bool(canonicalize_motion_order_on_load)
        if self._motion_order_canonicalized:
            self._canonicalize_motion_order(body_indexes, joint_indexes)
            logger.info(f"MotionLoader: canonicalized motion tensors to robot order: {motion_file}")
        else:
            self._joint_indexes = joint_indexes
            self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _canonicalize_motion_order(self, body_indexes: torch.Tensor, joint_indexes: torch.Tensor) -> None:
        """Store motion tensors in simulator body/joint order once at load time."""
        self._joint_pos = self._joint_pos[:, joint_indexes].contiguous()
        self._joint_vel = self._joint_vel[:, joint_indexes].contiguous()
        self._sim_joint_vel = self._sim_joint_vel[:, joint_indexes].contiguous()
        self._body_pos_w = self._body_pos_w[:, body_indexes].contiguous()
        self._body_quat_w = self._body_quat_w[:, body_indexes].contiguous()
        self._body_lin_vel_w = self._body_lin_vel_w[:, body_indexes].contiguous()
        self._body_ang_vel_w = self._body_ang_vel_w[:, body_indexes].contiguous()
        self._joint_indexes = torch.arange(self._joint_pos.shape[1], dtype=torch.long, device=self._joint_pos.device)
        self._body_indexes = torch.arange(self._body_pos_w.shape[1], dtype=torch.long, device=self._body_pos_w.device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
        "robot_asset_json",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            try:
                decode_robot_asset_json(data["robot_asset_json"], context=f"motion {motion_file}")
            except ValueError as exc:
                raise RuntimeError(f"Refusing motion with incompatible robot provenance: {motion_file}") from exc
            try:
                kinematics_provenance = decode_kinematics_provenance(
                    data.get("kinematics_provenance_json", None),
                    context=f"motion {motion_file}",
                )
            except ValueError as exc:
                raise RuntimeError(f"Refusing motion without Newton FK provenance: {motion_file}") from exc
            if "contact_force_part_w" in data:
                try:
                    decode_contact_force_provenance(
                        data.get("contact_force_provenance_json", None),
                        context=f"motion {motion_file}",
                        require_newton=True,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"Refusing force-bearing motion without Newton contact provenance: {motion_file}"
                    ) from exc

            self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isfinite(self.fps) or self.fps <= 0.0:
                raise ValueError(f"Motion FPS must be finite and positive: {motion_file} -> {self.fps}")
            if not np.isclose(
                self.fps,
                float(kinematics_provenance["output_fps"]),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(f"Motion FPS disagrees with kinematics provenance: {motion_file}")

            body_names = [str(name) for name in data["body_names"].tolist()]
            joint_names = [str(name) for name in data["joint_names"].tolist()]
            if body_names != [str(name) for name in kinematics_provenance["body_names"]]:
                raise ValueError(f"Motion body_names disagree with kinematics provenance: {motion_file}")
            if joint_names != [str(name) for name in kinematics_provenance["joint_order"]]:
                raise ValueError(f"Motion joint_names disagree with kinematics provenance: {motion_file}")

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            sim_joint_vel_raw = data["sim_joint_vel"] if "sim_joint_vel" in data else joint_vel_raw
            sim_root_lin_vel_raw = (
                data["sim_root_lin_vel"] if "sim_root_lin_vel" in data else joint_vel_raw[:, :3]
            )
            sim_root_ang_vel_raw = (
                data["sim_root_ang_vel"] if "sim_root_ang_vel" in data else joint_vel_raw[:, 3:6]
            )
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                raise ValueError(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7}. "
                    f"File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                raise ValueError(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6}. "
                    f"File: {motion_file}"
                )
            if sim_joint_vel_raw.shape != joint_vel_raw.shape:
                raise ValueError(
                    f"Unexpected sim_joint_vel shape: got {sim_joint_vel_raw.shape}, "
                    f"expected {joint_vel_raw.shape}. File: {motion_file}"
                )
            expected_root_vel_shape = (joint_pos_raw.shape[0], 3)
            if sim_root_lin_vel_raw.shape != expected_root_vel_shape:
                raise ValueError(
                    f"Unexpected sim_root_lin_vel shape: got {sim_root_lin_vel_raw.shape}, "
                    f"expected {expected_root_vel_shape}. File: {motion_file}"
                )
            if sim_root_ang_vel_raw.shape != expected_root_vel_shape:
                raise ValueError(
                    f"Unexpected sim_root_ang_vel shape: got {sim_root_ang_vel_raw.shape}, "
                    f"expected {expected_root_vel_shape}. File: {motion_file}"
                )
            if num_bodies != len(body_names):
                raise ValueError(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )
            if "pelvis" not in body_names:
                raise ValueError(f"Motion has no pelvis root body: {motion_file}")
            validate_root_body_consistency(
                joint_pos_raw,
                body_pos_w_raw,
                body_quat_w_raw,
                root_body_index=body_names.index("pelvis"),
            )
            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)
            self._sim_joint_vel = torch.tensor(sim_joint_vel_raw[:, 6:], dtype=torch.float32, device=device)
            self._sim_root_lin_vel = torch.tensor(sim_root_lin_vel_raw, dtype=torch.float32, device=device)
            self._sim_root_ang_vel = torch.tensor(sim_root_ang_vel_raw, dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)
            num_frames = self._joint_pos.shape[0]
            self._part_order = [str(x) for x in data["part_order"].tolist()] if "part_order" in data else []
            self._has_part_annotations = any(
                key in data for key in ("active_part_mask", "contact_keep_intent_mask", "support_part_mask", "contact_part_mask")
            )
            self._active_part_mask = self._load_part_mask(data, "active_part_mask", num_frames, device)
            # Historical field is a desired role, never measured support truth.
            keep_key = "contact_keep_intent_mask" if "contact_keep_intent_mask" in data else "support_part_mask"
            self._support_part_mask = self._load_part_mask(data, keep_key, num_frames, device)
            self._free_part_mask = self._load_part_mask(data, "free_part_mask", num_frames, device)
            self._contact_part_mask = self._load_part_mask(data, "contact_part_mask", num_frames, device)
            contact_force_part_w = self._load_contact_force_part(data, "contact_force_part_w", num_frames, device)
            contact_force_part_mask = self._load_part_mask(data, "contact_force_part_mask", num_frames, device)
            raw_contact_force_part_order = (
                [str(x) for x in data["contact_force_part_order"].tolist()]
                if "contact_force_part_order" in data
                else list(CONTACT_FORCE_PART_ORDER[: contact_force_part_w.shape[1]])
            )
            self._contact_force_part_w, self._contact_force_part_mask = self._align_contact_force_parts(
                contact_force_part_w,
                contact_force_part_mask,
                raw_contact_force_part_order,
                CONTACT_FORCE_PART_ORDER,
            )
            self._contact_force_part_order = list(CONTACT_FORCE_PART_ORDER)
            self._proto_start_idx, self._proto_end_idx = self._load_proto_bounds(
                data, num_frames, device, Path(motion_file)
            )

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @staticmethod
    def _load_part_mask(data: np.lib.npyio.NpzFile, key: str, num_frames: int, device: str) -> torch.Tensor:
        if key not in data:
            return torch.zeros(num_frames, 8, dtype=torch.bool, device=device)
        mask = np.asarray(data[key], dtype=np.bool_)
        if mask.ndim != 2 or mask.shape[0] != num_frames:
            raise ValueError(f"{key} must have shape [T, num_parts], got {mask.shape}, T={num_frames}")
        return torch.tensor(mask, dtype=torch.bool, device=device)

    @staticmethod
    def _load_contact_force_part(
        data: np.lib.npyio.NpzFile,
        key: str,
        num_frames: int,
        device: str,
    ) -> torch.Tensor:
        if key not in data:
            return torch.zeros(num_frames, len(CONTACT_FORCE_PART_ORDER), 3, dtype=torch.float32, device=device)
        force = np.asarray(data[key], dtype=np.float32)
        if force.ndim != 3 or force.shape[0] != num_frames or force.shape[2] != 3:
            raise ValueError(f"{key} must have shape [T, num_parts, 3], got {force.shape}, T={num_frames}")
        return torch.tensor(force, dtype=torch.float32, device=device)

    @staticmethod
    def _align_contact_force_parts(
        force: torch.Tensor,
        mask: torch.Tensor,
        source_order: list[str],
        target_order: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source_order) != force.shape[1]:
            source_order = list(target_order[: force.shape[1]])
        if mask.shape[1] != force.shape[1]:
            mask = torch.norm(force, dim=-1) > 10.0

        aligned_force = torch.zeros(force.shape[0], len(target_order), 3, dtype=force.dtype, device=force.device)
        aligned_mask = torch.zeros(force.shape[0], len(target_order), dtype=torch.bool, device=force.device)
        target_index = {part_name: i for i, part_name in enumerate(target_order)}
        for src_i, part_name in enumerate(source_order):
            dst_i = target_index.get(part_name)
            if dst_i is None:
                continue
            aligned_force[:, dst_i] = force[:, src_i]
            aligned_mask[:, dst_i] = mask[:, src_i]
        return aligned_force, aligned_mask

    @staticmethod
    def _load_proto_bounds(
        data: np.lib.npyio.NpzFile,
        num_frames: int,
        device: str,
        motion_file: Path | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "proto_start_idx" in data and "proto_end_idx" in data:
            starts = np.asarray(data["proto_start_idx"], dtype=np.int64)
            ends = np.asarray(data["proto_end_idx"], dtype=np.int64)
        else:
            starts, ends = MotionLoader._load_external_proto_bounds(motion_file, num_frames)
            if starts is None or ends is None:
                starts = np.asarray([0], dtype=np.int64)
                ends = np.asarray([num_frames], dtype=np.int64)
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                f"proto_start_idx/proto_end_idx must be 1D arrays with the same shape, got {starts.shape}/{ends.shape}"
            )
        keep = (ends > starts) & (starts >= 0) & (ends <= num_frames)
        starts = starts[keep]
        ends = ends[keep]
        if starts.size == 0:
            starts = np.asarray([0], dtype=np.int64)
            ends = np.asarray([num_frames], dtype=np.int64)
        return (
            torch.tensor(starts, dtype=torch.long, device=device),
            torch.tensor(ends, dtype=torch.long, device=device),
        )

    @staticmethod
    def _load_external_proto_bounds(
        motion_file: Path | None,
        num_frames: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if motion_file is None:
            return None, None
        base_stem = motion_file.stem.split("_force_demo", 1)[0]
        repo_root = Path(__file__).resolve().parents[6]
        split_bounds = MotionLoader._load_split_proto_manifest_bounds(repo_root, base_stem, num_frames)
        if split_bounds[0] is not None and split_bounds[1] is not None:
            logger.info(f"Loaded split proto bounds for {motion_file.name} from proto17 manifest")
            return split_bounds
        candidates = [
            repo_root / "OmniRetarget_Dataset/data/holosoma_motions_masked_50hz" / f"{base_stem}.npz",
            repo_root / "data/holosoma_motions_masked_50hz" / f"{base_stem}.npz",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            with np.load(candidate, allow_pickle=True) as data:
                if "proto_start_idx" not in data or "proto_end_idx" not in data:
                    continue
                starts = np.asarray(data["proto_start_idx"], dtype=np.int64)
                ends = np.asarray(data["proto_end_idx"], dtype=np.int64)
                if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
                    continue
                if starts.size == 0 or int(ends.max(initial=0)) > num_frames:
                    continue
                logger.info(f"Loaded proto bounds for {motion_file.name} from {candidate}")
                return starts, ends
        return None, None

    @staticmethod
    def _load_split_proto_manifest_bounds(
        repo_root: Path,
        base_stem: str,
        num_frames: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        match = re.match(r"(?P<terrain>.+)_z_scale_(?P<scale>[0-9.]+)$", base_stem)
        if match is None:
            return None, None
        terrain_key = match.group("terrain").replace("_", "")
        candidates = [
            repo_root / "configs/motion_matched" / f"{terrain_key}_unmasked_proto17_split_long7_manifest.json",
            repo_root / "configs/motion_matched" / f"{terrain_key}_masked_proto17_split_long7_manifest.json",
        ]
        for manifest_path in candidates:
            if not manifest_path.exists():
                continue
            try:
                manifest = load_motion_terrain_manifest(str(manifest_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to load split proto manifest {manifest_path}: {exc}")
                continue
            entries = []
            for entry in manifest.get("motion_files", []):
                entry_file = Path(str(entry.get("motion_file", ""))).name
                if base_stem not in entry_file:
                    continue
                if "source_start" in entry and "source_end" in entry:
                    start = int(entry["source_start"])
                    end = int(entry["source_end"])
                else:
                    frame_match = re.search(r"_frames_(\d+)_(\d+)\.npz$", entry_file)
                    if frame_match is None:
                        continue
                    start = int(frame_match.group(1))
                    end = int(frame_match.group(2))
                if 0 <= start < end <= num_frames:
                    entries.append((start, end))
            if not entries:
                continue
            entries = sorted(set(entries))
            starts = np.asarray([start for start, _ in entries], dtype=np.int64)
            ends = np.asarray([end for _, end in entries], dtype=np.int64)
            return starts, ends
        return None, None

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._joint_pos
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._joint_vel
        return self._joint_vel[:, self._joint_indexes]

    @property
    def sim_joint_vel(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._sim_joint_vel
        return self._sim_joint_vel[:, self._joint_indexes]

    @property
    def sim_root_lin_vel(self) -> torch.Tensor:
        return self._sim_root_lin_vel

    @property
    def sim_root_ang_vel(self) -> torch.Tensor:
        return self._sim_root_ang_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_pos_w
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_quat_w
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_lin_vel_w
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_ang_vel_w
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def part_order(self) -> list[str]:
        return self._part_order

    @property
    def active_part_mask(self) -> torch.Tensor:
        return self._active_part_mask

    @property
    def support_part_mask(self) -> torch.Tensor:
        """Compatibility name for declared contact-keep intent, not measured support."""
        return self._support_part_mask

    @property
    def free_part_mask(self) -> torch.Tensor:
        return self._free_part_mask

    @property
    def contact_part_mask(self) -> torch.Tensor:
        return self._contact_part_mask

    @property
    def contact_force_part_w(self) -> torch.Tensor:
        return self._contact_force_part_w

    @property
    def contact_force_part_mask(self) -> torch.Tensor:
        return self._contact_force_part_mask

    @property
    def contact_force_part_order(self) -> list[str]:
        return self._contact_force_part_order

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_terrain_ids(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_sampling_weights(self) -> torch.Tensor:
        return torch.tensor([1.0], dtype=torch.float32, device=self._joint_pos.device)

    @property
    def proto_start_idx(self) -> torch.Tensor:
        return self._proto_start_idx

    @property
    def proto_end_idx(self) -> torch.Tensor:
        return self._proto_end_idx

    @property
    def proto_motion_ids(self) -> torch.Tensor:
        return torch.zeros(len(self._proto_start_idx), dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_files(self) -> list[str]:
        return self._motion_files

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
            ("part_mask", "_active_part_mask"),
            ("part_mask", "_support_part_mask"),
            ("part_mask", "_free_part_mask"),
            ("part_mask", "_contact_part_mask"),
            ("contact_force_part", "_contact_force_part_w"),
            ("contact_force_part_mask", "_contact_force_part_mask"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        if prepend:
            added_frames = int(segments["joint_pos"].shape[0])
            self._proto_start_idx = self._proto_start_idx + added_frames
            self._proto_end_idx = self._proto_end_idx + added_frames

        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        motion_manifest: str = "",
        canonicalize_motion_order_on_load: bool = False,
    ):
        terrain_ids: list[int] = []
        weights: list[float] = []
        if motion_manifest:
            manifest = load_motion_terrain_manifest(motion_manifest)
            motion_entries = manifest["motion_files"]
            motion_files = [entry["motion_file"] for entry in motion_entries]
            terrain_ids = [int(entry["terrain_id"]) for entry in motion_entries]
            weights = [float(entry.get("weight", 1.0)) for entry in motion_entries]
            touchdown_masks = [
                _part_names_to_chain_mask(
                    entry.get("expected_touchdown_parts", entry.get("touchdown", "")), device=device
                )
                for entry in motion_entries
            ]
            logger.info(
                f"MultiMotionLoader: loading {len(motion_files)} motion file(s) from manifest {manifest['path']}"
            )
        else:
            # Support comma-separated directories for combining multiple datasets
            dirs = [d.strip() for d in motion_dir.split(",")]
            motion_files = []
            for d in dirs:
                expanded = os.path.expanduser(d)
                files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
                logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
                motion_files.extend(files)
            terrain_ids = [0 for _ in motion_files]
            weights = [1.0 for _ in motion_files]
            touchdown_masks = [
                torch.zeros(len(CHAIN_BOUNDARY_PART_ORDER), dtype=torch.bool, device=device) for _ in motion_files
            ]
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        kept_terrain_ids = []
        kept_weights = []
        kept_touchdown_masks = []
        kept_motion_files = []
        for mf, terrain_id, weight, touchdown_mask in zip(motion_files, terrain_ids, weights, touchdown_masks):
            loader = MotionLoader(
                mf,
                robot_body_names,
                robot_joint_names,
                device=device,
                canonicalize_motion_order_on_load=True,
            )
            loaders.append(loader)
            kept_terrain_ids.append(terrain_id)
            kept_weights.append(weight)
            kept_touchdown_masks.append(touchdown_mask)
            kept_motion_files.append(str(resolve_data_file_path(mf)))

        fps_values = [loader.fps for loader in loaders]
        if not all(np.isclose(fps, fps_values[0], atol=1.0e-6, rtol=0.0) for fps in fps_values[1:]):
            raise ValueError(f"All motions must use the same FPS, got {fps_values}")

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)
        self._motion_terrain_ids = torch.tensor(kept_terrain_ids, dtype=torch.long, device=device)
        self._motion_sampling_weights = torch.tensor(kept_weights, dtype=torch.float32, device=device)
        self._motion_sampling_weights = self._motion_sampling_weights / self._motion_sampling_weights.sum()
        self._motion_touchdown_part_mask = torch.stack(kept_touchdown_masks, dim=0)
        self._motion_files = kept_motion_files

        # Concatenate all motion data
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._sim_joint_vel = torch.cat([ld._sim_joint_vel for ld in loaders], dim=0)
        self._sim_root_lin_vel = torch.cat([ld._sim_root_lin_vel for ld in loaders], dim=0)
        self._sim_root_ang_vel = torch.cat([ld._sim_root_ang_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)
        self._active_part_mask = torch.cat([ld._active_part_mask for ld in loaders], dim=0)
        self._support_part_mask = torch.cat([ld._support_part_mask for ld in loaders], dim=0)
        self._free_part_mask = torch.cat([ld._free_part_mask for ld in loaders], dim=0)
        self._contact_part_mask = torch.cat([ld._contact_part_mask for ld in loaders], dim=0)
        self._contact_force_part_w = torch.cat([ld._contact_force_part_w for ld in loaders], dim=0)
        self._contact_force_part_mask = torch.cat([ld._contact_force_part_mask for ld in loaders], dim=0)
        self._has_part_annotations = any(ld._has_part_annotations for ld in loaders)
        self._part_order = loaders[0]._part_order
        self._contact_force_part_order = loaders[0]._contact_force_part_order
        proto_starts = []
        proto_ends = []
        proto_motion_ids = []
        offset = 0
        for motion_id, loader in enumerate(loaders):
            proto_starts.append(loader._proto_start_idx + offset)
            proto_ends.append(loader._proto_end_idx + offset)
            proto_motion_ids.append(
                torch.full_like(
                    loader._proto_start_idx, motion_id, dtype=torch.long, device=loader._proto_start_idx.device
                )
            )
            offset += loader.time_step_total
        self._proto_start_idx = torch.cat(proto_starts, dim=0)
        self._proto_end_idx = torch.cat(proto_ends, dim=0)
        self._proto_motion_ids = torch.cat(proto_motion_ids, dim=0)

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self._motion_order_canonicalized = all(ld._motion_order_canonicalized for ld in loaders)
        self.fps = loaders[0].fps
        self.time_step_total = self._joint_pos.shape[0]

        # Object support: only if ALL motions have objects
        self.has_object = all(ld.has_object for ld in loaders)
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)

        logger.info(f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames")

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def motion_terrain_ids(self) -> torch.Tensor:
        return self._motion_terrain_ids

    @property
    def motion_sampling_weights(self) -> torch.Tensor:
        return self._motion_sampling_weights

    @property
    def motion_touchdown_part_mask(self) -> torch.Tensor:
        return self._motion_touchdown_part_mask

    @property
    def proto_start_idx(self) -> torch.Tensor:
        return self._proto_start_idx

    @property
    def proto_end_idx(self) -> torch.Tensor:
        return self._proto_end_idx

    @property
    def proto_motion_ids(self) -> torch.Tensor:
        return self._proto_motion_ids

    @property
    def motion_files(self) -> list[str]:
        return self._motion_files

    @property
    def part_order(self) -> list[str]:
        return self._part_order

    @property
    def active_part_mask(self) -> torch.Tensor:
        return self._active_part_mask

    @property
    def support_part_mask(self) -> torch.Tensor:
        """Compatibility name for declared contact-keep intent, not measured support."""
        return self._support_part_mask

    @property
    def free_part_mask(self) -> torch.Tensor:
        return self._free_part_mask

    @property
    def contact_part_mask(self) -> torch.Tensor:
        return self._contact_part_mask

    @property
    def contact_force_part_w(self) -> torch.Tensor:
        return self._contact_force_part_w

    @property
    def contact_force_part_mask(self) -> torch.Tensor:
        return self._contact_force_part_mask

    @property
    def contact_force_part_order(self) -> list[str]:
        return self._contact_force_part_order

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._joint_pos
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._joint_vel
        return self._joint_vel[:, self._joint_indexes]

    @property
    def sim_joint_vel(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._sim_joint_vel
        return self._sim_joint_vel[:, self._joint_indexes]

    @property
    def sim_root_lin_vel(self) -> torch.Tensor:
        return self._sim_root_lin_vel

    @property
    def sim_root_ang_vel(self) -> torch.Tensor:
        return self._sim_root_ang_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_pos_w
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_quat_w
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_lin_vel_w
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._motion_order_canonicalized:
            return self._body_ang_vel_w
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Merge interpolated segments with motion data, mutating this MultiMotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
            ("part_mask", "_active_part_mask"),
            ("part_mask", "_support_part_mask"),
            ("part_mask", "_free_part_mask"),
            ("part_mask", "_contact_part_mask"),
            ("contact_force_part", "_contact_force_part_w"),
            ("contact_force_part_mask", "_contact_force_part_mask"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        # Update boundaries — shift all motion boundaries if prepending
        if prepend:
            self._proto_start_idx = self._proto_start_idx + added_frames
            self._proto_end_idx = self._proto_end_idx + added_frames
            self._motion_start_idx = self._motion_start_idx + added_frames
            self._motion_end_idx = self._motion_end_idx + added_frames
            self._proto_motion_ids = self._proto_motion_ids + 1
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=dev), self._motion_start_idx]
            )
            self._motion_end_idx = torch.cat(
                [torch.tensor([added_frames], dtype=torch.long, device=dev), self._motion_end_idx]
            )
            self._motion_terrain_ids = torch.cat([self._motion_terrain_ids[:1], self._motion_terrain_ids])
            self._motion_sampling_weights = torch.cat(
                [self._motion_sampling_weights[:1], self._motion_sampling_weights]
            )
            self._motion_files = [f"default_pose_prepend:{self._motion_files[0]}", *self._motion_files]
        else:
            old_total = self.time_step_total
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [self._motion_start_idx, torch.tensor([old_total], dtype=torch.long, device=dev)]
            )
            self._motion_end_idx = torch.cat(
                [self._motion_end_idx, torch.tensor([old_total + added_frames], dtype=torch.long, device=dev)]
            )
            self._motion_terrain_ids = torch.cat([self._motion_terrain_ids, self._motion_terrain_ids[-1:]])
            self._motion_sampling_weights = torch.cat(
                [self._motion_sampling_weights, self._motion_sampling_weights[-1:]]
            )
            self._motion_files = [*self._motion_files, f"default_pose_append:{self._motion_files[-1]}"]

        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = len(self._motion_start_idx)
        self._motion_sampling_weights = self._motion_sampling_weights / self._motion_sampling_weights.sum()
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_lengths: torch.Tensor | int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        if isinstance(motion_lengths, int):
            motion_lengths = torch.tensor([motion_lengths], dtype=torch.long, device=device)
        else:
            motion_lengths = motion_lengths.to(device=device, dtype=torch.long)
        self.motion_lengths = motion_lengths.clamp(min=1)
        self.num_motions = int(self.motion_lengths.numel())
        self.motion_time_step_total = int(self.motion_lengths.sum().item())
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        self.num_bins_per_motion = (self.motion_lengths // max(self.env_fps, 1) + 1).clamp(min=1)
        self.num_bins = int(self.num_bins_per_motion.max().item())
        bin_ids = torch.arange(self.num_bins, device=self.device).unsqueeze(0)
        self.valid_bin_mask = bin_ids < self.num_bins_per_motion.unsqueeze(1)
        self.bin_start_local, self.bin_end_local = self._build_fixed_bins()
        self.bin_lengths = (self.bin_end_local - self.bin_start_local).clamp(min=0)
        self.uniform_bin_probability = self.bin_lengths.to(torch.float32) / self.motion_lengths.to(
            torch.float32
        ).unsqueeze(1).clamp(min=1.0)
        self.uniform_bin_probability = self.uniform_bin_probability * self.valid_bin_mask
        self.uniform_bin_probability = self.uniform_bin_probability / self.uniform_bin_probability.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def _build_fixed_bins(self) -> tuple[torch.Tensor, torch.Tensor]:
        bin_ids = torch.arange(self.num_bins, device=self.device).unsqueeze(0)
        starts = torch.div(
            bin_ids * self.motion_lengths.unsqueeze(1),
            self.num_bins_per_motion.unsqueeze(1),
            rounding_mode="floor",
        )
        ends = torch.div(
            (bin_ids + 1) * self.motion_lengths.unsqueeze(1),
            self.num_bins_per_motion.unsqueeze(1),
            rounding_mode="floor",
        )
        starts = starts * self.valid_bin_mask
        ends = ends * self.valid_bin_mask
        return starts.long(), ends.long()

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(
            self.num_motions, self.num_bins, dtype=torch.float, device=self.device
        )
        self.bin_failed_count = torch.zeros(self.num_motions, self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, motion_ids: torch.Tensor, local_failed_at_time_step: torch.Tensor):
        """Update per-motion local failed-bin counts with terminated time steps."""
        if motion_ids.numel() == 0:
            return
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        local_failed_at_time_step = local_failed_at_time_step.to(device=self.device, dtype=torch.long).clamp(min=0)
        motion_bin_counts = self.num_bins_per_motion[motion_ids]
        motion_lengths = self.motion_lengths[motion_ids]
        failed_bin = (local_failed_at_time_step * motion_bin_counts) // motion_lengths.clamp(min=1)
        failed_bin = torch.minimum(
            torch.maximum(failed_bin, torch.zeros_like(failed_bin)), motion_bin_counts - 1
        ).long()
        flat_idx = motion_ids * self.num_bins + failed_bin
        counts = torch.bincount(flat_idx, minlength=self.num_motions * self.num_bins).to(torch.float)
        self.current_bin_failed_count += counts.view(self.num_motions, self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.bin_failed_count = self.bin_failed_count * self.valid_bin_mask
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + (self.adaptive_uniform_ratio * self.uniform_bin_probability)
        sampling_probabilities = sampling_probabilities * self.valid_bin_mask
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(1),
            (0, self.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).squeeze(
            1
        )
        sampling_probabilities = sampling_probabilities * self.valid_bin_mask
        return sampling_probabilities / sampling_probabilities.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def sample(self, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        probabilities = self.sampling_probabilities[motion_ids]
        sampled_bins = torch.multinomial(probabilities, 1, replacement=True).squeeze(1)
        starts = self.bin_start_local[motion_ids, sampled_bins]
        lengths = self.bin_lengths[motion_ids, sampled_bins].clamp(min=1).to(torch.float32)
        local_steps = starts.to(torch.float32) + torch.rand(motion_ids.numel(), device=self.device) * lengths
        return local_steps / self.motion_lengths[motion_ids].to(torch.float32).clamp(min=1.0)

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum(dim=1)
        H_norm = H / torch.log(self.num_bins_per_motion.to(torch.float).clamp(min=2.0))
        pmax_per_motion, imax_per_motion = prob.max(dim=1)
        pmax, motion_idx = pmax_per_motion.max(dim=0)
        imax = imax_per_motion[motion_idx]
        self.metrics["sampling_entropy"] = H_norm.mean()
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins_per_motion[motion_idx].to(torch.float)


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    _CHECKPOINT_SCHEMA = "holosoma_wbt_motion_sampler_v2"
    _PROBE_WINDOW_SIZE = 30
    _RECENT_PROBE_FIELDS = ("values", "count", "cursor")
    _PER_MOTION_CHECKPOINT_FIELDS = (
        "_normal_motion_sampling_weights",
        "_probe_recent30_values",
        "_probe_recent30_count",
        "_probe_recent30_cursor",
        "_probe_count",
        "_probe_fail_bin_hist",
        "_completion_success_streak",
        "_completion_fail_streak",
        "_completion_success_count",
        "_completion_fail_count",
        "_completion_episode_count",
        "_completion_progress_ema",
        "_completion_success_ema",
        "_completion_fail_ema",
        "_completion_learned_mask",
        "_failure_window_failure_hist",
    )

    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir or self.motion_cfg.motion_manifest, (
            "Either motion_file, motion_dir, or motion_manifest must be set in MotionConfig"
        )
        self.motion: MotionLoader | MultiMotionLoader
        if self.motion_cfg.motion_manifest:
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                motion_manifest=self.motion_cfg.motion_manifest,
                canonicalize_motion_order_on_load=bool(self.motion_cfg.canonicalize_motion_order_on_load),
            )
        elif self.motion_cfg.motion_dir:
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                canonicalize_motion_order_on_load=bool(self.motion_cfg.canonicalize_motion_order_on_load),
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                canonicalize_motion_order_on_load=bool(self.motion_cfg.canonicalize_motion_order_on_load),
            )

        if not robot_body_names_alias or robot_body_names_alias[0] != "pelvis":
            raise ValueError(f"WBT requires pelvis as simulator body 0, got {robot_body_names_alias[:1]}")
        control_fps = 1.0 / float(self._env.dt)
        if not np.isclose(float(self.motion.fps), control_fps, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                f"Motion FPS must equal WBT control frequency: motion={self.motion.fps}, control={control_fps}"
            )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)
        self._event_token_plan = self._load_event_token_plan()

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )
        self.a2a_limb_body_indexes_in_track = self._get_index_of_a_in_b(
            list(A2A_LIMB_REF_BODY_NAMES), self.motion_cfg.body_names_to_track, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACLAB3_NEWTON, (
                "Object is only supported in the IsaacLab3 Newton backend in this migration copy."
            )

        # 4. get the reset timestep sampler
        self._reset_sampler = str(self.motion_cfg.reset_sampler)
        if self._reset_sampler in ("completion_ema_failure_window", "hotspot_failure_window"):
            logger.warning(f"reset_sampler={self._reset_sampler!r} now uses completion_recent30_failure_window.")
            self._reset_sampler = "completion_recent30_failure_window"
        self._uses_failure_window_sampler = self._reset_sampler in (
            "failure_window",
            "completion_recent30_failure_window",
            "hotspot_failure_window",
        )
        self._uses_completion_recent30_failure_window_sampler = (
            self._reset_sampler
            in ("completion_recent30_failure_window", "hotspot_failure_window")
        )
        if self._reset_sampler == "hotspot_failure_window":
            logger.warning(
                "reset_sampler='hotspot_failure_window' is deprecated; use "
                "'completion_recent30_failure_window'"
            )
        self._use_completion_learning_sampler = bool(
            self.motion_cfg.use_completion_learning_sampler and int(self.motion.num_motions) > 1
        )
        if self._uses_failure_window_sampler:
            logger.info(
                "Failure-window reset sampler enabled: "
                f"mode={self._reset_sampler}, "
                f"pre_frames={int(self.motion_cfg.failure_window_pre_frames)}, "
                f"post_frames={int(self.motion_cfg.failure_window_post_frames)}, "
                f"before_prob={float(self.motion_cfg.failure_window_before_prob):.3f}, "
                f"success_horizon_frames={int(self.motion_cfg.failure_window_success_horizon_frames)}"
            )
        if self._uses_completion_recent30_failure_window_sampler:
            logger.info(
                "Recent-30 probe failure-window replay enabled: "
                f"uniform_mix={float(self.motion_cfg.hotspot_failure_uniform_mix):.3f}, "
                "center=completion_recent30_mean, window=30 completed episodes"
            )
        if self._use_completion_learning_sampler:
            logger.info(
                "Completion-learning motion sampler enabled: "
                f"success_streak_threshold={int(self.motion_cfg.completion_success_streak_threshold)}, "
                f"learned_replay_weight={float(self.motion_cfg.completion_learned_replay_weight):.3f}, "
                f"weight_beta={float(self.motion_cfg.completion_weight_beta):.3f}"
            )
        if self._reset_sampler == "adaptive":
            motion_lengths = self.motion.motion_end_idx - self.motion.motion_start_idx
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                motion_lengths,
                self.device,
                int(1 / (self._env.dt)),
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()
        self._future_joint_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._future_ref_body_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._policy_reference_window_cache: torch.Tensor | None = None

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACLAB3_NEWTON:
            self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._future_joint_cache.clear()
        self._future_ref_body_cache.clear()
        self._policy_reference_window_cache = None

        self._update_completion_learning_stats(env_ids)
        self._clear_pending_chain_checks(env_ids)
        self._clear_chain_completion_state(env_ids)
        self._update_start_probe_stats(env_ids)
        self._failure_window_reset_count.zero_()
        self._failure_window_before_count.zero_()
        self._failure_window_offset_sum.zero_()

        # 0. Update failed local bins from environments that terminated before this reset.
        if hasattr(self, "adaptive_timesteps_sampler"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                failed_env_ids = env_ids[episode_failed]
                failed_motion_ids = self.motion_ids[failed_env_ids]
                failed_start_idx = self.motion.motion_start_idx[failed_motion_ids]
                local_failed_at_time_step = self.time_steps[failed_env_ids] - failed_start_idx
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(
                    failed_motion_ids, local_failed_at_time_step
                )

        failure_window_mask = self._get_failure_window_reset_mask(env_ids)
        failure_window_env_ids = env_ids[failure_window_mask]
        failure_window_failed_time_steps = self.time_steps[failure_window_env_ids].clone()
        failure_window_failed_motion_ids = self.motion_ids[failure_window_env_ids].clone()
        normal_sampler_mask = ~failure_window_mask
        self._failure_window_retry_active[env_ids] = False

        if self._use_group_probe_envs:
            is_group_probe = self._group_probe_env_mask[env_ids]
            group_probe_env_ids = env_ids[is_group_probe]
            normal_env_ids = env_ids[~is_group_probe & normal_sampler_mask]

            if group_probe_env_ids.numel() > 0:
                group_ids = self._group_probe_group_ids[group_probe_env_ids]
                self.motion_ids[group_probe_env_ids] = self._sample_motion_ids_from_groups(group_ids)
                self._probe_episode_valid[group_probe_env_ids] = True

            if normal_env_ids.numel() > 0:
                sampled_group_ids = torch.multinomial(
                    self._normal_group_sampling_weights,
                    normal_env_ids.numel(),
                    replacement=True,
                )
                self.motion_ids[normal_env_ids] = self._sample_motion_ids_from_groups(sampled_group_ids)
                self._probe_episode_valid[normal_env_ids] = False
        elif self._use_start_probe_envs:
            is_probe = self._probe_env_mask[env_ids]
            probe_env_ids = env_ids[is_probe]
            normal_env_ids = env_ids[~is_probe & normal_sampler_mask]

            if probe_env_ids.numel() > 0:
                self.motion_ids[probe_env_ids] = self._probe_motion_ids[probe_env_ids]
                self._probe_episode_valid[probe_env_ids] = True

            if normal_env_ids.numel() > 0:
                sampled_motion_ids = torch.multinomial(
                    self._normal_motion_sampling_weights,
                    normal_env_ids.numel(),
                    replacement=True,
                )
                self.motion_ids[normal_env_ids] = sampled_motion_ids
                self._probe_episode_valid[normal_env_ids] = False
        else:
            # Original WBT sampling: choose a motion, then sample a phase inside that motion.
            sampler_env_ids = env_ids[normal_sampler_mask]
            normal_env_ids = sampler_env_ids
            n = sampler_env_ids.numel()
            num_motions = self.motion.num_motions
            if n > 0:
                if (
                    self._env.is_evaluating
                    and (bool(getattr(self.motion_cfg, "fixed_motion_assignment_in_eval", False))
                         or bool(self.motion_cfg.chain_motion_segments))
                    and float(self.motion_cfg.start_at_timestep_zero_prob) >= 1.0
                ):
                    self.motion_ids[sampler_env_ids] = self.env_motion_ids[sampler_env_ids]
                elif self._use_completion_learning_sampler:
                    self.motion_ids[sampler_env_ids] = torch.multinomial(
                        self._normal_motion_sampling_weights,
                        n,
                        replacement=True,
                    )
                else:
                    self.motion_ids[sampler_env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
            self._probe_episode_valid[env_ids] = False

        self._maybe_sample_motion_matched_origins(env_ids)
        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        if self._env.is_evaluating:
            self.time_steps[env_ids] = start_idx
        else:
            if hasattr(self, "adaptive_timesteps_sampler"):
                phase = self.adaptive_timesteps_sampler.sample(self.motion_ids[env_ids])
            else:
                phase = self.sample_reset_phase(env_ids)
            if self._use_start_probe_envs or self._use_group_probe_envs:
                phase[self._probe_env_mask[env_ids]] = 0.0
            motion_len = end_idx - start_idx
            self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()
            if self._uses_completion_recent30_failure_window_sampler:
                completion_window_env_ids = normal_env_ids
                if completion_window_env_ids.numel() > 0:
                    self.time_steps[completion_window_env_ids] = (
                        self._sample_completion_recent30_failure_window_time_steps(
                            completion_window_env_ids
                        )
                    )
            if failure_window_env_ids.numel() > 0:
                self.time_steps[failure_window_env_ids] = self._sample_failure_window_time_steps(
                    failure_window_env_ids,
                    failure_window_failed_time_steps,
                )
                self._failure_window_retry_active[failure_window_env_ids] = True
                self._failure_window_retry_motion_ids[failure_window_env_ids] = failure_window_failed_motion_ids
                self._failure_window_retry_target_steps[failure_window_env_ids] = failure_window_failed_time_steps

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        start_zero_env_ids = env_ids[normal_sampler_mask]
        start_zero_start_idx = self.motion.motion_start_idx[self.motion_ids[start_zero_env_ids]]
        if prob >= 1.0 and start_zero_env_ids.numel() > 0:
            self.time_steps[start_zero_env_ids] = start_zero_start_idx
        elif prob > 0.0 and start_zero_env_ids.numel() > 0:
            subset = self.time_steps[start_zero_env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_zero_start_idx, subset)
            self.time_steps[start_zero_env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])
        if self._use_completion_learning_sampler:
            assigned_start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
            is_start_zero = self.time_steps[env_ids] == assigned_start_idx
            is_probe = (
                self._probe_env_mask[env_ids] if hasattr(self, "_probe_env_mask") else torch.zeros_like(is_start_zero)
            )
            valid_episode = normal_sampler_mask & is_start_zero & (~is_probe)
            if self._env.is_evaluating:
                valid_episode = torch.zeros_like(valid_episode)
            self._completion_episode_valid[env_ids] = valid_episode
        elif hasattr(self, "_completion_episode_valid"):
            self._completion_episode_valid[env_ids] = False

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.reset_root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.reset_root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.reset_joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # Start probes measure whether the current policy can execute a motion
        # from its canonical first frame.  Applying reset randomization to
        # those environments makes a strict body-position threshold judge the
        # sampled initial pose before the policy has taken an action.  Keep
        # normal training environments randomized, but restore probe targets
        # to the exact reference state.
        probe_reset_mask = self._probe_env_mask[env_ids]
        if torch.any(probe_reset_mask):
            target_dof_pos[probe_reset_mask] = dof_pos[probe_reset_mask]
            target_root_pos[probe_reset_mask] = root_pos[probe_reset_mask]
            target_root_rot[probe_reset_mask] = root_rot[probe_reset_mask]
            target_root_lin_vel[probe_reset_mask] = root_lin_vel[probe_reset_mask]
            target_root_ang_vel[probe_reset_mask] = root_ang_vel[probe_reset_mask]

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise
            if torch.any(probe_reset_mask):
                target_obj_pos[probe_reset_mask] = obj_pos[probe_reset_mask]

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

        self._reset_local_segment_anchors(env_ids)
        self._sync_event_token_indices(env_ids)
        self._refresh_event_token_prev_contact(env_ids)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        self._future_joint_cache.clear()
        self._future_ref_body_cache.clear()
        self._policy_reference_window_cache = None
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()
        self._handle_event_token_switches()
        self._update_chain_touchdown_latches()

        # BeyondMimic-style behavior: when the clip ends, either resample a new
        # segment or keep rolling into the next consecutive segment.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            if (
                bool(self.motion_cfg.hold_at_motion_end_in_eval)
                and bool(self._env.is_evaluating)
                and not bool(self.motion_cfg.chain_motion_segments)
            ):
                self.time_steps[ended_env_ids] = per_motion_end[ended_env_ids] - 1
            else:
                self._handle_ended_motion_segments(ended_env_ids)

        self._handle_failure_window_success_horizon()

        self._update_relative_body_targets()
        self._update_pending_chain_checks()
        if hasattr(self, "adaptive_timesteps_sampler"):
            self.adaptive_timesteps_sampler.update_bin_failed_count()

    def _update_relative_body_targets(self) -> None:
        """Refresh relative reference poses without advancing the command clock."""

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_full_w = quat_mul(
            robot_ref_quat_w_repeat,
            quat_inverse(ref_quat_w_repeat, w_last=True),
            w_last=True,
        )
        if self.motion_cfg.relative_reference_rotation == "full":
            delta_quat_w = delta_quat_full_w
        else:
            delta_quat_w = yaw_quat(delta_quat_full_w, w_last=True)
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

    def sample_reset_phase(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Uniform by default; specialized commands may supply a phase curriculum."""
        return torch.rand(env_ids.numel(), device=self.device)

    def _handle_ended_motion_segments(self, env_ids: torch.Tensor) -> None:
        if not bool(self.motion_cfg.chain_motion_segments):
            if bool(self.motion_cfg.hold_at_motion_end_in_eval) and bool(self._env.is_evaluating):
                per_motion_end = self.motion.motion_end_idx[self.motion_ids[env_ids]]
                self.time_steps[env_ids] = per_motion_end - 1
                return
            self.reset(env_ids)
            self._flush_reset_states_to_sim(env_ids)
            return

        # Keep probe envs anchored to their assigned segment; only normal training
        # envs chain forward.
        self._update_start_probe_stats(env_ids)
        if getattr(self, "_use_start_probe_envs", False) or getattr(self, "_use_group_probe_envs", False):
            is_probe = self._probe_env_mask[env_ids]
        else:
            is_probe = torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)

        probe_env_ids = env_ids[is_probe]
        normal_env_ids = env_ids[~is_probe]
        reset_env_ids = [probe_env_ids] if probe_env_ids.numel() > 0 else []

        if normal_env_ids.numel() > 0:
            current_motion_ids = self.motion_ids[normal_env_ids]
            next_motion_ids = current_motion_ids + 1
            num_motions = int(self.motion.num_motions)
            has_next = next_motion_ids < num_motions
            safe_next_motion_ids = next_motion_ids.clamp(max=max(num_motions - 1, 0))

            consecutive = (
                self.motion.motion_start_idx[safe_next_motion_ids] == self.motion.motion_end_idx[current_motion_ids]
            )
            same_terrain = (
                self.motion.motion_terrain_ids[safe_next_motion_ids]
                == self.motion.motion_terrain_ids[current_motion_ids]
            )
            chain_mask = has_next & consecutive & same_terrain
            waiting_mask = torch.zeros_like(chain_mask)
            if bool(self.motion_cfg.require_chain_boundary_success) and bool(
                self.motion_cfg.chain_gate_before_transition
            ):
                candidate_env_ids = normal_env_ids[chain_mask]
                if candidate_env_ids.numel() > 0:
                    boundary_success = self._chain_transition_success(candidate_env_ids)
                    candidate_indices = torch.where(chain_mask)[0]
                    failed_candidate_indices = candidate_indices[~boundary_success]
                    if failed_candidate_indices.numel() > 0:
                        failed_env_ids = normal_env_ids[failed_candidate_indices]
                        failed_motion_ids = current_motion_ids[failed_candidate_indices]
                        failed_steps = self.motion.motion_end_idx[failed_motion_ids] - 1
                        self._record_adaptive_failures(failed_env_ids, failed_motion_ids, failed_steps)
                    chain_mask[candidate_indices] = boundary_success

            if bool(self.motion_cfg.chain_require_touchdown_completion):
                candidate_indices = torch.where(chain_mask)[0]
                if candidate_indices.numel() > 0:
                    candidate_env_ids = normal_env_ids[candidate_indices]
                    completion_success = self._chain_completion_success(candidate_env_ids)
                    next_wait_steps = self._chain_completion_wait_steps[candidate_env_ids] + 1
                    timed_out = next_wait_steps > max(int(self.motion_cfg.chain_completion_max_hold_steps), 0)
                    waiting = ~completion_success & ~timed_out

                    if torch.any(waiting):
                        waiting_env_ids = candidate_env_ids[waiting]
                        waiting_motion_ids = current_motion_ids[candidate_indices[waiting]]
                        self.time_steps[waiting_env_ids] = self.motion.motion_end_idx[waiting_motion_ids] - 1
                        self._chain_completion_wait_steps[waiting_env_ids] = next_wait_steps[waiting]

                    if torch.any(timed_out):
                        failed_env_ids = candidate_env_ids[timed_out]
                        failed_motion_ids = current_motion_ids[candidate_indices[timed_out]]
                        failed_steps = self.motion.motion_end_idx[failed_motion_ids] - 1
                        self._record_adaptive_failures(failed_env_ids, failed_motion_ids, failed_steps)
                        self._clear_chain_completion_state(failed_env_ids)

                    chain_mask[candidate_indices] = completion_success
                    waiting_mask[candidate_indices] = waiting

            chain_env_ids = normal_env_ids[chain_mask]
            if chain_env_ids.numel() > 0:
                chained_motion_ids = next_motion_ids[chain_mask]
                from_motion_ids = current_motion_ids[chain_mask]
                self.motion_ids[chain_env_ids] = chained_motion_ids
                self.time_steps[chain_env_ids] = self.motion.motion_start_idx[chained_motion_ids]
                self._reset_local_segment_anchors(chain_env_ids)
                self._clear_chain_completion_state(chain_env_ids)
                if not bool(self.motion_cfg.chain_gate_before_transition):
                    self._start_pending_chain_checks(chain_env_ids, from_motion_ids, chained_motion_ids)

            normal_reset_env_ids = normal_env_ids[~chain_mask & ~waiting_mask]
            if normal_reset_env_ids.numel() > 0:
                if bool(self.motion_cfg.hold_at_motion_end_in_eval) and bool(self._env.is_evaluating):
                    per_motion_end = self.motion.motion_end_idx[self.motion_ids[normal_reset_env_ids]]
                    self.time_steps[normal_reset_env_ids] = per_motion_end - 1
                    self._clear_chain_completion_state(normal_reset_env_ids)
                else:
                    reset_env_ids.append(normal_reset_env_ids)

        if reset_env_ids:
            reset_ids = torch.cat(reset_env_ids, dim=0)
            self._clear_chain_completion_state(reset_ids)
            self.reset(reset_ids)
            self._flush_reset_states_to_sim(reset_ids)

    def _clear_chain_completion_state(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_chain_completion_wait_steps"):
            return
        self._chain_completion_wait_steps[env_ids] = 0
        self._chain_contact_free_steps[env_ids] = 0
        self._chain_contact_stable_steps[env_ids] = 0
        self._chain_touchdown_latched[env_ids] = False

    def _load_event_token_plan(self) -> dict[str, Any] | None:
        plan_ref = str(getattr(self.motion_cfg, "event_token_plan", "") or "")
        if not plan_ref:
            return None
        plan_path = Path(plan_ref).expanduser()
        if not plan_path.is_absolute():
            plan_path = Path.cwd() / plan_path
        with plan_path.open("r", encoding="utf-8") as f:
            plan = json.load(f)
        if not isinstance(plan, dict) or plan.get("kind") != "contact_event_token_plan":
            raise ValueError(f"Unsupported event token plan: {plan_path}")
        tokens = plan.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Event token plan has no tokens: {plan_path}")
        logger.info(f"Loaded event token plan {plan_path} with {len(tokens)} tokens")
        return plan

    def _init_event_token_buffers(self) -> None:
        self._event_token_enabled = self._event_token_plan is not None
        self._event_token_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._event_token_prev_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._event_token_switch_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._event_token_timeout_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._event_token_contact_switch_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._event_token_last_switch_frame = torch.full((), -1.0, dtype=torch.float32, device=self.device)
        self._event_token_last_from = torch.full((), -1.0, dtype=torch.float32, device=self.device)
        self._event_token_last_to = torch.full((), -1.0, dtype=torch.float32, device=self.device)
        self._event_token_last_reason = torch.full((), -1.0, dtype=torch.float32, device=self.device)
        if not self._event_token_enabled:
            self._event_token_start = torch.zeros(1, dtype=torch.long, device=self.device)
            self._event_token_end = torch.zeros(1, dtype=torch.long, device=self.device)
            self._event_token_timeout = torch.zeros(1, dtype=torch.long, device=self.device)
            self._event_token_target_mask = torch.zeros(
                1, len(CHAIN_BOUNDARY_PART_ORDER), dtype=torch.bool, device=self.device
            )
            return

        tokens = sorted(self._event_token_plan["tokens"], key=lambda item: int(item["start_frame"]))  # type: ignore[index]
        part_index = {part: i for i, part in enumerate(CHAIN_BOUNDARY_PART_ORDER)}
        starts: list[int] = []
        ends: list[int] = []
        timeouts: list[int] = []
        masks: list[torch.Tensor] = []
        for token in tokens:
            starts.append(int(token["start_frame"]))
            ends.append(int(token["end_frame"]))
            timeout = int(token.get("timeout_frame", token["end_frame"])) + int(
                self.motion_cfg.event_token_timeout_margin_frames
            )
            timeouts.append(timeout)
            mask = torch.zeros(len(CHAIN_BOUNDARY_PART_ORDER), dtype=torch.bool, device=self.device)
            part = str(token.get("target_part", "")).upper()
            if part in part_index:
                mask[part_index[part]] = True
            masks.append(mask)
        self._event_token_start = torch.tensor(starts, dtype=torch.long, device=self.device)
        self._event_token_end = torch.tensor(ends, dtype=torch.long, device=self.device)
        self._event_token_timeout = torch.tensor(timeouts, dtype=torch.long, device=self.device)
        self._event_token_target_mask = torch.stack(masks, dim=0)

    def _sync_event_token_indices(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_event_token_enabled", False) or env_ids.numel() == 0:
            return
        token_idx = torch.searchsorted(self._event_token_end, self.time_steps[env_ids], right=True)
        self._event_token_index[env_ids] = token_idx.clamp(max=self._event_token_start.numel() - 1)

    def _event_token_target_contact(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0 or not hasattr(self._env.simulator, "contact_forces_history"):
            return torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        part_contact = self._chain_boundary_part_contact(env_ids, float(self.motion_cfg.event_token_contact_threshold))
        token_idx = self._event_token_index[env_ids].clamp(max=self._event_token_target_mask.shape[0] - 1)
        target = self._event_token_target_mask[token_idx, : part_contact.shape[1]]
        return (part_contact & target).any(dim=1)

    def _refresh_event_token_prev_contact(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_event_token_enabled", False) or env_ids.numel() == 0:
            return
        self._event_token_prev_contact[env_ids] = self._event_token_target_contact(env_ids)

    def _handle_event_token_switches(self) -> None:
        if not getattr(self, "_event_token_enabled", False):
            return
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        token_idx = self._event_token_index
        has_next = token_idx < (self._event_token_start.numel() - 1)
        if not torch.any(has_next):
            self._refresh_event_token_prev_contact(env_ids)
            return

        now_contact = self._event_token_target_contact(env_ids)
        rising_edge = now_contact & ~self._event_token_prev_contact
        timeout = self.time_steps >= self._event_token_timeout[token_idx]
        switch = has_next & (rising_edge | timeout)
        if torch.any(switch):
            switch_env_ids = env_ids[switch]
            prev_token = token_idx[switch_env_ids]
            next_token = token_idx[switch_env_ids] + 1
            switch_frames = self.time_steps[switch_env_ids].clone()
            timeout_switch = timeout[switch_env_ids]
            contact_switch = rising_edge[switch_env_ids] & ~timeout_switch
            from_motion_ids = self.motion_ids[switch_env_ids]
            self._event_token_index[switch_env_ids] = next_token
            self.time_steps[switch_env_ids] = self._event_token_start[next_token]
            self._reset_local_segment_anchors(switch_env_ids)
            self._clear_chain_completion_state(switch_env_ids)
            self._start_pending_chain_checks(switch_env_ids, from_motion_ids, self.motion_ids[switch_env_ids])
            self._event_token_switch_count += switch.to(torch.float32).sum()
            self._event_token_timeout_count += timeout_switch.to(torch.float32).sum()
            self._event_token_contact_switch_count += contact_switch.to(torch.float32).sum()

            first = 0
            self._event_token_last_switch_frame = switch_frames[first].to(torch.float32)
            self._event_token_last_from = prev_token[first].to(torch.float32)
            self._event_token_last_to = next_token[first].to(torch.float32)
            self._event_token_last_reason = torch.where(
                timeout_switch[first],
                torch.tensor(1.0, dtype=torch.float32, device=self.device),
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
            )
            logger.info(
                "event_token_switch envs={} frame={} token {}->{} reason={} total_switches={} timeouts={}",
                switch_env_ids.detach().cpu().tolist(),
                int(switch_frames[first].item()),
                int(prev_token[first].item()),
                int(next_token[first].item()),
                "timeout" if bool(timeout_switch[first].item()) else "contact",
                float(self._event_token_switch_count.item()),
                float(self._event_token_timeout_count.item()),
            )

        self._refresh_event_token_prev_contact(env_ids)

    def _update_chain_touchdown_latches(self) -> None:
        if not bool(self.motion_cfg.chain_require_touchdown_completion):
            return
        if not hasattr(self._env.simulator, "contact_forces_history"):
            return
        part_contact = self._chain_boundary_part_contact(
            torch.arange(self.num_envs, device=self.device),
            float(self.motion_cfg.chain_boundary_contact_threshold),
        )
        prev_free_steps = self._chain_contact_free_steps.clone()
        self._chain_contact_stable_steps = torch.where(
            part_contact,
            self._chain_contact_stable_steps + 1,
            torch.zeros_like(self._chain_contact_stable_steps),
        )
        self._chain_contact_free_steps = torch.where(
            part_contact,
            torch.zeros_like(self._chain_contact_free_steps),
            self._chain_contact_free_steps + 1,
        )

        expected = self._motion_touchdown_mask_for_envs(torch.arange(self.num_envs, device=self.device))
        if not torch.any(expected):
            return
        motion_ids = self.motion_ids
        exit_window = max(int(self.motion_cfg.chain_completion_exit_window_steps), 0)
        exit_start = self.motion.motion_end_idx[motion_ids] - max(exit_window, 1)
        in_exit_window = self.time_steps >= exit_start
        free_window = max(int(self.motion_cfg.chain_touchdown_free_window_steps), 0)
        stable_steps = max(int(self.motion_cfg.chain_touchdown_stable_steps), 1)
        touchdown = (prev_free_steps >= free_window) & (self._chain_contact_stable_steps >= stable_steps)
        self._chain_touchdown_latched |= in_exit_window.unsqueeze(1) & expected & touchdown

    def _motion_touchdown_mask_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        mask = getattr(self.motion, "motion_touchdown_part_mask", None)
        if mask is None:
            return torch.zeros(env_ids.numel(), len(CHAIN_BOUNDARY_PART_ORDER), dtype=torch.bool, device=self.device)
        return mask[self.motion_ids[env_ids], : len(CHAIN_BOUNDARY_PART_ORDER)].to(torch.bool)

    def _chain_completion_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        success = self._chain_transition_success(env_ids)
        expected = self._motion_touchdown_mask_for_envs(env_ids)
        has_expected_touchdown = expected.any(dim=1)
        if not torch.any(has_expected_touchdown):
            return success

        stable_steps = max(int(self.motion_cfg.chain_touchdown_stable_steps), 1)
        stable_contact = self._chain_contact_stable_steps[env_ids] >= stable_steps
        touchdown_ok_by_part = self._chain_touchdown_latched[env_ids] | stable_contact
        touchdown_ok = torch.all(~expected | touchdown_ok_by_part, dim=1)
        return success & (~has_expected_touchdown | touchdown_ok)

    def _handle_failure_window_success_horizon(self) -> None:
        if not self._uses_failure_window_sampler:
            return
        horizon = int(self.motion_cfg.failure_window_success_horizon_frames)
        if horizon <= 0:
            return

        retry_env_ids = torch.where(self._failure_window_retry_active)[0]
        if retry_env_ids.numel() == 0:
            return

        same_motion = self.motion_ids[retry_env_ids] == self._failure_window_retry_motion_ids[retry_env_ids]
        target_steps = self._failure_window_retry_target_steps[retry_env_ids]
        reached = self.time_steps[retry_env_ids] >= target_steps + horizon
        success_env_ids = retry_env_ids[same_motion & reached]
        if success_env_ids.numel() == 0:
            return

        self.reset(success_env_ids)
        self._flush_reset_states_to_sim(success_env_ids)

    def _flush_reset_states_to_sim(self, env_ids: torch.Tensor) -> None:
        # Flush mutated root/dof state into the simulator so rigid-body positions
        # are current for termination checks, observations, and rewards.
        sim = self._env.simulator
        sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)  # type: ignore[attr-defined]
        sim.refresh_sim_tensors()

    def _record_adaptive_failures(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0 or not hasattr(self, "adaptive_timesteps_sampler"):
            return
        failed_motion_ids = self.motion_ids[env_ids] if motion_ids is None else motion_ids
        failed_start_idx = self.motion.motion_start_idx[failed_motion_ids]
        failed_end_idx = self.motion.motion_end_idx[failed_motion_ids]
        failed_time_steps = self.time_steps[env_ids] if time_steps is None else time_steps
        failed_time_steps = torch.minimum(failed_time_steps, failed_end_idx - 1)
        local_failed_at_time_step = failed_time_steps - failed_start_idx
        self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_motion_ids, local_failed_at_time_step)

    def _start_pending_chain_checks(
        self,
        env_ids: torch.Tensor,
        from_motion_ids: torch.Tensor,
        to_motion_ids: torch.Tensor,
    ) -> None:
        if not bool(self.motion_cfg.require_chain_boundary_success) or env_ids.numel() == 0:
            return
        self._pending_chain_check[env_ids] = True
        self._pending_chain_from_motion_ids[env_ids] = from_motion_ids
        self._pending_chain_to_motion_ids[env_ids] = to_motion_ids
        self._pending_chain_success_steps[env_ids] = 0
        self._pending_chain_deadline_steps[env_ids] = self.time_steps[env_ids] + int(
            self.motion_cfg.chain_transition_grace_steps
        )

    def _update_pending_chain_checks(self) -> None:
        if not bool(self.motion_cfg.require_chain_boundary_success):
            return
        env_ids = torch.where(self._pending_chain_check)[0]
        if env_ids.numel() == 0:
            return

        success = self._chain_transition_success(env_ids)
        self._pending_chain_success_steps[env_ids] = torch.where(
            success,
            self._pending_chain_success_steps[env_ids] + 1,
            torch.zeros_like(self._pending_chain_success_steps[env_ids]),
        )

        required = max(int(self.motion_cfg.chain_transition_success_window), 1)
        accepted = self._pending_chain_success_steps[env_ids] >= required
        if torch.any(accepted):
            accepted_env_ids = env_ids[accepted]
            self._clear_pending_chain_checks(accepted_env_ids)

        remaining_env_ids = env_ids[~accepted]
        if remaining_env_ids.numel() == 0:
            return
        expired = self.time_steps[remaining_env_ids] >= self._pending_chain_deadline_steps[remaining_env_ids]
        if not torch.any(expired):
            return

        failed_env_ids = remaining_env_ids[expired]
        from_motion_ids = self._pending_chain_from_motion_ids[failed_env_ids]
        from_end_steps = self.motion.motion_end_idx[from_motion_ids] - 1
        self._record_adaptive_failures(failed_env_ids, from_motion_ids, from_end_steps)
        self._record_adaptive_failures(failed_env_ids)
        self._clear_pending_chain_checks(failed_env_ids)
        self.reset(failed_env_ids)
        self._flush_reset_states_to_sim(failed_env_ids)

    def _clear_pending_chain_checks(self, env_ids: torch.Tensor) -> None:
        self._pending_chain_check[env_ids] = False
        self._pending_chain_from_motion_ids[env_ids] = -1
        self._pending_chain_to_motion_ids[env_ids] = -1
        self._pending_chain_deadline_steps[env_ids] = 0
        self._pending_chain_success_steps[env_ids] = 0

    def _chain_transition_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not bool(self.motion_cfg.require_chain_boundary_success):
            return torch.ones(env_ids.shape, dtype=torch.bool, device=self.device)

        success = torch.ones(env_ids.shape, dtype=torch.bool, device=self.device)

        threshold = float(self.motion_cfg.chain_boundary_tracking_threshold)
        limb_idx = self.a2a_limb_body_indexes_in_track
        if threshold > 0.0 and limb_idx.numel() > 0:
            z_error = torch.abs(
                self.body_pos_relative_w[env_ids][:, limb_idx, -1] - self.robot_body_pos_w[env_ids][:, limb_idx, -1]
            )
            success &= torch.all(z_error <= threshold, dim=1)

        contact_threshold = float(self.motion_cfg.chain_boundary_contact_threshold)
        if contact_threshold > 0.0 and hasattr(self._env.simulator, "contact_forces_history"):
            part_contact = self._chain_boundary_part_contact(env_ids, contact_threshold)
            motion_ids = self.motion_ids[env_ids]
            boundary_steps = self.time_steps[env_ids].clamp(max=(self.motion.motion_end_idx[motion_ids] - 1))
            expected_support = self.motion.support_part_mask[boundary_steps, : part_contact.shape[1]].to(torch.bool)
            has_expected_support = expected_support.any(dim=1)
            support_contact_ok = torch.all(~expected_support | part_contact, dim=1)
            success &= ~has_expected_support | support_contact_ok

        return success

    def _chain_boundary_part_contact(self, env_ids: torch.Tensor, threshold: float) -> torch.Tensor:
        body_names = list(self._env.simulator.body_names)  # type: ignore[attr-defined]
        body_force = torch.norm(self._env.simulator.contact_forces_history[env_ids], dim=-1).max(dim=1)[0]
        body_contact = body_force > threshold
        contacts = []
        for part_body_names in CHAIN_BOUNDARY_CONTACT_BODY_NAMES:
            body_ids = [body_names.index(name) for name in part_body_names if name in body_names]
            if not body_ids:
                contacts.append(torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device))
            else:
                contacts.append(body_contact[:, body_ids].any(dim=1))
        return torch.stack(contacts, dim=1)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def reset_joint_vel(self) -> torch.Tensor:
        return self.motion.sim_joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        pos_w = self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
        quat_w = self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]
        pos_w, _ = self._localize_motion_pos_quat(pos_w, quat_w)
        return pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        pos_w = self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
        quat_w = self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]
        _, quat_w = self._localize_motion_pos_quat(pos_w, quat_w)
        return quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._localize_motion_velocity(self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._localize_motion_velocity(self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes])

    @property
    def ref_pos_w(self) -> torch.Tensor:
        pos_w = self.motion.body_pos_w[self.time_steps, self.ref_body_index][:, None, :]
        quat_w = self.motion.body_quat_w[self.time_steps, self.ref_body_index][:, None, :]
        pos_w, _ = self._localize_motion_pos_quat(pos_w, quat_w)
        return pos_w[:, 0, :]

    @property
    def ref_quat_w(self) -> torch.Tensor:
        pos_w = self.motion.body_pos_w[self.time_steps, self.ref_body_index][:, None, :]
        quat_w = self.motion.body_quat_w[self.time_steps, self.ref_body_index][:, None, :]
        _, quat_w = self._localize_motion_pos_quat(pos_w, quat_w)
        return quat_w[:, 0, :]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        vel_w = self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index][:, None, :]
        return self._localize_motion_velocity(vel_w)[:, 0, :]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        vel_w = self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index][:, None, :]
        return self._localize_motion_velocity(vel_w)[:, 0, :]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, 0] + self._reference_env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, 0]

    @property
    def reset_root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.sim_root_lin_vel[self.time_steps]

    @property
    def reset_root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.sim_root_ang_vel[self.time_steps]

    @property
    def active_part_mask(self) -> torch.Tensor:
        return self.motion.active_part_mask[self.time_steps]

    @property
    def support_part_mask(self) -> torch.Tensor:
        """Compatibility name for declared contact-keep intent, not measured support."""
        return self.motion.support_part_mask[self.time_steps]

    @property
    def free_part_mask(self) -> torch.Tensor:
        return self.motion.free_part_mask[self.time_steps]

    @property
    def contact_part_mask(self) -> torch.Tensor:
        return self.motion.contact_part_mask[self.time_steps]

    @property
    def contact_force_part_w(self) -> torch.Tensor:
        return self.motion.contact_force_part_w[self.time_steps]

    @property
    def contact_force_part_mask(self) -> torch.Tensor:
        return self.motion.contact_force_part_mask[self.time_steps]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._reference_env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def _build_motion_group_source_ids(self, num_motions: int) -> torch.Tensor:
        group_by = str(getattr(self.motion_cfg, "group_probe_by", "terrain_id"))
        if group_by == "terrain_id":
            return self.motion.motion_terrain_ids.to(device=self.device, dtype=torch.long)
        if group_by == "climb_id":
            motion_files = getattr(self.motion, "motion_files", [])
            terrain_ids = self.motion.motion_terrain_ids.detach().to("cpu")
            labels: list[int] = []
            for motion_id in range(num_motions):
                motion_file = motion_files[motion_id] if motion_id < len(motion_files) else ""
                match = re.search(r"climb[_-](\d+)", str(motion_file))
                if match is not None:
                    labels.append(int(match.group(1)))
                else:
                    labels.append(int(terrain_ids[motion_id].item()))
            return torch.tensor(labels, dtype=torch.long, device=self.device)
        raise ValueError(f"Unsupported group_probe_by={group_by!r}")

    def _init_motion_groups(self, num_motions: int, base_motion_weights: torch.Tensor) -> None:
        source_ids = self._build_motion_group_source_ids(num_motions)
        group_labels, motion_group_ids = torch.unique(source_ids, sorted=True, return_inverse=True)
        self._group_probe_by = str(getattr(self.motion_cfg, "group_probe_by", "terrain_id"))
        self._group_labels = group_labels.to(device=self.device, dtype=torch.long)
        self._motion_group_ids = motion_group_ids.to(device=self.device, dtype=torch.long)
        self._num_motion_groups = int(self._group_labels.numel())
        self._group_motion_mask = torch.zeros(
            self._num_motion_groups, num_motions, dtype=torch.bool, device=self.device
        )
        self._group_motion_mask[self._motion_group_ids, torch.arange(num_motions, device=self.device)] = True
        self._group_motion_counts = self._group_motion_mask.sum(dim=1).clamp(min=1).to(torch.long)
        max_group_motion_count = int(self._group_motion_counts.max().item())
        self._group_motion_ids_padded = torch.zeros(
            self._num_motion_groups, max_group_motion_count, dtype=torch.long, device=self.device
        )
        for group_id in range(self._num_motion_groups):
            motion_ids = torch.where(self._motion_group_ids == group_id)[0]
            self._group_motion_ids_padded[group_id, : motion_ids.numel()] = motion_ids
        self._group_variant_sample_count = max(int(getattr(self.motion_cfg, "group_variant_sample_count", 0)), 0)
        if self._group_variant_sample_count > 0:
            requested = torch.full_like(self._group_motion_counts, self._group_variant_sample_count)
            self._group_variant_effective_counts = torch.minimum(self._group_motion_counts, requested)
        else:
            self._group_variant_effective_counts = self._group_motion_counts.clone()
        group_weights = torch.zeros(self._num_motion_groups, dtype=torch.float32, device=self.device)
        group_weights.index_add_(0, self._motion_group_ids, base_motion_weights.to(dtype=torch.float32))
        self._normal_group_sampling_weights = group_weights / group_weights.sum().clamp(min=1e-8)

    def _refresh_normal_motion_weights_from_group_weights(self) -> None:
        per_motion = self._normal_group_sampling_weights[self._motion_group_ids] / self._group_motion_counts[
            self._motion_group_ids
        ].to(torch.float32)
        self._normal_motion_sampling_weights[:] = per_motion / per_motion.sum().clamp(min=1e-8)

    def _sample_motion_ids_from_groups(self, group_ids: torch.Tensor) -> torch.Tensor:
        group_ids = group_ids.to(device=self.device, dtype=torch.long)
        variant_count = int(getattr(self, "_group_variant_sample_count", 0))
        if variant_count <= 0:
            counts = self._group_motion_counts[group_ids]
            draws = torch.floor(torch.rand(group_ids.shape, device=self.device) * counts.to(torch.float32)).to(
                torch.long
            )
            return self._group_motion_ids_padded[group_ids, draws]

        sampled_motion_ids = torch.empty(group_ids.shape, dtype=torch.long, device=self.device)
        for group_id in group_ids.unique():
            selected = group_ids == group_id
            candidate_motion_ids = torch.where(self._motion_group_ids == group_id)[0]
            if variant_count > 0 and candidate_motion_ids.numel() > variant_count:
                active_idx = torch.randperm(candidate_motion_ids.numel(), device=self.device)[:variant_count]
                candidate_motion_ids = candidate_motion_ids[active_idx]
            draws = torch.randint(0, candidate_motion_ids.numel(), (int(selected.sum().item()),), device=self.device)
            sampled_motion_ids[selected] = candidate_motion_ids[draws]
        return sampled_motion_ids

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        num_motions = max(int(self.motion.num_motions), 1)
        self.env_motion_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device) % num_motions
        self.motion_ids = self.env_motion_ids.clone()
        base_motion_weights = getattr(
            self.motion,
            "motion_sampling_weights",
            torch.full((num_motions,), 1.0 / num_motions, dtype=torch.float32, device=self.device),
        ).to(device=self.device, dtype=torch.float32)
        self._base_motion_sampling_weights = base_motion_weights / base_motion_weights.sum().clamp(min=1e-8)
        self._normal_motion_sampling_weights = self._base_motion_sampling_weights.clone()
        self._init_motion_groups(num_motions, self._normal_motion_sampling_weights)

        group_probe_requested = bool(self.motion_cfg.use_group_probe_envs and num_motions > 1)
        if group_probe_requested and bool(self.motion_cfg.use_start_probe_envs):
            logger.info("Group-probe envs enabled; per-motion start probes are disabled for this run.")
        configured_probe_per_group = max(int(self.motion_cfg.probe_env_per_group), 0)
        probe_per_group = configured_probe_per_group if group_probe_requested else 0
        if group_probe_requested and self._num_motion_groups * probe_per_group > self.num_envs:
            probe_per_group = self.num_envs // self._num_motion_groups
            logger.warning(
                "Requested group probe envs exceed num_envs; clamping probe_env_per_group "
                f"from {configured_probe_per_group} to {probe_per_group}."
            )
        self._group_probe_env_per_group = probe_per_group
        self._num_group_probe_envs = self._num_motion_groups * probe_per_group
        self._use_group_probe_envs = bool(group_probe_requested and self._num_group_probe_envs > 0)

        probe_requested = bool(self.motion_cfg.use_start_probe_envs and not self._use_group_probe_envs)
        configured_probe_per_motion = max(int(self.motion_cfg.probe_env_per_motion), 0)
        probe_per_motion = configured_probe_per_motion if probe_requested else 0
        if probe_requested and num_motions * probe_per_motion > self.num_envs:
            probe_per_motion = self.num_envs // num_motions
            logger.warning(
                "Requested start probe envs exceed num_envs; clamping probe_env_per_motion "
                f"from {configured_probe_per_motion} to {probe_per_motion}."
            )
        self._probe_env_per_motion = probe_per_motion
        self._num_probe_envs = num_motions * probe_per_motion
        self._use_start_probe_envs = bool(probe_requested and self._num_probe_envs > 0)
        self._probe_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._probe_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._group_probe_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._group_probe_group_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        if self._use_start_probe_envs:
            probe_env_ids = torch.arange(self._num_probe_envs, dtype=torch.long, device=self.device)
            self._probe_env_mask[probe_env_ids] = True
            self._probe_motion_ids[probe_env_ids] = torch.arange(
                num_motions, dtype=torch.long, device=self.device
            ).repeat_interleave(probe_per_motion)
            logger.info(
                "Start-probe envs enabled: "
                f"{probe_per_motion} envs/motion, {self._num_probe_envs} probe envs, "
                f"{self.num_envs - self._num_probe_envs} normal envs."
            )
        elif self._use_group_probe_envs:
            probe_env_ids = torch.arange(self._num_group_probe_envs, dtype=torch.long, device=self.device)
            self._probe_env_mask[probe_env_ids] = True
            self._group_probe_env_mask[probe_env_ids] = True
            self._group_probe_group_ids[probe_env_ids] = torch.arange(
                self._num_motion_groups, dtype=torch.long, device=self.device
            ).repeat_interleave(probe_per_group)
            self._refresh_normal_motion_weights_from_group_weights()
            logger.info(
                "Group-probe envs enabled: "
                f"group_by={self._group_probe_by}, {probe_per_group} envs/group, "
                f"{self._num_motion_groups} groups, {self._num_group_probe_envs} probe envs, "
                f"{self.num_envs - self._num_group_probe_envs} normal envs, "
                f"group_variant_sample_count={self._group_variant_sample_count}."
            )
        self._probe_episode_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._probe_completion_recent30 = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._probe_success_recent30 = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._probe_fail_recent30 = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._probe_timeout_recent30 = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._probe_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._probe_fail_bin_count = max(int(self.motion_cfg.probe_fail_bin_count), 1)
        self._probe_fail_bin_hist = torch.zeros(
            num_motions, self._probe_fail_bin_count, dtype=torch.float32, device=self.device
        )
        self._completion_episode_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._completion_success_streak = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._completion_fail_streak = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._completion_success_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._completion_fail_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._completion_episode_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._completion_progress_ema = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._completion_success_ema = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._completion_fail_ema = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        self._completion_learned_mask = torch.zeros(num_motions, dtype=torch.bool, device=self.device)
        self._group_probe_completion_recent30 = torch.zeros(self._num_motion_groups, dtype=torch.float32, device=self.device)
        self._group_probe_success_recent30 = torch.zeros(self._num_motion_groups, dtype=torch.float32, device=self.device)
        self._group_probe_fail_recent30 = torch.zeros(self._num_motion_groups, dtype=torch.float32, device=self.device)
        self._group_probe_timeout_recent30 = torch.zeros(self._num_motion_groups, dtype=torch.float32, device=self.device)
        self._group_probe_count = torch.zeros(self._num_motion_groups, dtype=torch.long, device=self.device)
        self._init_recent_probe_window()
        self._init_recent_probe_window(group=True)
        self._group_probe_fail_bin_hist = torch.zeros(
            self._num_motion_groups, self._probe_fail_bin_count, dtype=torch.float32, device=self.device
        )
        self._failure_window_reset_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._failure_window_before_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._failure_window_offset_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._failure_window_retry_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure_window_retry_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._failure_window_retry_target_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._failure_window_log_bin_frames = max(int(self.motion_cfg.failure_window_log_bin_frames), 1)
        motion_lengths = (self.motion.motion_end_idx - self.motion.motion_start_idx).clamp(min=1)
        self._failure_window_log_bin_count = int(
            torch.div(
                motion_lengths.max() + self._failure_window_log_bin_frames - 1,
                self._failure_window_log_bin_frames,
                rounding_mode="floor",
            ).item()
        )
        self._failure_window_failure_hist = torch.zeros(
            num_motions, self._failure_window_log_bin_count, dtype=torch.float32, device=self.device
        )
        self._pending_chain_check = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_chain_from_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pending_chain_to_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pending_chain_deadline_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pending_chain_success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        num_chain_parts = len(CHAIN_BOUNDARY_PART_ORDER)
        self._chain_completion_wait_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._chain_contact_free_steps = torch.zeros(
            self.num_envs, num_chain_parts, dtype=torch.long, device=self.device
        )
        self._chain_contact_stable_steps = torch.zeros(
            self.num_envs, num_chain_parts, dtype=torch.long, device=self.device
        )
        self._chain_touchdown_latched = torch.zeros(
            self.num_envs, num_chain_parts, dtype=torch.bool, device=self.device
        )
        self._init_event_token_buffers()
        self._local_segment_anchor_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._local_segment_motion_ref_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._local_segment_motion_ref_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._local_segment_motion_ref_quat_w[:, 3] = 1.0
        self._local_segment_robot_ref_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._local_segment_robot_ref_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._local_segment_robot_ref_quat_w[:, 3] = 1.0
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if hasattr(self, "adaptive_timesteps_sampler"):
            self.adaptive_timesteps_sampler.init_buffers()

    def _init_recent_probe_window(self, *, group: bool = False) -> None:
        prefix = "_group_probe" if group else "_probe"
        count = getattr(self, prefix + "_count").numel()
        setattr(self, prefix + "_recent30_values", torch.zeros(
            count, self._PROBE_WINDOW_SIZE, 4, device=self.device
        ))
        for field in ("count", "cursor"):
            setattr(self, prefix + "_recent30_" + field, torch.zeros(count, dtype=torch.long, device=self.device))

    def _refresh_recent_probe_means(self, *, group: bool = False) -> None:
        prefix = "_group_probe" if group else "_probe"
        means = getattr(self, prefix + "_recent30_values").sum(dim=1) / getattr(
            self, prefix + "_recent30_count"
        ).clamp(min=1).unsqueeze(1)
        for column, metric in enumerate(("completion", "success", "fail", "timeout")):
            getattr(self, prefix + "_" + metric + "_recent30").copy_(means[:, column])

    def _append_recent_probe_results(self, index: int, completion: torch.Tensor, success: torch.Tensor,
                                     fail: torch.Tensor, timeout: torch.Tensor, *, group: bool = False) -> None:
        """Count individual finished episodes, including simultaneous probe finishes."""
        prefix = "_group_probe" if group else "_probe"
        if not hasattr(self, prefix + "_recent30_values"):
            self._init_recent_probe_window(group=group)
        values = getattr(self, prefix + "_recent30_values")
        counts = getattr(self, prefix + "_recent30_count")
        cursors = getattr(self, prefix + "_recent30_cursor")
        n = completion.numel()
        kept = min(n, self._PROBE_WINDOW_SIZE)
        if not kept:
            return
        offsets = torch.arange(n - kept, n, device=values.device)
        slots = (cursors[index] + offsets) % self._PROBE_WINDOW_SIZE
        outcomes = torch.stack((completion, success, fail, timeout), dim=-1).to(values.dtype)
        values[index, slots] = outcomes[-kept:]
        cursors[index] = (cursors[index] + n) % self._PROBE_WINDOW_SIZE
        counts[index] = (counts[index] + n).clamp(max=self._PROBE_WINDOW_SIZE)
        mean = values[index].sum(dim=0) / counts[index].clamp(min=1)
        for column, metric in enumerate(("completion", "success", "fail", "timeout")):
            getattr(self, prefix + "_" + metric + "_recent30")[index] = mean[column]

    def _update_start_probe_stats(
        self,
        env_ids: torch.Tensor,
        force_fail_mask: torch.Tensor | None = None,
    ) -> None:
        if not getattr(self, "_use_start_probe_envs", False) and not getattr(self, "_use_group_probe_envs", False):
            return
        is_valid_probe = self._probe_env_mask[env_ids] & self._probe_episode_valid[env_ids]
        if not torch.any(is_valid_probe):
            return

        probe_env_ids = env_ids[is_valid_probe]
        if force_fail_mask is None:
            force_fail = torch.zeros(probe_env_ids.shape, dtype=torch.bool, device=self.device)
        else:
            force_fail = force_fail_mask[is_valid_probe].to(torch.bool)
        motion_ids = self.motion_ids[probe_env_ids]
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        motion_len = (end_idx - start_idx - 1).clamp(min=1).to(torch.float32)
        elapsed = (self.time_steps[probe_env_ids] - start_idx).to(torch.float32)
        completion = (elapsed / motion_len).clamp(0.0, 1.0)

        terminated = self._env.termination_manager.terminated[probe_env_ids]
        time_outs = self._env.termination_manager.time_outs[probe_env_ids]
        motion_end_done = self._env.termination_manager.term_dones.get("motion_ends")
        if motion_end_done is None:
            motion_end_done = torch.zeros_like(terminated)
        else:
            motion_end_done = motion_end_done[probe_env_ids].to(torch.bool)
        failure_terminated = terminated & ~motion_end_done
        fixed_qualified = torch.ones_like(failure_terminated)
        curriculum_manager = getattr(self._env, "curriculum_manager", None)
        tracking_precision = (
            curriculum_manager.get_term("tracking_precision") if curriculum_manager is not None else None
        )
        completed_probe_qualification = getattr(tracking_precision, "completed_probe_qualification", None)
        if callable(completed_probe_qualification):
            fixed_qualified = completed_probe_qualification(probe_env_ids).to(device=self.device, dtype=torch.bool)
        fail = failure_terminated | force_fail | ~fixed_qualified
        timeout = time_outs
        success = (completion >= 0.98) & fixed_qualified & ~fail & ~timeout

        if getattr(self, "_use_group_probe_envs", False):
            group_ids = self._group_probe_group_ids[probe_env_ids]
            valid_group = group_ids >= 0
            if torch.any(valid_group):
                group_ids = group_ids[valid_group]
                completion = completion[valid_group]
                success = success[valid_group]
                fail = fail[valid_group]
                timeout = timeout[valid_group]
                for group_id in group_ids.unique():
                    mask = group_ids == group_id
                    g = int(group_id.item())
                    self._append_recent_probe_results(
                        g, completion[mask], success[mask], fail[mask], timeout[mask], group=True
                    )
                    self._group_probe_count[g] += int(mask.sum().item())

                    failed_completion = completion[mask & fail]
                    if failed_completion.numel() > 0:
                        fail_bins = torch.clamp(
                            (failed_completion * self._probe_fail_bin_count).long(),
                            0,
                            self._probe_fail_bin_count - 1,
                        )
                        self._group_probe_fail_bin_hist[g] += torch.bincount(
                            fail_bins, minlength=self._probe_fail_bin_count
                        ).to(self._group_probe_fail_bin_hist.dtype)
            self._probe_episode_valid[probe_env_ids] = False
            return

        for motion_id in motion_ids.unique():
            mask = motion_ids == motion_id
            m = int(motion_id.item())
            self._append_recent_probe_results(m, completion[mask], success[mask], fail[mask], timeout[mask])
            self._probe_count[m] += int(mask.sum().item())

            failed_completion = completion[mask & fail]
            if failed_completion.numel() > 0:
                fail_bins = torch.clamp(
                    (failed_completion * self._probe_fail_bin_count).long(),
                    0,
                    self._probe_fail_bin_count - 1,
                )
                self._probe_fail_bin_hist[m] += torch.bincount(fail_bins, minlength=self._probe_fail_bin_count).to(
                    self._probe_fail_bin_hist.dtype
                )

        self._probe_episode_valid[probe_env_ids] = False

    def _update_completion_learning_stats(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_use_completion_learning_sampler", False):
            return
        valid = self._completion_episode_valid[env_ids]
        if not torch.any(valid):
            return

        episode_env_ids = env_ids[valid]
        motion_ids = self.motion_ids[episode_env_ids]
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        motion_len = (end_idx - start_idx - 1).clamp(min=1).to(torch.float32)
        elapsed = (self.time_steps[episode_env_ids] - start_idx).to(torch.float32)
        completion = (elapsed / motion_len).clamp(0.0, 1.0)

        bad_tracking = _get_bad_tracking_done_mask(self._env.termination_manager.term_dones, episode_env_ids)
        if bad_tracking is None:
            bad_tracking = torch.zeros(episode_env_ids.shape, dtype=torch.bool, device=self.device)
        timeout = self._env.termination_manager.time_outs[episode_env_ids].to(torch.bool)
        terminated = self._env.termination_manager.terminated[episode_env_ids].to(torch.bool)
        motion_end_done = self._env.termination_manager.term_dones.get("motion_ends")
        if motion_end_done is None:
            motion_end_done = torch.zeros(episode_env_ids.shape, dtype=torch.bool, device=self.device)
        else:
            motion_end_done = motion_end_done[episode_env_ids].to(torch.bool)

        reached_motion_end = (self.time_steps[episode_env_ids] >= (end_idx - 1)) | motion_end_done
        failure = bad_tracking | timeout | (terminated & ~motion_end_done)
        success = reached_motion_end & ~failure
        alpha = float(self.motion_cfg.probe_completion_alpha)
        threshold = max(int(self.motion_cfg.completion_success_streak_threshold), 1)

        for motion_id in motion_ids.unique():
            mask = motion_ids == motion_id
            m = int(motion_id.item())
            success_count = int(success[mask].to(torch.long).sum().item())
            fail_count = int(failure[mask].to(torch.long).sum().item())
            episode_count = int(mask.sum().item())
            progress_mean = completion[mask].mean()
            success_mean = success[mask].to(torch.float32).mean()
            fail_mean = failure[mask].to(torch.float32).mean()

            self._completion_episode_count[m] += episode_count
            self._completion_success_count[m] += success_count
            self._completion_fail_count[m] += fail_count
            self._completion_progress_ema[m] = (1.0 - alpha) * self._completion_progress_ema[m] + alpha * progress_mean
            self._completion_success_ema[m] = (1.0 - alpha) * self._completion_success_ema[m] + alpha * success_mean
            self._completion_fail_ema[m] = (1.0 - alpha) * self._completion_fail_ema[m] + alpha * fail_mean

            for episode_success, episode_failure in zip(success[mask].tolist(), failure[mask].tolist()):
                if episode_success:
                    self._completion_success_streak[m] += 1
                    self._completion_fail_streak[m] = 0
                    if int(self._completion_success_streak[m].item()) >= threshold:
                        self._completion_learned_mask[m] = True
                elif episode_failure:
                    self._completion_fail_streak[m] += 1
                    if not bool(self._completion_learned_mask[m].item()):
                        self._completion_success_streak[m] = 0

        self._completion_episode_valid[episode_env_ids] = False

    def update_probe_motion_sampling_weights(self) -> None:
        if getattr(self, "_use_completion_learning_sampler", False):
            learned = self._completion_learned_mask
            base = self._base_motion_sampling_weights
            if torch.all(learned):
                target = base.clone()
            else:
                replay = min(max(float(self.motion_cfg.completion_learned_replay_weight), 0.0), 1.0)
                target = base.clone()
                target[learned] = target[learned] * replay
                target = target / target.sum().clamp(min=1e-8)
            beta = min(max(float(self.motion_cfg.completion_weight_beta), 0.0), 1.0)
            self._normal_motion_sampling_weights[:] = (
                1.0 - beta
            ) * self._normal_motion_sampling_weights + beta * target
            self._normal_motion_sampling_weights[:] = self._normal_motion_sampling_weights / (
                self._normal_motion_sampling_weights.sum().clamp(min=1e-8)
            )

        if getattr(self, "_use_group_probe_envs", False):
            difficulty = 1.0 - self._group_probe_success_recent30
            temperature = max(float(self.motion_cfg.probe_priority_temperature), 1e-6)
            priority = torch.softmax(difficulty / temperature, dim=0)
            uniform = torch.full_like(priority, 1.0 / max(priority.numel(), 1))
            uniform_mix = float(self.motion_cfg.probe_uniform_mix)
            target = uniform_mix * uniform + (1.0 - uniform_mix) * priority
            beta = float(self.motion_cfg.probe_weight_beta)
            self._normal_group_sampling_weights[:] = (1.0 - beta) * self._normal_group_sampling_weights + beta * target
            self._normal_group_sampling_weights[:] = self._normal_group_sampling_weights / (
                self._normal_group_sampling_weights.sum().clamp(min=1e-8)
            )
            self._refresh_normal_motion_weights_from_group_weights()
            return
        if not getattr(self, "_use_start_probe_envs", False):
            return
        difficulty = 1.0 - self._probe_success_recent30
        temperature = max(float(self.motion_cfg.probe_priority_temperature), 1e-6)
        priority = torch.softmax(difficulty / temperature, dim=0)
        uniform = torch.full_like(priority, 1.0 / max(priority.numel(), 1))
        uniform_mix = float(self.motion_cfg.probe_uniform_mix)
        target = uniform_mix * uniform + (1.0 - uniform_mix) * priority
        beta = float(self.motion_cfg.probe_weight_beta)
        self._normal_motion_sampling_weights[:] = (1.0 - beta) * self._normal_motion_sampling_weights + beta * target
        self._normal_motion_sampling_weights[:] = self._normal_motion_sampling_weights / (
            self._normal_motion_sampling_weights.sum().clamp(min=1e-8)
        )

    def _checkpoint_motion_keys(self) -> list[str]:
        from somaforge_core.published_data import relocate_published_values

        paths = relocate_published_values(self.motion.motion_files)
        return [str(Path(path).expanduser().resolve()) for path in paths]

    def _checkpoint_group_keys(self) -> list[str]:
        motion_keys = self._checkpoint_motion_keys()
        groups = self._motion_group_ids.detach().cpu().tolist()
        return [json.dumps(sorted(key for key, group in zip(motion_keys, groups) if group == index))
                for index in range(self._num_motion_groups)]

    def _valid_recent_probe_checkpoint(self, tensors: dict, prefix: str, rows: int) -> bool:
        """Restore the raw window atomically; EMA summaries cannot seed it."""
        for suffix, shape in (("values", (rows, self._PROBE_WINDOW_SIZE, 4)),
                              ("count", (rows,)), ("cursor", (rows,))):
            value = tensors.get(prefix + "_recent30_" + suffix)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                return False
            if suffix == "values":
                if ((value < 0) | (value > 1)).any():
                    return False
            elif value.is_floating_point() or ((value < 0) | (value >= self._PROBE_WINDOW_SIZE + (suffix == "count"))).any():
                return False
        return True

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return learned reset-sampler state keyed by canonical motion path."""

        motion_keys = self._checkpoint_motion_keys()
        tensors: dict[str, torch.Tensor] = {}
        for field in self._PER_MOTION_CHECKPOINT_FIELDS:
            value = getattr(self, field, None)
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == len(motion_keys):
                tensors[field] = value.detach().cpu().clone()

        adaptive: dict[str, torch.Tensor] = {}
        sampler = getattr(self, "adaptive_timesteps_sampler", None)
        if sampler is not None:
            for field in ("bin_failed_count", "current_bin_failed_count"):
                value = getattr(sampler, field, None)
                if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == len(motion_keys):
                    adaptive[field] = value.detach().cpu().clone()

        group_state = {}
        if getattr(self, "_use_group_probe_envs", False):
            fields = ["_group_probe_recent30_" + name for name in self._RECENT_PROBE_FIELDS]
            fields += ["_normal_group_sampling_weights", "_group_probe_count", "_group_probe_fail_bin_hist"]
            group_state = {"keys": self._checkpoint_group_keys(), "tensors": {
                name: getattr(self, name).detach().cpu().clone() for name in fields
            }}
        return {
            "schema": self._CHECKPOINT_SCHEMA,
            "probe_statistics": {"kind": "recent30", "window": self._PROBE_WINDOW_SIZE},
            "group_probe": group_state,
            "motion_keys": motion_keys,
            "tensors": tensors,
            "adaptive_timesteps": adaptive,
            "restore_provenance": {
                "restored": bool(getattr(self, "_sampler_checkpoint_restored", False)),
                "matched_motion_count": int(
                    getattr(self, "_sampler_checkpoint_matched_motion_count", 0)
                ),
                "saved_motion_count": int(
                    getattr(self, "_sampler_checkpoint_saved_motion_count", 0)
                ),
                "reason": str(getattr(self, "_sampler_checkpoint_restore_reason", "fresh_run")),
            },
        }

    @staticmethod
    def _restore_rows_by_motion_key(
        current: torch.Tensor,
        saved: Any,
        current_indices: torch.Tensor,
        saved_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        saved_tensor = torch.as_tensor(saved, device=current.device, dtype=current.dtype)
        if saved_tensor.ndim != current.ndim or saved_tensor.shape[1:] != current.shape[1:]:
            return None
        if saved_indices.numel() > 0 and int(saved_indices.max().item()) >= saved_tensor.shape[0]:
            return None
        restored = current.clone()
        restored[current_indices] = saved_tensor[saved_indices]
        return restored

    def _set_checkpoint_restore_status(self, *, restored: bool, matched: int, saved: int, reason: str) -> None:
        self._sampler_checkpoint_restored = bool(restored)
        self._sampler_checkpoint_matched_motion_count = int(matched)
        self._sampler_checkpoint_saved_motion_count = int(saved)
        self._sampler_checkpoint_restore_reason = str(reason)

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore learned sampler state, remapping rows if the manifest was reordered."""

        if not isinstance(state, dict):
            self._set_checkpoint_restore_status(restored=False, matched=0, saved=0, reason="legacy_checkpoint")
            logger.warning("WBT sampler state was not restored: legacy checkpoint has no sampler state.")
            return
        if state.get("schema") not in (self._CHECKPOINT_SCHEMA, "holosoma_wbt_motion_sampler_v1"):
            self._set_checkpoint_restore_status(restored=False, matched=0, saved=0, reason="unsupported_schema")
            logger.warning(
                "WBT sampler state was not restored: unsupported schema "
                f"{state.get('schema')!r}."
            )
            return

        from somaforge_core.published_data import relocate_published_values

        paths = relocate_published_values(state.get("motion_keys", []))
        saved_keys = [str(Path(path).expanduser().resolve()) for path in paths]
        current_keys = self._checkpoint_motion_keys()
        saved_index_by_key = {key: index for index, key in enumerate(saved_keys)}
        matched_pairs = [
            (current_index, saved_index_by_key[key])
            for current_index, key in enumerate(current_keys)
            if key in saved_index_by_key
        ]
        if not matched_pairs:
            self._set_checkpoint_restore_status(
                restored=False,
                matched=0,
                saved=len(saved_keys),
                reason="no_matching_motions",
            )
            logger.warning("WBT sampler state was not restored: no checkpoint motions match the current manifest.")
            return

        current_indices = torch.tensor([pair[0] for pair in matched_pairs], device=self.device, dtype=torch.long)
        saved_indices = torch.tensor([pair[1] for pair in matched_pairs], device=self.device, dtype=torch.long)
        restored_fields = 0
        tensors = state.get("tensors", {})
        current_statistics = state.get("probe_statistics") == {"kind": "recent30", "window": self._PROBE_WINDOW_SIZE}
        recent_ok = current_statistics and isinstance(tensors, dict) and self._valid_recent_probe_checkpoint(
            tensors, "_probe", len(saved_keys)
        )
        self._init_recent_probe_window()
        self._refresh_recent_probe_means()
        if not recent_ok:
            logger.info("Recent-30 probe window starts empty; historical EMA is not converted into observations.")
            if not getattr(self, "_use_completion_learning_sampler", False):
                base = getattr(self, "_base_motion_sampling_weights", None)
                if base is None:
                    base = torch.ones_like(self._normal_motion_sampling_weights)
                self._normal_motion_sampling_weights.copy_(base / base.sum().clamp(min=1e-8))
        if isinstance(tensors, dict):
            for field in self._PER_MOTION_CHECKPOINT_FIELDS:
                if field.startswith("_probe_recent30_") and not recent_ok:
                    continue
                if field == "_normal_motion_sampling_weights" and not recent_ok and not getattr(
                    self, "_use_completion_learning_sampler", False
                ):
                    continue
                current = getattr(self, field, None)
                if not torch.is_tensor(current) or field not in tensors:
                    continue
                restored = self._restore_rows_by_motion_key(
                    current,
                    tensors[field],
                    current_indices,
                    saved_indices,
                )
                if restored is None:
                    logger.warning(f"Skipping incompatible WBT sampler checkpoint field {field}.")
                    continue
                current.copy_(restored)
                restored_fields += 1

        self._refresh_recent_probe_means()
        if getattr(self, "_use_group_probe_envs", False):
            self._init_recent_probe_window(group=True)
            saved_group = state.get("group_probe", {})
            group_keys = saved_group.get("keys", [])
            # Group identity is a JSON-encoded set of canonical motion paths.
            # Relocate each member before matching; never alter saved tensors.
            relocated_group_keys = []
            for key in group_keys:
                try:
                    members = json.loads(key)
                except (TypeError, ValueError):
                    relocated_group_keys.append(key)
                    continue
                if isinstance(members, list) and all(isinstance(member, str) for member in members):
                    members = relocate_published_values(members)
                    key = json.dumps(sorted(str(Path(member).expanduser().resolve()) for member in members))
                relocated_group_keys.append(key)
            group_keys = relocated_group_keys
            group_tensors = saved_group.get("tensors", {})
            if current_statistics and self._valid_recent_probe_checkpoint(group_tensors, "_group_probe", len(group_keys)):
                lookup = {key: index for index, key in enumerate(group_keys)}
                pairs = [(i, lookup[key]) for i, key in enumerate(self._checkpoint_group_keys()) if key in lookup]
                if pairs:
                    ci = torch.tensor([i for i, _ in pairs], device=self.device)
                    si = torch.tensor([i for _, i in pairs], device=self.device)
                    for field, saved in group_tensors.items():
                        if field not in {"_normal_group_sampling_weights", "_group_probe_count", "_group_probe_fail_bin_hist",
                                         *("_group_probe_recent30_" + name for name in self._RECENT_PROBE_FIELDS)}:
                            continue
                        restored = self._restore_rows_by_motion_key(getattr(self, field), saved, ci, si)
                        if restored is not None:
                            getattr(self, field).copy_(restored)
                            restored_fields += 1
            self._refresh_recent_probe_means(group=True)
            self._normal_group_sampling_weights.div_(self._normal_group_sampling_weights.sum().clamp(min=1e-8))
            self._refresh_normal_motion_weights_from_group_weights()

        weights = getattr(self, "_normal_motion_sampling_weights", None)
        if torch.is_tensor(weights):
            weights.clamp_(min=0.0)
            weights.div_(weights.sum().clamp(min=1e-8))

        adaptive = state.get("adaptive_timesteps", {})
        sampler = getattr(self, "adaptive_timesteps_sampler", None)
        if sampler is not None and isinstance(adaptive, dict):
            for field in ("bin_failed_count", "current_bin_failed_count"):
                current = getattr(sampler, field, None)
                if not torch.is_tensor(current) or field not in adaptive:
                    continue
                restored = self._restore_rows_by_motion_key(
                    current,
                    adaptive[field],
                    current_indices,
                    saved_indices,
                )
                if restored is None:
                    logger.warning(f"Skipping incompatible WBT adaptive sampler checkpoint field {field}.")
                    continue
                current.copy_(restored)
                restored_fields += 1

        restored = restored_fields > 0
        reason = "restored" if restored else "no_compatible_fields"
        self._set_checkpoint_restore_status(
            restored=restored,
            matched=len(matched_pairs),
            saved=len(saved_keys),
            reason=reason,
        )
        log = logger.info if restored else logger.warning
        log(
            "WBT sampler checkpoint restore: "
            f"restored={restored}, matched_motions={len(matched_pairs)}/{len(current_keys)}, "
            f"saved_motions={len(saved_keys)}, restored_fields={restored_fields}, reason={reason}."
        )

    @staticmethod
    def _append_checkpoint_restore_metrics(owner: Any, metrics: dict[str, float]) -> dict[str, float]:
        if not hasattr(owner, "_sampler_checkpoint_restored"):
            return metrics
        metrics.update(
            {
                "sampler_checkpoint_restored": float(owner._sampler_checkpoint_restored),
                "sampler_checkpoint_matched_motion_count": float(
                    owner._sampler_checkpoint_matched_motion_count
                ),
                "sampler_checkpoint_saved_motion_count": float(owner._sampler_checkpoint_saved_motion_count),
            }
        )
        return metrics

    @staticmethod
    def _recent_probe_count_metrics(owner: Any, *, group: bool = False) -> dict[str, float]:
        prefix = "_group_probe" if group else "_probe"
        counts = getattr(owner, prefix + "_recent30_count")
        tag = "group_probe_" if group else ""
        return {
            tag + "recent30_window_size": float(MotionCommand._PROBE_WINDOW_SIZE),
            tag + "recent30_sample_count_min": float(counts.min().item()),
            tag + "recent30_sample_count_mean": float(counts.float().mean().item()),
            tag + "recent30_reported_fraction": float((counts > 0).float().mean().item()),
            tag + "recent30_full_window_fraction": float((counts == MotionCommand._PROBE_WINDOW_SIZE).float().mean().item()),
        }

    def get_motion_learning_progress_metrics(self) -> dict[str, float]:
        """Return probe learning progress metrics for TensorBoard."""
        if getattr(self, "_use_completion_learning_sampler", False):
            learned = self._completion_learned_mask.detach()
            weights = self._normal_motion_sampling_weights.detach()
            progress = self._completion_progress_ema.detach()
            success = self._completion_success_ema.detach()
            fail = self._completion_fail_ema.detach()
            max_weight, max_weight_motion = weights.max(dim=0)
            min_success, min_success_motion = success.min(dim=0)
            return MotionCommand._append_checkpoint_restore_metrics(self, {
                "completion_learned_frac": float(learned.to(torch.float32).mean().item()),
                "completion_progress_ema_mean": float(progress.mean().item()),
                "completion_success_ema_mean": float(success.mean().item()),
                "completion_fail_ema_mean": float(fail.mean().item()),
                "completion_success_ema_min": float(min_success.item()),
                "completion_success_ema_min_motion": float(min_success_motion.item()),
                "completion_normal_motion_weight_max": float(max_weight.item()),
                "completion_normal_motion_weight_max_motion": float(max_weight_motion.item()),
            })

        if getattr(self, "_use_group_probe_envs", False):
            completion = self._group_probe_completion_recent30.detach()
            success = self._group_probe_success_recent30.detach()
            fail = self._group_probe_fail_recent30.detach()
            timeout = self._group_probe_timeout_recent30.detach()
            weights = self._normal_group_sampling_weights.detach()
            min_completion, min_completion_group = completion.min(dim=0)
            min_success, min_success_group = success.min(dim=0)
            max_weight, max_weight_group = weights.max(dim=0)

            return MotionCommand._append_checkpoint_restore_metrics(self, {
                **MotionCommand._recent_probe_count_metrics(self, group=True),
                "group_probe_completion_recent30_mean": float(completion.mean().item()),
                "group_probe_completion_recent30_min": float(min_completion.item()),
                "group_probe_completion_recent30_min_group": float(min_completion_group.item()),
                "group_probe_start_to_end_success_recent30_mean": float(success.mean().item()),
                "group_probe_start_to_end_success_recent30_min": float(min_success.item()),
                "group_probe_start_to_end_success_recent30_min_group": float(min_success_group.item()),
                "group_probe_fail_recent30_mean": float(fail.mean().item()),
                "group_probe_timeout_recent30_mean": float(timeout.mean().item()),
                "group_probe_normal_group_weight_max": float(max_weight.item()),
                "group_probe_normal_group_weight_max_group": float(max_weight_group.item()),
            })

        if not getattr(self, "_use_start_probe_envs", False):
            return MotionCommand._append_checkpoint_restore_metrics(self, {})

        completion = self._probe_completion_recent30.detach()
        success = self._probe_success_recent30.detach()
        fail = self._probe_fail_recent30.detach()
        timeout = self._probe_timeout_recent30.detach()
        weights = self._normal_motion_sampling_weights.detach()
        min_completion, min_completion_motion = completion.min(dim=0)
        min_success, min_success_motion = success.min(dim=0)
        max_weight, max_weight_motion = weights.max(dim=0)

        return MotionCommand._append_checkpoint_restore_metrics(self, {
            **MotionCommand._recent_probe_count_metrics(self),
            "start_probe_episode_count": float(self._probe_count.sum().item()),
            "completion_recent30_mean": float(completion.mean().item()),
            "completion_recent30_min": float(min_completion.item()),
            "completion_recent30_min_motion": float(min_completion_motion.item()),
            "start_to_end_success_recent30_mean": float(success.mean().item()),
            "start_to_end_success_recent30_min": float(min_success.item()),
            "start_to_end_success_recent30_min_motion": float(min_success_motion.item()),
            "fail_recent30_mean": float(fail.mean().item()),
            "timeout_recent30_mean": float(timeout.mean().item()),
            "normal_motion_weight_max": float(max_weight.item()),
            "normal_motion_weight_max_motion": float(max_weight_motion.item()),
        })

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/root_error_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/root_error_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/root_error_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/root_error_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        body_pos_error = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
        body_rot_error = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w)
        body_lin_vel_error = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1)
        body_ang_vel_error = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1)

        self.metrics["motion/full_body_error_pos"] = body_pos_error.mean(dim=-1)
        self.metrics["motion/full_body_error_rot"] = body_rot_error.mean(dim=-1)
        self.metrics["motion/full_body_error_lin_vel"] = body_lin_vel_error.mean(dim=-1)
        self.metrics["motion/full_body_error_ang_vel"] = body_ang_vel_error.mean(dim=-1)
        self.metrics["motion/full_body_error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/full_body_error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion._has_part_annotations:
            limb_idx = self.a2a_limb_body_indexes_in_track
            active_mask = self.active_part_mask[:, : len(A2A_LIMB_REF_BODY_NAMES)].to(body_pos_error.dtype)
            support_mask = self.support_part_mask[:, : len(A2A_LIMB_REF_BODY_NAMES)].to(body_pos_error.dtype)
            active_count = active_mask.sum(dim=1)
            support_count = support_mask.sum(dim=1)
            active_den = active_count.clamp(min=1.0)

            limb_pos_error = body_pos_error[:, limb_idx]
            limb_rot_error = body_rot_error[:, limb_idx]

            self.metrics["motion/num_active_parts"] = active_count
            self.metrics["motion/num_contact_keep_intents"] = support_count
            self.metrics["motion/active_limb_error_pos"] = (limb_pos_error * active_mask).sum(dim=1) / active_den
            self.metrics["motion/active_limb_error_rot"] = (limb_rot_error * active_mask).sum(dim=1) / active_den

            support_slip = torch.norm(self.robot_body_lin_vel_w[:, limb_idx, :2], dim=-1)
            support_contact = self.contact_part_mask[:, : len(A2A_LIMB_REF_BODY_NAMES)].to(support_slip.dtype)
            support_slip = support_slip * support_contact * support_mask
            support_den = support_count.clamp(min=1.0)
            self.metrics["motion/keep_intent_body_origin_speed_proxy_mean"] = support_slip.sum(dim=1) / support_den
            self.metrics["motion/keep_intent_body_origin_speed_proxy_max"] = support_slip.max(dim=1)[0]

        if hasattr(self, "adaptive_timesteps_sampler"):
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]
        if getattr(self, "_event_token_enabled", False):
            self.metrics["motion/event_token_index_mean"] = self._event_token_index.to(torch.float32).mean()
            self.metrics["motion/event_token_index_max"] = self._event_token_index.to(torch.float32).max()
            self.metrics["motion/event_token_switch_count"] = self._event_token_switch_count
            self.metrics["motion/event_token_contact_switch_count"] = self._event_token_contact_switch_count
            self.metrics["motion/event_token_timeout_count"] = self._event_token_timeout_count
            self.metrics["motion/event_token_last_switch_frame"] = self._event_token_last_switch_frame
            self.metrics["motion/event_token_last_from"] = self._event_token_last_from
            self.metrics["motion/event_token_last_to"] = self._event_token_last_to
            self.metrics["motion/event_token_last_reason"] = self._event_token_last_reason
        if self._uses_failure_window_sampler:
            count = self._failure_window_reset_count.clamp(min=1.0)
            self.metrics["motion/failure_window_last_reset_count"] = self._failure_window_reset_count
            self.metrics["motion/failure_window_last_before_ratio"] = self._failure_window_before_count / count
            self.metrics["motion/failure_window_last_mean_offset_frames"] = self._failure_window_offset_sum / count
            hist_total = self._failure_window_failure_hist.sum().clamp(min=1.0)
            hist_prob = self._failure_window_failure_hist / hist_total
            flat_top = torch.argmax(hist_prob)
            top_motion = torch.div(flat_top, self._failure_window_log_bin_count, rounding_mode="floor")
            top_bin = flat_top - top_motion * self._failure_window_log_bin_count
            top_prob = hist_prob.flatten()[flat_top]
            top_frame = top_bin.to(torch.float32) * float(self._failure_window_log_bin_frames)
            hist_entropy = -(hist_prob * (hist_prob + 1e-12).log()).sum()
            hist_entropy = hist_entropy / torch.log(
                torch.tensor(float(hist_prob.numel()), dtype=torch.float32, device=self.device)
            ).clamp(min=1e-8)
            if self.motion.num_motions > 1:
                self.metrics["motion/failure_window_top_failure_motion"] = top_motion.to(torch.float32)
            self.metrics["motion/failure_window_top_failure_frame"] = top_frame
            self.metrics["motion/failure_window_top_failure_prob"] = top_prob
            self.metrics["motion/failure_window_failure_hist_entropy"] = hist_entropy
        for key, value in list(self.metrics.items()):
            if torch.is_tensor(value):
                self.metrics[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def write_adaptive_sampling_distribution(
        self, log_dir: str | os.PathLike[str], iteration: int, interval: int = 100
    ):
        """Append the full adaptive reset sampling distribution to a per-run CSV."""
        if not hasattr(self, "adaptive_timesteps_sampler"):
            return
        if interval <= 0 or iteration % interval != 0:
            return

        sampler = self.adaptive_timesteps_sampler
        probabilities = sampler.sampling_probabilities.detach().to("cpu")
        failed_count = sampler.bin_failed_count.detach().to("cpu")
        current_failed_count = sampler.current_bin_failed_count.detach().to("cpu")
        env_fps = float(max(sampler.env_fps, 1))
        motion_weights = (
            getattr(
                self.motion,
                "motion_sampling_weights",
                torch.full((self.motion.num_motions,), 1.0 / max(self.motion.num_motions, 1), device=self.device),
            )
            .detach()
            .to("cpu")
        )
        if getattr(self, "_use_start_probe_envs", False) or getattr(self, "_use_group_probe_envs", False):
            motion_weights = self._normal_motion_sampling_weights.detach().to("cpu")
        motion_files = getattr(self.motion, "motion_files", [])
        motion_start_idx = self.motion.motion_start_idx.detach().to("cpu")
        motion_end_idx = self.motion.motion_end_idx.detach().to("cpu")
        motion_lengths = (motion_end_idx - motion_start_idx).clamp(min=1)
        num_bins_per_motion = sampler.num_bins_per_motion.detach().to("cpu")
        bin_start_local = sampler.bin_start_local.detach().to("cpu")
        bin_end_local = sampler.bin_end_local.detach().to("cpu")

        path = Path(log_dir) / "adaptive_reset_sampling_distribution.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "motion_id",
                    "motion_file",
                    "motion_weight",
                    "num_bins",
                    "bin_id",
                    "bin_start_frame",
                    "bin_end_frame",
                    "bin_center_frame",
                    "global_bin_start_frame",
                    "global_bin_end_frame",
                    "global_bin_center_frame",
                    "bin_start_time_s",
                    "bin_end_time_s",
                    "bin_center_time_s",
                    "sampling_probability",
                    "bin_failed_count",
                    "current_bin_failed_count",
                ],
            )
            if write_header:
                writer.writeheader()
            for motion_id in range(int(self.motion.num_motions)):
                num_bins = int(num_bins_per_motion[motion_id].item())
                motion_start = int(motion_start_idx[motion_id].item())
                for bin_id in range(num_bins):
                    local_start = int(bin_start_local[motion_id, bin_id].item())
                    local_end = int(bin_end_local[motion_id, bin_id].item())
                    local_center = 0.5 * (local_start + local_end)
                    global_start = motion_start + local_start
                    global_end = motion_start + local_end
                    global_center = motion_start + local_center
                    writer.writerow(
                        {
                            "iteration": int(iteration),
                            "motion_id": motion_id,
                            "motion_file": motion_files[motion_id] if motion_id < len(motion_files) else "",
                            "motion_weight": f"{float(motion_weights[motion_id]):.8g}",
                            "num_bins": num_bins,
                            "bin_id": bin_id,
                            "bin_start_frame": local_start,
                            "bin_end_frame": local_end,
                            "bin_center_frame": f"{local_center:.3f}",
                            "global_bin_start_frame": global_start,
                            "global_bin_end_frame": global_end,
                            "global_bin_center_frame": f"{global_center:.3f}",
                            "bin_start_time_s": f"{local_start / env_fps:.6g}",
                            "bin_end_time_s": f"{local_end / env_fps:.6g}",
                            "bin_center_time_s": f"{local_center / env_fps:.6g}",
                            "sampling_probability": f"{float(probabilities[motion_id, bin_id]):.8g}",
                            "bin_failed_count": f"{float(failed_count[motion_id, bin_id]):.8g}",
                            "current_bin_failed_count": f"{float(current_failed_count[motion_id, bin_id]):.8g}",
                        }
                    )

    def write_start_probe_distribution(
        self, log_dir: str | os.PathLike[str], iteration: int, interval: int = 100
    ) -> None:
        """Append per-motion start-probe stats and normal-env motion weights to a CSV."""
        if not getattr(self, "_use_start_probe_envs", False):
            return
        if interval <= 0 or iteration % interval != 0:
            return

        motion_files = getattr(self.motion, "motion_files", [])
        weights = self._normal_motion_sampling_weights.detach().to("cpu")
        completion = self._probe_completion_recent30.detach().to("cpu")
        success = self._probe_success_recent30.detach().to("cpu")
        fail = self._probe_fail_recent30.detach().to("cpu")
        timeout = self._probe_timeout_recent30.detach().to("cpu")
        counts = self._probe_count.detach().to("cpu")
        recent_counts = self._probe_recent30_count.detach().to("cpu")
        fail_bins = self._probe_fail_bin_hist.detach().to("cpu")

        path = Path(log_dir) / "start_probe_recent30_distribution.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        bin_fields = [f"fail_bin_{i}" for i in range(self._probe_fail_bin_count)]
        with path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "iteration",
                "motion_id",
                "motion_file",
                "probe_env_per_motion",
                "probe_count",
                "recent30_count",
                "completion_recent30",
                "start_to_end_success_recent30",
                "fail_recent30",
                "timeout_recent30",
                "normal_motion_weight",
                *bin_fields,
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for motion_id in range(int(self.motion.num_motions)):
                row = {
                    "iteration": int(iteration),
                    "motion_id": motion_id,
                    "motion_file": motion_files[motion_id] if motion_id < len(motion_files) else "",
                    "probe_env_per_motion": int(self._probe_env_per_motion),
                    "probe_count": int(counts[motion_id].item()),
                    "recent30_count": int(recent_counts[motion_id].item()),
                    "completion_recent30": f"{float(completion[motion_id]):.8g}",
                    "start_to_end_success_recent30": f"{float(success[motion_id]):.8g}",
                    "fail_recent30": f"{float(fail[motion_id]):.8g}",
                    "timeout_recent30": f"{float(timeout[motion_id]):.8g}",
                    "normal_motion_weight": f"{float(weights[motion_id]):.8g}",
                }
                row.update(
                    {
                        f"fail_bin_{bin_id}": f"{float(fail_bins[motion_id, bin_id]):.8g}"
                        for bin_id in range(self._probe_fail_bin_count)
                    }
                )
                writer.writerow(row)

    def write_completion_learning_distribution(
        self, log_dir: str | os.PathLike[str], iteration: int, interval: int = 100
    ) -> None:
        """Append per-motion completion-learning stats and normal-env motion weights to a CSV."""
        if not getattr(self, "_use_completion_learning_sampler", False):
            return
        if interval <= 0 or iteration % interval != 0:
            return

        motion_files = getattr(self.motion, "motion_files", [])
        weights = self._normal_motion_sampling_weights.detach().to("cpu")
        learned = self._completion_learned_mask.detach().to("cpu")
        success_streak = self._completion_success_streak.detach().to("cpu")
        fail_streak = self._completion_fail_streak.detach().to("cpu")
        success_count = self._completion_success_count.detach().to("cpu")
        fail_count = self._completion_fail_count.detach().to("cpu")
        episode_count = self._completion_episode_count.detach().to("cpu")
        progress = self._completion_progress_ema.detach().to("cpu")
        success = self._completion_success_ema.detach().to("cpu")
        fail = self._completion_fail_ema.detach().to("cpu")

        path = Path(log_dir) / "completion_learning_motion_distribution.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "motion_id",
                    "motion_file",
                    "learned",
                    "success_streak",
                    "fail_streak",
                    "episode_count",
                    "success_count",
                    "fail_count",
                    "progress_ema",
                    "success_ema",
                    "fail_ema",
                    "normal_motion_weight",
                ],
            )
            if write_header:
                writer.writeheader()
            for motion_id in range(int(self.motion.num_motions)):
                writer.writerow(
                    {
                        "iteration": int(iteration),
                        "motion_id": motion_id,
                        "motion_file": motion_files[motion_id] if motion_id < len(motion_files) else "",
                        "learned": int(bool(learned[motion_id].item())),
                        "success_streak": int(success_streak[motion_id].item()),
                        "fail_streak": int(fail_streak[motion_id].item()),
                        "episode_count": int(episode_count[motion_id].item()),
                        "success_count": int(success_count[motion_id].item()),
                        "fail_count": int(fail_count[motion_id].item()),
                        "progress_ema": f"{float(progress[motion_id]):.8g}",
                        "success_ema": f"{float(success[motion_id]):.8g}",
                        "fail_ema": f"{float(fail[motion_id]):.8g}",
                        "normal_motion_weight": f"{float(weights[motion_id]):.8g}",
                    }
                )

    def write_group_probe_distribution(
        self, log_dir: str | os.PathLike[str], iteration: int, interval: int = 100
    ) -> None:
        """Append per-group probe stats and normal-env group weights to a CSV."""
        if not getattr(self, "_use_group_probe_envs", False):
            return
        if interval <= 0 or iteration % interval != 0:
            return

        group_labels = self._group_labels.detach().to("cpu")
        group_motion_mask = self._group_motion_mask.detach().to("cpu")
        group_motion_counts = self._group_motion_counts.detach().to("cpu")
        group_variant_effective_counts = self._group_variant_effective_counts.detach().to("cpu")
        weights = self._normal_group_sampling_weights.detach().to("cpu")
        completion = self._group_probe_completion_recent30.detach().to("cpu")
        success = self._group_probe_success_recent30.detach().to("cpu")
        fail = self._group_probe_fail_recent30.detach().to("cpu")
        timeout = self._group_probe_timeout_recent30.detach().to("cpu")
        counts = self._group_probe_count.detach().to("cpu")
        recent_counts = self._group_probe_recent30_count.detach().to("cpu")
        fail_bins = self._group_probe_fail_bin_hist.detach().to("cpu")

        path = Path(log_dir) / "group_probe_recent30_distribution.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        bin_fields = [f"fail_bin_{i}" for i in range(self._probe_fail_bin_count)]
        with path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "iteration",
                "group_id",
                "group_label",
                "group_by",
                "probe_env_per_group",
                "group_variant_sample_count",
                "group_variant_effective_count",
                "group_motion_count",
                "group_motion_ids",
                "probe_count",
                "recent30_count",
                "completion_recent30",
                "start_to_end_success_recent30",
                "fail_recent30",
                "timeout_recent30",
                "normal_group_weight",
                *bin_fields,
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for group_id in range(int(self._num_motion_groups)):
                motion_ids = torch.where(group_motion_mask[group_id])[0].tolist()
                row = {
                    "iteration": int(iteration),
                    "group_id": group_id,
                    "group_label": int(group_labels[group_id].item()),
                    "group_by": self._group_probe_by,
                    "probe_env_per_group": int(self._group_probe_env_per_group),
                    "group_variant_sample_count": int(self._group_variant_sample_count),
                    "group_variant_effective_count": int(group_variant_effective_counts[group_id].item()),
                    "group_motion_count": int(group_motion_counts[group_id].item()),
                    "group_motion_ids": "|".join(str(int(motion_id)) for motion_id in motion_ids),
                    "probe_count": int(counts[group_id].item()),
                    "recent30_count": int(recent_counts[group_id].item()),
                    "completion_recent30": f"{float(completion[group_id]):.8g}",
                    "start_to_end_success_recent30": f"{float(success[group_id]):.8g}",
                    "fail_recent30": f"{float(fail[group_id]):.8g}",
                    "timeout_recent30": f"{float(timeout[group_id]):.8g}",
                    "normal_group_weight": f"{float(weights[group_id]):.8g}",
                }
                row.update(
                    {
                        f"fail_bin_{bin_id}": f"{float(fail_bins[group_id, bin_id]):.8g}"
                        for bin_id in range(self._probe_fail_bin_count)
                    }
                )
                writer.writerow(row)

    def write_failure_window_distribution(
        self, log_dir: str | os.PathLike[str], iteration: int, interval: int = 100
    ) -> None:
        """Append failure-window failure-frame histogram stats to a per-run CSV."""
        if not self._uses_failure_window_sampler:
            return
        if interval <= 0 or iteration % interval != 0:
            return

        hist = self._failure_window_failure_hist.detach().to("cpu")
        total = hist.sum().clamp(min=1.0)
        probabilities = hist / total
        flat_top = int(torch.argmax(probabilities).item())
        top_motion = flat_top // int(self._failure_window_log_bin_count)
        top_bin = flat_top - top_motion * int(self._failure_window_log_bin_count)
        motion_files = getattr(self.motion, "motion_files", [])
        fps = float(max(self.motion.fps, 1))

        path = Path(log_dir) / "failure_window_reset_distribution.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "iteration",
                "motion_id",
                "motion_file",
                "bin_id",
                "bin_start_frame",
                "bin_end_frame",
                "bin_center_frame",
                "bin_start_time_s",
                "bin_end_time_s",
                "bin_center_time_s",
                "failure_count",
                "failure_probability",
                "is_top_failure_bin",
                "top_failure_motion",
                "top_failure_bin",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            bin_width = int(self._failure_window_log_bin_frames)
            for motion_id in range(int(self.motion.num_motions)):
                for bin_id in range(int(self._failure_window_log_bin_count)):
                    start = bin_id * bin_width
                    end = start + bin_width
                    center = 0.5 * (start + end)
                    writer.writerow(
                        {
                            "iteration": int(iteration),
                            "motion_id": motion_id,
                            "motion_file": motion_files[motion_id] if motion_id < len(motion_files) else "",
                            "bin_id": bin_id,
                            "bin_start_frame": start,
                            "bin_end_frame": end,
                            "bin_center_frame": f"{center:.3f}",
                            "bin_start_time_s": f"{start / fps:.6g}",
                            "bin_end_time_s": f"{end / fps:.6g}",
                            "bin_center_time_s": f"{center / fps:.6g}",
                            "failure_count": f"{float(hist[motion_id, bin_id]):.8g}",
                            "failure_probability": f"{float(probabilities[motion_id, bin_id]):.8g}",
                            "is_top_failure_bin": int(motion_id == top_motion and bin_id == top_bin),
                            "top_failure_motion": top_motion,
                            "top_failure_bin": top_bin,
                        }
                    )

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    @property
    def _reference_env_origins(self) -> torch.Tensor:
        terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
        terrain = getattr(terrain_state, "terrain", None)
        if getattr(terrain, "has_motion_matched_origins", False):
            return terrain_state.env_origins
        return self._env.simulator.scene.env_origins

    def _use_local_segment_reference(self) -> bool:
        return bool(self.motion_cfg.local_motion_segment_reference) and bool(self._env.is_evaluating)

    def _reset_local_segment_anchors(self, env_ids: torch.Tensor) -> None:
        if not self._use_local_segment_reference() or env_ids.numel() == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        start_steps = self.motion.motion_start_idx[motion_ids]
        self._local_segment_anchor_motion_ids[env_ids] = motion_ids
        self._local_segment_motion_ref_pos_w[env_ids] = self.motion.body_pos_w[start_steps, self.ref_body_index]
        self._local_segment_motion_ref_quat_w[env_ids] = self.motion.body_quat_w[start_steps, self.ref_body_index]
        self._local_segment_robot_ref_pos_w[env_ids] = self.robot_ref_pos_w[env_ids].detach()
        self._local_segment_robot_ref_quat_w[env_ids] = self.robot_ref_quat_w[env_ids].detach()

    def _ensure_local_segment_anchors(self) -> None:
        if not self._use_local_segment_reference():
            return
        missing = self._local_segment_anchor_motion_ids != self.motion_ids
        if torch.any(missing):
            self._reset_local_segment_anchors(torch.where(missing)[0])

    def _local_segment_delta_quat(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return yaw_quat(
            quat_mul(
                self._local_segment_robot_ref_quat_w[env_ids],
                quat_inverse(self._local_segment_motion_ref_quat_w[env_ids], w_last=True),
                w_last=True,
            ),
            w_last=True,
        )

    def _localize_motion_pos_quat(
        self,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(pos_w.shape[0], device=self.device)
        if not self._use_local_segment_reference():
            return pos_w + self._reference_env_origins[env_ids, None, :], quat_w
        self._ensure_local_segment_anchors()
        delta_quat = self._local_segment_delta_quat(env_ids)
        pos_delta = pos_w - self._local_segment_motion_ref_pos_w[env_ids, None, :]
        delta_quat = delta_quat[:, None, :].expand_as(quat_w)
        localized_pos = self._local_segment_robot_ref_pos_w[env_ids, None, :] + quat_apply(
            delta_quat, pos_delta, w_last=True
        )
        localized_quat = quat_mul(delta_quat, quat_w, w_last=True)
        return localized_pos, localized_quat

    def _localize_motion_velocity(
        self,
        vel_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(vel_w.shape[0], device=self.device)
        if not self._use_local_segment_reference():
            return vel_w
        self._ensure_local_segment_anchors()
        delta_quat = self._local_segment_delta_quat(env_ids)
        delta_quat = delta_quat[:, None, :].expand(*vel_w.shape[:-1], 4)
        return quat_apply(delta_quat, vel_w, w_last=True)

    def _localize_motion_pos_quat_future(
        self,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_local_segment_reference():
            return pos_w + self._reference_env_origins[:, None, None, :], quat_w
        self._ensure_local_segment_anchors()
        env_ids = torch.arange(self.num_envs, device=self.device)
        delta_quat = self._local_segment_delta_quat(env_ids)
        pos_delta = pos_w - self._local_segment_motion_ref_pos_w[:, None, None, :]
        delta_quat = delta_quat[:, None, None, :].expand_as(quat_w)
        localized_pos = self._local_segment_robot_ref_pos_w[:, None, None, :] + quat_apply(
            delta_quat, pos_delta, w_last=True
        )
        localized_quat = quat_mul(delta_quat, quat_w, w_last=True)
        return localized_pos, localized_quat

    def future_body_pos_quat_w(self, future_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos_w = self.motion.body_pos_w[future_steps][:, :, self.tracked_body_indexes]
        body_quat_w = self.motion.body_quat_w[future_steps][:, :, self.tracked_body_indexes]
        return self._localize_motion_pos_quat_future(body_pos_w, body_quat_w)

    def future_joint_pos_vel(self, offsets: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._future_joint_cache.get(offsets)
        if cached is not None:
            return cached
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
        future_steps = self.time_steps[:, None] + offsets_t[None, :]
        start_idx = self.motion.motion_start_idx[self.motion_ids][:, None]
        end_idx = self.motion.motion_end_idx[self.motion_ids][:, None]
        future_steps = future_steps.clamp(min=start_idx, max=end_idx - 1)
        result = self.motion.joint_pos[future_steps], self.motion.joint_vel[future_steps]
        self._future_joint_cache[offsets] = result
        return result

    def future_body_pos_quat_offsets(
        self, offsets: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
        future_steps = self.time_steps[:, None] + offsets_t[None, :]
        start_idx = self.motion.motion_start_idx[self.motion_ids][:, None]
        end_idx = self.motion.motion_end_idx[self.motion_ids][:, None]
        future_steps = future_steps.clamp(min=start_idx, max=end_idx - 1)
        return self.future_body_pos_quat_w(future_steps)

    def future_ref_body_pos_quat_offsets(
        self, offsets: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return localized future poses for only the reference body, cached per command step."""
        cached = self._future_ref_body_cache.get(offsets)
        if cached is not None:
            return cached

        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
        future_steps = self.time_steps[:, None] + offsets_t[None, :]
        start_idx = self.motion.motion_start_idx[self.motion_ids][:, None]
        end_idx = self.motion.motion_end_idx[self.motion_ids][:, None]
        future_steps = future_steps.clamp(min=start_idx, max=end_idx - 1)
        body_pos_w = self.motion.body_pos_w[future_steps, self.ref_body_index][:, :, None, :]
        body_quat_w = self.motion.body_quat_w[future_steps, self.ref_body_index][:, :, None, :]
        localized_pos, localized_quat = self._localize_motion_pos_quat_future(body_pos_w, body_quat_w)
        result = localized_pos[:, :, 0], localized_quat[:, :, 0]
        self._future_ref_body_cache[offsets] = result
        return result

    def future_contact_force_part_mask_offsets(self, offsets: tuple[int, ...]) -> torch.Tensor:
        """Return the recorded contact schedule at future reference offsets."""
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
        future_steps = self.time_steps[:, None] + offsets_t[None, :]
        start_idx = self.motion.motion_start_idx[self.motion_ids][:, None]
        end_idx = self.motion.motion_end_idx[self.motion_ids][:, None]
        future_steps = future_steps.clamp(min=start_idx, max=end_idx - 1)
        return self.motion.contact_force_part_mask[future_steps]

    def _maybe_sample_motion_matched_origins(self, env_ids: torch.Tensor) -> None:
        terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
        terrain = getattr(terrain_state, "terrain", None)
        if not getattr(terrain, "has_motion_matched_origins", False):
            return
        motion_terrain_ids = self.motion.motion_terrain_ids[self.motion_ids[env_ids]]
        sample_origins = getattr(terrain_state, "sample_motion_matched_origins", None)
        if not callable(sample_origins):
            raise RuntimeError("Motion-matched terrain state must provide sample_motion_matched_origins().")
        sample_origins(env_ids, motion_terrain_ids)

    def _get_failure_window_reset_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._env.is_evaluating or not self._uses_failure_window_sampler:
            return torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        mask = _get_bad_tracking_done_mask(self._env.termination_manager.term_dones, env_ids)
        if mask is None:
            return torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        if getattr(self, "_use_start_probe_envs", False) or getattr(self, "_use_group_probe_envs", False):
            mask = mask & ~self._probe_env_mask[env_ids]
        return mask

    def _sample_failure_window_time_steps(
        self,
        env_ids: torch.Tensor,
        failed_time_steps: torch.Tensor,
    ) -> torch.Tensor:
        motion_ids = self.motion_ids[env_ids]
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        last_idx = torch.maximum(start_idx, end_idx - 2)
        failed_idx = torch.minimum(torch.maximum(failed_time_steps, start_idx), last_idx)

        pre_frames = max(int(self.motion_cfg.failure_window_pre_frames), 0)
        post_frames = max(int(self.motion_cfg.failure_window_post_frames), 0)
        before_prob = min(max(float(self.motion_cfg.failure_window_before_prob), 0.0), 1.0)

        use_before = torch.rand(env_ids.numel(), device=self.device) < before_prob
        before_start = torch.maximum(start_idx, failed_idx - pre_frames)
        before_end = failed_idx
        after_start = failed_idx
        after_end = torch.minimum(last_idx, failed_idx + post_frames)

        sample_start = torch.where(use_before, before_start, after_start)
        sample_end = torch.where(use_before, before_end, after_end)
        span = (sample_end - sample_start + 1).clamp(min=1)
        offsets = torch.floor(torch.rand(env_ids.numel(), device=self.device) * span.to(torch.float32)).long()
        sampled = sample_start + offsets

        self._failure_window_reset_count = torch.tensor(float(env_ids.numel()), dtype=torch.float32, device=self.device)
        self._failure_window_before_count = use_before.to(torch.float32).sum()
        self._failure_window_offset_sum = (sampled - failed_idx).to(torch.float32).sum()
        local_failed_idx = failed_idx - start_idx
        hist_bins = torch.div(local_failed_idx, self._failure_window_log_bin_frames, rounding_mode="floor")
        hist_bins = hist_bins.clamp(min=0, max=self._failure_window_log_bin_count - 1)
        flat_idx = motion_ids * self._failure_window_log_bin_count + hist_bins
        counts = torch.bincount(
            flat_idx,
            minlength=int(self.motion.num_motions) * self._failure_window_log_bin_count,
        ).to(torch.float32)
        self._failure_window_failure_hist += counts.view(
            int(self.motion.num_motions), self._failure_window_log_bin_count
        )
        return sampled

    def _sample_completion_recent30_failure_window_time_steps(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        motion_ids = self.motion_ids[env_ids]
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        last_idx = torch.maximum(start_idx, end_idx - 2)
        motion_len = (end_idx - start_idx).clamp(min=1)
        sampled = (
            start_idx
            + torch.floor(
                torch.rand(env_ids.numel(), device=self.device) * (motion_len - 1).clamp(min=1).to(torch.float32)
            ).long()
        )

        uniform_mix = min(max(float(self.motion_cfg.hotspot_failure_uniform_mix), 0.0), 1.0)
        use_completion_window = (
            torch.rand(env_ids.numel(), device=self.device) >= uniform_mix
        )
        completion_candidate_indices = torch.where(use_completion_window)[0]
        if completion_candidate_indices.numel() == 0:
            return sampled

        candidate_motion_ids = motion_ids[completion_candidate_indices]
        completion_stats = self._completion_recent30_reset_centers(candidate_motion_ids)
        if completion_stats is None:
            return sampled
        completion_centers, completion_valid = completion_stats
        completion_candidate_indices = completion_candidate_indices[completion_valid]
        candidate_motion_ids = candidate_motion_ids[completion_valid]
        completion_centers = completion_centers[completion_valid]
        if completion_candidate_indices.numel() == 0:
            return sampled

        candidate_start_idx = self.motion.motion_start_idx[candidate_motion_ids]
        candidate_end_idx = self.motion.motion_end_idx[candidate_motion_ids]
        candidate_lengths = (candidate_end_idx - candidate_start_idx - 1).clamp(min=1)
        target_local = torch.round(completion_centers * candidate_lengths.to(torch.float32)).long()
        target_global = candidate_start_idx + target_local
        sampled[completion_candidate_indices] = self._sample_around_failure_frames(
            candidate_motion_ids,
            target_global,
        )
        completion_env_ids = env_ids[completion_candidate_indices]
        self._failure_window_retry_active[completion_env_ids] = True
        self._failure_window_retry_motion_ids[completion_env_ids] = candidate_motion_ids
        self._failure_window_retry_target_steps[completion_env_ids] = target_global

        sampled = torch.minimum(torch.maximum(sampled, start_idx), last_idx)
        return sampled

    def _completion_recent30_reset_centers(
        self, motion_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return completion-window centers and validity for the requested motions."""
        if getattr(self, "_use_group_probe_envs", False):
            group_ids = self._motion_group_ids[motion_ids]
            return (
                self._group_probe_completion_recent30[group_ids].clamp(0.0, 1.0),
                self._group_probe_recent30_count[group_ids] > 0,
            )
        if not getattr(self, "_use_start_probe_envs", False):
            return None
        return (
            self._probe_completion_recent30[motion_ids].clamp(0.0, 1.0),
            self._probe_recent30_count[motion_ids] > 0,
        )

    def _sample_around_failure_frames(self, motion_ids: torch.Tensor, failed_time_steps: torch.Tensor) -> torch.Tensor:
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        last_idx = torch.maximum(start_idx, end_idx - 2)
        failed_idx = torch.minimum(torch.maximum(failed_time_steps, start_idx), last_idx)
        pre_frames = max(int(self.motion_cfg.failure_window_pre_frames), 0)
        post_frames = max(int(self.motion_cfg.failure_window_post_frames), 0)
        before_prob = min(max(float(self.motion_cfg.failure_window_before_prob), 0.0), 1.0)
        use_before = torch.rand(motion_ids.numel(), device=self.device) < before_prob
        before_start = torch.maximum(start_idx, failed_idx - pre_frames)
        before_end = failed_idx
        after_start = failed_idx
        after_end = torch.minimum(last_idx, failed_idx + post_frames)
        sample_start = torch.where(use_before, before_start, after_start)
        sample_end = torch.where(use_before, before_end, after_end)
        span = (sample_end - sample_start + 1).clamp(min=1)
        offsets = torch.floor(torch.rand(motion_ids.numel(), device=self.device) * span.to(torch.float32)).long()
        return sample_start + offsets

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACLAB3_NEWTON, (
            "Default-pose interpolation only supports the IsaacLab3 Newton backend in this migration copy."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
            "part_mask": torch.zeros(
                alphas.shape[0],
                self.motion._active_part_mask.shape[1],
                dtype=torch.bool,
                device=alphas.device,
            ),
            "contact_force_part": torch.zeros(
                alphas.shape[0],
                self.motion._contact_force_part_w.shape[1],
                3,
                dtype=start["body_pos"].dtype,
                device=alphas.device,
            ),
            "contact_force_part_mask": torch.zeros(
                alphas.shape[0],
                self.motion._contact_force_part_w.shape[1],
                dtype=torch.bool,
                device=alphas.device,
            ),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
