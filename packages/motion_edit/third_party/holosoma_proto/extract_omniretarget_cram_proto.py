#!/usr/bin/env python3
"""Extract outcome-style CRAM proto descriptors from OmniRetarget motions.

The extractor intentionally stores contact/anchor/outcome descriptors instead
of full all-joint reference trajectories.  By default only foot and hand
contacts participate in segmentation.  Knee and hip parts are kept in the
schema and can be enabled for future datasets, but they do not affect the
current climb motions unless explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from somaforge_core.robot_assets import (
    canonical_g1_urdf_path,
    encode_robot_asset_json,
    validate_canonical_g1_urdf,
)

try:
    import pinocchio as pin
except ModuleNotFoundError as exc:
    raise SystemExit("This script requires pinocchio. Run it in the active Holosoma conda env.") from exc


PART_ORDER = [
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
]

DEFAULT_CONTACT_PARTS = [
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
]

ACTIVE_MOTION_PARTS = {"left_foot", "right_foot", "left_hand", "right_hand"}

PART_FRAMES = {
    "left_foot": ["left_ankle_roll_link"],
    "right_foot": ["right_ankle_roll_link"],
    "left_hand": ["left_wrist_yaw_link"],
    "right_hand": ["right_wrist_yaw_link"],
    "left_knee": ["left_knee_link"],
    "right_knee": ["right_knee_link"],
    "left_hip": ["left_hip_yaw_link"],
    "right_hip": ["right_hip_yaw_link"],
}

CONTACT_PROBE_FRAMES = {
    "left_foot": [
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "left_ankle_roll_sphere_5_link",
    ],
    "right_foot": [
        "right_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_2_link",
        "right_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_4_link",
        "right_ankle_roll_sphere_5_link",
    ],
    "left_hand": ["left_sphere_hand_link", "left_sphere_hand_tip_link"],
    "right_hand": ["right_sphere_hand_link", "right_sphere_hand_tip_link"],
    "left_knee": ["left_knee_link"],
    "right_knee": ["right_knee_link"],
    "left_hip": ["left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link"],
    "right_hip": ["right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link"],
}

SHORT_NAME = {
    "left_foot": "LF",
    "right_foot": "RF",
    "left_hand": "LH",
    "right_hand": "RH",
    "left_knee": "LK",
    "right_knee": "RK",
    "left_hip": "LHip",
    "right_hip": "RHip",
}


@dataclass(frozen=True)
class Box:
    center_xy: np.ndarray
    axis_x: np.ndarray
    axis_y: np.ndarray
    half_x: float
    half_y: float
    z_min: float
    z_max: float


@dataclass
class MotionContext:
    path: Path
    terrain_urdf: Path
    qpos: np.ndarray
    fps: float
    boxes: list[Box]
    part_pos: dict[str, np.ndarray]
    part_probe_pos: dict[str, np.ndarray]
    root_pos: np.ndarray
    root_yaw: np.ndarray
    root_vel_local: np.ndarray
    projected_gravity: np.ndarray


def _ground_box() -> Box:
    return Box(
        center_xy=np.array([0.0, 0.0], dtype=np.float64),
        axis_x=np.array([1.0, 0.0], dtype=np.float64),
        axis_y=np.array([0.0, 1.0], dtype=np.float64),
        half_x=5.0,
        half_y=5.0,
        z_min=-1.0,
        z_max=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("robot-terrain"))
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--robot-urdf", type=Path, default=None)
    parser.add_argument(
        "--motions",
        nargs="*",
        default=None,
        help="Motion filenames or paths. Defaults to all .npz files in --data-dir.",
    )
    parser.add_argument(
        "--contact-parts",
        nargs="+",
        default=DEFAULT_CONTACT_PARTS,
        choices=PART_ORDER,
        help="Parts allowed to generate contact anchors. Knee/hip are available but disabled by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/omniretarget_cram_proto_50hz"))
    parser.add_argument("--output-name", type=str, default="outcome_proto")
    parser.add_argument("--max-segment-frames", type=int, default=100)
    parser.add_argument(
        "--contact-force-demo-dir",
        type=Path,
        default=None,
        help="Optional directory of rollout force-demo .npz files. If set, LF/RF/LH/RH contact masks come from contact_force_part_mask.",
    )
    parser.add_argument(
        "--contact-force-demo-file",
        type=Path,
        default=None,
        help="Optional single rollout force-demo .npz file to use for the selected motion.",
    )
    parser.add_argument(
        "--merge-transition-window",
        type=int,
        default=2,
        help="Fill foot contact gaps up to this many frames per end-effector before extracting anchors.",
    )
    parser.add_argument(
        "--hand-merge-transition-window",
        type=int,
        default=2,
        help="Fill hand contact gaps up to this many frames per end-effector before extracting anchors.",
    )
    parser.add_argument(
        "--body-merge-transition-window",
        type=int,
        default=2,
        help="Fill knee/hip contact gaps up to this many frames per body support if those contacts are enabled.",
    )
    parser.add_argument(
        "--stable-touchdown-window",
        type=int,
        default=3,
        help="Frames of free-before/contact-after required before a new stable contact anchor.",
    )
    parser.add_argument("--stable-touchdown-speed-thresh", type=float, default=0.12)
    parser.add_argument("--stable-touchdown-cluster-window", type=int, default=6)
    parser.add_argument("--free-state-window", type=int, default=100)
    parser.add_argument("--free-transition-active-window", type=int, default=30)
    parser.add_argument("--min-root-delta", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Print segmentation summary without writing files.")
    parser.add_argument(
        "--write-holosoma-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write Holosoma motion npz copies with frame-wise A2A masks appended.",
    )
    parser.add_argument(
        "--holosoma-motion-dir",
        type=Path,
        default=None,
        help="Directory containing converted Holosoma .npz files. Defaults to <dataset-root>/data/holosoma_motions_50hz.",
    )
    parser.add_argument(
        "--holosoma-output-dir",
        type=Path,
        default=None,
        help="Output directory for masked Holosoma motion npz files. Defaults to <dataset-root>/data/holosoma_motions_masked_50hz.",
    )
    parser.add_argument(
        "--holosoma-motion-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the raw motion stem when looking up converted Holosoma motions.",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="With --write-holosoma-masks, skip proto/csv output and only write masked Holosoma motions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting masked Holosoma motion outputs.",
    )
    return parser.parse_args()


def _parse_motion_name(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<terrain>.+)_z_scale_(?P<scale>[0-9.]+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer terrain from {path.name}; expected '*_z_scale_<scale>.npz'.")
    return match.group("terrain"), match.group("scale")


def _motion_paths(data_dir: Path, motions: list[str] | None) -> list[Path]:
    if motions:
        paths = [Path(m) if Path(m).is_absolute() else data_dir / m for m in motions]
    else:
        paths = sorted(data_dir.glob("*_z_scale_1.0.npz"))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing motion file(s): {missing}")
    return paths


def _load_motion(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path) as data:
        qpos_key = "qpos" if "qpos" in data else "joint_pos"
        qpos = np.asarray(data[qpos_key], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item())
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise RuntimeError(f"{path} expected qpos shape [T,36], got {qpos.shape}")
    if fps <= 0.0:
        raise RuntimeError(f"{path} has invalid fps={fps}")
    return qpos, fps


def _parse_obj_vertices(obj_path: Path, scale: tuple[float, float, float]) -> np.ndarray:
    vertices: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, x, y, z = line.strip().split()[:4]
            vertices.append([float(x) * scale[0], float(y) * scale[1], float(z) * scale[2]])
    if not vertices:
        raise RuntimeError(f"No vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float64)


def _box_from_vertices(vertices: np.ndarray) -> Box:
    xy = vertices[:, :2]
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    axis_x = vh[0]
    axis_y = vh[1]
    rel = xy - center
    return Box(
        center_xy=center,
        axis_x=axis_x,
        axis_y=axis_y,
        half_x=float(np.ptp(rel @ axis_x) * 0.5),
        half_y=float(np.ptp(rel @ axis_y) * 0.5),
        z_min=float(vertices[:, 2].min()),
        z_max=float(vertices[:, 2].max()),
    )


def _terrain_boxes(terrain_urdf: Path) -> list[Box]:
    root = ET.parse(terrain_urdf).getroot()
    boxes: list[Box] = []
    seen: set[tuple[str, str]] = set()
    for mesh in root.findall(".//collision/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        scale_text = mesh.attrib.get("scale", "1 1 1")
        key = (filename, scale_text)
        if key in seen:
            continue
        seen.add(key)
        scale = tuple(float(v) for v in scale_text.split())
        if len(scale) != 3:
            raise RuntimeError(f"Invalid scale in {terrain_urdf}: {scale_text}")
        vertices = _parse_obj_vertices((terrain_urdf.parent / filename).resolve(), scale)
        boxes.append(_box_from_vertices(vertices))
    if not boxes:
        raise RuntimeError(f"No terrain collision meshes found in {terrain_urdf}")
    return [_ground_box()] + boxes


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rot_z(points: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return points @ rot.T


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _projected_gravity(qwxyz: np.ndarray) -> np.ndarray:
    out = np.zeros((len(qwxyz), 3), dtype=np.float64)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for i, q in enumerate(qwxyz):
        out[i] = _quat_wxyz_to_rot(q).T @ gravity_world
    return out


def _pin_qpos(qpos: np.ndarray, model: pin.Model) -> np.ndarray:
    out = np.zeros(model.nq, dtype=np.float64)
    out[:3] = qpos[4:7]
    out[3:7] = [qpos[1], qpos[2], qpos[3], qpos[0]]
    out[7:] = qpos[7 : 7 + model.nq - 7]
    return out


def _part_positions(qpos: np.ndarray, robot_urdf: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = pin.buildModelFromUrdf(str(robot_urdf), pin.JointModelFreeFlyer())
    data = model.createData()
    frame_ids = {
        part: [model.getFrameId(name) for name in names if model.existFrame(name)]
        for part, names in PART_FRAMES.items()
    }
    probe_frame_ids = {
        part: [model.getFrameId(name) for name in names if model.existFrame(name)]
        for part, names in CONTACT_PROBE_FRAMES.items()
    }
    missing = [part for part, ids in frame_ids.items() if not ids]
    missing += [f"{part} contact probes" for part, ids in probe_frame_ids.items() if not ids]
    if missing:
        raise RuntimeError(f"Missing robot frames for parts: {missing}")

    positions = {part: np.zeros((len(qpos), 3), dtype=np.float64) for part in PART_ORDER}
    probe_positions = {
        part: np.zeros((len(qpos), len(probe_frame_ids[part]), 3), dtype=np.float64) for part in PART_ORDER
    }
    for i, q in enumerate(qpos):
        pin.forwardKinematics(model, data, _pin_qpos(q, model))
        pin.updateFramePlacements(model, data)
        for part, ids in frame_ids.items():
            points = np.asarray([data.oMf[fid].translation for fid in ids], dtype=np.float64)
            positions[part][i] = points.mean(axis=0)
        for part, ids in probe_frame_ids.items():
            probe_positions[part][i] = np.asarray([data.oMf[fid].translation for fid in ids], dtype=np.float64)
    return positions, probe_positions


def _local_velocity(xyz: np.ndarray, yaw: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(xyz)
    vel[1:] = np.diff(xyz, axis=0) * fps
    for i in range(len(vel)):
        vel[i] = _rot_z(vel[i : i + 1], -float(yaw[i]))[0]
    return vel


def _load_context(motion: Path, dataset_root: Path, robot_urdf: Path) -> MotionContext:
    terrain_name, z_scale = _parse_motion_name(motion)
    terrain_urdf = dataset_root / "models/terrain" / terrain_name / f"multi_boxes_z_scale_{z_scale}.urdf"
    if not terrain_urdf.exists():
        raise FileNotFoundError(f"Missing terrain URDF: {terrain_urdf}")
    qpos, fps = _load_motion(motion)
    root_pos = qpos[:, 4:7].copy()
    root_yaw = _yaw_from_quat_wxyz(qpos[:, :4])
    part_pos, part_probe_pos = _part_positions(qpos, robot_urdf)
    return MotionContext(
        path=motion,
        terrain_urdf=terrain_urdf,
        qpos=qpos,
        fps=fps,
        boxes=_terrain_boxes(terrain_urdf),
        part_pos=part_pos,
        part_probe_pos=part_probe_pos,
        root_pos=root_pos,
        root_yaw=root_yaw,
        root_vel_local=_local_velocity(root_pos, root_yaw, fps),
        projected_gravity=_projected_gravity(qpos[:, :4]),
    )


def _interp_time(values: np.ndarray, target_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if len(values) == target_frames and abs(source_fps - target_fps) < 1.0e-6:
        return values.copy()
    source_t = np.arange(len(values), dtype=np.float64) / source_fps
    target_t = np.arange(target_frames, dtype=np.float64) / target_fps
    flat = values.reshape(len(values), -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(target_t, source_t, flat[:, i])
    return out.reshape((target_frames, *values.shape[1:]))


def _resample_context(ctx: MotionContext, target_frames: int, target_fps: float) -> MotionContext:
    qpos = _interp_time(ctx.qpos, target_frames, ctx.fps, target_fps)
    quat_norm = np.linalg.norm(qpos[:, :4], axis=1, keepdims=True)
    qpos[:, :4] /= np.maximum(quat_norm, 1.0e-8)
    root_pos = _interp_time(ctx.root_pos, target_frames, ctx.fps, target_fps)
    root_yaw = _interp_time(np.unwrap(ctx.root_yaw)[:, None], target_frames, ctx.fps, target_fps)[:, 0]
    part_pos = {part: _interp_time(pos, target_frames, ctx.fps, target_fps) for part, pos in ctx.part_pos.items()}
    part_probe_pos = {
        part: _interp_time(pos, target_frames, ctx.fps, target_fps) for part, pos in ctx.part_probe_pos.items()
    }
    return MotionContext(
        path=ctx.path,
        terrain_urdf=ctx.terrain_urdf,
        qpos=qpos,
        fps=target_fps,
        boxes=ctx.boxes,
        part_pos=part_pos,
        part_probe_pos=part_probe_pos,
        root_pos=root_pos,
        root_yaw=root_yaw,
        root_vel_local=_local_velocity(root_pos, root_yaw, target_fps),
        projected_gravity=_interp_time(ctx.projected_gravity, target_frames, ctx.fps, target_fps),
    )


def _holosoma_motion_info(raw_motion_path: Path, holosoma_motion_dir: Path, suffix: str) -> tuple[Path, int, float]:
    source = holosoma_motion_dir / f"{raw_motion_path.stem}{suffix}.npz"
    if not source.exists():
        raise FileNotFoundError(f"Missing converted Holosoma motion for {raw_motion_path.name}: {source}")
    with np.load(source, allow_pickle=True) as data:
        if "joint_pos" not in data:
            raise KeyError(f"{source} is not a converted Holosoma motion: missing 'joint_pos'")
        target_frames = int(data["joint_pos"].shape[0])
        target_fps = float(np.asarray(data.get("fps", 0.0)).item())
    if target_frames <= 0 or target_fps <= 0.0:
        raise ValueError(f"{source} has invalid frames/fps: {target_frames}, {target_fps}")
    return source, target_frames, target_fps


def _point_box_signed_distance(point: np.ndarray, box: Box) -> float:
    dxy = point[:2] - box.center_xy
    lx = float(dxy @ box.axis_x)
    ly = float(dxy @ box.axis_y)
    dx = abs(lx) - box.half_x
    dy = abs(ly) - box.half_y
    dz = max(box.z_min - float(point[2]), 0.0, float(point[2]) - box.z_max)
    if dx <= 0.0 and dy <= 0.0 and box.z_min <= point[2] <= box.z_max:
        return -min(box.half_x - abs(lx), box.half_y - abs(ly), point[2] - box.z_min, box.z_max - point[2])
    return float(np.linalg.norm([max(dx, 0.0), max(dy, 0.0), dz]))


def _nearest_surface_distance(point: np.ndarray, boxes: list[Box]) -> float:
    return min((_point_box_signed_distance(point, box) for box in boxes), key=abs)


def _top_gap(point: np.ndarray, boxes: list[Box]) -> float:
    candidates = []
    for box in boxes:
        dxy = point[:2] - box.center_xy
        if abs(float(dxy @ box.axis_x)) <= box.half_x + 0.04 and abs(float(dxy @ box.axis_y)) <= box.half_y + 0.04:
            candidates.append(float(point[2]) - box.z_max)
    if not candidates:
        candidates.append(float(point[2]))
    return min(candidates, key=abs)


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.r_[mask.astype(bool), False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _clean_contact(mask: np.ndarray, min_island: int = 3) -> np.ndarray:
    cleaned = mask.astype(bool).copy()
    for i in range(1, len(cleaned) - 1):
        if not cleaned[i] and cleaned[i - 1] and cleaned[i + 1]:
            cleaned[i] = True
    out = np.zeros_like(cleaned)
    for start, end in _intervals(cleaned):
        if end - start + 1 >= min_island:
            out[start : end + 1] = True
    return out


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    filled = mask.astype(bool).copy()
    if max_gap <= 0:
        return filled
    for start, end in _intervals(~filled):
        if start == 0 or end == len(filled) - 1:
            continue
        if end - start + 1 <= max_gap:
            filled[start : end + 1] = True
    return filled


def _debounce_window_for_part(part: str, foot_gap: int, hand_gap: int, body_gap: int) -> int:
    if "foot" in part:
        return foot_gap
    if "hand" in part:
        return hand_gap
    return body_gap


def _debounced_contact(contact: np.ndarray, foot_gap: int, hand_gap: int, body_gap: int) -> np.ndarray:
    debounced = contact.astype(bool).copy()
    for part_i, part in enumerate(PART_ORDER):
        max_gap = _debounce_window_for_part(part, foot_gap, hand_gap, body_gap)
        debounced[:, part_i] = _fill_short_gaps(contact[:, part_i], max_gap)
    return debounced


def _merge_anchor_events(
    num_frames: int,
    num_parts: int,
    events: list[tuple[int, int]],
    cluster_window: int,
) -> tuple[list[int], dict[int, np.ndarray]]:
    touchdown_by_anchor: dict[int, np.ndarray] = {}
    cluster_frame: int | None = None
    cluster_mask = np.zeros(num_parts, dtype=bool)
    for frame, part_i in sorted(events):
        if cluster_frame is None:
            cluster_frame = frame
            cluster_mask = np.zeros(num_parts, dtype=bool)
            cluster_mask[part_i] = True
            continue
        if frame - cluster_frame <= cluster_window:
            cluster_frame = frame
            cluster_mask[part_i] = True
            continue
        touchdown_by_anchor[cluster_frame] = cluster_mask.copy()
        cluster_frame = frame
        cluster_mask = np.zeros(num_parts, dtype=bool)
        cluster_mask[part_i] = True
    if cluster_frame is not None:
        touchdown_by_anchor[cluster_frame] = cluster_mask.copy()

    anchors = sorted(touchdown_by_anchor)
    if not anchors or anchors[0] != 0:
        anchors.insert(0, 0)
    if anchors[-1] != num_frames:
        anchors.append(num_frames)
    return anchors, touchdown_by_anchor


def _stable_contact_anchor_maps(
    ctx: MotionContext,
    contact: np.ndarray,
    enabled_indices: list[int],
    free_window: int,
    contact_window: int,
    stable_speed_thresh: float,
    cluster_window: int,
) -> tuple[list[int], dict[int, np.ndarray]]:
    events: list[tuple[int, int]] = []
    role_indices = [i for i in enabled_indices if PART_ORDER[i] in ACTIVE_MOTION_PARTS]
    free_window = max(1, int(free_window))
    contact_window = max(1, int(contact_window))
    for part_i in role_indices:
        part = PART_ORDER[part_i]
        part_contact = contact[:, part_i].astype(bool)
        vel = np.zeros_like(ctx.part_pos[part])
        vel[1:] = np.diff(ctx.part_pos[part], axis=0) * ctx.fps
        speed = np.linalg.norm(vel, axis=1)
        for frame in range(free_window, len(contact) - contact_window + 1):
            if part_contact[frame - 1] or not part_contact[frame]:
                continue
            if part_contact[frame - free_window : frame].any():
                continue
            for candidate in range(frame, len(contact) - contact_window + 1):
                if not part_contact[candidate]:
                    break
                if part_contact[candidate : candidate + contact_window].all() and np.all(
                    speed[candidate : candidate + contact_window] <= stable_speed_thresh
                ):
                    events.append((candidate, part_i))
                    break

    return _merge_anchor_events(len(contact), contact.shape[1], events, cluster_window)


def _anchor_maps(
    ctx: MotionContext,
    debounced_contact: np.ndarray,
    enabled_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, np.ndarray]]:
    return _stable_contact_anchor_maps(
        ctx,
        debounced_contact,
        enabled_indices,
        args.stable_touchdown_window,
        args.stable_touchdown_window,
        args.stable_touchdown_speed_thresh,
        args.stable_touchdown_cluster_window,
    )


def _contact_masks(ctx: MotionContext, enabled_parts: set[str]) -> np.ndarray:
    masks = np.zeros((len(ctx.qpos), len(PART_ORDER)), dtype=bool)
    for part_i, part in enumerate(PART_ORDER):
        if part not in enabled_parts:
            continue
        pos = ctx.part_pos[part]
        if "foot" in part:
            probes = ctx.part_probe_pos[part]
            gap = np.asarray(
                [[_top_gap(point, ctx.boxes) for point in frame_points] for frame_points in probes],
                dtype=np.float64,
            )
            gap_vel = np.zeros_like(gap)
            gap_vel[1:] = np.diff(gap, axis=0) * ctx.fps
            probe_contact = (np.abs(gap) < 0.035) & (np.abs(gap_vel) < 0.35)
            masks[:, part_i] = _clean_contact(probe_contact.any(axis=1))
        elif "hand" in part:
            probe_center = ctx.part_probe_pos[part].mean(axis=1)
            dist = np.asarray([_nearest_surface_distance(p, ctx.boxes) for p in probe_center], dtype=np.float64)
            strict = np.abs(dist) < 0.06
            keep = np.abs(dist) < 0.065
            hand_contact = np.zeros(len(pos), dtype=bool)
            for frame in range(len(pos)):
                if strict[frame] or (frame > 0 and hand_contact[frame - 1] and keep[frame]):
                    hand_contact[frame] = True
            masks[:, part_i] = _clean_contact(hand_contact)
        else:
            dist = np.asarray([_nearest_surface_distance(p, ctx.boxes) for p in pos], dtype=np.float64)
            masks[:, part_i] = _clean_contact(np.abs(dist) < 0.035)
    return masks


def _force_demo_candidates(motion_path: Path, demo_dir: Path) -> list[Path]:
    candidates = sorted(demo_dir.expanduser().glob(f"{motion_path.stem}_force_demo*.npz"))
    match = re.search(r"climb_(\d+)", motion_path.stem)
    if match is not None:
        climb_id = int(match.group(1))
        candidates.extend(
            [
                demo_dir.expanduser() / f"climb_{climb_id:02d}_rollout_ref_contact_force.npz",
                *sorted(demo_dir.expanduser().glob(f"climb_{climb_id:02d}_*contact_force*.npz")),
            ]
        )
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _load_force_demo_contact_mask(
    motion_path: Path,
    demo_dir: Path | None,
    demo_file: Path | None,
    num_frames: int,
) -> tuple[np.ndarray, set[str]] | None:
    if demo_file is not None:
        candidates = [demo_file.expanduser()]
    elif demo_dir is not None:
        candidates = _force_demo_candidates(motion_path, demo_dir)
    else:
        return None
    if not candidates:
        raise FileNotFoundError(f"No force-demo contact file found for {motion_path.name} in {demo_dir}")
    demo_path = candidates[-1]
    if not demo_path.exists():
        raise FileNotFoundError(f"Force-demo contact file does not exist: {demo_path}")
    with np.load(demo_path, allow_pickle=True) as data:
        if "contact_force_part_mask" not in data or "contact_force_part_order" not in data:
            raise KeyError(f"{demo_path} must contain contact_force_part_mask and contact_force_part_order")
        force_mask = np.asarray(data["contact_force_part_mask"], dtype=np.bool_)
        force_order = [str(name) for name in data["contact_force_part_order"].tolist()]

    if force_mask.ndim != 2:
        raise ValueError(f"{demo_path} contact_force_part_mask must be 2D, got {force_mask.shape}")
    if force_mask.shape[0] != num_frames:
        print(
            f"[force-contact] {motion_path.name}: skip {demo_path} for {num_frames} frames "
            f"(force demo has {force_mask.shape[0]} frames)"
        )
        return None

    contact = np.zeros((num_frames, len(PART_ORDER)), dtype=bool)
    short_to_part = {
        "LF": "left_foot",
        "RF": "right_foot",
        "LH": "left_hand",
        "RH": "right_hand",
        "LK": "left_knee",
        "RK": "right_knee",
    }
    present_parts = set()
    for src_i, short_name in enumerate(force_order):
        part = short_to_part.get(short_name)
        if part is None:
            continue
        contact[:, PART_ORDER.index(part)] = force_mask[:, src_i]
        present_parts.add(part)
    print(f"[force-contact] {motion_path.name}: using {demo_path}")
    return contact, present_parts


def _contact_masks_with_force_demo(
    ctx: MotionContext,
    enabled_parts: set[str],
    demo_dir: Path | None,
    demo_file: Path | None,
) -> np.ndarray:
    contact = _contact_masks(ctx, enabled_parts)
    loaded = _load_force_demo_contact_mask(ctx.path, demo_dir, demo_file, len(ctx.qpos))
    if loaded is None:
        return contact
    force_contact, present_parts = loaded
    for part in present_parts:
        if part in enabled_parts:
            part_i = PART_ORDER.index(part)
            contact[:, part_i] = force_contact[:, part_i]
    return contact


def _local_points(points: np.ndarray, root: np.ndarray, yaw: float) -> np.ndarray:
    return _rot_z(points - root[None, :], -yaw)


def _root_delta(ctx: MotionContext, start: int, end: int) -> np.ndarray:
    end_i = end - 1
    delta = ctx.root_pos[end_i] - ctx.root_pos[start]
    delta_local = _rot_z(delta[None, :], -float(ctx.root_yaw[start]))[0]
    yaw_delta = float(_wrap_angle(ctx.root_yaw[end_i] - ctx.root_yaw[start]))
    return np.asarray([delta_local[0], delta_local[1], delta_local[2], yaw_delta], dtype=np.float32)


def _root_summary(ctx: MotionContext, frame: int) -> np.ndarray:
    return np.r_[
        ctx.root_pos[frame, 2],
        ctx.root_yaw[frame],
        ctx.root_vel_local[frame],
        ctx.projected_gravity[frame],
    ].astype(np.float32)


def _global_role_masks(
    ctx: MotionContext,
    contact: np.ndarray,
    anchors: list[int],
    touchdown_by_anchor: dict[int, np.ndarray],
    stable_speed_thresh: float,
    stable_window: int,
    free_state_window: int,
    free_transition_active_window: int,
) -> dict[str, np.ndarray]:
    num_frames = contact.shape[0]
    stable_window = max(1, int(stable_window))
    free_state_window = max(1, int(free_state_window))
    free_transition_active_window = max(0, int(free_transition_active_window))
    role_indices = [PART_ORDER.index(part) for part in ACTIVE_MOTION_PARTS]
    support = np.zeros_like(contact, dtype=bool)
    active = np.zeros_like(contact, dtype=bool)
    free = np.zeros_like(contact, dtype=bool)
    release_event = np.zeros_like(contact, dtype=bool)
    acquire_event = np.zeros_like(contact, dtype=bool)
    active_owner = np.full(contact.shape, -1, dtype=np.int32)

    for part_i in role_indices:
        part = PART_ORDER[part_i]
        c = contact[:, part_i].astype(bool)
        for frame in range(1, num_frames):
            if not c[frame - 1] and c[frame]:
                acquire_event[frame, part_i] = True
            if c[frame - 1] and not c[frame]:
                release_event[frame, part_i] = True

        # Rebuild from event intervals so support spans acquire->release and
        # active spans release->acquire. The contact mask is used only to find
        # event boundaries, not to gate support by speed within the interval.
        support[:, part_i] = False
        active[:, part_i] = False
        cursor = 0
        in_support = bool(c[0])
        for frame in range(1, num_frames):
            if not c[frame - 1] and c[frame]:
                active[cursor:frame, part_i] = True
                cursor = frame
                in_support = True
            elif c[frame - 1] and not c[frame]:
                support[cursor:frame, part_i] = True
                cursor = frame
                in_support = False
        if in_support:
            support[cursor:num_frames, part_i] = True
        else:
            active[cursor:num_frames, part_i] = True

        for start, end in _intervals(~c):
            length = end - start + 1
            if length < free_state_window:
                continue
            free_start = start
            free_end = end + 1
            if start > 0:
                free_start = min(free_end, start + free_transition_active_window)
            if end < num_frames - 1:
                free_end = max(free_start, end + 1 - free_transition_active_window)
            free[free_start:free_end, part_i] = True

        support[active[:, part_i], part_i] = False
        active[free[:, part_i], part_i] = False
        support[free[:, part_i], part_i] = False

    anchors = sorted(anchors)
    proto_ranges = list(zip(anchors[:-1], anchors[1:]))
    for part_i in role_indices:
        for active_start, active_end_inclusive in _intervals(active[:, part_i]):
            active_end = active_end_inclusive + 1
            best_proto_id = None
            best_overlap = 0
            for proto_id, (proto_start, proto_end) in enumerate(proto_ranges):
                overlap = min(active_end, proto_end) - max(active_start, proto_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_proto_id = proto_id
            if best_proto_id is not None and best_overlap > 0:
                active_owner[active_start:active_end, part_i] = best_proto_id

    for proto_id, (start, end) in enumerate(zip(anchors[:-1], anchors[1:])):
        owned = active[start:end] & (active_owner[start:end] < 0)
        owner_slice = active_owner[start:end]
        owner_slice[owned] = proto_id
        active_owner[start:end] = owner_slice

    return {
        "support": support,
        "active": active,
        "free": free,
        "release_event": release_event,
        "acquire_event": acquire_event,
        "active_owner": active_owner,
    }


def _segment_role_summary(
    contact_start: np.ndarray,
    end_touchdown_event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(end_touchdown_event, dtype=bool).copy()
    support = np.asarray(contact_start, dtype=bool).copy()
    support[active] = False
    free = ~(active | support)
    return active.astype(bool), support.astype(bool), free.astype(bool)


def _segment_records(
    ctx: MotionContext,
    contact: np.ndarray,
    anchors: list[int],
    touchdown_by_anchor: dict[int, np.ndarray],
    min_root_delta: float,
    role_masks: dict[str, np.ndarray],
) -> list[dict]:
    records: list[dict] = []
    for proto_id, (start, end) in enumerate(zip(anchors[:-1], anchors[1:])):
        end_i = min(end, len(ctx.qpos) - 1)
        delta_pos = ctx.root_pos[end_i] - ctx.root_pos[start]
        delta_local = _rot_z(delta_pos[None, :], -float(ctx.root_yaw[start]))[0]
        root_delta = np.asarray(
            [
                delta_local[0],
                delta_local[1],
                delta_local[2],
                float(_wrap_angle(ctx.root_yaw[end_i] - ctx.root_yaw[start])),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(root_delta[:3]) < min_root_delta:
            continue

        c0 = contact[start].copy()
        c1 = contact[end_i].copy()
        start_touchdown_event = touchdown_by_anchor.get(start, np.zeros_like(c0))
        end_active_event = touchdown_by_anchor.get(end, np.zeros_like(c0))
        delta = np.logical_xor(c0, c1)
        displacements = np.asarray(
            [np.linalg.norm(ctx.part_pos[p][end_i] - ctx.part_pos[p][start]) for p in PART_ORDER],
            dtype=np.float64,
        )
        active, support, free = _segment_role_summary(c0, end_active_event)
        segment_len = max(0, end - start)
        active_frame = np.broadcast_to(active, (segment_len, active.shape[0])).copy()
        support_frame = np.broadcast_to(support, (segment_len, support.shape[0])).copy()
        free_frame = np.broadcast_to(free, (segment_len, free.shape[0])).copy()
        records.append(
            {
                "start": start,
                "end": end,
                "end_state": end_i,
                "contact_start": c0,
                "contact_end": c1,
                "touchdown_part": start_touchdown_event.copy(),
                "contact_delta": delta,
                "active_part": active,
                "support_part": support,
                "free_part": free,
                "active_frame_part": active_frame,
                "support_frame_part": support_frame,
                "free_frame_part": free_frame,
                "root_delta": root_delta,
                "root_start": _root_summary(ctx, start),
                "root_end": _root_summary(ctx, end_i),
                "part_displacement": displacements.astype(np.float32),
            }
        )
    return records


def _pack_records(contexts: list[MotionContext], by_motion: list[list[dict]], max_len: int) -> dict:
    records = [(ctx, rec) for ctx, recs in zip(contexts, by_motion) for rec in recs]
    n = len(records)
    p = len(PART_ORDER)
    contact_start = np.zeros((n, p), dtype=bool)
    contact_end = np.zeros((n, p), dtype=bool)
    touchdown_part = np.zeros((n, p), dtype=bool)
    contact_delta = np.zeros((n, p), dtype=bool)
    active_part = np.zeros((n, p), dtype=bool)
    support_part = np.zeros((n, p), dtype=bool)
    active_start = np.zeros((n, p, 3), dtype=np.float32)
    active_end = np.zeros((n, p, 3), dtype=np.float32)
    active_traj = np.zeros((n, max_len, p, 3), dtype=np.float32)
    traj_mask = np.zeros((n, max_len), dtype=bool)
    root_delta = np.zeros((n, 4), dtype=np.float32)
    root_start = np.zeros((n, 8), dtype=np.float32)
    root_end = np.zeros((n, 8), dtype=np.float32)
    anchor = np.zeros((n, 2), dtype=np.int64)
    duration = np.zeros(n, dtype=np.float32)
    dt = np.zeros(n, dtype=np.float32)
    part_displacement = np.zeros((n, p), dtype=np.float32)
    motion_index = np.zeros(n, dtype=np.int64)
    motion_name: list[str] = []

    for i, (ctx, rec) in enumerate(records):
        start = int(rec["start"])
        end = int(rec["end"])
        end_state = int(rec["end_state"])
        length = min(end_state - start + 1, max_len)
        root = ctx.root_pos[start]
        yaw = float(ctx.root_yaw[start])
        contact_start[i] = rec["contact_start"]
        contact_end[i] = rec["contact_end"]
        touchdown_part[i] = rec["touchdown_part"]
        contact_delta[i] = rec["contact_delta"]
        active_part[i] = rec["active_part"]
        support_part[i] = rec["support_part"]
        root_delta[i] = rec["root_delta"]
        root_start[i] = rec["root_start"]
        root_end[i] = rec["root_end"]
        anchor[i] = [start, end]
        duration[i] = (end - start) / ctx.fps
        dt[i] = 1.0 / ctx.fps
        part_displacement[i] = rec["part_displacement"]
        motion_index[i] = contexts.index(ctx)
        motion_name.append(ctx.path.name)

        for part_i, part in enumerate(PART_ORDER):
            if not active_part[i, part_i]:
                continue
            traj = _local_points(ctx.part_pos[part][start : start + length], root, yaw)
            active_traj[i, :length, part_i] = traj.astype(np.float32)
            active_start[i, part_i] = traj[0].astype(np.float32)
            active_end[i, part_i] = _local_points(ctx.part_pos[part][end_state : end_state + 1], root, yaw)[0].astype(np.float32)
        traj_mask[i, :length] = True

    return {
        "schema_version": "omniretarget_outcome_proto_v1",
        "part_order": PART_ORDER,
        "default_contact_parts": DEFAULT_CONTACT_PARTS,
        "motion_files": [ctx.path.name for ctx in contexts],
        "motion_name": motion_name,
        "motion_index": torch.as_tensor(motion_index),
        "anchor": torch.as_tensor(anchor),
        "duration": torch.as_tensor(duration),
        "dt": torch.as_tensor(dt),
        "contact_start_mask": torch.as_tensor(contact_start),
        "contact_end_mask": torch.as_tensor(contact_end),
        "touchdown_part_mask": torch.as_tensor(touchdown_part),
        "query_touchdown_mask": torch.as_tensor(touchdown_part[:, :4]),
        "contact_delta_mask": torch.as_tensor(contact_delta),
        "active_part_mask": torch.as_tensor(active_part),
        "support_part_mask": torch.as_tensor(support_part),
        "root_delta": torch.as_tensor(root_delta),
        "root_start": torch.as_tensor(root_start),
        "root_end": torch.as_tensor(root_end),
        "active_start_local": torch.as_tensor(active_start),
        "active_end_local": torch.as_tensor(active_end),
        "active_traj_local": torch.as_tensor(active_traj),
        "active_traj_valid_mask": torch.as_tensor(traj_mask),
        "part_displacement": torch.as_tensor(part_displacement),
        "ref_feat": torch.cat(
            [
                torch.as_tensor(root_delta),
                torch.as_tensor(active_end.reshape(n, -1)),
                torch.as_tensor(contact_end.astype(np.float32)),
                torch.as_tensor(duration[:, None]),
            ],
            dim=1,
        ),
        "quality": torch.ones(n, dtype=torch.float32),
        "usage_count": torch.zeros(n, dtype=torch.long),
        "age": torch.zeros(n, dtype=torch.long),
    }


def _frame_role_masks(contact: np.ndarray, records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.zeros_like(contact, dtype=bool)
    support = np.zeros_like(contact, dtype=bool)
    free = np.zeros_like(contact, dtype=bool)
    for rec in records:
        start = int(rec["start"])
        end = int(rec["end"])
        overlap = np.logical_and(rec["active_part"], rec["support_part"])
        if overlap.any():
            parts = [PART_ORDER[i] for i in np.flatnonzero(overlap)]
            raise ValueError(f"active/support overlap in segment {start}->{end}: {parts}")
        if rec.get("active_frame_part") is not None and rec.get("support_frame_part") is not None:
            active[start:end] = rec["active_frame_part"]
            support[start:end] = rec["support_frame_part"]
            if rec.get("free_frame_part") is not None:
                free[start:end] = rec["free_frame_part"]
        else:
            active[start:end] = rec["active_part"]
            support[start:end] = rec["support_part"]
            free[start:end] = rec.get("free_part", ~(rec["active_part"] | rec["support_part"]))
    return active, support, free


def _records_to_target_axis(
    ctx: MotionContext,
    contact: np.ndarray,
    records: list[dict],
    source_fps: float,
) -> list[dict]:
    target_records: list[dict] = []
    for rec in records:
        start = int(round(float(rec["start"]) / source_fps * ctx.fps))
        end = int(round(float(rec["end"]) / source_fps * ctx.fps))
        start = max(0, min(start, len(ctx.qpos) - 1))
        end = max(start + 1, min(end, len(ctx.qpos)))
        end_state = min(end, len(ctx.qpos) - 1)
        active = np.asarray(rec["active_part"], dtype=bool).copy()
        support = np.asarray(rec["support_part"], dtype=bool).copy()
        free = np.asarray(rec.get("free_part", ~(active | support)), dtype=bool).copy()
        overlap = active & support
        if overlap.any():
            parts = [PART_ORDER[i] for i in np.flatnonzero(overlap)]
            raise ValueError(f"active/support overlap in source segment {rec['start']}->{rec['end']}: {parts}")
        free[active | support] = False

        c0 = contact[start].copy()
        c1 = contact[end_state].copy()
        displacements = np.asarray(
            [np.linalg.norm(ctx.part_pos[p][end_state] - ctx.part_pos[p][start]) for p in PART_ORDER],
            dtype=np.float32,
        )
        target_records.append(
            {
                "start": start,
                "end": end,
                "end_state": end_state,
                "contact_start": c0,
                "contact_end": c1,
                "touchdown_part": np.asarray(rec["touchdown_part"], dtype=bool).copy(),
                "contact_delta": np.logical_xor(c0, c1),
                "active_part": active,
                "support_part": support,
                "free_part": free,
                "root_delta": _root_delta(ctx, start, end),
                "root_start": _root_summary(ctx, start),
                "root_end": _root_summary(ctx, end_state),
                "part_displacement": displacements,
            }
        )
    return target_records


def _resample_frame_mask(mask: np.ndarray, target_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if mask.shape[0] == target_frames:
        return mask
    if source_fps <= 0.0 or target_fps <= 0.0:
        indices = np.linspace(0, mask.shape[0] - 1, target_frames).round().astype(np.int64)
    else:
        source_time = np.arange(target_frames, dtype=np.float64) / target_fps
        indices = np.rint(source_time * source_fps).astype(np.int64)
    indices = np.clip(indices, 0, mask.shape[0] - 1)
    return mask[indices]


def _write_holosoma_motion_with_masks(
    raw_motion_path: Path,
    holosoma_motion_dir: Path,
    output_dir: Path,
    suffix: str,
    source_fps: float,
    contact: np.ndarray,
    records: list[dict],
    role_masks: dict[str, np.ndarray] | None,
    overwrite: bool,
) -> Path:
    source = holosoma_motion_dir / f"{raw_motion_path.stem}{suffix}.npz"
    if not source.exists():
        raise FileNotFoundError(f"Missing converted Holosoma motion for {raw_motion_path.name}: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / source.name
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite to replace it.")

    with np.load(source, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    num_frames = arrays["joint_pos"].shape[0]
    active, support, free = _frame_role_masks(contact, records)
    target_fps = float(np.asarray(arrays.get("fps", source_fps)).item())
    if contact.shape[0] != num_frames:
        print(
            f"[resample] {source.name}: raw masks {contact.shape[0]} frames @ {source_fps:g} fps -> "
            f"Holosoma {num_frames} frames @ {target_fps:g} fps"
        )
        contact = _resample_frame_mask(contact, num_frames, source_fps, target_fps)
        active = _resample_frame_mask(active, num_frames, source_fps, target_fps)
        support = _resample_frame_mask(support, num_frames, source_fps, target_fps)
        free = _resample_frame_mask(free, num_frames, source_fps, target_fps)
    overlap = (active & support) | (active & free) | (support & free)
    if overlap.any():
        first = []
        for frame_i, part_i in np.argwhere(overlap)[:16]:
            first.append(f"frame={int(frame_i)} part={PART_ORDER[int(part_i)]}")
        raise ValueError(f"{raw_motion_path.name} frame role overlap: {', '.join(first)}")
    arrays["part_order"] = np.asarray(PART_ORDER)
    arrays["contact_part_mask"] = contact.astype(np.bool_)
    arrays["active_part_mask"] = active.astype(np.bool_)
    arrays["support_part_mask"] = support.astype(np.bool_)
    arrays["free_part_mask"] = free.astype(np.bool_)
    arrays["proto_start_idx"] = np.asarray([int(rec["start"]) for rec in records], dtype=np.int64)
    arrays["proto_end_idx"] = np.asarray([int(rec["end"]) for rec in records], dtype=np.int64)
    arrays["robot_asset_json"] = np.asarray(encode_robot_asset_json())
    np.savez(output, **arrays)
    return output


def _names_from_mask(mask: np.ndarray) -> str:
    names = [SHORT_NAME[part] for part, value in zip(PART_ORDER, mask) if value]
    return "|".join(names)


def _write_summary_csv(path: Path, contexts: list[MotionContext], by_motion: list[list[dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "motion",
                "start",
                "end",
                "duration",
                "contact_start",
                "contact_end",
                "touchdown",
                "contact_delta",
                "active_part",
                "support_part",
                "root_dx",
                "root_dy",
                "root_dz",
                "root_dyaw",
            ]
        )
        for ctx, records in zip(contexts, by_motion):
            for rec in records:
                rd = rec["root_delta"]
                writer.writerow(
                    [
                        ctx.path.name,
                        rec["start"],
                        rec["end"],
                        f"{(rec['end'] - rec['start']) / ctx.fps:.4f}",
                        _names_from_mask(rec["contact_start"]),
                        _names_from_mask(rec["contact_end"]),
                        _names_from_mask(rec["touchdown_part"]),
                        _names_from_mask(rec["contact_delta"]),
                        _names_from_mask(rec["active_part"]),
                        _names_from_mask(rec["support_part"]),
                        f"{rd[0]:.5f}",
                        f"{rd[1]:.5f}",
                        f"{rd[2]:.5f}",
                        f"{rd[3]:.5f}",
                    ]
                )


def _write_per_motion_summary_csvs(output_dir: Path, output_name: str, contexts: list[MotionContext], by_motion: list[list[dict]]) -> None:
    for ctx, records in zip(contexts, by_motion):
        _write_summary_csv(output_dir / f"{ctx.path.stem}_{output_name}_summary.csv", [ctx], [records])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = dataset_root / data_dir
    data_dir = data_dir.resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = dataset_root / output_dir
    args.output_dir = output_dir.resolve()
    if args.holosoma_motion_dir is None:
        args.holosoma_motion_dir = dataset_root / "data" / "holosoma_motions_50hz"
    if args.holosoma_output_dir is None:
        args.holosoma_output_dir = dataset_root / "data" / "holosoma_motions_masked_50hz"
    robot_urdf = (
        args.robot_urdf.expanduser().resolve()
        if args.robot_urdf
        else canonical_g1_urdf_path()
    )
    validate_canonical_g1_urdf(robot_urdf)
    paths = _motion_paths(data_dir, args.motions)
    enabled_parts = set(args.contact_parts)
    enabled_indices = [PART_ORDER.index(part) for part in args.contact_parts]

    contexts: list[MotionContext] = []
    by_motion: list[list[dict]] = []
    contacts_by_motion: list[np.ndarray] = []
    total = 0
    for path in paths:
        ctx = _load_context(path, dataset_root, robot_urdf)
        contact = _contact_masks_with_force_demo(
            ctx, enabled_parts, args.contact_force_demo_dir, args.contact_force_demo_file
        )
        debounced_contact = _debounced_contact(
            contact,
            args.merge_transition_window,
            args.hand_merge_transition_window,
            args.body_merge_transition_window,
        )
        anchors, touchdown_by_anchor = _anchor_maps(ctx, debounced_contact, enabled_indices, args)
        role_masks = _global_role_masks(
            ctx,
            contact,
            anchors,
            touchdown_by_anchor,
            args.stable_touchdown_speed_thresh,
            args.stable_touchdown_window,
            args.free_state_window,
            args.free_transition_active_window,
        )
        records = _segment_records(
            ctx,
            contact,
            anchors,
            touchdown_by_anchor,
            args.min_root_delta,
            role_masks,
        )
        contexts.append(ctx)
        by_motion.append(records)
        contacts_by_motion.append(contact)
        total += len(records)
        print(f"{path.name}: frames={len(ctx.qpos)} anchors={len(anchors)} protos={len(records)}")
        for part_i, part in enumerate(PART_ORDER):
            if part not in enabled_parts and part not in {"left_knee", "right_knee", "left_hip", "right_hip"}:
                continue
            intervals = _intervals(contact[:, part_i])
            if intervals:
                print(f"  {SHORT_NAME[part]} contact {intervals[:8]}")
        for rec in records[:12]:
            print(
                f"  {rec['start']:4d}->{rec['end']:4d} "
                f"active={_names_from_mask(rec['active_part']) or '-'} "
                f"support={_names_from_mask(rec['support_part']) or '-'} "
                f"C0={_names_from_mask(rec['contact_start']) or '-'} "
                f"C1={_names_from_mask(rec['contact_end']) or '-'}"
            )

    if args.dry_run:
        print(f"dry-run: extracted {total} protos, no files written")
        return

    if args.mask_only and not args.write_holosoma_masks:
        raise ValueError("--mask-only requires --write-holosoma-masks")

    holosoma_motion_dir = args.holosoma_motion_dir.expanduser().resolve()
    output_contexts = []
    output_contacts_by_motion = []
    output_by_motion = []
    output_role_masks_by_motion = []
    for ctx, _contact, _records in zip(contexts, contacts_by_motion, by_motion):
        _source, target_frames, target_fps = _holosoma_motion_info(
            ctx.path,
            holosoma_motion_dir,
            args.holosoma_motion_suffix,
        )
        target_ctx = _resample_context(ctx, target_frames, target_fps)
        target_contact = _contact_masks_with_force_demo(
            target_ctx, enabled_parts, args.contact_force_demo_dir, args.contact_force_demo_file
        )
        target_debounced_contact = _debounced_contact(
            target_contact,
            args.merge_transition_window,
            args.hand_merge_transition_window,
            args.body_merge_transition_window,
        )
        target_anchors, target_touchdown_by_anchor = _anchor_maps(
            target_ctx,
            target_debounced_contact,
            enabled_indices,
            args,
        )
        target_role_masks = _global_role_masks(
            target_ctx,
            target_contact,
            target_anchors,
            target_touchdown_by_anchor,
            args.stable_touchdown_speed_thresh,
            args.stable_touchdown_window,
            args.free_state_window,
            args.free_transition_active_window,
        )
        target_records = _segment_records(
            target_ctx,
            target_contact,
            target_anchors,
            target_touchdown_by_anchor,
            args.min_root_delta,
            target_role_masks,
        )
        output_contexts.append(target_ctx)
        output_contacts_by_motion.append(target_contact)
        output_by_motion.append(target_records)
        output_role_masks_by_motion.append(target_role_masks)
        print(
            f"[target-fps] {ctx.path.name}: {len(ctx.qpos)} frames @ {ctx.fps:g} fps -> "
            f"{target_frames} frames @ {target_fps:g} fps, anchors={len(target_anchors)} protos={len(target_records)}"
        )
    total = sum(len(records) for records in output_by_motion)

    if args.write_holosoma_masks:
        holosoma_output_dir = args.holosoma_output_dir.expanduser().resolve()
        for ctx, contact, records, role_masks in zip(
            output_contexts,
            output_contacts_by_motion,
            output_by_motion,
            output_role_masks_by_motion,
        ):
            output = _write_holosoma_motion_with_masks(
                ctx.path,
                holosoma_motion_dir,
                holosoma_output_dir,
                args.holosoma_motion_suffix,
                ctx.fps,
                contact,
                records,
                role_masks,
                args.overwrite,
            )
            print(f"wrote masked Holosoma motion to {output}")

    if args.mask_only:
        print(f"mask-only: extracted frame masks for {len(contexts)} motion(s), no proto files written")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    proto = _pack_records(output_contexts, output_by_motion, args.max_segment_frames)
    pt_path = args.output_dir / f"{args.output_name}.pt"
    csv_path = args.output_dir / f"{args.output_name}_summary.csv"
    torch.save(proto, pt_path)
    _write_summary_csv(csv_path, output_contexts, output_by_motion)
    _write_per_motion_summary_csvs(args.output_dir, args.output_name, output_contexts, output_by_motion)
    print(f"wrote {total} protos to {pt_path}")
    print(f"wrote summary to {csv_path}")


if __name__ == "__main__":
    main()
