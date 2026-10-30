"""Fuse timestamped, deskewed point-cloud recordings without a simulator.

Input NPZ: points_sensor [T,P,3], valid [T,P], position_w [T,3],
quaternion_xyzw [T,4], timestamps [T]. Optional episode_start [T] resets history.
Policy samples also require root_position_w [T,3], root_quaternion_xyzw [T,4].
Output: accumulated surface point clouds/maps per episode and policy scans.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from holosoma.acp.mapping import MappingPipeline, load_mapping_network


@dataclass(frozen=True)
class Options:
    recording: Path
    output_dir: Path
    checkpoint: Path | None = None
    device: str = "cpu"


def reconstruct(options):
    with np.load(options.recording, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in
                  ("points_sensor", "valid", "position_w", "quaternion_xyzw", "timestamps")}
        starts = data["episode_start"].astype(bool) if "episode_start" in data else np.zeros(len(arrays["timestamps"]), bool)
        # Policy scans are centered on the robot root, not the sensor. Require
        # an explicit base pose, avoiding silent misuse of sensor height/yaw.
        for key in ("root_position_w", "root_quaternion_xyzw"):
            arrays[key] = data[key].copy()
    length = len(arrays["timestamps"])
    if not length or any(len(value) != length for value in arrays.values()) or starts.shape != (length,):
        raise ValueError("Recording arrays must have matching nonempty time axes")
    options.output_dir.mkdir(parents=True, exist_ok=False)
    network = load_mapping_network(options.checkpoint, options.device) if options.checkpoint else None
    pipeline = MappingPipeline(1, options.device, network=network)
    scans, masks, heights, height_masks = [], [], [], []
    episode = 0
    for index in range(length):
        if starts[index] and index:
            pipeline.map.export(options.output_dir/f"map_{episode:04d}.npz")
            episode += 1
            pipeline.map.reset([0])
        frame = {key: torch.as_tensor(value[index:index+1], device=options.device)
                 for key, value in arrays.items()}
        time = frame["timestamps"].float()
        pipeline.integrate(frame["points_sensor"].float(), frame["valid"].bool(),
                           frame["position_w"].float(), frame["quaternion_xyzw"].float(), time)
        args = (frame["root_position_w"].float(), frame["root_quaternion_xyzw"].float(), time)
        scan, valid = pipeline.scan(*args)
        tracker, tracker_valid = pipeline.scan(*args, rows=9, columns=7)
        scans.append(scan[0].cpu().numpy())
        masks.append(valid[0].cpu().numpy())
        heights.append(tracker[0, :, 2].cpu().numpy().reshape(9, 7))
        height_masks.append(tracker_valid[0].cpu().numpy().reshape(9, 7))
    pipeline.map.export(options.output_dir/f"map_{episode:04d}.npz")
    np.savez_compressed(options.output_dir/"scans.npz", planner_points=np.stack(scans), planner_valid=np.stack(masks),
                        tracker_heights=np.stack(heights), tracker_valid=np.stack(height_masks),
                        timestamps=arrays["timestamps"], episode_start=starts)
    report = {"frames": length, "episodes": episode+1, "mode": "fused_neural" if network else "fused_raw",
              "planner_coverage": float(np.mean(masks)), "tracker_coverage": float(np.mean(height_masks)),
              "point_cloud_representation": "accumulated_2.5d_surface_cell_centers"}
    (options.output_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    print(reconstruct(tyro.cli(Options)))
