#!/usr/bin/env python3
"""Build a mixed original/rollout WBT fine-tuning manifest from lightweight eval recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk
from somaforge_core import sha256_file


ROLLOUT_JOINT_VELOCITY_FILTER_WINDOW = 11
ROLLOUT_JOINT_VELOCITY_FILTER_POLYORDER = 3


def _complete_segment(steps: np.ndarray, terminated: np.ndarray, frame_count: int) -> np.ndarray | None:
    """Return indices for one uninterrupted 0..frame_count-1 episode."""
    target = np.arange(frame_count, dtype=steps.dtype)
    for start in np.flatnonzero(steps == 0):
        stop = int(start) + frame_count
        if stop > steps.shape[0]:
            continue
        if np.array_equal(steps[start:stop], target) and not np.any(terminated[start:stop]):
            return np.arange(start, stop, dtype=np.int64)
    return None


def _metadata(recording: np.lib.npyio.NpzFile) -> dict:
    raw = recording["_metadata_json"].item()
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _resolve(path: str, manifest: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (manifest.parent / value).resolve()


def _write_source(
    recording: np.lib.npyio.NpzFile,
    metadata: dict,
    indices: np.ndarray,
    env_id: int,
    original: np.lib.npyio.NpzFile,
    output: Path,
) -> np.ndarray:
    root_pos = np.asarray(recording["root_pos"])[indices, env_id]
    root_xyzw = np.asarray(recording["root_quat_xyzw"])[indices, env_id]
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    dof_pos = np.asarray(recording["dof_pos"])[indices, env_id]
    joint_pos = np.concatenate([root_pos, root_wxyz, dof_pos], axis=1).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=np.asarray(float(metadata["fps"])),
        joint_names=np.asarray(metadata["dof_names"]),
        joint_pos=joint_pos,
        robot_asset_json=original["robot_asset_json"],
        rollout_recording=np.asarray(str(Path(recording.zip.filename).resolve())),
        rollout_env_id=np.asarray(env_id, dtype=np.int32),
        sim_joint_vel=np.concatenate(
            [
                np.asarray(recording["root_lin_vel"])[indices, env_id],
                np.asarray(recording["root_ang_vel"])[indices, env_id],
                np.asarray(recording["dof_vel"])[indices, env_id],
            ],
            axis=1,
        ).astype(np.float32),
    )
    return joint_pos[:, 7:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", required=True, type=Path)
    parser.add_argument("--original-motion", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument("--original-weight", type=float, default=0.5)
    parser.add_argument("--min-dof-rms", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not 0.0 < args.original_weight < 1.0:
        raise ValueError("--original-weight must be between zero and one")
    if args.output_manifest.exists():
        raise FileExistsError(args.output_manifest)

    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    if len(base["motion_files"]) != 1 or len(base["terrains"]) != 1:
        raise ValueError("This builder requires a single-motion, single-terrain base manifest")
    original_path = args.original_motion.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "sources"
    motion_dir = output_dir / "motions"
    source_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    accepted_dofs: list[np.ndarray] = []
    rollout_entries: list[dict] = []
    with np.load(original_path, allow_pickle=False) as original:
        frame_count = int(original["joint_pos"].shape[0])
        for recording_path in args.recording:
            if len(rollout_entries) >= args.max_rollouts:
                break
            with np.load(recording_path.resolve(), allow_pickle=False) as recording:
                metadata = _metadata(recording)
                steps = np.asarray(recording["motion_time_step"])
                terminated = np.asarray(recording["terminated"], dtype=bool)
                if steps.ndim != 2:
                    raise ValueError(f"Expected all-env recording, got motion_time_step {steps.shape}")
                for env_id in range(steps.shape[1]):
                    if len(rollout_entries) >= args.max_rollouts:
                        break
                    indices = _complete_segment(steps[:, env_id], terminated[:, env_id], frame_count)
                    if indices is None:
                        continue
                    index = len(rollout_entries)
                    source_path = source_dir / f"{recording_path.stem}_env_{env_id:04d}_source.npz"
                    canonical_path = motion_dir / f"climb_00_rollout_{index:04d}.npz"
                    dofs = _write_source(recording, metadata, indices, env_id, original, source_path)
                    if accepted_dofs:
                        rms = min(float(np.sqrt(np.mean((dofs - prior) ** 2))) for prior in accepted_dofs)
                        if rms < args.min_dof_rms:
                            continue
                    canonicalize_motion_with_direct_newton_fk(
                        source_path,
                        canonical_path,
                        device=args.device,
                        joint_velocity_filter_window=ROLLOUT_JOINT_VELOCITY_FILTER_WINDOW,
                        joint_velocity_filter_polyorder=ROLLOUT_JOINT_VELOCITY_FILTER_POLYORDER,
                    )
                    accepted_dofs.append(dofs)
                    rollout_entries.append(
                        {
                            "motion_id": index + 1,
                            "motion_file": str(canonical_path),
                            "terrain_id": 0,
                            "source_file": str(source_path),
                            "source_sha256": sha256_file(source_path),
                            "motion_sha256": sha256_file(canonical_path),
                            "kinematics_schema": "somaforge_canonical_motion_v1",
                            "kinematics_backend": "newton_direct_fk",
                            "velocity_derivation": "pose_local_polynomial_derivative",
                            "joint_velocity_filter_window": ROLLOUT_JOINT_VELOCITY_FILTER_WINDOW,
                            "joint_velocity_filter_polyorder": ROLLOUT_JOINT_VELOCITY_FILTER_POLYORDER,
                        }
                    )

    if not rollout_entries:
        raise RuntimeError("No complete, distinct rollouts were found")
    original_canonical_path = motion_dir / "climb_00_original_canonical.npz"
    canonicalize_motion_with_direct_newton_fk(
        original_path,
        original_canonical_path,
        device=args.device,
    )
    rollout_weight = (1.0 - args.original_weight) / len(rollout_entries)
    for entry in rollout_entries:
        entry["weight"] = rollout_weight
    original_entry = {
        "motion_id": 0,
        "motion_file": str(original_canonical_path),
        "terrain_id": 0,
        "weight": args.original_weight,
        "source_file": str(original_path),
        "source_sha256": sha256_file(original_path),
        "motion_sha256": sha256_file(original_canonical_path),
        "kinematics_schema": "somaforge_canonical_motion_v1",
        "kinematics_backend": "newton_direct_fk",
        "velocity_derivation": "pose_finite_difference",
    }
    terrain = dict(base["terrains"][0])
    terrain["terrain_file"] = str(_resolve(terrain["terrain_file"], base_path))
    manifest = {
        **{key: value for key, value in base.items() if key not in ("motion_files", "terrains", "kinematics_backend")},
        "description": f"50/50 original and {len(rollout_entries)} direct-Newton rollout references.",
        "kinematics_backend": "newton_direct_fk",
        "terrains": [terrain],
        "motion_files": [original_entry, *rollout_entries],
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"accepted_rollouts": len(rollout_entries), "manifest": str(args.output_manifest)}, indent=2))


if __name__ == "__main__":
    main()
