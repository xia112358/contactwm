#!/usr/bin/env python3
"""Build one canonical source motion from successful parallel policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from somaforge_core.newton_direct_fk import (
    canonicalize_motion_with_direct_newton_fk,
)
from somaforge_core.rollout_source import (
    complete_rollout_env_ids,
    merge_part_force,
    merge_qpos,
    merge_raw_contacts,
    source_metadata_json,
)
from somaforge_core.contact_schema import (
    CONTACT_FORCE_PART_BODY_NAMES,
    CONTACT_FORCE_PART_ORDER,
    encode_contact_force_provenance,
    newton_contact_provenance,
)


RAW_KEYS = (
    "raw_contact_shape0",
    "raw_contact_shape1",
    "raw_contact_body0",
    "raw_contact_body1",
    "raw_contact_point0_w",
    "raw_contact_point1_w",
    "raw_contact_normal_w",
    "raw_contact_force_w",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--reference-motion", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = _parser().parse_args()
    recording_path = args.recording.expanduser().resolve()
    reference_path = args.reference_motion.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    for path in (recording_path, reference_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)

    with np.load(reference_path, allow_pickle=False) as reference:
        frame_count = int(reference["joint_pos"].shape[0])
        robot_asset_json = np.asarray(reference["robot_asset_json"])

    with np.load(recording_path, allow_pickle=False) as recording:
        metadata = json.loads(str(recording["_metadata_json"].item()))
        env_ids = complete_rollout_env_ids(
            recording["motion_time_step"],
            frame_count=frame_count,
        )
        if env_ids.size < 3:
            raise ValueError(
                f"multi-rollout consensus requires at least 3 complete runs, got {env_ids.tolist()}"
            )
        selection = env_ids.astype(np.int64)
        qpos = merge_qpos(
            root_pos=np.asarray(recording["root_pos"])[:frame_count, selection],
            root_quat_xyzw=np.asarray(recording["root_quat_xyzw"])[
                :frame_count, selection
            ],
            dof_pos=np.asarray(recording["dof_pos"])[:frame_count, selection],
        )
        force, raw_mask, stable_mask = merge_part_force(
            np.asarray(recording["contact_sensor_forces"])[
                :frame_count, selection
            ],
            metadata["contact_sensor_body_names"],
        )

        counts = np.asarray(recording["raw_contact_count"])[:frame_count].copy()
        active_width = max(int(counts.max(initial=0)), 1)
        compact_raw: dict[str, np.ndarray] = {"raw_contact_count": counts}
        for name in RAW_KEYS:
            compact_raw[name] = np.asarray(recording[name])[
                :frame_count, :active_width
            ].copy()
        raw = merge_raw_contacts(
            compact_raw,
            metadata,
            frame_count=frame_count,
            env_ids=env_ids,
        )

    solver_config = metadata.get("newton_solver_config")
    if not isinstance(solver_config, dict):
        raise ValueError("recording metadata has no Newton solver config")
    contact_provenance = newton_contact_provenance(
        solver_config=solver_config,
        source_recording=str(recording_path),
        threshold_n=10.0,
        mask_off_threshold_n=5.0,
        mask_close_gap_frames=2,
    )
    contact_provenance.update(
        {
            "source_kind": "multi_rollout_measured_consensus",
            "source_checkpoint": str(checkpoint_path),
            "successful_rollout_count": int(env_ids.size),
            "time_alignment": "motion_time_step_exact",
            "force_aggregation": "world_vector_coordinatewise_median",
            "raw_contact_aggregation": (
                "stable_label_external_contact_group_median"
            ),
            "raw_contact_persisted": True,
            "runtime_env_identity_persisted": False,
            "dynamic_replay_used": False,
        }
    )
    fps = float(metadata.get("fps", 50.0))
    seed = {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_pos": qpos,
        "joint_names": np.asarray(metadata["dof_names"]),
        "robot_asset_json": robot_asset_json,
        "contact_force_part_w": force,
        "contact_force_part_mask": stable_mask,
        "contact_force_part_mask_raw": raw_mask,
        "contact_force_part_order": np.asarray(CONTACT_FORCE_PART_ORDER),
        "contact_force_part_body_names_json": np.asarray(
            json.dumps(
                {
                    key: list(value)
                    for key, value in CONTACT_FORCE_PART_BODY_NAMES.items()
                },
                sort_keys=True,
            )
        ),
        "contact_force_demo_threshold": np.asarray(10.0, dtype=np.float32),
        "contact_force_demo_off_threshold": np.asarray(5.0, dtype=np.float32),
        "contact_force_demo_close_gap_frames": np.asarray(2, dtype=np.int32),
        "contact_force_sample_semantics": np.asarray(
            "motion_timestep_aligned_original_policy_rollout_vector_median"
        ),
        "contact_force_provenance_json": np.asarray(
            encode_contact_force_provenance(contact_provenance)
        ),
        "rollout_source_provenance_json": source_metadata_json(
            source_recording=str(recording_path),
            checkpoint=str(checkpoint_path),
            env_count=int(env_ids.size),
        ),
        "rollout_source_recording": np.asarray(str(recording_path)),
        "rollout_source_checkpoint": np.asarray(str(checkpoint_path)),
        "rollout_source_count": np.asarray(env_ids.size, dtype=np.int32),
        **raw,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_path = work_dir / "source_seed.npz"
    np.savez_compressed(seed_path, **seed)
    result = canonicalize_motion_with_direct_newton_fk(
        seed_path,
        output_path,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "frames": result.frame_count,
                "body_count": len(result.body_names),
                "joint_count": len(result.joint_names),
                "successful_rollout_count": int(env_ids.size),
                "raw_contact_max_count": int(raw["raw_contact_max_count"]),
                "force_norm_max_n": float(
                    np.linalg.norm(force, axis=-1).max(initial=0.0)
                ),
                "dynamic_replay_used": False,
                "runtime_env_identity_persisted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
