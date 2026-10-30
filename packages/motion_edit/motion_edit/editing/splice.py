from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_edit.adapters.holosoma_npz import load_motion_npz

FRAME_KEYS = (
    "joint_pos",
    "qpos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_vel",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "contact_part_mask",
    "active_part_mask",
    "contact_keep_intent_mask",
    "support_part_mask",
    "free_part_mask",
)


def splice_motions(inputs: list[str | Path], output_path: str | Path) -> Path:
    if not inputs:
        raise ValueError("splice_motions requires at least one input")
    loaded = [load_motion_npz(path) for path in inputs]
    from somaforge_core.support_evidence import reference_only_support
    for arrays in loaded:
        reference_only_support(arrays, reason='splice input roles are contact-keep intent, not output execution evidence')
    out = {}
    frame_counts = []
    for arrays in loaded:
        for key in ("joint_pos", "qpos", "body_pos_w"):
            if key in arrays:
                frame_counts.append(arrays[key].shape[0])
                break
    for key, first_value in loaded[0].items():
        if key.startswith("_"):
            continue
        if key in FRAME_KEYS and all(key in arrays for arrays in loaded):
            out[key] = np.concatenate([arrays[key] for arrays in loaded], axis=0)
        else:
            out[key] = np.asarray(first_value).copy()
    out["splice_source_paths"] = np.asarray([str(Path(path).expanduser().resolve()) for path in inputs])
    out["splice_source_lengths"] = np.asarray(frame_counts, dtype=np.int64)
    reference_only_support(out, reason='spliced trajectory has no observed execution across its boundaries')
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return path
