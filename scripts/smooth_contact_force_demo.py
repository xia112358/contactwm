#!/usr/bin/env python3
"""Smooth contact-force vectors within existing contact segments only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[0::2], changes[1::2])]


def _smooth_segment(values: np.ndarray, kernel: np.ndarray, passes: int) -> np.ndarray:
    if values.shape[0] < 3:
        return values.copy()
    out = values.astype(np.float32, copy=True)
    pad = kernel.shape[0] // 2
    for _ in range(passes):
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        out = np.stack(
            [
                np.convolve(padded[:, axis], kernel, mode="valid")
                for axis in range(out.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
    return out


def smooth_contact_force(
    force: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    passes: int,
    preserve_magnitude: bool,
) -> np.ndarray:
    if force.ndim != 3 or force.shape[-1] != 3:
        raise ValueError(f"contact_force_part_w must be [T,P,3], got {force.shape}")
    if mask.shape != force.shape[:2]:
        raise ValueError(f"contact_force_part_mask must be {force.shape[:2]}, got {mask.shape}")

    smoothed = np.zeros_like(force, dtype=np.float32)
    for part_idx in range(force.shape[1]):
        part_mask = mask[:, part_idx].astype(bool)
        for start, stop in _segments(part_mask):
            segment = force[start:stop, part_idx]
            filtered = _smooth_segment(segment, kernel, passes)
            if preserve_magnitude:
                original_mag = np.linalg.norm(segment, axis=-1, keepdims=True)
                filtered_mag = np.linalg.norm(filtered, axis=-1, keepdims=True)
                filtered = np.where(filtered_mag > 1e-6, filtered / filtered_mag * original_mag, filtered)
            smoothed[start:stop, part_idx] = filtered

    smoothed[~mask.astype(bool)] = 0.0
    return smoothed.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--kernel", choices=("3", "5"), default="3")
    parser.add_argument("--preserve-magnitude", action="store_true")
    args = parser.parse_args()

    data = _load_npz(args.input)
    force = np.asarray(data["contact_force_part_w"], dtype=np.float32)
    mask = np.asarray(data["contact_force_part_mask"], dtype=bool)
    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
    if args.kernel == "5":
        kernel = np.asarray([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)

    out = dict(data)
    out["contact_force_part_w_raw"] = force
    out["contact_force_part_w"] = smooth_contact_force(
        force,
        mask,
        kernel=kernel,
        passes=max(1, int(args.passes)),
        preserve_magnitude=bool(args.preserve_magnitude),
    )
    out["contact_force_smoothing"] = np.asarray(
        f"segment_only_kernel_{args.kernel}_passes_{args.passes}_preserve_magnitude_{args.preserve_magnitude}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    raw_noncontact = float(np.linalg.norm(force[~mask], axis=-1).max(initial=0.0))
    new_noncontact = float(np.linalg.norm(out["contact_force_part_w"][~mask], axis=-1).max(initial=0.0))
    print(
        f"Wrote {args.output} force_shape={force.shape} "
        f"raw_noncontact_max={raw_noncontact:.6g} new_noncontact_max={new_noncontact:.6g}"
    )


if __name__ == "__main__":
    main()
