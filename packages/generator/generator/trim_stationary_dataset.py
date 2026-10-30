"""Remove the validated climb00 stationary plateau from an aligned manifest.

The 207-way climb00 augmentation preserves the source 1005-frame timeline.
Frames [481, 681) are a long, unchanged-support dwell rather than a contact
transition.  This tool applies the previously validated 16-frame quintic
crossfade to every motion and to every shared contact source, then writes an
independent manifest whose plans point at the trimmed contact timelines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk

LEFT_START = 481
RIGHT_START = 681
BLEND_FRAMES = 16
RIGHT_KEEP = RIGHT_START + BLEND_FRAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value, axis=-1, keepdims=True).clip(min=1.0e-8)


def _slerp(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    first = _normalize_quaternion(np.asarray(left, dtype=np.float64))
    second = _normalize_quaternion(np.asarray(right, dtype=np.float64))
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.abs(dot).clip(-1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    linear = sine < 1.0e-7
    weight_shape = (len(alpha),) + (1,) * (first.ndim - 1)
    weight = alpha.reshape(weight_shape)
    first_weight = np.sin((1.0 - weight) * angle) / np.where(linear, 1.0, sine)
    second_weight = np.sin(weight * angle) / np.where(linear, 1.0, sine)
    output = first_weight * first + second_weight * second
    output = np.where(linear, (1.0 - weight) * first + weight * second, output)
    return _normalize_quaternion(output)


def _quintic_alpha() -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, BLEND_FRAMES, dtype=np.float64)
    return alpha**3 * (alpha * (alpha * 6.0 - 15.0) + 10.0)


def trim_frame_array(key: str, value: np.ndarray, frame_count: int) -> np.ndarray:
    """Trim one frame-aligned array while preserving its representation."""

    if value.ndim == 0 or value.shape[0] != frame_count:
        return value
    left = value[LEFT_START : LEFT_START + BLEND_FRAMES]
    right = value[RIGHT_START:RIGHT_KEEP]
    alpha = _quintic_alpha()
    if key == "joint_pos":
        weight = alpha.reshape((BLEND_FRAMES,) + (1,) * (value.ndim - 1))
        blend = (1.0 - weight) * left + weight * right
        blend[..., 3:7] = _slerp(left[..., 3:7], right[..., 3:7], alpha)
    elif key == "body_quat_w":
        blend = _slerp(left, right, alpha)
    elif np.issubdtype(value.dtype, np.floating):
        weight = alpha.reshape((BLEND_FRAMES,) + (1,) * (value.ndim - 1))
        blend = (1.0 - weight) * left + weight * right
    else:
        choose_right = alpha.reshape((BLEND_FRAMES,) + (1,) * (value.ndim - 1)) >= 0.5
        blend = np.where(choose_right, right, left)
    return np.concatenate((value[:LEFT_START], blend.astype(value.dtype), value[RIGHT_KEEP:]), axis=0)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _write_trimmed_source(source: Path, output: Path) -> None:
    values = _load(source)
    frame_count = int(values["joint_pos"].shape[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    trimmed = {key: trim_frame_array(key, value, frame_count) for key, value in values.items()}
    np.savez_compressed(output, **trimmed)


def _write_trimmed_motion(source: Path, output: Path, *, device: str) -> None:
    values = _load(source)
    frame_count = int(values["joint_pos"].shape[0])
    qpos = trim_frame_array("joint_pos", np.asarray(values["joint_pos"]), frame_count)
    provenance = {
        "schema": "somaforge_stationary_trim_v1",
        "source_motion": str(source.resolve()),
        "left_window": [LEFT_START, LEFT_START + BLEND_FRAMES],
        "right_window": [RIGHT_START, RIGHT_KEEP],
        "removed_frames": RIGHT_START - LEFT_START,
        "blend": "quintic_crossfade_root_slerp",
        "source_frame_count": frame_count,
        "output_frame_count": len(qpos),
    }
    static = {key: value for key, value in values.items() if value.ndim == 0 or value.shape[0] != frame_count}
    static["joint_pos"] = qpos
    static["stationary_trim_provenance_json"] = np.asarray(json.dumps(provenance, sort_keys=True))
    pre_fk = output.with_suffix(".pre_fk.npz")
    pre_fk.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pre_fk, **static)
    canonicalize_motion_with_direct_newton_fk(pre_fk, output, device=device, overwrite=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["motion_files"]
    source_paths = sorted({Path(entry["edit_plan_file"]) for entry in entries})
    plans = {path: json.loads(path.read_text(encoding="utf-8")) for path in source_paths}
    contact_sources = sorted({Path(plan["source_motion_path"]) for plan in plans.values()})
    trimmed_contact: dict[Path, Path] = {}
    for index, source in enumerate(contact_sources):
        output = args.output / "contact_sources" / f"source_{index:02d}_{source.parent.name}_{source.name}"
        _write_trimmed_source(source, output)
        trimmed_contact[source] = output.resolve()

    output_entries = []
    for index, entry in enumerate(entries):
        source_motion = Path(entry["motion_file"])
        output_motion = args.output / "motions" / source_motion.name
        _write_trimmed_motion(source_motion, output_motion, device=args.device)

        source_plan = Path(entry["edit_plan_file"])
        plan = dict(plans[source_plan])
        original_contact = Path(plan["source_motion_path"])
        plan["source_motion_path"] = str(trimmed_contact[original_contact])
        plan["output_motion_path"] = str(output_motion.resolve())
        metadata = dict(plan.get("metadata", {}))
        metadata["stationary_trim"] = {
            "left_start": LEFT_START,
            "right_start": RIGHT_START,
            "blend_frames": BLEND_FRAMES,
            "removed_frames": RIGHT_START - LEFT_START,
            "frame_indexed_edit_annotations": (
                "retained as source provenance; training reads only geometry and contact source"
            ),
        }
        plan["metadata"] = metadata
        output_plan = args.output / "plans" / f"{source_motion.stem}.json"
        output_plan.parent.mkdir(parents=True, exist_ok=True)
        output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        output_entry = dict(entry)
        output_entry.update(
            motion_file=str(output_motion.resolve()),
            motion_sha256=_sha256(output_motion),
            edit_plan_file=str(output_plan.resolve()),
            edit_plan_sha256=_sha256(output_plan),
            stationary_trim_source_motion=str(source_motion.resolve()),
        )
        output_entries.append(output_entry)
        print(f"trimmed {index + 1}/{len(entries)} {source_motion.name}", flush=True)

    output_manifest = dict(manifest)
    output_manifest["description"] = str(manifest.get("description", "")) + " Stationary plateau removed."
    output_manifest["motion_files"] = output_entries
    output_manifest["stationary_trim"] = {
        "schema": "somaforge_stationary_trim_v1",
        "source_manifest": str(args.manifest.resolve()),
        "left_start": LEFT_START,
        "right_start": RIGHT_START,
        "blend_frames": BLEND_FRAMES,
        "removed_frames": RIGHT_START - LEFT_START,
        "source_frames": 1005,
        "output_frames": 805,
    }
    manifest_path = args.output / "training_manifest_207_trimmed.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "motions": len(output_entries)}, indent=2))


if __name__ == "__main__":
    main()
