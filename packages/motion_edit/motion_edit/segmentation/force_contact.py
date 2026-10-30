from __future__ import annotations

from pathlib import Path

from motion_edit.force_proto import segments_from_masked_motion
from motion_edit.layers import write_layer
from motion_edit.paths import layer_dir


def import_force_proto_dir(motion_dir: str | Path, *, layer_name: str = "force_contact", pattern: str = "*.npz") -> int:
    out_dir = layer_dir("candidate", layer_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for motion_path in sorted(Path(motion_dir).expanduser().resolve().glob(pattern)):
        segments = segments_from_masked_motion(motion_path, source="force_contact", status="candidate")
        write_layer(out_dir / f"{motion_path.stem}.jsonl", segments)
        total += len(segments)
    return total
