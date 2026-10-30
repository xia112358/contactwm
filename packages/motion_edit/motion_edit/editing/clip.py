from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_edit.adapters.holosoma_npz import load_motion_npz, subset_arrays


def clip_motion(input_path: str | Path, output_path: str | Path, start: int, end: int) -> Path:
    arrays = load_motion_npz(input_path)
    clipped = subset_arrays(arrays, start, end)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **clipped)
    return out
