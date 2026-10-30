import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parents[1] / "scripts" / "build_rollout_ref_finetune_dataset.py"
_SPEC = importlib.util.spec_from_file_location("build_rollout_ref_finetune_dataset", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_complete_segment_skips_duplicate_bootstrap_frame() -> None:
    steps = np.asarray([0, 0, 1, 2, 3, 0])
    terminated = np.zeros_like(steps, dtype=bool)
    np.testing.assert_array_equal(_MODULE._complete_segment(steps, terminated, 4), [1, 2, 3, 4])


def test_complete_segment_rejects_terminated_episode() -> None:
    steps = np.arange(4)
    terminated = np.asarray([False, False, True, False])
    assert _MODULE._complete_segment(steps, terminated, 4) is None
