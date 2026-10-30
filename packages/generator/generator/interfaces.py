"""Runtime interface for next-keyframe prediction."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from somaforge_core.motion_contracts import SparseKeyframe


class KeyframeSelector(Protocol):
    """Predict exactly one complete next keyframe from the realized boundary."""

    def predict(self, current: SparseKeyframe, terrain_scan: NDArray[np.float32]) -> SparseKeyframe: ...
