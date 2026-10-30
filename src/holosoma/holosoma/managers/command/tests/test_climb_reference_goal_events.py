from __future__ import annotations

import numpy as np
from holosoma.managers.command.terms.climb_reference_goal import load_trimmed_contact_events

MANIFEST = "runtime/current/motions/wbt/climb00_continuous_coverage_trimmed/training_manifest_207_trimmed.json"
EXPECTED_FRAMES = (
    29,
    103,
    126,
    148,
    158,
    170,
    197,
    221,
    227,
    239,
    260,
    280,
    312,
    320,
    326,
    407,
    640,
    654,
    666,
    681,
    695,
    725,
    761,
    786,
)


def test_trimmed_manifest_has_shared_24_event_contract() -> None:
    table = load_trimmed_contact_events(MANIFEST)
    assert table["event_frames"].shape == (207, 24)
    assert table["touchdown_masks"].shape == (207, 24, 6)
    assert table["end_contact_masks"].shape == (207, 24, 6)
    assert np.all(table["frame_counts"] == 805)
    assert tuple(table["event_frames"][0].tolist()) == EXPECTED_FRAMES
    assert np.all(table["event_frames"] == table["event_frames"][0])

