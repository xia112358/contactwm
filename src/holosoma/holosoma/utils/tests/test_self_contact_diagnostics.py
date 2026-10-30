import json

import numpy as np

from holosoma.utils.self_contact_diagnostics import SelfContactDiagnostics


def test_self_contact_diagnostics_keeps_only_same_robot_pairs(tmp_path) -> None:
    output = tmp_path / "contacts.json"
    diagnostics = SelfContactDiagnostics(output)
    diagnostics.update(
        {
            "count": np.asarray(3, dtype=np.int32),
            "body0": np.asarray([0, 0, 0], dtype=np.int32),
            "body1": np.asarray([1, 2, 3], dtype=np.int32),
            "force_w": np.asarray([[3.0, 4.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32),
        },
        [
            "/World/envs/env_0/Robot/pelvis",
            "/World/envs/env_0/Robot/left_hand",
            "/World/envs/env_1/Robot/right_hand",
            "/World/ground",
        ],
    )
    diagnostics.write()

    payload = json.loads(output.read_text())
    assert payload["sample_count"] == 1
    assert payload["raw_contact_count"] == 3
    assert payload["robot_external_contact_count"] == 2
    assert payload["unknown_body_contact_count"] == 0
    assert payload["frames_with_self_contact"] == 1
    assert payload["max_contacts_per_sample"] == 1
    assert payload["pairs"] == [
        {
            "body_a": "left_hand",
            "body_b": "pelvis",
            "active_samples": 1,
            "contact_count": 1,
            "mean_force_n": 5.0,
            "max_force_n": 5.0,
        }
    ]


def test_self_contact_diagnostics_ignores_same_body_pairs(tmp_path) -> None:
    output = tmp_path / "contacts.json"
    diagnostics = SelfContactDiagnostics(output)
    diagnostics.update(
        {
            "count": np.asarray(2, dtype=np.int32),
            "body0": np.asarray([0, 0], dtype=np.int32),
            "body1": np.asarray([0, 0], dtype=np.int32),
            "force_w": np.zeros((2, 3), dtype=np.float32),
        },
        [
            "/World/envs/env_0/Robot/pelvis",
            "/World/envs/env_0/Robot/left_hand",
        ],
    )
    diagnostics.write()

    payload = json.loads(output.read_text())
    assert payload["raw_contact_count"] == 2
    assert payload["frames_with_self_contact"] == 0
    assert payload["max_contacts_per_sample"] == 0
    assert payload["pairs"] == []
