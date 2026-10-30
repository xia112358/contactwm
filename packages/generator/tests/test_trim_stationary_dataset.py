import numpy as np
from generator.trim_stationary_dataset import (
    BLEND_FRAMES,
    LEFT_START,
    RIGHT_KEEP,
    RIGHT_START,
    trim_frame_array,
)


def test_trim_frame_array_removes_plateau_and_preserves_crossfade_endpoints() -> None:
    values = np.arange(1005, dtype=np.float32)[:, None]
    output = trim_frame_array("value", values, len(values))

    assert output.shape == (805, 1)
    np.testing.assert_array_equal(output[:LEFT_START], values[:LEFT_START])
    np.testing.assert_array_equal(output[LEFT_START], values[LEFT_START])
    np.testing.assert_array_equal(output[LEFT_START + BLEND_FRAMES - 1], values[RIGHT_KEEP - 1])
    np.testing.assert_array_equal(output[LEFT_START + BLEND_FRAMES], values[RIGHT_KEEP])


def test_joint_trim_keeps_unit_quaternions() -> None:
    qpos = np.zeros((1005, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[RIGHT_START:RIGHT_KEEP, 3] = -1.0

    output = trim_frame_array("joint_pos", qpos, len(qpos))

    np.testing.assert_allclose(np.linalg.norm(output[:, 3:7], axis=-1), 1.0, atol=1.0e-6)
