from __future__ import annotations

import numpy as np
import pytest

from somaforge_core.newton_direct_fk import _local_polynomial_derivative


def test_local_polynomial_derivative_is_exact_for_cubic_at_boundaries() -> None:
    fps = 50.0
    time = np.arange(31, dtype=np.float64) / fps
    values = np.stack((time**3 - 2.0 * time**2 + time, 0.5 * time**2), axis=1)
    expected = np.stack((3.0 * time**2 - 4.0 * time + 1.0, time), axis=1)

    actual = _local_polynomial_derivative(values, fps, window=11, polyorder=3)

    np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=0.0)


def test_local_polynomial_derivative_rejects_invalid_window() -> None:
    values = np.zeros((20, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="odd"):
        _local_polynomial_derivative(values, 50.0, window=10, polyorder=3)
    with pytest.raises(ValueError, match="fewer"):
        _local_polynomial_derivative(values[:5], 50.0, window=11, polyorder=3)


def test_local_polynomial_derivative_reduces_high_frequency_pose_noise() -> None:
    fps = 50.0
    time = np.arange(201, dtype=np.float64) / fps
    clean = np.sin(2.0 * np.pi * time)
    noisy = clean + 0.02 * np.sin(2.0 * np.pi * 20.0 * time)
    expected = 2.0 * np.pi * np.cos(2.0 * np.pi * time)

    filtered = _local_polynomial_derivative(noisy[:, None], fps, window=11, polyorder=3)[:, 0]
    raw = np.gradient(noisy, 1.0 / fps)

    assert np.sqrt(np.mean((filtered - expected) ** 2)) < np.sqrt(np.mean((raw - expected) ** 2))
