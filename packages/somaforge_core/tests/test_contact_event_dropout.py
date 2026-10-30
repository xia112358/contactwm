"""Shared observed-contact dropout handling remains independent of archived clustering."""
import numpy as np
import pytest
from somaforge_core.contact_events import bridge_dropouts


@pytest.mark.parametrize('gap', [1, 2, 3, 4, 40])
def test_bounded_dropout(gap):
    raw = np.array([True] + [False]*gap + [True])
    saved = raw.copy()
    surface = np.where(raw, 2, -1)
    result = bridge_dropouts(raw, surface)
    assert result.all() == (gap <= 3)
    np.testing.assert_array_equal(raw, saved)


def test_edges_and_different_surface():
    raw = np.array([False, True, False, True, False])
    np.testing.assert_array_equal(bridge_dropouts(raw, np.array([-1, 2, -1, 3, -1])), raw)
    np.testing.assert_array_equal(bridge_dropouts(raw, np.array([-1, 2, 3, 2, -1])), raw)
