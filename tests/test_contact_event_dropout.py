import sys
from pathlib import Path
import numpy as np
import pytest
from somaforge_core.contact_events import bridge_dropouts, group_endpoint

from generator.research.archived_experiments.event_clustering import Config, cluster_events, events_for_part


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


@pytest.mark.parametrize('gap', [1, 2, 3, 4])
def test_cluster_waits_for_real_endpoint(gap):
    mask = np.zeros((20, 2), bool)
    mask[5:, 0] = True
    mask[6:6+gap, 0] = False
    mask[6:, 1] = True
    surface = np.where(mask, 2, -1)
    events = [dict(frame=5, part_index=0, surface=2), dict(frame=6, part_index=1, surface=2)]
    groups = cluster_events(events, mask, 50, Config(), surface)
    assert len(groups) == (1 if gap <= 3 else 2)
    if gap <= 3:
        assert group_endpoint(groups[0]) == 6+gap
        assert mask[group_endpoint(groups[0])].all()
        assert [e['frame'] for e in groups[0]] == [5, 6]
    assert 'cluster_endpoint_frame' not in events[0]


def test_three_frame_gap_does_not_create_touchdown():
    mask = np.ones(30, bool)
    mask[10:13] = False
    pos = np.zeros((30, 3)); pos[:, 0] = np.arange(30)*.1
    margins = np.full(len(mask), .02)
    events, _ = events_for_part(mask, None, None, pos, 50, Config(),
                                np.where(mask, 2, -1), np.array([0., 0., 1.]), margins)
    assert not events


def test_same_surface_threshold_chatter_is_not_touchdown():
    mask = np.ones(30, bool)
    mask[10:15] = False
    pos = np.zeros((30, 3))
    pos[9:16, 2] = np.array([0., .001, .002, .003, .004, .003, .002])
    margins = np.full(len(mask), .02)
    events, rejected = events_for_part(mask, None, None, pos, 50, Config(),
        np.where(mask, 2, -1), np.array([0., 0., 1.]), margins)
    assert len(events) == 1
    assert not events[0]['motion_qualified']
    assert rejected[0]['reason'] == 'insufficient_surface_normal_motion'
    tagged = [dict(events[0], part_index=0, surface=2)]
    assert not cluster_events(tagged, mask[:, None], 50, Config(),
                              np.where(mask[:, None], 2, -1))


def test_same_surface_real_lift_remains_touchdown():
    mask = np.ones(30, bool)
    mask[10:15] = False
    pos = np.zeros((30, 3))
    pos[9:16, 2] = np.array([0., .005, .012, .018, .012, .005, 0.])
    margins = np.full(len(mask), .02)
    events, rejected = events_for_part(mask, None, None, pos, 50, Config(),
        np.where(mask, 2, -1), np.array([0., 0., 1.]), margins)
    assert len(events) == 1
    assert events[0]['motion_qualified']
    assert not rejected
