"""Action grouping must not smooth truth, lose motion frames, or invent goals."""
import numpy as np
import pytest

from somaforge_core.contact_action_segments import action_segments
from somaforge_core.demonstration_events import materialize
from somaforge_core.newton_contacts import PARTS


def labels(n=100):
    mask = np.zeros((n, len(PARTS)), bool)
    return dict(contact_part_mask=mask, contact_surface=np.zeros(mask.shape, int), fps=50.)


def bounds(events):
    return [(e['start_frame'], e['end_frame']) for e in events]


def test_short_same_surface_recontact_does_not_cut_or_modify_truth():
    data = labels(); data['contact_part_mask'][:, 0] = True
    data['contact_part_mask'][30:36, 0] = False
    before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    events, audit = action_segments(data)
    assert bounds(events) == [(0, 99)]
    assert audit['suppressed_recontacts'][0]['gap_frames'] == 6
    assert events[0]['touchdown_events'][0]['frame'] == 36
    assert not events[0]['persistent_parts']
    for key, value in before.items(): np.testing.assert_array_equal(data[key], value)


def test_genuine_swing_and_surface_transfer_survive_debounce():
    data = labels(); data['contact_part_mask'][:, 0] = True
    data['contact_part_mask'][25:32, 0] = False
    data['contact_surface'][60:, 0] = 1
    events, _ = action_segments(data)
    assert bounds(events) == [(0, 32), (32, 60), (60, 99)]


def test_cooperative_contacts_share_an_actual_stable_endpoint():
    data = labels(); data['contact_part_mask'][30:, 2] = True
    data['contact_part_mask'][36:, 5] = True
    events, audit = action_segments(data)
    assert bounds(events) == [(0, 36), (36, 99)]
    assert set(events[0]['target_contact_parts']) == {PARTS[2], PARTS[5]}
    assert len(audit['retained_boundaries'][0]['events']) == 2


def test_lost_contact_cannot_be_promoted_to_cooperative_target():
    data = labels(); data['contact_part_mask'][30:34, 2] = True
    data['contact_part_mask'][36:, 5] = True
    events, audit = action_segments(data)
    assert len(audit['rejected_boundaries']) == 1
    assert bounds(events) == [(0, 30), (30, 99)]
    assert events[0]['target_contact_parts'] == [PARTS[2]]


def test_clustering_cannot_chain_past_the_declared_total_span():
    data = labels()
    for part, frame in zip((0, 2, 4), (30, 36, 42)):
        data['contact_part_mask'][frame:, part] = True
    _, audit = action_segments(data, minimum_segment_frames=1)
    assert [g['frame'] for g in audit['retained_boundaries']] == [36, 42]


def test_surface_changes_then_hand_knee_actions_take_priority_over_footsteps():
    data = labels(130); data['contact_part_mask'][30:, 0] = True
    data['contact_part_mask'][40:, 2] = True
    data['contact_surface'][50:, 2] = 1
    events, audit = action_segments(data)
    assert bounds(events) == [(0, 30), (30, 50), (50, 129)]
    assert audit['rejected_boundaries'][0]['frame'] == 40
    assert audit['retained_boundaries'][-1]['priority'] == 2


def test_prefix_tail_and_priority_conflicts_are_audited_without_losing_frames():
    data = labels(101)
    for part, frame in ((0, 10), (2, 30), (4, 90)):
        data['contact_part_mask'][frame:, part] = True
    events, audit = action_segments(data)
    assert bounds(events) == [(0, 30), (30, 100)]
    assert [g['frame'] for g in audit['rejected_boundaries']] == [10, 90]
    assert sum(e['duration_frames'] for e in events) == 100
    original, _ = materialize(data, events)
    for actual, expected in zip(events, original):
        for key, value in expected.items(): assert actual[key] == value


def test_initial_contact_transient_and_short_whole_motion():
    data = labels(10); data['contact_part_mask'][:2, 0] = True
    data['contact_part_mask'][4:6, 1] = True
    events, audit = action_segments(data)
    assert bounds(events) == [(0, 9)]
    assert audit['candidate_events'] == []
    assert len(audit['short_segments']) == 1


@pytest.mark.parametrize('kwargs', [dict(stable_frames=0), dict(same_face_gap=-1),
    dict(cluster_window=True), dict(minimum_segment_frames=0)])
def test_invalid_rules_are_rejected(kwargs):
    with pytest.raises(ValueError): action_segments(labels(), **kwargs)
