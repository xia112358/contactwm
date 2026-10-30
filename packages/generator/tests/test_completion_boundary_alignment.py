"""Observed labels follow the demonstration without shifting its action clock."""
import numpy as np
import pytest

from somaforge_core.demonstration_events import materialize


def evidence():
    mask = np.zeros((11, 6), bool); mask[:, :2] = True
    surface = np.full((11, 6), -1); surface[:, :2] = 0
    return dict(contact_part_mask=mask, contact_surface=surface, fps=50.)


def intervals():
    return [dict(start_frame=0, end_frame=5, active_parts=['right_foot'],
                 target_contact_parts=['left_foot', 'right_foot'], include_for_infiller=False),
            dict(start_frame=5, end_frame=10, persistent_parts=['right_foot'])]


def test_old_required_endpoint_contact_and_inclusion_flag_are_not_inherited():
    actual = evidence(); actual['contact_part_mask'][5, 1] = False
    events, audit = materialize(actual, intervals())
    assert [(e['start_frame'], e['end_frame']) for e in events] == [(0, 5), (5, 10)]
    assert events[0]['target_contact_parts'] == ['left_foot']
    assert events[0]['target_surfaces'] == [0, -1, -1, -1, -1, -1]
    assert events[0]['released_parts'] == ['right_foot']
    assert all(e['include_for_infiller'] for e in events)
    assert events[1]['active_parts'] == ['right_foot']
    assert events[1]['touchdown_events'][0]['frame'] == 6
    assert audit['omitted_event_count'] == audit['boundary_shifts'] == 0
    assert not actual['contact_part_mask'][5, 1]


def test_same_face_continuous_contact_is_keep_not_authored_touchdown():
    events, _ = materialize(evidence(), intervals())
    assert events[0]['active_parts'] == []
    assert events[0]['persistent_parts'] == ['left_foot', 'right_foot']
    assert events[0]['terminal_event'] == 'pose_adjustment'


def test_contact_churn_does_not_create_frame_level_actions():
    actual = evidence(); actual['contact_part_mask'][2, 0] = False
    events, _ = materialize(actual, intervals())
    assert len(events) == 2
    assert events[0]['active_parts'] == ['left_foot']
    assert events[0]['persistent_parts'] == ['right_foot']
    assert events[0]['touchdown_events'][0]['frame'] == 3


def test_surface_change_is_observed_from_own_labels():
    actual = evidence(); actual['contact_surface'][3:, 0] = 1
    events, _ = materialize(actual, intervals())
    assert events[0]['touchdown_events'][0]['surface'] == 1
    assert events[0]['released_parts'] == ['left_foot']
    assert events[1]['persistent_parts'] == ['left_foot', 'right_foot']


def test_no_contact_samples_and_terminal_adjustment_are_retained():
    actual = evidence(); actual['contact_part_mask'][5:] = False
    events, audit = materialize(actual, intervals())
    assert events[0]['target_contact_parts'] == []
    assert events[1]['terminal_event'] == 'pose_adjustment'
    assert audit['retained_frame_range'] == [0, 10]


@pytest.mark.parametrize('clock', [
    [dict(start_frame=0, end_frame=5), dict(start_frame=6, end_frame=10)],
    [dict(start_frame=0, end_frame=5)],
    [dict(start_frame=1, end_frame=10)],
    [dict(start_frame=0, end_frame=0), dict(start_frame=0, end_frame=10)],
])
def test_incomplete_clock_is_a_structural_error_not_event_pruning(clock):
    with pytest.raises(ValueError, match='complete demonstration continuously'):
        materialize(evidence(), clock)
