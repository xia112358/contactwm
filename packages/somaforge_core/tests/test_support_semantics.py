import json

import numpy as np
import pytest

from somaforge_core.support_evidence import reference_only_support
from somaforge_core.support_semantics import assess_support, assess_support_path


def test_missing_zero_and_positive_load_are_distinct():
    unknown = assess_support([True])
    assert unknown['load_state'].tolist() == ['unknown']
    observed = assess_support([True, True, False], [0, 100, 0],
                             [np.nan, .2, np.nan], load_known=[True]*3)
    assert observed['load_state'].tolist() == ['unloaded','load_bearing','no_contact']
    assert observed['loaded_motion_known'].tolist() == [False,True,False]


def test_sliding_contact_stays_load_bearing_and_intent_does_not_create_load():
    result = assess_support([True,True], [100,0], [.2,np.nan], load_known=[True]*2,
                            speed_limit=.01, keep_intent=[False,True])
    assert result['slip_state'].tolist() == ['slipping','not_load_bearing']
    assert result['load_bearing'].tolist() == [True,False]
    assert result['keep_intent'].tolist() == [False,True]


def test_force_cannot_create_contact_or_replace_unknown_evidence():
    with pytest.raises(ValueError, match='create contact'):
        assess_support([False],[100],load_known=[True])
    with pytest.raises(ValueError, match='invalid observed'):
        assess_support([True],load_known=[True])
    assert not assess_support([True],[100],load_known=[False])['load_bearing'].any()


def test_load_transfer_is_not_a_static_single_foot_obligation():
    force = np.zeros((4,6)); force[:2,0] = 100; force[2:,1] = 100
    contact = np.ones((4,6),bool)
    observations = assess_support(contact,force,np.zeros_like(force),load_known=contact)
    report = assess_support_path(observations,0,3)
    assert report['load_coverage_passed'] and report['unloaded_sample_frames'] == []
    assert report['stationary_support_status'].startswith('unknown')
    assert assess_support_path(observations,0,3,speed_limit=.01)['stationary_support_status']=='passed_at_recorded_solve_samples'


def test_unknown_path_sample_cannot_certify_support():
    observations = assess_support(np.ones((3,6),bool))
    report = assess_support_path(observations,0,2)
    assert not report['load_coverage_passed'] and report['unknown_sample_frames']==[0,1,2]


def test_edit_preserves_original_evidence_only_as_reference():
    force = np.array([[[0.,0,100]]])
    payload = dict(joint_pos=np.zeros((1,36)),contact_force_part_w=force.copy(),
                   solver_contact_force_on_body1_w=force.copy(),
                   support_part_mask=np.ones((1,6),bool),
                   observed_support_load_bearing=np.ones((1,6),bool))
    reference_only_support(payload, reason='edited trajectory')
    np.testing.assert_array_equal(payload['source_reference_contact_force_part_w'],force)
    assert 'contact_force_part_w' not in payload and 'observed_support_load_bearing' not in payload
    assert payload['contact_keep_intent_mask'].all()
    assert json.loads(payload['support_assessment_json'].item())['status']=='unknown'
