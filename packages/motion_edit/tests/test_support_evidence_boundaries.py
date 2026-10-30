"""Keep edit intentions while preventing inherited execution truth at boundaries."""
import json
import hashlib

import numpy as np
import pytest

from motion_edit.adapters.holosoma_npz import subset_arrays
from motion_edit.editing.splice import splice_motions
from somaforge_core.support_evidence import reference_only_support


def test_crop_preserves_native_pose_and_intent_but_requires_rebinding():
    q = np.arange(4*36).reshape(4,36)
    intent = np.zeros((4,6),bool); intent[1:3,0] = True
    observed = np.ones((4,6),bool)
    source = dict(joint_pos=q,contact_keep_intent_mask=intent,
                  observed_support_load_bearing=observed,
                  support_assessment_json=np.asarray(json.dumps(dict(status='observed'))))
    cropped = subset_arrays(source,1,3)
    np.testing.assert_array_equal(cropped['joint_pos'],q[1:3])
    np.testing.assert_array_equal(cropped['contact_keep_intent_mask'],intent[1:3])
    np.testing.assert_array_equal(cropped['source_reference_observed_support_load_bearing'],observed[1:3])
    assert 'observed_support_load_bearing' not in cropped
    assert json.loads(cropped['support_assessment_json'].item())['status'] == 'unknown'
    assert json.loads(source['support_assessment_json'].item())['status'] == 'observed'
    assert 'source_reference_observed_support_load_bearing' not in source


def test_splice_converts_legacy_roles_and_preserves_new_intent(tmp_path):
    a,b,out = (tmp_path/name for name in ('a.npz','b.npz','combined.npz'))
    keep_a = np.zeros((2,6),bool); keep_a[:,0] = True
    keep_b = np.zeros((3,6),bool); keep_b[:,1] = True
    np.savez(a,joint_pos=np.zeros((2,36)),support_part_mask=keep_a,
             observed_support_load_bearing=keep_a)
    np.savez(b,joint_pos=np.ones((3,36)),contact_keep_intent_mask=keep_b,
             observed_support_load_bearing=keep_b)
    original_a,original_b = a.read_bytes(),b.read_bytes()
    splice_motions([a,b],out)
    with np.load(out,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['contact_keep_intent_mask'],np.concatenate([keep_a,keep_b]))
        np.testing.assert_array_equal(z['joint_pos'],np.concatenate([np.zeros((2,36)),np.ones((3,36))]))
        assert 'observed_support_load_bearing' not in z
        assert json.loads(z['support_assessment_json'].item())['status'] == 'unknown'
    assert a.read_bytes() == original_a and b.read_bytes() == original_b


def test_repeated_edit_does_not_overwrite_original_force_provenance():
    original_provenance = np.asarray(json.dumps(dict(source='original_native_solve')))
    payload = dict(contact_force_part_w=np.ones((2,6,3)),
                   contact_force_provenance_json=original_provenance,
                   contact_keep_intent_mask=np.ones((2,6),bool),
                   support_assessment_json=np.asarray(json.dumps(dict(status='observed'))))
    reference_only_support(payload,reason='first edit',retain_force_targets=True)
    reference_only_support(payload,reason='second edit',retain_force_targets=True)
    assert payload['source_reference_contact_force_provenance_json'].item() == original_provenance.item()
    assert json.loads(payload['source_reference_support_assessment_json'].item())['status'] == 'observed'
    assert json.loads(payload['support_assessment_json'].item())['status'] == 'unknown'
    assert payload['contact_keep_intent_mask'].all()
