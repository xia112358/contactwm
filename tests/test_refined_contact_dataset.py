import copy
import json

import numpy as np
import pytest

from somaforge_core import refined_contact_dataset as release
from somaforge_core import trajectory_refinement as io
from somaforge_core.contact_events import load_segment_rows
from somaforge_core.newton_contacts import PARTS


def labels(mask, surfaces=None):
    mask = np.asarray(mask, bool)
    return dict(contact_part_mask=mask, contact_surface=np.where(mask, 0, -1)
                if surfaces is None else surfaces, fps=50.)


def test_own_clock_preserves_prefix_tail_and_observed_contact_roles(tmp_path):
    mask = np.zeros((25, 6), bool)
    mask[:, 0] = True
    mask[7:, 2] = True
    mask[16:, 1] = True
    data = labels(mask)
    rows, audit = release.self_observed_segments(data)
    assert [(r['start_frame'], r['end_frame']) for r in rows] == [(0, 7), (7, 16), (16, 24)]
    assert rows[0]['active_parts'] == [PARTS[2]]
    assert rows[0]['persistent_parts'] == [PARTS[0]]
    assert rows[-1]['terminal_event'] == 'pose_adjustment'
    assert audit['retained_frame_range'] == [0, 24]
    assert not audit['old_contact_requirements_used']
    path = tmp_path/'events.jsonl'
    release._jsonl(path, rows)
    entry = dict(event_segments_file=str(path), event_segments_sha256=io.sha256_file(path),
                 event_selection=audit['selection'])
    assert load_segment_rows(entry, data, 50.) == rows


def test_nearby_touchdowns_do_not_require_old_order_or_minimum_duration():
    mask = np.zeros((12, 6), bool)
    mask[3:7, 2] = True
    mask[8:, 1] = True
    before = mask.copy()
    rows, _ = release.self_observed_segments(labels(mask))
    assert [(r['start_frame'], r['end_frame']) for r in rows] == [(0, 3), (3, 8), (8, 11)]
    assert rows[1]['released_parts'] == [PARTS[2]]
    assert np.array_equal(mask, before)


def test_full_timeline_kept_even_without_touchdowns():
    rows, _ = release.self_observed_segments(labels(np.zeros((2, 6), bool)))
    assert len(rows) == 1 and rows[0]['end_frame'] == 1
    assert rows[0]['active_parts'] == rows[0]['persistent_parts'] == []


def test_same_part_new_surface_is_an_observed_event():
    mask = np.zeros((15, 6), bool); mask[:, 0] = True
    faces = np.where(mask, 0, -1); faces[8:, 0] = 1
    rows, _ = release.self_observed_segments(labels(mask, faces))
    assert rows[0]['touchdown_events'] == [dict(frame=8, part_index=0, part=PARTS[0], surface=1)]
    assert rows[0]['persistent_parts'] == []


@pytest.mark.parametrize('changes', [dict(stable_frames=0), dict(merge_gap=-1)])
def test_invalid_event_configuration_rejected(changes):
    with pytest.raises(ValueError):
        release.self_observed_segments(labels(np.zeros((10, 6), bool)), **changes)


def reviewed_case(tmp_path, monkeypatch):
    motion, contacts = tmp_path/'motion', tmp_path/'contacts'
    motion.write_bytes(b'canonical motion'); contacts.write_bytes(b'actual Newton contacts')
    result = dict(motion=str(motion), contacts=str(contacts), task_acceptance=dict(passed=True),
        clock_validation=dict(passed=True), joint_limits=dict(passed=True), aggregation_written=True,
        contributing_env_ids=[0, 1, 2], optimization=dict(representative_fallback=False))
    result_path = tmp_path/'result.json'; io.write_json(result_path, result)
    pin = io.fingerprint(result_path)
    review = dict(decision='accept', authorized_by='user', authorization_text='那就接收吧',
        source_case_key='case', evidence=dict(original_result=pin, motion=io.fingerprint(motion),
                                            contacts=io.fingerprint(contacts)))
    review_path = tmp_path/'review.json'; io.write_json(review_path, review)
    entry = dict(source_case_key='case', motion_file=str(motion), motion_sha256=io.sha256_file(motion),
        contacts=str(contacts), contacts_sha256=io.sha256_file(contacts), validation_result=pin,
        acceptance=dict(accepted=True, basis='explicit_user_review', review=io.fingerprint(review_path)))
    monkeypatch.setattr(release.extraction, 'verify_validation', lambda _: None)
    monkeypatch.setattr(release.extraction, 'classification', lambda _: 'needs_refinement')
    return entry, result, review


def test_explicit_acceptance_is_bound_to_the_exact_trajectory(tmp_path, monkeypatch):
    entry, _, _ = reviewed_case(tmp_path, monkeypatch)
    assert release.verify_acceptance(entry) == entry['acceptance']['review']
    altered = copy.deepcopy(entry); altered['contacts_sha256'] = '0'*64
    with pytest.raises(ValueError, match='changed'):
        release.verify_acceptance(altered)


@pytest.mark.parametrize('gate', ['task_acceptance', 'clock_validation', 'joint_limits'])
def test_review_does_not_waive_required_physics_or_clock_gates(tmp_path, monkeypatch, gate):
    entry, result, review = reviewed_case(tmp_path, monkeypatch)
    result[gate]['passed'] = False
    io.write_json(entry['validation_result']['path'], result)
    entry['validation_result'] = io.fingerprint(entry['validation_result']['path'])
    review['evidence']['original_result'] = entry['validation_result']
    io.write_json(entry['acceptance']['review']['path'], review)
    entry['acceptance']['review'] = io.fingerprint(entry['acceptance']['review']['path'])
    with pytest.raises(ValueError, match='required gate'):
        release.verify_acceptance(entry)
