import json

import pytest
from generator.support_path_acceptance import SCHEMA, file_evidence, validate_support_paths
from somaforge_core.demonstration_events import SCHEMA as EVENT_SCHEMA


def fixture(tmp_path):
    motion = tmp_path / 'motion.npz'; motion.write_bytes(b'canonical motion')
    labels = tmp_path / 'labels.npz'; labels.write_bytes(b'actual Newton evidence')
    clock = tmp_path / 'clock.json'
    intervals = [dict(start_frame=0, end_frame=10), dict(start_frame=10, end_frame=20)]
    clock.write_text(json.dumps(dict(intervals=intervals)))
    events = [dict(schema=EVENT_SCHEMA, **e) for e in intervals]
    event_path = tmp_path / 'events.jsonl'
    event_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    certificate = tmp_path / 'certificate.json'
    certificate.write_text(json.dumps(dict(schema=SCHEMA, frame_count=21,
        evidence=dict(motion=file_evidence(motion), labels=file_evidence(labels),
                      events=file_evidence(event_path), action_clock=file_evidence(clock)),
        segments=events, zero_contact_frames=[10])))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(
        demonstration_event_evidence=[dict(file_evidence(certificate), motion_id=0)],
        motion_files=[dict(motion_file=str(motion), newton_contact_file=str(labels),
            event_segments_file=str(event_path), event_segments_sha256=file_evidence(event_path)['sha256'])])))
    return manifest, motion, labels


def sample(a=0, b=10):
    return dict(motion_id=0, current_frame=a, target_frame=b)


def test_complete_observed_clock_including_contactless_endpoint(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    result = validate_support_paths(manifest, [sample(), sample(10, 20)])
    assert result['status'] == 'verified_demonstration_observations'
    assert result['event_count'] == 2


@pytest.mark.parametrize('a,b', [(0, 20), (1, 10), (10, 19), (20, 30)])
def test_samples_must_match_their_own_clock(tmp_path, a, b):
    manifest, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='matching demonstration observation'):
        validate_support_paths(manifest, [sample(a, b)])


def test_changed_contact_evidence_is_an_integrity_error(tmp_path):
    manifest, _, labels = fixture(tmp_path)
    labels.write_bytes(b'changed contact evidence')
    with pytest.raises(ValueError, match='Stale demonstration evidence'):
        validate_support_paths(manifest, [sample()])


def test_certificate_cannot_be_reused_for_other_motion(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    other = tmp_path / 'other.npz'; other.write_bytes(b'other motion')
    data = json.loads(manifest.read_text()); data['motion_files'][0]['motion_file'] = str(other)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='does not belong'):
        validate_support_paths(manifest, [sample()])


def test_old_filtered_manifest_cannot_be_silently_reused(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(support_path_acceptance=[])))
    with pytest.raises(ValueError, match='Rebuild predictor labels'):
        validate_support_paths(manifest, [sample()])


def test_empty_declaration_cannot_disable_identity_checks(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(demonstration_event_evidence=[])))
    with pytest.raises(ValueError, match='Empty demonstration'):
        validate_support_paths(manifest, [sample()])
