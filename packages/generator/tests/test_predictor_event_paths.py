"""The exporter retains the complete clock, including contactless samples."""
import json

import numpy as np
import prepare_predictor_event_paths as module


def test_export_ignores_old_requirements_and_does_not_partition_samples(tmp_path, monkeypatch):
    motion, labels, intervals = (tmp_path / name for name in ('motion.npz', 'labels.npz', 'intervals.jsonl'))
    motion.write_bytes(b'canonical motion fixture')
    np.savez(labels, contact_semantics_json=np.asarray(json.dumps(
        dict(provenance=dict(newton_runtime='fixture')))))
    intervals.write_text('\n'.join(json.dumps(e) for e in [
        dict(start_frame=0, end_frame=2, include_for_infiller=False,
             persistent_parts=['right_foot'], active_parts=['right_foot']),
        dict(start_frame=2, end_frame=4, include_for_infiller=False)]) + '\n')
    mask = np.zeros((5, 6), bool); mask[:2, 0] = True
    faces = np.where(mask, 0, -1)
    monkeypatch.setattr(module, 'observations', lambda *args: dict(
        contact_part_mask=mask, contact_surface=faces, fps=50.))
    out = tmp_path / 'observed'
    report = module.prepare(motion, labels, intervals, out)
    assert report['event_count'] == 2
    assert report['zero_contact_frames'] == [2, 3, 4]
    assert report['observed_support_status'] == 'unknown_without_same_solve_loads'
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    assert events[0]['target_contact_parts'] == []
    assert all('accepted' not in e for e in events)
    assert not (out / 'accepted_events.jsonl').exists()
    assert not (out / 'rejected_events.jsonl').exists()
    clock = json.loads((out / 'action_intervals.json').read_text())
    assert clock['consumed_fields'] == ['start_frame', 'end_frame']
    assert clock['intervals'] == [dict(start_frame=0, end_frame=2), dict(start_frame=2, end_frame=4)]
    manifest = json.loads((out / 'manifest.json').read_text())
    assert 'support_path_acceptance' not in manifest
    assert len(manifest['demonstration_event_evidence']) == 1
