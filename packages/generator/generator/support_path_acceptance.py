"""Bind predictor samples to complete, self-observed demonstration events.

Integrity validation checks evidence identity and frame correspondence. It
does not impose contact goals or decide which demonstrations deserve training.
Old filtered-event manifests must be explicitly rebuilt.
"""
import hashlib
import json

from somaforge_core.demonstration_events import SCHEMA as EVENT_SCHEMA
from somaforge_core.published_data import load_published_json, relocate_published_values, resolve_published_path

SCHEMA = 'predictor_demonstration_observation_evidence_v1'


def file_evidence(path):
    path = resolve_published_path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_support_paths(manifest_path, samples):
    manifest_path = resolve_published_path(manifest_path)
    manifest = load_published_json(manifest_path)
    declarations = manifest.get('demonstration_event_evidence')
    if declarations is None:
        raise ValueError('Rebuild predictor labels from each demonstration; old filtered event manifests are unsupported')
    if not declarations:
        raise ValueError('Empty demonstration evidence declaration')
    observed, motion_ids, verified = set(), set(), []
    def resolve(value):
        return resolve_published_path(value, manifest_path.parent)
    for declaration in declarations:
        path = resolve(declaration['path'])
        if file_evidence(path)['sha256'] != declaration['sha256']:
            raise ValueError('Stale demonstration certificate')
        certificate = load_published_json(path)
        if certificate.get('schema') != SCHEMA:
            raise ValueError('Unknown demonstration observation schema')
        for evidence in certificate['evidence'].values():
            if file_evidence(evidence['path'])['sha256'] != evidence['sha256']:
                raise ValueError('Stale demonstration evidence')
        motion_id = int(declaration['motion_id'])
        if motion_id in motion_ids or not 0 <= motion_id < len(manifest['motion_files']):
            raise ValueError('Duplicate or invalid demonstration motion ID')
        motion_ids.add(motion_id)
        entry = manifest['motion_files'][motion_id]
        for field, evidence_key in (('motion_file', 'motion'), ('newton_contact_file', 'labels'),
                                    ('event_segments_file', 'events')):
            if file_evidence(resolve(entry[field]))['sha256'] != certificate['evidence'][evidence_key]['sha256']:
                raise ValueError('Certificate does not belong to manifest motion/contact/event entry')
        event_path = resolve(entry['event_segments_file'])
        if file_evidence(event_path)['sha256'] != entry['event_segments_sha256']:
            raise ValueError('Stale observed event table')
        events = [relocate_published_values(json.loads(line))
                  for line in event_path.read_text().splitlines() if line.strip()]
        if events != certificate['segments']:
            raise ValueError('Event table differs from demonstration certificate')
        clock = load_published_json(certificate['evidence']['action_clock']['path'])['intervals']
        boundaries = [dict(start_frame=e['start_frame'], end_frame=e['end_frame']) for e in events]
        if clock != boundaries:
            raise ValueError('Observed events do not preserve the demonstration action clock')
        if (not events or events[0]['start_frame'] != 0
                or events[-1]['end_frame'] != certificate['frame_count']-1
                or any(a['end_frame'] != b['start_frame'] for a, b in zip(events, events[1:]))):
            raise ValueError('Incomplete demonstration action chain')
        for event in events:
            if event.get('schema') != EVENT_SCHEMA or 'accepted' in event:
                raise ValueError('Old intent or accepted/rejected event semantics are unsupported')
            key = (motion_id, int(event['start_frame']), int(event['end_frame']))
            if key in observed or not 0 <= key[1] < key[2] < certificate['frame_count']:
                raise ValueError('Duplicate or invalid demonstration interval')
            observed.add(key)
        verified.append(str(path.resolve()))
    if motion_ids != set(range(len(manifest['motion_files']))):
        raise ValueError('Missing demonstration evidence for manifest motion')
    for sample in samples:
        key = tuple(int(sample[k]) for k in ('motion_id', 'current_frame', 'target_frame'))
        if key not in observed:
            raise ValueError(f'Predictor sample lacks matching demonstration observation: {key}')
    return dict(status='verified_demonstration_observations', samples=len(samples),
        event_count=len(observed), certificates=verified,
        scope='identity and complete action clock, no quality filtering',
        predicted_path_status='not_observed',
        force_bearing_slip_status='unknown_without_same_solve_loads')
