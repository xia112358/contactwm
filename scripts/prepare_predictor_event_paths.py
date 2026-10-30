"""Export all action observations from a demonstration's own Newton evidence.

The input interval file supplies timestamps only. Old targets, roles, and
inclusion flags are ignored. There is no accepted/rejected event partition.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from generator.support_path_acceptance import SCHEMA, file_evidence
from somaforge_core.demonstration_events import materialize, observations


def prepare(motion, labels, intervals_path, output):
    motion, labels, intervals_path, output = map(Path, (motion, labels, intervals_path, output))
    data = observations(motion, labels)
    intervals = [json.loads(line) for line in intervals_path.read_text().splitlines() if line.strip()]
    events, audit = materialize(data, intervals)
    output.mkdir(parents=True, exist_ok=True)
    clock_path = output / 'action_intervals.json'
    clock_path.write_text(json.dumps(dict(
        source=file_evidence(intervals_path), consumed_fields=['start_frame', 'end_frame'],
        intervals=[dict(start_frame=e['start_frame'], end_frame=e['end_frame']) for e in events]),
        indent=2))
    for event in events:
        event.update(source_path=str(motion.resolve()), contact_source_path=str(labels.resolve()))
    event_path = output / 'events.jsonl'
    event_path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    report = dict(schema=SCHEMA,
        evidence=dict(motion=file_evidence(motion), labels=file_evidence(labels),
                      action_clock=file_evidence(clock_path), events=file_evidence(event_path)),
        segments=events, audit=audit, frame_count=len(data['contact_part_mask']),
        event_count=len(events), sampling='fixed_q_forward_no_integration',
        scope='demonstration_observation_identity; not physical trajectory acceptance',
        observed_support_status='unknown_without_same_solve_loads',
        zero_contact_frames=np.flatnonzero(~data['contact_part_mask'].any(-1)).tolist(),
        training_ready=False)
    certificate = output / 'event_evidence.json'
    certificate.write_text(json.dumps(report, indent=2, allow_nan=False))
    with np.load(labels, allow_pickle=False) as z:
        semantics = json.loads(z['contact_semantics_json'].item())
    manifest = dict(
        demonstration_event_evidence=[dict(file_evidence(certificate), motion_id=0)],
        newton_runtime=semantics['provenance']['newton_runtime'],
        motion_files=[dict(motion_file=str(motion.resolve()),
            newton_contact_file=str(labels.resolve()), event_segments_file=str(event_path.resolve()),
            event_segments_sha256=file_evidence(event_path)['sha256'])],
        training_ready=False, scope=report['scope'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('motion', 'labels', 'intervals', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.motion, args.labels, args.intervals, args.output)
    print(json.dumps(dict(events=result['event_count'], omitted_events=0)))
