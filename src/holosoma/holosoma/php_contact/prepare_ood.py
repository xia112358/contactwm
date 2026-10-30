"""Prepare an unseen native-contact task for frozen student inference only."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from holosoma.php_contact.dataset import SCHEMA, digest, interval_targets, load_prepared
from somaforge_core.contact_action_segments import action_segments
from somaforge_core.newton_contact_data import load_contact_labels
from somaforge_core.newton_contacts import PARTS
from somaforge_core.published_data import load_published_json, resolve_published_path
from somaforge_core.robot_assets import decode_robot_asset_json, validate_g1_asset_metadata


@dataclass(frozen=True)
class Options:
    source_manifest: Path
    labels: Path
    checkpoint: Path
    output_dir: Path
    motion_id: int = 0


def prepare(options):
    import torch
    source_path = resolve_published_path(options.source_manifest)
    source = load_published_json(source_path)
    row = source['motion_files'][options.motion_id]
    motion = resolve_published_path(row['motion_file'], base_dir=source_path.parent)
    terrain = next(t for t in source['terrains'] if t['terrain_id'] == row['terrain_id'])
    terrain_path = resolve_published_path(terrain['terrain_file'], base_dir=source_path.parent)
    if digest(motion) != row['motion_sha256'] or digest(terrain_path) != terrain['terrain_sha256']:
        raise ValueError('OOD task source hash mismatch')
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    validate_g1_asset_metadata(saved['robot_asset'], context=str(options.checkpoint))
    params = saved['experiment_config']['command']['setup_terms']['motion_command']['params']
    training = load_published_json(resolve_published_path(params['motion_config']['motion_manifest']))
    original = load_published_json(resolve_published_path(training['teacher']['training_manifest']))
    for key, field, value in (('motion_files', 'motion_sha256', digest(motion)),
                              ('motion_files', 'source_sha256', row['source_sha256']),
                              ('terrains', 'terrain_sha256', digest(terrain_path))):
        if value in {r.get(field) for r in original[key]}:
            raise ValueError(f'OOD task overlaps the student training source: {field}')
    labels_path = options.labels.expanduser().resolve()
    labels = load_contact_labels(motion, labels_path)
    with np.load(motion, allow_pickle=False) as z:
        decode_robot_asset_json(z['robot_asset_json'], context=str(motion))
        labels['fps'] = float(z['fps'])
        n = len(z['joint_pos'])
    if not np.isclose(labels['fps'], 50.):
        raise ValueError('OOD reference must use the trained 50 Hz control clock')
    with np.load(labels_path, allow_pickle=False) as z:
        semantics = json.loads(z['contact_semantics_json'].item())
    events, audit = action_segments(labels, minimum_segment_frames=20)
    arrays = interval_targets(labels, events)
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    event_file = output/'events.jsonl'
    event_file.write_text(''.join(json.dumps(e)+'\n' for e in events))
    report = dict(source_manifest=str(source_path), source_manifest_sha256=digest(source_path),
        checkpoint=str(options.checkpoint.resolve()), checkpoint_sha256=digest(options.checkpoint),
        original_training_manifest=training['teacher']['training_manifest'],
        source_motion=str(motion), motion_sha256=digest(motion),
        source_labels=str(labels_path), labels_sha256=digest(labels_path),
        terrain_file=str(terrain_path), terrain_sha256=digest(terrain_path),
        source_motion_id=options.motion_id, frames=n, duration_s=(n-1)/labels['fps'],
        intervals=len(events), cuts=audit, overlap_with_training_source=False,
        goal_source='own native fixed-q contacts; action endpoints and root goals from unseen reference',
        training_started=False, physical_execution_verified=False)
    evidence = output/'preparation_report.json'
    evidence.write_text(json.dumps(report, indent=2)+'\n')
    metadata = dict(schema=SCHEMA, parts=PARTS, source_motion=str(motion), source_sha256=digest(motion),
        source_labels=str(labels_path), labels_sha256=digest(labels_path),
        events=str(event_file), events_sha256=digest(event_file),
        evidence=str(evidence), evidence_sha256=digest(evidence),
        split='evaluation', interval_count=len(events), frame_count=n,
        reference_sampling=semantics['sampling'], source_newton_runtime=semantics['provenance']['newton_runtime'])
    contact = output/'contacts.npz'
    np.savez_compressed(contact, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    manifest = dict(schema=SCHEMA, parts=PARTS, split='evaluation',
        reference_origin='out_of_distribution_native_reference',
        reference_sampling=semantics['sampling'], newton_runtime=semantics['provenance']['newton_runtime'],
        training_ready=False, physical_execution_verified=False,
        terrain_spacing=source['terrain_spacing'],
        terrains=[dict(terrain_id=0, terrain_file=str(terrain_path), terrain_sha256=digest(terrain_path))],
        motion_files=[dict(motion_id=0, motion_file=str(motion), motion_sha256=digest(motion),
            terrain_id=0, contact_file=str(contact), contact_sha256=digest(contact), source_motion_id=options.motion_id)],
        evaluation_evidence=str(evidence), evaluation_evidence_sha256=digest(evidence))
    manifest_path = output/'evaluation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    load_prepared(manifest_path)
    return report


def main():
    import tyro
    report = prepare(tyro.cli(Options))
    print(json.dumps({key:report[key] for key in ('frames','duration_s','intervals','training_started')}))


if __name__ == '__main__':
    main()
