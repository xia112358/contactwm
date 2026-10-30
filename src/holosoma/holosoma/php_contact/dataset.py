"""Adapt immutable Predictor evidence to PHP without certifying physical execution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from somaforge_core.demonstration_events import materialize
from somaforge_core.newton_contact_data import load_contact_labels, require_current_newton_manifest
from somaforge_core.newton_contacts import PARTS
from somaforge_core.published_data import load_published_json, resolve_published_path
from somaforge_core.robot_assets import decode_robot_asset_json

SCHEMA = 'somaforge_php_predictor_contacts_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def split_name(height):
    # Frozen Predictor split: generator.training_data.split_name.
    return 'test' if height >= 1.075 else 'validation' if height <= 0.925 else 'train'


def interval_targets(labels, events):
    """Keep authored shared boundaries; the next command starts at that boundary."""
    rebuilt, _ = materialize(labels, events)
    for old, new in zip(events, rebuilt, strict=True):
        for key, value in new.items():
            if old.get(key) != value:
                raise ValueError(f'Event evidence differs from current native labels: {key}')
    frames = len(labels['contact_part_mask'])
    target = np.full(frames, -1, np.int64)
    segment = np.full(frames, -1, np.int64)
    for i, event in enumerate(events):
        a, b = event['start_frame'], event['end_frame']
        stop = b+1 if i == len(events)-1 else b
        target[a:stop] = b
        segment[a:stop] = i
    if (target < 0).any():
        raise ValueError('Contact intervals leave uncovered frames')
    active = labels['contact_part_mask'][target]
    return dict(reference_active=labels['contact_part_mask'],
        target_active=active, target_position_w=np.where(active[..., None], labels['contact_position_w'][target], 0),
        target_frame=target, segment_id=segment)


@dataclass(frozen=True)
class Options:
    source_manifest: Path
    output_dir: Path


def prepare(options):
    source = resolve_published_path(options.source_manifest)
    manifest = load_published_json(source)
    require_current_newton_manifest(manifest, context=str(source))
    if manifest.get('event_contract', {}).get('contact_targets') != 'own_actual_Newton_observations':
        raise ValueError('Expected the current self-observed Predictor dataset')
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'contacts').mkdir()
    groups = {name: [] for name in ('train', 'validation', 'test')}
    counts = dict.fromkeys(groups, 0)
    terrains = []
    terrain_ids = {}
    for row in manifest['terrains']:
        key = str(row['terrain_id'])
        if key in terrain_ids:
            raise ValueError('Duplicate Predictor terrain identity')
        terrain_ids[key] = len(terrains)
        terrain = resolve_published_path(row['terrain_file'], base_dir=source.parent)
        terrains.append(dict(terrain_id=len(terrains), terrain_file=str(terrain), terrain_sha256=digest(terrain)))
    declarations = {int(row['motion_id']): row for row in manifest['demonstration_event_evidence']}
    audit = []
    for index, row in enumerate(manifest['motion_files']):
        paths = {key: resolve_published_path(row[key], base_dir=source.parent)
                 for key in ('motion_file', 'newton_contact_file', 'event_segments_file', 'edit_plan_file')}
        motion = paths['motion_file']
        evidence_path = resolve_published_path(declarations[index]['path'], base_dir=source.parent)
        if digest(evidence_path) != declarations[index]['sha256']:
            raise ValueError('Predictor event certificate hash mismatch')
        certificate = load_published_json(evidence_path)
        for key, path in (('motion', motion), ('labels', paths['newton_contact_file']),
                          ('events', paths['event_segments_file'])):
            if digest(path) != certificate['evidence'][key]['sha256']:
                raise ValueError(f'Predictor {key} evidence changed')
        if digest(paths['event_segments_file']) != row['event_segments_sha256']:
            raise ValueError('Predictor event hash mismatch')
        with np.load(motion, allow_pickle=False) as z:
            decode_robot_asset_json(z['robot_asset_json'], context=str(motion))
            fps = float(z['fps'])
            frame_count = len(z['joint_pos'])
            if not np.isclose(fps, 50.):
                raise ValueError('PHP contact references must use the 50 Hz control clock')
            if z['joint_pos'].shape[1] != 36:
                raise ValueError('Expected canonical floating-root plus 29-joint motion')
        with np.load(paths['newton_contact_file'], allow_pickle=False) as z:
            semantics = json.loads(z['contact_semantics_json'].item())
        labels = load_contact_labels(motion, paths['newton_contact_file'], frame_count=frame_count)
        labels['fps'] = fps
        events = [json.loads(line) for line in paths['event_segments_file'].read_text().splitlines() if line.strip()]
        if certificate['segments'] != events:
            raise ValueError('Event certificate and interval file disagree')
        arrays = interval_targets(labels, events)
        plan = load_published_json(paths['edit_plan_file'])
        height = float(plan['metadata']['height_scale'])
        split = split_name(height)
        metadata = dict(schema=SCHEMA, parts=PARTS, source_motion=str(motion), source_sha256=digest(motion),
            source_labels=str(paths['newton_contact_file']), labels_sha256=digest(paths['newton_contact_file']),
            events=str(paths['event_segments_file']), events_sha256=digest(paths['event_segments_file']),
            evidence=str(evidence_path), evidence_sha256=digest(evidence_path),
            reference_sampling=semantics['sampling'], source_newton_runtime=manifest['newton_runtime'],
            split=split, height_scale=height, interval_count=len(events), frame_count=frame_count,
            physical_execution_verified=False, support_loads='unknown')
        target = output/'contacts'/f'{index:04d}.npz'
        np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
        groups[split].append(dict(motion_id=len(groups[split]), motion_file=str(motion),
            terrain_id=terrain_ids[str(row['terrain_id'])], contact_file=str(target),
            contact_sha256=digest(target), motion_sha256=metadata['source_sha256'],
            predictor_motion_id=index, predictor_motion_name=row['motion_id'], height_scale=height))
        counts[split] += len(events)
        audit.append(metadata)
        if (index+1) % 20 == 0:
            print(json.dumps(dict(prepared=index+1, total=len(manifest['motion_files']))), flush=True)
    report = dict(schema=SCHEMA, source_manifest=str(source), source_manifest_sha256=digest(source),
        motion_counts={name: len(rows) for name, rows in groups.items()}, interval_counts=counts,
        source_training_ready=manifest.get('training_ready'), training_ready=False,
        reference_role='fixed_q_contact_intent_for_experimental_tracking',
        physical_execution_verified=False, motions=audit)
    for split, rows in groups.items():
        payload = dict(schema=SCHEMA, source_kind='predictor_fixed_q_contact_intervals',
            split=split, motion_files=rows, terrains=terrains, terrain_spacing=[8., 8.],
            parts=PARTS, reference_sampling='fixed_q_forward_no_integration',
            newton_runtime=manifest['newton_runtime'], training_ready=False,
            physical_execution_verified=False, source_manifest=str(source), source_manifest_sha256=digest(source))
        (output/f'{split}.json').write_text(json.dumps(payload, indent=2)+'\n')
    (output/'preparation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def load_prepared(path):
    path = resolve_published_path(path)
    manifest = load_published_json(path)
    if manifest.get('schema') != SCHEMA or tuple(manifest.get('parts', ())) != PARTS:
        raise ValueError('Expected a prepared six-part PHP/Predictor contact manifest')
    require_current_newton_manifest(manifest, context=str(path))
    if manifest.get('reference_sampling') != 'fixed_q_forward_no_integration':
        raise ValueError('Unknown PHP contact reference sampling')
    evaluation_only = manifest.get('reference_origin') == 'out_of_distribution_native_reference'
    if evaluation_only and (manifest['split'] != 'evaluation' or manifest.get('training_ready') is not False):
        raise ValueError('Out-of-distribution references must remain evaluation-only')
    for terrain in manifest['terrains']:
        terrain['terrain_file'] = str(resolve_published_path(terrain['terrain_file'], base_dir=path.parent))
        if digest(terrain['terrain_file']) != terrain['terrain_sha256']:
            raise ValueError('Prepared PHP terrain hash mismatch')
    for row in manifest['motion_files']:
        for key, sha in (('motion_file', 'motion_sha256'), ('contact_file', 'contact_sha256')):
            row[key] = str(resolve_published_path(row[key], base_dir=path.parent))
            if digest(row[key]) != row[sha]:
                raise ValueError(f'Prepared PHP {key} hash mismatch')
        with np.load(row['contact_file'], allow_pickle=False) as z:
            meta = json.loads(z['metadata_json'].item())
            if (meta.get('schema') != SCHEMA or tuple(meta.get('parts', ())) != PARTS
                    or meta.get('source_sha256') != row['motion_sha256']
                    or meta.get('split') != manifest['split']
                    or (not evaluation_only and split_name(row['height_scale']) != manifest['split'])
                    or meta.get('reference_sampling') != 'fixed_q_forward_no_integration'):
                raise ValueError('Prepared PHP contact provenance/split mismatch')
            n = meta['frame_count']
            for key in ('reference_active', 'target_active'):
                if z[key].shape != (n, 6) or not np.isin(z[key], [0, 1]).all():
                    raise ValueError('Invalid PHP reference contact mask')
            if z['target_position_w'].shape != (n, 6, 3) or not np.isfinite(z['target_position_w']).all():
                raise ValueError('Invalid PHP contact target positions')
            if (z['target_frame'].shape != (n,) or z['segment_id'].shape != (n,)
                    or not np.issubdtype(z['target_frame'].dtype, np.integer)
                    or not np.issubdtype(z['segment_id'].dtype, np.integer)
                    or (z['target_frame'] < np.arange(n)).any() or (z['target_frame'] >= n).any()
                    or (z['segment_id'] < 0).any()):
                raise ValueError('Invalid PHP contact target clock')
            for key, sha in (('source_labels', 'labels_sha256'), ('events', 'events_sha256'),
                             ('evidence', 'evidence_sha256')):
                if digest(resolve_published_path(meta[key])) != meta[sha]:
                    raise ValueError(f'Prepared PHP original {key} evidence changed')
            if manifest.get('reference_origin') == 'matched_teacher_existing_extraction_standard':
                reference = resolve_published_path(row['teacher_reference_file'], base_dir=path.parent)
                if (digest(reference) != row['teacher_reference_sha256']
                        or meta.get('teacher_reference_sha256') != row['teacher_reference_sha256']):
                    raise ValueError('Original teacher task reference changed')
                with np.load(reference, allow_pickle=False) as original:
                    original_frames = len(original['joint_pos'])
                if ('teacher_frame' not in z or z['teacher_frame'].shape != (n,)
                        or not np.issubdtype(z['teacher_frame'].dtype, np.integer)
                        or (z['teacher_frame'] < 0).any() or (z['teacher_frame'] >= original_frames).any()
                        or (np.diff(z['teacher_frame']) < 0).any()):
                    raise ValueError('Original teacher task frame mapping is invalid')
                row['teacher_reference_file'] = str(reference)
    return manifest


def main():
    import tyro
    report = prepare(tyro.cli(Options))
    print(json.dumps({key: report[key] for key in ('motion_counts', 'interval_counts', 'training_ready')}))


if __name__ == '__main__':
    main()
