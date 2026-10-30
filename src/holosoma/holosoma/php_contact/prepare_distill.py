"""Adapt verified existing rollout extraction to segmented distillation inputs.

This is a downstream format adapter. Collection, event consensus, admission and
action cuts are supplied by the shared native-contact segmentation module.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

from holosoma.php_contact.dataset import SCHEMA, digest, interval_targets
from somaforge_core import extraction_standard
from somaforge_core.refined_contact_dataset import self_observed_segments
from somaforge_core.contact_action_segments import action_segments
from somaforge_core.newton_contact_data import load_contact_labels
from somaforge_core.newton_contacts import PARTS
from somaforge_core.published_data import load_published_json, resolve_published_path
from somaforge_core.trajectory_refinement import verify_files


@dataclass(frozen=True)
class Options:
    source_manifest: Path
    teacher_checkpoint: Path
    output_dir: Path
    segmentation: Literal['action', 'touchdown'] = 'action'
    minimum_segment_frames: int = 20


def selected_clock(maps, source_env_ids, selected_env, original_frames):
    row = np.flatnonzero(np.asarray(source_env_ids) == selected_env)
    if len(row) != 1:
        raise ValueError('Extraction representative has no unique retained source clock')
    clock = np.asarray(maps)[row[0]]
    if (clock.ndim != 1 or not np.isfinite(clock).all() or (np.diff(clock) < 0).any()
            or clock.min() < 0 or clock.max() > original_frames-1):
        raise ValueError('Invalid extraction-to-original-reference clock')
    return np.rint(clock).astype(np.int64)


def prepare(options):
    import torch
    from holosoma.utils.eval_utils import _experiment_config_from_serialized
    source_path = resolve_published_path(options.source_manifest)
    source = extraction_standard.verified_manifest(source_path)
    if not source['motion_files']:
        raise ValueError('No trajectories passed the existing extraction standard; nothing admitted')
    checkpoint = resolve_published_path(options.teacher_checkpoint)
    receipt = load_published_json(source['reaggregation_plan']['path'])
    collection = Path(receipt['source_collection'])
    collection_receipt = load_published_json(collection/'collection_plan.json')
    if digest(checkpoint) != collection_receipt['checkpoint']['sha256']:
        raise ValueError('Selected teacher differs from the actual rollout collection policy')
    config = _experiment_config_from_serialized(torch.load(checkpoint, map_location='cpu',
        weights_only=False, mmap=True)['experiment_config'])
    original_manifest = resolve_published_path(config.command.setup_terms['motion_command'].params['motion_config'].motion_manifest)
    original = load_published_json(original_manifest)
    trained = {str(resolve_published_path(row['motion_file'], base_dir=original_manifest.parent)): row
               for row in original['motion_files']}
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'contacts').mkdir()
    (output/'events').mkdir()
    rows, audit = [], []
    runtime = None
    for index, row in enumerate(source['motion_files']):
        reference = resolve_published_path(row['reference_motion_file'])
        if str(reference) not in trained:
            raise ValueError('Extracted task is absent from the selected teacher training manifest')
        motion, labels_path = Path(row['motion_file']), Path(row['contacts'])
        labels = load_contact_labels(motion, labels_path)
        with np.load(motion, allow_pickle=False) as z:
            fps = float(z['fps']); n = len(z['joint_pos'])
        with np.load(reference, allow_pickle=False) as z:
            original_frames = len(z['joint_pos'])
        if not np.isclose(fps, 50.):
            raise ValueError('Segment distillation requires 50 Hz references')
        labels['fps'] = fps
        if options.segmentation == 'action':
            events, cut_audit = action_segments(labels,
                minimum_segment_frames=options.minimum_segment_frames)
        elif options.segmentation == 'touchdown':
            events, cut_audit = self_observed_segments(labels)
        else:
            raise ValueError('Unknown segmentation rule')
        arrays = interval_targets(labels, events)
        result_path = Path(row['validation_result']['path'])
        result = load_published_json(result_path)
        if result.get('strategies'):
            matches = [load_published_json(child['result_path']) for child in result['strategies']]
            matches = [r for r in matches if r.get('motion') == str(motion)]
            if len(matches) != 1:
                raise ValueError('Ambiguous extraction strategy for the selected trajectory')
            result = matches[0]
        alignment = motion.parent/'alignment.npz'
        pins = [pin for pin in result['artifacts'] if Path(pin['path']).resolve() == alignment.resolve()]
        if len(pins) != 1:
            raise ValueError('Unpinned original-reference alignment')
        verify_files(pins)
        with np.load(alignment, allow_pickle=False) as z:
            arrays['teacher_frame'] = selected_clock(z['source_frame_maps'], z['source_env_ids'],
                result['pose_initialization']['selected_env_id'], original_frames)
        if len(arrays['teacher_frame']) != n:
            raise ValueError('Teacher reference clock and extracted trajectory lengths differ')
        with np.load(labels_path, allow_pickle=False) as z:
            semantics = json.loads(z['contact_semantics_json'].item())
        runtime = semantics['provenance']['newton_runtime']
        event_file = output/'events'/f'{index:04d}.jsonl'
        event_file.write_text(''.join(json.dumps(e)+'\n' for e in events))
        meta = dict(schema=SCHEMA, parts=PARTS, source_motion=str(motion), source_sha256=digest(motion),
            source_labels=str(labels_path), labels_sha256=digest(labels_path),
            events=str(event_file), events_sha256=digest(event_file),
            evidence=str(result_path), evidence_sha256=digest(result_path),
            split='train', height_scale=1., interval_count=len(events), frame_count=n,
            reference_sampling=semantics['sampling'], source_newton_runtime=runtime,
            physical_execution_verified=False, support_loads='not inherited onto fitted kinematics',
            teacher_reference_file=str(reference), teacher_reference_sha256=digest(reference),
            teacher_clock='retained representative source execution map; nearest original 50Hz control frame',
            alignment=pins[0], original_task_matched=True)
        meta['segmentation'] = cut_audit['selection']
        contact = output/'contacts'/f'{index:04d}.npz'
        np.savez_compressed(contact, **arrays, metadata_json=np.asarray(json.dumps(meta)))
        rows.append(dict(motion_id=index, motion_file=str(motion), motion_sha256=digest(motion),
            terrain_id=row['terrain_id'], contact_file=str(contact), contact_sha256=digest(contact),
            height_scale=1., teacher_reference_file=str(reference), teacher_reference_sha256=digest(reference),
            source_case_key=row['source_case_key']))
        audit.append(dict(source_case_key=row['source_case_key'], intervals=len(events),
            cuts=cut_audit, teacher_reference=str(reference), alignment=pins[0]))
    teacher = dict(path=str(checkpoint), sha256=digest(checkpoint), training_manifest=str(original_manifest),
                   training_manifest_sha256=digest(original_manifest))
    manifest = dict(schema=SCHEMA, split='train', reference_origin='matched_teacher_existing_extraction_standard',
        reference_sampling='fixed_q_forward_no_integration', parts=PARTS, motion_files=rows,
        terrains=source['terrains'], terrain_spacing=original.get('terrain_spacing', [8.,8.]),
        newton_runtime=runtime, source_manifest=str(source_path), source_manifest_sha256=digest(source_path),
        teacher=teacher, training_ready=False, physical_execution_verified=False,
        segmentation=audit[0]['cuts']['selection'],
        heldout_split='not configured; teacher-task collection only')
    (output/'train.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'teachers.json').write_text(json.dumps({'teachers':[dict(checkpoint=str(checkpoint),
        manifest=str(output/'train.json'))]}, indent=2)+'\n')
    report = dict(source_manifest=str(source_path), teacher=teacher, accepted_motions=len(rows),
        segmentation=audit[0]['cuts']['selection'],
        intervals=sum(r['intervals'] for r in audit), motions=audit, training_started=False,
        extraction_summary=source['validation_summary'], reference_role='verified fitted references; original recordings retained separately')
    (output/'preparation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    import tyro
    report = prepare(tyro.cli(Options))
    print(json.dumps({key:report[key] for key in ('accepted_motions','intervals','training_started')}))


if __name__ == '__main__':
    main()
