"""Publish accepted motions as an independent, self-observed segmentation version."""
import copy
import inspect
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import trajectory_refinement as io
from . import extraction_standard as extraction
from .contact_dataset import contact_dataset_fingerprint
from .contact_events import event_contract, load_segment_rows
from .demonstration_events import materialize, SCHEMA as EVENT_SCHEMA
from .newton_contact_data import load_contact_labels, touchdown_events, require_current_newton_provenance
from .robot_assets import decode_robot_asset_json

SCHEMA = 'somaforge_self_observed_segmentation_release_v1'


def safe_name(value):
    value = str(value)
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('Version and motion names must be single path components')
    return value


def copy_verified(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    digest = io.sha256_file(source)
    if expected is not None and digest != expected:
        raise ValueError(f'Source hash differs: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if io.sha256_file(destination) != digest:
        raise ValueError(f'Archived file differs: {destination}')
    return io.fingerprint(destination)


def verify_acceptance(entry):
    """A manual continuity decision never waives contacts, clock, or provenance."""
    pin = entry['validation_result']
    io.verify_files([pin])
    result = io.read_json(pin['path'])
    if result.get('motion') != entry['motion_file'] or result.get('contacts') != entry['contacts']:
        raise ValueError('Acceptance belongs to a different trajectory')
    extraction.verify_validation(result)
    io.verify_files([dict(path=entry['motion_file'], sha256=entry['motion_sha256']),
                     dict(path=entry['contacts'], sha256=entry['contacts_sha256'])])
    if extraction.classification(result) == 'accepted':
        return None
    decision = entry.get('acceptance', {})
    if not decision.get('accepted') or decision.get('basis') != 'explicit_user_review':
        raise ValueError('Motion lacks automatic or explicit reviewed acceptance')
    review_pin = decision['review']
    io.verify_files([review_pin])
    review = io.read_json(review_pin['path'])
    evidence = review['evidence']
    if (review.get('decision') != 'accept' or review.get('authorized_by') != 'user'
            or not review.get('authorization_text')
            or review.get('source_case_key') != entry['source_case_key']
            or evidence['original_result'] != pin
            or evidence['motion']['sha256'] != entry['motion_sha256']
            or evidence['contacts']['sha256'] != entry['contacts_sha256']
            or not result['task_acceptance']['passed'] or not result['clock_validation']['passed']
            or not result['joint_limits']['passed'] or not result['aggregation_written']
            or result['optimization'].get('representative_fallback', True)
            or len(set(result['contributing_env_ids'])) < 3):
        raise ValueError('Reviewed acceptance is unbound or waives a required gate')
    io.verify_files(list(evidence.values()))
    return review_pin


def self_observed_segments(data, *, stable_frames=3, merge_gap=4):
    """Use shared Newton touchdown semantics; preserve prefix and terminal tail."""
    if type(stable_frames) is not int or stable_frames < 1 or type(merge_gap) is not int or merge_gap < 0:
        raise ValueError('Invalid touchdown stability or grouping configuration')
    n = len(data['contact_part_mask'])
    if n < 2:
        raise ValueError('Segmentation needs at least two frames')
    touchdowns = touchdown_events(data['contact_part_mask'], surfaces=data['contact_surface'],
                                 stable_frames=stable_frames, merge_gap=merge_gap)
    cuts = sorted({0, n-1, *(frame for frame, _ in touchdowns)})
    intervals = [dict(start_frame=a, end_frame=b) for a, b in zip(cuts, cuts[1:])]
    events, audit = materialize(data, intervals)
    selection = dict(schema='own_native_touchdown_complete_timeline_v1', stable_frames=stable_frames,
        merge_gap=merge_gap, minimum_segment_frames=None, original_reference_boundaries_used=False,
        endpoint_convention='inclusive; adjacent segments share one endpoint sample')
    for event in events:
        event.update(event_contract=event_contract(data['fps']), event_selection=selection)
    audit.update(action_clock_source='own actual Newton touchdown groups plus full prefix/tail',
                 selection=selection, complete_timeline=True)
    return events, audit


def _jsonl(path, rows):
    value = ''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value: raise ValueError(f'Existing segmentation differs: {path}')
    else:
        with path.open('x') as stream: stream.write(value)


def _publish_motion(entry, baseline, output, version, materialize_labels, stable_frames, merge_gap):
    name = safe_name(entry.get('motion_name', entry['source_case_key']))
    report_path = output/'provenance'/name/'release.json'
    if report_path.exists():
        saved = io.read_json(report_path)
        io.verify_files(saved['files'])
        return saved
    review = verify_acceptance(entry)
    motion = output/'motions'/name/'motion.npz'
    copy_verified(entry['motion_file'], motion, entry['motion_sha256'])
    labels = output/'contacts'/f'{name}.npz'
    if not labels.exists(): materialize_labels(motion, Path(entry['contacts']), labels)
    data = load_contact_labels(motion, labels)
    with np.load(motion, allow_pickle=False) as z:
        decode_robot_asset_json(z['robot_asset_json'], context=str(motion))
        fps = float(z['fps'].reshape(-1)[0]); count = len(z['joint_pos'])
    data['fps'] = fps
    with np.load(labels, allow_pickle=False) as z:
        semantics = json.loads(z['contact_semantics_json'].item())
    runtime = require_current_newton_provenance(semantics, context=str(labels))
    # Materialization may update task arrays and touchdown caches. Preserve all
    # raw solver evidence byte-for-byte at the array level, not just the mask.
    raw_names = ('contact_part_mask', 'contact_surface', 'contact_position_w',
        'contact_pairs_json', 'unallocated_active_contact', 'contact_semantics_json', 'robot_asset_json')
    with np.load(entry['contacts'], allow_pickle=False) as original, np.load(labels, allow_pickle=False) as archived:
        for field in raw_names:
            equal_nan = np.issubdtype(original[field].dtype, np.inexact)
            if not np.array_equal(original[field], archived[field], equal_nan=equal_nan):
                raise ValueError(f'Raw Newton evidence changed: {field}')
    events, audit = self_observed_segments(data, stable_frames=stable_frames, merge_gap=merge_gap)
    events_path = output/'events'/f'{name}.jsonl'
    for index, event in enumerate(events):
        event.update(segment_id=f'{name}_event_{index:04d}', motion_id=entry['motion_id'],
            motion_name=name, source_path=str(motion), contact_source_path=str(labels),
            motion_version_id=version)
    _jsonl(events_path, events)
    scene_path = output/'scenes'/f'{name}.json'
    io.write_json(scene_path, semantics['scene'], immutable=True)
    acceptance = copy.deepcopy(entry.get('acceptance', dict(accepted=True, automated_passed=True,
                                                          basis='automated_extraction_checks')))
    files = [io.fingerprint(p) for p in (motion, labels, events_path, scene_path)]
    for key in ('validation_result',):
        pin = entry[key]
        files.append(copy_verified(pin['path'], report_path.parent/'source_result.json', pin['sha256']))
    continuity_pin = io.fingerprint(entry['continuity']['path'])
    files.append(copy_verified(continuity_pin['path'], report_path.parent/'source_continuity.json', continuity_pin['sha256']))
    if review is not None:
        archived_review = copy_verified(review['path'], report_path.parent/'acceptance_review.json', review['sha256'])
        acceptance.update(source_review=review, review=archived_review)
        files.append(archived_review)
    row = dict(motion_id=entry['motion_id'], motion_name=name, source_case_key=entry['source_case_key'],
        motion_file=str(motion), motion_sha256=io.sha256_file(motion),
        newton_contact_file=str(labels), contacts=str(labels), contacts_sha256=io.sha256_file(labels),
        event_segments_file=str(events_path), event_segments_sha256=io.sha256_file(events_path),
        event_selection=audit['selection'], scene_catalog_file=str(scene_path),
        terrain_id=entry['terrain_id'], weight=entry.get('weight', 1.),
        motion_version_id=version, acceptance=acceptance,
        kinematics_backend=entry.get('kinematics_backend', 'newton_direct_fk'),
        kinematics_schema=entry.get('kinematics_schema', 'somaforge_canonical_motion_v1'),
        velocity_derivation=entry.get('velocity_derivation', 'pose_finite_difference'),
        training_ready=False)
    load_segment_rows(row, data, fps)
    report = dict(schema=SCHEMA, motion=row, frame_count=count, fps=fps, event_count=len(events),
        newton_runtime=runtime, contact_contract=data['contact_label_contract'], audit=audit,
        baseline_motion_id=baseline['motion_id'], baseline_files=baseline.get('files', {}),
        source_motion=io.fingerprint(entry['motion_file']), source_labels=io.fingerprint(entry['contacts']),
        raw_newton_evidence_preserved=True, poses_changed=False, files=files)
    io.write_json(report_path, report, immutable=True)
    return report


def publish(source_manifest, dataset_root, version, display_name, baseline_index, *,
            materialize_labels, workers=4, stable_frames=3, merge_gap=4, resume=False):
    source_manifest, dataset_root, baseline_index = map(lambda p: Path(p).resolve(),
        (source_manifest, dataset_root, baseline_index))
    version = safe_name(version); output = dataset_root/'versions'/version
    if type(workers) is not int or workers < 1: raise ValueError('workers must be a positive integer')
    source, baseline = io.read_json(source_manifest), io.read_json(baseline_index)
    entries = source['motion_files']
    names = [str(e.get('motion_name', e['source_case_key'])) for e in entries]
    baseline_rows = {str(e['motion_id']): e for e in baseline['motion_files']}
    if (not entries or len(set(names)) != len(names)
            or len(baseline_rows) != len(baseline['motion_files']) or set(names)-set(baseline_rows)):
        raise ValueError('Motions require unique identities and a matching baseline entry')
    if (not source.get('complete') or not (source.get('all_accepted') or source.get('all_passed'))
            or set(source.get('expected_case_keys', [])) != {e['source_case_key'] for e in entries}):
        raise ValueError('Source collection is incomplete or not entirely accepted')
    source_terrains = {str(t['terrain_id']): t for t in source['terrains']}
    baseline_terrains = {str(t['terrain_id']): t for t in baseline['terrains']}
    for entry, name in zip(entries, names):
        current_terrain = Path(source_terrains[str(entry['terrain_id'])]['terrain_file'])
        old_terrain = dataset_root / baseline_terrains[str(baseline_rows[name]['terrain_id'])]['file']
        if io.sha256_file(current_terrain) != io.sha256_file(old_terrain):
            raise ValueError(f'Baseline terrain differs for {name}')
    implementation = [io.fingerprint(p) for p in (
        Path(__file__), Path(__file__).with_name('demonstration_events.py'),
        Path(__file__).with_name('contact_events.py'), Path(__file__).with_name('newton_contact_data.py'),
        Path(__file__).with_name('contact_face_selection.py'), Path(__file__).with_name('robot_assets.py'),
        Path(__file__).with_name('contact_labels.py'), inspect.getsourcefile(materialize_labels))]
    receipt = dict(schema=SCHEMA, version=version, display_name=display_name,
        source_manifest=io.fingerprint(source_manifest), baseline_index=io.fingerprint(baseline_index),
        workers=workers, executor='spawned_cpu_processes', stable_frames=stable_frames,
        merge_gap=merge_gap, implementation=implementation)
    if output.exists() and not resume: raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    io.write_json(output/'publication_plan.json', receipt, immutable=True)
    copy_verified(source_manifest, output/'provenance/source_manifest.json')
    copy_verified(baseline_index, output/'provenance/baseline_index.json')
    terrains = []
    for terrain in source['terrains']:
        path = output/'terrains'/f"{safe_name(terrain['terrain_id'])}.obj"
        pinned = copy_verified(terrain['terrain_file'], path, terrain.get('terrain_sha256'))
        terrains.append(dict(terrain, terrain_file=str(path), terrain_sha256=pinned['sha256']))
    results = {}
    # Contact pair processing is Python-heavy. Separate CPU processes provide
    # actual parallelism without initializing or sharing a simulator/CUDA state.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(_publish_motion, entry, baseline_rows[name], output, version,
            materialize_labels, stable_frames, merge_gap): name for entry, name in zip(entries, names)}
        for future in as_completed(futures):
            report = future.result(); results[futures[future]] = report
            progress = dict(processed=len(results), total=len(entries),
                segments=sum(r['event_count'] for r in results.values()), complete=False)
            io.write_json(output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    ordered = [results[name] for name in names]
    runtimes = [r['newton_runtime'] for r in ordered]
    if any(runtime != runtimes[0] for runtime in runtimes): raise ValueError('Mixed Newton runtimes')
    rows = [r['motion'] for r in ordered]
    summary = dict(schema=SCHEMA, version=version, display_name=display_name, motions=len(rows),
        segments=sum(r['event_count'] for r in ordered), complete=True, all_accepted=True,
        automatically_accepted=sum(r['acceptance']['automated_passed'] for r in rows),
        accepted_by_user_review=sum(not r['acceptance']['automated_passed'] for r in rows),
        full_timeline_preserved=True, original_boundaries_used=False, training_started=False,
        training_ready=False)
    io.write_json(output/'summary.json', summary, immutable=True)
    manifest = dict(io.manifest_header('newton_direct_fk'), motion_files=rows, terrains=terrains,
        dataset_version=version, display_name=display_name, variant='dynamics_corrected',
        newton_runtime=runtimes[0], event_contract=dict(schema=EVENT_SCHEMA,
            boundaries='own actual Newton touchdown events', full_prefix_and_tail=True),
        acceptance_summary=summary, complete=True, all_accepted=True, training_ready=False,
        contact_dataset_fingerprint=None,
        provenance=dict(source_manifest=io.fingerprint(output/'provenance/source_manifest.json'),
            baseline_index=io.fingerprint(output/'provenance/baseline_index.json'),
            original_paths='historical provenance only; operational motion/labels/events/terrain files are version-local'))
    # Compute identity against the final manifest structure before publishing it.
    pending = output/'manifest.pending.json'
    io.write_json(pending, manifest)
    manifest['contact_dataset_fingerprint'] = contact_dataset_fingerprint(pending)
    mapping = [dict(motion_name=name, source_case_key=row['source_case_key'],
        baseline_motion_id=baseline_rows[name]['motion_id'], baseline_files=baseline_rows[name].get('files', {}),
        corrected_motion=row['motion_file'], corrected_labels=row['newton_contact_file'],
        corrected_events=row['event_segments_file'], corrected_event_count=report['event_count'])
        for name, row, report in zip(names, rows, ordered)]
    io.write_json(output/'comparison.json', dict(baseline=receipt['baseline_index'],
        corrected_version=version, correspondence='whole motion identity; no inherited frame boundaries',
        motions=mapping), immutable=True)
    archive = dict(schema='somaforge_archived_dataset_index_v1', display_name=display_name,
        variant='dynamics_corrected', manifest=str((output/'manifest.json').relative_to(dataset_root)),
        paths_relative_to='dataset_root',
        motion_files=[dict(motion_id=name, source_case_key=row['source_case_key'], terrain_id=row['terrain_id'],
            files={key:str(Path(row[key]).relative_to(dataset_root)) for key in
                   ('motion_file', 'newton_contact_file', 'event_segments_file')},
            acceptance=row['acceptance']) for name, row in zip(names, rows)],
        terrains=[dict(terrain_id=t['terrain_id'], file=str(Path(t['terrain_file']).relative_to(dataset_root)),
                       terrain_sha256=t['terrain_sha256']) for t in terrains], summary=summary)
    io.write_json(output/'index.json', archive, immutable=True)
    # Nothing in the catalog references this version until all cases validate.
    io.verify_files(implementation)
    io.write_json(output/'manifest.json', manifest, immutable=True)
    _register(dataset_root, output, baseline_index, version, display_name, summary)
    io.write_json(output/'progress.json', dict(summary, processed=len(rows), total=len(rows)))
    return summary


def _register(root, output, baseline_index, version, display_name, summary):
    import fcntl
    import uuid
    with (root/'.catalog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'catalog.json'; catalog = io.read_json(path)
        backup = output/'provenance/catalog_before_registration.json'
        if not backup.exists(): copy_verified(path, backup)
        baseline_name = baseline_index.parent.name
        row = dict(name=version, display_name=display_name, variant='dynamics_corrected',
            index=str((output/'index.json').relative_to(root)),
            manifest=str((output/'manifest.json').relative_to(root)),
            motions=summary['motions'], segments=summary['segments'], baseline_version=baseline_name,
            current_experiment_source=False, training_ready=False, storage='version-local independent copies')
        existing = [v for v in catalog['versions'] if v['name'] == version]
        if existing and existing != [row]: raise ValueError('Catalog version already differs')
        if not existing: catalog['versions'].append(row)
        for previous in catalog['versions']:
            if previous['name'] == baseline_name:
                previous.setdefault('display_name', '直接生成切分版')
                previous.setdefault('variant', 'generated_direct')
        catalog['policy_execution_dataset_status'] = dict(version=version, display_name=display_name,
            motions=summary['motions'], segmentation_published=True, training_manifest_changed=False)
        temporary = root/f'.catalog.pending.{uuid.uuid4().hex}.json'
        io.write_json(temporary, catalog)
        temporary.replace(path)
