"""Materialize a shorter event source using existing heel/toe Newton anchors.

The source ContactLayer keeps shape-level heel and toe evidence. This script
only materializes a new motion/label/event version; it never edits the source
or publishes predictor training data.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.newton_contact_data import load_contact_labels
from somaforge_core.robot_assets import decode_robot_asset_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor_shape(anchor):
    match = re.search(r'_shape(\d+)_face(\d+)_', anchor['anchor_id'])
    if match is None:
        raise ValueError('Anchor lacks audited Newton shape and face IDs')
    return int(match[1]), int(match[2])


def patch_evidence(pairs, catalog, anchors, frame_count):
    """Reuse original body names, verifying every patch frame against native pairs."""
    order = ('left_toe', 'left_heel', 'right_toe', 'right_heel')
    selected = select_contact_pairs(pairs, catalog)['contact_pairs']
    expected = np.zeros((frame_count, len(order)), bool)
    actual = np.zeros_like(expected)
    surfaces = np.full(expected.shape, -1, np.int64)
    shape_roles = {}
    for anchor in anchors:
        body = anchor.get('body')
        if body not in order:
            continue
        shape, surface = anchor_shape(anchor)
        key = (shape, surface)
        if key in shape_roles and shape_roles[key] != body:
            raise ValueError('One Newton shape/face maps to multiple contact bodies')
        shape_roles[key] = body
        a, b = int(anchor['start_frame']), int(anchor['end_frame'])
        if not 0 <= a < b <= frame_count:
            raise ValueError('Patch anchor outside the source motion')
        expected[a:b, order.index(body)] = True
    for frame, row in enumerate(selected):
        for pair in row:
            role = shape_roles.get((int(pair['robot_shape']), int(pair['surface'])))
            if role is None:
                continue
            index = order.index(role)
            actual[frame, index] = True
            previous = surfaces[frame, index]
            if previous >= 0 and previous != pair['surface']:
                raise ValueError('Patch occupies different primary surfaces in one frame')
            surfaces[frame, index] = int(pair['surface'])
    if not np.array_equal(expected, actual):
        bad = np.argwhere(expected != actual)
        raise ValueError(f'ContactLayer/native patch evidence mismatch: {bad[:8].tolist()}')
    return order, actual, surfaces, shape_roles


def completion_window(mask, surface, order, event, requirement, stable_frames):
    a, b = int(event['start_frame']), int(event['end_frame'])
    members = tuple(requirement['members'])
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError('Multi-contact completion requires distinct bodies')
    ids = [order.index(name) for name in members]
    face = int(requirement['surface'])
    concurrent = (mask[a:b+1, ids] & (surface[a:b+1, ids] == face)).all(axis=-1)
    edges = np.diff(np.r_[False, concurrent, False].astype(int))
    runs = [(a+int(x), a+int(y)-1) for x, y in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    stable = [(x, y) for x, y in runs if y-x+1 >= stable_frames]
    if not stable:
        raise ValueError('No stable native multi-contact window in terminal event')
    # Preserve the complete first establishment window, then stop before the
    # next patch release or unrelated terminal adjustment.
    return stable[0], runs


def materialize(source, labels, events, anchors_file, phases_file, output, *, stable_frames=3):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already exists: {output}')
    with np.load(source, allow_pickle=False) as z:
        motion = {k: z[k].copy() for k in z.files}
    decode_robot_asset_json(motion['robot_asset_json'], context=str(source))
    frame_count = len(motion['joint_pos'])
    load_contact_labels(source, labels)
    with np.load(labels, allow_pickle=False) as z:
        native = {k: z[k].copy() for k in z.files}
    pairs = json.loads(native['contact_pairs_json'].item())
    semantics = json.loads(native['contact_semantics_json'].item())
    anchors = [json.loads(line) for line in anchors_file.read_text().splitlines() if line.strip()]
    order, patch_mask, patch_surface, shape_roles = patch_evidence(
        pairs, semantics['scene']['surface_catalog'], anchors, frame_count)
    rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    terminal = rows[-1]
    if terminal.get('terminal_event') != 'terminal_tail' or terminal.get('touchdown_events'):
        raise ValueError('Last event is not an empty-contact terminal tail')
    required = dict(members=('left_toe', 'left_heel'), surface=0)
    (first, last), windows = completion_window(
        patch_mask, patch_surface, order, terminal, required, stable_frames)
    if last >= frame_count-1:
        raise ValueError('No terminal adjustment remains to crop')
    keep = last+1
    output.mkdir(parents=True, exist_ok=True)
    target = output/'source'
    (target/'keyframe_segments').mkdir(parents=True, exist_ok=True)
    for key, value in motion.items():
        if value.ndim and value.shape[0] == frame_count:
            motion[key] = value[:keep].copy()
    motion['event_source_crop_json'] = np.asarray(json.dumps(dict(
        schema='native_multi_contact_terminal_crop_v1', source=str(source.resolve()),
        source_sha256=sha(source), retained=[0,last], removed=[keep,frame_count-1])))
    motion_path = target/'motion.npz'
    np.savez_compressed(motion_path, **motion)
    for key, value in native.items():
        if value.ndim and value.shape[0] == frame_count:
            native[key] = value[:keep].copy()
    for key in ('contact_pairs_json', 'task_contact_pairs_json', 'task_abnormal_contact_pairs_json'):
        if key in native:
            native[key] = np.asarray(json.dumps(json.loads(native[key].item())[:keep]))
    if 'touchdown_frames' in native:
        valid = native['touchdown_frames'] < keep
        native['touchdown_frames'] = native['touchdown_frames'][valid]
        native['touchdown_parts'] = native['touchdown_parts'][valid]
    native['source_path'] = np.asarray(str(motion_path.resolve()))
    native['source_sha256'] = np.asarray(sha(motion_path))
    semantics['derived_prefix'] = dict(parent_labels=str(labels.resolve()),
                                       parent_sha256=sha(labels), retained=[0,last])
    native['contact_semantics_json'] = np.asarray(json.dumps(semantics))
    native['patch_order'] = np.asarray(order)
    native['patch_contact_mask'] = patch_mask[:keep]
    native['patch_contact_surface'] = patch_surface[:keep]
    native['patch_contact_contract_json'] = np.asarray(json.dumps(dict(
        schema='native_shape_face_contact_patches_v1',
        original_layer=str(anchors_file.resolve()), original_layer_sha256=sha(anchors_file),
        rule='existing Motion Edit toe/heel anchors verified framewise against eligible Newton pairs',
        shape_roles=[dict(shape=shape, surface=surface, body=body)
                     for (shape, surface), body in sorted(shape_roles.items())])))
    labels_path = target/'effective_contacts.npz'
    np.savez_compressed(labels_path, **native)
    load_contact_labels(motion_path, labels_path)
    if not (patch_mask[first:last+1, [0,1]].all() and
            (patch_surface[first:last+1, [0,1]] == required['surface']).all()):
        raise ValueError('Completion window lost dual native contact')
    terminal.update(end_frame=last, duration_frames=last-int(terminal['start_frame']),
                    duration_s=(last-int(terminal['start_frame']))/float(motion['fps']),
                    terminal_event='multi_contact_completion',
                    completion_patches=dict(required, stable_frames=stable_frames,
                                            native_interval=[first,last], source='existing_contact_layer_verified_by_newton'))
    for row in rows:
        row['source_path'] = str(motion_path.resolve())
        row['contact_source_path'] = str(labels_path.resolve())
    event_path = target/'keyframe_segments/atoms.jsonl'
    event_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    phases = json.loads(phases_file.read_text())
    for phase in phases:
        if phase['event_id'] == terminal['segment_id']:
            phase['end'] = last
    (target/'phase_tasks.json').write_text(json.dumps(phases, indent=2))
    report = dict(schema='native_multi_contact_terminal_crop_v1', training_ready=False,
                  source_frames=frame_count, retained_frames=keep, cut_after=last,
                  dual_contact_window=[first,last], all_terminal_dual_windows=windows,
                  required_patches=required, stable_frames=stable_frames,
                  source_motion_sha256=sha(source), new_motion_sha256=sha(motion_path),
                  source_labels_sha256=sha(labels), new_labels_sha256=sha(labels_path),
                  source_events_sha256=sha(events), new_events_sha256=sha(event_path),
                  contact_layer_sha256=sha(anchors_file),
                  interpretation='Toe and heel are separate native contacts; whole-foot activation alone cannot complete this event.',
                  pending='Regenerate cropped ContactLayer, edit plans, augmentation references, and training labels')
    (output/'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'labels', 'events', 'anchors', 'phases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.labels, args.events, args.anchors, args.phases, args.output)
