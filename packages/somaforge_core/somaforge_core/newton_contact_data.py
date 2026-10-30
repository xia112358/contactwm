"""Verified labels for the actual motion, not inherited source topology."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .newton_contacts import PARTS, SCHEMA
from .robot_assets import decode_robot_asset_json


def require_current_newton_provenance(semantics, *, context):
    """Reject labels made by an unrecorded or superseded collision runtime."""

    from .newton_runtime_compat import NEWTON_RUNTIME_ID

    runtime = semantics.get('provenance', {}).get('newton_runtime')
    actual = None if not isinstance(runtime, dict) else runtime.get('runtime_id')
    if actual != NEWTON_RUNTIME_ID:
        raise ValueError(
            f'{context}: Newton contact labels use runtime {actual!r}; '
            f'expected {NEWTON_RUNTIME_ID}. Relabel the unchanged motion with '
            'the pinned Newton main worker before training or segmentation'
        )
    return runtime


def require_current_newton_manifest(manifest, *, context):
    """Require a dataset-level declaration of the contact-runtime contract."""

    from .newton_runtime_compat import NEWTON_RUNTIME_ID

    runtime = manifest.get('newton_runtime')
    actual = None if not isinstance(runtime, dict) else runtime.get('runtime_id')
    if actual != NEWTON_RUNTIME_ID:
        raise ValueError(
            f'{context}: dataset uses Newton runtime {actual!r}; expected '
            f'{NEWTON_RUNTIME_ID}. Rebuild labels, events, and caches together'
        )
    return runtime


def load_raw_contact_labels(motion_path, labels_path=None, *, frame_count=None):
    motion_path = Path(motion_path)
    path = Path(labels_path) if labels_path else motion_path
    if frame_count is None:
        with np.load(motion_path, allow_pickle=False) as motion:
            frame_count = len(motion['joint_pos'])
    with np.load(path, allow_pickle=False) as z:
        decode_robot_asset_json(z['robot_asset_json'], context=str(path))
        if 'contact_semantics_json' not in z:
            raise ValueError(f'{path}: missing Newton contact semantics; relabel this actual motion')
        semantics = json.loads(z['contact_semantics_json'].item())
        if semantics.get('schema') != SCHEMA or semantics.get('sampling') != 'fixed_q_forward_no_integration':
            raise ValueError(f'{path}: contact and q are not certified same-state; use fixed-q Newton relabeling')
        if semantics.get('position') != 'robot geometry point from the exact Newton raw-to-solver contact mapping':
            raise ValueError(f'{path}: contact geometry-point semantics need regeneration')
        require_current_newton_provenance(semantics, context=str(path))
        if labels_path and z['source_sha256'].item() != hashlib.sha256(motion_path.read_bytes()).hexdigest():
            raise ValueError('Newton labels belong to a different motion; source topology cannot be inherited')
        if tuple(z['part_order'].astype(str)) != PARTS:
            raise ValueError('Expected explicit six-part Newton labels')
        result = {k: z[k].copy() for k in ('contact_part_mask', 'contact_position_w', 'contact_surface')}
        mask = result['contact_part_mask']; n = len(mask)
        if mask.shape != (n, 6) or (frame_count is not None and n != frame_count):
            raise ValueError('Contact timeline/part dimensions mismatch')
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('Contact truth must be binary, not interpolated probabilities')
        result['contact_part_mask'] = mask.astype(bool)
        if result['contact_position_w'].shape != (n, 6, 3) or not np.isfinite(result['contact_position_w']).all():
            raise ValueError('Invalid contact positions')
        if result['contact_surface'].shape != (n, 6) or np.any(result['contact_surface'][mask.astype(bool)] < 0):
            raise ValueError('Invalid active contact surfaces')
        if 'unallocated_active_contact' not in z or z['unallocated_active_contact'].any():
            raise ValueError('Unverified constraint allocation')
    return result


def load_contact_labels(motion_path, labels_path=None, *, frame_count=None, rebuild_task_cache=False,
                        consensus_evidence=None):
    """All training/segmentation clients receive the same task-label policy.

    Raw-only archives are valid inputs, not preprocessed task-label caches.
    Persisted processed labels must have an exact current contract and content.
    """
    from .contact_face_selection import process_face_contacts
    raw = load_raw_contact_labels(motion_path, labels_path, frame_count=frame_count)
    with np.load(motion_path, allow_pickle=False) as motion:
        fps = float(motion['fps'].reshape(-1)[0])
    with np.load(labels_path or motion_path, allow_pickle=False) as z:
        semantics = json.loads(z['contact_semantics_json'].item())
        pairs = json.loads(z['contact_pairs_json'].item())
        if len(pairs) != len(raw['contact_part_mask']):
            raise ValueError('Newton pair timeline mismatch')
        if consensus_evidence is not None and not rebuild_task_cache:
            raise ValueError('Changing consensus evidence requires explicit cache rebuild')
        if consensus_evidence is None and 'contact_consensus_evidence_json' in z:
            consensus_evidence = json.loads(z['contact_consensus_evidence_json'].item())
        if consensus_evidence is not None:
            from .contact_consensus_selection import process_with_consensus
            result = process_with_consensus(pairs, semantics['scene']['surface_catalog'],
                                            fps=fps, evidence=consensus_evidence)
        else:
            result = process_face_contacts(pairs, semantics['scene']['surface_catalog'], fps=fps)
        task_keys = ['task_'+k for k in raw]
        if not rebuild_task_cache and (any(k in z for k in task_keys) or 'contact_label_contract_json' in z):
            if not all(k in z for k in task_keys) or 'contact_label_contract_json' not in z:
                raise ValueError('Incomplete processed contact archive')
            if json.loads(z['contact_label_contract_json'].item()) != result['contact_label_contract']:
                raise ValueError('Task contact contract mismatch; explicitly materialize current face-filtered labels')
            for k in raw:
                if not np.array_equal(z['task_'+k], result[k]):
                    raise ValueError(f'Processed contact archive differs from shared processing: {k}')
            for key in ('contact_pairs', 'abnormal_contact_pairs'):
                cache_key = 'task_'+key+'_json'
                if cache_key not in z or json.loads(z[cache_key].item()) != result[key]:
                    raise ValueError(f'Processed pair archive mismatch: {key}')
    result.update({'raw_'+k:v for k,v in raw.items()})
    return result


def load_entry_contacts(entry, frame_count=None):
    return load_contact_labels(entry['motion_file'], entry.get('newton_contact_file'), frame_count=frame_count)


def touchdown_events(mask, *, stable_frames=3, merge_gap=4, surfaces=None):
    """Debounce events only; leave raw labels intact and require contact at cut."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or stable_frames < 1 or merge_gap < 0:
        raise ValueError('Invalid touchdown inputs')
    surfaces = np.zeros_like(mask,int) if surfaces is None else np.asarray(surfaces)
    if surfaces.shape != mask.shape:
        raise ValueError('Contact surface timeline mismatch')
    rises = mask[1:] & (~mask[:-1] | (surfaces[1:] != surfaces[:-1])); groups = []
    for f in np.flatnonzero(rises.any(-1))+1:
        if f+stable_frames > len(mask):
            continue
        parts = rises[f-1] & mask[f:f+stable_frames].all(0) & (surfaces[f:f+stable_frames] == surfaces[f]).all(0)
        if not parts.any():
            continue
        if (groups and f-groups[-1][0] <= merge_gap and mask[f, groups[-1][1]].all()
                and (surfaces[f,groups[-1][1]] == surfaces[groups[-1][0],groups[-1][1]]).all()):
            groups[-1] = (int(f), groups[-1][1] | parts)
        else:
            groups.append((int(f), parts.copy()))
    return groups
