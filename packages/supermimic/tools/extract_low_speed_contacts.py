"""Pool successful 10k native captures and export independently restartable contact slices."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import shutil
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trim_bounds(data, begin, end):
    """Only quiet, contact-stable boundary intervals are removable."""
    from contact_boundary_waits import boundary_wait_bounds
    start, stop, _ = boundary_wait_bounds(data, begin, end)
    return start, stop


def reference_action_bounds(task, length, lookahead, settling):
    """Reference waits tolerate balance jitter; retain anticipation and landing response."""
    if sha(task['csv']) != task['csv_sha256']:
        raise ValueError('Reference CSV changed')
    q = np.loadtxt(task['csv'], delimiter=',')
    v = np.diff(q, axis=0)*task['fps']
    # Quaternion sign changes are equivalent rotations, not angular motion.
    dot = np.abs((q[1:,3:7]*q[:-1,3:7]).sum(-1)).clip(0,1)
    angular = 2*np.arccos(dot)*task['fps']
    moving = ((np.linalg.norm(v[:,:3],axis=-1) >= .05)
              | (angular >= .15) | (np.sqrt((v[:,7:]**2).mean(-1)) >= .2))
    active = np.flatnonzero(moving)
    if not len(active):
        return 0, length
    first, last = int(active[0]), int(active[-1])+1
    start = max(0,first-lookahead-2) if first >= 10 else 0
    stop = min(length,last+settling+2) if length-last >= 10 else length
    return start, stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, nargs='+', required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    selection = json.loads(a.selection.read_text())
    pass
    from supermimic.contact_library import build_contact_clock
    from supermimic.contact_control import PARTS
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    from somaforge_core.collision_profiles import require_profile
    tasks = {t['source_task_index']: t for t in selection['tasks']}
    chosen = {}; attempts = []; checkpoint = None; contract = None
    for capture in a.captures:
        config = json.loads((capture/'config.json').read_text())
        current_checkpoint = str(Path(config['checkpoint']).resolve())
        if checkpoint is not None and checkpoint != current_checkpoint:
            raise ValueError('Different teacher checkpoints')
        checkpoint = current_checkpoint
        for path in sorted(capture.glob('motion_*/rollout.pt')):
            saved = torch.load(path, map_location='cpu', weights_only=False)
            meta, data = saved['metadata'], saved['data']; task = meta['motion']
            if task not in tasks:
                raise ValueError(f'Unselected motion: {task}')
            asset = json.loads(meta['robot_asset_json'])
            validate_g1_asset_metadata(asset, context=str(path))
            require_profile(asset, 'surface_v2', context=str(path))
            if meta.get('initialization_transfer') is not None or meta['task_manifest_sha256'] != selection['selection_source_manifest_sha256']:
                raise ValueError('Capture/teacher/selection contract mismatch')
            physical = {k: meta[k] for k in ('task_manifest_sha256', 'task_schema', 'robot_asset_json',
                'observation_schema', 'robot_collision_schema', 'force_contract', 'collision_profile',
                'collision_profile_sha256', 'body_contact_schema', 'body_contact_names', 'body_contact_scope', 'contact_response_contract', 'terrain_bounding_volume_schema', 'contact_timing', 'dt', 'state_quaternion') if k in meta}
            if contract is not None and physical != contract:
                raise ValueError('Mixed native contracts')
            contract = physical
            success = bool(meta['successful_rollout']) and not bool(data['term'].any()) and bool(torch.isfinite(data['s1']).all())
            attempts.append(dict(motion=task, name=tasks[task]['name'], source=str(path.resolve()),
                                 successful=success, frames=len(data['s']), expected=meta['expected_frames']))
            if success and task not in chosen:
                chosen[task] = (path, saved)
    missing = sorted(set(tasks)-set(chosen))
    if missing:
        raise ValueError(f'Complete capture coverage required; missing: {[tasks[t]["name"] for t in missing]}')
    chunks = []; spans = []; offset = 0
    for task, (path, saved) in sorted(chosen.items()):
        data = saved['data']; length = len(data['s'])
        if not bool((data['xi'][:, 33].long() == task).all()):
            raise ValueError('Terrain task identity changed inside capture')
        if not bool((data['f'][1:]-data['f'][:-1] == 1).all()):
            raise ValueError('Discontinuous reference clock')
        spans.append(dict(motion=task, name=tasks[task]['name'], group=tasks[task]['group'],
                          begin=offset, end=offset+length, source=str(path.resolve()), source_sha256=sha(path)))
        chunks.append(data); offset += length
    data = {k: torch.cat([d[k] for d in chunks]) for k in chunks[0]}
    metadata = dict(**contract, schema='supermimic_native_contact_library_v1', motion=None,
                    recording_spans=spans, source_checkpoint=checkpoint, source_checkpoint_sha256=sha(checkpoint),
                    selection_manifest=str(a.selection.resolve()), selection_manifest_sha256=sha(a.selection),
                    unit='whole contact-transfer episode', slice_contract='Use cases.pt explicit starts/stops/goals; original pooled clock is preserved')
    library = dict(data=data, metadata=metadata)
    clock = build_contact_clock(library, hold=1, role_aware=True)
    teacher_args = torch.load(checkpoint, map_location='cpu', weights_only=False)['args']
    hist = int(teacher_args['hist'])
    reference_bounds = {s['motion']: reference_action_bounds(tasks[s['motion']],s['end']-s['begin'],
                        int(teacher_args['goal_frames']),hist) for s in spans}
    cases = []; starts = []; stops = []; goals = []; history = []; preceding = []; motion_ids = []
    excluded_wait_cases = []
    span_by_motion = {s['motion']: s for s in spans}
    for index, (begin, end) in enumerate(zip(clock.bounds, clock.bounds[1:])):
        task = int(clock.segment_motion[index]); span = span_by_motion[task]
        start, stop = trim_bounds(data, begin, end)
        target = end-1; new = clock.target_masks[target] & ~clock.keep_masks[target]
        floor = span['begin']
        action_begin, action_end = reference_bounds[task]
        if stop <= start:
            excluded_wait_cases.append(dict(original_case=index+1, motion=task,
                source_frames=[begin-floor,end-floor],
                reason='entire interval has no contact conversion and negligible measured motion'))
            continue
        starts.append(start); stops.append(stop); goals.append(target); motion_ids.append(task)
        history.append(torch.arange(start-hist, start).clamp_min(floor)); preceding.append(max(start-1, floor))
        cases.append(dict(case=len(cases)+1, original_case=index+1, name=span['name'], motion=task, group=span['group'],
            kind='contact_transfer' if bool(new.any()) else 'support_motion',
            original_source_frames=[begin-floor, end-floor], source_frames=[start-floor, stop-floor],
            pooled_frames=[start, stop], original_goal_source_frame=target-floor,
            reference_start=int(data['f'][start]), reference_goal=int(data['f'][target])+1,
            frames=stop-start, duration_s=(stop-start)*metadata['dt'],
            removed_head=start-begin, removed_tail=end-stop,
            new_contacts=[PARTS[j] for j in new.nonzero().flatten().tolist()],
            maintained_contacts=[PARTS[j] for j in clock.keep_masks[target].nonzero().flatten().tolist()]))
    starts, stops, goals = map(torch.tensor, (starts, stops, goals))
    packed = dict(schema='native_contact_slices_v1', starts=starts, stops=stops, goals=goals,
                  motion_ids=torch.tensor(motion_ids), history_indices=torch.stack(history),
                  previous_contact_indices=torch.tensor(preceding),
                  target_masks=clock.target_masks[goals], keep_masks=clock.keep_masks[goals],
                  support_masks=clock.support_masks[goals], metadata=metadata)
    catalogue = dict(schema=packed['schema'], checkpoint=checkpoint, checkpoint_sha256=metadata['source_checkpoint_sha256'],
        motions=len(spans), case_count=len(cases), kinds=dict(Counter(c['kind'] for c in cases)),
        groups=dict(Counter(t['group'] for t in tasks.values())),
        removed_wait_frames=sum(c['removed_head']+c['removed_tail'] for c in cases),
        excluded_wait_cases=excluded_wait_cases, reference_action_bounds=reference_bounds,
        trim_contract=dict(linear_speed_max=.05, angular_speed_max=.15, joint_rms_speed_max=.2,
                           minimum_quiet_frames=10, boundary_context_frames=2, preserve_original_goals=True,
                           reference_waits=False, measured_motion_and_stable_contact_required=True,
                           preparation_frames=int(teacher_args['goal_frames']),
                           landing_response_frames=hist),
        native_restart_validation='pending', attempts=attempts, recording_spans=spans, cases=cases)
    a.output.mkdir(parents=True)
    torch.save(library, a.output/'rollout.pt'); torch.save(packed, a.output/'cases.pt')
    for name in ('solver-collision.mjb','solver-collision.mjb.json'):
        shutil.copyfile(a.captures[0]/name,a.output/name)
    (a.output/'catalogue.json').write_text(json.dumps(catalogue, indent=2)+'\n')
    print(json.dumps({k: catalogue[k] for k in ('motions','case_count','kinds','groups','removed_wait_frames')}, indent=2))


if __name__ == '__main__':
    main()
