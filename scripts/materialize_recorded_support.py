"""Bind actual original Newton solve loads to an unchanged native trajectory.

No pose correction, event warp, interpolation, force imitation or simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from somaforge_core.newton_contact_data import require_current_newton_manifest
from somaforge_core.newton_support import SUPPORT_SCHEMA, SUPPORT_FIELDS, observed_support
from somaforge_core.robot_assets import decode_robot_asset_json, validate_g1_asset_metadata
from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA


def materialize(motion_path, original_state_path, output):
    with np.load(original_state_path, allow_pickle=False) as state:
        provenance = json.loads(state['rollout_provenance_json'].item())
        original_q = state['joint_pos'].copy()
        original_names = state['joint_names'].astype(str)
        decode_robot_asset_json(state['robot_asset_json'], context=str(original_state_path))
    recording = Path(provenance['recording'])
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    if digest != provenance['sha256']:
        raise ValueError('Original recording has changed')
    with np.load(motion_path, allow_pickle=False) as motion:
        decode_robot_asset_json(motion['robot_asset_json'], context=str(motion_path))
        q = motion['joint_pos'].copy()
        if not np.array_equal(motion['joint_names'].astype(str), original_names):
            raise ValueError('Motion joint order differs from original execution')
        fps = float(motion['fps'].item())
    if q.shape != original_q.shape or not np.allclose(q, original_q, atol=1.e-6, rtol=0):
        raise ValueError('Edited/warped poses cannot inherit original support loads')
    env = int(provenance['env'])
    frames = np.asarray(provenance['native_frame_indices'], int)
    if len(frames) != len(q) or np.any(frames < 0) or np.any(np.diff(frames) != 1):
        raise ValueError('Only contiguous native physical execution is supported')
    with np.load(recording, allow_pickle=False) as z:
        meta = json.loads(z['_metadata_json'].item())
        require_current_newton_manifest(meta, context=str(recording))
        validate_g1_asset_metadata(meta.get('robot_asset'), context=str(recording))
        if meta.get('solver_support_semantics', {}).get('schema') != SUPPORT_SCHEMA:
            raise ValueError('Missing same-solve source force semantics')
        if not all('solver_contact_' + key in z for key in SUPPORT_FIELDS):
            raise ValueError('Missing original solver force/velocity channels')
        if not np.isclose(fps, 1/meta['dt'], atol=1.e-6, rtol=0):
            raise ValueError('Resampling changes the native solve clock')
        saved_env = int(meta['env_id'])
        def choose(array):
            return array[frames, env] if saved_env < 0 else array[frames]
        if saved_env >= 0 and saved_env != env:
            raise ValueError('Requested environment not recorded')
        if not np.array_equal(np.asarray(meta['dof_names']), original_names):
            raise ValueError('Recording joint order differs from motion')
        position = choose(z['root_pos']); joints = choose(z['dof_pos'])
        quaternion = choose(z['root_quat_xyzw'])[:, [3,0,1,2]]
        quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
        target_quaternion = q[:,3:7] / np.linalg.norm(q[:,3:7], axis=-1, keepdims=True)
        orientation_error = np.minimum(np.linalg.norm(target_quaternion-quaternion,axis=-1),
                                       np.linalg.norm(target_quaternion+quaternion,axis=-1)).max()
        if (not np.allclose(q[:,:3], position, atol=1.e-6, rtol=0)
                or not np.allclose(q[:,7:], joints, atol=1.e-6, rtol=0) or orientation_error > 1.e-6):
            raise ValueError('Trajectory is not the original recorded physical poses')
        counts = z['solver_contact_count']
        if len(frames) == 0 or frames[-1] >= len(counts):
            raise ValueError('Native frame mapping outside recording')
        channels = {key[len('solver_contact_'):]: z[key] for key in z.files
                    if key.startswith('solver_contact_') and key != 'solver_contact_count'}
        tables = {}
        for index, frame in enumerate(frames):
            count = int(counts[frame])
            belongs = channels['worldid'][frame,:count] == env
            snapshot = {key:value[frame,:count][belongs] for key,value in channels.items()}
            snapshot['count'] = int(belongs.sum())
            result = observed_support(snapshot, meta['newton_scene_binding'], meta['newton_body_labels'], meta['num_envs'])
            for key,value in result.items():
                tables.setdefault(key, []).append(value[env])
            if index % 200 == 0:
                print(f'actual support {index+1}/{len(frames)}', flush=True)
    arrays = {key:np.asarray(value) for key,value in tables.items()}
    arrays['native_record_frames'] = frames
    arrays['support_assessment_json'] = np.asarray(json.dumps(dict(
        schema=SUPPORT_ASSESSMENT_SCHEMA, status='observed',
        motion_sha256=hashlib.sha256(Path(motion_path).read_bytes()).hexdigest(),
        recording=dict(path=str(recording.resolve()),sha256=digest),
        robot_asset=meta['robot_asset'], newton_runtime=meta['newton_runtime'],
        scene_fingerprint=meta['newton_scene_binding']['model_fingerprint'], env=env,
        sampling=meta['solver_support_semantics'], dt_s=meta['dt'], sim_dt_s=meta['sim_dt'],
        pose_sampling='unchanged original post-integration pose; distinct from support solve geometry',
        unrecorded_substep_slip='unknown', stationary_support_budget='not_calibrated')))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return arrays


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('motion', 'original_state', 'output'):
        parser.add_argument('--' + key.replace('_','-'), type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.motion, args.original_state, args.output)
    print(json.dumps(dict(frames=len(result['load_known']),
        known_load_samples=int(result['load_known'].all(-1).sum()),
        samples_without_load=int((~result['load_bearing'].any(-1)).sum()))))
