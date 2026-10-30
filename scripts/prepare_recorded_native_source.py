"""Extract an unchanged native-clock policy execution and its loaded evidence.

An explicit recording window supplies the action-reference extent. This tool
does not average poses, resample time, optimize support, or rank by slide budget.
Recorded solver geometry remains distinct from post-integration motion FK.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from materialize_recorded_support import materialize
from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk
from somaforge_core.newton_contact_data import require_current_newton_manifest
from somaforge_core.newton_support import SUPPORT_SCHEMA, SUPPORT_FIELDS
from somaforge_core.robot_assets import encode_robot_asset_json, validate_g1_asset_metadata


def extract(recording, output, *, env_id, frame_count, resume=False):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError(output)
    with np.load(recording, allow_pickle=False) as z:
        meta = json.loads(z['_metadata_json'].item())
        require_current_newton_manifest(meta, context=str(recording))
        validate_g1_asset_metadata(meta['robot_asset'], context=str(recording))
        if meta.get('solver_support_semantics', {}).get('schema') != SUPPORT_SCHEMA:
            raise ValueError('Recording lacks original same-solve loaded material evidence')
        missing = ['solver_contact_'+k for k in SUPPORT_FIELDS if 'solver_contact_'+k not in z]
        if missing:
            raise ValueError('Missing recorded support channels: '+', '.join(missing))
        if not 0 <= env_id < meta['num_envs'] or not 1 < frame_count <= len(z['motion_time_step']):
            raise ValueError('Native recording window outside recorded data')
        if meta['env_id'] >= 0 and meta['env_id'] != env_id:
            raise ValueError('Requested environment was not recorded')
        def select(key):
            value=z[key][:frame_count]
            return value[:,env_id].copy() if meta['env_id'] < 0 else value.copy()
        if not np.array_equal(select('motion_time_step'), np.arange(frame_count)):
            raise ValueError('Execution is not one continuous native-clock sequence')
        if select('terminated').any() or select('timeout')[:-1].any():
            raise ValueError('Requested source window includes a failed/reset execution')
        quaternion = select('root_quat_xyzw')[:,[3,0,1,2]]
        quaternion /= np.linalg.norm(quaternion,axis=-1,keepdims=True)
        q=np.concatenate((select('root_pos'),quaternion,select('dof_pos')),axis=1)
        provenance=dict(recording=str(recording),sha256=hashlib.sha256(recording.read_bytes()).hexdigest(),
            env=env_id,native_frame_indices=list(range(frame_count)),dt_s=meta['dt'],sim_dt_s=meta['sim_dt'],
            sampling=meta['solver_support_semantics'],poses='unchanged original post-integration physical execution',
            alignment='native recording clock; no pose averaging, interpolation or support correction',
            source_selection='explicit environment; no motion-budget-based quality selection')
        seed=dict(joint_pos=q,joint_names=np.asarray(meta['dof_names']),fps=np.asarray(meta['fps']),
            robot_asset_json=np.asarray(encode_robot_asset_json(meta['robot_asset'])),
            rollout_provenance_json=np.asarray(json.dumps(provenance)))
    output.mkdir(parents=True,exist_ok=resume)
    state=output/'original_state.npz'
    if state.exists():
        with np.load(state,allow_pickle=False) as existing:
            if any(not np.array_equal(existing[k],v) for k,v in seed.items()):
                raise ValueError('Incomplete source belongs to different recording or native window')
    else:
        with state.open('xb') as stream:
            np.savez_compressed(stream,**seed)
    motion=output/'motion.npz'
    if not motion.exists():
        canonicalize_motion_with_direct_newton_fk(state,motion,device='cpu')
    else:
        with np.load(motion,allow_pickle=False) as z:
            if not np.allclose(z['joint_pos'],q,atol=1.e-6,rtol=0):
                raise ValueError('Existing source poses differ from original execution')
    observations=output/'observations.npz'
    if not observations.exists():
        materialize(motion,state,observations)
    else:
        from somaforge_core.support_evidence import load_support_observations
        load_support_observations(motion,observations)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return dict(output=str(output),frames=frame_count,env_id=env_id,
        motion_sha256=hashlib.sha256((output/'motion.npz').read_bytes()).hexdigest(),
        recording_sha256=provenance['sha256'],support_motion_truth='same-solve original recording; post-integration FK separate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-id',type=int,required=True)
    parser.add_argument('--frame-count',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    print(json.dumps(extract(args.recording,args.output,env_id=args.env_id,frame_count=args.frame_count,resume=args.resume)),flush=True)
