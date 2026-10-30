"""Extract six-part contact labels from recorded Newton solver decisions.

No distances recomputed, no force threshold, no inferred margin. Legacy raw
contact-only recordings are rejected because candidates are not active contacts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from somaforge_core.robot_assets import validate_g1_asset_metadata

PARTS=('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')


def extract(arrays,metadata,env_id):
    from somaforge_core.newton_contacts import extract_recording_contacts
    result = extract_recording_contacts(arrays, metadata, env_id)
    return result['active'], result['unallocated']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recording',type=Path,required=True);parser.add_argument('--env-id',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.recording,allow_pickle=False) as z:
        metadata=json.loads(z['_metadata_json'].item());validate_g1_asset_metadata(metadata['robot_asset'],context=str(args.recording))
        if not 0<=args.env_id<int(metadata['num_envs']):raise ValueError('env-id out of range')
        contact,overflow=extract(z,metadata,args.env_id)
        timestep=z['motion_time_step'][:,args.env_id].copy()
        terminated=z['terminated'][:,args.env_id].copy();timeout=z['timeout'][:,args.env_id].copy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,contact_part_mask=contact,part_order=np.array(PARTS),
        unallocated_active_contact=overflow,motion_time_step=timestep,terminated=terminated,timeout=timeout,
        recording_path=np.array(str(args.recording.resolve())),env_id=np.array(args.env_id),
        robot_asset_json=np.array(json.dumps(metadata['robot_asset'])),
        contact_semantics_json=np.array(json.dumps(metadata['solver_contact_semantics'])))
    print(json.dumps(dict(frames=len(contact),active_part_frames=int(contact.sum()),
        unallocated_active_part_frames=int(overflow.sum()),output=str(args.output))))


if __name__=='__main__':main()
