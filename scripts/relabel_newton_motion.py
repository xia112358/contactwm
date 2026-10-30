"""Relabel an unchanged canonical motion through the dedicated Newton worker."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from somaforge_core.newton_contact_query import query_contacts
from somaforge_core.newton_contacts import SCHEMA, PARTS
from somaforge_core.newton_contact_data import touchdown_events
from somaforge_core.robot_assets import decode_robot_asset_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True, help='JSON: scene=[center,rotation,half,ground], matching worker binding')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    return relabel(args.motion, args.scene, args.output, args.batch_size)


def relabel(motion, scene, output, batch_size=128):
    from types import SimpleNamespace
    args = SimpleNamespace(motion=Path(motion), scene=Path(scene), output=Path(output), batch_size=batch_size)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    with np.load(args.motion, allow_pickle=False) as motion:
        asset = motion['robot_asset_json'].copy()
        decode_robot_asset_json(asset, context=str(args.motion))
        from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER
        if tuple(motion['joint_names'].astype(str)) != G1_29DOF_JOINT_ORDER:
            raise ValueError('Noncanonical joint order')
        q = motion['joint_pos'].copy(); fps = motion['fps'].copy()
    description = json.loads(args.scene.read_text())
    scene = None if description['scene'] is None else [np.asarray(x, np.float32) for x in description['scene']]
    chunks = []
    for start in range(0, len(q), args.batch_size):
        batch = q[start:start+args.batch_size]
        result = query_contacts(batch, None if scene is None else [np.broadcast_to(x, (len(batch), *x.shape)) for x in scene])
        if chunks and result['provenance']['model_fingerprint'] != chunks[0]['provenance']['model_fingerprint']:
            raise ValueError('Newton worker model changed during relabeling')
        chunks.append(result)
        print(f'Newton relabel {min(start+len(batch),len(q))}/{len(q)}', flush=True)
    mask = np.concatenate([c['active'] for c in chunks])
    from somaforge_core.contact_face_selection import process_face_contacts
    task = process_face_contacts([p for c in chunks for p in c['pairs']],
        description['surface_catalog'], fps=float(fps.reshape(-1)[0]))
    events = touchdown_events(task['contact_part_mask'], surfaces=task['contact_surface'])
    semantics = dict(schema=SCHEMA, sampling='fixed_q_forward_no_integration',
        position='robot geometry point from the exact Newton raw-to-solver contact mapping',
        scene=description, provenance=chunks[0]['provenance'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing labels, including concurrent jobs.
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, contact_part_mask=mask, part_order=np.array(PARTS), fps=fps,
            contact_label_contract_json=np.array(json.dumps(task['contact_label_contract'])),
            task_contact_pairs_json=np.array(json.dumps(task['contact_pairs'])),
            task_abnormal_contact_pairs_json=np.array(json.dumps(task['abnormal_contact_pairs'])),
            **{'task_'+k:task[k] for k in ('contact_part_mask','contact_surface','contact_position_w')},
            contact_position_w=np.concatenate([c['position_w'] for c in chunks]),
            contact_surface=np.concatenate([c['surface'] for c in chunks]),
            unallocated_active_contact=np.concatenate([c['unallocated'] for c in chunks]),
            full_robot_separation_json=np.array(json.dumps([r for c in chunks for r in c['full_robot_separation']])),
            contact_pairs_json=np.array(json.dumps([p for c in chunks for p in c['pairs']])),
            source_path=np.array(str(args.motion.resolve())),
            source_sha256=np.array(hashlib.sha256(args.motion.read_bytes()).hexdigest()),
            robot_asset_json=asset, contact_semantics_json=np.array(json.dumps(semantics)),
            touchdown_frames=np.asarray([f for f,_ in events],np.int64),
            touchdown_parts=np.asarray([p for _,p in events],bool).reshape(-1,6))
    print(json.dumps(dict(output=str(args.output), frames=len(q), events=len(events),
                          active_part_frames=int(mask.sum()), q_unchanged=True)))


if __name__ == '__main__':
    main()
