"""Rebuild augmentation source patches from actual shape/face observations.

No projection, mean contact point, force threshold, or inherited intervals.
This is a source layer; edited trajectories still need independent validation.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import numpy as np
import torch
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.contact_labels import DEFAULT_POLICY, ContactLabelPolicy
from somaforge_core.newton_contact_data import load_contact_labels
from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.layers import write_contact_layer, read_verified_contact_graph
from motion_edit.contact.io import write_contact_jsonl
from motion_edit.contact.schema import ContactAnchorRecord, ContactPatchRecord, ContactEventRecord
from motion_edit.contact.surface_catalog import surfaces_from_obj_mesh_faces
from materialize_task_contact_labels import materialize


def semantic_body(shape_label, part):
    if part.endswith('foot'):
        match = re.search(r'ankle_roll_sphere_([1-5])(?:_|/|$)', shape_label)
        if match:
            return part.split('_')[0] + ('_heel' if int(match[1]) <= 2 else '_toe')
    return part


def contiguous_runs(frames):
    frames = sorted(frames)
    if not frames:
        return []
    runs = [[frames[0], frames[0]+1]]
    for f in frames[1:]:
        if f == runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([f,f+1])
    return runs


def manifold_runs(observations):
    runs=[]
    for f in sorted(observations):
        count=len(observations[f])
        if runs and f==runs[-1][1] and count==runs[-1][2]:
            runs[-1][1]+=1
        else:
            runs.append([f,f+1,count])
    return [(a,b) for a,b,_ in runs]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--terrain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--motion-id',default='climb_00')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if DEFAULT_POLICY != ContactLabelPolicy():
        raise ValueError('Shape-level task filtering/aggregation contract needs migration before source-layer rebuild')
    labels=load_contact_labels(args.motion,args.labels)
    with np.load(args.labels,allow_pickle=False) as z:
        pairs=labels['contact_pairs']
        semantics=json.loads(z['contact_semantics_json'].item())
        part_names=z['part_order'].astype(str).tolist()
    model=semantics['provenance']; binding=semantics['scene']
    surfaces=surfaces_from_obj_mesh_faces(motion_id=args.motion_id,obj_path=args.terrain,
                                         include_sides=True,include_downward=True)
    surface_by_id={}
    used_faces={p['surface'] for row in pairs for p in row}
    for face in binding['surface_catalog']:
        if face['surface'] not in used_faces:
            continue
        matches=[s for s in surfaces if np.allclose(s.normal,face['normal_w'],atol=1e-6,rtol=0)
                 and abs(np.dot(s.normal,s.origin)-face['plane_offset'])<1e-6]
        if len(matches)!=1:
            raise ValueError(f'Actual surface mapping ambiguous/missing: {face}')
        surface_by_id[face['surface']]=matches[0]
    channels=defaultdict(dict)
    for f,row in enumerate(pairs):
        for p in row:
            key=(p['robot_shape'],p['surface'])
            channels[key].setdefault(f,[]).append(p)
    bodies=sorted({model['body_labels'][p['body']].rsplit('/',1)[-1] for row in pairs for p in row})
    with np.load(args.motion,allow_pickle=False) as z:
        q=z['joint_pos'].copy(); joint_names=z['joint_names'].astype(str).tolist()
    if len(joint_names) != 29 or set(joint_names) != set(G1_29DOF_JOINT_ORDER):
        raise ValueError('Expected canonical G1 joint names')
    indices = [joint_names.index(name) for name in G1_29DOF_JOINT_ORDER]
    q = np.concatenate((q[:, :7], q[:, 7:][:, indices]), axis=-1)
    fk=CanonicalG1ForwardKinematics()
    with torch.no_grad():
        pos,rotation=fk.link_poses(torch.as_tensor(q,dtype=torch.float32),tuple(bodies))
    pos,rotation=pos.numpy(),rotation.numpy()
    anchors,patches,events=[],[],[]
    for (shape,sid), observations in sorted(channels.items()):
        shape_label=model['shape_labels'][shape]
        face=surface_by_id[sid]
        for a,b in manifold_runs(observations):
            rows=[sorted(observations[f],key=lambda p:tuple(p['position_w'])) for f in range(a,b)]
            body_label=model['body_labels'][rows[0][0]['body']]
            if any(model['body_labels'][p['body']]!=body_label for row in rows for p in row):
                raise ValueError('Shape parent changed inside contact interval')
            part=part_names[rows[0][0]['part']]
            body=semantic_body(shape_label,part)
            idx=bodies.index(body_label.rsplit('/',1)[-1])
            points=np.asarray([[p['position_w'] for p in row] for row in rows])
            local=np.einsum('tji,tpj->tpi',rotation[a:b,idx],points-pos[a:b,idx,None])
            # Both trajectories retain every actual robot geometry point.
            # No counterpart-plane projection or coordinatewise median.
            aid=f'{args.motion_id}_{body}_shape{shape}_face{sid}_{a:06d}_{b:06d}'
            pid=aid+'_patch'; delta=points[0,0]-np.asarray(face.origin)
            metadata=dict(contact_label_contract=labels['contact_label_contract'],
                newton_shape_label=shape_label,newton_surface_id=sid,
                source_target_contract='newton_robot_geometry_point_trajectory',
                surface_bindings=[dict(polygon_surface_coordinates=face.metadata.get('polygon_surface_coordinates'))])
            anchors.append(ContactAnchorRecord(motion_id=args.motion_id,anchor_id=aid,body=body,
                start_frame=a,end_frame=b,role='unknown',world_position=points[0,0].tolist(),
                object_id=face.object_id,surface_id=face.surface_id,surface_type=face.surface_type,
                surface_normal=face.normal,surface_origin=face.origin,surface_tangent_u=face.tangent_u,
                surface_tangent_v=face.tangent_v,surface_bounds=face.bounds,
                surface_coordinates=dict(u=float(delta@face.tangent_u),v=float(delta@face.tangent_v)),
                surface_binding_source='actual_newton_face',patch_id=pid,
                position_source='first_actual_robot_geometry_point',source='shared_task_contacts',metadata=metadata))
            patches.append(ContactPatchRecord(motion_id=args.motion_id,patch_id=pid,anchor_id=aid,body=body,
                start_frame=a,end_frame=b,patch_type='point',patch_center_world=points[0,0].tolist(),
                link_names=[body_label.rsplit('/',1)[-1]],newton_body_label=body_label,
                newton_shape_labels=[shape_label]*points.shape[1],robot_points_local=local[0].tolist(),
                source_target_frames=list(range(a,b)),source_target_points_w=points.tolist(),
                source_points_local_by_frame=local.tolist(),
                robot_asset_fingerprint=model['robot_asset']['asset_bundle_sha256'],
                robot_binding_backend='newton_mjwarp',robot_binding_source='shared_task_contacts',metadata=metadata))
            for frame,kind in [(a,'touchdown'),(b,'liftoff')]:
                if (kind=='touchdown' and a-1 in observations) or (kind=='liftoff' and b in observations):
                    continue  # Manifold point-count changes are not contact transitions.
                if 0<frame<len(q):
                    events.append(ContactEventRecord(motion_id=args.motion_id,event_id=f'{aid}_{kind}',frame=frame,
                        body=body,event_type=kind,source='shape_face_contact_transition',metadata=metadata))
    args.output.mkdir(parents=True,exist_ok=False)
    graph=ContactGraph(motion_id=args.motion_id,anchors=anchors,patches=patches,events=events)
    write_contact_layer(args.output,graph)
    write_contact_jsonl(args.output/'surfaces'/f'{args.motion_id}.jsonl',surfaces)
    new_labels=args.output/'contact_labels'/f'{args.motion_id}.npz'
    materialize(args.motion,args.labels,new_labels)
    read_verified_contact_graph(args.output,args.motion_id,motion_path=args.motion,labels_path=new_labels)
    report=dict(frames=len(q),shape_face_intervals=len(anchors),patches=len(patches),
        represented_contact_point_samples=sum((p.end_frame-p.start_frame)*len(p.robot_points_local) for p in patches),
        input_contact_pairs=sum(map(len,pairs)),contact_label_contract=labels['contact_label_contract'],
        source_motion=str(args.motion.resolve()),training_ready=False,
        note='Shape intervals are not action segments. Source geometry points are retained; no dynamic validation implied.')
    with (args.output/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
