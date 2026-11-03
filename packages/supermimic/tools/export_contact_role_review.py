#!/usr/bin/env python3
"""Export native contact-role trajectories for read-only browser review."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Rebuilt dataset directory containing library/')
    parser.add_argument('--task-manifest',type=Path,required=True,help='Original task manifest matching the recording hash')
    parser.add_argument('--output',type=Path,required=True,help='Directory for generated HTML, meshes and trajectory JSON')
    parser.add_argument('--previous',type=Path,help='Previous rebuilt dataset, for source-frame overlap mapping')
    args=parser.parse_args()
    import ast,json,hashlib,xml.etree.ElementTree as ET
    import numpy as np
    import torch,trimesh
    from somaforge_core.robot_assets import canonical_g1_urdf_path,canonical_g1_source_metadata,validate_g1_asset_metadata
    from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics,_rpy_matrix
    from somaforge_core import G1_29DOF_JOINT_ORDER
    from supermimic.contact_topology import support_topology
    r=Path(__file__).resolve().parents[3];out=args.output;lib=args.dataset/'library'
    out.mkdir(parents=True,exist_ok=True)
    import shutil
    assets=r/'packages/supermimic/supermimic/resources/contact_role_review'
    for name in ('index.html','player.js','style.css'):shutil.copyfile(assets/name,out/name)
    torch.set_num_threads(4);rec=torch.load(lib/'rollout.pt',map_location='cpu',weights_only=False);data=rec['data'];meta=rec['metadata'];asset=json.loads(meta['robot_asset_json']);validate_g1_asset_metadata(asset,context='role player source')
    cat=json.loads((lib/'catalogue.json').read_text());cases=cat['cases']
    assert sum(c['stop']-c['start'] for c in cases)==int(data['source_supervision_valid'].sum())
    old_cases=json.loads((args.previous/'library/catalogue.json').read_text())['cases'] if args.previous else []
    from collections import Counter
    role_names={'S':'支撑','T':'转移','N':'不参与'}
    counts=Counter(c['conversion_class'] for c in cases)
    labels={'STNN':'单下肢支撑 · 另一侧转移','SSNN':'双下肢支撑','STTT':'下肢支撑/转移 · 双上肢转移'}
    classes=[dict(key=k,label=labels[k] if k in labels else '下肢：'+'/'.join(role_names[x] for x in k[:2])+' · 上肢：'+'/'.join(role_names[x] for x in k[2:]),count=n) for k,n in counts.most_common()]
    urdf=canonical_g1_urdf_path();xml=ET.parse(urdf).getroot();links=[l.get('name') for l in xml.findall('link')];parts=[]
    for link in xml.findall('link'):
     name=link.get('name');limb='LL' if name.startswith('left_') and any(v in name for v in ('hip','knee','ankle','foot')) else 'RL' if name.startswith('right_') and any(v in name for v in ('hip','knee','ankle','foot')) else 'LA' if name.startswith('left_') else 'RA' if name.startswith('right_') else None
     for vis in link.findall('visual'):
      g=vis.find('geometry');shape=list(g)[0]
      if shape.tag=='mesh':
       mesh=trimesh.load(urdf.parent/shape.get('filename'),force='mesh',process=False);scale=np.array([float(v) for v in shape.get('scale','1 1 1').split()]);v=np.asarray(mesh.vertices)*scale;f=np.asarray(mesh.faces)
      elif shape.tag=='sphere':
       mesh=trimesh.creation.icosphere(subdivisions=1,radius=float(shape.get('radius')));v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
      elif shape.tag=='cylinder':
       mesh=trimesh.creation.cylinder(radius=float(shape.get('radius')),height=float(shape.get('length')),sections=12);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
      elif shape.tag=='box':
       mesh=trimesh.creation.box(extents=[float(v) for v in shape.get('size').split()]);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
      else:raise ValueError(shape.tag)
      # Mesh-only visual simplification; never used as contact evidence.
      cells=np.round(v/(.025 if shape.tag=='mesh' else .004)).astype(int);_,inverse=np.unique(cells,axis=0,return_inverse=True);counts=np.bincount(inverse);vv=np.stack([np.bincount(inverse,weights=v[:,i])/counts for i in range(3)],axis=1);ff=inverse[f];ff=ff[(ff[:,0]!=ff[:,1])&(ff[:,1]!=ff[:,2])&(ff[:,0]!=ff[:,2])];ff=np.unique(ff,axis=0)
      origin=vis.find('origin');xyz=np.array([float(x) for x in origin.get('xyz','0 0 0').split()]) if origin is not None else np.zeros(3);rot=_rpy_matrix(tuple(float(x) for x in origin.get('rpy','0 0 0').split())).numpy() if origin is not None else np.eye(3);vv=vv@rot.T+xyz
      parts.append(dict(link=links.index(name),limb=limb,vertices=vv.round(5).tolist(),faces=ff.tolist()))
    robot=dict(asset=canonical_g1_source_metadata(),links=links,parts=parts)
    (out/'robot.json').write_text(json.dumps(robot,separators=(',',':')))
    node=ast.parse((r/'packages/supermimic/supermimic/robot.py').read_text());names=next(ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in n.targets));order=[names.index(n) for n in G1_29DOF_JOINT_ORDER]
    fk=CanonicalG1ForwardKinematics();taskpath=args.task_manifest;assert hashlib.sha256(taskpath.read_bytes()).hexdigest()==meta['task_manifest_sha256'];tasks=json.loads(taskpath.read_text())['tasks'];traces=[]
    for span in meta['recording_spans']:
     tid=span['trace_id'];b,e=span['begin'],span['end'];s=data['s1'][b:e];qpos=torch.cat((s[:,:7],s[:,13:42][:,order]),-1)
     with torch.no_grad():P,R=fk.link_poses(qpos,tuple(links))
     assert torch.isfinite(P).all() and torch.isfinite(R).all()
     states=support_topology({k:v[b:e] for k,v in data.items()},[[[] for _ in range(6)] for _ in range(e-b)])
     payload=dict(effective_source_frames=span['effective_source_frames'],edge_exclusions=span['edge_exclusions'],posture_phases=span['posture_phases'],transfers=span['transfer_events'],name=span['name'],dt=meta['dt'],boxes=tasks[span['motion']]['boxes'],positions=P.numpy().round(5).tolist(),rotations=R.numpy().round(5).tolist(),active=data['active'][b:e].tolist(),stable=data['contact_stable'][b:e].tolist(),support=[s['mask'] for s in states],points=data['point'][b:e].numpy().round(5).tolist(),loads=data['contact_load'][b:e].numpy().round(2).tolist(),source_frames=data['f'][b:e].tolist())
     (out/f'trace-{tid}.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False));traces.append(dict(id=tid,name=span['name'],frames=e-b))
    manifest=dict(source_motion_contract=meta['source_motion_contract'],temporal_contract=meta['segmentation_temporal_contract'],previous_dataset=str(args.previous) if args.previous else None,previous_case_count=len(old_cases),dt=meta['dt'],classes=classes,cases=[dict(kind=c['kind'],id=c['case'],trace=c['trace_id'],name=c['name'],start=c['source_start'],stop=c['source_stop'],category=c['conversion_class'],roles=c['topology']['limb_roles'],forms=c['topology']['contact_forms'],boundary_reasons=c['boundary_reasons'],transfer_ids=c['transfer_ids'],supervised=c['frame_point_supervision'],point_valid_frames=int(data['point_loss_valid'][c['start']:c['stop']].any(-1).sum()),previous_ids=[o['case'] for o in old_cases if o['trace']==c['trace'] and max(o['source_start'],c['source_start'])<min(o['source_stop'],c['source_stop'])]) for c in cases],traces=traces,provenance=dict(dataset=str(lib),asset=asset,timing='robot: recorded post-step s1; contacts: recorded final native solver sample, not recomputed geometry',training_started=False))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
    print(json.dumps(dict(cases=len(cases),traces=len(traces),triangles=sum(len(x['faces']) for x in parts),bytes=sum(p.stat().st_size for p in out.glob('*.json')))))

    review=[]
    for i,c in enumerate(cases):
     n=c['source_stop']-c['source_start']
     if 10<n<=40:continue
     previous=cases[i-1] if i and cases[i-1]['trace_id']==c['trace_id'] else None
     review.append(dict(id=c['case'],name=c['name'],start=c['source_start'],stop=c['source_stop'],frames=n,kind=c['kind'],role=c['topology']['role_key'],start_reasons=previous['boundary_reasons'] if previous else ['source_start'],end_reasons=c['boundary_reasons'],arrivals=list(zip(c['arrival_frames'],c['arrival_parts']))))
    (out/'length-review.json').write_text(json.dumps(review,ensure_ascii=False))


if __name__=='__main__':
    main()
