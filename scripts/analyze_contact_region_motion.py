"""Audit stepwise motion of verified native contacts on canonical trajectories.

Reports geometric region statistics only. Motion acceptance instead uses the
original loaded-material RMS phase metric. This tool does not classify contact,
infer force support, or change the input trajectory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.contact_motion import (
    CONTACT_MOTION_SCHEMA, contact_region_statistics,calibrate_pivot_budgets)
from somaforge_core.newton_contact_data import load_contact_labels
from somaforge_core.robot_assets import decode_robot_asset_json


def analyze(motion, labels, phases):
    load_contact_labels(motion, labels)
    with np.load(motion, allow_pickle=False) as z:
        decode_robot_asset_json(z['robot_asset_json'], context=str(motion))
        if 'newton_direct_fk_metadata_json' not in z:
            raise ValueError('Motion needs certified canonical Newton FK')
        names=z['body_names'].astype(str).tolist()
        pos=z['body_pos_w'].astype(float); quat=z['body_quat_w']
        rot=Rotation.from_quat(quat[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*quat.shape[:-1],3,3)
    with np.load(labels,allow_pickle=False) as z:
        semantics=json.loads(z['contact_semantics_json'].item())
        selected=select_contact_pairs(json.loads(z['contact_pairs_json'].item()),
                                      semantics['scene']['surface_catalog'])
        parts=z['part_order'].astype(str).tolist()
    n=len(pos); rows=[];steps=torch.full((n-1,6),float('nan'),dtype=torch.float64)
    for phase in phases:
        a,b,p,face=(int(phase[k]) for k in ('start','end','part','surface'))
        if not 0<=a<b<n: raise ValueError('Phase outside motion')
        distances=[];groups=[]
        for t in range(a,b):
            for pair in selected['contact_pairs'][t]:
                if pair['part']!=p or pair['surface']!=face:continue
                name=pair['body_name'].rsplit('/',1)[-1]
                i=names.index(name)
                local=rot[t,i].T@(np.asarray(pair['position_w'])-pos[t,i])
                delta=pos[t+1,i]-pos[t,i]+(rot[t+1,i]-rot[t,i])@local
                normal=np.asarray(pair['normal_w'])
                distances.append(float(np.linalg.norm(delta-normal*np.dot(delta,normal))))
                groups.append(t-a)
        stats={key:value.numpy() for key,value in contact_region_statistics(
            torch.tensor(distances,dtype=torch.float64),torch.tensor(groups,dtype=torch.long),b-a).items()}
        complete=bool(np.isfinite(stats['pivot']).all())
        on=selected['contact_part_mask'][a:b+1,p] & (selected['contact_surface'][a:b+1,p]==face)
        rows.append(dict(event_id=phase.get('event_id'),start=a,end=b,part=parts[p],surface=face,
            measurement_complete=complete,missing_measurement_frames=(np.flatnonzero(~np.isfinite(stats['pivot']))+a).tolist(),
            actual_off_frames=(np.flatnonzero(~on)+a).tolist(),
            region_path_cm={key:float(np.nansum(value)*100) for key,value in stats.items() if key!='count'},
            step_max_cm=float(np.nanmax(stats['pivot'])*100) if complete else None,
            step_sample_count=stats['count'].tolist(),
            pivot_step_cm=(stats['pivot']*100).tolist()))
        steps[a:b,p]=torch.tensor(stats['pivot'])
    complete_phases=[(int(p['start']),int(p['end']),int(p['part']))
                     for p,row in zip(phases,rows) if row['measurement_complete']]
    calibration=iter(calibrate_pivot_budgets(steps,complete_phases))
    for row in rows:
        if row['measurement_complete']:row['source_calibration']=next(calibration)
    return dict(schema=CONTACT_MOTION_SCHEMA,geometric_metric='actual_contact_pivot_minimum',
                motion=str(motion.resolve()),labels=str(labels.resolve()),frame_count=n,
                interpretation='Actual-contact pivot existence plus full region distribution; force-bearing slip unknown. Part-pooled median+3MAD budgets are geometric only.',
                contact_semantics=semantics,phases=rows,training_ready=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('motion','labels','phases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.motion,args.labels,json.loads(args.phases.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    for row in report['phases']:
        print(row['start'],row['end'],row['part'],row['region_path_cm'],row['measurement_complete'])
