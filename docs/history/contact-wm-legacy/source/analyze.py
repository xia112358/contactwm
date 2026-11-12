"""Audit native contact events with current-contact material-point velocities.

Kinematics estimates velocity only. Activation/allocation comes exclusively
from recorded Newton evidence. Force points are last-solve, poses post-step;
this finite-difference estimate retains that explicit timing limitation.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics
from holosoma.contact_wm.geometry import LINKS
from holosoma.contact_wm.validation import native_success


def material_point_velocity(point,pos,rotation,dt):
    """Velocity of the current material point, not of a moving contact centroid."""
    if dt<=0:raise ValueError('Positive timestep required')
    local=(rotation.transpose(-1,-2)@(point-pos)[...,None]).squeeze(-1)
    previous=pos[:-1]+(rotation[:-1]@local[1:,...,None]).squeeze(-1)
    return torch.cat((torch.zeros_like(point[:1]),(point[1:]-previous)/dt))


def audit(directory,settle_steps=5,hold=10,position_tolerance=.05,slip_tolerance=.05):
    directory=Path(directory);z=np.load(directory/'transitions.npz',allow_pickle=False)
    meta=json.loads((directory/'report.json').read_text());s=torch.tensor(z['next_state'])
    order=[meta['joint_names'].index(n) for n in G1_29DOF_JOINT_ORDER]
    qpos=torch.cat((s[:,:3],s[:,[6,3,4,5]],s[:,13:42][:,order]),-1)
    pos,R=BatchedG1ForwardKinematics().link_poses(qpos,LINKS)
    point=torch.tensor(z['solver_point_w'])
    # At each t, ask where the SAME current material point was at t-1.
    velocity=material_point_velocity(point,pos,R,meta['control_dt']).numpy()
    results=[]
    for event in meta['transitions']:
        onset=event['arrival'];start=onset-10;end=onset+settle_steps+hold
        if start<1 or end>len(s):continue
        establish=np.array(event['establish'],bool);maintain=np.array(event['maintain'],bool)
        normal=np.zeros((6,3));normal[:,2]=1. # Shared selector accepts horizontal tops only.
        goal=dict(position_w=z['target_position_w'][start],normal_w=normal,establish=establish,maintain=maintain,
                  support_position_w=z['solver_point_w'][start],min_normal_force=maintain*10.+establish*1.)
        result=native_success(z['active'][start:end],z['known'][start:end],z['force_w'][start:end],z['solver_point_w'][start:end],
            velocity[start:end],z['terminated'][start:end],goal,arrival=onset+settle_steps-start,hold=hold,
            position_tolerance=position_tolerance,slip_tolerance=slip_tolerance)
        continuous=bool((z['active'][onset:end]&z['known'][onset:end])[:,establish].all())
        result['checks']['continuous_after_onset']=continuous;result['success'] &= continuous
        result.update(onset=onset,start=start,hold_start=onset+settle_steps,end=end,
                      establish=establish.tolist(),maintain=maintain.tolist(),
                      target_position_w=goal['position_w'].tolist(),support_position_w=goal['support_position_w'].tolist())
        results.append(result)
    return dict(motion=meta['motion_index'],recording=str(directory.resolve()),events=results,
                stable_events=sum(e['success'] for e in results),settle_steps=settle_steps,hold_steps=hold,
                position_tolerance=position_tolerance,slip_tolerance=slip_tolerance,
                velocity_method='current native solver force-point material velocity, canonical FK finite difference',
                strict_pose_contact_same_state=False)

@dataclass(frozen=True)
class Options:
    directory:Path
    output:Path
    settle_steps:int=5

def main():
    import tyro
    o=tyro.cli(Options);torch.set_num_threads(1)
    directories=sorted(o.directory.glob('motion_*')) if not (o.directory/'transitions.npz').exists() else [o.directory]
    report=[audit(p,o.settle_steps) for p in directories]
    with o.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([dict(motion=r['motion'],events=len(r['events']),stable=r['stable_events']) for r in report]))

if __name__=='__main__':main()
