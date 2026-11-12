"""Score native execution of a fixed goal-conditioned policy and save trajectory."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from holosoma.contact_wm.analyze import material_point_velocity
from holosoma.contact_wm.geometry import LINKS
from holosoma.contact_wm.validation import native_success
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics


@dataclass(frozen=True)
class Options:
    recording:Path
    output:Path


def main():
    import tyro
    o=tyro.cli(Options);torch.set_num_threads(1)
    meta=json.loads((o.recording/'report.json').read_text());z=np.load(o.recording/'transitions.npz')
    start=meta['execution_start'];end=start+meta['execution_horizon']
    goal={k:np.asarray(v)[0] if isinstance(v,list) else v for k,v in meta['execution_goal'].items()}
    s=torch.tensor(z['next_state']);order=[meta['joint_names'].index(n) for n in G1_29DOF_JOINT_ORDER]
    qpos=torch.cat((s[:,:3],s[:,[6,3,4,5]],s[:,13:42][:,order]),-1)
    pos,R=BatchedG1ForwardKinematics().link_poses(qpos,LINKS)
    velocity=material_point_velocity(torch.tensor(z['solver_point_w']),pos,R,meta['control_dt']).numpy()
    if end>len(s):result=dict(success=False,reason='episode ended before complete contact hold',available_steps=len(s)-start)
    else:
        result=native_success(z['active'][start:end],z['known'][start:end],z['force_w'][start:end],
            z['solver_point_w'][start:end],velocity[start:end],z['terminated'][start:end],goal,
            arrival=goal['arrival_step']-1,hold=goal['hold_steps'],position_tolerance=.05,slip_tolerance=.05,
            position_required=goal['establish'] if meta.get('support_contract','fixed_points')=='sustained_contact' else None)
        support_distance=np.linalg.norm(z['solver_point_w'][start:end]-goal['support_position_w'],axis=-1)
        result['max_support_point_displacement']=float(support_distance[:,goal['maintain']].max())
    result.update(policy=meta['contact_policy'],recording=str(o.recording.resolve()),
        goal_offset=meta['contact_goal_offset'],native_backend=meta['native_backend'],
        support_contract=meta.get('support_contract','fixed_points'),
        executed_actions_path=str((o.recording/'transitions.npz').resolve()),
        failure_contract=meta['failure_contract'],strict_pose_contact_same_state=False)
    with o.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
