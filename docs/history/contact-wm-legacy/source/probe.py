"""Compare inherited/calibrated WM Jacobians with matched native action pulses.

This is open-loop local sensitivity validation, not policy optimization and not
proof that finite-difference sensitivities at contact discontinuities generalize.
"""
from dataclasses import dataclass
from pathlib import Path
from supermimic.paths import runtime_path
import json
import numpy as np
import torch
from holosoma.contact_wm.supermimic import SuperMimicForceWM
from holosoma.contact_wm.height_context import HeightRaster
from holosoma.contact_wm.geometry import WitnessGeometry

@dataclass(frozen=True)
class Options:
    recording:Path
    calibrated:Path
    output:Path
    source:Path=runtime_path('logs/climb_low_force6_resume1000/ckpt_006000.pt')
    rotations:Path|None=None


def cosine(a,b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator>1e-12 else None


def relative_error(predicted,actual):
    norm=float(np.linalg.norm(actual))
    return float(np.linalg.norm(predicted-actual)/norm) if norm>1e-12 else None


def main():
    import tyro
    o=tyro.cli(Options);torch.set_num_threads(1)
    records=[];metadata=[]
    for i in range(9):
        directory=o.recording/f'probe_{i:02d}'
        records.append(np.load(directory/'transitions.npz',allow_pickle=False))
        metadata.append(json.loads((directory/'report.json').read_text()))
    meta=metadata[0];start=meta['probe_step'];end=len(records[0]['action']);epsilon=meta['probe_amplitude'];joints=meta['probe_joints']
    if start<0 or start>=end or epsilon<=0 or any(len(r['state'])!=end or r['terminated'][start:].any() for r in records):raise ValueError('Incomplete/failed paired experiment')
    if any(m['motion_index']!=meta['motion_index'] for m in metadata):raise ValueError('Different paired motions')
    for i,r in enumerate(records[1:],1):
        delta=np.zeros_like(records[0]['action'][start])
        delta[joints[(i-1)//2]]=epsilon if i%2 else -epsilon
        if not np.allclose(r['action'][start]-records[0]['action'][start],delta,atol=1e-6,rtol=1e-5):
            raise ValueError('Incorrect paired pulse')
    before=np.stack([r['state'][start].copy() for r in records]);origins=np.array([m['reference_origin_w'] for m in metadata]);before[:,:3]-=origins
    mismatch=float(np.abs(before-before[:1]).max())
    if mismatch>1e-4:raise ValueError('Mismatched native initial state')
    for r in records[1:]:
        if not np.array_equal(r['action'][start+1:],records[0]['action'][start+1:]):raise ValueError('Future actions are not paired')
    geometry=WitnessGeometry(meta['joint_names'],torch.zeros(6,3))
    states=torch.tensor(np.stack([r['next_state'][-1] for r in records]));hand=geometry(states)[:,2].detach().numpy()-origins
    force=np.stack([r['force_w'][-1,2] for r in records]);actual_position=np.stack([(hand[1+2*i]-hand[2+2*i])/(2*epsilon) for i in range(4)],-1)
    actual_force=np.stack([(force[1+2*i]-force[2+2*i])/(2*epsilon) for i in range(4)],-1)
    atlas=np.load(o.recording/'terrain_atlas.npz');reader=HeightRaster(torch.tensor(atlas['height'][:1]),torch.tensor(atlas['minimum'][:1]),float(atlas['resolution']))
    def evaluate(model):
        model.requires_grad_(False)
        state=torch.tensor(records[0]['state'][start:start+1]);previous=torch.tensor(records[0]['previous_force_w'][start:start+1])
        first=torch.tensor(records[0]['action'][start:start+1],requires_grad=True)
        physics=torch.zeros(1,33);physics[:,0]=(1.-.75)/.45
        for step in range(start,end):
            action=first if step==start else torch.tensor(records[0]['action'][step:step+1])
            prediction=model.source_step(state,action,physics,reader(state),previous)
            state=prediction['state'];previous=prediction['force_w']
        hand=geometry(state)[0,2];force=previous[0,2]
        pos_j=torch.stack([torch.autograd.grad(hand[k],first,retain_graph=True)[0][0,joints] for k in range(3)]).detach().numpy()
        force_j=torch.stack([torch.autograd.grad(force[k],first,retain_graph=True)[0][0,joints] for k in range(3)]).detach().numpy()
        if not np.isfinite(pos_j).all() or not np.isfinite(force_j).all():raise ValueError('Nonfinite WM Jacobian')
        return dict(position_jacobian=pos_j.tolist(),force_jacobian=force_j.tolist(),
            position_cosine=cosine(pos_j,actual_position),force_cosine=cosine(force_j,actual_force),
            position_relative_error=relative_error(pos_j,actual_position),
            force_relative_error=relative_error(force_j,actual_force),
            force_absolute_error=float(np.linalg.norm(force_j-actual_force)),
            predicted_terminal_hand=(hand.detach().numpy()-origins[0]).tolist(),predicted_terminal_force=force.detach().tolist())
    inherited=SuperMimicForceWM(o.rotations);inherited.load_source(o.source);before_wm=evaluate(inherited)
    checkpoint=torch.load(o.calibrated,map_location='cpu',weights_only=False,mmap=True)
    inherited.load_state_dict(checkpoint['wm'],strict=True);after_wm=evaluate(inherited)
    report=dict(motion=meta['motion_index'],initial_state_mismatch=mismatch,start=start,horizon=end-start,epsilon=epsilon,joints=joints,
        terminal_hand_frame='world axes relative to reference environment origin',
        native_position_jacobian=actual_position.tolist(),native_force_jacobian=actual_force.tolist(),
        native_terminal_hand=hand[0].tolist(),native_terminal_force=force[0].tolist(),
        native_terminal_hand_active=[bool(r['active'][-1,2]) for r in records],
        inherited=before_wm,calibrated=after_wm,policy_updated=False,optimization_allowed=False,
        limitations=['one local open-loop probe','no unperturbed replica noise-floor measurement','oracle terrain atlas','finite-difference contact sensitivity may be nonsmooth',
                     'stable target-execution baseline still required before policy optimization'])
    with o.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))

if __name__=='__main__':main()
