"""Compare contact objectives on a saved trajectory, without physics or updates.

Use --output for an explicit project runtime path, or read the JSON on stdout.
The fixed target is declared by --segment; physical contacts stay native data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--collision-model', type=Path, required=True)
    p.add_argument('--segment', type=int, required=True, help='One-based demonstration segment')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--temperature', type=float, default=.02)
    p.add_argument('--normal', nargs=3, type=float, default=(0.,0.,1.), help='Explicit task world normal')
    p.add_argument('--normal-weight', type=float, default=1.)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    from supermimic.contact_control import ContactClock, ContactGeometry
    from supermimic.contact_surface import SurfaceRegionObjective
    from supermimic.mdp import unflatten_state
    from somaforge_core.contact_system.diagnostics import variation, witness_motion
    from somaforge_core.contact_system.objectives import region_cost, temporal_region_loss
    torch.set_num_threads(4)
    original = torch.load(a.recording, map_location='cpu', weights_only=False)
    demo = original['data']
    bundle = torch.load(a.trajectory, map_location='cpu', weights_only=False)
    data = bundle['data']
    # Accept a single saved trajectory or time x environment rollouts.
    if data['s1'].ndim == 2:
        data = {k:v[:,None] if isinstance(v,torch.Tensor) and len(v)==len(data['s1']) else v
                for k,v in data.items()}
    clock = ContactClock(demo, hold=1, role_aware=True)
    if not 1 <= a.segment <= len(clock.event_ends):
        raise ValueError('Segment outside demonstration')
    endpoint = int(clock.event_ends[a.segment-1])-1
    geometry = ContactGeometry(demo, 'cpu')
    surface = SurfaceRegionObjective(a.collision_model, geometry, samples_per_part=a.samples,
                                    temperature=a.temperature, target_normal=a.normal)
    time, batch = data['s1'].shape[:2]
    target = demo['point'][endpoint][None].expand(time*batch,-1,-1)
    new = clock.target_masks[endpoint] & ~clock.keep_masks[endpoint]
    parts = new.nonzero().flatten().tolist()
    if not parts:
        parts = clock.target_masks[endpoint].nonzero().flatten().tolist()
    if not parts:
        raise ValueError('Segment has no contact task')
    finite_state = torch.isfinite(data['s1']).all(-1)
    flat = data['s1'].reshape(-1,71)
    placeholder = torch.zeros_like(flat);placeholder[:,3]=1
    # Computational placeholders are excluded from every diagnostic; they do
    # not become repaired trajectory states or physical contact evidence.
    state = unflatten_state(torch.where(finite_state.flatten()[:,None],flat,placeholder))
    pos, rot = geometry.poses(state)
    normals = target.new_tensor(a.normal).expand_as(target)
    old_point = surface.contact_points(state,target,parts)
    old_cost, _, _ = region_cost(old_point[:,:,None],target,normals,normal_weight=a.normal_weight)
    smooth = surface.region_loss(state,target,parts,normals=normals,normal_weight=a.normal_weight)
    valid = finite_state.clone()
    if 'ep_step' in data:
        valid[1:] &= data['ep_step'][1:]>data['ep_step'][:-1]
    if 'done' in data:
        valid &= data['done'].long().cumsum(0).roll(1,0)==0
        valid[0] = torch.isfinite(data['s1'][0]).all(-1)
    if 'goal_index' in data:
        valid &= data['goal_index']==a.segment-1
    mask = valid[...,None].expand(time,batch,len(parts))
    costs = {'old_euclidean_selection':old_cost.squeeze(-1)[:,parts],
             'metric_consistent_hard':smooth['hard_loss'][:,parts],
             'smooth_surface':smooth['loss'][:,parts]}
    report = dict(schema='somaforge_fixed_surface_audit_v1',segment=a.segment,
                  trajectory=str(a.trajectory.resolve()),surface=surface.contract,
                  scope='fixed saved states and fixed target; geometric diagnostics, not executed improvement',
                  valid_samples=int(mask.sum()),variation={})
    for name, value in costs.items():
        value = value.reshape(time,batch,len(parts))
        report['variation'][name] = variation(value,mask)
    complete = mask.any(0)
    if complete.any():
        selected = costs['smooth_surface'].reshape(time,batch,len(parts))[:,complete]
        report['temporal_region_loss'] = temporal_region_loss(selected,mask[:,complete],dt=.02).tolist()
    if 'point' in data:
        known = data['known'][:,:,parts] & data['active'][:,:,parts] & mask
        report['native_witness_motion'] = witness_motion(data['point'][:,:,parts],
            pos[:,parts].reshape(time,batch,len(parts),3),
            rot[:,parts].reshape(time,batch,len(parts),3,3),known)
    # Check the geometry derivative against finite differences at retained states.
    ids = valid.flatten().nonzero().flatten()[:16]
    if len(ids):
        fixed = tuple(v[ids].detach().double() for v in state)
        surface.double()
        root = fixed[0].clone().requires_grad_()
        def objective(position):
            return surface.region_loss((position,)+fixed[1:],target[ids].double(),parts,
                normals=normals[ids].double(),normal_weight=a.normal_weight)['loss'][:,parts].sum()
        gradient = torch.autograd.grad(objective(root),root)[0]
        eps = 1e-5
        finite = []
        for axis in range(3):
            delta = torch.zeros_like(root);delta[:,axis]=eps
            finite.append((objective(root.detach()+delta)-objective(root.detach()-delta))/(2*eps))
        finite = torch.stack(finite)
        analytic = gradient.sum(0)
        report['geometry_gradient'] = dict(analytic=analytic.tolist(),finite_difference=finite.tolist(),
            max_abs_error=float((finite-analytic).abs().max()),epsilon_m=eps,
            scope='geometry derivative only; world-model and simulator Jacobians not certified')
    text = json.dumps(report,indent=2,allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:f.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
