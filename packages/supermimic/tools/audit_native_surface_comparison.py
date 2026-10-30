"""Audit fresh native runs; historical demonstration supplies goals only.

Within-run loss comparisons hold states/targets fixed. Between-profile native
metrics include trajectory changes of the same frozen policy. Goal-window
coverage is a diagnostic, not the full contact actor's completion criterion.
"""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--temperature',type=float,default=.02)
    p.add_argument('--frames',type=int,help='Limit both analyses to a common prefix; source recordings stay intact')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    from supermimic.contact_control import ContactClock,ContactGeometry,load_contact_tolerances
    from supermimic.contact_surface import SurfaceRegionObjective
    from supermimic.mdp import unflatten_state
    from supermimic.mdp import RewardCfg
    from supermimic.rotations import quat_to_mat
    from somaforge_core.collision_profiles import require_profile
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    from somaforge_core.contact_system.objectives import region_cost
    from somaforge_core.contact_system.diagnostics import witness_motion
    reference=torch.load(a.reference,map_location='cpu',weights_only=False)
    demo=reference['data'];clock=ContactClock(demo,hold=1,role_aware=True)
    tolerances=load_contact_tolerances(clock,reference['metadata'],ROOT/'experiments/contact_completion.json')
    geometry=ContactGeometry(demo,'cpu').double()
    def describe(v):
        v=v.detach().double().flatten()
        if not len(v):return dict(count=0,mean=None,p95=None,max=None)
        return dict(count=len(v),mean=float(v.mean()),p95=float(torch.quantile(v,.95)),max=float(v.max()))
    report=dict(schema='somaforge_native_surface_comparison_v1',reference_role='fixed task intent only; never new contact truth',
        samples=a.samples,temperature=a.temperature,runs=[])
    for directory in a.runs:
        recording=torch.load(directory/'rollout.pt',map_location='cpu',weights_only=False)
        data=recording['data'];meta=recording['metadata']
        collected_frames=len(data['s1'])
        if a.frames is not None:
            data={key:value[:a.frames] if isinstance(value,torch.Tensor) and len(value)==collected_frames else value
                  for key,value in data.items()}
        asset=json.loads(meta['robot_asset_json']);profile=asset.get('collision_profile','legacy_v1')
        validate_g1_asset_metadata(asset,context=str(directory))
        surface=SurfaceRegionObjective(directory/'solver-collision.mjb',geometry,
            samples_per_part=a.samples,temperature=a.temperature,target_normal=[0.,0.,1.],collision_profile=profile).double()
        raw_samples=raw_rows=active_rows=unallocated=0
        for file in sorted(directory.glob('raw-*.npz')):
            with np.load(file,allow_pickle=False) as z:
                if a.frames is not None and int(z['control_step'].max())>a.frames:continue
                evidence=json.loads(str(z['evidence_metadata_json'].item()))
                require_profile(evidence['provenance']['robot_asset'],profile,context=str(file))
                raw_samples+=1;raw_rows+=int(z['count']);active_rows+=int(z['active'].sum())
                unallocated+=int((z['active']&~z['constraint_allocated']).sum())
        if unallocated:raise ValueError('Unallocated native contacts in '+str(directory))
        finite=torch.isfinite(data['s1']).all(-1)
        if not finite.all():raise ValueError('Nonfinite native trajectory in '+str(directory))
        t=len(data['s1']);state=unflatten_state(data['s1'].double())
        pos,rot=geometry.poses(state)
        known=data['known'].bool();active=data['active'].bool()&known
        witness=witness_motion(data['point'].double(),pos,rot,active)
        costs={key:[] for key in ('euclidean_witness','smooth_region')}
        gradients={key:[] for key in costs};stages=[]
        for index,(begin,end) in enumerate(zip(clock.bounds,clock.bounds[1:])):
            endpoint=end-1
            new=clock.target_masks[endpoint]&~clock.keep_masks[endpoint]
            parts=new.nonzero().flatten().tolist()
            stop=min(end,t)
            if not parts or begin>=stop:
                stages.append(dict(segment=index+1,covered=False,reached=False));continue
            target=demo['point'][endpoint].double()[None].expand(stop-begin,-1,-1)
            normal=target.new_tensor([0.,0.,1.]).expand_as(target)
            subset=tuple(v[begin:stop].detach() for v in state)
            root=subset[0].clone().requires_grad_();subset=(root,)+subset[1:]
            point=surface.contact_points(subset,target,parts)
            old=region_cost(point[:,:,None],target,normal,normal_weight=4.,tangent_weight=1.)[0].squeeze(-1)[:,parts].mean(-1)
            smooth=surface.region_loss(subset,target,parts,normals=normal,normal_weight=4.,tangent_weight=1.)['loss'][:,parts].mean(-1)
            for name,value in [('euclidean_witness',old),('smooth_region',smooth)]:
                gradient=torch.autograd.grad(value.sum(),root,retain_graph=True)[0].detach()
                costs[name].append((value[1:]-value[:-1]).abs().detach())
                gradients[name].append((gradient[1:]-gradient[:-1]).norm(dim=-1))
            actual=data['point'][begin:stop].double()
            distance=(actual-target).norm(dim=-1)
            observed=active[begin:stop]
            hits=observed&(distance<=tolerances[index])
            achieved=hits[:,parts].any(0)
            best=distance[:,parts].masked_fill(~observed[:,parts],float('inf')).amin(0)
            stages.append(dict(segment=index+1,covered=stop==end,reached=bool(achieved.all()),
                parts=parts,best_actual_contact_m=[float(v) if torch.isfinite(v) else None for v in best]))
        aligned=min(t,len(demo['s1']))
        forces=data['force'].double().norm(dim=-1)
        termination_checks=None
        if bool(data['term'][-1]):
            config=json.loads((directory/'config.json').read_text())
            manifest=Path(config['task_manifest'])
            if not manifest.is_absolute():manifest=runtime_root()/manifest
            task_spec=json.loads(manifest.read_text())
            if float(task_spec['source_fps'])!=50.:
                raise ValueError('Termination audit needs the installed 50 Hz task-motion contract')
            task=task_spec['tasks'][meta['motion']]
            source=torch.tensor(np.loadtxt(task['csv'],delimiter=','),dtype=torch.float32)
            frame=int(data['f'][-1])+1-int(data['f'][0])
            expected=source[frame];actual=data['s1'][-1]
            expected_R=quat_to_mat(expected[3:7][[3,0,1,2]][None])[0]
            actual_R=quat_to_mat(actual[3:7][None])[0]
            height=float((expected[2]-actual[2]).abs())
            tilt=float((expected_R[2,2]-actual_R[2,2]).abs())
            horizontal=float((expected[:2]-actual[:2]).norm())
            cfg=RewardCfg()
            termination_checks=dict(frame=frame,height_error_m=height,height_limit_m=cfg.term_height,
                height_triggered=height>cfg.term_height,tilt_error=tilt,tilt_limit=cfg.term_tilt,
                tilt_triggered=tilt>cfg.term_tilt,horizontal_error_m=horizontal,horizontal_limit_m=.5,
                horizontal_triggered=horizontal>.5)
        result=dict(path=str(directory.resolve()),profile=profile,collected_frames=collected_frames,frames=t,terminated=bool(data['term'].any()),
            termination_checks=termination_checks,
            finite_states=bool(finite.all()),collision_catalog=surface.catalog,
            raw_audit=dict(snapshots=raw_samples,rows=raw_rows,active_rows=active_rows,unallocated_active=unallocated),
            native_active_fraction=active.double().mean(0).tolist(),force_norm_n=describe(forces[known]),
            root_reference_error_m=describe((data['s1'][:aligned,:3]-demo['s1'][:aligned,:3]).norm(dim=-1)),
            witness_motion=witness,goal_window_coverage=dict(reached=sum(s.get('reached',False) for s in stages),total=len(stages),
                semantics='each requested part makes native contact in the explicit radius during the original reference window; root/orientation/keep duties not included'),
            stages=stages,loss_comparison={})
        for name in costs:
            result['loss_comparison'][name]=dict(adjacent_loss_delta=describe(torch.cat(costs[name])),
                adjacent_root_gradient_delta=describe(torch.cat(gradients[name])))
        report['runs'].append(result)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
