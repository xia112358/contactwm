"""Paired DMO loss/gradient/timing checks on saved data, without optimizer steps.

Provide a pre-change contact_dmo.py with --reference-source to compare revisions.
The optional CUDA check tiles six retained real trajectories; it is a component
benchmark, not a new simulation or estimate of a complete training iteration.
"""
from supermimic.paths import runtime_path
import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import statistics
import sys
import time
from unittest.mock import patch
import warnings

ROOT=Path(__file__).resolve().parents[1]
import torch
from supermimic import contact_dmo as current


def load_reference(path):
    spec=importlib.util.spec_from_file_location('supermimic._dmo_efficiency_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check_small(reference):
    fixture=runpy.run_path(str(ROOT/'tools/check_contact_pairs.py'))
    reports=[]
    for paired in (False,True):
        for cutoff in ([None] if paired else [None,torch.tensor([2])]):
            data={k:v.clone() for k,v in fixture['d'].items()}
            if not paired:data['goal_index'].zero_()
            answers=[]
            for module in (reference,current):
                actor=fixture['Actor']();detailed={};attempts={};components={}
                with patch.object(module,'observation',lambda S,*a:S[0][:,:1]):
                    loss,details=module.replay_loss(actor,fixture['WM'](),fixture['clock'],fixture['Geometry'](),None,lambda x:x,None,data,
                        completion_mode='accumulated_target',stop_steps=cutoff,
                        stage_plan=fixture['plan'] if paired else None,
                        detailed_losses=detailed,per_attempt_losses=attempts,component_losses=components,check=True)
                loss.backward()
                answers.append((loss.detach(),actor.weight.grad.clone(),details,detailed,attempts,components))
            torch.testing.assert_close(answers[0][0],answers[1][0],rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(answers[0][1],answers[1][1],rtol=1e-6,atol=1e-6)
            for old,new in zip(answers[0][3:],answers[1][3:]):
                assert old.keys()==new.keys()
                for key in old:torch.testing.assert_close(old[key],new[key],rtol=1e-6,atol=1e-6)
            for key in answers[0][2]:
                assert answers[0][2][key]==answers[1][2][key],(key,answers[0][2],answers[1][2])
            reports.append(dict(paired=paired,cutoff=None if cutoff is None else int(cutoff[0]),loss=float(answers[1][0]),gradient_max_error=float((answers[0][1]-answers[1][1]).abs().max())))
    return reports


def check_event_schedule():
    """Compare batched scheduling to the original stateful, per-frame rule."""
    from types import SimpleNamespace
    rng=torch.Generator().manual_seed(84)
    for paired in (False,True):
        for mode in ('onset','accumulated_target'):
            for _ in range(20):
                T,B=11,7;initial=torch.randint(0,3,(B,),generator=rng)
                split=torch.randint(3,6,(B,),generator=rng)
                stop=torch.randint(8,T+1,(B,),generator=rng)
                cut=torch.rand(B,generator=rng)<.4
                times=torch.arange(T)[:,None]
                events=initial[None].expand(T,-1).clone()
                if paired:events+=(times>=split[None])
                raw=torch.rand(T,B,6,generator=rng)<.25
                masks=torch.rand(5,6,generator=rng)<.5
                clock=SimpleNamespace(event_ends=torch.arange(1,6),target_masks=masks,keep_masks=torch.zeros_like(masks))
                data=dict(goal_index=events,contact_goal=raw,contact_first=raw)
                plan=None
                if paired:
                    plan=dict(length=2,ends=torch.stack((split-1,stop)),enabled=torch.rand(2,B,generator=rng)>.2,
                              truncated=torch.stack((cut,torch.zeros_like(cut))))
                actual,terminal,_=current._replay_events(data,clock,initial,stop,cut,mode,plan)
                seen=torch.zeros(B,6,dtype=torch.bool);previous=initial.clone();expected=[];ends=[]
                for t in range(int(stop.max())):
                    live=t<stop;final=t+1==stop;eligible=live;truncated=cut
                    event=initial
                    if paired:
                        event=events[t];seen&=(event==previous)[:,None];previous=event
                        stage=event-initial;columns=torch.arange(B)
                        endpoint=plan['ends'][stage,columns];enabled=plan['enabled'][stage,columns]
                        truncated=plan['truncated'][stage,columns]
                        final=(t+1==endpoint)&enabled&live;eligible=live&enabled&(t<endpoint)
                    new=masks[event]
                    hit=raw[t]&new&eligible[:,None]&~seen&~truncated[:,None]
                    expected.append(hit|(final[:,None]&new&~seen));ends.append(final)
                    seen|=hit
                assert torch.equal(actual,torch.stack(expected))
                assert torch.equal(terminal,torch.stack(ends))
    return dict(cases=80,result='exact masks: onset/accumulated, first hits, truncation, disabled stages, handoffs')


def check_cuda(reference,batch,repeats,objective,source_root):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from supermimic.lowstep import install
    spec=install(runtime_path('data/lowclimb_boxes/task.json'),contact_model='force6')
    from supermimic.contact_control import ContactActor,ContactClock,ContactGeometry,observation
    from supermimic.contact_surface import SurfaceRegionObjective,CollisionSurface
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.mdp import unflatten_state,normalize_xi
    from supermimic.networks import WorldModel
    import supermimic.networks as networks
    from supermimic.robot import default_joint_pos_list
    networks.AMP_DTYPE=torch.bfloat16
    base=source_root
    source=base/'dmo-continuous-aligned-band-100-resume-v1'
    cfg=json.loads((source/'config.json').read_text())
    ck=torch.load(source/'checkpoint_000140.pt',map_location='cuda',weights_only=False)
    actor=ContactActor().cuda();actor.load_state_dict(ck['actor'])
    wm=WorldModel().cuda().requires_grad_(False);wm.load_state_dict(ck['wm'])
    prior=ContactActor().cuda().requires_grad_(False)
    prior.load_state_dict(torch.load(runtime_path(cfg['prior_checkpoint']),map_location='cuda',weights_only=False)['actor'])
    demo=torch.load(runtime_path(cfg['recording']),map_location='cuda',weights_only=False)['data']
    clock=ContactClock(demo,hold=1,role_aware=True)
    geometry=ContactGeometry(demo,'cuda');geometry.load_state_dict(ck['geometry'])
    context=HeightForceContext(spec['tasks']);default=torch.tensor(default_joint_pos_list(),device='cuda')
    raw=torch.load(base/'isolated-surface-normal4-v1/rollout_000.pt',map_location='cuda',weights_only=False)
    ix=torch.arange(batch,device='cuda')%6
    data={key:value[:,ix] for key,value in raw['data'].items()}
    pack=torch.load(source/'isolated-entry-pack.pt',map_location='cuda',weights_only=False)
    previous=pack['previous_history']
    history=observation(unflatten_state(previous['s'].flatten(0,1)),previous['a_prev'].flatten(0,1),previous['xi'].flatten(0,1),default,context).reshape(6,actor.hist,-1)
    first=observation(unflatten_state(raw['data']['s'][0]),raw['data']['a_prev'][0],raw['data']['xi'][0],default,context)
    history=torch.cat((history[:,1:],first[:,None]),1)[ix]
    surface=(SurfaceRegionObjective if objective=='surface_region' else CollisionSurface)(base/'initialization-fast-v3-solver.mjb',geometry,'cuda')
    normals=torch.tensor(raw['config']['normals'],device='cuda')[ix]
    del ck,pack
    def run(module):
        actor.zero_grad(set_to_none=True)
        loss,details=module.replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,data,
            completion_mode='accumulated_target',prior_actor=prior,prior_weight=.01,
            initial_history=history,goal_normals=normals,normal_weight=4.,tangent_weight=1.,contact_surface=surface)
        loss.backward()
        grad=torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
        return loss.detach(),grad.detach(),details
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        old=run(reference);new=run(current)
        relative=float((old[1]-new[1]).norm()/old[1].norm().clamp_min(1e-20))
        torch.testing.assert_close(old[0],new[0],rtol=1e-5,atol=1e-5)
        assert relative<1e-4,relative
        rows=[]
        for repeat in range(repeats):
            for name,module in ([('before',reference),('after',current)] if repeat%2==0 else [('after',current),('before',reference)]):
                actor.zero_grad(set_to_none=True);torch.cuda.synchronize()
                baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                loss,grad,details=run(module);torch.cuda.synchronize()
                rows.append(dict(revision=name,seconds=time.perf_counter()-started,peak_extra_bytes=torch.cuda.max_memory_allocated()-baseline,loss=float(loss)))
    # Same saved trajectories; metrics should retain all values and exclusions.
    native_before=reference.native_attempt_metrics(data,clock,None)
    native_after=current.native_attempt_metrics(data,clock,None)
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:compare(a[key],b[key])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):assert abs(a-b)<=max(1e-5,abs(a)*2e-6),(a,b)
        else:assert a==b,(a,b)
    compare(native_before,native_after)
    metric_rows=[]
    for name,module in [('before',reference),('after',current)]:
        torch.cuda.synchronize();started=time.perf_counter()
        for _ in range(20):module.native_attempt_metrics(data,clock,None)
        torch.cuda.synchronize();metric_rows.append(dict(revision=name,milliseconds=(time.perf_counter()-started)*50))
    return dict(batch=batch,horizon=len(data['s']),objective=objective,wm_precision='bf16',
        loss_before=float(old[0]),loss_after=float(new[0]),gradient_relative_error=relative,
        action_replay_max_error=new[2]['action_replay_max_error'],timings=rows,
        median_seconds={name:statistics.median(r['seconds'] for r in rows if r['revision']==name) for name in ('before','after')},
        native_metrics=metric_rows,optimizer_updates=0,simulation_steps=0,
        scope='Six recorded real trajectories tiled across the requested batch; GPU shared with existing workloads.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,help='Existing experiment inputs; required with --cuda')
    parser.add_argument('--cuda',action='store_true');parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--objective',choices=['surface_region','collision_surface'],default='surface_region')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.cuda and args.source_root is None:parser.error('--cuda requires --source-root')
    reference=load_reference(args.reference_source)
    result=dict(cpu_cases=check_small(reference),event_schedule=check_event_schedule(),optimizer_updates=0)
    if args.cuda:result['cuda']=check_cuda(reference,args.batch,args.repeats,args.objective,args.source_root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
