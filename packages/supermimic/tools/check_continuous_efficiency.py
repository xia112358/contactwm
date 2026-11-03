"""Compare continuous stage planning/extraction revisions; no training updates."""
from supermimic.paths import runtime_path
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
import torch
from supermimic import contact_sequence as sequence,contact_duration as duration


def load(path,name):
    spec=importlib.util.spec_from_file_location('supermimic._continuous_reference_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(batch,device,groups=19):
    centers=[93,29,20,72,21,17,42,19,40,40,39,36,25,47,28,26,24,21,15][:groups]
    worlds=torch.arange(batch,device=device);first=worlds%groups
    stages=torch.arange(groups,device=device)[:,None]
    count=1+(worlds*7)%(groups-first)
    reached=stages<count[None]
    durations=torch.tensor(centers,device=device)[(first[None]+stages).clamp_max(groups-1)]+(worlds[None]+stages*3)%7-3
    ends=(durations*reached).cumsum(0);native_stop=(durations*reached).sum(0)
    T=int(native_stop.max());times=torch.arange(T,device=device)[:,None]
    goal=first[None].expand(T,-1).clone();stage_done=torch.zeros(T,batch,device=device,dtype=torch.bool)
    for k in range(groups):
        goal+=((times>=ends[k])&(k+1<count)[None]).long()
        stage_done|=(times+1==ends[k])&reached[k][None]
    done=times+1==native_stop[None]
    stage_success=stage_done&((times+1<native_stop[None])|((worlds%3)==0)[None])
    data=dict(goal_index=goal,done=done,term=done&~stage_success,stage_done=stage_done,stage_success=stage_success)
    return data,centers


def compare(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)


def add_states(data,device):
    T,B=data['done'].shape
    # Deliberately distinct row/world values expose wrong gather indices.
    t=torch.arange(T,device=device)[:,None];b=torch.arange(B,device=device)[None]
    p=(t*.001+b*.00001)
    s=torch.zeros(T,B,71,device=device);s[:,:,0]=p;s[:,:,2]=1;s[:,:,3]=1
    data.update(s=s,s1=s.clone(),xi=torch.zeros(T,B,52,device=device),next_xi=torch.zeros(T,B,52,device=device),
        a=torch.zeros(T,B,29,device=device),a_prev=torch.zeros(T,B,29,device=device),ep_step=t.expand(-1,B),
        contact_goal=torch.zeros(T,B,6,device=device,dtype=torch.bool),active=torch.ones(T,B,6,device=device,dtype=torch.bool),
        known=torch.ones(T,B,6,device=device,dtype=torch.bool),point=torch.zeros(T,B,6,3,device=device))


def check_cpu(old_sequence,old_duration):
    checked=0
    for groups in (1,2,5,19):
        data,centers=fixture(23,'cpu',groups)
        for kind in ('none','fixed','dynamic'):
            histories=[]
            for module in (old_duration,duration):
                histories.append(None if kind=='none' else module.FixedDurationBand(centers) if kind=='fixed' else module.DurationBandHistory(centers,bootstrap_batches=0))
            old=old_sequence.sequence_plan(data,groups,histories[0]);new=sequence.sequence_plan(data,groups,histories[1])
            compare(old,new);checked+=1
        add_states(data,'cpu');actor=SimpleNamespace(hist=5);rows=torch.arange(len(data['done']))
        for sorted_ in (False,True):
            # Rotate ring storage: extraction must follow rows, not assume zero.
            shifted=SimpleNamespace(data={key:value.roll(7,0) for key,value in data.items()})
            def obs(S,previous,xi,*args):return torch.cat((S[0],previous,xi),-1)
            with patch('supermimic.contact_control.observation',obs):
                before=list(old_sequence.stage_batches(shifted,(rows+7)%len(rows),old[1],actor,None,None,13,sort_by_duration=sorted_))
                after=list(sequence.stage_batches(shifted,(rows+7)%len(rows),new[1],actor,None,None,13,sort_by_duration=sorted_))
            compare(before,after);checked+=1
    return dict(cases=checked,result='exact plans, fixed/dynamic reports, every copied field/history, weights, stable ordering and wrapped ring rows')


def bench_cuda(old_sequence,old_duration,batch,repeats):
    torch.set_num_threads(4)
    data,centers=fixture(batch,'cuda')
    old_band=old_duration.FixedDurationBand(centers);new_band=duration.FixedDurationBand(centers)
    old=old_sequence.sequence_plan(data,19,old_band);new=sequence.sequence_plan(data,19,new_band);compare(old,new)
    timings=[]
    def measure(name,fn):
        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
        timings.append(dict(revision=name,milliseconds=(time.perf_counter()-start)*1000));return result
    for repeat in range(repeats):
        ordered=[('before',old_sequence,old_band),('after',sequence,new_band)]
        for name,module,history in (ordered if repeat%2==0 else ordered[::-1]):
            measure(name,lambda:module.sequence_plan(data,19,history))
    planning=timings;timings=[]
    add_states(data,'cuda')
    from supermimic.lowstep import install
    spec=install(runtime_path('data/lowclimb_boxes/task.json'),contact_model='force6')
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.robot import default_joint_pos_list
    context=HeightForceContext(spec['tasks']);default=torch.tensor(default_joint_pos_list(),device='cuda')
    actor=SimpleNamespace(hist=5);buffer=SimpleNamespace(data=data);rows=torch.arange(len(data['done']),device='cuda')
    def extract(module):
        shapes=[]
        for values,cutoffs,history,weight,total in module.stage_batches(buffer,rows,new[1],actor,context,default,4096):
            shapes.append([len(cutoffs),len(values['s'])])
        return shapes
    history_error=0.
    before=old_sequence.stage_batches(buffer,rows,new[1],actor,context,default,4096)
    after=sequence.stage_batches(buffer,rows,new[1],actor,context,default,4096)
    for a,b in zip(before,after):
        torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
        history_error=max(history_error,float((a[2]-b[2]).abs().max()))
    extract(old_sequence);extract(sequence)
    for repeat in range(repeats):
        ordered=[('before',old_sequence),('after',sequence)]
        for name,module in (ordered if repeat%2==0 else ordered[::-1]):shapes=measure(name,lambda:extract(module))
    return dict(batch=batch,horizon=len(rows),retained_stages=int(new[1]['enabled'].sum()),batches=shapes,history_max_error=history_error,
        plan_timings=planning,extract_timings=timings,
        median_ms={section:{name:statistics.median(r['milliseconds'] for r in values if r['revision']==name) for name in ('before','after')}
            for section,values in [('sequence_plan',planning),('stage_batches',timings)]},
        scope='Synthetic continuous stage schedule and distinct stored states; actual height/contact observation encoder; shared GPU; zero simulator/optimizer steps')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--cuda',action='store_true');p.add_argument('--batch',type=int,default=4057);p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',type=Path)
    args=p.parse_args();old_sequence=load(args.reference_dir/'contact_sequence.py','sequence');old_duration=load(args.reference_dir/'contact_duration.py','duration')
    report=dict(cpu=check_cpu(old_sequence,old_duration),optimizer_updates=0)
    if args.cuda:report['cuda']=bench_cuda(old_sequence,old_duration,args.batch,args.repeats)
    if args.output:args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
