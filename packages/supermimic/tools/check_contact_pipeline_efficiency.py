"""Matched replay/WM/metrics microbenchmarks; no simulator or optimizer steps."""
from supermimic.paths import runtime_path
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True,help='Pre-change source snapshot')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=8)
    a=p.parse_args()
    import torch
    torch.set_num_threads(4);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False
    from supermimic.lowstep import install
    install(runtime_path('data/lowclimb_boxes/task.json'),contact_model='force6',collision_profile='legacy_v1')
    from supermimic import networks
    from supermimic.contact_control import wm_one_step_loss
    from supermimic.contact_sequence import sequence_attempt_metrics
    from supermimic.contact_io import load_checkpoint_cpu
    from supermimic.buffer import ReplayBuffer
    from supermimic.mdp import normalize_xi

    def reference(name):
        spec=importlib.util.spec_from_file_location('supermimic._efficiency_reference_'+name,a.reference_dir/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    old_control=reference('contact_control');old_sequence=reference('contact_sequence')
    report=dict(actor_updates=0,wm_updates=0,scope='paired fixed-input microbenchmarks, shared GPU; not a full training iteration',
                gpu=torch.cuda.get_device_name(),results={})

    def bench(name,old,new):
        for _ in range(2):old();new()
        samples={'before':[],'after':[]}
        for repeat in range(a.repeats):
            for key,fn in ([('before',old),('after',new)] if repeat%2==0 else [('after',new),('before',old)]):
                torch.cuda.synchronize();begin=time.perf_counter();result=fn();torch.cuda.synchronize()
                samples[key].append((time.perf_counter()-begin)*1000)
                del result
        row={key:dict(median_ms=statistics.median(v),samples_ms=v) for key,v in samples.items()}
        row['speedup']=row['before']['median_ms']/row['after']['median_ms']
        report['results'][name]=row;print(json.dumps({name:row}),flush=True)

    base=runtime_path('tmp/contact-tracker-20261027-v1')
    checkpoint=load_checkpoint_cpu(base/'dmo-continuous-aligned-band-100-resume-v1/checkpoint_000140.pt')
    wm=networks.WorldModel().cuda();wm.load_state_dict(checkpoint['wm']);del checkpoint
    demo=torch.load(base/'demo0-roles-v1/rollout.pt',map_location='cuda',weights_only=False)['data']
    index=torch.arange(512,device='cuda')%len(demo['s'])
    w={k:demo[k][index,None] for k in ('s','s1','a','xi','next_xi')}
    w['mask']=torch.ones(512,1,device='cuda')
    def wm_pass(fn):
        wm.zero_grad(set_to_none=True)
        loss=fn(wm,w,normalize_xi);loss.backward()
        return loss.detach()
    for precision in (None,torch.bfloat16):
        networks.AMP_DTYPE=precision
        old=wm_pass(old_control.wm_one_step_loss)
        old_grad=torch.cat([v.grad.flatten() for v in wm.parameters() if v.grad is not None])
        new=wm_pass(wm_one_step_loss)
        new_grad=torch.cat([v.grad.flatten() for v in wm.parameters() if v.grad is not None])
        relative=float((new_grad-old_grad).norm()/old_grad.norm().clamp_min(1e-20))
        torch.testing.assert_close(old,new,rtol=1e-6,atol=1e-6)
        assert relative<2e-5,relative
        key='wm_one_step_'+str(precision)
        bench(key,lambda:wm_pass(old_control.wm_one_step_loss),lambda:wm_pass(wm_one_step_loss))
        report['results'][key].update(loss_absolute_error=float((old-new).abs()),gradient_relative_error=relative)
        del old_grad,new_grad
    wm.zero_grad(set_to_none=True);del wm

    extra=dict(next_xi=(52,torch.float32),goal_index=((),torch.long),term=((),torch.bool),
               point=((6,3),torch.float32),active=(6,torch.bool),known=(6,torch.bool),
               force=((6,3),torch.float32),contact_goal=(6,torch.bool),contact_first=(6,torch.bool),
               load=(6,torch.float32),surface_local=(6,torch.long),paired_force_w=((6,3),torch.float32),
               contact_count=(6,torch.long),sequence_live=((),torch.bool))
    buffer=ReplayBuffer(96,4096,'cuda',extra_fields=extra);buffer.ptr=0;buffer.size=96
    for k,v in buffer.data.items():
        if v.is_floating_point():v.uniform_(-1,1)
    fields=('s','s1','a','xi','next_xi','sequence_live')
    rng=torch.cuda.get_rng_state();old=buffer.sample_windows(512,1)
    torch.cuda.set_rng_state(rng);new=buffer.sample_windows(512,1,fields=fields)
    assert all(torch.equal(v,old[k]) for k,v in new.items())
    bench('wm_replay_sample',lambda:buffer.sample_windows(512,1),lambda:buffer.sample_windows(512,1,fields=fields))
    report['results']['wm_replay_sample'].update(fields_before=len(old),fields_after=len(new),sampled_values_exact=True)
    rows=torch.arange(96,device='cuda')[:,None];slots=torch.arange(4096,device='cuda')[None]
    needed=('s','s1','xi','next_xi','a','a_prev','ep_step','goal_index','done','contact_goal','contact_first','active','known','point')
    bench('dmo_replay_gather',lambda:{k:v[rows,slots] for k,v in buffer.data.items()},
          lambda:{k:buffer.data[k][rows,slots] for k in needed})
    report['results']['dmo_replay_gather'].update(fields_before=len(buffer.data),fields_after=len(needed),
        before_mib=sum(v.numel()*v.element_size() for v in buffer.data.values())/2**20,
        after_mib=sum(buffer.data[k].numel()*buffer.data[k].element_size() for k in needed)/2**20)
    del buffer,old,new

    # Different stage entries, missing contacts, successful and failed endings.
    T,B,G=96,4096,19;device='cuda'
    first=torch.arange(B,device=device)%(G-1);t=torch.arange(T,device=device)[:,None]
    handoff=torch.arange(B,device=device)%30+20;stop=handoff+torch.arange(B,device=device)%20+10
    goal=first[None]+(t>=handoff[None]).long()
    end=(t==handoff[None]-1)|(t==stop[None]-1)
    passed=end&((t==handoff[None]-1)|((torch.arange(B,device=device)%3)!=0)[None])
    active=torch.rand(T,B,6,device=device)>.5
    data=dict(done=t==stop[None]-1,term=(t==stop[None]-1)&~passed,
              goal_index=goal,stage_done=end,stage_success=passed,stage_step=torch.where(t<handoff[None],t,t-handoff[None]),
              s1=demo['s1'][:T,None].expand(T,B,-1),active=active,known=torch.ones_like(active),
              point=torch.randn(T,B,6,3,device=device)*.01)
    target=torch.zeros(G,6,dtype=torch.bool,device=device);target[:,0]=True
    clock=SimpleNamespace(event_ends=torch.arange(1,G+1,device=device),target_masks=target,
                          keep_masks=torch.zeros_like(target),data=dict(point=torch.zeros(G,6,3,device=device),s1=demo['s1'][:G]))
    old=old_sequence.sequence_attempt_metrics(data,clock,G,continuous=True)
    new=sequence_attempt_metrics(data,clock,G,continuous=True)
    def compare(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:compare(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for u,v in zip(x,y):compare(u,v)
        elif isinstance(x,float):assert abs(x-y)<max(1e-5,abs(x)*2e-6),(x,y)
        else:assert x==y,(x,y)
    compare(old,new)
    bench('sequence_metrics',lambda:old_sequence.sequence_attempt_metrics(data,clock,G,continuous=True),
          lambda:sequence_attempt_metrics(data,clock,G,continuous=True))
    report['results']['sequence_metrics']['same_counts_rates_and_errors']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS no optimizer steps; losses, parameter gradients, sampled data, and metrics agree',flush=True)


if __name__=='__main__':main()
