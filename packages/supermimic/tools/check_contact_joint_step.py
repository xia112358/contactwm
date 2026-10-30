"""Matched numerical and timing check for shared WM state/context encoding."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];pass
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from supermimic.lowstep import install
install(runtime_path('data/lowclimb_boxes/task.json'),contact_model='force6')
from supermimic.networks import WorldModel
import supermimic.networks as networks
from supermimic.mdp import unflatten_state,normalize_xi
b=runtime_path('tmp/contact-tracker-20261027-v1');p=b/'dmo-continuous-aligned-band-100-resume-v1';ck=torch.load(p/'checkpoint_000140.pt',map_location='cuda',weights_only=False);wm=WorldModel().cuda().requires_grad_(False);wm.load_state_dict(ck['wm']);networks.AMP_DTYPE=torch.bfloat16
pack=torch.load(p/'isolated-entry-pack.pt',map_location='cuda',weights_only=False);ix=torch.arange(1024,device='cuda')%6;state=pack['initial']['s'][ix];xi=pack['initial']['xi'][ix];action=pack['initial']['a_prev'][ix];results=[]
def run(joint):
 s=state.clone().requires_grad_();x=xi.clone().requires_grad_();a=action.clone().requires_grad_();S=unflatten_state(s);xn=normalize_xi(x)
 if joint:next_state,next_xi=wm.joint_step(S,a,x,xn)
 else:next_state=wm.step(S,a,xn);next_xi=wm.next_context(S,a,x,xn)
 loss=sum(v.square().mean() for v in next_state)+next_xi.square().mean();gradient=torch.autograd.grad(loss,(s,a,x));return tuple(v.detach() for v in next_state)+(next_xi.detach(),),gradient
old,og=run(False);new,ng=run(True)
maxdiff=max(float((a-b).abs().max()) for a,b in zip(old,new));relative=float(torch.cat([(a-b).flatten() for a,b in zip(og,ng)]).norm()/torch.cat([v.flatten() for v in og]).norm());assert maxdiff<1e-6 and relative<2e-5,(maxdiff,relative)
for _ in range(10):run(False);run(True)
paired={False:[],True:[]}
for repeat in range(60):
    for joint in ([False,True] if repeat%2==0 else [True,False]):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();run(joint);end.record();paired[joint].append((begin,end))
torch.cuda.synchronize()
import statistics
results=[]
for joint,events in paired.items():
    values=[a.elapsed_time(b) for a,b in events]
    results.append(dict(joint=joint,median_ms=statistics.median(values),mean_ms=statistics.mean(values),min_ms=min(values),max_ms=max(values)))
report=dict(batch=1024,output_max_error=maxdiff,gradient_relative_error=relative,timings=results)
out=b/'baselines-20261029/joint-step-efficiency.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
