"""Memory-only DMO backward profile by tiling a saved on-policy native batch."""
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
from supermimic.lowstep import install
cfg=json.loads((a.run/'config.json').read_text());spec=install(cfg['task_manifest'],contact_model='force6')
from supermimic.contact_control import ContactActor,ContactGeometry,ContactClock
from supermimic.contact_dmo import replay_loss
from supermimic.networks import WorldModel
import supermimic.networks as networks
from supermimic.lowstep_observations import HeightForceContext
from supermimic.mdp import normalize_xi
from supermimic.robot import default_joint_pos_list
last=json.loads((a.run/'metrics.jsonl').read_text().splitlines()[-1])
ck=torch.load(a.run/f"checkpoint_{last['iteration']-1:06d}.pt",map_location='cuda',weights_only=False)
source=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
actor=ContactActor().cuda();actor.load_state_dict(ck['actor'])
prior=ContactActor().cuda().requires_grad_(False);prior.load_state_dict(torch.load(cfg['prior_checkpoint'],map_location='cuda',weights_only=False)['actor'])
wm=WorldModel().cuda().requires_grad_(False);wm.load_state_dict(ck['wm'])
demo=torch.load(cfg['recording'],map_location='cuda',weights_only=False)['data'];clock=ContactClock(demo,hold=1,role_aware=True)
geometry=ContactGeometry(demo,'cuda');geometry.load_state_dict(ck['geometry'])
context=HeightForceContext(spec['tasks']);default=torch.tensor(default_joint_pos_list(),device='cuda')
buf=torch.load(a.run/'replay.pt',map_location='cuda',weights_only=False)
T=last['objective_parts']['unroll_steps'];rows=(buf['ptr']-T+torch.arange(T,device='cuda'))%buf['T']
idx=torch.arange(a.batch,device='cuda')%buf['N'];batch={k:v[rows[:,None],idx[None,:]] for k,v in buf['data'].items()}
del ck,source,buf
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();t=time.perf_counter()
try:
 loss,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,batch,prior_actor=prior)
 loss.backward();torch.cuda.synchronize()
 result=dict(result='PASS',batch=a.batch,horizon=T,seconds=time.perf_counter()-t,baseline_gib=base/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,details=details,scope='Memory profile only: tiled recorded trajectories, no simulation or optimizer update')
except torch.cuda.OutOfMemoryError:
 result=dict(result='OOM',batch=a.batch,horizon=T,baseline_gib=base/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
