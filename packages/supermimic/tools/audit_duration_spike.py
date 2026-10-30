"""Recompute exact forward contact loss and cutoff counterfactual on saved replay."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import json,sys
from pathlib import Path
pass
import torch
from supermimic.lowstep import install
from supermimic.contact_control import ContactClock,ContactGeometry
from supermimic.contact_duration import select_duration_batch
from supermimic.mdp import unflatten_state

torch.set_num_threads(4)
p=Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());install(cfg['task_manifest'],contact_model='force6');device='cuda:0'
demo=torch.load(cfg['recording'],map_location=device,weights_only=False)['data'];clock=ContactClock(demo,hold=1,role_aware=True)
ck=torch.load(p/'checkpoint_000000.pt',map_location=device,weights_only=False);geom=ContactGeometry(demo,device);geom.load_state_dict(ck['geometry']);del ck
b=torch.load(p/'replay.pt',map_location=device,weights_only=False);ix=(b['ptr']-b['size']+torch.arange(b['size'],device=device))%b['T'];d={k:v[ix] for k,v in b['data'].items()};del b
starts=(d['ep_step']==0).all(-1).nonzero().flatten().tolist();spans=[(s,e) for s,e in zip(starts,starts[1:]+[len(ix)]) if d['done'][s:e].any(0).all()];s,e=spans[-1];d={k:v[s:e] for k,v in d.items()}
selected,stops,report=select_duration_batch(d,cfg['duration_prior_centers'],cfg['duration_halfwidth'],cfg['duration_min_success'],cfg['duration_min_mass'])
rows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];latest=rows[-1];assert report['segments']==latest['duration_filter']['segments']
native=d['done'].long().argmax(0)+1;event=d['goal_index'][0];ends=clock.event_ends[event];new=clock.target_masks[ends-1]&~clock.keep_masks[ends-1];target=demo['point'][ends-1]
@torch.no_grad()
def score(stop):
 truncated=stop<native;T,B=d['done'].shape;ts=torch.arange(T,device=device)[:,None,None]
 valid=d['contact_goal']&new[None]&(ts<stop[None,:,None])&~truncated[None,:,None]
 first=valid.long().argmax(0);when=torch.where(valid.any(0),first,(stop-1)[:,None].expand(-1,6))
 worlds,parts=new.nonzero(as_tuple=True);t=when[worlds,parts]
 point=d['point'][t,worlds,parts].clone();observed=d['known'][t,worlds,parts]&d['active'][t,worlds,parts]
 if (~observed).any():
  proxy=geom(unflatten_state(d['s1'][t[~observed],worlds[~observed]]),ends[worlds[~observed]]-1)
  point[~observed]=proxy[torch.arange(len(proxy),device=device),parts[~observed]]
 error=((point-target[worlds,parts])/.05).square().sum(-1)/new.sum(-1)[worlds]
 per=torch.zeros(B,device=device).scatter_add_(0,worlds,error)
 result=[]
 for g in event.unique().tolist():
  ids=selected[event[selected]==g];pairs=(event[worlds]==g)&torch.isin(worlds,ids)
  result.append(dict(segment=g+1,n=len(ids),mean=float(per[ids].mean()) if len(ids) else None,contribution=float(per[ids].sum()/len(selected)),missing=int((pairs&~observed).sum())))
 return dict(contact_loss=float(per[selected].mean()),segments=result)
actual=score(stops);assert abs(actual['contact_loss']-latest['objective_parts']['terminal_contact'])<1e-5
previous_centers=torch.tensor([r['center'] for r in rows[-2]['duration_filter']['segments']],device=device)
cf=stops.clone();cf[event==0]=torch.minimum(native[event==0],previous_centers[0]+cfg['duration_halfwidth'])
result=dict(iteration=latest['iteration'],scope='Identical saved simulator trajectories and retained cohort; change only segment1 cutoff to previous batch upper bound. No simulation or optimization.',actual=actual,previous_segment1_upper= int(previous_centers[0])+cfg['duration_halfwidth'],counterfactual=score(cf))
(p/'cutoff-spike-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
