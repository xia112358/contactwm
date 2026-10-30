"""Independent forward-contract audit and scoring-rule sensitivity on real replay."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,json,sys
from pathlib import Path
pass
import torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)
cfg=json.loads((a.run/'config.json').read_text());metrics=[json.loads(s) for s in (a.run/'metrics.jsonl').read_text().splitlines()]
from supermimic.lowstep import install
install(cfg['task_manifest'],contact_model='force6')
from supermimic.contact_control import ContactClock,ContactGeometry
from supermimic.mdp import unflatten_state
record=torch.load(cfg['recording'],map_location='cpu',weights_only=False)['data'];clock=ContactClock(record,hold=1,role_aware=True);geometry=ContactGeometry(record,'cpu')
b=torch.load(a.run/'replay.pt',map_location='cpu',weights_only=False);idx=(b['ptr']-b['size']+torch.arange(b['size']))%b['T'];data={k:v[idx] for k,v in b['data'].items()};del b
starts=(data['ep_step']==0).all(-1).nonzero().flatten().tolist();blocks=[]
for start,end in zip(starts,starts[1:]+[len(idx)]):
 d={k:v[start:end] for k,v in data.items()}
 if d['done'].any(0).all():blocks.append(d)
results=[]
for d,m in zip(blocks[-2:],metrics[-2:]):
 B=d['done'].shape[1];slot=torch.arange(B);stop=d['done'].long().argmax(0);time=torch.arange(len(d['done']))[:,None];live=time<=stop;event=d['goal_index'][0];end=clock.event_ends[event]
 target=record['point'][end-1];new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
 observed=d['known']&d['active'];distance=(d['point']-target).norm(dim=-1)
 tolerance=torch.tensor(cfg.get('contact_tolerances_m',[.05]*len(clock.event_ends)))[event,None]
 valid=observed&(distance<=tolerance)&new[None]&live[...,None]
 first_valid=valid&~torch.cat((torch.zeros_like(valid[:1]),valid[:-1].long().cummax(0).values.bool()),0)
 stored=d['contact_goal']&new[None]&live[...,None]
 assert torch.equal(first_valid,stored),'stored goal event does not match native region crossing'
 has=valid.any(0);hit=valid.long().argmax(0);score_time=torch.where(has,hit,stop[:,None]);parts=torch.arange(6)[None].expand(B,6);slots=slot[:,None].expand_as(parts)
 measured=d['point'][score_time,slots,parts];real=observed[score_time,slots,parts]
 S=unflatten_state(d['s1'][score_time,slots]);proxy=geometry(S,end[:,None]-1)
 proxy=proxy.gather(-2,parts[...,None,None].expand(B,6,1,3)).squeeze(-2)
 selected=torch.where(real[...,None],measured,proxy)
 errors=((selected-target)/.05).square().sum(-1)
 per_attempt=(errors*new).sum(-1)/new.sum(-1).clamp_min(1)
 contact=float(per_attempt.mean());expected=m.get('objective_parts',{}).get('terminal_contact') if m.get('objective_parts') else None
 if expected is not None:assert abs(expected-contact)<1e-5,(expected,contact)
 between=live[:-1]&(time[:-1]+1<=stop)
 # All transitions of a first attempt must be consecutive and pre-reset.
 ep=d['ep_step'];assert bool((ep==time).masked_select(live).all())
 continuity=(d['s1'][:-1]-d['s'][1:]).abs().amax(-1)
 success=~d['term'][stop,slot]
 groups={}
 for label,mask in [('success',success),('failure',~success)]:
  groups[label]=dict(count=int(mask.sum()),mean_loss=float(per_attempt[mask].mean()) if mask.any() else None,contribution=float(per_attempt.masked_fill(~mask,0).mean()))
 # First actual contact, earliest valid goal and terminal-only scores on the
 # identical recorded attempts. These are diagnostics, not task rule changes.
 scoring=[]
 for radius in [0.05,0.051,0.055,0.06]:
  good=observed&(distance<=radius)&new[None]&live[...,None];has_good=good.any(0)
  ti=torch.where(has_good,good.long().argmax(0),stop[:,None]);ob=observed[ti,slots,parts]
  v=d['point'][ti,slots,parts];score=((v-target)/.05).square().sum(-1)
  all_real=((~new)|ob).all(-1);vals=(score*new).sum(-1)/new.sum(-1).clamp_min(1)
  scoring.append(dict(radius_m=radius,all_real_attempts=int(all_real.sum()),mean_loss_on_all_real=float(vals[all_real].mean()) if all_real.any() else None,changed_scoring_parts=int(((ti!=score_time)&new).sum())))
 # Distance-vector bias and variance: E||e||^2 = ||E e||^2 + E||e-Ee||^2.
 statistics=[]
 for part in range(6):
  choose=new[:,part]&real[:,part]
  if not choose.any():continue
  err=selected[choose,part]-target[choose,part];mean=err.mean(0);variance=(err-mean).square().sum(-1).mean()
  statistics.append(dict(part=part,samples=int(choose.sum()),mean_error_cm=(mean*100).tolist(),bias_loss=float(mean.square().sum()/.05**2),variance_loss=float(variance/.05**2),rms_error_cm=float(err.square().sum(-1).mean().sqrt()*100)))
 results.append(dict(sampled_update=m['collected_policy_updates'],attempts=B,contact_loss=contact,logged_loss=expected,scoring_count=int(new.sum()),missing_contact_terms=int((new&~real).sum()),continuity_max=float(continuity[between].max()) if between.any() else None,groups=groups,radius_sensitivity=scoring,error_bias_variance=statistics))
report=dict(scope='Independent reconstruction of last two complete first-attempt batches. Threshold alternatives use identical recorded trajectories and are diagnostics only. The 5 cm normalization stays fixed. No gradients or task settings changed.',results=results)
if len(blocks)>=2 and bool((blocks[-1]['goal_index'][0]==6).all()):
 x,y=blocks[-2:];frame=45;part=0
 common=x['contact_goal'][frame,:,part]&y['contact_goal'][frame,:,part]
 pair=[]
 for v in [x,y]:
  end=clock.event_ends[v['goal_index'][0]];target=record['point'][end-1,part]
  error=(v['point'][frame,:,part]-target)[common];mean=error.mean(0)
  pair.append(dict(contact_loss=float(error.square().sum(-1).mean()/.05**2),bias_loss=float(mean.square().sum()/.05**2),variance_loss=float((error-mean).square().sum(-1).mean()/.05**2)))
 report['same_frame46_cohort']=dict(attempts=int(common.sum()),before=pair[0],after=pair[1])
(a.run/'contact-loss-contract-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
