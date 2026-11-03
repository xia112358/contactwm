"""Audit exact contact-loss witnesses in saved native replay, without simulation."""
from supermimic.paths import runtime_path
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch
torch.set_num_threads(4)
b=runtime_path('tmp/contact-tracker-20261027-v1');run=b/'dmo-continuous-aligned-band-100-resume-v1';cfg=json.loads((run/'config.json').read_text());spec=json.loads((runtime_path(cfg['task_manifest'])).read_text())
from supermimic.contact_control import ContactClock,ContactGeometry
from supermimic.mdp import unflatten_state
from supermimic.rotations import rot_angle_sq
rec=torch.load(cfg['recording'],map_location='cpu',weights_only=False);demo=rec['data'];clock=ContactClock(demo,hold=1,role_aware=True);ck=torch.load(run/'checkpoint_000140.pt',map_location='cpu',weights_only=False);geom=ContactGeometry(demo,'cpu');geom.load_state_dict(ck['geometry']);del ck
buf=torch.load(run/'replay.pt',map_location='cpu',weights_only=False,mmap=True);ix=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T'];data=buf['data'];ss=(data['ep_step'][ix]==0).all(-1).nonzero().flatten().tolist();start,finish=[(a,z) for a,z in zip(ss,ss[1:]+[len(ix)]) if data['done'][ix[a:z]].any(0).all()][-1];rows=ix[start:finish];goals=data['goal_index'][rows];et,w=data['stage_done'][rows].nonzero(as_tuple=True);g=goals[et,w];duration=data['stage_step'][rows[et],w]+1;entry=et-duration+1;success=data['stage_success'][rows[et],w];new=clock.target_masks[clock.event_ends[g]-1]&~clock.keep_masks[clock.event_ends[g]-1];centers=torch.tensor(cfg['duration_prior_centers']);retained=duration>=centers[g]-2
# Reproduce replay_loss: first valid goal contact per part, otherwise final frame.
T=int(duration.max());local=torch.arange(T)[:,None];indices=entry[None]+torch.minimum(local,duration[None]-1)
hits=data['contact_goal'][rows[indices],w[None]]&(local<duration[None])[...,None];has=hits.any(0);first=hits.long().argmax(0);selected=torch.where(has,entry[:,None]+first,et[:,None]);sample,part=new.nonzero(as_tuple=True);t=selected[sample,part];world=w[sample];event=g[sample];good=success[sample];keep=retained[sample]
observed=data['known'][rows[t],world,part]&data['active'][rows[t],world,part];measured=data['point'][rows[t],world,part];fk=[]
with torch.no_grad():
 for offset in range(0,len(t),1024):
  sl=slice(offset,offset+1024);S=unflatten_state(data['s1'][rows[t[sl]],world[sl]]);v=geom(S,clock.event_ends[event[sl]]-1);fk.append(v[torch.arange(len(v)),part[sl]])
fk=torch.cat(fk);target=demo['point'][clock.event_ends[event]-1,part];point=torch.where(observed[:,None],measured,fk);error=(point-target).norm(dim=-1);loss=(error/.05).square();radii=torch.tensor(cfg['contact_tolerances_m'])[event];labels=['左脚','右脚','左手','右手','左膝','右膝']
summary=[]
for stage in range(19):
 for outcome in [False,True]:
  mask=(event==stage)&(good==outcome)&keep
  if not mask.any():continue
  n=int(mask.sum());actual=mask&observed;proxy=mask&~observed
  summary.append(dict(segment=stage+1,success=outcome,parts=n,actual_parts=int(actual.sum()),fk_parts=int(proxy.sum()),mean_error_cm=float(error[mask].mean()*100),mean_loss=float(loss[mask].mean()),actual_loss=float(loss[actual].mean()) if actual.any() else None,fk_loss=float(loss[proxy].mean()) if proxy.any() else None,failed_fk_inside_goal=int((proxy&(error<=radii)&~good).sum())))
old=json.loads((runtime_path('results/mixed_comparison_web/aligned-failure-videos-v1/report.json')).read_text());examples=[]
for x in old['examples']:
 cohort=(g==x['segment']-1)&(w==x['world'])&(et==x['global_steps']-1);sid=int(cohort.nonzero()[0]);parts=[]
 for z in (sample==sid).nonzero().flatten().tolist():
  parts.append(dict(part=labels[int(part[z])],loss_frame_local=int(t[z]-entry[sid])+1,source='actual_contact' if observed[z] else 'FK_material_point',loss_error_cm=float(error[z]*100),contact_loss=float(loss[z]),point=point[z].tolist(),target=target[z].tolist(),inside_goal=bool(error[z]<=radii[z]),first_goal_hit=bool(has[sid,part[z]])))
 examples.append(dict(segment=x['segment'],entry=x['entry'],world=x['world'],parts=parts))
# Aggregate each stage identically to training, including per-part averaging.
per=torch.zeros(len(et));per.index_add_(0,sample,loss/new.sum(-1)[sample].clamp_min(1));reconstructed={}
metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()];m=metrics[-1];assert m['collected_policy_updates']==139
for sl in m['objective_parts']['stage_losses']:
 mask=(g==sl['segment']-1)&retained
 value=float(per[mask].mean());assert abs(value-sl['contact'])<1e-3,(sl['segment'],value,sl['contact']);reconstructed[sl['segment']]=value
report=dict(policy_updates=139,scope='Exact saved native contact-loss frame/source reconstruction; matches recorded per-stage contact losses within 1e-3',summary=summary,examples=examples,reconstructed_contact_losses=reconstructed)
out=run/'terminal-contact-source-audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
