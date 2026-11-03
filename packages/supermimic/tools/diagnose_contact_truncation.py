"""Audit saved native batches without collecting or updating networks."""
import argparse,json,sys
from pathlib import Path
import torch
from supermimic.contact_control import ContactClock
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)
cfg=json.loads((a.run/'config.json').read_text());metrics=[json.loads(x) for x in (a.run/'metrics.jsonl').read_text().splitlines()]
demo=torch.load(cfg['recording'],map_location='cpu',weights_only=False)['data'];clock=ContactClock(demo,hold=1,role_aware=True)
buf=torch.load(a.run/'replay.pt',map_location='cpu',weights_only=False);rows=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T'];data={k:v[rows] for k,v in buf['data'].items()}
starts=(data['ep_step']==0).all(-1).nonzero().flatten().tolist();print('Synchronized reset rows',starts)
blocks=[]
for j,start in enumerate(starts):
 end=starts[j+1] if j+1<len(starts) else len(rows)
 d={k:v[start:end] for k,v in data.items()}
 if d['done'].any(0).all():blocks.append(d)
assert len(blocks)>=2
result=[];saved=[]
for d,metric in zip(blocks[-2:],metrics[-2:]):
 B=d['done'].shape[1];slot=torch.arange(B);stop=d['done'].long().argmax(0)
 event=d['goal_index'][0];ends=clock.event_ends[event];new=clock.target_masks[ends-1]&~clock.keep_masks[ends-1];assert (new.sum(-1)==1).all()
 part=new.long().argmax(-1);target=demo['point'][ends-1][slot,part]
 observed=d['known'][:,slot,part]&d['active'][:,slot,part]
 dist=(d['point'][:,slot,part]-target).norm(dim=-1)
 hit=d['contact_goal'][:,slot,part]&(torch.arange(len(d['done']))[:,None]<=stop)
 evaluate=torch.where(hit.any(0),hit.long().argmax(0),stop)
 assert observed[evaluate,slot].all(),'This audit needs real contacts, not FK fallback'
 loss=(dist[evaluate,slot]/.05).square()
 if metric.get('objective_parts'):
  expected=metric['objective_parts']['terminal_contact'];assert abs(float(loss.mean())-expected)<1e-5
 success=~d['term'][stop,slot]
 bytime=[]
 for t in evaluate.unique().tolist():
  m=evaluate==t;bytime.append(dict(frame=t+1,count=int(m.sum()),mean_contact_loss=float(loss[m].mean()),success_rate=float(success[m].float().mean())))
 row=dict(sampled_policy_update=metric['collected_policy_updates'],contact_loss=float(loss.mean()),mean_stop_frame=float((stop+1).float().mean()),mean_scoring_frame=float((evaluate+1).float().mean()),by_scoring_frame=bytime)
 result.append(row);saved.append(dict(data=d,stop=stop,evaluate=evaluate,loss=loss,observed=observed,dist=dist,success=success))
x,y=saved;same=x['evaluate']==y['evaluate'];same_stop=x['stop']==y['stop'];delta=y['loss']-x['loss'];at46=(x['evaluate']==45)&(y['evaluate']==45)
paired=dict(same_scoring_frame_fraction=float(same.float().mean()),same_stop_frame_fraction=float(same_stop.float().mean()),mean_loss_change=float(delta.mean()),change_from_same_frame=float(delta.masked_fill(~same,0).mean()),change_from_changed_frame=float(delta.masked_fill(same,0).mean()),both_frame46_count=int(at46.sum()),both_frame46_loss_before=float(x['loss'][at46].mean()),both_frame46_loss_after=float(y['loss'][at46].mean()))
fixed=[]
for t in [43,44,45,46,47,53]:
 if min(len(x['data']['done']),len(y['data']['done']))<=t:continue
 m=(x['stop']>=t)&(y['stop']>=t)&x['observed'][t]&y['observed'][t]
 fixed.append(dict(frame=t+1,paired_observed_count=int(m.sum()),loss_before=float((x['dist'][t,m]/.05).square().mean()) if m.any() else None,loss_after=float((y['dist'][t,m]/.05).square().mean()) if m.any() else None))
report=dict(scope='Last two saved fresh batches; all scoring contacts are native. Same-frame subsets exclude post-reset states; cannot infer unrecorded continuation after success.',batches=result,paired=paired,fixed_frames=fixed)
# Decompose same-frame point displacement into rigid body motion and a change
# of selected body-local contact witness; this is geometric, not causal proof.
from supermimic.lowstep import install
install(cfg['task_manifest'],contact_model='force6')
from supermimic.contact_control import ContactGeometry
from supermimic.mdp import unflatten_state
geo=ContactGeometry(demo,'cpu')
with torch.no_grad():
    k=45;slot=torch.arange(len(same));event=x['data']['goal_index'][0];end=clock.event_ends[event]
    part=(clock.target_masks[end-1]&~clock.keep_masks[end-1]).long().argmax(-1)
    poses=[];points=[];states=[]
    for v in [x,y]:
        state=unflatten_state(v['data']['s1'][k]);pos,rot=geo.poses(state)
        poses.append((pos[slot,part],rot[slot,part]));points.append(v['data']['point'][k,slot,part]);states.append(state)
    p0,R0=poses[0];p1,R1=poses[1];q0,q1=points
    local0=(R0.transpose(-1,-2)@(q0-p0).unsqueeze(-1)).squeeze(-1)
    transported=p1+(R1@local0.unsqueeze(-1)).squeeze(-1)
    goal=demo['point'][end-1][slot,part]
    describe=lambda v:dict(mean=float(v[at46].mean()),median=float(v[at46].median()),p90=float(torch.quantile(v[at46],.9)))
    report['same_frame46_geometry']=dict(samples=int(at46.sum()),
        body_origin_displacement_cm=describe((p1-p0).norm(dim=-1)*100),
        contact_point_displacement_cm=describe((q1-q0).norm(dim=-1)*100),
        old_witness_rigid_motion_cm=describe((transported-q0).norm(dim=-1)*100),
        witness_change_residual_cm=describe((q1-transported).norm(dim=-1)*100),
        loss_old_witness_at_new_pose=float(((transported[at46]-goal[at46])/.05).square().sum(-1).mean()),
        mean_joint_change_rad=float((states[1][4][at46]-states[0][4][at46]).square().mean().sqrt()))
report['trajectory_divergence']=[]
for k in [0,1,4,9,19,29,39,43,44,45]:
    live=(x['stop']>=k)&(y['stop']>=k)
    a0=x['data']['a'][k,live];a1=y['data']['a'][k,live]
    s0=unflatten_state(x['data']['s'][k,live]);s1=unflatten_state(y['data']['s'][k,live])
    f0=x['data']['xi'][k,live,34:52];f1=y['data']['xi'][k,live,34:52]
    report['trajectory_divergence'].append(dict(frame=k+1,count=int(live.sum()),
        state_max_difference=float((x['data']['s'][k,live]-y['data']['s'][k,live]).abs().max()),
        root_position_difference_cm=float((s1[0]-s0[0]).norm(dim=-1).mean()*100),
        action_rms_difference=float((a1-a0).square().mean().sqrt()),
        action_max_difference=float((a1-a0).abs().max()),
        normalized_force_rms_difference=float((f1-f0).square().mean().sqrt()),
        normalized_force_max_difference=float((f1-f0).abs().max())))
(a.run/'truncation-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
