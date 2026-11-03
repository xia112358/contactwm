"""Read-only gradient and feedback audit on frozen-policy native replay."""
import argparse,json,sys,copy
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'config.json').read_text())
from supermimic.lowstep import install
spec=install(cfg['task_manifest'],contact_model='force6')
from supermimic.contact_control import ContactActor,ContactClock,ContactGeometry,observation
from supermimic.contact_dmo import replay_loss
from supermimic.lowstep_observations import HeightForceContext
from supermimic.networks import WorldModel
from supermimic import networks
from supermimic.mdp import unflatten_state,normalize_xi
from supermimic.robot import default_joint_pos_list
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;dev='cuda'
ck=torch.load(cfg['resume'],map_location=dev,weights_only=False);source=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
actor=ContactActor().to(dev);actor.load_state_dict(ck['actor']);wm=WorldModel().to(dev);wm.load_state_dict(ck['wm'])
prior=copy.deepcopy(actor).requires_grad_(False);prior.load_state_dict(torch.load(cfg['prior_checkpoint'],map_location=dev,weights_only=False)['actor'])
demo=torch.load(cfg['recording'],map_location=dev,weights_only=False)['data'];clock=ContactClock(demo,hold=1,role_aware=True);geometry=ContactGeometry(demo,dev);geometry.load_state_dict(ck['geometry']);context=HeightForceContext(spec['tasks']);default=torch.tensor(default_joint_pos_list(),device=dev)
buf=torch.load(a.run/'replay.pt',map_location=dev,weights_only=False);rows=(buf['ptr']-buf['size']+torch.arange(buf['size'],device=dev))%buf['T'];data={k:v[rows] for k,v in buf['data'].items()};starts=(data['ep_step']==0).all(-1).nonzero().flatten().tolist();blocks=[{k:v[s:e] for k,v in data.items()} for s,e in zip(starts,starts[1:]+[len(rows)])];assert len(blocks)>=2;x,y=blocks[-2:]
report=dict(checkpoint=cfg['resume'],scope='Read-only, same frozen actor and same replay. Counterfactual force swaps are actor-input diagnostics, not simulated interventions.')
feedback=[]
with torch.no_grad():
 for t in [0,19,29,39,43,45]:
  hs=[];cs=[];preds=[]
  for d in [x,y]:
   idx=torch.arange(t-actor.hist+1,t+1,device=dev).clamp_min(0)
   h=observation(unflatten_state(d['s'][idx]),d['a_prev'][idx],d['xi'][idx],default,context).transpose(0,1)
   event=d['goal_index'][0];c=clock.condition(unflatten_state(d['s'][t]),torch.tensor(clock.bounds[:-1],device=dev)[event],ends=clock.event_ends[event])
   pred=actor(h,c);assert float((pred-d['a'][t]).abs().max())<1e-4
   hs.append(h);cs.append(c);preds.append(pred)
  mixed=hs[0].clone();mixed[:,:,-18:]=hs[1][:,:,-18:];force_only=actor(mixed,cs[0])-preds[0]
  mixed=hs[1].clone();mixed[:,:,-18:]=hs[0][:,:,-18:];other_only=actor(mixed,cs[1])-preds[0]
  difference=preds[1]-preds[0];rms=lambda v:float(v.square().mean().sqrt())
  feedback.append(dict(frame=t+1,full_action_rms=rms(difference),force_only_action_rms=rms(force_only),other_inputs_action_rms=rms(other_only),force_projection_on_full=float((force_only*difference).sum()/difference.square().sum().clamp_min(1e-30))))
report['feedback']=feedback;print('Feedback',json.dumps(feedback),flush=True)
params=list(actor.parameters());flatten=lambda gs:torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,params)])
cos=lambda a,b:float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
parts={};loss,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,y,prior_actor=prior,completion_mode=cfg['completion_mode'],component_losses=parts)
gradients={k:flatten(torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)) for k,v in parts.items()}
g=flatten(torch.autograd.grad(loss,params));report['loss_parts']={k:float(v.detach()) for k,v in parts.items()};report['gradient_norms']={k:float(v.norm()) for k,v in gradients.items()};report['gradient_norms']['total']=float(g.norm());report['gradient_cosines']={a+'__'+b:cos(gradients[a],gradients[b]) for a,b in [('contact','body'),('contact','regularization'),('body','regularization')]}
for k,v in gradients.items():report['gradient_cosines'][k+'__total']=cos(v,g)
wm.load_state_dict(source['wm']);loss0,_=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,y,prior_actor=prior,completion_mode=cfg['completion_mode']);g0=flatten(torch.autograd.grad(loss0,params));assert abs(float(loss.detach()-loss0.detach()))<1e-6
report['wm_gradient_comparison']=dict(original_norm=float(g0.norm()),updated_norm=float(g.norm()),cosine=cos(g0,g),same_forward_loss=float(loss.detach()))
opt=torch.optim.Adam(params);opt.load_state_dict(ck['actor_optimizer']);before=torch.cat([p.detach().flatten() for p in params]);cursor=0
for p in params:p.grad=g[cursor:cursor+p.numel()].reshape_as(p).clone();cursor+=p.numel()
torch.nn.utils.clip_grad_norm_(params,1.);opt.step();after=torch.cat([p.detach().flatten() for p in params]);delta=after-before
report['adam_proposal']=dict(lr=opt.param_groups[0]['lr'],step_norm=float(delta.norm()),step_negative_gradient_cosine=cos(delta,-g),predicted_linear_loss_change=float(torch.dot(g,delta)))
report['predicted_change_by_part']={k:float(torch.dot(v,delta)) for k,v in gradients.items()}
report['original_wm_predicted_change']=float(torch.dot(g0,delta))
# Trial weights only, never promoted to the accepted baseline or a training checkpoint.
torch.save(dict(base_actor=ck['actor'],trial_actor=actor.state_dict(),source_checkpoint=cfg['resume'],gradient=g.cpu(),delta=delta.cpu(),report=report),a.run/'diagnostic-adam-proposal.pt')
(a.run/'gradient-sources.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
