"""Decompose the unchanged DMO objective and gradients on one frozen replay."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,copy,json,sys,time
from pathlib import Path
pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    import torch
    cfg=json.loads((a.run/'config.json').read_text())
    from supermimic.lowstep import install
    spec=install(cfg['task_manifest'],contact_model='force6')
    from supermimic.contact_control import ContactActor,ContactClock,ContactGeometry
    from supermimic.contact_dmo import replay_loss
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.networks import WorldModel
    from supermimic import networks
    from supermimic.mdp import normalize_xi
    from supermimic.robot import default_joint_pos_list
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;device='cuda:0';started=time.perf_counter()
    ck=torch.load(cfg['resume'],map_location=device,weights_only=False)
    source=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
    actor=ContactActor().to(device);actor.load_state_dict(ck['actor']);actor.eval()
    wm=WorldModel().to(device);wm.load_state_dict(ck['wm']);wm.eval()
    prior=copy.deepcopy(actor).requires_grad_(False);prior.load_state_dict(torch.load(cfg['prior_checkpoint'],map_location=device,weights_only=False)['actor'])
    demo=torch.load(cfg['recording'],map_location=device,weights_only=False)['data'];clock=ContactClock(demo,hold=1,role_aware=True)
    geometry=ContactGeometry(demo,device);geometry.load_state_dict(ck['geometry']);context=HeightForceContext(spec['tasks']);default=torch.tensor(default_joint_pos_list(),device=device)
    buf=torch.load(a.run/'replay.pt',map_location=device,weights_only=False)
    rows=(buf['ptr']-buf['size']+torch.arange(buf['size'],device=device))%buf['T']
    data={k:v[rows] for k,v in buf['data'].items()};del buf
    starts=(data['ep_step']==0).all(-1).nonzero().flatten();assert len(starts)
    data={k:v[int(starts[-1]):] for k,v in data.items()}
    assert data['done'].any(0).all()
    params=list(actor.parameters())
    def flatten(gs):return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
    def cosine(x,y):return float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-30))
    components={};per_attempt={};coarse={}
    loss,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,data,
        prior_actor=prior,prior_weight=cfg['prior_weight'],pose_weight=cfg['pose_weight'],completion_mode=cfg['completion_mode'],
        check=True,component_losses=coarse,detailed_losses=components,per_attempt_losses=per_attempt)
    values={k:float(v.detach()) for k,v in components.items()}
    assert abs(sum(values.values())-float(loss.detach()))<2e-6
    report=dict(checkpoint=cfg['resume'],policy_updates=ck['policy_updates'],trajectories=data['done'].shape[1],scope='One frozen actor, frozen updated WM and same native batch; original loss unchanged. Linear projections are WM-based diagnostics, not actual simulator improvement.',loss=float(loss.detach()),components=values,forward_checks=details)
    group=data['goal_index'][0]
    report['segments']=[dict(segment=g+1,n=int((group==g).sum()),components={k:float(v[group==g].mean().detach()) for k,v in per_attempt.items()}) for g in group.unique().tolist()]
    print('Loss split',json.dumps(values),flush=True)
    gradients={}
    for name,value in components.items():
        gradients[name]=flatten(torch.autograd.grad(value,params,retain_graph=True,allow_unused=True))
        print('Gradient',name,float(gradients[name].norm()),'elapsed',time.perf_counter()-started,flush=True)
    total=flatten(torch.autograd.grad(loss,params,allow_unused=True))
    summed=sum(gradients.values());relative=float((summed-total).norm()/total.norm().clamp_min(1e-30));assert relative<(0.01 if networks.AMP_DTYPE is not None else 1e-4),relative
    body=sum(gradients[k] for k in ('root_position','root_orientation','root_velocity'))
    regular=sum(gradients[k] for k in ('dagger_prior','action_smoothness','action_magnitude'))
    task=gradients['contact']+body
    report['gradient_sum_relative_error']=relative
    report['amp_dtype']=str(networks.AMP_DTYPE)
    # bf16 reverse passes round each component separately. Verify the optional
    # diagnostic calculations leave the original total objective unchanged.
    baseline_loss,_=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default,data,prior_actor=prior,
        prior_weight=cfg['prior_weight'],pose_weight=cfg['pose_weight'],completion_mode=cfg['completion_mode'])
    baseline_gradient=flatten(torch.autograd.grad(baseline_loss,params,allow_unused=True))
    report['diagnostic_vs_original_loss_error']=abs(float(baseline_loss.detach())-report['loss'])
    report['diagnostic_vs_original_gradient_relative_error']=float((baseline_gradient-total).norm()/total.norm().clamp_min(1e-30))
    assert report['diagnostic_vs_original_loss_error']<1e-6
    assert report['diagnostic_vs_original_gradient_relative_error']<1e-4
    report['gradient_norms']={k:float(v.norm()) for k,v in gradients.items()};report['gradient_norms'].update(total=float(total.norm()),task=float(task.norm()),body=float(body.norm()),regularization=float(regular.norm()))
    report['gradient_cosines']={x+'__'+y:cosine(gradients[x],gradients[y]) for i,x in enumerate(gradients) for y in list(gradients)[i+1:]}
    report['gradient_cosines'].update(task__dagger_prior=cosine(task,gradients['dagger_prior']),contact__body=cosine(gradients['contact'],body),task__regularization=cosine(task,regular))
    report['cancellation_ratio']=float(total.norm()/sum(v.norm() for v in gradients.values()))
    proposals=[]
    for scale in [1.,.5,.25,0.]:
        gradient=total-(1-scale)*gradients['dagger_prior']
        trial=[torch.nn.Parameter(p.detach().clone()) for p in params]
        opt=torch.optim.Adam(trial);opt.load_state_dict(copy.deepcopy(ck['actor_optimizer']))
        cursor=0
        for p in trial:p.grad=gradient[cursor:cursor+p.numel()].view_as(p).clone();cursor+=p.numel()
        torch.nn.utils.clip_grad_norm_(trial,1.);opt.step()
        delta=torch.cat([(p.detach()-base.detach()).flatten() for p,base in zip(trial,params)])
        proposals.append(dict(prior_scale=scale,learning_rate=opt.param_groups[0]['lr'],gradient_norm=float(gradient.norm()),cosine_with_original_gradient=cosine(gradient,total),step_norm=float(delta.norm()),predicted_task_change=float(torch.dot(task,delta)),predicted_components={k:float(torch.dot(v,delta)) for k,v in gradients.items()}))
    report['same_momentum_adam_proposals']=proposals
    report['actor_unchanged']=all(torch.equal(v,ck['actor'][k]) for k,v in actor.state_dict().items());report['wm_unchanged']=all(torch.equal(v,ck['wm'][k]) for k,v in wm.state_dict().items())
    assert report['actor_unchanged'] and report['wm_unchanged']
    report['elapsed_s']=time.perf_counter()-started
    (a.run/'loss-components.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    torch.save({k:v.cpu() for k,v in gradients.items()},a.run/'component-gradients.pt')

if __name__=='__main__':main()
