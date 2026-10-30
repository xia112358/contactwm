"""Finite native probes along a saved Adam proposal; restores all actor weights."""
import json
import torch
from .contact_dmo import replay_loss,native_attempt_metrics
from .mdp import unflatten_state


def probe_update(path,actor,wm,prior,task,buffer,clock,geometry,context,normalize,default,collect,horizon,iteration,output,material_probe=False,contact_surface=None):
    proposal=torch.load(path,map_location=task.env.device,weights_only=False)
    base=proposal['base_actor'];trial=proposal['trial_actor']
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in base.items())
    results=[]
    baseline_stops=None
    fixed=[];baseline_data=None
    plan=proposal.get('probes', [dict(label=label,scale=scale,variant='adam') for label,scale in
        [('base_before',0.),('small_forward',.25),('full_forward',1.),('small_reverse',-.25),('full_reverse',-1.),('base_after',0.)]])
    try:
        for item in plan:
            label,scale=item['label'],item['scale']
            variant=item.get('variant','adam')
            selected=proposal.get('trial_actors',{}).get(variant,trial)
            actor.load_state_dict({k:v+scale*(selected[k]-v) for k,v in base.items()})
            task.reset(torch.arange(task.N,device=task.env.device))
            n=collect(horizon,iteration,until_first_done=True)
            rows=(buffer.ptr-n+torch.arange(n,device=task.env.device))%buffer.T
            data={k:v[rows] for k,v in buffer.data.items()}
            with torch.no_grad():
                loss,parts=replay_loss(actor,wm,clock,geometry,context,normalize,default,data,
                    prior_actor=prior,completion_mode=task.completion_mode,contact_surface=contact_surface)
                native=native_attempt_metrics(data,clock,task.target_faces_by_segment)
            stops=data['done'].long().argmax(0)
            if baseline_stops is None:baseline_stops=stops.clone()
            if material_probe:
                if baseline_data is None:baseline_data=data
                slot=torch.arange(task.N,device=task.env.device);frame=45
                event=data['goal_index'][0];end=clock.event_ends[event]
                new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
                assert bool((new.sum(-1)==1).all()),'Material probe currently requires one target part'
                part=new.long().argmax(-1)
                with torch.no_grad():
                    pos,rot=geometry.poses(unflatten_state(data['s1'][frame]))
                    fixed.append(dict(label=label,scale=scale,pos=pos[slot,part],rot=rot[slot,part],
                        point=data['point'][frame,slot,part],goal=clock.data['point'][end-1][slot,part],
                        live=(stops>=frame)&data['active'][frame,slot,part]&data['known'][frame,slot,part]))
            result=dict(label=label,scale=scale,variant=variant,loss=float(loss),parts=parts,native=native,
                same_stop_fraction=float((stops==baseline_stops).float().mean()))
            results.append(result)
            print('PROBE',json.dumps(result),flush=True)
            (output/'update-probe.json').write_text(json.dumps(dict(
                scope='Diagnostic native rollouts, no optimizer or WM updates; actor restored afterward. Actual objective includes discrete native stopping/contact selection.',
                proposal=proposal['report']['adam_proposal'],results=results),indent=2))
    finally:
        actor.load_state_dict(base)
    if material_probe:
        material_point_audit(actor,wm,prior,clock,geometry,context,normalize,default,
            baseline_data,fixed,base,trial,task.completion_mode,output)
    return results


def material_point_audit(actor,wm,prior,clock,geometry,context,normalize,default,data,fixed,base,trial,mode,output):
    """Same frame and same worlds: separate material transport from reselection."""
    torch.save([{k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in item.items()} for item in fixed],output/'material-point-samples.pt')
    flags=[p.requires_grad for p in actor.parameters()]
    try:
        actor.requires_grad_(True)
        probe=dict(frame=45)
        _,details=replay_loss(actor,wm,clock,geometry,context,normalize,default,data,
            prior_actor=prior,completion_mode=mode,endpoint_probe=probe)
        first=fixed[0];common=torch.stack([v['live'] for v in fixed]).all(0)
        assert int(common.sum())>0
        slot=torch.arange(len(common),device=common.device);event=data['goal_index'][0];end=clock.event_ends[event]
        part=(clock.target_masks[end-1]&~clock.keep_masks[end-1]).long().argmax(-1)
        pos,rot=geometry.poses(probe['state']);pos=pos[slot,part];rot=rot[slot,part]
        local=(first['rot'].transpose(-1,-2)@(first['point']-first['pos']).unsqueeze(-1)).squeeze(-1)
        point=pos+(rot@local.unsqueeze(-1)).squeeze(-1)
        assert float((point.detach()-first['point']).abs().max())<1e-5
        objective=((point[common]-first['goal'][common])/.05).square().sum(-1).mean()
        delta=torch.cat([(trial[k]-base[k]).flatten() for k,_ in actor.named_parameters()])
        params=list(actor.parameters())
        def directional(value,retain):
            grad=torch.autograd.grad(value,params,retain_graph=retain,allow_unused=True)
            flat=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(grad,params)])
            return float(torch.dot(flat,delta))
        displacement=[directional(point[common,j].mean(),True) for j in range(3)]
        predicted=directional(objective,False)
        baseline_loss=float(objective.detach());rows=[]
        for v in fixed:
            material=v['pos']+(v['rot']@local.unsqueeze(-1)).squeeze(-1)
            L=lambda p:float(((p[common]-v['goal'][common])/.05).square().sum(-1).mean())
            lm,ln=L(material),L(v['point'])
            e0=first['point'][common]-v['goal'][common]
            dm=material[common]-first['point'][common];dn=v['point'][common]-first['point'][common]
            linear=lambda d:float(2*(e0*d).sum(-1).mean()/.05**2)
            quadratic=lambda d:float(d.square().sum(-1).mean()/.05**2)
            rows.append(dict(label=v['label'],scale=v['scale'],material_loss=lm,native_loss=ln,
                material_linear_change=linear(dm),material_quadratic_change=quadratic(dm),
                native_linear_change=linear(dn),native_quadratic_change=quadratic(dn),
                predicted_loss=baseline_loss+v['scale']*predicted,
                dynamics_and_finite_step_residual=lm-baseline_loss-v['scale']*predicted,
                contact_reselection_residual=ln-lm,
                material_point_mean_displacement_cm=((material-first['point'])[common].mean(0)*100).tolist(),
                native_point_mean_displacement_cm=((v['point']-first['point'])[common].mean(0)*100).tolist(),
                material_to_native_rms_cm=float((v['point'][common]-material[common]).square().sum(-1).mean().sqrt()*100)))
        report=dict(scope='Frame 46 only, identical cohort observed and not reset in every trial. Baseline body-local witness fixed per world. Gradients recomputed on this same baseline cohort. Finite step residual includes WM derivative error, physical nonlinearities, and simulation variability.',
            common_worlds=int(common.sum()),total_worlds=len(common),baseline_material_loss=baseline_loss,
            predicted_loss_change_per_full_step=predicted,
            predicted_mean_displacement_cm_per_full_step=[v*100 for v in displacement],rows=rows)
        (output/'material-point-audit.json').write_text(json.dumps(report,indent=2))
        print('MATERIAL_AUDIT',json.dumps(report),flush=True)
    finally:
        for p,flag in zip(actor.parameters(),flags):p.requires_grad_(flag)
