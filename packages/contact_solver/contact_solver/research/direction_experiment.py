"""Protected AL descent in pose and network spaces, without pose teachers."""
import json
import time
import torch
from contact_solver.constraint_direction import project_direction
from contact_solver.constraint_learning import AugmentedLagrangian,violation_metrics
from contact_solver.feasibility_filter import FeasibilityFilter,ProtectionRule
from contact_solver.pose_coordinates import pose,coordinates
from .step14_gradient_field import plain,cpu,InvalidWitnessNormal


def experiment(c,model,inputs,base,checkpoint,cfg,evaluate,reference):
    out=c['out'];scale=c['scale'];results={};traces={};attempts={}
    if cfg.get('self_pair_certificates',False):
        from .self_pair_certificate import install
        install(c)
    guard=FeasibilityFilter([ProtectionRule(**r) for r in cfg['filter_rules']])
    parts=c['initial']['contact_part_mask'].nonzero().flatten().tolist()
    surfaces=c['initial']['contact_surface']
    required={str(p):True for p in parts}
    def evidence(raw):return {str(p):bool(raw['contact_part_mask'][p] and raw['contact_surface'][p]==surfaces[p]) for p in parts}
    def protected(rows):
        values=[]
        for row in rows:
            rules=[r for r in guard.rules if row.constraint_id.startswith(r.prefix)]
            if not rules:continue
            tolerance=min(r.tolerance for r in rules)
            for i,component in enumerate(row.component_ids):
                value=row.value[i]
                # Each signed equality side has its own outward normal.
                for sign in ((1.,-1.) if row.kind=='eq' else (1.,)):
                    values.append(((*row.key,component,sign),sign*value,tolerance))
        return values
    for method in cfg['direction_methods']:
        network=method.startswith('network');projected=method.endswith('projected')
        if network and cfg.get('network_float64',False):
            model.double()
            inputs={k:(v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in inputs.items()}
        model.load_state_dict(checkpoint['model']);al=AugmentedLagrangian(rho=cfg['rho'],max_rho=1e3)
        z=torch.zeros_like(scale,requires_grad=True);q=reference.clone()
        if network:
            with torch.no_grad():q=model(**inputs).qpos[0].double()
        initial_precision_shift=float((q-reference).abs().max())
        if initial_precision_shift>1e-4:raise ValueError('Precision conversion changed the network fixture')
        raw=c['observe'](q,method+'_initial')
        trace=[];trials=[];forced=set();status='step_budget'
        c['record'](method+'_initial',q,raw)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();starting_memory=torch.cuda.memory_allocated();durations=[];max_active=0
        first_query_id=raw['query_id']
        for step in range(1,cfg['steps']+1):
            step_started=time.perf_counter()
            # Geometry derivatives live in a small local chart. For the network
            # arm, pull them back through its actual output map before projecting.
            local=coordinates(q,reference,scale).detach().requires_grad_()
            prior,rows=evaluate(pose(local,reference,scale),raw)
            value=al.loss(prior,rows)
            g=torch.autograd.grad(value,local,retain_graph=True)[0]
            guard_rows=c['protection_residuals'](pose(local,reference,scale),raw) if 'protection_residuals' in c else rows
            monitored=protected(guard_rows)
            active=[(key,v,t) for key,v,t in monitored if float(v.detach())>=
                    (t*(1-cfg['active_slack_fraction']) if t>0 else -cfg.get('zero_tolerance_activation_slack',0.)) or key in forced]
            jac=torch.stack([torch.autograd.grad(v,local,retain_graph=True)[0] for _,v,_ in active]) if active else local.new_zeros((0,len(local)))
            if network:
                graph=coordinates(model(**inputs).qpos[0].double(),reference,scale)
                params=[p for p in model.parameters() if p.requires_grad]
                def pullback(v):
                    gradients=torch.autograd.grad(graph,params,grad_outputs=v,retain_graph=True,allow_unused=True)
                    return torch.cat([(torch.zeros_like(p) if d is None else d).detach().flatten() for p,d in zip(params,gradients)])
                grad=pullback(g);normals=torch.stack([pullback(j) for j in jac]) if len(jac) else grad.new_zeros((0,len(grad)))
            else:
                params=[z];grad=g.detach();normals=jac.detach()
            max_active=max(max_active,len(active))
            start_eta=cfg['learning_rate'] if network else .005
            reserve=cfg.get('interior_reserve_fraction',0.)
            def inward_bound(v,t):
                if t==0 and cfg.get('zero_tolerance_interior_reserve',False):
                    slack=cfg.get('zero_tolerance_activation_slack',0.)
                    return -min(reserve*slack,max(0.,float(v.detach())+slack))/start_eta
                return -min(reserve*t,max(0.,float(v.detach())-t*(1-cfg['active_slack_fraction'])))/start_eta
            bounds=grad.new_tensor([inward_bound(v,t) for _,v,t in active])
            direction,info=project_direction(-grad,normals,bounds=bounds,algorithm=cfg.get('projection_algorithm','coordinate')) if projected else (-grad,dict(rows=len(active),converged=True))
            if cfg.get('relax_infeasible_reserve',False) and not info['converged'] and (info.get('feasibility') or {}).get('status')==2:
                # An interior reserve is a heuristic, not contact truth. If its
                # halfspaces are infeasible, keep non-worsening constraints and
                # the identical nonlinear/evidence guards. Record both solves.
                reserve_failure=info
                direction,info=project_direction(-grad,normals,bounds=torch.zeros_like(bounds),algorithm=cfg['projection_algorithm'])
                info['infeasible_reserve_attempt']=reserve_failure
            if cfg.get('require_converged_projection',False) and not info['converged']:
                status='projection_not_converged'
                trace.append(dict(step=step,accepted=False,event=status,projection=info))
                break
            slope=float(grad.double()@direction.double())
            weights=[p.detach().clone() for p in params]
            chunks=list(direction.split([p.numel() for p in params]))
            before=float(value.detach());accepted=False;blocked_keys=set();last_prediction=[]
            for trial in range(24):
                eta=start_eta*.5**trial
                with torch.no_grad():
                    for p,w,d in zip(params,weights,chunks):p.copy_(w+eta*d.reshape_as(p))
                    if network:pred=model(**inputs);candidate=pred.qpos[0].double()
                    else:candidate=pose(z,reference,scale)
                fresh=c['observe'](candidate,f'{method}_{step}_{trial}')
                try:
                    next_prior,next_rows=evaluate(candidate,fresh);number=float(al.loss(next_prior,next_rows))
                except InvalidWitnessNormal as exc:
                    trials.append(dict(step=step,trial=trial,accepted=False,error=str(exc),query_id=fresh['query_id']));continue
                next_guard_rows=c['protection_residuals'](candidate,fresh) if 'protection_residuals' in c else next_rows
                decision=guard.assess(guard_rows,next_guard_rows,required_evidence=required,candidate_evidence=evidence(fresh))
                moved=float((candidate-q).abs().max())>1e-9
                armijo=slope<0 and number<before and number<=before+1e-4*eta*slope
                accepted=armijo and moved and decision['accepted']
                next_map={key:float(v.detach()) for key,v,_ in protected(next_guard_rows)}
                delta=coordinates(candidate,reference,scale)-local.detach()
                last_prediction=[dict(key=list(key),before=float(v.detach()),linear_change=float(j@delta),actual_change=next_map.get(key,float('nan'))-float(v.detach()),variable_space_linear_change=eta*float(normal.double()@direction.double())) for (key,v,_),j,normal in zip(active,jac,normals)]
                trials.append(dict(step=step,trial=trial,eta=eta,query_id=fresh['query_id'],accepted=accepted,armijo=armijo,guard_accepted=decision['accepted'],moved=moved,loss_before=before,loss_after=number,reasons=decision['reasons']))
                if accepted:q,raw=candidate.detach(),fresh;break
                for reason in decision['reasons']:
                    if reason['kind']=='protected_residual_worsened':
                        key=tuple(reason['key'])
                        for signed_key,v,_ in monitored:
                            if signed_key[:-1]==key and next_map.get(signed_key,-1)>0:blocked_keys.add(signed_key)
            if not accepted:
                with torch.no_grad():
                    for p,w in zip(params,weights):p.copy_(w)
                torch.cuda.synchronize();durations.append(time.perf_counter()-step_started)
                if projected and blocked_keys-forced:
                    forced.update(blocked_keys)
                    trace.append(dict(step=step,accepted=False,event='activate_blockers',keys=[list(k) for k in blocked_keys],projection=info,prediction=last_prediction))
                    continue
                status='no_acceptable_step'
                trace.append(dict(step=step,accepted=False,event=status,projection=info,slope=slope,prediction=last_prediction))
                break
            # Forced blockers are local, not permanent equality locks.
            forced.clear()
            torch.cuda.synchronize();durations.append(time.perf_counter()-step_started)
            count=sum(t.get('accepted',False) for t in trace)+1
            trace.append(dict(step=step,accepted=True,eta=eta,query_id=raw['query_id'],loss_before=before,loss_after=number,projection=info,slope=slope,active_count=len(active),prediction=last_prediction,
                required_contacts_kept=all(evidence(raw).values()),plan_unchanged=True if not network else bool(torch.equal(pred.conditioned_contact,base.conditioned_contact) and torch.equal(pred.cell,base.cell)),metrics=violation_metrics(next_rows)))
            if count%cfg['dual_update_every']==0:al.update(next_rows)
            if count==1 or count%20==0:
                c['record'](f'{method}_step{step}',q,raw)
                print('DIRECTION_PROGRESS',method,step,count,flush=True)
            (out/'direction_progress.json').write_text(json.dumps(plain(dict(method=method,results=results,trace=trace)),indent=2))
        final=c['record'](method+'_final',q,raw);prior,rows=evaluate(q,raw)
        final.update(initial_precision_shift=initial_precision_shift,parameter_dtype=str(next(model.parameters()).dtype),performance=dict(wall_seconds=time.perf_counter()-started,step_seconds=durations,query_count=(max(t['query_id'] for t in trials)-first_query_id if trials else 0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),initial_allocated_bytes=starting_memory,parameter_count=sum(p.numel() for p in model.parameters()),max_active_rows=max_active),status=status,accepted_steps=sum(t.get('accepted',False) for t in trace),trial_count=len(trials),metrics=violation_metrics(rows),prior=float(prior))
        if network:
            fitted={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
            artifact=dict(schema='constraint_network_fit_v1',model=fitted,model_config=checkpoint['config'],inputs=cpu(inputs),q_final=q.detach().cpu(),robot_asset_json=c['saved']['robot_asset_json'],experiment_config=cfg)
            torch.save(artifact,out/(method+'_model.pt'))
            # Reload actual learned parameters, then use a plain forward pass.
            model.load_state_dict(checkpoint['model'])
            reloaded_artifact=torch.load(out/(method+'_model.pt'),map_location='cpu',weights_only=False)
            model.load_state_dict(reloaded_artifact['model'])
            with torch.no_grad():reloaded=model(**inputs).qpos[0].double()
            final['checkpoint_reload_max_q_error']=float((reloaded-q).abs().max())
            if final['checkpoint_reload_max_q_error']>1e-6:raise ValueError('Saved network does not reproduce its accepted final output')
        results[method]=final;traces[method]=trace;attempts[method]=trials
        result=dict(schema='constraint_direction_experiment_v1',config=cfg,results=results,traces=traces,attempts=attempts,records=c['records'],forward_audit=c.get('filter_forward_audit'),terrain=c['saved']['terrain'],robot_asset_json=c['saved']['robot_asset_json'],joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
            contract='AL objective and identical nonlinear guards. Projected arms modify update directions using local protected halfspaces in their optimization variable space. No QP pose teacher. Projection is an optimizer operation, not an additional scalar loss. Finite trials do not establish infeasibility.')
        (out/'diagnostic.json').write_text(json.dumps(plain(result),indent=2))
    print('DIRECTION_EXPERIMENT_COMPLETE',flush=True)
