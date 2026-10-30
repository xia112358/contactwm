"""Isolate feasibility, primal convergence and network metric; QP is audit only."""
import json
from pathlib import Path
import torch
from contact_solver.constraint_learning import AugmentedLagrangian,violation_metrics
from contact_solver.pose_coordinates import coordinates,pose
from .step14_gradient_field import plain,InvalidWitnessNormal


def diagnose(c,model,inputs,base,checkpoint,cfg,evaluate,reference,task_specs):
    out=c['out'];scale=c['scale'];sample=cfg['cases'][0]['id'];results={};traces={};directions={};dual_logs={}
    initial=c['observe'](reference,'diagnostic_reference')
    required=c['initial']['contact_part_mask']
    required_surfaces=c['initial']['contact_surface']
    keep_prefixes=tuple('contact/'+spec['id']+'/' for spec,_,task in task_specs if task.mode=='keep')
    def kept(raw):return bool(((raw['contact_part_mask'] & (raw['contact_surface']==required_surfaces)) | ~required).all())
    def stats(rows,al):
        state={tuple(e['key']):e for e in al.state_dict()['entries']};data={}
        for r in rows:
            entry=state.get(r.key);rho=cfg['rho'] if entry is None else entry['rho']
            dual=r.value.new_tensor([0. if entry is None else entry['dual'].get(i,0.) for i in r.component_ids])
            coefficient=dual+rho*r.value
            if r.kind=='le':coefficient=coefficient.relu()
            data[r.constraint_id]=dict(kind=r.kind,al_active_components=int((coefficient.detach().abs()>0).sum()),
                components=r.value.numel(),rho=rho,multiplier_max=float(dual.abs().max()),
                violation_max=float((r.value.abs() if r.kind=='eq' else r.value.relu()).max().detach()))
        return data
    # This endpoint is never used in loss, initialization or parameter updates.
    old=json.loads(Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_step14_static_interval_20260925/diagnostic.json').read_text())
    if old['robot_asset_json']!=c['saved']['robot_asset_json']:raise ValueError('Audit asset mismatch')
    qp=reference.new_tensor(next(r['q'] for r in old['records'] if r['label']=='face3_iter5'))
    compatibility={}
    for label,q in [('initial',reference),('qp_audit_only',qp)]:
        raw=c['observe'](q,label);prior,rows=evaluate(q,raw)
        record=c['record'](label,q,raw)
        compatibility[label]=dict(prior=float(prior),metrics=violation_metrics(rows),
            rows={r.constraint_id:dict(kind=r.kind,value=plain(r.value),component_ids=r.component_ids) for r in rows},
            old_static_pass=record['static_recovery_pass'])
    (out/'compatibility.json').write_text(json.dumps(plain(compatibility),indent=2))
    def breakdown(q,raw,al,actual_delta):
        z=coordinates(q,reference,scale).detach().requires_grad_()
        prior,rows=evaluate(pose(z,reference,scale),raw)
        groups={'prior':prior}
        for r in rows:
            if r.constraint_id.startswith(keep_prefixes):name='keep'
            elif r.constraint_id.startswith('contact/'):name='new_contact'
            elif r.constraint_id.startswith('collision/'):name='collision'
            else:name='limits'
            groups[name]=groups.get(name,z.sum()*0)+al.loss(z.sum()*0,[r])
        keep_rows=[r for r in rows if r.constraint_id.startswith(keep_prefixes)]
        keep_error=sum(.5*(r.value if r.kind=='eq' else r.value.relu()).square().sum() for r in keep_rows)
        kgrad=torch.autograd.grad(keep_error,z,retain_graph=True)[0]
        terms={}
        for name,value in groups.items():
            grad=torch.autograd.grad(value,z,retain_graph=True)[0]
            terms[name]=dict(gradient=plain(grad),gradient_norm=float(grad.norm()),
                keep_error_derivative_along_negative_gradient=float(-kgrad@grad))
        # Positive derivative means this direction worsens current keep residual.
        return dict(terms=terms,keep_error=float(keep_error.detach()),
            keep_error_derivative_along_actual_step=float(kgrad@actual_delta),
            actual_delta=plain(actual_delta),states=stats(rows,al))
    budget=240
    for space in ('pose','network'):
        for schedule in ('every10','stationary'):
            label=f'{space}_{schedule}';al=AugmentedLagrangian(rho=cfg['rho'],max_rho=1e3)
            model.load_state_dict(checkpoint['model'])
            z=torch.zeros(len(scale),device=reference.device,dtype=reference.dtype,requires_grad=True)
            q=reference.clone();raw=c['observe'](q,label+'_initial');trace=[];events=[];directions[label]=[]
            inner=0;block_initial_grad=None;status='step_budget';lost_seen=False
            def objective(q,observed):
                prior,rows=evaluate(q,observed)
                return al.loss(prior,rows),rows
            for step in range(1,budget+1):
                if space=='pose':
                    q_graph=pose(z,reference,scale);value,_=objective(q_graph,raw)
                    gradient=torch.autograd.grad(value,z)[0];params=[z];grads=[gradient]
                else:
                    model.zero_grad(set_to_none=True);q_graph=model(**inputs).qpos[0].double();value,_=objective(q_graph,raw);value.backward()
                    params=[p for p in model.parameters() if p.grad is not None];grads=[p.grad.detach().clone() for p in params]
                norm2=sum(float(g.double().square().sum()) for g in grads);gnorm=norm2**.5
                if block_initial_grad is None:block_initial_grad=gnorm
                # The same relative criterion in each space; raw gradient norms
                # across pose coordinates and network parameters are not comparable.
                stationary=gnorm<=max(1e-3,.01*block_initial_grad)
                if schedule=='stationary' and inner>=10 and stationary:
                    _,rows=evaluate(q,raw);al.update(rows)
                    events.append(dict(before_step=step,reason='gradient_tolerance',inner_steps=inner,gradient_norm=gnorm,block_initial_gradient=block_initial_grad))
                    inner=0;block_initial_grad=None
                    continue
                weights=[p.detach().clone() for p in params];accepted=False
                start_eta=.005 if space=='pose' else 1e-7
                before=q.detach().clone();raw_before=raw
                for trial in range(22):
                    eta=start_eta*.5**trial
                    with torch.no_grad():
                        for p,w,g in zip(params,weights,grads):p.copy_(w-eta*g)
                        if space=='pose':candidate=pose(z,reference,scale)
                        else:pred=model(**inputs);candidate=pred.qpos[0].double()
                    fresh=c['observe'](candidate,f'{label}_{step}_{trial}')
                    try:next_value,rows=objective(candidate,fresh);number=float(next_value)
                    except InvalidWitnessNormal:continue
                    if number<=float(value.detach())-1e-4*eta*norm2:
                        q,raw=candidate.detach(),fresh;accepted=True;break
                if not accepted:
                    with torch.no_grad():
                        for p,w in zip(params,weights):p.copy_(w)
                    status='line_search_stalled_without_stationarity' if not stationary else 'stationary_line_search_stalled';break
                inner+=1;keep=kept(raw)
                delta=coordinates(q,reference,scale)-coordinates(before,reference,scale)
                if step==1 or step%40==0 or (not keep and not lost_seen):
                    details=breakdown(before,raw_before,al,delta)
                    details.update(step=step,query_before=raw_before['query_id'],query_after=raw['query_id'],actual_contact_after=plain(raw['contact_part_mask']),all_required_kept_after=keep)
                    directions[label].append(details)
                lost_seen|=not keep
                trace.append(dict(step=step,loss_before=float(value.detach()),loss_after=number,eta=eta,gradient_norm=gnorm,
                    relative_block_gradient=gnorm/max(block_initial_grad,1e-30),query_id=raw['query_id'],all_required_kept=keep,
                    actual_contact=plain(raw['contact_part_mask']),constraint_states=stats(rows,al),
                    plan_unchanged=True if space=='pose' else bool(torch.equal(pred.conditioned_contact,base.conditioned_contact) and torch.equal(pred.cell,base.cell))))
                if schedule=='every10' and inner==10:
                    al.update(rows);events.append(dict(after_step=step,reason='fixed_schedule',inner_steps=inner,gradient_norm=gnorm,block_initial_gradient=block_initial_grad,stationary=stationary))
                    inner=0;block_initial_grad=None
                if step%40==0 or step==1 or step==budget:
                    record=c['record'](f'{label}_step{step}',q,raw)
                    record.update(gradient_norm=gnorm,relative_block_gradient=trace[-1]['relative_block_gradient'],dual_updates=len(events))
                    (out/'progress_diagnostics.json').write_text(json.dumps(plain(dict(results=results,trace=trace,directions=directions,dual_events=events)),indent=2))
            final=c['record'](label+'_final',q,raw);prior,rows=evaluate(q,raw)
            final.update(status=status,steps=len(trace),dual_updates=len(events),prior=float(prior),constraint_metrics=violation_metrics(rows),
                last_gradient_norm=trace[-1]['gradient_norm'] if trace else None,last_relative_gradient=trace[-1]['relative_block_gradient'] if trace else None,
                primal_convergence_claimed=False,remaining_inner_steps=inner)
            results[label]=final;traces[label]=trace;dual_logs[label]=events
            (out/(label+'_multipliers.json')).write_text(json.dumps(al.state_dict(),indent=2))
    result=dict(schema='constraint_diagnostics_v1',config=cfg,budget_per_run=budget,compatibility=compatibility,results=results,
        traces=traces,directions=directions,dual_events=dual_logs,records=c['records'],terrain=c['saved']['terrain'],
        robot_asset_json=c['saved']['robot_asset_json'],joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
        contract='QP endpoint is evaluated only, never a target or initializer. All four runs use identical full residuals and prior. Stationary schedule updates duals only at recorded gradient tolerance, otherwise retains multipliers. Finite budget is not proof of convergence or infeasibility.')
    (out/'diagnostic.json').write_text(json.dumps(plain(result),indent=2));print('CONSTRAINT_DIAGNOSTICS_COMPLETE',flush=True)
