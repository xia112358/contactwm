"""Same AL gradient and trial steps; only candidate acceptance differs."""
import json
import torch
from contact_solver.constraint_learning import AugmentedLagrangian,violation_metrics
from contact_solver.feasibility_filter import FeasibilityFilter,ProtectionRule
from .step14_gradient_field import plain,InvalidWitnessNormal


def experiment(c,model,inputs,base,checkpoint,cfg,evaluate,reference):
    out=c['out'];results={};traces={};attempts={};restorations={}
    guard=FeasibilityFilter([ProtectionRule(**r) for r in cfg['filter_rules']])
    required_parts=c['initial']['contact_part_mask'].nonzero().flatten().tolist()
    surface=c['initial']['contact_surface']
    required={f'part/{part}/surface/{int(surface[part])}':True for part in required_parts}
    def evidence(raw):return {key:bool(raw['contact_part_mask'][part] and raw['contact_surface'][part]==surface[part]) for part,key in zip(required_parts,required)}
    for method in cfg.get('filter_methods',('armijo_only','protected')):
        if method not in ('armijo_only','protected','protected_dual_restore'):raise ValueError('Unknown filter method')
        model.load_state_dict(checkpoint['model']);al=AugmentedLagrangian(rho=cfg['rho'],max_rho=1e3)
        q=reference.clone();raw=c['observe'](q,method+'_initial');trace=[];trials=[];status='step_budget';restore_events=[];consecutive_restorations=0
        c['record'](method+'_initial',q,raw)
        def objective(q,observed):
            prior,rows=evaluate(q,observed)
            return al.loss(prior,rows),rows
        for step in range(1,cfg['steps']+1):
            model.zero_grad(set_to_none=True)
            value,_=objective(model(**inputs).qpos[0].double(),raw);value.backward()
            current_value,previous=objective(q,raw)
            actual_before=float(current_value.detach())
            params=[p for p in model.parameters() if p.grad is not None];weights=[p.detach().clone() for p in params];grads=[p.grad.detach().clone() for p in params]
            norm2=sum(float(g.double().square().sum()) for g in grads);accepted=False
            for trial in range(24):
                eta=cfg['learning_rate']*.5**trial
                with torch.no_grad():
                    for p,w,g in zip(params,weights,grads):p.copy_(w-eta*g)
                    pred=model(**inputs);candidate=pred.qpos[0].double()
                fresh=c['observe'](candidate,f'{method}_{step}_{trial}')
                try:next_value,rows=objective(candidate,fresh);number=float(next_value)
                except InvalidWitnessNormal as exc:
                    trials.append(dict(step=step,trial=trial,eta=eta,query_id=fresh['query_id'],accepted=False,reasons=[dict(kind='invalid_newton_normal',error=str(exc))]));continue
                armijo=number<actual_before and number<=actual_before-1e-4*eta*norm2
                decision=guard.assess(previous,rows,required_evidence=required,candidate_evidence=evidence(fresh))
                moved=float((candidate-q).abs().max())>1e-9
                accepted=armijo and moved and (method=='armijo_only' or decision['accepted'])
                trials.append(dict(step=step,trial=trial,eta=eta,query_id=fresh['query_id'],loss_before=actual_before,autograd_loss=float(value.detach()),loss_after=number,
                    armijo=armijo,guard_accepted=decision['accepted'],moved=moved,accepted=accepted,reasons=decision['reasons']))
                if accepted:q,raw=candidate.detach(),fresh;break
            if not accepted:
                with torch.no_grad():
                    for p,w in zip(params,weights):p.copy_(w)
                if method=='protected_dual_restore' and consecutive_restorations<cfg.get('max_blocked_dual_updates',12):
                    # Use the smallest moving, objective-decreasing rejected
                    # trial to identify active blockers. No target pose is used.
                    blocked=next((t for t in reversed(trials) if t['step']==step and t.get('armijo') and t.get('moved') and not t.get('guard_accepted')),None)
                    keys=set() if blocked is None else {tuple(r['key'][:3]) for r in blocked['reasons'] if r['kind']=='protected_residual_worsened'}
                    selected=[r for r in previous if r.key in keys]
                    if selected:
                        al.update(selected)
                        consecutive_restorations+=1
                        restore_events.append(dict(step=step,blocking_query=blocked['query_id'],constraint_keys=[list(r.key) for r in selected],consecutive=consecutive_restorations))
                        continue
                status='no_acceptable_step';break
            consecutive_restorations=0
            _,rows=evaluate(q,raw)
            trace.append(dict(step=step,eta=eta,query_id=raw['query_id'],loss=number,
                actual_contacts=plain(raw['contact_part_mask']),required_contacts_kept=all(evidence(raw).values()),
                plan_unchanged=bool(torch.equal(pred.conditioned_contact,base.conditioned_contact) and torch.equal(pred.cell,base.cell)),
                metrics=violation_metrics(rows)))
            if len(trace)%cfg['dual_update_every']==0:al.update(rows)
            if step==1 or step%20==0 or step==cfg['steps']:
                c['record'](f'{method}_step{step}',q,raw)
                (out/'filter_progress.json').write_text(json.dumps(plain(dict(results=results,method=method,trace=trace,attempts=trials)),indent=2))
        final=c['record'](method+'_final',q,raw);prior,rows=evaluate(q,raw)
        final.update(status=status,accepted_steps=len(trace),trial_count=len(trials),metrics=violation_metrics(rows),prior=float(prior),blocked_dual_updates=len(restore_events))
        results[method]=final;traces[method]=trace;attempts[method]=trials;restorations[method]=restore_events
        (out/(method+'_multipliers.json')).write_text(json.dumps(al.state_dict(),indent=2))
    (out/'diagnostic.json').write_text(json.dumps(plain(dict(schema='feasibility_filter_experiment_v1',config=cfg,results=results,traces=traces,attempts=attempts,restorations=restorations,
        records=c['records'],forward_audit=c.get('filter_forward_audit'),terrain=c['saved']['terrain'],robot_asset_json=c['saved']['robot_asset_json'],joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
        contract='Shared AL residuals, gradient formula and Armijo trials. Protected arms check per-component feasibility and actual Newton contact retention. protected_dual_restore additionally updates only blocking constraint multipliers on rejected steps. No QP teacher or projected gradient. Tolerances are optimization guards, not contact thresholds.')),indent=2))
    print('FEASIBILITY_FILTER_COMPLETE',flush=True)
