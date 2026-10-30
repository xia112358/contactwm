"""Configuration-driven experiments using a static canonical geometry adapter.

The existing fixture owns FK, actual Newton queries and the environment. The
loss and AL engine know none of its link names or scene-specific thresholds.
No QP solution is loaded or used. This adapter currently supports planar tasks
on the fixture's box, whole-body sampled environment SDF and queried self rows.
"""
import json
import os
from pathlib import Path
import torch
from contact_solver.constraint_learning import penalty_loss,AugmentedLagrangian,violation_metrics
from contact_solver.constraint_residuals import (PlaneSurface,ContactTask,contact_residuals,
    collision_residual,joint_limit_residuals,quadratic_prior)
from contact_solver.pose_coordinates import coordinates
from .step14_gradient_field import plain,InvalidWitnessNormal


def experiment(c,model,inputs,base,checkpoint,metadata):
    if os.environ.get('FEASIBILITY_FILTER_EXPERIMENT') == '1' or os.environ.get('DIRECTION_EXPERIMENT') == '1':
        # Inference-only MHA fast paths can differ numerically from autograd
        # forward; that difference must not masquerade as line-search progress.
        before=model(**inputs).qpos.detach()
        with torch.no_grad(): inference=model(**inputs).qpos.detach()
        mismatch_before=float((before-inference).abs().max())
        torch.backends.mha.set_fastpath_enabled(False)
        after=model(**inputs).qpos.detach()
        with torch.no_grad(): inference=model(**inputs).qpos.detach()
        c['filter_forward_audit']=dict(before_max_abs=mismatch_before,after_max_abs=float((after-inference).abs().max()),mha_fastpath_enabled=False)
    cfg=json.loads(Path(os.environ.get('CONSTRAINT_CASE_CONFIG','configs/contact_learning/step14_static.json')).read_text())
    if cfg['schema']!='constraint_learning_case_v1':raise ValueError('Unknown experiment config')
    out=c['out'];geo=c['geometry'];fk=c['fk'];device=c['q0'].device
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.double)
    n=geo.normal[geo.top];tangents=geo.basis[:,:2].T
    sides=torch.where((geo.normal[:6]@n).abs()<1e-6)[0]
    surface=PlaneSurface(n*geo.offset[geo.top],n,tangents,geo.normal[sides],geo.offset[sides])
    # The actual realized solver margin is required; no geometric fallback.
    margins=c['query'].configured_terrain_includemargins
    if not margins or len(set(float(x) for x in margins))!=1:raise ValueError('Fixture adapter requires an unambiguous realized terrain margin')
    margin=float(margins[0])
    solver_band=cfg.get('contact_semantics')=='solver_activation_with_penetration_v1'
    from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE
    penetration_tolerance=DEFAULT_ACCEPTANCE.shallow_penetration_m if solver_band else 0.
    anchor_map={a['link']:a for a in c['anchors']}
    task_specs=[]
    for spec in cfg['contacts']:
        if spec['surface']!='terrain_top':raise ValueError('Surface not present in this fixture adapter')
        ids=[i for i,s in enumerate(geo.shapes) if s['name']==spec['link']]
        if not ids:raise ValueError('Contact link lacks canonical geometry')
        if spec.get('keep_tangent') and spec['link'] not in anchor_map:raise ValueError('Missing actual material anchor for requested retention')
        upper=margin-cfg['activation_numerical_reserve_m'] if solver_band else margin*spec['max_gap_margin_fraction']
        task=ContactTask(spec['id'],spec['mode'],cfg['distance_scale_m'],-penetration_tolerance,upper,coverage=spec.get('coverage',False))
        task_specs.append((spec,ids,task))
    names=tuple(s['link'] for s,_,_ in task_specs)
    if solver_band:
        original_record=c['record']
        def record_with_semantics(label,q,raw,*args,**kwargs):
            result=original_record(label,q,raw,*args,**kwargs)
            result['legacy_static_recovery_pass']=result['static_recovery_pass']
            pair=raw['pairs'];actual={}
            for spec,_,_ in task_specs:
                link=raw['link_names'].index(spec['link'])
                match=(pair['body_link1']==link)&pair['task_pair']&pair['upward']&(pair['primary_surface']==spec['surface_id'])
                actual[spec['id']]=bool((match & pair['eligible']).any())
            result['required_contact_established']=actual
            result['penetration_accepted']=max(result['full_depth_cm'],result['shape_plane_violation_cm'],result['sampled_sdf_depth_cm'])<=100*penetration_tolerance
            result['contact_satisfied']=all(actual.values()) and result['penetration_accepted']
            result['static_recovery_pass']=result['contact_satisfied'] and result['joint_violation_rad']<=1e-6 and result['anchor_error_cm']<=1.
            return result
        c['record']=record_with_semantics
    summaries={};histories={};queryfailures=[]
    for case in cfg['cases']:
        from somaforge_core.heightmap import render_root_yaw_box_heightmaps
        case_inputs=dict(inputs);current=inputs['current_q'][0].detach().clone()
        current[:3]+=current.new_tensor(case['input_root_offset'])
        for name,delta in case['input_joint_offsets'].items():current[7+c['saved']['joint_names'].index(name)]+=delta
        current_raw=c['observe'](current.double(),case['id']+'_input')
        hm=render_root_yaw_box_heightmaps(geo.center.cpu().numpy()[None],geo.basis.cpu().numpy()[None],geo.half.cpu().numpy()[None],[0.],current.cpu().numpy()[None],supersample=checkpoint['config'].get('heightmap_supersample',1))
        case_inputs.update(current_q=current[None],current_contact=current_raw['contact_part_mask'][None],current_anchor=current_raw['contact_position_w'][None],heightmap=torch.as_tensor(hm,device=device,dtype=torch.float32))
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():case_base=model(**case_inputs);reference=case_base.qpos[0].double().detach()
        prior_weights=tensor([cfg['root_relative_cost']]*6+[1.]*(len(reference)-7))*cfg['prior_weight']
        def evaluate(q,observed,*,protection=False):
            world,sep=geo.evaluate(q);p,r=fk.link_poses(q[None],names)
            rows=[];sample=case['id'];context=cfg['context_id'];dist_scale=cfg['distance_scale_m']
            clearance=0. if protection else -penetration_tolerance
            ids=tuple(str(i) for i in range(len(geo.shapes)))
            rows.append(collision_residual(sample,context,'environment',geo.sdf(world).amin(-1),distance_scale=dist_scale,component_ids=ids,clearance=clearance))
            rows.append(collision_residual(sample,context,'ground',sep[:,geo.ground],distance_scale=dist_scale,component_ids=ids,clearance=clearance))
            distances=c['distances'](q,observed);pairs=observed['pairs'];own=(pairs['full_kind']==1).nonzero().flatten()
            if cfg.get('acceptance_geometry_residuals',False):
                # Exactly the full-shape plane diagnostic used by record(), not
                # another random sample of the collision mesh.
                rows.append(collision_residual(sample,context,'shape_environment',sep[:,:6].amax(-1),distance_scale=dist_scale,component_ids=ids,clearance=clearance))
                full=pairs['full_kind']>=0
                # No reported pair means zero reported penetration, NOT a
                # geometric separation certificate; shape rows remain above.
                reported=distances[full].amin().clamp_max(0) if bool(full.any()) else q.sum()*0
                # A newly discovered witness can deepen the reported Newton
                # maximum while the actual deep embedding is improving. Fuse
                # the SAME maximum used by acceptance, rather than protecting
                # each incomplete estimator's visibility independently.
                verified=torch.stack((reported,sep[:,:6].amax(-1).amin(),sep[:,geo.ground].amin(),geo.sdf(world).amin())).amin()
                rows.append(collision_residual(sample,context,'verified_fullbody',verified.reshape(1),distance_scale=dist_scale,component_ids=('maximum_penetration',),clearance=clearance))
            # Aggregate changing witness counts by stable shape-pair identity.
            pair_ids=sorted(set((int(pairs['shape0'][i]),int(pairs['shape1'][i])) for i in own))
            for a,b in pair_ids:
                mask=(pairs['full_kind']==1)&(pairs['shape0']==a)&(pairs['shape1']==b)
                rows.append(collision_residual(sample,context,f'self/{a}/{b}',distances[mask].amin().reshape(1),distance_scale=dist_scale,component_ids=('minimum',),clearance=clearance))
            for certificate in observed.get('self_separation_certificates',[]):
                a,b=certificate['shape0'],certificate['shape1']
                if (a,b) in pair_ids:raise ValueError('Certificate cannot replace a current Newton pair')
                rows.append(collision_residual(sample,context,f'self/{a}/{b}',q.new_tensor([certificate['gap_m']]),distance_scale=dist_scale,component_ids=('minimum',)))
            for i,(spec,shape_ids,task) in enumerate(task_specs):
                kwargs={}
                if solver_band:
                    link=observed['link_names'].index(spec['link'])
                    match=(pairs['body_link1']==link)&pairs['task_pair']&pairs['upward']&(pairs['primary_surface']==spec['surface_id'])
                    if bool((match & pairs['active'] & ~pairs['constraint_allocated']).any()):
                        raise ValueError('Active target contact has no allocated constraint')
                    if not protection:kwargs['realized']=bool((match & pairs['eligible']).any())
                    if bool(match.any()):kwargs['approach_gap']=distances[match].amin()
                if spec.get('keep_tangent'):
                    anchor=anchor_map[spec['link']]
                    kwargs.update(anchor_world=p[0,i]+r[0,i]@anchor['local'],anchor_target=anchor['target'])
                if 'align_local_normal' in spec:kwargs.update(normal_world=r[0,i]@tensor(spec['align_local_normal']),normal_target=-surface.normal)
                if task.coverage:kwargs['coverage_points']=world[shape_ids].reshape(-1,3)
                rows+=contact_residuals(sample,context,task,surface,support_gap=sep[shape_ids,geo.top].amin(),**kwargs)
            rows+=joint_limit_residuals(sample,context,q[7:],fk.joint_lower,fk.joint_upper,angle_scale=cfg['angle_scale_rad'],joint_names=tuple(c['saved']['joint_names']))
            prior=quadratic_prior(coordinates(q,reference,c['scale']),prior_weights)
            return prior,rows
        if os.environ.get('DIRECTION_EXPERIMENT') == '1':
            if solver_band:c['protection_residuals']=lambda q,raw:evaluate(q,raw,protection=True)[1]
            from .direction_experiment import experiment as direction_experiment
            return direction_experiment(c,model,case_inputs,case_base,checkpoint,cfg,evaluate,reference)
        if os.environ.get('FEASIBILITY_FILTER_EXPERIMENT') == '1':
            from .filter_experiment import experiment as filter_experiment
            return filter_experiment(c,model,case_inputs,case_base,checkpoint,cfg,evaluate,reference)
        if os.environ.get('CONSTRAINT_DIAGNOSTICS') == '1':
            from .constraint_diagnostics import diagnose
            return diagnose(c,model,case_inputs,case_base,checkpoint,cfg,evaluate,reference,task_specs)
        for method in ('penalty','al'):
            model.load_state_dict(checkpoint['model']);al=AugmentedLagrangian(rho=cfg['rho'],max_rho=1e3)
            label=case['id']+'_'+method;q=reference.clone();raw=c['observe'](q,label+'_initial');trace=[];status='step_budget'
            c['record'](label+'_initial',q,raw)
            def objective(q,observed):
                prior,rows=evaluate(q,observed)
                return (penalty_loss(prior,rows,rho=cfg['rho']) if method=='penalty' else al.loss(prior,rows)),rows
            for step in range(1,cfg['steps']+1):
                model.zero_grad(set_to_none=True)
                value,_=objective(model(**case_inputs).qpos[0].double(),raw);value.backward()
                params=[p for p in model.parameters() if p.grad is not None]
                old=[p.detach().clone() for p in params];grads=[p.grad.detach().clone() for p in params]
                norm2=sum(float(g.double().square().sum()) for g in grads);accepted=False
                for trial in range(18):
                    eta=cfg['learning_rate']*.5**trial
                    with torch.no_grad():
                        for p,w,g in zip(params,old,grads):p.copy_(w-eta*g)
                        pred=model(**case_inputs);candidate=pred.qpos[0].double()
                    fresh=c['observe'](candidate,f'{label}_{step}_{trial}')
                    try:next_value,rows=objective(candidate,fresh);number=float(next_value)
                    except InvalidWitnessNormal as exc:
                        queryfailures.append(dict(label=label,step=step,trial=trial,query_id=fresh['query_id'],error=str(exc)));continue
                    if number<=float(value.detach())-1e-4*eta*norm2:
                        q,raw=candidate,fresh;accepted=True;break
                if not accepted:
                    with torch.no_grad():
                        for p,w in zip(params,old):p.copy_(w)
                    status='line_search_stalled';break
                metrics=violation_metrics(rows)
                trace.append(dict(step=step,loss=number,eta=eta,query_id=raw['query_id'],max_normalized_violation=max(v['max'] for v in metrics.values()),
                    plan_unchanged=bool(torch.equal(pred.conditioned_contact,case_base.conditioned_contact) and torch.equal(pred.cell,case_base.cell))))
                if method=='al' and step%cfg['dual_update_every']==0:al.update(rows)
                if step==1 or step%20==0 or step==cfg['steps']:
                    row=c['record'](f'{label}_step{step}',q,raw)
                    row.update(constraint_metrics=metrics,prior=float(evaluate(q,raw)[0]),sample_id=case['id'],method=method)
                    (out/'constraint_progress.json').write_text(json.dumps(plain(dict(summaries=summaries,records=c['records'],trace=trace)),indent=2))
            final=c['record'](label+'_final',q,raw)
            prior,rows=evaluate(q,raw);final.update(constraint_metrics=violation_metrics(rows),prior=float(prior),status=status,sample_id=case['id'],method=method,
                root_from_case_baseline_cm=100*float((q[:3]-reference[:3]).norm()))
            summaries[label]=final;histories[label]=trace
            (out/(label+'_multipliers.json')).write_text(json.dumps(al.state_dict(),indent=2))
    (out/'diagnostic.json').write_text(json.dumps(plain(dict(schema='constraint_learning_experiment_v1',config=cfg,summaries=summaries,histories=histories,records=c['records'],invalid_queries=queryfailures,
        terrain=c['saved']['terrain'],joint_names=c['saved']['joint_names'],robot_asset_json=c['saved']['robot_asset_json'],provenance=c['query'].provenance,
        contract='No QP target or solve in training. Fixed physical task anchors. Geometry guides; actual Newton contacts audit. Sampled environment coverage is approximate; absent self pairs do not prove no self penetration. Static cases, not generalization evidence.')),indent=2))
    print('CONSTRAINT_LEARNING_COMPLETE',flush=True)
