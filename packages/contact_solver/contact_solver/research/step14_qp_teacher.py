"""Fixed QP endpoint as a stopped-gradient diagnostic teacher, never a projector."""
import json
import math
import os
from pathlib import Path
import torch
from .step14_gradient_field import plain


from contact_solver.pose_coordinates import multiply, coordinates, pose


def comparisons(a,b):
    groups=dict(all=list(range(35)),root=list(range(6)),root_translation=list(range(3)),
                root_rotation=list(range(3,6)),left_hip=[6,7,8],left_knee=[9],left_ankle=[10,11])
    result={}
    for name,ids in groups.items():
        u,v=a[ids],b[ids];nu,nv=float(u.norm()),float(v.norm())
        result[name]=dict(cosine=float(u@v/(u.norm()*v.norm())) if min(nu,nv)>1e-10 else None,norm=nu,teacher_norm=nv)
    return result


def experiment(c,model,inputs,base,checkpoint,metadata):
    reference=base.qpos[0].detach().double();scale=c['scale'];out=c['out']
    source=Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_step14_static_interval_20260925/diagnostic.json')
    saved=json.loads(source.read_text())
    if saved['robot_asset_json']!=c['saved']['robot_asset_json']:raise ValueError('Teacher asset mismatch')
    target=reference.new_tensor(next(r['q'] for r in saved['records'] if r['label']=='face3_iter5')).detach()
    target_z=coordinates(target,reference,scale).detach()
    origin=c['observe'](reference,'teacher_network_origin')
    def loss(q):return .5*(coordinates(q,reference,scale)-target_z).square().sum()
    def audit(q,label,**extra):
        q=q.detach();raw=c['observe'](q,label);row=c['record'](label,q,raw)
        row.update(teacher_loss=float(loss(q)),teacher_error_norm=float((coordinates(q,reference,scale)-target_z).norm()),**extra)
        return row
    teacher_row=audit(target,'qp_teacher_endpoint')
    if not teacher_row['static_recovery_pass']:raise ValueError('QP endpoint fails fresh static audit')
    z=torch.zeros_like(target_z,requires_grad=True)
    g=torch.autograd.grad(loss(pose(z,reference,scale)),z)[0]
    if not torch.allclose(-g,target_z,atol=1e-9,rtol=1e-9):raise ValueError('Teacher gradient mismatch')
    z=torch.zeros_like(target_z,requires_grad=True)
    _,parts=c['objective'](pose(z,reference,scale),origin,'sdf_sum')
    sdf=-torch.autograd.grad(parts['environment'],z)[0]
    model.load_state_dict(checkpoint['model']);model.zero_grad(set_to_none=True)
    loss(model(**inputs).qpos[0].double()).backward()
    params=[p for p in model.parameters() if p.grad is not None]
    weights=[p.detach().clone() for p in params];grads=[p.grad.detach().clone() for p in params]
    with torch.no_grad():
        for p,w,g in zip(params,weights,grads):p.copy_(w-1e-9*g)
        pred=model(**inputs);network_probe=pred.qpos[0].double()
    net=coordinates(network_probe,reference,scale)
    audit(network_probe,'network_teacher_small_probe',learning_rate=1e-9)
    directions=dict(teacher=plain(target_z),sdf_descent=plain(sdf),network_displacement=plain(net),
        sdf_vs_teacher=comparisons(sdf,target_z),network_vs_teacher=comparisons(net,target_z),
        network_vs_sdf=comparisons(net,sdf))
    # Equal normalized displacement length; this is an output-space diagnostic,
    # not an assertion that the network produced this rescaled output itself.
    length=.1*float(target_z.norm())
    for name,direction in [('teacher',target_z),('sdf',sdf),('network_direction',net)]:
        audit(pose(direction/direction.norm()*length,reference,scale),'matched_'+name,normalized_step_length=length)
    # A separate same-loss probe isolates K K^T acting on the SDF gradient.
    # The teacher-driven network vector above must not be mistaken for it.
    model.load_state_dict(checkpoint['model']);model.zero_grad(set_to_none=True)
    _,sdf_parts=c['objective'](model(**inputs).qpos[0].double(),origin,'sdf_sum')
    sdf_parts['environment'].backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:p.add_(p.grad,alpha=-1e-9)
        sdf_prediction=model(**inputs);sdf_network_q=sdf_prediction.qpos[0].double()
    net_sdf=coordinates(sdf_network_q,reference,scale)
    audit(sdf_network_q,'network_sdf_small_probe',learning_rate=1e-9)
    directions.update(network_sdf_displacement=plain(net_sdf),
        network_sdf_vs_sdf=comparisons(net_sdf,sdf),network_sdf_vs_teacher=comparisons(net_sdf,target_z))
    audit(pose(net_sdf/net_sdf.norm()*length,reference,scale),'matched_network_sdf_direction',normalized_step_length=length)
    # Direct SGD in the fixed chart: step 0.5 halves teacher error each time.
    direct=torch.zeros_like(target_z)
    for step in range(1,9):
        direct.requires_grad_();value=.5*(direct-target_z).square().sum()
        direct=(direct-.5*torch.autograd.grad(value,direct)[0]).detach()
        audit(pose(direct,reference,scale),f'direct_step{step}')
    model.load_state_dict(checkpoint['model'])
    trace=[];steps=int(os.environ.get('STEP14_TEACHER_STEPS','300'));status='step_budget'
    for step in range(1,steps+1):
        model.zero_grad(set_to_none=True);value=loss(model(**inputs).qpos[0].double());value.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        weights=[p.detach().clone() for p in params];grads=[p.grad.detach().clone() for p in params]
        norm2=sum(float(g.double().square().sum()) for g in grads)
        accepted=False
        for trial in range(20):
            eta=1e-6*.5**trial
            with torch.no_grad():
                for p,w,g in zip(params,weights,grads):p.copy_(w-eta*g)
                pred=model(**inputs);q=pred.qpos[0].double();next_value=float(loss(q))
            if next_value<=float(value.detach())-1e-4*eta*norm2:
                accepted=True;break
        if not accepted:
            with torch.no_grad():
                for p,w in zip(params,weights):p.copy_(w)
                pred=model(**inputs);q=pred.qpos[0].double()
            status='line_search_stalled';break
        raw=c['observe'](q,f'network_teacher_{step}')
        kept=raw['contact_part_mask']&origin['contact_part_mask']&(raw['contact_surface']==origin['contact_surface'])
        trace.append(dict(step=step,loss=next_value,eta=eta,query_id=raw['query_id'],
            all_required_kept=bool((kept|~origin['contact_part_mask']).all()),
            plan_unchanged=bool(torch.equal(pred.conditioned_contact,base.conditioned_contact) and torch.equal(pred.cell,base.cell))))
        if step==1 or step%25==0 or step==steps:
            row=c['record'](f'network_teacher_step{step}',q,raw)
            row.update(teacher_loss=next_value,teacher_error_norm=float((coordinates(q,reference,scale)-target_z).norm()))
            (out/'teacher_progress.json').write_text(json.dumps(plain(dict(trace=trace,records=c['records'])),indent=2))
    final=audit(q,'network_teacher_final',status=status)
    result=dict(schema='step14_qp_teacher_v1',records=c['records'],directions=directions,trace=trace,
        teacher_source=str(source),teacher_label='face3_iter5',teacher_q=plain(target),reference_q=plain(reference),
        coordinate_scale=plain(scale),network=metadata,terrain=c['saved']['terrain'],
        robot_asset_json=c['saved']['robot_asset_json'],joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
        contract='Stopped fixed QP teacher; SO(3) log coordinates. Only teacher squared error updates network. Geometry and Newton contacts audit, never participate in teacher loss. Matched direction probes are synthetic poses.')
    (out/'diagnostic.json').write_text(json.dumps(plain(result),indent=2))
    print('QP_TEACHER_COMPLETE',flush=True)
