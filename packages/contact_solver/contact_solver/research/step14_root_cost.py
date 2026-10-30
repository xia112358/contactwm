"""Root-expensive scalar loss, differentiated through the actual predictor."""
import json
import os
import torch
from .step14_gradient_field import plain, InvalidWitnessNormal


def displacement_cost(q, reference, root_weight):
    """Per-coordinate normalized cost; quaternion chordal rotation, sign invariant.

    2 cm translation and 0.1 rad joint/rotation are unit displacements. Root
    translation and rotation each cost root_weight times a joint coordinate.
    """
    a=q[3:7]/q[3:7].norm(); b=reference[3:7]/reference[3:7].norm()
    sign=torch.where((a*b).sum()>=0, 1., -1.)
    translation=((q[:3]-reference[:3])/.02).square().sum()
    rotation=4*((a-sign*b)/.1).square().sum()
    joints=((q[7:]-reference[7:])/.1).square().sum()
    return .5*(root_weight*(translation+rotation)+joints)


def experiment(c,model,inputs,base,checkpoint,metadata):
    reference=base.qpos[0].detach().double()
    out=c['out']; records=[]; attempts=[]; results={}
    steps=int(os.environ.get('STEP14_ROOT_STEPS','40'))
    for weight in (1.,100.,1000.):
        model.load_state_dict(checkpoint['model'])
        q=reference.clone(); observed=c['observe'](q,f'root{weight}_start')
        def loss(q, raw):
            _,parts=c['objective'](q,raw,'sdf_sum')
            return parts['environment']+parts['self_collision']+10*parts['contact']+displacement_cost(q,reference,weight)
        initial_loss=float(loss(q,observed))
        status='step_budget'
        for step in range(steps):
            model.zero_grad(set_to_none=True)
            value=loss(model(**inputs).qpos[0].double(),observed)
            value.backward()
            params=[p for p in model.parameters() if p.grad is not None]
            old=[p.detach().clone() for p in params]
            gradients=[p.grad.detach().clone() for p in params]
            norm2=sum(float(g.double().square().sum()) for g in gradients)
            accepted=False
            for trial in range(18):
                eta=1e-9*(.5**trial)
                with torch.no_grad():
                    for p,o,g in zip(params,old,gradients):p.copy_(o-eta*g)
                    pred=model(**inputs);candidate=pred.qpos[0].double()
                fresh=c['observe'](candidate,f'root{weight}_{step}_{trial}')
                try: next_value=float(loss(candidate,fresh))
                except InvalidWitnessNormal: next_value=float('inf')
                accepted=next_value<=float(value.detach())-1e-4*eta*norm2
                attempts.append(dict(weight=weight,step=step,trial=trial,eta=eta,
                    before=float(value.detach()),after=next_value if next_value<float('inf') else None,
                    accepted=accepted,query_id=fresh['query_id']))
                if accepted:
                    q,observed=candidate,fresh
                    break
            if not accepted:
                with torch.no_grad():
                    for p,o in zip(params,old):p.copy_(o)
                status='line_search_stalled'
                break
            if step%5==0 or step==steps-1:
                row=c['record'](f'root{weight}_step{step+1}',q,observed)
                row.update(root_weight=weight,iteration=step+1,loss=next_value,
                    displacement_cost=float(displacement_cost(q,reference,weight)),
                    root_from_network_baseline_cm=100*float((q[:3]-reference[:3]).norm()),
                    root_rotation_deg=float(torch.rad2deg(2*torch.acos((q[3:7]*reference[3:7]).sum().abs().clamp(max=1)))),
                    plan_unchanged=bool(torch.equal(pred.conditioned_contact,base.conditioned_contact) and torch.equal(pred.cell,base.cell)))
                records.append(row)
        row=c['record'](f'root{weight}_final',q,observed)
        row.update(root_weight=weight,status=status,initial_loss=initial_loss,loss=float(loss(q,observed)),
            displacement_cost=float(displacement_cost(q,reference,weight)))
        results[str(weight)]=row
        (out/'root_progress.json').write_text(json.dumps(plain(dict(results=results,records=records,attempts=attempts)),indent=2))
    result=dict(schema='step14_root_cost_v1',results=results,records=c['records'],attempts=attempts,
        network=metadata,terrain=c['saved']['terrain'],robot_asset_json=c['saved']['robot_asset_json'],
        joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
        contract='Scalar normalized correction cost, root/joint ratio 1/100/1000. Real parameter gradient descent with fresh Newton Armijo backtracking. No QP, output projection, or gradient preconditioning. Reference is original network output, not prior frame.')
    (out/'diagnostic.json').write_text(json.dumps(plain(result),indent=2))
    print('ROOT_COST_COMPLETE',flush=True)
