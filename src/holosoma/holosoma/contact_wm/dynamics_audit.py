"""Held-out, same-action dynamics checks. No desired contacts enter WM scoring."""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class DynamicsLimits:
    # Initial engineering tolerances, not a certificate of gradient correctness.
    root_p95_m: float = .03
    effector_p95_m: float = .03
    joint_rmse_rad: float = .10
    rotation_p95_rad: float = .15
    force_rmse_N: float = 100.
    false_positive_rate: float = .10
    false_negative_rate: float = .10


def errors(predictions, batch, geometry):
    state=torch.stack([p['state'] for p in predictions])
    target=batch['next_state'][:len(state)]
    known=batch['known'][:len(state)].bool()
    active=batch['active'][:len(state)].bool()
    contact=torch.stack([p['contact_logits'] for p in predictions])>=0
    force=torch.stack([p['force_w'] for p in predictions])
    H,B=state.shape[:2]
    def body_points(s):
        if hasattr(geometry,'local_witness'):
            # Task witnesses are defined only for the requested support/new
            # contacts. Audit all six actual link origins instead of projecting
            # a missing contact's zero world-coordinate placeholder through FK.
            from holosoma.contact_wm.geometry import LINKS
            qpos=torch.cat((s[:,:3],s[:,[6,3,4,5]],s[:,13:42].index_select(-1,geometry.order)),-1)
            return geometry.fk.link_poses(qpos,LINKS)[0]
        return geometry(s)
    witness=(body_points(state.flatten(0,1))-body_points(target.flatten(0,1))).reshape(H,B,6,3).norm(dim=-1)
    root=(state[...,:3]-target[...,:3]).norm(dim=-1)
    q=torch.nn.functional.normalize(state[...,3:7],dim=-1)
    qt=torch.nn.functional.normalize(target[...,3:7],dim=-1)
    angle=2*torch.acos((q*qt).sum(-1).abs().clamp(max=1))
    switch=((active!=batch['previous_active'][:H])&known&batch['previous_known'][:H]).any(-1)
    def subset(mask):
        if not bool(mask.any()):return None
        k=known&mask[...,None];positive=k&active;negative=k&~active
        safe_force=torch.where(k[...,None],force-batch['force_w'][:H],0.)
        rate=lambda numerator,denominator:float(numerator.sum()/denominator.sum()) if bool(denominator.any()) else None
        return dict(samples=int(mask.sum()),root_p95_m=float(torch.quantile(root[mask],.95)),
            effector_p95_m=float(torch.quantile(witness[mask].flatten(),.95)),
            joint_rmse_rad=float((state[...,13:42]-target[...,13:42])[mask].square().mean().sqrt()),
            rotation_p95_rad=float(torch.quantile(angle[mask],.95)),
            force_rmse_N=float((safe_force.square().sum()/(3*k.sum()).clamp_min(1)).sqrt()),
            false_positive_rate=rate(contact&negative,negative),
            false_negative_rate=rate(~contact&positive,positive),
            positive_contacts=int(positive.sum()),negative_contacts=int(negative.sum()))
    end=torch.zeros(H,B,device=state.device,dtype=torch.bool);end[-1]=True
    return dict(endpoint=subset(end),all=subset(torch.ones_like(end)),switch=subset(switch),steady=subset(~switch))


@torch.no_grad()
def audit_dynamics(learner,replay,batch_size=128,horizons=(1,8,16,35)):
    report=dict(horizons={},source='held-out native transitions; identical recorded actions',
                effector_points='six canonical contact-link origins, not solver contact positions',
                gradient_validated=False)
    for horizon in horizons:
        try:x=replay.windows(batch_size,horizon)
        except (ValueError,RuntimeError) as exc:
            report['horizons'][str(horizon)]=dict(available=False,reason=str(exc));continue
        modes={}
        for mode in ('recorded_height','predicted_height'):
            state=x['state'][0];force=x['previous_force_w'][0];predictions=[];valid=True
            for t in range(horizon):
                if mode=='predicted_height':
                    height,inside=learner.height(state,learner.source_grid)
                    valid=valid and bool(inside.all())
                else:height=x['height'][t]
                p=learner.wm.source_step(state,x['action'][t],learner.physics(state),height,force)
                if not all(bool(torch.isfinite(p[k]).all()) for k in ('state','force_w','contact_logits')):
                    valid=False;break
                predictions.append(p);state=p['state'];force=p['force_w']
            modes[mode]=dict(valid=valid,metrics=errors(predictions,x,learner.geometry) if valid else None)
            if valid and 'motion_replay' in x:
                groups={}
                for name,mask in [('motion',x['motion_replay'][0]),('policy',~x['motion_replay'][0])]:
                    if not bool(mask.any()):groups[name]=None;continue
                    sub=[{k:v[mask] for k,v in p.items()} for p in predictions]
                    groups[name]=errors(sub,{k:v[:,mask] for k,v in x.items()},learner.geometry)
                modes[mode]['by_controller']=groups
        report['horizons'][str(horizon)]=dict(available=True,**modes)
    return report


def policy_gate(report,limits=DynamicsLimits(),reference=None):
    """Require measured accuracy, including contact switches when observed.

    Missing contact classes are reported rather than counted as perfect. This
    gate cannot establish accuracy in unobserved contact modes or WM Jacobians.
    """
    reasons=[];missing=[]
    for horizon in (1,8,16,35):
        row=report.get('horizons',{}).get(str(horizon),{})
        if not row.get('available'):
            reasons.append(f'h{horizon}: no valid held-out windows');continue
        mode=row['predicted_height']
        if not mode['valid']:
            reasons.append(f'h{horizon}: invalid prediction or terrain query');continue
        groups=mode.get('by_controller',{'combined':mode['metrics']})
        for group,summary in groups.items():
            if summary is None:
                missing.append(f'h{horizon}/{group}');continue
            for subset in ('endpoint','all','switch'):
                metrics=summary[subset]
                if metrics is None:
                    missing.append(f'h{horizon}/{group}/{subset}');continue
                for key,limit in asdict(limits).items():
                    value=metrics[key]
                    if value is None:missing.append(f'h{horizon}/{group}/{subset}/{key}')
                    elif not math.isfinite(value) or value>limit:
                        reasons.append(f'h{horizon}/{group}/{subset}/{key}={value:.5g} > {limit}')
    if reference is not None:
        fixed=policy_gate(reference,limits)
        reasons.extend('fixed_reference/'+reason for reason in fixed['reasons'])
        missing.extend('fixed_reference/'+reason for reason in fixed['unobserved'])
    return dict(allowed=not reasons,reasons=reasons,unobserved=missing,limits=asdict(limits),gradient_validated=False)
