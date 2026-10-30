"""Audit scalar loss gradients in pose space and through the real predictor.

No projection, contact metric, clipping, optimizer momentum, or QP targets.
Every probe starts from the same pose/checkpoint; steps are declared in advance.
"""
import json
import os
from pathlib import Path
import runpy
import numpy as np
import torch
from .recovery_field import pose_increment
from .step14_gradient_field import run, plain


def experiment(c):
    q0, geo, fk = c['q0'], c['geometry'], c['fk']
    out, initial = c['out'], c['initial']
    anchors, scale = c['anchors'], c['scale']
    observe, record = c['observe'], c['record']
    report = json.loads(Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_motion8_raw29_20260925/report.json').read_text())
    def terms(q):
        _, old = c['objective'](q, initial, 'sdf_sum')
        _, sep = geo.evaluate(q)
        foot = sep[geo.foot_ids, geo.top].amin()
        support = torch.stack([sep[a['shape_ids'], geo.top].amin() for a in anchors])
        # Explicit relative-height auxiliary: root translation cancels exactly.
        # This is geometry guidance, not a declaration of achieved contact.
        relative = ((foot-support)/.01).square().mean()
        target = (foot/.01).square()
        return dict(sdf=old['environment'], self_collision=old['self_collision'],
                    retention=10*old['contact'], posture=1e-4*old['reference'],
                    target_top=target, relative_support=relative)
    def loss(q, method):
        t = terms(q)
        base = t['sdf']+t['self_collision']+t['retention']+t['posture']
        return base if method == 'sdf_sum' else base+t['relative_support']
    def audit(q, label, **extra):
        q = q.detach()
        row = record(label, q, observe(q, label), **extra)
        row['terms'] = plain(terms(q))
        return row
    directions = {}
    for name in terms(q0):
        x = torch.zeros(35, device=q0.device, dtype=q0.dtype, requires_grad=True)
        value = terms(pose_increment(q0, x, scale))[name]
        g = torch.autograd.grad(value, x)[0]
        eps = 1e-6
        dq = pose_increment(q0, -eps*g, scale)
        p0, _ = fk.link_poses(q0[None], ('left_ankle_roll_link',))
        p1, _ = fk.link_poses(dq[None], ('left_ankle_roll_link',))
        fd = []
        for axis in range(35):
            d = torch.zeros_like(x); d[axis] = 1e-5
            fd.append(float((terms(pose_increment(q0,d,scale))[name]-terms(pose_increment(q0,-d,scale))[name])/2e-5))
        fd = g.new_tensor(fd)
        directions[name] = dict(value=float(value.detach()), gradient=plain(g),
            fd_relative_l2=float((g-fd).norm()/g.norm().clamp_min(1e-12)),
            root_velocity=plain((dq[:3]-q0[:3])/eps),
            foot_velocity=plain((p1-p0)[0,0]/eps))
    for method in ('sdf_sum', 'relative_auxiliary'):
        x = torch.zeros(35,device=q0.device,dtype=q0.dtype,requires_grad=True)
        g = torch.autograd.grad(loss(pose_increment(q0,x,scale),method),x)[0]
        for eta in (1e-5,1e-4,1e-3):
            q = pose_increment(q0,-eta*g,scale)
            audit(q, f'pose_{method}_{eta}', step_size=eta, loss_before=float(loss(q0,method)),loss_after=float(loss(q,method)))
    from generator.full1000_position_predictor import Full1000PositionPredictor
    from somaforge_core.heightmap import render_root_yaw_box_heightmaps
    cp = torch.load(report['checkpoint'], map_location=q0.device, weights_only=False)
    cfg = cp['config']
    model = Full1000PositionPredictor(cfg['width'],cfg['layers'],cfg.get('location_width',32),
        **{k:bool(cfg.get(k,False)) for k in ('body_geometry','part_geometry','region_plan','unified_contact','event_roles','execution_observation_gradients')}).to(q0.device).eval()
    model.load_state_dict(cp['model'])
    current = np.asarray(report['visualization']['predicted_q_world'][13],dtype=np.float32)
    observed = observe(torch.as_tensor(current,device=q0.device,dtype=torch.float64), 'network_input')
    hm = render_root_yaw_box_heightmaps(geo.center.cpu().numpy()[None],geo.basis.cpu().numpy()[None],
        geo.half.cpu().numpy()[None],np.zeros(1,dtype=np.float32),current[None],supersample=cfg.get('heightmap_supersample',1))
    inputs = dict(current_q=torch.as_tensor(current[None],device=q0.device),
        current_contact=observed['contact_part_mask'][None],current_anchor=observed['contact_position_w'][None],
        heightmap=torch.as_tensor(hm,device=q0.device,dtype=torch.float32))
    with torch.no_grad(): base = model(**inputs)
    error = float((base.qpos[0].double()-q0).abs().max())
    metadata = dict(checkpoint=report['checkpoint'], reproduction_max_abs_error=error,
        reproduction_root_error_m=float((base.qpos[0,:3].double()-q0[:3]).norm()),
        reproduction_joint_error_rad=float((base.qpos[0,7:].double()-q0[7:]).abs().max()),
        reproduction_contract='Fresh Newton input and mesh-derived heightmap; numerical replay, not bit-exact replay. Maximum component tolerance 1e-4; network steps compared to its own unmodified forward.',
        original_predicted_contact=report['events'][13]['predicted_contact'],
        scope='Left-foot recovery and preservation of initially realized hand/knee contacts; not full predicted-plan feasibility.')
    (out/'network_input.json').write_text(json.dumps(plain(dict(metadata=metadata,inputs=inputs)),indent=2))
    if error > 1e-4:
        raise ValueError(f'Network fixture reproduction failed: {error}; no parameter experiment claimed')
    audit(base.qpos[0].double(), 'network_baseline')
    if os.environ.get('CONSTRAINT_LEARNING') == '1':
        from .constraint_learning_experiment import experiment as constraint_experiment
        return constraint_experiment(c, model, inputs, base, cp, metadata)
    if os.environ.get('STEP14_QP_TEACHER') == '1':
        from .step14_qp_teacher import experiment as teacher_experiment
        return teacher_experiment(c, model, inputs, base, cp, metadata)
    if os.environ.get('STEP14_ROOT_COST') == '1':
        from .step14_root_cost import experiment as root_experiment
        return root_experiment(c, model, inputs, base, cp, metadata)
    for method in ('sdf_sum','relative_auxiliary'):
        model.load_state_dict(cp['model']); model.zero_grad(set_to_none=True)
        prediction = model(**inputs)
        objective = loss(prediction.qpos[0].double(),method)
        objective.backward()
        gradients = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        norm = sum(float(g.double().square().sum()) for g in gradients.values())**.5
        for eta in (1e-9,1e-8,1e-7):
            model.load_state_dict(cp['model'])
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in gradients: p.add_(gradients[n],alpha=-eta)
                updated = model(**inputs)
                q = updated.qpos[0].double()
            audit(q,f'network_{method}_{eta}',step_size=eta,parameter_gradient_norm=norm,
                  plan_unchanged=bool(torch.equal(updated.conditioned_contact,base.conditioned_contact) and torch.equal(updated.cell,base.cell)),
                  loss_before=float(objective.detach()),loss_after=float(loss(q,method)))
    result = dict(schema='step14_loss_direction_v1',records=c['records'],directions=directions,
        network=metadata,terrain=c['saved']['terrain'],robot_asset_json=c['saved']['robot_asset_json'],
        joint_names=c['saved']['joint_names'],provenance=c['query'].provenance,
        contract='Fixed coordinate pose SGD and real parameter SGD; no projection or QP teacher. Geometry metrics do not define contact.')
    (out/'diagnostic.json').write_text(json.dumps(plain(result),indent=2))
    print('LOSS_DIRECTION_COMPLETE',flush=True)


if __name__ == '__main__':
    import somaforge_core.newton_tensor_transport as transport
    transport.serve_tensor_queries = lambda query,*a,**kw: run(query,experiment=experiment)
    runpy.run_path('scripts/serve_newton_contact_queries.py',run_name='__main__')
