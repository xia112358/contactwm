"""Fixed step14: Newton witnesses versus sampled SDF versus coherent exit faces.

Run through the ordinary dedicated Newton worker CLI; the tensor-server entry
is replaced in this process by the experiment (no listener, no training).
Environment: STEP14_FIELD_OUTPUT, STEP14_FIELD_STEPS, STEP14_FIELD_METHODS.
"""
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np
import torch

from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from somaforge_core.robot_assets import decode_robot_asset_json
from .recovery_field import RecoveryGeometry, pose_increment


class InvalidWitnessNormal(ValueError):
    """A trial pose cannot supply a valid Newton witness derivative."""


def plain(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    return value


def run(query, experiment=None):
    torch.set_num_threads(2)
    out = Path(os.environ['STEP14_FIELD_OUTPUT']); out.mkdir(parents=True, exist_ok=True)
    if (out / 'diagnostic.json').exists(): raise FileExistsError('Use a new experiment output directory')
    (out / 'queries').mkdir(exist_ok=True)
    source = Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_step14_gradient_20260925_v2/diagnostic.json')
    saved = json.loads(source.read_text())
    decode_robot_asset_json(saved['robot_asset_json'], context='step14 recovery fields')
    old_model = json.loads(Path('runtime/current/archives/preserved_inputs_20261010/baseline1000_step14_static_interval_20260925/model.json').read_text())
    if query.provenance['model_fingerprint'] != old_model['model_fingerprint']:
        raise ValueError('Initialized Newton model differs from the original static fixture')
    device = str(query.model.device)
    fk = CanonicalG1ForwardKinematics().double().to(device)
    q0 = torch.tensor(saved['records'][0]['q'], device=device, dtype=torch.float64)
    scale = q0.new_tensor([.02]*3 + [.1]*32)
    geometry = RecoveryGeometry(saved['terrain'], fk, device=device, samples=512)
    steps = int(os.environ.get('STEP14_FIELD_STEPS', '180'))
    optimizer = os.environ.get('STEP14_FIELD_OPTIMIZER', 'steepest')
    if optimizer not in ('steepest', 'contact_metric'):
        raise ValueError('Unknown research optimizer')
    methods = os.environ.get('STEP14_FIELD_METHODS', 'newton,sdf,shared_top,shared_auto').split(',')
    start = time.monotonic()
    records, attempts, directions, summaries = [], [], {}, {}
    query_count = 0

    def observe(q, label):
        nonlocal query_count
        raw = query.query_device(q[None].float())
        n = int(raw['count'])
        fields = ('dist', 'includemargin', 'type', 'worldid', 'active', 'constraint_allocated', 'constraint_rows',
                  'efc_address', 'shape0', 'shape1', 'body0', 'body1', 'source_key', 'primary_surface', 'part',
                  'task_pair', 'eligible', 'upward', 'full_kind', 'body_link0', 'body_link1', 'normal_w',
                  'geometry_point0_w', 'geometry_point1_w')
        pairs = {k: raw[k][:n].clone() for k in fields}
        if bool((pairs['active'] & ~pairs['constraint_allocated']).any()):
            raise ValueError('Active but unallocated Newton constraints')
        observed = dict(pairs=pairs, link_names=query.tensor_reader.link_names,
                        contact_part_mask=raw['contact_part_mask'][0].clone(), contact_surface=raw['contact_surface'][0].clone(),
                        contact_position_w=raw['contact_position_w'][0].clone(), query_id=query_count)
        torch.save(dict(q=q.detach().cpu(), observed=cpu(observed), label=label,
                        sampling='same fixed-q Newton pass; no integration'), out / 'queries' / f'{query_count:05d}.pt')
        query_count += 1
        # Freeze material witnesses at the queried pose, also for finite differences.
        with torch.no_grad():
            pos, rot = fk.link_poses(q[None], observed['link_names'])
            for side in (0, 1):
                idx = pairs[f'body_link{side}'].clamp_min(0)
                point = pairs[f'geometry_point{side}_w'].double()
                observed[f'local{side}'] = torch.einsum('nij,nj->ni', rot[0, idx].transpose(-1, -2), point-pos[0, idx])
        return observed

    def distances(q, observed):
        pairs = observed['pairs']; p, r = fk.link_poses(q[None], observed['link_names'])
        moved = []
        for side in (0, 1):
            idx = pairs[f'body_link{side}']; safe = idx.clamp_min(0)
            point = p[0, safe] + torch.einsum('nij,nj->ni', r[0, safe], observed[f'local{side}'])
            moved.append(torch.where((idx >= 0)[:, None], point-pairs[f'geometry_point{side}_w'], 0))
        normal = pairs['normal_w'].double()
        valid = pairs['full_kind'] >= 0
        invalid = valid & ~torch.isclose(normal.norm(dim=-1), normal.new_ones(len(normal)), atol=1e-4, rtol=0)
        if bool(invalid.any()):
            raise InvalidWitnessNormal(json.dumps(dict(query_id=observed['query_id'],
                shape0=plain(pairs['shape0'][invalid]), shape1=plain(pairs['shape1'][invalid]),
                normals=plain(normal[invalid]), distance=plain(pairs['dist'][invalid]))))
        return pairs['dist'].double() + ((moved[1]-moved[0])*normal).sum(-1)

    initial = observe(q0, 'original')
    anchors = []
    p0, r0 = fk.link_poses(q0[None], initial['link_names'])
    pairs = initial['pairs']
    for part in initial['contact_part_mask'].nonzero().flatten().tolist():
        mask = pairs['eligible'] & (pairs['part'] == part) & (pairs['primary_surface'] == initial['contact_surface'][part])
        candidates = mask.nonzero().flatten(); j = int(candidates[pairs['dist'][candidates].argmin()])
        link = int(pairs['body_link1'][j]); point = pairs['geometry_point1_w'][j].double()
        if not torch.allclose(pairs['normal_w'][j].double(), q0.new_tensor([0, 0, 1]), atol=1e-6):
            raise ValueError('Expected original horizontal-top material anchors')
        anchors.append(dict(part=part, link=initial['link_names'][link],
                            local=(r0[0, link].T @ (point-p0[0, link])).detach(), target=point.detach(),
                            margin=float(pairs['includemargin'][j]), surface=int(initial['contact_surface'][part])))
    with torch.no_grad():
        _, sep0 = geometry.evaluate(q0)
        auto_faces = sep0[:, :6].argmax(1)
    top_faces = auto_faces.clone(); top_faces[geometry.foot_ids] = geometry.top
    # Original intended contact links keep their known surface in both variants.
    for anchor in anchors:
        ids = [i for i, s in enumerate(geometry.shapes) if s['name'] == anchor['link']]
        anchor['shape_ids'] = ids
        auto_faces[ids] = geometry.top; top_faces[ids] = geometry.top

    def contact_residual(q, sep):
        p, r = fk.link_poses(q[None], tuple(a['link'] for a in anchors))
        residual = []
        for i, a in enumerate(anchors):
            point = p[0, i] + r[0, i] @ a['local']
            gap = sep[a['shape_ids'], geometry.top].amin()
            normal = gap.clamp_max(0) + (gap-.9*a['margin']).relu()
            residual.append(torch.cat((point[:2]-a['target'][:2], normal.reshape(1))))
        return torch.stack(residual)

    def contact_coordinates(q):
        # Metric measures material tangential motion and whole-link normal
        # support even inside the allowed interval; it does not add targets.
        p, r = fk.link_poses(q[None], tuple(a['link'] for a in anchors))
        _, sep = geometry.evaluate(q)
        values = []
        for i, a in enumerate(anchors):
            point = p[0, i] + r[0, i] @ a['local']
            values.append(torch.cat((point[:2], sep[a['shape_ids'], geometry.top].amin().reshape(1))))
        return torch.cat(values)/.005

    def descent(seed, gradient):
        direction = -gradient
        if optimizer == 'contact_metric':
            origin = torch.zeros(35, device=device, dtype=torch.float64)
            jacobian = torch.autograd.functional.jacobian(
                lambda x: contact_coordinates(pose_increment(seed, x, scale)), origin, vectorize=True)
            metric = torch.eye(35, device=device, dtype=torch.float64) + 20*jacobian.T@jacobian
            direction = torch.linalg.solve(metric, direction)
            # Restrict the metric to feasible joint directions at active limits.
            # Clipping an already coupled direction can destroy its descent sign.
            free = torch.ones(35, dtype=torch.bool, device=device)
            for _ in range(30):
                blocked = ((seed[7:] <= fk.joint_lower+1e-7) & (direction[6:] < 0)) | ((seed[7:] >= fk.joint_upper-1e-7) & (direction[6:] > 0))
                if not bool((blocked & free[6:]).any()): break
                free[6:] &= ~blocked
                direction = torch.zeros_like(gradient)
                direction[free] = torch.linalg.solve(metric[free][:, free], -gradient[free])
        return direction / direction.abs().max().clamp_min(1e-12)

    def reduced(depth):
        if not depth.numel(): return depth.sum()
        cost = (depth/.01).square()
        return cost.mean() + cost.amax()

    def objective(q, observed, method):
        world, sep = geometry.evaluate(q)
        d = distances(q, observed); kind = observed['pairs']['full_kind']
        if method == 'newton':
            env = reduced((-d[kind == 0]).relu())
        elif method in ('sdf', 'sdf_sum'):
            depth = (-geometry.sdf(world)).relu()
            # Equal surface sample count per shape; no tessellation-density weight.
            cost = (depth/.01).square()
            env = cost.mean(-1).sum() if method == 'sdf_sum' else cost.mean() + cost.amax()
            env = env + reduced((-sep[:, geometry.ground]).relu())
        else:
            faces = top_faces if method == 'shared_top' else auto_faces
            gap = sep[torch.arange(len(geometry.shapes), device=device), faces]
            env = ((-gap).relu()/.01).square().sum() if method == 'shared_sum' else reduced((-gap).relu())
            env = env + reduced((-sep[:, geometry.ground]).relu())
        own = reduced((-d[kind == 1]).relu())
        anchor = (contact_residual(q, sep)/.005).square().sum()
        reference = ((q[:3]-q0[:3])/.1).square().sum() + ((q[7:]-q0[7:])/.5).square().mean()
        total = env + own + 10*anchor + 1e-4*reference
        return total, dict(environment=env, self_collision=own, contact=anchor, reference=reference)

    def record(label, q, raw, method=None, **extra):
        with torch.no_grad():
            p = raw['pairs']; full = p['full_kind'] >= 0
            depth = lambda mask: max(0., -float(p['dist'][mask].min()))*100 if bool(mask.any()) else 0.
            foot_links = torch.tensor(['ankle' in n for n in raw['link_names']], device=device)
            foot_pair = ((p['body_link0'] >= 0) & foot_links[p['body_link0'].clamp_min(0)]) | ((p['body_link1'] >= 0) & foot_links[p['body_link1'].clamp_min(0)])
            world, sep = geometry.evaluate(q)
            violation = max(float((-sep[:, :6].amax(1)).relu().max()), float((-sep[:, 6]).relu().max()))
            residual = contact_residual(q, sep)
            kept = raw['contact_part_mask'] & initial['contact_part_mask'] & (raw['contact_surface'] == initial['contact_surface'])
            all_kept = bool((kept | ~initial['contact_part_mask']).all())
            joint = float(torch.maximum(fk.joint_lower-q[7:], q[7:]-fk.joint_upper).relu().max())
            full_depth, own_depth = depth(full), depth(p['full_kind'] == 1)
            penetration = full & (p['dist'] < 0)
            row = dict(label=label, method=method, q=plain(q), query_id=raw['query_id'],
                       full_depth_cm=full_depth, feet_depth_cm=depth(full & foot_pair), self_depth_cm=own_depth,
                       shape_plane_violation_cm=100*violation, sampled_sdf_depth_cm=100*float((-geometry.sdf(world)).relu().max()),
                       anchor_error_cm=100*float(residual.norm(dim=-1).max()), anchor_xy_max_cm=100*float(residual[:, :2].norm(dim=-1).max()),
                       all_required_kept=all_kept, original_contact_kept=plain(kept), actual_contact=plain(raw['contact_part_mask']),
                       actual_surface=plain(raw['contact_surface']), contact_position_w=plain(raw['contact_position_w']),
                       joint_violation_rad=joint, root_translation_cm=100*float((q[:3]-q0[:3]).norm()),
                       joint_delta_max_deg=float(torch.rad2deg((q[7:]-q0[7:]).abs().max())),
                       geometry_diagnostic={'left_ankle_roll_link': geometry.foot_audit(q)},
                       penetrating_pairs={k: plain(v[penetration]) for k, v in p.items()},
                       static_recovery_pass=bool(full_depth <= .1 and violation <= .0002 and all_kept and joint <= 1e-6 and float(residual.norm(dim=-1).max()) <= .01),
                       **extra)
            if method:
                loss, parts = objective(q, raw, method)
                row['loss'] = float(loss); row['loss_parts'] = plain(parts)
        records.append(row)
        (out/'progress.json').write_text(json.dumps(dict(records=records, summaries=summaries), indent=2))
        if not method or extra.get('iteration', 0) % 20 == 0 or row['static_recovery_pass'] or 'final' in label:
            print(json.dumps({k: row[k] for k in ('label','full_depth_cm','shape_plane_violation_cm','anchor_error_cm','all_required_kept','static_recovery_pass')}), flush=True)
        return row

    record('original', q0, initial)
    if experiment is not None:
        return experiment(locals())
    # Audit gradients against a genuinely frozen witness surrogate, and SDF/FK.
    for method in methods:
        x = torch.zeros(35, device=device, dtype=torch.float64, requires_grad=True)
        value, parts = objective(pose_increment(q0, x, scale), initial, method)
        gradient = torch.autograd.grad(value, x)[0]
        direction = descent(q0, gradient)
        fd = []
        for axis in range(35):
            delta = torch.zeros_like(x); delta[axis] = 1e-5
            plus = objective(pose_increment(q0, delta, scale), initial, method)[0]
            minus = objective(pose_increment(q0, -delta, scale), initial, method)[0]
            fd.append(float((plus-minus)/2e-5))
        fd = gradient.new_tensor(fd)
        world0, _ = geometry.evaluate(q0)
        probe = pose_increment(q0, .05*direction, scale).detach()
        world1, _ = geometry.evaluate(probe)
        feet = geometry.foot_ids
        field_points = world0[feet].reshape(-1, 3)[::16]
        dx = (world1[feet]-world0[feet]).reshape(-1, 3)[::16]
        directions[method] = dict(gradient=plain(gradient), descent=plain(direction), finite_difference=plain(fd),
                                  fd_relative_l2=float((gradient-fd).norm()/gradient.norm().clamp_min(1e-12)),
                                  probe_scale=.05, foot_dx_mean_m=plain(dx.mean(0)),
                                  dx_points=plain(field_points), dx_vectors=plain(dx), objective=float(value.detach()))
        record(method+'_probe', probe, observe(probe, method+'_probe'), method, dx=plain(dx), dx_points=plain(field_points))
        seed, raw = q0.clone(), initial
        last = None; status = 'iteration_budget'
        for iteration in range(1, steps+1):
            x = torch.zeros(35, device=device, dtype=torch.float64, requires_grad=True)
            value, _ = objective(pose_increment(seed, x, scale), raw, method)
            gradient = torch.autograd.grad(value, x)[0]
            direction = descent(seed, gradient)
            # Feasible tangent direction at limits; no change to the robot limits.
            direction[6:] = torch.where((seed[7:] <= fk.joint_lower+1e-9) & (direction[6:] < 0), 0., direction[6:])
            direction[6:] = torch.where((seed[7:] >= fk.joint_upper-1e-9) & (direction[6:] > 0), 0., direction[6:])
            accepted = False
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625, .001953125, .0009765625, .000244140625, .00006103515625, .0000152587890625):
                candidate = pose_increment(seed, alpha*direction, scale).detach()
                # Project onto the true joint box rather than stopping merely
                # because every discrete trial step overshoots an almost-active limit.
                candidate[7:] = torch.maximum(torch.minimum(candidate[7:], fk.joint_upper), fk.joint_lower)
                fresh = observe(candidate, f'{method}_{iteration}_{alpha}')
                try:
                    with torch.no_grad(): next_value, _ = objective(candidate, fresh, method)
                except InvalidWitnessNormal as exc:
                    # Preserve raw invalid trial evidence, reject it, and shrink.
                    # Never replace an unknown derivative with geometry or zero.
                    attempts.append(dict(method=method, iteration=iteration, alpha=alpha,
                                         query_id=fresh['query_id'], rejected='invalid_newton_normal', details=str(exc)))
                    continue
                attempts.append(dict(method=method, iteration=iteration, alpha=alpha, before=float(value.detach()), after=float(next_value), query_id=fresh['query_id']))
                if float(next_value) < float(value.detach())-1e-9:
                    seed, raw = candidate, fresh; accepted = True
                    last = record(f'{method}_iter{iteration}', seed, raw, method, iteration=iteration, alpha=alpha, gradient_norm=float(gradient.norm()))
                    break
            if not accepted:
                status = 'line_search_stalled'; break
            if last['static_recovery_pass']:
                status = 'static_recovery_pass'; break
        final = record(method+'_final', seed, raw, method, iteration=iteration, status=status,
                       last_gradient=plain(gradient), last_direction=plain(direction),
                       last_directional_derivative=float(gradient@direction),
                       active_joint_limits=plain(((seed[7:] <= fk.joint_lower+1e-7) | (seed[7:] >= fk.joint_upper-1e-7)).nonzero().flatten()))
        summaries[method] = final
        # Audit solver rows at the same final pose, independently of the loss.
        observe(seed, method+'_audit')
        (out/(method+'_newton_audit.json')).write_text(json.dumps(plain(query.audit_current(seed[None].float())), indent=2))
    result = dict(schema='step14_gradient_fields_v1', source=str(source), terrain=saved['terrain'],
                  robot_asset_json=saved['robot_asset_json'], joint_names=saved['joint_names'], records=records,
                  summaries=summaries, directions=directions, attempts=attempts, anchors=plain(anchors),
                  auto_faces=plain(auto_faces), top_faces=plain(top_faces), face_normals=plain(geometry.normal),
                  canonical_shapes=[dict(name=s['name'], kind=s['kind']) for s in geometry.shapes],
                  provenance=query.provenance, coordinate_scale=plain(scale), query_count=query_count,
                  elapsed_seconds=time.monotonic()-start, samples_per_shape=512,
                  optimizer=optimizer, optimizer_contract='scaled gradient descent with fresh-query backtracking; no QP; '
                  'contact_metric uses (I+20 J_contact^T J_contact)^-1 on the unchanged loss gradient',
                  contract='Fixed pose research. SDF and plane violations are geometric guidance, never contact truth. '
                           'Newton actual active/allocated primary-horizontal-face contacts determine retention. '
                           'Auto faces chosen once at initial pose; top variant is explicitly privileged diagnostic. '
                           'Independent foot audit uses all original collision-mesh vertices. No dynamics or general recovery guarantee.')
    (out/'diagnostic.json').write_text(json.dumps(plain(result), indent=2))
    print('STEP14_GRADIENT_FIELD_COMPLETE', flush=True)


if __name__ == '__main__':
    import somaforge_core.newton_tensor_transport as transport
    transport.serve_tensor_queries = lambda query, *_args, **_kwargs: run(query)
    runpy.run_path('scripts/serve_newton_contact_queries.py', run_name='__main__')
