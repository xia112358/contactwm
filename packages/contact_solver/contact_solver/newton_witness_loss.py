"""Local separation derivatives from freshly queried Newton geometry pairs.

This is not a differentiable contact classifier. At the supplied pose the
value is exactly the solver distance; the derivative freezes its current
normal/material witness. Each invocation must query again. Discrete face and
activation decisions remain Newton's, and absent pairs have no such gradient.
"""
import numpy as np
import torch
from somaforge_core.newton_contact_query import query_contacts

SCHEMA = 'newton_refreshed_witness_v1'


def full_body_violation(fk,q,separation,*,world_frame=None,signed=False,invalid_policy='error',return_validity=False,audit_path=None):
    """Refreshed Newton worst terrain/self-pair derivative on BOTH robot bodies.

The supplied summaries must be from a fresh query at q. Both witness motion
terms are essential for self-collision. No inverse-kinematics/output projection.
"""
    if invalid_policy not in ('error','detach'):raise ValueError('Unknown invalid witness policy')
    if invalid_policy=='detach' and not return_validity:raise ValueError('Detached invalid witnesses require explicit validity reporting')
    if len(separation)!=len(q):raise ValueError('Full-body witness batch mismatch')
    keys = ('closest_terrain', 'closest_self') if signed else ('worst_terrain', 'worst_self')
    flat = [(sample, slot, row[key]) for sample, row in enumerate(separation)
            for slot, key in enumerate(keys) if row[key] is not None]
    counts = q.new_zeros(len(q))
    if not flat:
        result = q.sum(-1) * 0
        if signed: result = result - float('inf')
        return (result, counts) if return_validity else result
    if any(not np.isfinite(w['dist']) for _, _, w in flat):
        raise ValueError('Nonfinite full-body distance')
    si = torch.tensor([sample for sample, _, _ in flat], device=q.device)
    slots = torch.tensor([slot for _, slot, _ in flat], device=q.device)
    # Transfer the ragged Newton records together; only exceptional witnesses
    # take the detailed Python audit path below.
    values = q.new_tensor([w['normal_w'] + w['point0_w'] + w['point1_w'] + [w['dist']]
                           for _, _, w in flat])
    normal, points = values[:, :3], values[:, 3:9].reshape(-1, 2, 3)
    if world_frame is not None:
        origin, basis = world_frame
        normal = torch.einsum('ni,nij->nj', normal, basis[si])
        points = torch.einsum('npi,nij->npj', points-origin[si, None], basis[si])
    norm = normal.norm(dim=-1)
    valid = torch.isfinite(normal).all(-1) & torch.isclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0)
    invalid_indices = (~valid).nonzero(as_tuple=False).flatten().cpu().tolist()
    for index in invalid_indices:
        import json
        sample, _, witness = flat[index]
        details = dict(batch_index=sample, witness=witness, q=q[sample].detach().cpu().tolist(),
            transformed_normal=normal[index].detach().cpu().tolist(),
            transformed_norm=float(norm[index].detach()),
            basis=None if world_frame is None else world_frame[1][sample].detach().cpu().tolist())
        if invalid_policy == 'error':
            raise ValueError('Invalid full-body normal: ' + json.dumps(details))
        details.update(schema='newton_invalid_witness_detached_v1', action='raw_distance_retained_no_collision_gradient', geometry_valid=False)
        if audit_path is not None:
            with open(audit_path, 'a') as stream: stream.write(json.dumps(details)+'\n')
        else:
            import warnings
            warnings.warn('Invalid full-body normal, gradient detached: '+json.dumps(details), RuntimeWarning)
    counts.scatter_add_(0, si, (~valid).to(q))
    # Invalid normals must not contaminate backward with NaN * zero. Retain
    # their solver distance, with zero pose derivative and explicit invalidity.
    normal = torch.where(valid[:, None], normal, torch.zeros_like(normal))
    points = torch.where(valid[:, None, None], points, torch.zeros_like(points))
    names = tuple(sorted({w[f'body_name{side}'] for _, _, w in flat for side in (0, 1)
                          if w.get(f'body_name{side}') is not None}))
    displacement = q[si, None, :3].expand(-1, 2, -1) * 0
    if names:
        positions, rotations = fk.link_poses(q, names)
        lookup = {name: index for index, name in enumerate(names)}
        indices = torch.tensor([[lookup.get(w[f'body_name{side}'], 0) for side in (0, 1)]
                                for _, _, w in flat], device=q.device)
        moving_body = torch.tensor([[w[f'body_name{side}'] is not None for side in (0, 1)]
                                    for _, _, w in flat], device=q.device)
        position, rotation = positions[si[:, None], indices], rotations[si[:, None], indices]
        material = torch.matmul(rotation.detach().transpose(-1, -2), (points-position.detach()).unsqueeze(-1)).detach()
        moved = position + torch.matmul(rotation, material).squeeze(-1)
        displacement = (moved-moved.detach()) * (moving_body & valid[:, None])[..., None]
    distance = values[:, 9] + (normal * (displacement[:, 1]-displacement[:, 0])).sum(-1)
    violation = -distance if signed else (-distance).relu()
    # amax distributes ties like the original scalar max; max(dim) would
    # incorrectly assign a tied terrain/self collision to only one witness.
    table = q.sum(-1, keepdim=True).expand(-1, 2) * 0 - float('inf')
    table = table.index_put((si, slots), violation)
    if not signed: table = torch.cat((table, q.sum(-1, keepdim=True)*0), -1)
    result = table.amax(-1)
    return (result, counts) if return_validity else result


def query_local_distances(fk, q, scene=None, *, query=None, world_frame=None, fingerprints=None):
    """Return ragged (pair, local_distance) rows in q's coordinate frame.

The query must own the matching authoritative scene. Native-scene callers
pass None and native world q; frame-bound callers pass their explicit scene.
No cached response is accepted by this API.
"""
    provider = query_contacts if query is None else query
    geometry = None if scene is None else [scene[k].detach().cpu().numpy() for k in
        ('box_center', 'box_rotation', 'box_half_extents', 'ground_height')]
    query_q = q.detach().cpu().numpy().copy()
    if world_frame is not None:
        # Explicit caller-owned frame of the initialized native scene; not an
        # automatic fallback when a frame-bound scene query is rejected.
        import numpy as np
        from scipy.spatial.transform import Rotation
        origin, basis = [x.detach().cpu().numpy() for x in world_frame]
        if origin.shape != (len(q),3) or basis.shape != (len(q),3,3):
            raise ValueError('Native query frame batch mismatch')
        if (not np.isfinite(origin).all() or not np.isfinite(basis).all()
                or not np.allclose(basis.transpose(0,2,1)@basis, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(basis), 1., atol=1e-5)):
            raise ValueError('Invalid native query rigid frame')
        query_q[:,:3] = np.einsum('bij,bj->bi',basis,query_q[:,:3])+origin
        query_q[:,3:7] = (Rotation.from_matrix(basis)*Rotation.from_quat(query_q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
        geometry = None
    if fingerprints is not None:
        if geometry is not None:raise ValueError('Fingerprint routing requires an explicit native-world frame')
        from somaforge_core.newton_scene_router import configured_router
        encoded = fingerprints.detach().cpu().numpy()
        if encoded.shape != (len(q),32) or encoded.dtype.name != 'uint8':
            raise ValueError('Expected loss-only [B,32] fingerprint bytes')
        result = configured_router()(query_q, [bytes(row).hex() for row in encoded])
    else:
        result = provider(query_q, geometry)
    if 'candidate_pairs' not in result:
        raise ValueError('Missing Newton candidate pairs; no geometric fallback')
    if len(result['candidate_pairs']) != len(q):
        raise ValueError('Newton witness batch mismatch')
    names = tuple(sorted({p['body_name'] for row in result['candidate_pairs'] for p in row
                          if 'body_name' in p}))
    if names:
        positions, rotations = fk.link_poses(q, names)
        name_index = {name:i for i,name in enumerate(names)}
    output = []; flat = []; sample_ids = []; body_ids = []
    for i, pairs in enumerate(result['candidate_pairs']):
        rows = []
        for original_pair in pairs:
            pair = dict(original_pair)
            if world_frame is not None:
                for field in ('position_w','terrain_position_w'):
                    if pair.get(field) is not None:
                        pair[field] = ((np.asarray(pair[field])-origin[i])@basis[i]).tolist()
                pair['normal_w'] = (np.asarray(pair['normal_w'])@basis[i]).tolist()
            if pair.get('witness_schema') != 'newton_geometry_pair_v1' or not pair.get('geometry_witness_available'):
                raise ValueError('Missing actual Newton geometry witnesses; refresh query worker')
            if not (int(pair['constraint_type']) & 1):
                raise ValueError('Non-constraint pair in Newton witness loss')
            active = float(pair['dist']) < float(pair['includemargin'])
            if bool(pair['constraint_active']) != active:
                raise ValueError('Inconsistent Newton activation')
            if active and (not pair['allocated'] or any(a < 0 for a in pair['efc_address'])):
                raise ValueError('Unallocated active Newton contact')
            rows.append(pair);flat.append(pair);sample_ids.append(i)
            body_ids.append(name_index[pair['body_name']])
        output.append(rows)
    if flat:
        # One transfer and two aggregate checks, not five transfers/checks per pair.
        values=q.new_tensor([p['position_w']+p['normal_w']+p['terrain_position_w']+
                             [p['dist'],p['includemargin']] for p in flat])
        if not torch.isfinite(values).all():raise ValueError('Nonfinite Newton geometry pair')
        normal=values[:,3:6]
        if not torch.isclose(normal.norm(dim=-1),torch.ones_like(values[:,0]),atol=1e-4,rtol=0).all():
            raise ValueError('Non-unit Newton normal')
        si=torch.tensor(sample_ids,device=q.device);bi=torch.tensor(body_ids,device=q.device)
        p,r=positions[si,bi],rotations[si,bi]
        local=torch.bmm(r.detach().transpose(1,2),(values[:,:3]-p.detach()).unsqueeze(-1)).detach()
        moved=p+torch.bmm(r,local).squeeze(-1)
        distances=values[:,9]+(normal*(moved-moved.detach())).sum(-1)
        offset=0
        for i,pairs in enumerate(output):
            output[i]=list(zip(pairs,distances[offset:offset+len(pairs)].unbind()))
            offset+=len(pairs)
    return output, result


def selected_contact_cost(q, rows, active, surface, *, activation_buffer_fraction=0., continuous_gap=False):
    """Any actual pair can realize a part/face intent; do not pin every point.

Missing target pairs are reported separately, NOT scored as achieved contact.
Callers supply a separately named demonstrated approach loss for those rows.
With continuous_gap=True the upper-gap residual is independent of activation;
the caller must supply the lower bound once through full-body safety. The
legacy buffered mode remains available for existing projector callers.
"""
    if not 0 <= activation_buffer_fraction < 1:
        raise ValueError('Activation buffer must be a fraction in [0,1)')
    if continuous_gap and activation_buffer_fraction:
        raise ValueError('Continuous gap objective uses the actual margin without an interior target')
    active=active.detach().cpu().tolist();surface=surface.detach().cpu().tolist()
    costs = []; missing = []; realized = []
    for i, pairs in enumerate(rows):
        part_cost = []; part_missing = []; part_realized = []
        for part in range(6):
            wanted = bool(active[i][part])
            matches = [(p, d) for p, d in pairs if int(p['part']) == part
                       and int(p['surface']) == int(surface[i][part])]
            is_realized = wanted and any(
                p['constraint_active'] and p['allocated'] for p, _ in matches
            )
            if wanted and matches and (continuous_gap or not is_realized):
                # Training target only. Actual realized contact ALWAYS comes
                # from the unchanged solver activation/allocation flags below.
                # A positive buffer avoids an asymptotic approach to the strict
                # dist < margin boundary without ever entering it.
                if activation_buffer_fraction and any(float(p['includemargin'])<=0 for p,_ in matches):
                    raise ValueError('Buffered approach requires positive actual includemargin')
                part_cost.append(matches)
            else:
                part_cost.append([])
            part_missing.append(wanted and not matches)
            part_realized.append(is_realized)
        costs.extend(part_cost); missing.append(part_missing); realized.append(part_realized)
    return _group_penalties(q,costs,minimum=True,margin_scale=1-activation_buffer_fraction), torch.tensor(missing, device=q.device), torch.tensor(realized, device=q.device)


def selected_contact_activation_deficit(
    q,
    rows,
    active,
    surface,
    *,
    interior_fraction=0.05,
):
    """Return the normalized distance still needed to activate each intent.

    This is privileged training supervision, not an observation.  Every
    inactive candidate is compared with its own Newton ``includemargin`` and
    the best candidate for a limb/surface intent supplies the gradient.  A
    small relative interior target avoids the strict ``dist < margin`` edge;
    an already active and allocated contact has exactly zero cost.
    """

    if not 0.0 <= interior_fraction < 1.0:
        raise ValueError("interior_fraction must be in [0,1)")
    if active.dtype != torch.bool or active.shape != surface.shape:
        raise ValueError("active must be bool and match surface")
    if active.shape != (len(rows), 6):
        raise ValueError("contact activation intent must be [B,6]")

    wanted_rows = active.detach().cpu().tolist()
    surfaces = surface.detach().cpu().tolist()
    losses = []
    missing = []
    realized = []
    zero = q.sum() * 0.0
    for sample, pairs in enumerate(rows):
        sample_losses = []
        sample_missing = []
        sample_realized = []
        for part in range(6):
            wanted = bool(wanted_rows[sample][part])
            matches = [
                (pair, distance)
                for pair, distance in pairs
                if int(pair["part"]) == part
                and int(pair["surface"]) == int(surfaces[sample][part])
            ] if wanted else []
            is_realized = wanted and any(
                bool(pair["constraint_active"]) and bool(pair["allocated"])
                for pair, _ in matches
            )
            if is_realized or not matches:
                sample_losses.append(zero)
            else:
                candidate_losses = []
                for pair, distance in matches:
                    margin = float(pair["includemargin"])
                    if margin <= 0.0:
                        raise ValueError(
                            "contact activation deficit requires positive actual includemargin"
                        )
                    target = margin * (1.0 - interior_fraction)
                    # Keep a non-vanishing gradient right up to activation.
                    # Squaring this hinge made the final millimetres—the only
                    # ones that decide Newton contact—needlessly difficult.
                    candidate_losses.append(((distance - target) / margin).relu())
                sample_losses.append(torch.stack(candidate_losses).amin())
            sample_missing.append(wanted and not matches)
            sample_realized.append(is_realized)
        losses.append(torch.stack(sample_losses))
        missing.append(sample_missing)
        realized.append(sample_realized)
    return (
        torch.stack(losses),
        torch.tensor(missing, dtype=torch.bool, device=q.device),
        torch.tensor(realized, dtype=torch.bool, device=q.device),
    )


def _group_penalties(q,groups,*,minimum,margin_scale=1.):
    """Batched arithmetic and tie-preserving reduction over ragged manifolds."""
    flat=[item for group in groups for item in group]
    zero=q.sum(-1,keepdim=True).expand(-1,6)*0
    if not flat:return zero
    distances=torch.stack([d for _,d in flat])
    margins=q.new_tensor([float(p['includemargin'])*margin_scale for p,_ in flat])
    penalties=((distances-margins) if minimum else (margins-distances)).relu().square()
    ids=torch.tensor([j for j,g in enumerate(groups) for _ in g],device=q.device)
    initial=q.new_full((len(groups),),float('inf') if minimum else -float('inf'))
    values=initial.scatter_reduce(0,ids,penalties,reduce='amin' if minimum else 'amax',include_self=False)
    present=torch.tensor([bool(g) for g in groups],device=q.device).reshape_as(zero)
    return torch.where(present,values.reshape_as(zero),zero)


def unwanted_contact_cost(q, rows, active, eligible_surfaces):
    """Separate unwanted task contacts; do not pin each manifold point.

Only demonstrated non-contact parts are pushed away. Multiple surfaces of a
wanted part are not independently forbidden: the shared task selector assigns
one representative surface and valid demonstrations can contain edge pairs.
Max per part avoids multiplying a loss by manifold point count. Activation
remains Newton's unchanged strict condition, not this optimization residual.
"""
    active=active.detach().cpu().tolist()
    costs=[];counts=[]
    for i,pairs in enumerate(rows):
        per_part=[];per_count=[]
        for part in range(6):
            bad=[(p,d) for p,d in pairs if int(p['part'])==part
                 and int(p['surface']) in eligible_surfaces[i] and not bool(active[i][part])]
            per_part.append(bad)
            per_count.append(any(p['constraint_active'] and p['allocated'] for p,_ in bad))
        costs.extend(per_part);counts.append(per_count)
    return _group_penalties(q,costs,minimum=False),torch.tensor(counts,device=q.device)
