"""GPU loss/acceptance adapters for freshly queried actual Newton tensors."""
import torch
from contact_solver.contact_layout import endpoint_position_statistics, endpoint_position_loss
from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE


def reduce_groups(values, groups, valid, size, *, minimum=False):
    initial = torch.inf if minimum else -torch.inf
    out = values.new_full((size,), initial)
    out = out.scatter_reduce(0, groups, torch.where(valid, values, initial),
        reduce='amin' if minimum else 'amax', include_self=False)
    return torch.where(torch.isfinite(out), out, 0)


def group_any(groups, mask, size):
    out = torch.zeros(size, dtype=torch.long, device=groups.device)
    return out.scatter_add(0, groups, mask.long()) > 0


def first_minimum(values, groups, valid, size):
    minimum = values.new_full((size,), torch.inf)
    minimum.scatter_reduce_(0, groups, torch.where(valid, values, torch.inf), reduce='amin')
    row = torch.arange(len(values), device=values.device)
    winner = torch.full((size,), len(values), device=values.device, dtype=torch.long)
    winner.scatter_reduce_(0, groups, torch.where(valid & (values == minimum[groups]), row, len(values)), reduce='amin')
    return winner, winner < len(values)


def intended_surfaces(points, active, observed):
    distance = (torch.einsum('bpi,bfi->bpf', points.detach(), observed['face_normal'])
                - observed['face_offset'][:, None]).abs()
    distance = distance.masked_fill(observed['face_surface'][:, None] < 0, torch.inf)
    surface = observed['face_surface'].gather(1, distance.argmin(-1))
    return torch.where(active, surface, -1)


def acceptance(observed, active, surface):
    pair = observed['pairs']; sample = pair['sample']; part = pair['part'].clamp(0, 5)
    batch = len(active); group = sample*6+part
    matching = active[sample, part] & (pair['primary_surface'] == surface[sample, part])
    realized = group_any(group, pair['eligible'] & matching, batch*6).reshape(batch, 6)
    extra = pair['eligible'] & ~matching
    overlap = pair['includemargin']-pair['dist']
    ignored = extra & (pair['includemargin'] > 0) & (overlap <=
        DEFAULT_ACCEPTANCE.extra_activation_margin_fraction*pair['includemargin'])
    blocking = extra & ~ignored
    ignored_count = torch.zeros(batch, device=active.device).scatter_add(0, sample, ignored.float())
    blocking_count = torch.zeros_like(ignored_count).scatter_add(0, sample, blocking.float())
    maximum = reduce_groups(overlap, sample, extra, batch)
    accepted = ((realized | ~active).all(-1)) & (blocking_count == 0)
    return dict(contact_accepted=accepted, realized=realized, ignored=ignored_count,
        blocking=blocking_count, maximum=maximum)


class DeviceWitnessRows:
    """One fresh query, with frozen geometry witnesses and differentiable FK."""
    def __init__(self, fk, q, observed, *, world_frame=None, collision_aggregation='max'):
        if observed.get('schema') != 'newton_device_witness_batch_v1':
            raise ValueError('Expected fresh device Newton witnesses')
        self.q, self.observed, self.pair = q, observed, observed['pairs']
        self.world_frame = world_frame
        if collision_aggregation not in ('max', 'body_mean_plus_max'):
            raise ValueError('unknown collision aggregation')
        self.collision_aggregation = collision_aggregation
        p = self.pair
        self.sample = p['sample']; self.part = p['part'].clamp(0, 5)
        self.group = self.sample*6+self.part
        points = torch.stack((p['geometry_point0_w'], p['geometry_point1_w']), 1)
        normal = p['normal_w']
        if world_frame is not None:
            origin, basis = world_frame
            points = torch.einsum('npi,nij->npj', points.double()-origin[self.sample, None].double(),
                                  basis[self.sample].double()).to(q)
            normal = torch.einsum('ni,nij->nj', normal.double(), basis[self.sample].double()).to(q)
        self.normal_valid = torch.isfinite(normal).all(-1) & torch.isclose(normal.norm(dim=-1),
            torch.ones_like(normal[:, 0]), atol=1e-4, rtol=0)
        if bool((p['task_pair'] & ~self.normal_valid).any()):
            raise ValueError('Invalid actual task-pair normal in GPU witness loss')
        self.points = points
        self.transformed_normal = normal
        position, rotation = fk.link_poses(q, observed['link_names'])
        links = torch.stack((p['body_link0'], p['body_link1']), -1)
        pos = position[self.sample[:, None], links.clamp_min(0)]
        rot = rotation[self.sample[:, None], links.clamp_min(0)]
        material = torch.matmul(rot.detach().transpose(-1, -2), (points-pos.detach())[..., None]).detach()
        moved = pos+torch.matmul(rot, material).squeeze(-1)
        self.moving = torch.where((links >= 0)[..., None], moved, points)
        displacement = self.moving-self.moving.detach()
        normal = torch.where(self.normal_valid[:, None], normal, 0)
        self.distances = p['dist']+(normal*(displacement[:, 1]-displacement[:, 0])).sum(-1)

    def __len__(self):
        return len(self.q)

    def full_body(self):
        p = self.pair; size = len(self)*2
        group = self.sample*2+p['full_kind'].clamp(0, 1)
        chosen, present = first_minimum(p['dist'], group, (p['full_kind'] >= 0) & (p['dist'] < 0), size)
        self.full_body_selected = chosen[present]
        # Append a differentiable zero for groups with no penetration.
        distance = torch.cat((self.distances, self.q.sum().reshape(1)*0))
        depth = torch.where(present, -distance[chosen], 0).reshape(len(self), 2)
        validity = torch.cat((self.normal_valid, torch.ones(1, dtype=torch.bool, device=self.q.device)))
        invalid = (present & ~validity[chosen]).reshape(len(self), 2).sum(-1).to(self.q)
        return torch.cat((depth, self.q.sum(-1, keepdim=True)*0), -1).amax(-1), invalid

    def representatives(self, active, surface, *, actual=False):
        p = self.pair
        match = p['task_pair'] & active[self.sample, self.part] & (p['primary_surface'] == surface[self.sample, self.part])
        if actual:
            match &= p['eligible']
        chosen, present = first_minimum(p['dist'], self.group, match, len(self)*6)
        return chosen, present.reshape(len(self), 6)

    def body_collision_loss(self):
        """Worst witness per robot link, averaged over penetrating links.

        Duplicate shapes/manifold points cannot increase a link's weight.
        Self-collision witnesses supervise both involved robot links. The
        original global worst-depth penalty is retained by the caller.
        """
        p = self.pair
        links = len(self.observed['link_names'])
        values, groups, valid = [], [], []
        for name in ('body_link0', 'body_link1'):
            link = p[name]
            values.append((-self.distances-DEFAULT_ACCEPTANCE.shallow_penetration_m).relu())
            groups.append(self.sample * links + link.clamp_min(0))
            valid.append((p['full_kind'] >= 0) & (p['dist'] < 0) & (link >= 0))
        depth = reduce_groups(torch.cat(values), torch.cat(groups), torch.cat(valid),
                              len(self) * links).reshape(len(self), links)
        present = group_any(torch.cat(groups), torch.cat(valid), len(self)*links).reshape(len(self), links)
        return (depth/.005).square().sum(-1) / present.sum(-1).clamp_min(1)

    def spatial_representatives(self, points, active, surface, *, actual=False, regions=None):
        """Closest matching geometric witness; activation is never inferred here.

        Points are local for differentiable FK, world for actual raw witnesses.
        Region IDs must be classified once from canonical body-local geometry.
        """
        p = self.pair
        values = p['geometry_point1_w'] if actual else self.moving[:, 1]
        match = p['task_pair'] & active[self.sample, self.part] & (p['primary_surface'] == surface[self.sample, self.part])
        if actual:
            match &= p['eligible']
        if regions is not None:
            match &= regions[self.sample, self.part, self.contact_region_ids]
        error = (values.detach()[..., :2]-points.detach()[self.sample, self.part, :2]).square().sum(-1)
        chosen, present = first_minimum(error, self.group, match, len(self)*6)
        return chosen, present.reshape(len(self), 6)

    def frozen_layout_diagnostic(self, points, active, surface, *, regions=None):
        chosen, present = self.spatial_representatives(points, active, surface, regions=regions)
        values = torch.cat((self.moving[:, 1], self.q[:1, :3]*0), 0)
        witnesses = values[chosen].reshape(len(self), 6, 3)
        error, count = endpoint_position_statistics(witnesses, points.detach(), present)
        return endpoint_position_loss(error), dict(
            relative_layout_rms_cm=torch.where(count >= 1, 100*error.clamp_min(1e-16).sqrt(), 0),
            relative_layout_observed_parts=count.to(self.q))

    def actual_witness_position_statistics(self, points, active, surface):
        chosen, present = self.spatial_representatives(points, active, surface, actual=True)
        values = torch.cat((self.pair['geometry_point1_w'], points.new_zeros(1, 3)), 0)
        witnesses = values[chosen].reshape(len(self), 6, 3)
        error, count = endpoint_position_statistics(witnesses, points, present)
        return error, active.any(-1) & (present | ~active).all(-1), count

    def witness_position_statistics(self, points_world, active, surface, *, actual=False, regions=None):
        """Issued points relative to the issuing observation, including one contact.

        The saved world points are transport coordinates, not world-position
        supervision: shifting the entire observation and plan changes no error.
        This is a historical witness-location diagnostic. Production position
        checks use native_position_statistics over actual part surfaces.
        It never replaces contact/safety or temporal touchdown verification.
        """
        target = points_world.detach()
        if actual:
            values = self.pair['geometry_point1_w']
        else:
            values = self.moving[:, 1]
            if self.world_frame is not None:
                origin, basis = self.world_frame
                target = torch.einsum('bji,bpj->bpi', basis, target-origin[:, None])
        chosen, present = self.spatial_representatives(target, active, surface, actual=actual, regions=regions)
        values = torch.cat((values, target.new_zeros(1, 3)), 0)
        witnesses = values[chosen].reshape(len(self), 6, 3)
        count = present.sum(-1)
        error = (((witnesses[..., :2]-target[..., :2])**2).sum(-1)*present).sum(-1)/count.clamp_min(1)
        complete = active.any(-1) & (present | ~active).all(-1)
        return error, complete, count


def realization(rows, active, surface, *, invalid_policy='error', audit_path=None):
    if invalid_policy not in ('error', 'detach'):
        raise ValueError('Unknown invalid witness policy')
    q, p, observed = rows.q, rows.pair, rows.observed
    sample, part, group = rows.sample, rows.part, rows.group
    batch = len(q); size = batch*6
    wanted = active[sample, part]
    matching = p['task_pair'] & wanted & (p['primary_surface'] == surface[sample, part])
    exists = group_any(group, matching, size).reshape(batch, 6)
    realized = group_any(group, matching & p['active'] & p['constraint_allocated'], size).reshape(batch, 6)
    # One gap interval: upper violations here, lower violations exactly once
    # in the full-body safety term. Activation only supplies diagnostics.
    pair_cost = reduce_groups((rows.distances-p['includemargin']).relu().square(),
                             group, matching, size, minimum=True).reshape(batch, 6)
    missing = active & ~exists
    unwanted_pairs = p['task_pair'] & p['upward'] & ~wanted
    unwanted = reduce_groups((p['includemargin']-rows.distances).relu().square(),
                             group, unwanted_pairs, size).reshape(batch, 6)
    unwanted_active = group_any(group, unwanted_pairs & p['active'] & p['constraint_allocated'], size).reshape(batch, 6)
    depth, invalid = rows.full_body()
    if bool((invalid > 0).any()):
        import json
        details = dict(schema='newton_invalid_device_witness_v1', action='raw_distance_retained_no_collision_gradient',
            samples=(invalid > 0).nonzero().flatten().cpu().tolist(),
            invalid_counts=invalid.detach().cpu().tolist())
        selected = rows.full_body_selected
        bad = selected[~rows.normal_valid[selected]]
        details['witnesses'] = {key: value[bad].detach().cpu().tolist() for key, value in p.items()}
        details['q'] = q[rows.sample[bad]].detach().cpu().tolist()
        details['transformed_normal'] = rows.transformed_normal[bad].detach().cpu().tolist()
        details['transformed_points'] = rows.points[bad].detach().cpu().tolist()
        details['basis'] = None if rows.world_frame is None else rows.world_frame[1][rows.sample[bad]].detach().cpu().tolist()
        if invalid_policy == 'error':
            raise ValueError('Invalid full-body normal: '+json.dumps(details))
        if audit_path is not None:
            with open(audit_path, 'a') as stream:
                stream.write(json.dumps(details)+'\n')
        else:
            import warnings
            warnings.warn('Invalid full-body normal, gradient detached: '+json.dumps(details), RuntimeWarning)
    count = active.sum(-1).clamp_min(1)
    normalized = pair_cost/.02**2
    contact_loss = (normalized*active).sum(-1)/count+(normalized*active).amax(-1)
    normalized_unwanted = unwanted/.02**2
    unwanted_loss = normalized_unwanted.mean(-1)+normalized_unwanted.amax(-1)
    collision_loss = ((depth-DEFAULT_ACCEPTANCE.shallow_penetration_m).relu()/.005).square()
    body_collision = q.sum(-1) * 0
    if rows.collision_aggregation == 'body_mean_plus_max':
        # Any invalid full-body pair must be visible to safety/advance checks,
        # including witnesses that are not the globally deepest pair.
        bad = (p['full_kind'] >= 0) & (p['dist'] < 0) & ~rows.normal_valid
        if bool(bad.any()):
            raise ValueError('Invalid full-body witness in multi-body collision objective')
        body_collision = rows.body_collision_loss()
        collision_loss = collision_loss + body_collision
    audit = acceptance(observed, active, surface)
    exact = (observed['contact_part_mask'] == active).all(-1) & (
        (observed['contact_surface'] == surface) | ~active).all(-1)
    metrics = dict(
        newton_contact_accepted=audit['contact_accepted'].float(),
        newton_generation_accepted=(audit['contact_accepted'] & torch.isfinite(depth) & (depth >= 0) &
            (depth <= DEFAULT_ACCEPTANCE.shallow_penetration_m) & (invalid == 0)).float(),
        newton_ignored_extra_pairs=audit['ignored'], newton_blocking_extra_pairs=audit['blocking'],
        newton_max_extra_activation_overlap_m=audit['maximum'],
        newton_plan_realization_loss=contact_loss, newton_unwanted_contact_loss=unwanted_loss,
        newton_collision_loss=collision_loss, newton_plan_exact=exact.float(),
        newton_generation_valid=(exact & (depth == 0) & (invalid == 0)).float(),
        newton_intended_contacts=active.sum(-1).float(),
        newton_realized_intended_contacts=(active & realized).sum(-1).float(),
        newton_unrealized_intended_contacts=(active & ~realized).sum(-1).float(),
        newton_missing_intended_pairs=missing.sum(-1).float(), newton_missing_intended_mask=missing,
        newton_unwanted_contact_parts=unwanted_active.sum(-1).float(), newton_penetration_cm=100*depth,
        newton_invalid_fullbody_witnesses=invalid, newton_configured_includemargin=observed['configured_margin'],
        newton_body_collision_loss=body_collision)
    return contact_loss+unwanted_loss+collision_loss, metrics
