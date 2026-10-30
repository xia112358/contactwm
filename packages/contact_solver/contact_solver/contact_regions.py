"""Compact canonical collision regions and actual-Newton witness attribution.

Regions partition physical body coordinates; they never define contact truth.
Only selected, activated and allocated Newton pairs produce contact labels.
"""
import itertools

import torch
from torch import nn

from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.motion_contracts import BODY_NAMES
from contact_solver.part_collision_geometry import PartCollisionGeometry
from contact_solver.constraint_penalty import constraint_penalty
from contact_solver.endpoint_penetration import huber_squared
from contact_solver.endpoint_task import PENETRATION_LOSS_SCALE_M
from contact_solver.contact_surface_interval import finite_face_band_cost, SurfaceIntervalBounds
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_RESOLUTION_M


CONTACT_INTERVAL_SCHEMA = 'native_activation_upper_gap_v1'
PART_NAMES = ('left_foot', 'right_foot', 'left_hand', 'right_hand', 'left_knee', 'right_knee')


def interval_penalty_components(lower,upper):
    """One target-query penalty with exactly additive bound diagnostics."""
    total=lower+upper
    penalty=constraint_penalty(total)
    ratio=torch.where(total>0,penalty/total.clamp_min(torch.finfo(total.dtype).tiny),1)
    return lower*ratio,upper*ratio


class ContactRegions(nn.Module):
    schema = 'canonical_contact_regions_4_4_2_2_3_3_v1'
    feature_count = 32
    names = (('rear_negative_y', 'rear_positive_y', 'front_negative_y', 'front_positive_y'),)*2 + (
        ('proximal', 'distal'),)*2 + (('lower_shin', 'middle_shin', 'knee_end'),)*2

    def __init__(self, fk):
        super().__init__()
        self.material_skin = None
        source = PartCollisionGeometry(fk)
        # Fixed, actual surface samples provide compact regional landmarks.
        # Spheres use 26 analytical surface directions, not an enclosing box.
        directions = torch.tensor([x for x in itertools.product((-1., 0., 1.), repeat=3) if any(x)])
        directions = directions/directions.norm(dim=-1, keepdim=True)
        lows, highs, clouds = [], [], []
        for part in range(6):
            v, c, r = (getattr(source, f'{name}_{part}') for name in ('vertices', 'centers', 'radii'))
            sphere = (c[:, None]+r[:, None, None]*directions.to(c)).reshape(-1, 3)
            cloud = torch.cat((v, sphere))
            clouds.append(cloud)
            lows.append(cloud.amin(0)); highs.append(cloud.amax(0))
        self.register_buffer('low', torch.stack(lows), persistent=False)
        self.register_buffer('high', torch.stack(highs), persistent=False)
        self.register_buffer('valid', torch.arange(4)[None] < torch.tensor([4, 4, 2, 2, 3, 3])[:, None], persistent=False)
        landmarks, extents = [], []
        self.asset_counts = source.asset_counts
        for part, cloud in enumerate(clouds):
            ids = self.classify_local(torch.full((len(cloud),), part, dtype=torch.long), cloud)
            points, sizes = [], []
            for region in range(4):
                if not self.valid[part, region]:
                    points.append(cloud.new_zeros(3)); sizes.append(cloud.new_zeros(3)); continue
                section = cloud[ids == region]
                if not len(section): raise ValueError(f'Empty canonical region {part}/{region}')
                lo, hi = section.amin(0), section.amax(0)
                desired = (lo+hi)/2
                # Sole / hand underside; anterior shin for kneeling contacts.
                desired[2 if part < 4 else 0] = lo[2] if part < 4 else hi[0]
                point = section[(section-desired).square().sum(-1).argmin()]
                points.append(point); sizes.append(hi-lo)
                self.register_buffer(f'cloud_{part}_{region}', section, persistent=False)
            landmarks.append(torch.stack(points)); extents.append(torch.stack(sizes))
        self.register_buffer('landmark', torch.stack(landmarks), persistent=False)
        self.register_buffer('extent', torch.stack(extents), persistent=False)
        subsets = ((torch.arange(1, 16)[:, None] >> torch.arange(4)) & 1).bool()
        self.register_buffer('subsets', subsets, persistent=False)
        self.register_buffer('valid_subsets', ~(subsets[None] & ~self.valid[:, None]).any(-1), persistent=False)

    def classify_local(self, part, points):
        middle = (self.low[part]+self.high[part])/2
        foot = 2*(points[..., 0] >= middle[..., 0]).long()+(points[..., 1] >= middle[..., 1]).long()
        hand = (points[..., 0] >= middle[..., 0]).long()
        shin = ((points[..., 2]-self.low[part, 2])/(self.high[part, 2]-self.low[part, 2])*3).floor().long().clamp(0, 2)
        return torch.where(part < 2, foot, torch.where(part < 4, hand, shin))

    def witness_regions(self, fk, q, sample, part, points):
        position, rotation = fk.link_poses(q, BODY_NAMES[1:7])
        local = torch.einsum('nij,nj->ni', rotation[sample, part].transpose(-1, -2), points-position[sample, part])
        return self.classify_local(part, local.detach())

    def actual_mask(self, fk, rows, surface):
        from contact_solver.device_contact_objective import group_any
        p = rows.pair; part = p['part'].clamp(0, 5)
        region = self.witness_regions(fk, rows.q, p['sample'], part, rows.points[:, 1])
        eligible = p['eligible'] & (p['primary_surface'] == surface[p['sample'], part])
        return group_any((p['sample']*6+part)*4+region, eligible, len(rows)*24).reshape(len(rows), 6, 4)

    def audit_pairs(self, fk, q, pairs, surface, planned_regions):
        """Audit a single pose using already primary-face-selected Newton pairs."""
        actual = torch.zeros(6, 4, dtype=torch.bool, device=q.device)
        records = [p for p in pairs if p['surface'] == int(surface[p['part']])]
        if records:
            if any(not p['allocated'] or not p['constraint_active'] for p in records):
                raise ValueError('Region audit requires actual activated+allocated pairs')
            part = torch.tensor([p['part'] for p in records], device=q.device)
            point = q.new_tensor([p['position_w'] for p in records])
            region = self.witness_regions(fk, q, torch.zeros_like(part), part, point)
            actual[part, region] = True
        required = planned_regions.to(device=q.device, dtype=torch.bool)
        missing = required & ~actual
        return {'schema': self.schema, 'actual_regions': actual.tolist(),
            'planned_regions': required.tolist(), 'missing_regions': missing.tolist(),
            'realized': bool(required.any() and not missing.any())}

    def forward(self, position, rotation, heightmap):
        point = position[:, :, None]+torch.einsum('bpij,prj->bpri', rotation, self.landmark)
        x = (point[..., 0]-HEIGHTMAP_FORWARD_MIN_M)/HEIGHTMAP_RESOLUTION_M
        y = (point[..., 1]-HEIGHTMAP_LATERAL_MIN_M)/HEIGHTMAP_RESOLUTION_M
        visible = (x >= 0) & (x <= HEIGHTMAP_ROWS-1) & (y >= 0) & (y <= HEIGHTMAP_COLS-1) & self.valid
        cell = x.round().long().clamp(0, HEIGHTMAP_ROWS-1)*HEIGHTMAP_COLS+y.round().long().clamp(0, HEIGHTMAP_COLS-1)
        gap = point[..., 2]-heightmap.flatten(1).gather(1, cell.flatten(1)).reshape_as(x)
        offset = point-position[:, :, None]
        feature = torch.cat((offset/.1, self.extent[None].expand(len(point), -1, -1, -1)/.1,
            torch.where(visible, gap, 0)[..., None]/.1, visible[..., None].to(point)), -1)
        return (feature*self.valid[None, :, :, None]).flatten(2)

    def missing_region_distance(self, fk, q, surface, scene, margin):
        """Loss-only finite-face approach proxy, never a contact label.

        Uses fixed training Newton contact materials when bound. Unobserved
        regions retain the canonical asset proxy and are reported separately.
        Material geometry remains in the scalar loss after native candidates
        appear, including during deep embedding. It never replaces contact
        truth or complete-solid clearance.
        """
        position, rotation = fk.link_poses(q, BODY_NAMES[1:7])
        top_normal = scene['box_rotation'][:, :, 2]
        top_center = scene['box_center']+top_normal*scene['box_half_extents'][:, 2:3]
        result = q.new_zeros(len(q), 6, 4)
        for part in range(6):
            for region in range(len(self.names[part])):
                cloud = getattr(self, f'cloud_{part}_{region}')
                point = position[:, part, None]+torch.einsum('bij,vj->bvi', rotation[:, part], cloud)
                ground_gap = point[..., 2]-scene['ground_height'][:, None]
                delta = point-top_center[:, None]
                box_local = torch.einsum('bvi,bij->bvj', delta, scene['box_rotation'])
                outside = (box_local[..., :2].abs()-scene['box_half_extents'][:, None, :2]).relu()
                gap = torch.where(surface[:, part, None] == 0, ground_gap, box_local[..., 2])
                outside = torch.where((surface[:, part, None] == 0)[..., None], 0, outside)
                cost = finite_face_band_cost(gap, outside, margin[:, None])
                result[:, part, region] = cost.amin(-1)/(.25*margin).square()
        if self.material_skin is not None:
            result = self.material_skin.region_distance(fk, q, surface, scene, margin, result)
        return result


def unified_region_objective(model, rows, active, surface, scene, planned_regions=None, *, audit_path=None, release_mask=None,
                             surface_interval=None, penetration_penalty='log1p',
                             penetration_scale_m=PENETRATION_LOSS_SCALE_M,
                             target_recovery=None, target_recovery_mask=None):
    """One interval-violation objective across intended regions and all bodies."""
    from contact_solver.device_contact_objective import reduce_groups, group_any
    import math
    if penetration_penalty not in ('log1p', 'huber') or not math.isfinite(penetration_scale_m) or penetration_scale_m <= 0:
        raise ValueError('Invalid penetration penalty or scale')
    geometry = model.region_geometry
    if release_mask is not None:
        if release_mask.shape != active.shape or release_mask.dtype != torch.bool:
            raise ValueError('Release mask must be boolean with the contact intent shape')
        if bool((release_mask & active).any()):
            raise ValueError('Required contact and release intents conflict')
    p = rows.pair; q = rows.q; batch = len(q); sample = p['sample']
    solid = getattr(rows, 'solid', None)
    if solid is None:
        raise ValueError('Unified collision loss requires complete realized solid distances; triangle penetration gradients are unsupported')
    solid_distance = solid.distances
    if not bool(torch.isfinite(solid_distance).all() & torch.isfinite(solid.points).all()):
        raise ValueError('Undefined complete-solid distance/geometry; no substitute gradient')
    scale = .25*rows.observed['configured_margin']
    if bool((scale <= 0).any()): raise ValueError('Unified loss needs actual positive configured margins')
    relevant = (p['full_kind'] >= 0)
    invalid = relevant & (p['dist'] < 0) & ~rows.normal_valid
    invalid_counts = torch.zeros(batch, device=q.device).scatter_add(0, sample, invalid.float())
    if bool(invalid.any()):
        if audit_path is None:
            raise ValueError('Invalid native penetrating normal requires explicit audit and rollout rejection')
        import json
        ids=(invalid_counts > 0).nonzero().flatten()
        with open(audit_path, 'a') as stream:
            stream.write(json.dumps({'schema':'newton_unknown_region_gradient_v2',
                'action':'invalid_native_distance_rows_excluded; complete_solid_and_other_gradients_retained; rollout_rejected',
                'q':q[ids].detach().cpu().tolist(), 'sample_indices':ids.cpu().tolist(),
                'link_names':list(rows.observed['link_names']),
                'world_frame':None if rows.world_frame is None else [x[ids].detach().cpu().tolist() for x in rows.world_frame],
                'pairs':{k:v[invalid].detach().cpu().tolist() for k,v in p.items()},
                'transformed_normal':rows.transformed_normal[invalid].detach().cpu().tolist()})+'\n')
    part = p['part'].clamp(0, 5)
    region = geometry.witness_regions(model.fk, q, sample, part, rows.points[:, 1])
    region_group = (sample*6+part)*4+region
    match = p['task_pair'] & p['upward'] & rows.normal_valid & (p['primary_surface'] == surface[sample, part])
    desired = (geometry.valid[None] & active[..., None] if planned_regions is None else planned_regions & active[..., None])
    # A coarse intended part needs one region; an explicit plan needs every
    # selected region. Neither forces all other regions to touch or lift.
    # Use the same upper boundary as the existing execution interval loss.
    # Activation/allocation remain native evidence; a zero gap residual alone
    # neither certifies contact nor turns off a frozen-witness gradient.
    upper = ((rows.distances-p['includemargin']).relu()/scale[sample]).square()
    near = reduce_groups(upper, region_group, match, batch*24, minimum=True).reshape(batch, 6, 4)
    exists = group_any(region_group, match, batch*24).reshape(batch, 6, 4)
    field_lower=q.new_zeros(batch,6,4)
    bounds=None
    if surface_interval is None:
        approach = geometry.missing_region_distance(model.fk, q, surface, scene, rows.observed['configured_margin'])
    else:
        # Both sides of an interval come from one finite-face geometry query.
        # Complete terrain and self clearances below remain independent;
        # only an identical selected affine material query can be deduplicated.
        bounds=surface_interval if isinstance(surface_interval,SurfaceIntervalBounds) else None
        value=surface_interval if bounds is None else bounds.cost
        if value.shape != (batch,6,4):
            raise ValueError('Shape/surface interval must retain anatomical region identity')
        needed = geometry.valid[None] & active[...,None]
        parts=(value,) if bounds is None else (bounds.lower,bounds.upper)
        if any(v.shape!=value.shape or not bool(torch.isfinite(v[needed]).all() & (v[needed]>=0).all()) for v in parts):
            raise ValueError('Undefined realized shape/surface interval; no geometric fallback')
        if bounds is not None and bounds.scalar is not None:
            if bounds.scalar.shape!=value.shape or not bool(torch.isfinite(bounds.scalar[needed]).all() & (bounds.scalar[needed]>=0).all()):
                raise ValueError('Undefined coherent interval scalar')
            if not bool(torch.isclose(bounds.scalar[needed],(bounds.lower+bounds.upper)[needed],
                rtol=32*torch.finfo(value.dtype).eps,atol=64*torch.finfo(value.dtype).tiny).all()):
                raise ValueError('Interval scalar must equal its coherent lower/upper geometry')
        approach = torch.where(needed,value,0)/scale[:,None,None].square()
        if bounds is not None:
            field_lower=torch.where(needed,bounds.lower,0)/scale[:,None,None].square()
    # A triangle candidate does not make an embedded material leave the
    # requested face interval. Retain the field on both sides of the band;
    # native upper bounds can add evidence but cannot switch it off.
    native_upper = torch.where(exists, near, 0)
    material_query_selected = approach > native_upper
    near = torch.maximum(native_upper, approach)
    if (target_recovery is None) != (target_recovery_mask is None):
        raise ValueError('Target recovery requires both regional distances and actual missing-contact mask')
    if target_recovery is not None:
        if (target_recovery.shape != (batch,6,4) or target_recovery_mask.shape != active.shape
                or target_recovery_mask.dtype != torch.bool
                or bool((target_recovery_mask & ~active).any())):
            raise ValueError('Invalid regional target recovery')
        if bool((target_recovery_mask & geometry.actual_mask(model.fk, rows, surface).any(-1)).any()):
            raise ValueError('Target recovery cannot replace an actual same-part/surface contact')
        recovery_candidates = desired & target_recovery_mask[...,None]
        if bool((target_recovery_mask & ~recovery_candidates.any(-1)).any()):
            raise ValueError('Target recovery requires a nonempty intended anatomical region')
        if not bool((torch.isfinite(target_recovery[recovery_candidates]) & (target_recovery[recovery_candidates]>=0)).all()):
            raise ValueError('Undefined target recovery geometry; no fallback')
        recovery_region = target_recovery.masked_fill(~recovery_candidates, torch.inf).argmin(-1)
        recovered = torch.nn.functional.one_hot(recovery_region,4).bool() & target_recovery_mask[...,None]
        # Replace the generic native/material attraction, never add a second
        # approach residual. Recovery is one full XYZ shape-to-target segment
        # query for the closest intended region of each missing relation.
        # A single representative point must not force heel and toe onto the
        # same location. Other intended regions retain their surface interval.
        # Complete solids below still own physical penetration; different
        # queries cannot be deduplicated.
        near = torch.where(recovered, target_recovery/scale[:,None,None].square(), near)
        field_lower = torch.where(recovered, 0., field_lower)
        material_query_selected &= ~recovered
    if planned_regions is None:
        best = near.masked_fill(~geometry.valid, torch.inf).argmin(-1)
        if target_recovery is not None:
            best = torch.where(target_recovery_mask, recovery_region, best)
        desired = torch.nn.functional.one_hot(best, 4).bool() & active[..., None]
    near = torch.where(desired, near, 0)
    field_lower=torch.where(desired,field_lower,0)
    # Contact regions share groups across all attached shapes. Other body
    # links retain independent groups; every raw terrain/self pair is audited.
    link_names = rows.observed['link_names']; groups_per_sample = 24+len(link_names)
    mapping = {name:i for i,key in enumerate(PART_NAMES) for name in CONTACT_BODY_NAMES_BY_PART[key]}
    link_part = torch.tensor([mapping.get(n, -1) for n in link_names], device=q.device)
    costs, groups, valid, target_valid, physical = [], [], [], [], []
    target_mask=None if bounds is None else bounds.solid_target_mask
    if target_mask is not None:
        if target_mask.shape!=(len(solid.sample),2) or target_mask.dtype!=torch.bool:
            raise ValueError('Interval/solid identity binding must match pair sides')
        if bool((target_mask & (solid.pair['full_kind']==1)[:,None]).any()):
            raise ValueError('Self collision cannot be deduplicated as a target surface interval')
    for side in (0, 1):
        link = p[f'body_link{side}']; lp = link_part[link.clamp_min(0)]
        rp = geometry.witness_regions(model.fk, q, sample, lp.clamp_min(0), rows.points[:, side])
        slot = torch.where(lp >= 0, lp*4+rp, 24+link.clamp_min(0))
        # Existing task acceptance permits only shallow extra activations;
        # derive the unwanted-part lower bound from each real pair margin.
        released = ~active if release_mask is None else release_mask
        unwanted = (side == 1) & p['task_pair'] & p['upward'] & released[sample, part]
        from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE
        # Legacy complete plans tolerate shallow extra activations. An explicit
        # event release instead targets the actual activation boundary.
        fraction = 1-DEFAULT_ACCEPTANCE.extra_activation_margin_fraction if release_mask is None else 1.
        lower = torch.where(unwanted, fraction*p['includemargin'], 0)
        # Native rows continue to supervise explicit release/extra activation.
        # Penetration itself uses complete solids below, never a two-sided
        # terrain triangle's local exit branch.
        costs.append(torch.where(unwanted, ((lower-rows.distances).relu()/scale[sample]).square(), 0))
        groups.append(sample*groups_per_sample+slot)
        valid.append(relevant & (link >= 0) & unwanted & rows.normal_valid)
        target_valid.append(torch.zeros_like(valid[-1]))
        physical.append(torch.zeros_like(valid[-1]))
    sp = solid.pair; ss = sp['sample']
    for side in (0, 1):
        link = sp[f'body_link{side}']; lp = link_part[link.clamp_min(0)]
        region = geometry.witness_regions(model.fk, q, ss, lp.clamp_min(0), solid.points[:, side])
        slot = torch.where(lp >= 0, lp*4+region, 24+link.clamp_min(0))
        costs.append(((-solid_distance).relu()/scale[ss]).square())
        groups.append(ss*groups_per_sample+slot)
        valid.append(link >= 0)
        belongs=torch.zeros_like(link,dtype=torch.bool)
        if target_mask is not None:
            if bool((target_mask[:,side] & ((link<0)|(lp<0))).any()):
                raise ValueError('Target interval binding refers to a noncontact or static body')
            from contact_solver.contact_surface_interval import same_interval_query
            belongs=(target_mask[:,side] & desired[ss,lp.clamp_min(0),region]
                & material_query_selected[ss,lp.clamp_min(0),region]
                & same_interval_query(solid,bounds.query,link_names,lp,region,side,bounds=bounds))
        target_valid.append(belongs)
        physical.append(torch.ones_like(valid[-1]))
    flat_cost,flat_group,flat_valid=torch.cat(costs),torch.cat(groups),torch.cat(valid)
    lower_cost = reduce_groups(flat_cost,flat_group,flat_valid,batch*groups_per_sample).reshape(batch,groups_per_sample)
    # Entity/region matches are only candidates for duplicate evidence. Merge
    # only the SAME affine material query selected by the coherent interval.
    # Different physical features and unselected evidence remain independent.
    field_lower=torch.cat((field_lower.flatten(1),q.new_zeros(batch,len(link_names))),-1)
    independent_lower=None
    if bounds is not None:
        matched=torch.cat(target_valid)
        target_lower=reduce_groups(flat_cost,flat_group,flat_valid & matched,batch*groups_per_sample).reshape(batch,groups_per_sample)
        independent_lower=reduce_groups(flat_cost,flat_group,flat_valid & ~matched,batch*groups_per_sample).reshape(batch,groups_per_sample)
        # Every qualified lower witness is already covered by that interval.
        # Do not fuse a physical feature with another material's upper bound.
    upper_cost = torch.cat((near.flatten(1), q.new_zeros(batch, len(link_names))), -1)
    # Robustify each complete query once. A native candidate and a material
    # query are alternatives here; taking bounds from different queries
    # would invent a new residual and alter their derivatives.
    if penetration_penalty == 'huber':
        # Keep release/extra-contact penalties unchanged. Penetration uses one
        # metric scale, independent of solver activation margins. Only exact
        # shared material queries may remove a duplicate solid lower bound.
        physical = torch.cat(physical)
        matched = torch.cat(target_valid)
        release_cost = reduce_groups(flat_cost, flat_group, flat_valid & ~physical,
            batch*groups_per_sample).reshape(batch, groups_per_sample)
        solid_cost = reduce_groups(flat_cost, flat_group, flat_valid & physical & ~matched,
            batch*groups_per_sample).reshape(batch, groups_per_sample)
        factor = (scale/penetration_scale_m).square()[:, None]
        lower_penalty = constraint_penalty(release_cost) + huber_squared(solid_cost*factor)
        # Split the two bounds of the SAME selected material query. Do not
        # penalize the solid witness again when its lower bound lives here.
        selected_lower = torch.cat((torch.where(material_query_selected,
            field_lower[:, :24].reshape(batch, 6, 4), 0).flatten(1),
            q.new_zeros(batch, len(link_names))), -1)
        upper_penalty = (constraint_penalty((upper_cost-selected_lower).clamp_min(0))
                         + huber_squared(selected_lower*factor))
    elif bounds is not None:
        # Only the actually bound target entity shares this interval. Self,
        # other static components and release constraints remain independent;
        # their derivative cannot disappear behind a large recovery bound.
        lower_penalty=constraint_penalty(independent_lower)
        upper_penalty=constraint_penalty(upper_cost)
    else:
        lower_penalty, upper_penalty = constraint_penalty(lower_cost), constraint_penalty(upper_cost)
    # Different witnesses in a region can violate opposite bounds. Both
    # remain in the scalar; neither bound hides the other behind max().
    error = lower_penalty+upper_penalty
    # Fixed anatomical normalization: adding a new small collision must not
    # lower the average by increasing a count of currently violating groups.
    group_count = geometry.valid.sum()+(link_part < 0).sum()
    loss = error.sum(-1)/group_count.clamp_min(1)+error.amax(-1)
    # Exact additive diagnostics, including the equal-share subgradient used
    # by amax at ties. These terms sum to loss; they are not extra penalties.
    dominant = (error == error.amax(-1, keepdim=True)).to(error)
    dominant = (dominant/dominant.sum(-1, keepdim=True)).detach()
    attraction = upper_penalty.sum(-1)/group_count.clamp_min(1)+(upper_penalty*dominant).sum(-1)
    separation = lower_penalty.sum(-1)/group_count.clamp_min(1)+(lower_penalty*dominant).sum(-1)
    # Complete-solid distances have independent, verified material derivatives.
    # An unusable native triangle normal cannot erase those or other FK losses.
    # Native unknowns remain audited and reject rollout; they are not repaired
    # by relabeling a geometric approach residual as an actual contact.
    actual = geometry.actual_mask(model.fk, rows, surface)
    realized = ((actual | ~desired).all((-1, -2)) & active.any(-1))
    solid_depth = solid.depths()
    skin = getattr(geometry, 'material_skin', None)
    material_missing = (desired & ~skin.known[None]).sum((-1, -2)).float() if skin is not None else q.new_zeros(batch)
    return loss, dict(unified_region_loss=loss, region_plan_realized=realized.float(),
        region_approach_material_unknown=material_missing,
        solid_penetration_cm=100*solid_depth.amax(-1),
        solid_terrain_penetration_cm=100*solid_depth[:, 0], solid_self_penetration_cm=100*solid_depth[:, 1],
        unified_attraction_component=attraction,
        unified_separation_component=separation,
        unified_contact_interval_component=attraction,
        unified_physical_clearance_component=separation,
        region_missing_count=(desired & ~actual).sum((-1, -2)).float(),
        region_lower_bound_violating_groups=(lower_cost > 0).sum(-1).float(),
        region_gradient_valid=q.new_ones(batch), region_invalid_penetrating_normals=invalid_counts,
        region_invalid_nonpenetrating_normals=torch.zeros(batch, device=q.device).scatter_add(0, sample,
            (relevant & (p['dist'] >= 0) & ~rows.normal_valid).float()))
