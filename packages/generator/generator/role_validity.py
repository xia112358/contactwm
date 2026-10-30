"""State-dependent action support from actual Newton primary-face contacts.

Surface association describes raster intentions; only the supplied verified
Newton pairs establish observed contact. No proximity test defines contact.
"""
import torch

from somaforge_core.contact_face_selection import upward_face_mask
from somaforge_core.heightmap import heightmap_grid, _root_yaw_basis

ROLE_VALIDITY_SCHEMA = 'actual_newton_part_surface_role_validity_v1'


def mask_role_logits(logits, current_contact):
    if current_contact.dtype != torch.bool or logits.shape != (*current_contact.shape, 4):
        raise ValueError('Role validity requires logits [B,6,4] and actual bool contact [B,6]')
    allowed = torch.ones_like(logits, dtype=torch.bool)
    allowed[..., 2] = current_contact
    return logits.masked_fill(~allowed, -torch.inf)


def conditional_location_logits(logits, roles, keep_allowed):
    if keep_allowed is None:
        return logits
    if keep_allowed.shape != logits.shape or keep_allowed.dtype != torch.bool:
        raise ValueError('Invalid keep location support')
    allowed = (roles != 2)[..., None] | keep_allowed
    if bool((~allowed.any(-1)).any()):
        raise ValueError('Keep has no observed same-surface location in the current raster')
    return logits.masked_fill(~allowed, -torch.inf)


def keep_location_support(current_contact, relations, surface_ids, heightmap_surface):
    if relations is None or surface_ids is None or heightmap_surface is None:
        raise ValueError('Surface role validity requires actual contact relations and raster surface identity')
    if (relations.dtype != torch.bool or relations.ndim != 3 or relations.shape[:2] != current_contact.shape
            or surface_ids.shape != (len(current_contact), relations.shape[-1])
            or surface_ids.dtype != torch.long or heightmap_surface.dtype != torch.long
            or len(heightmap_surface) != len(current_contact)):
        raise ValueError('Invalid actual contact relation/surface mapping shapes')
    if not torch.equal(relations.any(-1), current_contact):
        raise ValueError('Actual contact mask disagrees with complete part/surface relations')
    if bool((relations & (surface_ids < 0)[:, None]).any()):
        raise ValueError('Observed contact has an unknown surface identity')
    raster = heightmap_surface.flatten(1)
    return ((raster[:, None, :, None] == surface_ids[:, None, None])
            & relations[:, :, None] & (surface_ids >= 0)[:, None, None]).any(-1)


@torch.no_grad()
def contact_validity_observation(current_q, heightmap, observed):
    """Retain every eligible relation, including multiple surfaces per part."""
    batch, device = len(current_q), current_q.device
    if observed.get('schema') == 'newton_device_witness_batch_v1':
        from somaforge_core.newton_contact_tensors import validate_observed
        validate_observed(observed)
        ids = observed['face_surface'].long()
        normals, offsets = observed['face_normal'], observed['face_offset']
        pair = observed['pairs']
        required = ('sample', 'part', 'primary_surface', 'eligible', 'active',
                    'constraint_allocated', 'task_pair')
        if any(k not in pair for k in required):
            raise ValueError('Missing actual Newton contact fields for role validity')
        if bool((pair['active'] & ~pair['constraint_allocated']).any()):
            raise ValueError('Unallocated active Newton contact in role validity')
        good = pair['eligible']
        if bool((good & ~(pair['active'] & pair['constraint_allocated'] & pair['task_pair'])).any()):
            raise ValueError('Invalid eligible Newton contact for role validity')
        part = pair['part'][good].long()
        sample = pair['sample'][good].long()
        surface = pair['primary_surface'][good].long()
        if bool(((part < 0) | (part >= 6) | (sample < 0) | (sample >= batch)).any()):
            raise ValueError('Invalid Newton part/sample mapping')
        match = surface[:, None] == ids[sample]
        if not bool(match.any(-1).all()):
            raise ValueError('Eligible contact missing from actual upward surface catalog')
        groups = (sample * 6 + part)[:, None] * ids.shape[1] + torch.arange(ids.shape[1], device=device)
        counts = torch.zeros(batch * 6 * ids.shape[1], device=device, dtype=torch.long)
        counts.scatter_add_(0, groups.flatten(), match.long().flatten())
        relations = counts.reshape(batch, 6, ids.shape[1]) > 0
    else:
        catalogs = observed.get('surface_catalog_by_sample')
        if catalogs is None:
            catalogs = [observed['surface_catalog']] * batch
        catalogs = [[f for f, valid in zip(c, upward_face_mask([f['normal_w'] for f in c])) if valid]
                    for c in catalogs]
        if len(catalogs) != batch or any(not c for c in catalogs):
            raise ValueError('Missing actual upward surface catalog')
        count = max(map(len, catalogs))
        ids = torch.full((batch, count), -1, device=device, dtype=torch.long)
        normals = current_q.new_zeros(batch, count, 3)
        offsets = current_q.new_zeros(batch, count)
        relations = torch.zeros(batch, 6, count, device=device, dtype=torch.bool)
        pairs = observed['pairs']
        if batch == 1 and (not pairs or isinstance(pairs[0], dict)):
            pairs = [pairs]
        if len(pairs) != batch:
            raise ValueError('Actual Newton pair batch mismatch')
        for b, (catalog, row) in enumerate(zip(catalogs, pairs)):
            ids[b, :len(catalog)] = ids.new_tensor([f['surface'] for f in catalog])
            normals[b, :len(catalog)] = normals.new_tensor([f['normal_w'] for f in catalog])
            offsets[b, :len(catalog)] = offsets.new_tensor([f['plane_offset'] for f in catalog])
            from somaforge_core.contact_face_selection import select_contact_pairs
            selected = select_contact_pairs([row], observed.get('surface_catalog_by_sample',
                [observed.get('surface_catalog')] * batch)[b])
            for p in selected['contact_pairs'][0]:
                if not p['constraint_active'] or not (p['constraint_type'] & 1) or not all(a >= 0 for a in p['efc_address']):
                    raise ValueError('Invalid actual Newton activation/allocation for role validity')
                relations[b, int(p['part'])] |= ids[b] == int(p['surface'])
    if bool(((ids >= 0).sum(-1) == 0).any()):
        raise ValueError('No actual upward surfaces')
    actual_mask = observed.get('contact_part_mask', observed.get('mask'))
    if actual_mask is None or not torch.equal(relations.any(-1), torch.as_tensor(actual_mask, device=device, dtype=torch.bool).reshape(batch, 6)):
        raise ValueError('Actual contact mask disagrees with full eligible Newton relations')
    if heightmap is None:
        return dict(current_contact_relations=relations, contact_surface_ids=ids)
    grid = current_q.new_tensor(heightmap_grid()).reshape(-1, 2)
    points = torch.cat((grid[None].expand(batch, -1, -1), heightmap.flatten(1)[..., None]), -1)
    basis, _ = _root_yaw_basis(current_q)
    world = current_q[:, None, :3] + torch.einsum('bij,bnj->bni', basis, points)
    # Same catalog association used by endpoint intent audits, never contact truth.
    distance = (torch.einsum('bni,bfi->bnf', world, normals) - offsets[:, None]).abs()
    distance = distance.masked_fill((ids < 0)[:, None], torch.inf)
    nearest = distance.argmin(-1)
    minimum = distance.gather(-1, nearest[..., None])
    ambiguous = (distance == minimum).sum(-1) != 1
    raster = ids.gather(1, nearest).masked_fill(ambiguous, -1).reshape_as(heightmap)
    return dict(current_contact_relations=relations, contact_surface_ids=ids,
                heightmap_surface=raster)


def validity_inputs(model, current_q, heightmap, observed, *, scan_provider=None, scene_ids=None):
    if not getattr(model, 'role_validity', False):
        return {}
    mesh = getattr(model, 'world_surface_scan', False)
    result = contact_validity_observation(current_q, None if mesh else heightmap, observed)
    if getattr(model, 'depth_observation', False):
        from contact_solver.endpoint_contact_points import relation_points
        points, present = relation_points(observed, result['contact_surface_ids'], current_q)
        if not torch.equal(present, result['current_contact_relations']):
            raise ValueError('Actual contact point and relation observations disagree')
        result['contact_relation_points'] = points
        from contact_solver.contact_point_sets import contact_patch_observation
        result.update(contact_patch_observation(observed, current_q))
    if mesh:
        if scan_provider is None or scene_ids is None:
            raise ValueError('World-mesh Predictor requires a current scan provider and scene IDs')
        from somaforge_core.world_mesh_scan import append_current_contact_nodes, DEPTH_OBSERVATION_KEYS
        packet = append_current_contact_nodes(scan_provider(current_q, scene_ids, model.fk), observed)
        if not getattr(model, 'depth_observation', False):
            packet = {k: v for k, v in packet.items() if k not in DEPTH_OBSERVATION_KEYS}
        result.update(packet)
    return result


def subset_observed(observed, indices):
    """Subset an existing same-state query without mixing its pair sample IDs."""
    if observed.get('schema') != 'newton_device_witness_batch_v1':
        raise ValueError('Expected actual device Newton observation')
    result = dict(observed)
    for key in ('contact_part_mask', 'contact_surface', 'contact_position_w',
                'face_normal', 'face_offset', 'face_surface', 'configured_margin'):
        if key in observed:
            result[key] = observed[key][indices]
    pair = observed['pairs']
    lookup = indices.new_full((len(observed['contact_part_mask']),), -1)
    lookup[indices] = torch.arange(len(indices), device=indices.device)
    chosen = lookup[pair['sample']] >= 0
    result['pairs'] = {k: v[chosen] for k, v in pair.items()}
    result['pairs']['sample'] = lookup[pair['sample'][chosen]]
    return result


def role_consistency_loss(logits, actual_contact):
    """A keep role requires current actual contact; other roles remain free.

    This is learning from Newton observations, not an inference-time mask or
    a claim that an intended future contact has been realized.
    """
    if logits.shape != (*actual_contact.shape, 4) or actual_contact.dtype != torch.bool:
        raise ValueError('Expected role logits [...,6,4] and actual boolean contact')
    valid = logits.clone()
    valid[..., 2] = valid[..., 2].masked_fill(~actual_contact, -torch.inf)
    return (torch.logsumexp(logits, -1)-torch.logsumexp(valid, -1)).mean(-1)
