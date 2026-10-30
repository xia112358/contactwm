"""Refreshed Newton witness loss and current-material persistence.

Absent target candidates receive demonstration-anchor approach supervision,
not a fabricated contact. Network/input contract remains unchanged.
"""
import torch
from contact_solver.surface_geometry import nearest_face
from generator.next_interaction import objective as reference_objective
from generator.next_interaction_joint import prepare_supervision as prepare_margins
from somaforge_core.g1_kinematics import _matrix_from_rotation6d
from somaforge_core.contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA
from contact_solver.surface_contact_proxy import primary_face_cost

SCHEMA = 'surface_region_newton_contact_topology_fullbody_v9_structural_pose'


def prepare_supervision(manifest, inputs, target, samples, model):
    report = prepare_margins(manifest, inputs, target, samples)
    from generator.newton_query_supervision import prepare_query_metadata
    report.update(prepare_query_metadata(manifest, inputs, target, samples))
    with torch.no_grad():
        pos, rot = model.fk(inputs['current_q'][:, None])
        rotation = _matrix_from_rotation6d(rot[:, 0])[:, 1:7]
        target['start_material_local'] = torch.einsum('bpji,bpj->bpi', rotation,
            inputs['current_anchor'] - pos[:, 0, 1:7])
    target['start_anchor'] = inputs['current_anchor']
    persistent = target['role'] == 2
    if (persistent & (~inputs['current_contact'] | (inputs['current_surface'] != target['surface']))).any():
        raise ValueError('Persistent label lacks same-surface current contact')
    report.update(objective_schema=SCHEMA, fixed_touchdown_anchor=False,
                  persistence_reference='same observed part and surface; no fixed-material constraint',
                  start_material_motion_diagnostic_only=True, geometric_proxy_not_truth=True,
                  contact_proxy='refreshed_actual_newton_witness',
                  missing_pair_supervision='demonstrated material anchor, not realized contact')
    return report


def region_cost(points, parts, faces, surface, active, margin, known):
    per_cost,per_part,per_excess,_=primary_face_cost(points,parts,faces,margin)
    index = surface.clamp_min(0)[..., None]
    selected = per_part.gather(2, index)[..., 0]
    available = known.gather(2, index)[..., 0]
    if (active & ~available).any():
        raise ValueError('Supervised contact has unknown Newton margin')
    excess = per_excess.gather(2,index)[...,0] * active
    normalized = per_cost.gather(2,index)[...,0] * active / .02**2
    cost = normalized.sum(-1) / active.sum(-1).clamp_min(1) + .25 * normalized.amax(-1)
    return cost, selected, excess


def contact_terms(model, prediction, target, scene, *, invalid_witness_policy='error',
                  witness_audit_path=None, query_override=None,
                  predicted_intent_mode='agreement', intent_consistency_weight=1.):
    from contact_solver.newton_witness_loss import query_local_distances, selected_contact_cost, unwanted_contact_cost, full_body_violation
    from somaforge_core.contact_face_selection import upward_face_mask, select_contact_pairs
    q = prediction.qpos
    active, persistent = target['role'] != 0, target['role'] == 2
    world_frame = None
    if 'newton_world_origin' in target:
        world_frame = (target['newton_world_origin'], target['newton_world_basis'])
    if query_override is None:
        rows, observed = query_local_distances(model.fk, q, scene, world_frame=world_frame,
                                       fingerprints=target.get('newton_model_fingerprint'))
    else:
        rows, observed = query_override
        if len(rows) != len(q) or len(observed.get('pairs', ())) != len(q):
            raise ValueError('Closed-loop Newton query batch mismatch')
    if observed.get('surface_attribution_schema') != SOURCE_NORMAL_FAN_SCHEMA:
        raise ValueError('Predictor loss requires source-aware Newton workers (--capture-contact-sources); '
                         'legacy nearest-face labels cannot silently substitute')
    if 'full_robot_separation' not in observed:raise ValueError('Full-body Newton separation is required')
    full_depth,invalid_witnesses=full_body_violation(model.fk,q,observed['full_robot_separation'],world_frame=world_frame,
        invalid_policy=invalid_witness_policy,return_validity=True,audit_path=witness_audit_path)
    pair_cost, missing, realized = selected_contact_cost(q, rows, active, target['surface'],
                                                       activation_buffer_fraction=.25)
    pos, rot = model.fk(q)
    rotation = _matrix_from_rotation6d(rot)[:, 1:7]
    # A demonstrated material point is supervision only. It provides an
    # approach direction when Newton has no target-face candidate; it never
    # changes the observed contact mask or enters model.forward.
    demonstrated_material = pos[:, 1:7] + torch.einsum('bpij,bpj->bpi', rotation, target['material_local'])
    approach = (demonstrated_material-target['anchor']).square().sum(-1) * missing
    normalized = (pair_cost+approach)/.02**2
    cost = normalized.sum(-1)/active.sum(-1).clamp_min(1)+.25*normalized.amax(-1)
    material = pos[:, 1:7] + torch.einsum('bpij,bpj->bpi', rotation, target['start_material_local'])
    error = (material - target['start_anchor']).norm(dim=-1) * persistent
    # A persistent topology label does not certify no-slip of this particular
    # material point (e.g. the foot may roll). Endpoint contact is already
    # supervised by `cost` for every active part, persistent ones included.
    # Keep observed material motion as a diagnostic, never a training penalty.
    persistent_normalized=normalized*persistent
    persistent_cost=persistent_normalized.sum(-1)/persistent.sum(-1).clamp_min(1)+.25*persistent_normalized.amax(-1)
    intended = prediction.contact.detach()
    intent_surface=prediction.surface.detach()
    intent_cost, intent_missing, intent_realized = selected_contact_cost(q, rows, intended, intent_surface,
                                                                       activation_buffer_fraction=.25)
    # Never reinforce a mistaken categorical prediction against its teacher.
    # Missing candidates receive approach supervision once, in `cost` above.
    # Consistency uses actual Newton candidates only; do not duplicate the
    # demonstrated-anchor force for already-agreeing categorical intentions.
    if predicted_intent_mode == 'agreement':
        supervised_intent=active & intended & (intent_surface==target['surface'])
    elif predicted_intent_mode == 'all':
        supervised_intent=intended
    else:
        raise ValueError(f'Unknown predicted intent mode: {predicted_intent_mode}')
    intent_normalized = (intent_cost*supervised_intent)/.02**2
    consistency = intent_normalized.sum(-1)/intended.sum(-1).clamp_min(1)+intent_normalized.amax(-1)
    catalogs=observed.get('surface_catalog_by_sample')
    if catalogs is None:catalogs=[observed['surface_catalog']]*len(q)
    eligible=[{int(f['surface']) for f,keep in zip(cat,upward_face_mask([f['normal_w'] for f in cat])) if keep}
              for cat in catalogs]
    unwanted,unwanted_active=unwanted_contact_cost(q,rows,active,eligible)
    separation=(unwanted/.02**2).mean(-1)+(unwanted/.02**2).amax(-1)
    exact=[]
    for i,cat in enumerate(catalogs):
        selected=select_contact_pairs([observed['pairs'][i]],cat)
        actual=q.new_tensor(selected['contact_part_mask'][0],dtype=torch.bool)
        surface=q.new_tensor(selected['contact_surface'][0],dtype=torch.long)
        exact.append((actual==active[i]).all() & (surface[active[i]]==target['surface'][i,active[i]]).all()
                     & (prediction.contact[i]==active[i]).all()
                     & (prediction.surface[i,active[i]]==target['surface'][i,active[i]]).all())
    extra = dict(intent_missing_newton_pairs=intent_missing.sum(-1).float(),
                 intent_unrealized_contacts=(intended & ~intent_realized).sum(-1).float(),
                 newton_missing_target_pairs=missing.sum(-1).float(),
                 newton_unrealized_target_contacts=(active & ~realized).sum(-1).float(),
                 demonstrated_approach_loss=approach.sum(-1)/.02**2,
                 unwanted_contact_loss=separation,
                 newton_unwanted_contact_parts=unwanted_active.sum(-1).float())
    extra.update(newton_next_topology_exact=torch.stack(exact).float(),
                 newton_invalid_fullbody_witnesses=invalid_witnesses,
                 newton_fullbody_geometry_valid=(invalid_witnesses==0).float(),
                 newton_fullbody_penetration_cm=100*full_depth,
                 _full_depth=full_depth)
    return cost + separation, intent_consistency_weight*consistency, dict(
        surface_margin_excess_cm=100*pair_cost.amax(-1).sqrt(),
        start_material_motion_cm=100*error.sum(-1)/persistent.sum(-1).clamp_min(1),
        start_material_motion_max_cm=100*error.amax(-1), surface_contact_loss=cost,
        persistent_contact_loss=persistent_cost, consistency_loss=consistency, **extra)


def objective(model, prediction, target, scene, *, invalid_witness_policy='error',
              witness_audit_path=None, query_override=None,
              predicted_intent_mode='agreement', intent_consistency_weight=1.):
    q=prediction.qpos
    model.fk.begin_link_pose_cache(q)
    try:
        contact, consistency, extra = contact_terms(model, prediction, target, scene,
            invalid_witness_policy=invalid_witness_policy,witness_audit_path=witness_audit_path,
            query_override=query_override,predicted_intent_mode=predicted_intent_mode,
            intent_consistency_weight=intent_consistency_weight)
        full_depth=extra.pop('_full_depth')
        loss, metrics = reference_objective(model, prediction, target, scene, contact_override=contact,
                                            additional_penetration=full_depth)
    finally:
        model.fk.end_link_pose_cache()
    loss = loss + consistency
    metrics['demonstration_anchor_mean_cm'] = metrics.pop('contact_cm')
    metrics['demonstration_anchor_max_cm'] = metrics.pop('contact_max_cm')
    metrics.update(extra)
    metrics['valid_next_contact']=metrics['newton_next_topology_exact']*(metrics['penetration_cm']==0).float()*metrics['newton_fullbody_geometry_valid']
    metrics['loss'] = loss
    return loss, metrics
