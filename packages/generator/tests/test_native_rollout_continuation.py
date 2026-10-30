import pytest
import torch

from generator.parallel_rollout_pool import ParallelRolloutPool
from generator.rollout_termination import native_continuation_masks
from contact_solver.endpoint_penetration import solid_generation_mask
from contact_solver.interaction_acceptance import configured_acceptance


def evidence():
    q = torch.zeros(3, 36)
    q[:, 3] = 1
    contact = torch.zeros(3, 6, dtype=torch.bool)
    contact[:, 0] = True
    metrics = dict(newton_penetration_cm=torch.tensor([0., 3.1, 0.]),
        solid_penetration_cm=torch.tensor([0., 3.1, 0.]),
        newton_contact_accepted=torch.ones(3),
        newton_invalid_fullbody_witnesses=torch.zeros(3),
        joint_violation_rad=torch.zeros(3),
        conservative_geometry_depth_cm=torch.full((3,), 5.1))
    pair = dict(type=torch.ones(3, dtype=torch.long), dist=torch.zeros(3),
        includemargin=torch.full((3,), .02), active=torch.ones(3, dtype=torch.bool),
        constraint_allocated=torch.tensor([True, True, False]), sample=torch.arange(3))
    return q, contact, metrics, pair


def test_native_pose_commits_despite_negative_proxy_but_real_failures_roll_back():
    q, contact, metrics, pair = evidence()
    continuable, reasons = native_continuation_masks(q, metrics, contact, pair=pair)
    assert continuable.tolist() == [True, True, False]
    assert reasons['severe_penetration'].tolist() == [False, True, False]
    assert reasons['invalid'].tolist() == [False, False, True]
    reference = dict(current_q=q.clone(), current_contact=contact.clone())
    pool = ParallelRolloutPool(reference, torch.arange(3), torch.full((3,), -1), 3)
    pool.indices[:] = torch.arange(3)
    pool.observation = {k:v.clone() for k,v in reference.items()}
    predicted = q.clone()
    predicted[:, 0] = 1

    def observe(candidate, indices, observed, chosen):
        return dict(current_q=candidate, current_contact=observed['contact'][chosen])

    accepted = solid_generation_mask(metrics, limits=configured_acceptance({'acceptance_penetration_m': .005}))
    result = pool.update(torch.arange(3), predicted, dict(contact=contact),
        accepted, continuable, observe, reset_reasons=reasons)
    assert pool.observation['current_q'][:, 0].tolist() == [1., 0., 0.]
    assert result['accepted_transitions'] == 1
    assert result['severe_penetration_rejections'] == 1
    assert result['invalid_rejections'] == 1


def test_missing_solid_depth_cannot_fall_back_to_native_or_proxy():
    q, contact, metrics, pair = evidence()
    metrics.pop('solid_penetration_cm')
    with pytest.raises(ValueError, match='complete-solid'):
        native_continuation_masks(q, metrics, contact, pair=pair)


def test_complete_containment_fails_single_acceptance_gate_even_with_small_triangle_depth():
    q, contact, metrics, pair = evidence()
    metrics['solid_penetration_cm'] = torch.tensor([5.4, 0., 0.])
    continuable, reasons = native_continuation_masks(q, metrics, contact, pair=pair)
    assert continuable.tolist() == [True, True, False]
    assert reasons['severe_penetration'].tolist() == [True, False, False]
    accepted = solid_generation_mask(metrics, limits=configured_acceptance({'acceptance_penetration_m': .005}))
    assert (accepted & continuable).tolist() == [False, True, False]


def test_inconsistent_activation_and_missing_allocation_are_explicit_errors():
    q, contact, metrics, pair = evidence()
    pair['active'][0] = False
    with pytest.raises(ValueError, match='activation metadata mismatch'):
        native_continuation_masks(q, metrics, contact, pair=pair)
    pair.pop('constraint_allocated')
    with pytest.raises(ValueError, match='activation/allocation fields'):
        native_continuation_masks(q, metrics, contact, pair=pair)
