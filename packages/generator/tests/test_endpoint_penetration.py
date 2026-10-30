import numpy as np
import pytest
import torch

from contact_solver.endpoint_penetration import (
    huber_squared, penetration_checks, solid_generation_mask, solid_separation_record)
from contact_solver.interaction_acceptance import configured_acceptance
from generator.full1000_training_variants import (endpoint_feedback_checks, endpoint_rejection_masks,
    endpoint_validation_score)
from generator.rollout_termination import TerminationLimits, geometry_failure, native_continuation_masks


LIMITS = configured_acceptance({'acceptance_penetration_m': .005})


def evidence(depths):
    n = len(depths)
    return dict(solid_penetration_cm=torch.tensor(depths, dtype=torch.float64)*100,
        newton_penetration_cm=torch.full((n,), 50.),
        newton_contact_accepted=torch.ones(n), newton_invalid_fullbody_witnesses=torch.zeros(n),
        joint_violation_rad=torch.zeros(n), own_endpoint_layout_accepted=torch.ones(n),
        persistent_role_accepted=torch.ones(n), task_endpoint_layout_accepted=torch.ones(n))


@pytest.mark.parametrize('tolerance', [.005, .03])
def test_one_solid_depth_boundary_for_training_feedback_and_scalar_evaluation(tolerance):
    limits = configured_acceptance({'acceptance_penetration_m': tolerance})
    depths = [0., .98*tolerance, tolerance, 1.02*tolerance, 4*tolerance, 6.2*tolerance]
    metrics = evidence(depths)
    # Native triangle depths may over- or under-report; neither overrides solids.
    metrics['newton_penetration_cm'][3:] = 0
    expected = [True, True, True, False, False, False]
    assert solid_generation_mask(metrics, limits=limits).tolist() == expected
    checks = endpoint_feedback_checks(metrics, torch.ones(6, dtype=torch.bool),
        torch.ones(6, dtype=torch.bool), limits=limits)
    assert checks['accepted'].tolist() == expected
    assert checks['penetration_only'].tolist() == [not x for x in expected]
    for depth, passed in zip(depths, expected):
        pair = dict(full_kind=np.array([0]), dist=np.array([-depth]), normal_w=np.array([[0., 0., 1.]]))
        native = dict(terrain_penetration_m=.5, self_penetration_m=.5,
            worst_terrain={'normal_w': [0., 0., 0.]}, closest_self={'normal_w': [0., 0., 0.]})
        result = solid_separation_record(native, [depth, 0.], pair)
        assert (geometry_failure(result, TerminationLimits(penetration_m=tolerance)) is None) == passed
        assert result['self_penetration_m'] == 0


def test_missing_or_invalid_solid_evidence_is_never_zero_or_native_fallback():
    metrics = evidence([float('nan'), float('inf'), -.001])
    checks = penetration_checks(metrics, limits=LIMITS)
    assert checks['penetration_unknown'].all()
    assert not solid_generation_mask(metrics, limits=LIMITS).any()
    metrics.pop('solid_penetration_cm')
    with pytest.raises(ValueError, match='complete-solid'):
        solid_generation_mask(metrics, limits=LIMITS)


def test_overlapping_rejection_reasons_do_not_masquerade_as_penetration_only():
    metrics = evidence([.006]*5)
    metrics['newton_contact_accepted'][1] = 0
    metrics['own_endpoint_layout_accepted'][2] = 0
    metrics['newton_invalid_fullbody_witnesses'][3] = 1
    checks = endpoint_feedback_checks(metrics, torch.tensor([True]*4+[False]),
        torch.ones(5, dtype=torch.bool), limits=LIMITS)
    assert checks['penetration_exceeded'].all()
    assert checks['penetration_only'].tolist() == [True, False, False, False, False]
    assert checks['penetration_and_contact'].tolist() == [False, True, False, False, False]


def test_huber_has_finite_zero_derivative_and_nondecaying_deep_penetration_signal():
    depth = torch.tensor([0., .001, .005, .03, .1], dtype=torch.float64, requires_grad=True)
    loss = huber_squared((depth/.005).square())
    gradient, = torch.autograd.grad(loss.sum(), depth)
    torch.testing.assert_close(gradient, depth.new_tensor([0., 40., 200., 200., 200.]))
    assert torch.autograd.gradcheck(lambda p: huber_squared((p/.005).square()), (depth,))


def test_penetration_only_excludes_invalid_allocation_and_progress_failures():
    metrics = evidence([.04]*3)
    checks = endpoint_feedback_checks(metrics, torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool), limits=LIMITS)
    masks = endpoint_rejection_masks(checks, dict(severe_penetration=torch.ones(3, dtype=torch.bool),
        invalid=torch.tensor([False, True, False]), no_forward_progress=torch.tensor([False, False, True])))
    assert masks['penetration_only'].tolist() == [True, False, False]
    assert masks['penetration_exceeded'].all()


def test_checkpoint_ranking_uses_solid_acceptance_not_native_zero_depth():
    metrics = dict(plan_exact=1., newton_generation_accepted=.9, newton_generation_valid=0.,
        newton_unrealized_intended_contacts=.1, root_cm=1., body_cm=1., joint_rmse_rad=.1,
        solid_penetration_cm=.2, penetration_cm=9., maxima=dict(solid_penetration_cm=.4, penetration_cm=99.))
    score = endpoint_validation_score(metrics)
    metrics.update(newton_generation_valid=1., penetration_cm=0.)
    metrics['maxima']['penetration_cm'] = 0.
    assert endpoint_validation_score(metrics) == score
    metrics['newton_generation_accepted'] = .8
    assert endpoint_validation_score(metrics) > score


def test_severity_classification_cannot_create_a_second_acceptance_threshold():
    metrics = evidence([.004, .006])
    q = torch.zeros(2, 36); q[:, 3] = 1
    contact = torch.ones(2, 6, dtype=torch.bool)
    for threshold in (.001, .03, .1):
        continuable, reasons = native_continuation_masks(q, metrics, contact,
            limits=TerminationLimits(penetration_m=threshold))
        assert continuable.all()
        assert (continuable & solid_generation_mask(metrics, limits=LIMITS)).tolist() == [True, False]
