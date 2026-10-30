"""Small, explicit training ablations; never redefine Newton contact truth."""
import torch
from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE
from contact_solver.endpoint_penetration import penetration_checks


def endpoint_feedback_checks(metrics, task_contact_accepted, supervised, *, limits=DEFAULT_ACCEPTANCE):
    """Commit own feasible intent; require demonstration targets only while supervised."""
    checks = penetration_checks(metrics, limits=limits)
    other = (metrics['newton_contact_accepted'].bool()
        & (metrics['newton_invalid_fullbody_witnesses'] == 0)
        & (metrics['joint_violation_rad'] <= limits.joint_excess_rad))
    # Endpoint events require actual part/surface contacts. The observed
    # quadrant distribution is an audit of solver witnesses, not a second
    # conjunction of required contacts. Spatial completeness still uses the
    # intended regions and independent actual Newton evidence.
    for key in ('own_endpoint_layout_accepted', 'persistent_role_accepted'):
        if key in metrics:
            other &= metrics[key] > .5
    task = task_contact_accepted.bool()
    if 'task_endpoint_layout_accepted' in metrics:
        task = task & (metrics['task_endpoint_layout_accepted'] > .5)
    other &= ~supervised | task
    checks.update(accepted=other & checks['penetration_passed'],
        penetration_only=checks['penetration_exceeded'] & other,
        penetration_and_contact=checks['penetration_exceeded'] & ~metrics['newton_contact_accepted'].bool())
    return checks


def endpoint_feedback_mask(metrics, task_contact_accepted, supervised, *, limits=DEFAULT_ACCEPTANCE):
    return endpoint_feedback_checks(metrics, task_contact_accepted, supervised, limits=limits)['accepted']


def endpoint_rejection_masks(checks, continuation_reasons):
    """Overlapping reasons plus a truly penetration-only subset of rejections."""
    result = {key: checks[key] for key in ('penetration_exceeded', 'penetration_unknown',
        'penetration_only', 'penetration_and_contact')}
    other = torch.zeros_like(checks['penetration_only'])
    for key, value in continuation_reasons.items():
        if key != 'severe_penetration':
            other |= value
    result['penetration_only'] = result['penetration_only'] & ~other
    return result


def endpoint_validation_score(metrics):
    """Checkpoint ranking uses the same accepted flag and complete-solid depth."""
    return (20 * (1-metrics['plan_exact']) + 40 * (1-metrics['newton_generation_accepted'])
        + 5 * metrics['newton_unrealized_intended_contacts'] + metrics['root_cm'] + metrics['body_cm']
        + 10 * metrics['joint_rmse_rad'] + metrics['solid_penetration_cm']
        + metrics['maxima']['solid_penetration_cm'])


def endpoint_layout_pass(error_squared, complete, count, tolerance_m):
    """Spatial endpoint check, NOT proof of lift-off/touchdown between poses.

    The caller supplies the native part-surface position error and independent
    actual Newton completeness. Geometry alone cannot establish contact.
    """
    if tolerance_m <= 0:
        raise ValueError('Spatial layout tolerance must be positive')
    return complete & (count >= 1) & torch.isfinite(error_squared) & (error_squared <= tolerance_m**2)


def teacher_probability(step, *, execution_only=False, fixed=None):
    """Original full1000 course: full teacher through 200, zero at 700."""
    if fixed is not None:
        if not 0 <= fixed <= 1: raise ValueError('Invalid teacher probability')
        return float(fixed)
    return 1. if execution_only else max(0., min(1., (700-step)/500))


def teacher_mask_for_batch(probability, contact, cell_valid, *, region_known=None):
    """Only use teacher plans whose required locations are actually visible."""
    visible = (cell_valid | ~contact).all(-1)
    if region_known is not None:
        if region_known.dtype != torch.bool or region_known.shape != contact.shape:
            raise ValueError('Native region availability must be bool with the contact shape')
        visible &= (region_known | ~contact).all(-1)
    return (torch.rand(len(contact), device=contact.device) < probability) & visible


def reference_batch_teacher_probability(probability, *, parallel_feedback, continuous_demonstrations):
    """Reference examples keep complete teacher conditions as feedback grows.

    The autonomous rollout schedule is unchanged. Reference examples never
    commit a pose to the rollout pool and retain their demonstrated target.
    """
    return 1.0 if parallel_feedback and continuous_demonstrations else probability


def training_course(step, *, reference_warmup_steps=0, execution_only=False, fixed=None):
    """Prepend reference-only warmup without consuming the autonomous-plan schedule."""
    if step < 1 or reference_warmup_steps < 0:
        raise ValueError('Invalid training course interval')
    if step <= reference_warmup_steps:
        return True, 1.
    return False, teacher_probability(step-reference_warmup_steps,
        execution_only=execution_only, fixed=fixed)


def event_consistent_roles(reference_role, observed_contact):
    """Preserve within-stage touchdown even when a limb starts in contact.

    A demonstrated persistent contact missing from a generated state must be
    re-established. Starting in contact does not cancel a later re-touchdown.
    """
    return torch.where((reference_role == 2) & ~observed_contact, 1, reference_role)


def recovery_transition(indices, successor, completed, recoverable):
    """Incomplete but recoverable outputs retain the same supervision stage."""
    next_indices = successor[indices]
    advance = completed & (next_indices >= 0)
    retry = ~completed & recoverable
    keep = advance | retry
    return torch.where(advance, next_indices, indices), keep, advance, retry
