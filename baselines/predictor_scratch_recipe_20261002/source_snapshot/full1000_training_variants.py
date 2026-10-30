"""Small, explicit training ablations; never redefine Newton contact truth."""
import torch


def endpoint_layout_pass(error_squared, complete, count, tolerance_m):
    """Spatial endpoint check, NOT proof of lift-off/touchdown between poses.

    Scene-fixed positions also constrain a single actual Newton witness.
    Missing/underdetermined observations must never count as successful layout.
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


def teacher_mask_for_batch(probability, contact, cell_valid):
    """Only use teacher plans whose required locations are actually visible."""
    visible = (cell_valid | ~contact).all(-1)
    return (torch.rand(len(contact), device=contact.device) < probability) & visible


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
