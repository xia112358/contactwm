"""One next-contact prediction per event; no retry or pose correction."""
from contact_solver.endpoint_task import ENDPOINT_POSITION_TOLERANCE_M


def endpoint_failure_reasons(record, *, require_spatial=False,
                             tolerance_cm=100*ENDPOINT_POSITION_TOLERANCE_M, require_observed_support=False):
    reasons=[]
    # Historical endpoint displacement fields cannot judge loaded path motion.
    # Actual contact/safety stay mandatory; physical support is checked only
    # against independent execution evidence when explicitly requested.
    if require_observed_support:
        from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
        support = record.get('observed_support', {})
        if (support.get('schema') != SUPPORT_ASSESSMENT_SCHEMA
                or not support.get('load_evidence_complete')):
            reasons.append('actual_support_evidence_unknown')
        elif (not support.get('load_coverage_passed')
                or support.get('stationary_support_status') != 'passed_at_recorded_solve_samples'):
            reasons.append('actual_support_not_verified')
    task = record.get('task_acceptance')
    own = record.get('own_plan_acceptance')
    if task is None and own is None:
        raise ValueError('Missing actual endpoint acceptance evidence')
    if task is not None and not task['accepted']:
        reasons.append('task_contact_or_safety')
    if not (own if own is not None else task)['accepted']:
        reasons.append('own_contact_or_safety')
    # Keep region_plan_acceptance as a witness-distribution audit. An endpoint
    # need not reproduce every quadrant from a demonstrated contact snapshot.
    # Actual contact/safety and region-filtered spatial completeness are
    # checked independently below and by own_plan_acceptance above.
    if record.get('persistent_role_consistent') is False:
        reasons.append('persistent_role_without_current_contact')
    if require_spatial:
        error=record.get('own_endpoint_position_rms_cm')
        if error is None or not __import__('math').isfinite(error) or error > tolerance_cm:
            reasons.append('own_spatial_endpoint')
    return reasons
