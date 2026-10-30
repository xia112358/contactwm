from generator.endpoint_termination import endpoint_failure_reasons


def event():
    return dict(task_acceptance={'accepted':True},own_plan_acceptance={'accepted':True},
                region_plan_acceptance={'accepted':True},own_endpoint_position_rms_cm=1.)


def test_any_failed_required_check_stops_even_with_valid_contact():
    for key in ('task_acceptance','own_plan_acceptance'):
        row=event();row[key]['accepted']=False
        assert endpoint_failure_reasons(row)
    assert not endpoint_failure_reasons(event(),require_spatial=True)


def test_partial_quadrant_evidence_does_not_reject_completed_endpoint():
    row = event()
    row['region_plan_acceptance'].update(accepted=False, missing_regions=[[True, False, False, False]])
    assert not endpoint_failure_reasons(row, require_spatial=True)
    row['own_endpoint_position_rms_cm'] = None
    assert endpoint_failure_reasons(row, require_spatial=True) == ['own_spatial_endpoint']
    row['own_plan_acceptance']['accepted'] = False
    assert endpoint_failure_reasons(row) == ['own_contact_or_safety']


def test_spatial_unknown_failure_is_explicit_not_a_contact_redefinition():
    for error in (None,float('nan'),float('inf'),10.):
        row=event();row['own_endpoint_position_rms_cm']=error
        assert endpoint_failure_reasons(row,require_spatial=True)==['own_spatial_endpoint']
        assert not endpoint_failure_reasons(row,require_spatial=False)


def test_retired_endpoint_displacement_cannot_reject_or_certify_contact_transfer():
    from somaforge_core.loaded_material_motion import unknown_material_path
    row=event()
    row.update(endpoint_contact_retention_valid=False, endpoint_material_displacement_cm=7.01,
               support_transition_valid=False, support_motion=unknown_material_path())
    assert not endpoint_failure_reasons(row)
    assert endpoint_failure_reasons(row,require_observed_support=True)==['actual_support_evidence_unknown']
    row['task_acceptance']['accepted']=False
    assert endpoint_failure_reasons(row)==['task_contact_or_safety']


def test_real_execution_support_still_requires_load_coverage_and_stationarity():
    from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
    row=event()
    row['observed_support']=dict(schema=SUPPORT_ASSESSMENT_SCHEMA,load_evidence_complete=True,
        load_coverage_passed=True,stationary_support_status='passed_at_recorded_solve_samples')
    assert not endpoint_failure_reasons(row,require_observed_support=True)
    row['observed_support']['load_coverage_passed']=False
    assert endpoint_failure_reasons(row,require_observed_support=True)==['actual_support_not_verified']


def test_autonomous_endpoint_has_no_future_task_but_requires_actual_own_checks():
    import pytest
    row = event(); row['task_acceptance'] = None
    assert not endpoint_failure_reasons(row, require_spatial=True)
    row['own_plan_acceptance']['accepted'] = False
    assert endpoint_failure_reasons(row) == ['own_contact_or_safety']
    row['own_plan_acceptance'] = None
    with pytest.raises(ValueError, match='actual endpoint acceptance'):
        endpoint_failure_reasons(row)
