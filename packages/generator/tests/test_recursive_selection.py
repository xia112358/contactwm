from copy import deepcopy

import pytest

from generator.recursive_selection import recursive_selection


def test_stopped_prefix_is_valid_but_continuing_after_failure_is_rejected():
    data=report([True,False],[True,True])
    data['contract']['failure_policy']='stop_on_first_failure'
    data['summary']['planned_events']=5
    data['summary']['demonstrated_steps']=5
    for e,ok in zip(data['events'],[True,False]): e['endpoint_accepted']=ok
    assert recursive_selection(data)['metrics']['task_and_own_safe_prefix']==1
    data['events'][0]['endpoint_accepted']=False
    with pytest.raises(ValueError,match='fed to a later event'):
        recursive_selection(data)


def test_spatial_failure_is_not_counted_as_successful_event():
    data=report([True],[True])
    data['contract']['failure_policy']='stop_on_first_failure'
    data['summary']['planned_events']=3
    data['events'][0]['endpoint_accepted']=False
    assert recursive_selection(data)['metrics']['task_and_own_safe_prefix']==0


@pytest.mark.parametrize('contract', ['newton_scene_fixed_witness_xy_v2', 'newton_region_matched_witness_xy_v3'])
def test_scene_fixed_selection_uses_endpoint_error_not_centered_diagnostic(contract):
    data=report([True,True],[True,True])
    data['contract'].update(failure_policy='stop_on_first_failure', relative_layout_contract=contract)
    data['summary']['planned_events']=3
    data['events'][0].update(endpoint_accepted=True,own_endpoint_position_rms_cm=2.,relative_layout_rms_cm=99.)
    data['events'][1].update(endpoint_accepted=False,own_endpoint_position_rms_cm=None)
    result=recursive_selection(data)
    assert result['metrics']['task_and_own_safe_prefix']==1
    assert result['metrics']['negative_safe_endpoint_position_rms_cm']==-2.
    assert result['endpoint_position_accuracy_used']
    data['events'][0]['own_endpoint_position_rms_cm']=None
    with pytest.raises(ValueError,match='lacks actual spatial error'):
        recursive_selection(data)


@pytest.mark.parametrize('CONTACT_POSITION_SCHEMA', [
    'native_part_surface_to_normal_interval_rms_v1', 'newton_endpoint_contact_points_xyz_v1',
    'newton_matched_endpoint_contact_points_xyz_v2'])
def test_native_interval_selection_uses_verified_endpoint_error(CONTACT_POSITION_SCHEMA):
    data = report([True, True], [True, True])
    data['contract'].update(failure_policy='stop_on_first_failure',
        trained_relative_layout_contract=CONTACT_POSITION_SCHEMA,
        relative_layout_contract='historical_centered_witness_diagnostic_only',
        endpoint_position_contract={'schema': CONTACT_POSITION_SCHEMA})
    data['summary']['planned_events'] = 3
    data['events'][0].update(endpoint_accepted=True, own_endpoint_position_rms_cm=2.,
        relative_layout_complete=False, relative_layout_rms_cm=99.)
    data['events'][1].update(endpoint_accepted=False, own_endpoint_position_rms_cm=None)
    result = recursive_selection(data)
    assert result['endpoint_position_accuracy_used']
    assert not result['relative_layout_accuracy_used']
    assert result['metrics']['task_and_own_safe_prefix'] == 1
    assert result['metrics']['negative_safe_endpoint_position_rms_cm'] == -2.
    data['events'][0]['own_endpoint_position_rms_cm'] = None
    with pytest.raises(ValueError, match='lacks actual spatial error'):
        recursive_selection(data)


def test_native_position_contract_cannot_fall_back_to_historical_layout():
    from contact_solver.native_contact_position import CONTACT_POSITION_SCHEMA
    data = report([True], [True])
    data['contract'].update(trained_relative_layout_contract=CONTACT_POSITION_SCHEMA,
        relative_layout_contract='newton_representative_witness_xy_v1')
    with pytest.raises(ValueError, match='missing native endpoint position contract'):
        recursive_selection(data)
    data['contract']['endpoint_position_contract'] = {'schema': 'unknown'}
    with pytest.raises(ValueError, match='unknown native endpoint position contract'):
        recursive_selection(data)


def report(task, own):
    return {
        "contract": {
            "recurrence": "raw predicted q becomes next current q",
            "projector": False, "penetration_correction": False,
            "teacher_pose_input": False, "event_index_input": False,
            "initial_perturbation": {"backward_m": 0, "world_offset_m": [0, 0, 0]},
            "task_acceptance": {"schema": "newton_task_acceptance_v1"},
        },
        "summary": {"steps": len(task), "demonstrated_steps": len(task),
                    "extra_steps": 0, "continuous_groups": 1},
        "events": [
            {"step": i + 1, "task_acceptance": {"accepted": t, "safety_failure": None},
             "own_plan_acceptance": {"accepted": o, "safety_failure": None},
             "predicted_contact": [True, True, False, False, False, False],
             "actual_contact": [True, True, False, False, False, False],
             "terrain_penetration_cm": 0, "self_penetration_cm": 0}
            for i, (t, o) in enumerate(zip(task, own, strict=True))
        ],
    }


def test_contiguous_prefix_beats_many_disconnected_successes():
    contiguous = report([True, True, False, False], [True, True, False, False])
    disconnected = report([True, False, True, True], [True, False, True, True])
    assert recursive_selection(contiguous)["key"] > recursive_selection(disconnected)["key"]


def test_repeating_obsolete_plan_cannot_outscore_task_progress():
    stationary = report([True, False, False, False], [True] * 4)
    progressing = report([True, True, False, False], [True, True, False, False])
    assert recursive_selection(progressing)["key"] > recursive_selection(stationary)["key"]


def test_task_success_without_own_plan_realization_does_not_count():
    result = recursive_selection(report([True, True], [True, False]))
    assert result["metrics"]["task_and_own_safe_prefix"] == 1


@pytest.mark.parametrize("field,value", [("backward_m", 0.1), ("world_offset_m", [0.1, 0, 0])])
def test_rejects_held_out_perturbations(field, value):
    data = report([True], [True])
    data["contract"]["initial_perturbation"][field] = value
    with pytest.raises(ValueError, match="held-out"):
        recursive_selection(data)


@pytest.mark.parametrize("field", ["projector", "penetration_correction", "teacher_pose_input"])
def test_rejects_corrected_or_teacher_rollouts(field):
    data = report([True], [True])
    data["contract"][field] = True
    with pytest.raises(ValueError):
        recursive_selection(data)


def test_pose_and_absolute_cell_errors_do_not_change_ranking():
    data = report([True, True], [True, True])
    moved = deepcopy(data)
    for event in moved["events"]:
        event.update(root_error_cm=100, intended_point_error_cm=50,
                     own_spatial_plan_realized=False, predicted_plan_matches_gt=False)
    assert recursive_selection(data)["key"] == recursive_selection(moved)["key"]


def test_empty_intention_cannot_claim_realization():
    data = report([True], [True])
    data["events"][0]["predicted_contact"] = [False] * 6
    assert recursive_selection(data)["metrics"]["own_plan_safe_steps"] == 0


def test_missing_authoritative_acceptance_cannot_fall_back_to_distance():
    data = report([True], [True])
    del data["events"][0]["own_plan_acceptance"]
    with pytest.raises(KeyError):
        recursive_selection(data)


def test_safety_failure_breaks_success_prefix():
    data = report([True, True], [True, True])
    for name in ("task_acceptance", "own_plan_acceptance"):
        data["events"][1][name]["safety_failure"] = "penetration"
    result = recursive_selection(data)
    assert result["metrics"]["task_and_own_safe_prefix"] == 1
    assert result["metrics"]["safe_steps"] == 1


def test_later_penetration_is_a_tiebreaker_not_an_average_reward():
    data = report([True, False], [True, False])
    worse = deepcopy(data)
    worse["events"][1]["self_penetration_cm"] = 2
    assert recursive_selection(data)["key"] > recursive_selection(worse)["key"]


def test_relative_layout_changes_spatial_ranking_without_penalizing_common_offset():
    data = report([True, True], [True, True])
    data['contract']['relative_layout_contract'] = 'newton_representative_witness_xy_v1'
    for event in data['events']:
        event.update(relative_layout_complete=True, relative_layout_rms_cm=1., common_contact_offset_xy_cm=[0., 0.])
    moved, worse = deepcopy(data), deepcopy(data)
    for event in moved['events']:
        event['common_contact_offset_xy_cm'] = [50., -20.]
    worse['events'][1]['relative_layout_rms_cm'] = 20.
    assert recursive_selection(data)['key'] == recursive_selection(moved)['key']
    assert recursive_selection(data)['key'] > recursive_selection(worse)['key']
    del worse['events'][1]['relative_layout_rms_cm']
    with pytest.raises(KeyError):
        recursive_selection(worse)


def test_cannot_score_truncated_chain():
    data = report([True, True], [True, True])
    data["events"].pop()
    with pytest.raises(ValueError, match="complete"):
        recursive_selection(data)
