"""Rank unperturbed deployment rollouts using already verified Newton facts."""

import math


SELECTION_FIELDS = (
    "task_and_own_safe_prefix",
    "own_plan_safe_prefix",
    "task_and_own_safe_steps",
    "own_plan_safe_steps",
    "safe_steps",
    "negative_max_penetration_cm",
)


def recursive_selection(report: dict) -> dict:
    """Do not infer contacts from distances, plan intentions, or pose errors.

    Task acceptance checks required task contacts, not equality of the emitted
    plan to the demonstration. Repeating an obsolete support plan cannot earn
    task progress. Failed outputs are never retried or reset in this evaluator.
    """
    contract, summary, events = report["contract"], report["summary"], report["events"]
    if contract["recurrence"] != "raw predicted q becomes next current q":
        raise ValueError("selection requires raw continuous recursion")
    if any(contract[key] for key in (
        "projector", "penetration_correction", "teacher_pose_input", "event_index_input"
    )):
        raise ValueError("selection forbids pose correction and future model inputs")
    perturbation = contract["initial_perturbation"]
    if perturbation["backward_m"] != 0 or any(
        value != 0 for value in perturbation["world_offset_m"]
    ):
        raise ValueError("held-out initial perturbations cannot select checkpoints")
    if contract["task_acceptance"]["schema"] != "newton_task_acceptance_v1":
        raise ValueError("unknown Newton acceptance contract")
    stopped = contract.get('failure_policy') == 'stop_on_first_failure'
    if stopped:
        planned = summary.get('planned_events', summary['demonstrated_steps'])
        if not 0 < len(events) <= planned:
            raise ValueError('Invalid stopped-chain length')
        if any(type(e.get('endpoint_accepted')) is not bool for e in events):
            raise ValueError('Stopped chain requires explicit endpoint acceptance')
        if not all(e['endpoint_accepted'] for e in events[:-1]):
            raise ValueError('Failed output was fed to a later event')
        if len(events) < planned and events[-1]['endpoint_accepted']:
            raise ValueError('Incomplete chain lacks a terminal failure')
    if (not events or summary["continuous_groups"] != 1 or summary["extra_steps"] != 0
            or (not stopped and summary["demonstrated_steps"] != len(events)) or summary["steps"] != len(events)):
        raise ValueError("selection requires a complete single task chain")
    joint, own, safe, penetration = [], [], [], []
    for step, event in enumerate(events, 1):
        if event["step"] != step:
            raise ValueError("missing or repeated recursive evaluation step")
        task_acceptance, own_acceptance = event["task_acceptance"], event["own_plan_acceptance"]
        for acceptance in (task_acceptance, own_acceptance):
            if type(acceptance["accepted"]) is not bool:
                raise ValueError("missing/unknown Newton acceptance")
        if task_acceptance["safety_failure"] != own_acceptance["safety_failure"]:
            raise ValueError("task and own plan must share physical safety facts")
        safe_step = own_acceptance["safety_failure"] is None
        own_step = (safe_step and own_acceptance["accepted"]
                    and any(event["predicted_contact"]) and any(event["actual_contact"]))
        own.append(own_step)
        joint.append(own_step and task_acceptance["accepted"] and (not stopped or event['endpoint_accepted']))
        safe.append(safe_step)
        for name in ("terrain_penetration_cm", "self_penetration_cm"):
            value = float(event[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid Newton penetration diagnostic")
            penetration.append(value)

    def prefix(values):
        return next((i for i, value in enumerate(values) if not value), len(values))

    values = (prefix(joint), prefix(own), sum(joint), sum(own), sum(safe), -max(penetration))
    fields = SELECTION_FIELDS
    spatial = contract.get('relative_layout_contract')
    native_schema = 'native_part_surface_to_normal_interval_rms_v1'
    point_schema = 'newton_endpoint_contact_points_xyz_v1'
    set_schema = 'newton_matched_endpoint_contact_points_xyz_v2'
    native_position = contract.get('endpoint_position_contract')
    if native_position is not None:
        if not isinstance(native_position, dict) or native_position.get('schema') not in (native_schema, point_schema, set_schema):
            raise ValueError('unknown native endpoint position contract')
        # The native interval distance is authoritative. The centered witness
        # layout in current reports is explicitly retained for diagnostics.
        spatial = native_position['schema']
    elif contract.get('trained_relative_layout_contract') in (native_schema, point_schema, set_schema):
        raise ValueError('missing native endpoint position contract')
    scene_fixed = spatial in ('newton_scene_fixed_witness_xy_v2', 'newton_region_matched_witness_xy_v3', native_schema, point_schema, set_schema)
    if spatial is not None:
        if spatial != 'newton_representative_witness_xy_v1' and not scene_fixed:
            raise ValueError('unknown relative layout contract')
        errors = []
        for event, accepted in zip(events, own, strict=True):
            if not accepted:
                continue
            if scene_fixed:
                # Topology acceptance alone does not certify region/position.
                # A failed terminal event is valid diagnostic evidence, not a
                # malformed report and not a zero-error successful endpoint.
                if not event.get('endpoint_accepted', False):
                    continue
                value = event.get('own_endpoint_position_rms_cm')
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError('accepted endpoint lacks actual spatial error')
                errors.append(value)
                continue
            if not event['relative_layout_complete']:
                raise ValueError('accepted own plan lacks actual layout witnesses')
            if event.get('relative_layout_parts', sum(event['predicted_contact'])) < 2:
                # One point has no translation-invariant spatial layout.
                continue
            value = event['relative_layout_rms_cm']
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError('missing/unknown actual relative layout')
            errors.append(value)
        # Contact/safety/progress retain priority. Among otherwise equal chains,
        # spatial execution accuracy matters; no distance redefines contact.
        relative = -sum(errors) / len(errors) if errors else 0.0
        error_field = 'negative_safe_endpoint_position_rms_cm' if scene_fixed else 'negative_safe_relative_layout_rms_cm'
        fields = (*SELECTION_FIELDS[:5], error_field, SELECTION_FIELDS[5])
        values = (*values[:5], relative, values[5])
    return {
        "schema": ("newton_recursive_checkpoint_selection_v3" if scene_fixed else
                   "newton_recursive_checkpoint_selection_v2" if spatial else "newton_recursive_checkpoint_selection_v1"),
        "ordering": "lexicographic, larger is better; preserve earlier checkpoint on exact ties",
        "fields": list(fields),
        "key": list(values),
        "metrics": dict(zip(fields, values, strict=True)),
        "steps": len(events),
        "initial_perturbations_used": False,
        "absolute_pose_and_cell_errors_used": False,
        "relative_layout_accuracy_used": bool(spatial) and not scene_fixed,
        "endpoint_position_accuracy_used": scene_fixed,
        "relative_layout_minimum_parts": 1 if scene_fixed else 2,
        "training_retry_history_used": False,
        "task_progress": "required task contacts accepted at each successive evaluation stage",
    }
