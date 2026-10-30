import torch


def test_endpoint_layout_rejects_stale_step_and_future_shift_but_accepts_one_known_contact():
    from contact_solver.contact_layout import endpoint_position_statistics
    from generator.full1000_training_variants import endpoint_layout_pass
    # A foot moves relative to a fixed observed support; future-only shift is penalized.
    start = torch.tensor([[[0., 0., 0.], [0., .2, 0.]]])
    target = start.clone(); target[:, 0, 0] += .2
    present = torch.ones(1, 2, dtype=torch.bool)
    stale, count = endpoint_position_statistics(start, target, present)
    complete = torch.ones(1, dtype=torch.bool)
    assert not endpoint_layout_pass(stale, complete, count, .04).item()
    shifted, count = endpoint_position_statistics(target + 3., target, present)
    assert not endpoint_layout_pass(shifted, complete, count, .04).item()
    assert endpoint_layout_pass(torch.zeros_like(shifted), complete, torch.ones_like(count), .04).item()
    assert not endpoint_layout_pass(shifted, ~complete, count, .04).item()


def test_device_realized_layout_uses_world_points_even_when_fk_is_local():
    from contact_solver.device_contact_objective import DeviceWitnessRows
    from generator.full1000_training_variants import endpoint_layout_pass
    rows = object.__new__(DeviceWitnessRows)
    rows.q = torch.zeros(1, 36)
    world = torch.zeros(1, 6, 3)
    world[0, :2] = torch.tensor([[10., 20., 0.], [10., 20.2, 0.]])
    rows.pair = {'geometry_point1_w': world[0]}
    active = torch.tensor([[True, True, False, False, False, False]])
    rows.spatial_representatives = lambda points, active, surface, actual=False, regions=None: (torch.arange(6)[None], active.clone())
    surface = torch.zeros(1, 6, dtype=torch.long)
    error, complete, count = rows.actual_witness_position_statistics(world, active, surface)
    assert endpoint_layout_pass(error, complete, count, .04).item()
    local = torch.zeros_like(world); local[0, 1, 0] = .2
    error, complete, count = rows.actual_witness_position_statistics(local, active, surface)
    assert not endpoint_layout_pass(error, complete, count, .04).item()

from generator.full1000_training_variants import (event_consistent_roles, recovery_transition,
    teacher_probability, teacher_mask_for_batch)


def test_touchdown_inside_stage_is_not_relabelled_as_persistent_at_same_input():
    role = torch.tensor([[1, 2, 0, 1, 2, 0]])
    observed = torch.tensor([[True, True, False, False, False, False]])
    assert event_consistent_roles(role, observed).tolist() == [[1, 2, 0, 1, 1, 0]]


def test_recovery_retains_target_and_only_completed_stages_advance():
    indices = torch.tensor([0, 1, 2, 3])
    successor = torch.tensor([1, 2, 3, -1])
    completed = torch.tensor([False, True, False, True])
    recoverable = torch.tensor([True, True, False, True])
    nxt, keep, advance, retry = recovery_transition(indices, successor, completed, recoverable)
    assert nxt.tolist() == [0, 2, 2, 3]
    assert keep.tolist() == [True, True, False, False]
    assert advance.tolist() == [False, True, False, False]
    assert retry.tolist() == [True, False, False, False]


def test_full1000_teacher_course_and_execution_only_override():
    assert [teacher_probability(s) for s in (1,200,450,700,1000,5000)] == [1.,1.,.5,0.,0.,0.]
    assert teacher_probability(5000,execution_only=True) == 1.
    assert teacher_probability(1,fixed=0) == 0.
    assert teacher_probability(5000,fixed=.5) == .5


def test_teacher_visibility_and_autonomous_inputs_do_not_modify_state():
    contact=torch.tensor([[True,False],[True,True],[False,False]])
    visible=torch.tensor([[True,False],[True,False],[False,False]])
    assert teacher_mask_for_batch(1.,contact,visible).tolist() == [True,False,True]
    assert not teacher_mask_for_batch(0.,contact,visible).any()
    assert contact.tolist() == [[True,False],[True,True],[False,False]]


def test_teacher_region_evidence_is_required_only_for_active_parts():
    contact = torch.tensor([[True, False], [True, True], [False, False]])
    known = torch.tensor([[True, False], [True, False], [False, False]])
    assert teacher_mask_for_batch(1., contact, torch.ones_like(contact), region_known=known).tolist() == [True, False, True]


def test_demonstration_batches_keep_teacher_conditions_after_autonomous_schedule_ends():
    from generator.full1000_training_variants import reference_batch_teacher_probability
    for probability in (1., .5, 0.):
        assert reference_batch_teacher_probability(probability, parallel_feedback=True,
            continuous_demonstrations=True) == 1.
        assert reference_batch_teacher_probability(probability, parallel_feedback=True,
            continuous_demonstrations=False) == probability
        assert reference_batch_teacher_probability(probability, parallel_feedback=False,
            continuous_demonstrations=True) == probability


def test_reference_warmup_preserves_full_teacher_course_after_transition():
    from generator.full1000_training_variants import training_course
    assert training_course(100, reference_warmup_steps=100, fixed=0.) == (True, 1.)
    assert training_course(101, reference_warmup_steps=100) == (False, 1.)
    assert training_course(550, reference_warmup_steps=100) == (False, .5)
    assert training_course(800, reference_warmup_steps=100) == (False, 0.)
    assert training_course(101, reference_warmup_steps=100, fixed=0.) == (False, 0.)
    for step in (1, 200, 450, 700, 1000):
        assert training_course(step) == (False, teacher_probability(step))


def test_post_demo_commit_ignores_stale_demonstration_but_checks_own_feasibility():
    from contact_solver.interaction_acceptance import configured_acceptance
    from generator.full1000_training_variants import endpoint_feedback_mask
    metrics = dict(newton_contact_accepted=torch.ones(2), solid_penetration_cm=torch.full((2,), .2),
        newton_invalid_fullbody_witnesses=torch.zeros(2), joint_violation_rad=torch.zeros(2),
        region_plan_realized=torch.ones(2), own_endpoint_layout_accepted=torch.ones(2),
        persistent_role_accepted=torch.ones(2), task_endpoint_layout_accepted=torch.tensor([0., float('nan')]))
    limits = configured_acceptance({'acceptance_penetration_m': .005})
    supervised = torch.tensor([True, False])
    assert endpoint_feedback_mask(metrics, torch.zeros(2, dtype=torch.bool), supervised,
                                  limits=limits).tolist() == [False, True]
    # The source event does not require reproducing every quadrant witness;
    # actual contact and region-filtered position checks still apply.
    partial = {k:v.clone() for k,v in metrics.items()}
    partial['region_plan_realized'].zero_()
    assert endpoint_feedback_mask(partial, torch.zeros(2, dtype=torch.bool), supervised,
                                  limits=limits).tolist() == [False, True]
    for key, value in [('newton_contact_accepted', 0.), ('solid_penetration_cm', .501),
                       ('newton_invalid_fullbody_witnesses', 1.), ('joint_violation_rad', .151),
                       ('own_endpoint_layout_accepted', 0.),
                       ('persistent_role_accepted', 0.)]:
        changed = {k:v.clone() for k,v in metrics.items()}
        changed[key][1] = value
        assert not endpoint_feedback_mask(changed, torch.zeros(2, dtype=torch.bool), supervised,
                                           limits=limits).any(), key
