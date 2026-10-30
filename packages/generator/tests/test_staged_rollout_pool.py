import copy
import pytest
import torch
from generator.staged_rollout_pool import StagedRolloutPool, validate_stage_transition
from generator.training_state import validate_resume_config


def pool(stage='demonstration_rollout', bank=None):
    ref={'current_q': torch.tensor([[0.], [1.], [2.]])}
    return StagedRolloutPool(ref, torch.arange(3), torch.tensor([1,2,-1]), 1,
        stage=stage, terminal_bank=bank, max_episode_steps=4)


def update(p, value, success=True):
    slots, _, _ = p.batch(1)
    return p.update(slots, torch.tensor([[value]]), None, torch.tensor([success]),
        torch.tensor([True]), lambda q, indices, actual, chosen: {'current_q':q},
        reset_reasons={'own_contact':torch.tensor([not success])})


def test_stage2_random_start_terminal_capture_and_reset():
    p=pool()
    assert p.seed_indices.tolist()==[0,1,2]
    p.indices.fill_(1);p.observation['current_q'].fill_(1)
    update(p, 12.)
    assert p.indices.item()==2 and not p.terminal_bank
    update(p, 23.)
    assert p.supervised.all() and p.age.item()==0
    assert p.terminal_bank[2]['current_q'].item()==23
    assert p.finished_episodes[-1]['accepted_steps']==2
    assert p.finished_episodes[-1]['completed_demonstration']


def test_stage3_counts_only_extra_steps_and_resets_to_actual_endpoint():
    p=pool('post_demonstration',{2:{'current_q':torch.tensor([23.])}})
    assert p.age.item()==0 and not p.supervised.any()
    update(p, 24.)
    feedback=update(p, -100., False)
    episode=p.finished_episodes[-1]
    assert episode['accepted_steps']==1 and episode['attempts']==2 and episode['retries']==1
    assert episode['first_failure_depth']==1 and episode['first_failure_reasons']==['own_contact']
    assert episode['reset_reason']=='prediction_rejected'
    assert p.age.item()==0 and p.observation['current_q'].item()==23
    assert p.attempts.item()==0 and p.retries.item()==0
    assert p.first_failure_depth.item()==-1 and p.episode_retries.item()==0
    assert feedback['rejection_resets'].item()==1 and feedback['rollbacks'].item()==0
    assert feedback['periodic_resamples'].item()==0
    assert not p.supervised.any()


def test_stage3_refuses_missing_actual_terminal_bank():
    with pytest.raises(ValueError,match='actual successful stage-2'):
        pool('post_demonstration')


def test_resume_preserves_seed_bank_cursor_depth_failures_and_next_transition():
    p=pool('post_demonstration',{2:{'current_q':torch.tensor([23.])}})
    update(p, 24.)
    state=copy.deepcopy(p.state_dict());rng=torch.get_rng_state()
    b=pool('post_demonstration',state['staged']['terminal_bank']);b.load_state_dict(state)
    assert b.stage_metrics()==p.stage_metrics()
    torch.set_rng_state(rng);update(p,-100.,False);update(p,26.)
    torch.set_rng_state(rng);update(b,-100.,False);update(b,26.)
    assert torch.equal(p.observation['current_q'],b.observation['current_q'])
    assert b.stage_metrics()==p.stage_metrics()


def test_stage_transitions_are_explicit_and_do_not_change_optimizer_schedule():
    a=dict(training_stage='single_segment',steps=100,schedule_steps=1100)
    b=dict(a,training_stage='demonstration_rollout',steps=600)
    assert validate_stage_transition(a,b)
    validate_resume_config(a,b)
    validate_resume_config(b,dict(b,steps=700))
    with pytest.raises(ValueError,match='configuration differs'):
        validate_resume_config(a,dict(b,schedule_steps=600))
    with pytest.raises(ValueError,match='Unsupported'):
        validate_stage_transition(a,dict(b,training_stage='post_demonstration'))
    with pytest.raises(ValueError,match='Unsupported'):
        validate_stage_transition({},b)


def test_rejected_terminal_never_seeds_stage3():
    p=pool();p.indices.fill_(2);p.observation['current_q'].fill_(2)
    update(p, 99.,False)
    assert not p.terminal_bank and p.age.item()==0
    assert p.observation['current_q'].item()==2
    assert p.first_failure_depth.item()==0
    assert p.finished_episodes==[]
    assert p.attempts.item()==1


def test_stage2_bank_survives_restore_and_stage3_uses_generated_pose():
    p=pool();p.indices.fill_(2)
    update(p, 37.)
    state=copy.deepcopy(p.state_dict());b=pool();b.load_state_dict(state)
    c=pool('post_demonstration',b.terminal_bank)
    assert c.observation['current_q'].item()==37 and c.age.item()==0
    assert not c.supervised.any()


def test_terminal_on_attempt_limit_still_retains_accepted_endpoint():
    p=pool();p.indices.fill_(2);p.attempts.fill_(3)
    update(p, 48.)
    assert p.terminal_bank[2]['current_q'].item()==48
    assert p.finished_episodes[-1]['completed_demonstration']
    assert p.episode_totals['count']==1


def test_stage3_continuation_rejection_resets_only_failed_slots():
    ref={'current_q':torch.tensor([[0.],[1.],[2.]])}
    p=StagedRolloutPool(ref,torch.arange(3),torch.tensor([1,2,-1]),2,
        stage='post_demonstration',terminal_bank={2:{'current_q':torch.tensor([23.])}},
        max_episode_steps=4)
    slots,_,_=p.batch(2)
    feedback=p.update(slots,torch.tensor([[24.],[-100.]]),None,
        torch.tensor([True,True]),torch.tensor([True,False]),
        lambda q,indices,actual,chosen:{'current_q':q},
        reset_reasons={'invalid':torch.tensor([False,True])})
    assert p.observation['current_q'].flatten().tolist()==[24.,23.]
    assert p.age.tolist()==[1,0] and p.attempts.tolist()==[1,0]
    assert not p.supervised.any()
    assert feedback['accepted_transitions'].item()==1
    assert feedback['rejection_resets'].item()==1
    assert feedback['rollbacks'].item()==0
    assert p.finished_episodes[0]['final_rejection_reasons']==['invalid']
    assert p.finished_episodes[0]['accepted_steps']==0


def test_stage3_accepted_attempt_budget_and_rejection_priority():
    p=pool('post_demonstration',{2:{'current_q':torch.tensor([23.])}})
    for value in (24.,25.,26.,27.):
        feedback=update(p,value)
    assert p.finished_episodes[-1]['reset_reason']=='attempt_budget'
    assert p.finished_episodes[-1]['accepted_steps']==4
    assert feedback['periodic_resamples'].item()==1
    assert feedback['rejection_resets'].item()==0
    for value in (24.,25.,26.):update(p,value)
    feedback=update(p,-100.,False)
    assert p.finished_episodes[-1]['reset_reason']=='prediction_rejected'
    assert p.finished_episodes[-1]['accepted_steps']==3
    assert feedback['periodic_resamples'].item()==0
    assert feedback['rejection_resets'].item()==1
    assert p.observation['current_q'].item()==23


def test_resume_rejection_policy_is_explicit_and_stage2_is_compatible():
    p=pool('post_demonstration',{2:{'current_q':torch.tensor([23.])}})
    state=copy.deepcopy(p.state_dict())
    assert state['staged']['rejection_policy']=='reset_episode'
    del state['staged']['rejection_policy']
    with pytest.raises(ValueError,match='rejection policy differs'):
        p.load_state_dict(state)
    b=pool();old=copy.deepcopy(b.state_dict())
    del old['staged']['rejection_policy']
    b.load_state_dict(old)
    assert b.rejection_policy=='rollback'
