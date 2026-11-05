from types import SimpleNamespace
import torch
from supermimic.contact_dmo import loss_aligned_attempt_metrics


def fixture():
    T,B=3,4
    state=torch.zeros(T,B,71);state[...,2]=.8;state[...,3]=1
    masks=torch.zeros(1,6,dtype=torch.bool);masks[0,0]=True
    clock=SimpleNamespace(event_ends=torch.tensor([1]),target_masks=masks,keep_masks=torch.zeros_like(masks),
        data=dict(point=torch.zeros(1,6,3),s1=state[0,:1].clone()))
    stable=torch.zeros(T,B,6,dtype=torch.bool);stable[0,0,0]=True;stable[1,0,0]=True;stable[0,1,0]=True;stable[1,3,0]=True
    points=torch.zeros(T,B,6,3);points[0,0,0,0]=.01;points[1,0,0,0]=.08;points[0,1,0,0]=.02;points[1,3,0,0]=.01
    state[-1,3,0]=.3
    data=dict(done=torch.tensor([[False]*B,[False]*B,[True]*B]),s1=state,
        goal_index=torch.zeros(T,B,dtype=torch.long),contact_stable=stable,contact_loss_point=points)
    return data,clock


def test_loss_witness_success_error_and_missing_are_consistent():
    data,clock=fixture();r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    # Last stable point for attempt 0 is 8cm even though an earlier one is 1cm.
    # Attempt 1 succeeds, attempt 2 lacks a witness, attempt 3 fails root.
    assert r['attempts']==4 and r['success_rate']==.25
    assert r['missing_witness_attempts']==1 and r['contact_error_samples']==3
    assert abs(r['best_contact_error_cm']-(8+2+1)/3)<1e-5
    assert r['segments'][0]['success_rate']==.25


def test_part_averaging_matches_loss_and_requires_all_parts():
    data,clock=fixture();clock.target_masks[0,1]=True
    data['contact_stable'][1,1,1]=True;data['contact_loss_point'][1,1,1,0]=.04
    r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['contact_error_samples']==1 and r['missing_witness_attempts']==3
    assert abs(r['best_contact_error_cm']-3.)<1e-5
    assert r['success_rate']==.25


def test_samples_past_terminal_never_define_a_witness():
    data,clock=fixture();data['done'][1,2]=True;data['contact_stable'][2,2,0]=True
    r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['missing_witness_attempts']==1
