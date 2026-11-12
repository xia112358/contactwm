import torch
from supermimic.contact_loss_sampling import stable_duration_stops


def test_all_required_parts_stable_retains_late_witness_without_error_gate():
    done=torch.tensor([[False],[False],[False],[True]])
    stable=torch.zeros(4,1,6,dtype=torch.bool)
    stable[1,0,2]=True;stable[3,0,3]=True
    required=torch.tensor([[False,False,True,True,False,False]])
    assert stable_duration_stops(done,stable,required,torch.tensor([2])).tolist()==[4]


def test_one_missing_target_truncates_despite_other_stable_support():
    done=torch.tensor([[False],[False],[False],[True]])
    stable=torch.zeros(4,1,6,dtype=torch.bool);stable[:,:,2]=True
    required=torch.tensor([[False,False,True,True,False,False]])
    assert stable_duration_stops(done,stable,required,torch.tensor([2])).tolist()==[2]


def test_post_terminal_contact_and_no_target_do_not_exempt_cutoff():
    done=torch.tensor([[False,False],[False,False],[True,True],[False,False]])
    stable=torch.zeros(4,2,6,dtype=torch.bool);stable[3,:,2]=True
    required=torch.zeros(2,6,dtype=torch.bool);required[0,2]=True
    assert stable_duration_stops(done,stable,required,torch.tensor([2,2])).tolist()==[2,2]


def test_failed_attempt_uses_band_upper_not_legacy_failure_stop():
    from supermimic.contact_duration import select_duration_batch
    done=torch.zeros(8,2,dtype=torch.bool);done[-1]=True
    data=dict(done=done,term=done.clone(),goal_index=torch.zeros(8,2,dtype=torch.long))
    _,legacy,report=select_duration_batch(data,[3],halfwidth=2)
    assert legacy.tolist()==[8,8]
    upper=torch.tensor([r['upper'] for r in report['segments']])
    cutoffs=torch.minimum(done.long().argmax(0)+1,upper[data['goal_index'][0]])
    stable=torch.zeros(8,2,6,dtype=torch.bool);stable[6,1,2]=True
    required=torch.zeros(2,6,dtype=torch.bool);required[:,2]=True
    assert stable_duration_stops(done,stable,required,cutoffs).tolist()==[5,8]
