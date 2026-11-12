import pytest
import torch
import warp as wp
from supermimic.contact_onset import NativeContactOnset, first_contact, held_joints, part_joint_mask, contact_decision


class Sequence:
    def __init__(self, initial, required, device='cpu'):
        self.device=device;self.onset=NativeContactOnset(1,device,exit_steps=3)
        self.onset.begin_stage(torch.tensor([0],device=device),torch.tensor([initial],device=device),torch.tensor([required],device=device))
        self.known=wp.ones(1,dtype=wp.bool,device=device);self.substep=wp.zeros(1,dtype=wp.int32,device=device)
        self.stable=wp.zeros(6,dtype=wp.bool,device=device)
    def launch(self):
        o=self.onset
        wp.launch(first_contact,6,inputs=[o.counts,o.counts,self.known,o.required,o.blocked,o.absent,o.delivered,o.exit_steps,self.substep,o.event,o.stop_substep,o.pending,o.event_parts,o.done,o.lost,o.motion_release],device=self.device)
    def sample(self,counts,known=True,substep=0):
        wp.to_torch(self.onset.counts).copy_(torch.tensor(counts,device=self.device))
        wp.to_torch(self.known)[0]=known;wp.to_torch(self.substep)[0]=substep
        self.launch();return wp.to_torch(self.onset.event).tolist()
    def confirm(self,stable):
        wp.to_torch(self.stable).copy_(torch.tensor(stable,device=self.device))
        self.onset.confirm(self.stable,self.known)
        return wp.to_torch(self.onset.done).tolist()


def test_airborne_entry_first_contact_needs_no_release():
    s=Sequence([False]*6,[True,False,False,False,False,False])
    assert s.sample([1,0,0,0,0,0])==[True,False,False,False,False,False]


def test_initial_support_row_changes_and_short_dropout_are_not_new_contact():
    s=Sequence([True]*6,[True]*6)
    for counts in ([1]*6,[4]*6,[2]*6,[0]*6,[0]*6,[1]*6):assert not any(s.sample(counts))
    for _ in range(3):assert not any(s.sample([0]*6))
    assert all(s.sample([1]*6,substep=2))


def test_unknown_evidence_does_not_count_as_release():
    s=Sequence([True]*6,[True]*6)
    s.sample([0]*6);s.sample([0]*6);s.sample([0]*6,known=False);s.sample([0]*6)
    assert not any(s.sample([1]*6))


def test_later_part_has_its_own_onset_and_confirmation():
    s=Sequence([False]*6,[True]*6)
    s.sample([1,0,0,0,0,0],substep=1);assert s.confirm([True,False,False,False,False,False])==[True,False,False,False,False,False]
    s.onset.begin_control()
    assert s.sample([1,1,0,0,0,0],substep=3)==[False,True,False,False,False,False]
    assert wp.to_torch(s.onset.stop_substep).tolist()==[-1,3,-1,-1,-1,-1]
    assert s.confirm([False,True,False,False,False,False])==[True,True,False,False,False,False]


def test_stage_reset_reinitializes_airborne_and_support_paths():
    s=Sequence([False]*6,[True]*6);s.sample([1]*6);s.confirm([True]*6)
    s.onset.begin_stage(torch.tensor([0]),torch.tensor([[True,False,True,True,True,True]]),torch.ones(1,6,dtype=torch.bool))
    assert not any(s.sample([1,0,1,1,1,1]))
    assert s.sample([1]*6)==[False,True,False,False,False,False]
    assert not wp.to_torch(s.onset.done).any()


def test_non_target_support_cannot_end_attempt():
    s=Sequence([False]*6,[False,True,False,False,False,False])
    assert not any(s.sample([2,0,3,1,0,0]))
    assert s.sample([2,1,3,1,0,0])==[False,True,False,False,False,False]


def test_confirmation_has_no_timeout_from_release_debounce():
    s=Sequence([False]*6,[True]*6);s.sample([1,0,0,0,0,0])
    for _ in range(100):
        s.sample([1,0,0,0,0,0]);assert not any(s.confirm([False]*6))
    assert int(wp.to_torch(s.onset.confirmation_steps)[0])==100
    assert s.confirm([True,False,False,False,False,False])==[True,False,False,False,False,False]


def test_contact_lost_is_measured_absence_not_time_spent_confirming():
    s=Sequence([False]*6,[True]*6);s.sample([1,0,0,0,0,0])
    for _ in range(2):
        s.sample([0]*6);assert not any(s.confirm([False]*6))
    s.sample([1,0,0,0,0,0]);assert not any(s.confirm([False]*6))
    for _ in range(3):s.sample([0]*6);s.confirm([False]*6)
    assert wp.to_torch(s.onset.lost).tolist()==[True,False,False,False,False,False]


def test_confirmation_survives_control_boundary():
    s=Sequence([False]*6,[True]*6);s.sample([1,0,0,0,0,0],substep=3)
    s.onset.begin_control();assert not any(s.sample([1,0,0,0,0,0]))
    assert wp.to_torch(s.onset.pending).tolist()==[True,False,False,False,False,False]


def test_good_first_target_never_ends_multi_target_attempt():
    required=torch.tensor([[True,True,False,False,False,False]])
    one=torch.tensor([[True,False,False,False,False,False]])
    failed,all_confirmed=contact_decision(required,one,one,torch.zeros_like(one))
    assert not failed.any() and not all_confirmed.any()
    failed,all_confirmed=contact_decision(required,required,required,torch.zeros_like(one))
    assert not failed.any() and all_confirmed.all()


def test_wrong_or_lost_contact_is_failure_not_silently_ignored():
    required=torch.tensor([[True,True,False,False,False,False]]);one=torch.tensor([[True,False,False,False,False,False]])
    failed,_=contact_decision(required,one,torch.zeros_like(one),torch.zeros_like(one));assert failed[0,0]
    failed,_=contact_decision(required,one,one,one);assert failed[0,0]


def test_holding_one_limb_leaves_other_targets_and_waist_controlled():
    names=['left_hip_pitch_joint','right_hip_pitch_joint','waist_pitch_joint','left_wrist_pitch_joint','right_elbow_joint']
    mapping=part_joint_mask(names)
    pending=torch.tensor([[True,False,False,False,False,False]])
    assert held_joints(pending,mapping).tolist()==[[True,False,False,False,False]]
    pending[:]=False;pending[:,4]=True
    assert held_joints(pending,mapping).tolist()==[[True,False,False,False,False]]
    pending[:,3]=True
    assert held_joints(pending,mapping).tolist()==[[True,False,False,False,True]]


def test_held_joint_gradient_flows_to_previous_action_only_for_that_limb():
    previous=torch.tensor([[1.,2.]],requires_grad=True);fresh=torch.tensor([[3.,4.]],requires_grad=True)
    action=torch.where(torch.tensor([[True,False]]),previous,fresh);action.sum().backward()
    assert previous.grad.tolist()==[[1.,0.]] and fresh.grad.tolist()==[[0.,1.]]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph test')
def test_gpu_graph_uses_runtime_entry_mask_and_independent_parts():
    s=Sequence([True]*6,[True]*6,'cuda:0');o=s.onset
    wp.load_module(module='supermimic.contact_onset',device=s.device)
    with wp.ScopedCapture(device=s.device) as capture:s.launch()
    wp.to_torch(o.counts).fill_(1)
    wp.get_stream(s.device).wait_stream(wp.stream_from_torch(torch.cuda.current_stream()));wp.capture_launch(capture.graph);wp.synchronize_device(s.device)
    assert not wp.to_torch(o.pending).any()
    o.begin_stage(torch.tensor([0],device=s.device),torch.zeros(1,6,dtype=torch.bool,device=s.device),torch.ones(1,6,dtype=torch.bool,device=s.device))
    wp.get_stream(s.device).wait_stream(wp.stream_from_torch(torch.cuda.current_stream()));wp.capture_launch(capture.graph);wp.synchronize_device(s.device)
    assert wp.to_torch(o.pending).all()
