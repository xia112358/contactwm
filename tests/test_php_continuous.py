"""No physical resets across action boundaries; whole-head statistics only."""
from types import SimpleNamespace as State

import pytest
import torch

from holosoma.php_contact.continuous import ContinuousContactCommand, trajectory_complete
from holosoma.php_contact.trajectory_statistics import TrajectoryStatistics
from holosoma.php_contact import segments


def test_twenty_percent_is_the_whole_trajectory_head_not_arbitrary_action_heads():
    cmd=ContinuousContactCommand.__new__(ContinuousContactCommand)
    n=20000;cmd.device='cpu';cmd.motion_ids=torch.zeros(n,dtype=torch.long);cmd.start_probability=.2
    cmd.motion=State(motion_start_idx=torch.tensor([100]),motion_end_idx=torch.tensor([201]))
    with torch.random.fork_rng():
        torch.manual_seed(12)
        local=(cmd.sample_reset_phase(torch.arange(n))*100).long()
    assert .18 < (local==0).float().mean().item() < .22
    assert ((local[local!=0]>=1)&(local[local!=0]<100)).all()


def test_native_continuous_step_switches_only_goal_and_never_calls_reset(monkeypatch):
    cmd=ContinuousContactCommand.__new__(ContinuousContactCommand)
    cmd.segment_mode='continuous';cmd.num_envs=1;cmd.device='cpu'
    cmd.time_steps=torch.tensor([9]);cmd.motion_ids=torch.tensor([0]);cmd.segment_offsets=torch.tensor([0])
    cmd.contact_data={'segment_id':torch.tensor([0]*10+[1]*11)}
    calls=[]
    def native_step(self): self.time_steps.add_(1)
    monkeypatch.setattr(segments.ContactCommand,'step',native_step)
    cmd._set_segments=lambda ids,selected:calls.append(selected.tolist())
    cmd.reset=lambda *_:pytest.fail('Action boundary must not reset state')
    cmd.step()
    assert calls==[[1]] and cmd.time_steps.item()==10
    monkeypatch.setattr(segments,'command',lambda _:cmd)
    assert not segments.segment_boundary(State(num_envs=1,device='cpu')).any()


def test_full_end_requires_the_complete_motion_clock(monkeypatch):
    import holosoma.php_contact.continuous as module
    cmd=State(time_steps=torch.tensor([8,18]),motion_ids=torch.tensor([0,1]),
        motion=State(motion_end_idx=torch.tensor([11,20])))
    monkeypatch.setattr(module,'command',lambda _:cmd)
    assert trajectory_complete(None).tolist()==[False,True]


def test_statistics_keep_complete_contact_success_separate_from_running_to_end(tmp_path):
    stats=TrajectoryStatistics(['a','b'],[('a',0,10),('b',0,10)],'cpu')
    mids=torch.tensor([0,1]);heads=torch.tensor([True,True])
    stats.sampled(mids,heads)
    stats.boundary(mids,heads,torch.tensor([True,False]),torch.tensor([.1,.3]),torch.tensor([.2,.4]))
    stats.finished(mids,mids,heads,torch.tensor([True,False]),torch.ones(2))
    m=stats.metrics()
    assert m['trajectory_head_run_completion_rate']==1
    assert m['trajectory_head_all_boundary_contacts_rate']==.5
    assert m['trajectory_boundary_contact_success_rate']==.5
    assert m['trajectory_boundary_root_error_m']==pytest.approx(.2)
    restored=TrajectoryStatistics(['a','b'],[('a',0,10),('b',0,10)],'cpu')
    restored.load_state_dict(stats.state_dict())
    assert restored.metrics()['trajectory_head_attempts']==0
    assert restored.metrics(True)==stats.metrics(True)
    restored.write_csv(tmp_path,100)
    assert (tmp_path/'trajectory_statistics.csv').exists() and (tmp_path/'trajectory_boundaries.csv').exists()


def test_first_failure_and_unknown_are_preserved_in_checkpoint():
    stats=TrajectoryStatistics(['a'],[('a',0,10)],'cpu')
    ids=torch.tensor([0]);yes=torch.tensor([True]);no=~yes
    stats.sampled(ids,yes)
    stats.boundary(ids,no,yes,torch.tensor([.2]),torch.tensor([.3]))
    stats.finished(ids,ids,no,no,torch.tensor([.5]))
    assert stats.metrics()['trajectory_boundary_unknown_count']==1
    assert stats.metrics()['trajectory_head_first_failures']==1
    saved=stats.state_dict();saved['counts'][0,4]=1
    with pytest.raises(ValueError,match='counts'):
        TrajectoryStatistics(['a'],[('a',0,10)],'cpu').load_state_dict(saved)


def test_native_reference_refresh_does_not_advance_time_or_write_physical_state():
    from holosoma.managers.command.terms.wbt import MotionCommand
    identity=torch.tensor([[0.,0.,0.,1.]])
    reference=torch.tensor([[[0.,0.,1.],[.2,0.,.5]]])
    root=torch.tensor([[10.,20.,1.]])
    cmd=State(time_steps=torch.tensor([442]),
        _env=State(episode_length_buf=torch.tensor([0])),motion_cfg=State(body_names_to_track=['pelvis','foot'],relative_reference_rotation='yaw'),
        root_pos_w=reference[:,0],ref_pos_w=reference[:,0],root_quat_w=identity,ref_quat_w=identity,
        robot_root_pos_w=root,robot_ref_pos_w=root,robot_root_quat_w=identity,robot_ref_quat_w=identity,
        body_pos_w=reference,body_quat_w=identity[:,None,:].expand(-1,2,-1))
    before=root.clone()
    MotionCommand._update_relative_body_targets(cmd)
    torch.testing.assert_close(cmd.body_pos_relative_w,reference+torch.tensor([10.,20.,0.]))
    assert cmd.time_steps.item()==442
    torch.testing.assert_close(root,before)


def test_boundary_is_recorded_before_goal_switch_and_terminal_failure_is_not_success(monkeypatch):
    import holosoma.php_contact.continuous as module
    cmd=ContinuousContactCommand.__new__(ContinuousContactCommand)
    cmd.num_envs=2;cmd.device='cpu';cmd.motion_ids=torch.tensor([0,0]);cmd.time_steps=torch.tensor([9,19])
    cmd.segment_end=torch.tensor([10,20]);cmd.selected_segment=torch.tensor([0,1]);cmd.segment_counts=torch.tensor([2])
    cmd._episode_from_head=torch.tensor([True,True]);cmd._episode_open=torch.tensor([True,True])
    cmd._trajectory_open=torch.tensor([True,True]);cmd._trajectory_all_contacts=torch.tensor([True,True])
    cmd._trajectory_observed=torch.tensor([0,0]);cmd._last_boundary=torch.tensor([-1,-1])
    cmd._last_boundary_root_error=torch.zeros(2);cmd._last_boundary_limb_error=torch.zeros(2)
    cmd._endpoint_body_ids=torch.arange(6);cmd._body_indexes_in_motion=torch.arange(6)
    cmd.motion=State(motion_start_idx=torch.tensor([0]),motion_end_idx=torch.tensor([21]),
                     body_pos_w=torch.zeros(21,6,3))
    monkeypatch.setattr(ContinuousContactCommand,'_reference_env_origins',property(lambda _:torch.zeros(2,3)))
    monkeypatch.setattr(ContinuousContactCommand,'robot_body_pos_w',property(lambda _:torch.zeros(2,6,3)))
    cmd.goal_active=torch.zeros(2,6,dtype=torch.bool)
    term=State(terminated=torch.tensor([False,True]),time_outs=torch.tensor([False,True]),
               term_dones={'motion_ends':torch.tensor([False,True])})
    cmd.env=State(termination_manager=term,episode_length_buf=torch.tensor([10,20]),
                  simulator=State(robot_root_states=torch.zeros(2,13)))
    cmd.trajectory_statistics=TrajectoryStatistics(['a'],[('a',0,10),('a',10,20)],'cpu')
    cmd.trajectory_statistics.sampled(cmd.motion_ids,cmd._episode_from_head)
    monkeypatch.setattr(module.ContactCommand,'update_metrics',lambda _:None)
    monkeypatch.setattr(module,'measured_contacts',lambda _:dict(active=cmd.goal_active.clone(),
        known=torch.ones(2,6,dtype=torch.bool)))
    cmd.update_metrics();cmd.update_metrics()
    assert cmd.trajectory_statistics.boundaries[:,0].tolist()==[1,1]
    assert cmd.trajectory_statistics.boundaries[:,1].tolist()==[1,0]
    assert cmd.trajectory_statistics.metrics()['trajectory_head_attempts']==1
    # First episode continues in the same physical state into its final action.
    cmd.time_steps[0]=19;cmd.segment_end[0]=20;cmd.selected_segment[0]=1
    term.time_outs[0]=True;term.term_dones['motion_ends'][0]=True
    cmd.update_metrics()
    assert cmd.trajectory_statistics.counts[0,2:].tolist()==[2,1,1]
    assert cmd._trajectory_observed.tolist()==[2,1]
    assert not cmd._trajectory_open.any()
