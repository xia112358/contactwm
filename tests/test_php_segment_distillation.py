"""Independent segment episodes, goal locking, teacher context and contact rewards."""
from types import SimpleNamespace as State

import numpy as np
import pytest
import torch

from holosoma.php_contact import segments
from holosoma.php_contact.prepare_distill import selected_clock


def segment_command():
    cmd = segments.SegmentCommand.__new__(segments.SegmentCommand)
    cmd.device, cmd.num_envs = 'cpu', 2
    starts, ends = torch.tensor([0,5]), torch.tensor([5,10])
    target = torch.tensor([2,2,4,4,4,7,7,9,9,9])
    ids = torch.tensor([0,0,1,1,1,0,0,1,1,1])
    active = torch.zeros(10,6,dtype=torch.bool)
    active[:2,0] = True
    active[2:,1] = True
    cmd.contact_data = dict(segment_id=ids, target_frame=target, target_active=active,
        target_position_w=torch.arange(180,dtype=torch.float).reshape(10,6,3))
    cmd.bounds, cmd.segment_offsets, cmd.segment_counts = segments.episode_bounds(starts,ends,ids,target)
    cmd.motion = State(motion_start_idx=starts,motion_end_idx=ends)
    cmd.motion_ids = torch.tensor([0,1])
    cmd.selected_segment = torch.zeros(2,dtype=torch.long)
    cmd.segment_start = torch.zeros(2,dtype=torch.long)
    cmd.segment_end = torch.zeros(2,dtype=torch.long)
    cmd.goal_active = torch.zeros(2,6,dtype=torch.bool)
    cmd.goal_position_w = torch.zeros(2,6,3)
    cmd.segment_mode='independent'
    cmd.start_probability=.5
    cmd._episode_from_head=torch.tensor([True,False])
    return cmd


def test_shared_endpoint_keeps_its_own_future_goal_until_episode_reset():
    cmd=segment_command()
    cmd._set_segments(torch.tensor([0]),torch.tensor([0]))
    torch.testing.assert_close(cmd.bounds,torch.tensor([[0,2],[2,4],[5,7],[7,9]]))
    # At frame 2 the stream has switched, but the first episode's goal is frame 2.
    assert cmd.contact_data['target_active'][2,1]
    assert cmd.goal_active[0,0] and not cmd.goal_active[0,1]
    torch.testing.assert_close(cmd.goal_position_w[0],cmd.contact_data['target_position_w'][1])


@pytest.mark.parametrize('probability',[0.,.5,1.])
def test_rsi_never_crosses_selected_segment_or_motion_bounds(probability):
    cmd=segment_command();cmd.start_probability=probability
    ids=torch.arange(2)
    for _ in range(100):
        phase=cmd.sample_reset_phase(ids)
        start=cmd.motion.motion_start_idx[cmd.motion_ids]
        length=cmd.motion.motion_end_idx[cmd.motion_ids]-start-1
        frame=start+(phase*length).long()
        assert ((frame>=cmd.segment_start)&(frame<cmd.segment_end)).all()
        if probability==1:torch.testing.assert_close(frame,cmd.segment_start)


def test_segment_boundary_is_a_timeout_not_an_automatic_contact_success(monkeypatch):
    cmd=segment_command();cmd.segment_end=torch.tensor([4,9]);cmd.time_steps=torch.tensor([3,7])
    monkeypatch.setattr(segments,'command',lambda _:cmd)
    env=State(num_envs=2,device='cpu')
    torch.testing.assert_close(segments.segment_boundary(env),torch.tensor([True,False]))
    cmd.segment_mode='continuous'
    assert not segments.segment_boundary(env).any()


def test_contact_reward_does_not_reward_missing_required_contact_via_inactive_parts():
    desired=torch.tensor([[True,False,False,False,False,False]])
    active=torch.zeros(1,6,dtype=torch.bool);known=torch.ones_like(active)
    assert segments.balanced_contact_score(active,known,desired).item()==0
    active[0,0]=True
    assert segments.balanced_contact_score(active,known,desired).item()==1
    active[0,1]=True
    assert segments.balanced_contact_score(active,known,desired).item()==pytest.approx(.8)
    known.zero_()
    assert segments.balanced_contact_score(active,known,desired).item()==0


def test_terminal_contact_reward_uses_realized_contacts_only_near_endpoint(monkeypatch):
    cmd=segment_command();cmd.segment_end=torch.tensor([10,10]);cmd.time_steps=torch.tensor([2,8])
    cmd.goal_active[:,0]=True
    monkeypatch.setattr(segments,'command',lambda _:cmd)
    data=dict(active=cmd.goal_active.clone(),known=torch.ones(2,6,dtype=torch.bool),force_w=torch.zeros(2,6,3))
    monkeypatch.setattr(segments,'measured_contacts',lambda _:data)
    env=State(log_dict={})
    torch.testing.assert_close(segments.goal_contact_reward(env),torch.tensor([0.,1.]))
    assert env.log_dict['PHPContact/head_terminal_contact_observations']==0
    assert env.log_dict['PHPContact/head_terminal_contact_satisfied']==0
    from holosoma.utils.average_meters import TensorAverageMeterDict
    meter=TensorAverageMeterDict()
    meter.add(env.log_dict)
    assert all(torch.isfinite(value) for value in meter.mean_and_clear().values())
    data['known'].zero_()
    assert not segments.goal_contact_reward(env).any()
    assert env.log_dict['PHPContact/head_terminal_contact_satisfied']==0


def test_teacher_context_shares_physics_but_not_student_reference():
    student_cmd=State(reference='executed/fitted trajectory')
    original_cmd=State(reference='original teacher task')
    env=State(simulator=object(),command_manager=State(get_state=lambda _:student_cmd))
    proxy=segments.TeacherEnvironment(env,original_cmd)
    assert proxy.simulator is env.simulator
    assert proxy.command_manager.get_state('motion_command') is original_cmd
    assert env.command_manager.get_state('motion_command') is student_cmd


def test_extraction_clock_requires_the_retained_representative_and_valid_native_time():
    maps=np.asarray([[0.,1.3,2.,3.],[0.,.7,2.,3.]])
    np.testing.assert_array_equal(selected_clock(maps,[4,9],9,4),[0,1,2,3])
    with pytest.raises(ValueError,match='unique'):
        selected_clock(maps,[4,9],8,4)
    with pytest.raises(ValueError,match='Invalid'):
        selected_clock(np.array([[0.,2.,1.,3.]]),[4],4,4)


def test_student_command_allowlist_does_not_expose_reference_observations_to_other_students():
    from holosoma.utils.wbt_experiment import assert_student_observations, SEGMENT_DISTILL_ALGO, ROUTED_ALGO
    term=State(func='holosoma.php_contact.segments:future_contacts')
    cfg=State(algo=State(_target_=SEGMENT_DISTILL_ALGO,config=State(module_dict=State(actor=State(input_dim=['actor'])))),
        observation=State(groups={'actor':State(terms={'command':term})}))
    assert_student_observations(cfg)
    cfg.algo._target_=ROUTED_ALGO
    with pytest.raises(ValueError):assert_student_observations(cfg)
    cfg.algo._target_=SEGMENT_DISTILL_ALGO
    term.func='holosoma.php_contact.segments:teacher_observation'
    with pytest.raises(ValueError):assert_student_observations(cfg)


def test_segment_rollout_keeps_native_dagger_but_does_not_activate_full_motion_probes(monkeypatch):
    from holosoma.php_contact.distill import SegmentDistillPPO
    from holosoma.agents.ppo.distill_ppo import DistillPPO
    monkeypatch.setattr(DistillPPO,'_rollout_step',lambda self,obs:('dagger',obs))
    algo=SegmentDistillPPO.__new__(SegmentDistillPPO)
    assert algo._rollout_step('student states')==('dagger','student states')


def test_segment_distillation_rejects_unmatched_old_predictor_augmentations(monkeypatch):
    from holosoma.php_contact import dataset
    from holosoma.php_contact.distill import Options,build_config
    monkeypatch.setattr(dataset,'load_prepared',lambda _:{'reference_origin':'old fixed-q augmentation'})
    with pytest.raises(ValueError,match='task-matched'):
        build_config(Options(manifest='old.json',teacher_spec='unused.json'))


def test_dagger_requires_original_teacher_validity_as_well_as_fitted_reference(monkeypatch):
    from holosoma.php_contact.distill import SegmentDistillPPO
    from holosoma.agents.ppo.distill_ppo import DistillPPO
    teacher_cmd=State(tag='original')
    student_cmd=State(tag='fitted',_teacher_view=None)
    env=State(command_manager=State(get_state=lambda _:student_cmd))
    student_cmd._teacher_view=segments.TeacherEnvironment(env,teacher_cmd)
    def valid(self):
        tag=self.env.command_manager.get_state('motion_command').tag
        return torch.tensor([1.,0.,1.]) if tag=='fitted' else torch.tensor([0.,1.,1.])
    monkeypatch.setattr(DistillPPO,'_compute_dagger_valid_mask',valid)
    algo=SegmentDistillPPO.__new__(SegmentDistillPPO);algo.env=env
    mask=algo._compute_dagger_valid_mask()
    assert mask.dtype == torch.float32 and torch.equal(mask,torch.tensor([0.,0.,1.]))
    assert algo.env is env and env.command_manager.get_state('motion_command') is student_cmd


def prepared_fixture(tmp_path, teacher_frames):
    import json
    from holosoma.php_contact.dataset import SCHEMA,digest,PARTS
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    motion=tmp_path/'fitted.npz';reference=tmp_path/'original.npz'
    np.savez(motion,joint_pos=np.zeros((5,36)))
    np.savez(reference,joint_pos=np.ones((5,36)))
    evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    meta=dict(schema=SCHEMA,parts=PARTS,source_sha256=digest(motion),split='train',frame_count=5,
        reference_sampling='fixed_q_forward_no_integration',source_labels=str(evidence),
        labels_sha256=digest(evidence),events=str(evidence),events_sha256=digest(evidence),
        evidence=str(evidence),evidence_sha256=digest(evidence),teacher_reference_sha256=digest(reference))
    contacts=tmp_path/'contacts.npz'
    np.savez(contacts,metadata_json=np.asarray(json.dumps(meta)),reference_active=np.zeros((5,6),bool),
        target_active=np.zeros((5,6),bool),target_position_w=np.zeros((5,6,3)),
        target_frame=np.array([2,2,4,4,4]),segment_id=np.array([0,0,1,1,1]),teacher_frame=teacher_frames)
    manifest=tmp_path/'train.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA,parts=PARTS,split='train',terrains=[],
        newton_runtime={'runtime_id':NEWTON_RUNTIME_ID},reference_sampling='fixed_q_forward_no_integration',
        reference_origin='matched_teacher_existing_extraction_standard',motion_files=[dict(motion_file=str(motion),
            motion_sha256=digest(motion),contact_file=str(contacts),contact_sha256=digest(contacts),height_scale=1.,
            teacher_reference_file=str(reference),teacher_reference_sha256=digest(reference))])))
    return manifest,reference


def test_changed_original_task_cannot_silently_replace_teacher_context(tmp_path):
    from holosoma.php_contact.dataset import load_prepared
    manifest,reference=prepared_fixture(tmp_path,np.arange(5))
    np.savez(reference,joint_pos=np.full((5,36),2.))
    with pytest.raises(ValueError,match='teacher task reference changed'):
        load_prepared(manifest)


def test_teacher_clock_outside_original_task_is_rejected_before_simulator(tmp_path):
    from holosoma.php_contact.dataset import load_prepared
    manifest,_=prepared_fixture(tmp_path,np.array([0,1,2,3,8]))
    with pytest.raises(ValueError,match='frame mapping is invalid'):
        load_prepared(manifest)
