"""Pinned inference must always initialize the selected action's own head."""
from types import SimpleNamespace as State

import pytest
import torch

from holosoma.php_contact.play_segment import PlaybackSegmentCommand, refresh_reset_reference, playback_bounds


def command():
    cmd=PlaybackSegmentCommand.__new__(PlaybackSegmentCommand)
    cmd.cfg=State(params={'playback_segment':1})
    cmd.motion=State(num_motions=1,motion_start_idx=torch.tensor([0]),motion_end_idx=torch.tensor([101]))
    cmd.motion_ids=torch.tensor([0]);cmd.segment_counts=torch.tensor([3])
    cmd.bounds=torch.tensor([[0,30],[30,70],[70,100]])
    cmd.selected_segment=torch.zeros(1,dtype=torch.long)
    cmd.segment_start=torch.zeros(1,dtype=torch.long);cmd.segment_end=torch.zeros(1,dtype=torch.long)
    target=torch.cat((torch.full((30,),30),torch.full((40,),70),torch.full((31,),100)))
    cmd.contact_data=dict(target_frame=target,target_active=torch.zeros(101,6,dtype=torch.bool),
        target_position_w=torch.zeros(101,6,3))
    cmd.contact_data['target_active'][30:70,2]=True
    cmd.goal_active=torch.zeros(1,6,dtype=torch.bool);cmd.goal_position_w=torch.zeros(1,6,3)
    return cmd


def test_every_reset_starts_at_same_selected_segment_and_locks_its_own_goal():
    cmd=command()
    for _ in range(10):
        phase=cmd.sample_reset_phase(torch.tensor([0]))
        assert (phase*100).long().item()==30
        assert cmd.segment_start.item()==30 and cmd.segment_end.item()==70
        assert cmd.goal_active[0,2]


def test_invalid_or_multiple_motion_selection_is_rejected():
    cmd=command();cmd.cfg.params['playback_segment']=3
    with pytest.raises(ValueError,match='outside'):cmd.sample_reset_phase(torch.tensor([0]))
    cmd.motion.num_motions=2
    with pytest.raises(ValueError,match='exactly one'):cmd.sample_reset_phase(torch.tensor([0]))


def test_reset_reference_is_refreshed_at_the_unchanged_frame():
    cmd=State(time_steps=torch.tensor([442]), relative_frame=None)
    def update():
        cmd.relative_frame=int(cmd.time_steps[0])
    cmd._update_relative_body_targets=update
    refresh_reset_reference(cmd)
    assert cmd.time_steps.item()==cmd.relative_frame==442


def test_whole_playback_uses_complete_motion_bounds_after_action_switch():
    cmd=command()
    cmd.segment_start[0]=30;cmd.segment_end[0]=70
    assert playback_bounds(cmd,False)==(30,70)
    assert playback_bounds(cmd,True)==(0,100)
    cmd.segment_start[0]=70;cmd.segment_end[0]=100
    assert playback_bounds(cmd,True)==(0,100)


def test_ood_inference_keeps_actor_and_control_but_removes_teacher(tmp_path):
    from dataclasses import asdict, replace
    from holosoma.config_types.algo import DistillPPOConfig
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_climb00_php_expert as base
    from holosoma.php_contact.play_segment import ood_inference_config
    cfg=replace(base, algo=replace(base.algo, _target_='holosoma.php_contact.distill.SegmentDistillPPO',
                                 config=DistillPPOConfig(**asdict(base.algo.config)) ))
    source=dict(reference_origin='out_of_distribution_native_reference',split='evaluation',terrain_spacing=[9.,10.])
    manifest=tmp_path/'evaluation.json'
    result=ood_inference_config({'experiment_config':asdict(cfg)},manifest,source)
    assert result.algo.config.module_dict.actor==cfg.algo.config.module_dict.actor
    assert result.robot.control==cfg.robot.control
    assert result.observation.groups['actor_obs']==cfg.observation.groups['actor_obs']
    assert not any(name.startswith('expert_') for name in result.observation.groups)
    assert not result.algo.config.distill.enable_kl
    assert result.algo.config.distill.teachers==()
    assert result.algo.config.distill.teacher_obs_keys is None
    assert result.command.setup_terms['motion_command'].params['motion_config'].motion_manifest==str(manifest)
    assert result.terrain.terrain_term.motion_matched_manifest==str(manifest)


def test_ood_manifest_remains_rejected_by_training(monkeypatch):
    from holosoma.php_contact import dataset
    from holosoma.php_contact.train import build_config, Options
    monkeypatch.setattr(dataset,'load_prepared',lambda _:dict(split='evaluation'))
    with pytest.raises(ValueError,match='only.*train split'):
        build_config(Options(manifest='evaluation.json'))


def test_disable_tracking_keeps_reference_endpoint_and_timeout():
    from holosoma.php_contact.play_segment import configure_tracking_termination
    from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_climb00_php_expert as base
    from dataclasses import replace
    terms=dict(base.termination.terms,motion_ends=TerminationTermCfg(func='reference_endpoint',is_timeout=True))
    cfg=replace(base,termination=TerminationManagerCfg(terms=terms))
    result=configure_tracking_termination(cfg,False)
    assert 'bad_tracking' not in result.termination.terms
    assert result.termination.terms['motion_ends']==terms['motion_ends']
    assert result.termination.terms['timeout']==terms['timeout']
    assert 'bad_tracking' in cfg.termination.terms
