"""Predictor interval identity and contact-conditioned PHP actor contracts."""
from types import SimpleNamespace as State

import numpy as np
import pytest
import torch

from holosoma.php_contact import dataset, terms
from holosoma.php_contact.train import Options, build_config
from somaforge_core.demonstration_events import materialize


def timeline():
    active = np.zeros((5, 6), bool)
    active[:3, 0] = True
    active[2:, 1] = True
    positions = np.zeros((5, 6, 3), np.float32)
    positions[..., 0] = np.arange(5)[:, None]
    return dict(contact_part_mask=active, contact_surface=np.where(active, 0, -1),
                contact_position_w=positions, fps=50.)


def test_shared_event_boundary_switches_command_without_frame_shift_or_cross_clip_target():
    data = timeline()
    events, _ = materialize(data, [dict(start_frame=0, end_frame=2), dict(start_frame=2, end_frame=4)])
    result = dataset.interval_targets(data, events)
    np.testing.assert_array_equal(result['target_frame'], [2, 2, 4, 4, 4])
    np.testing.assert_array_equal(result['segment_id'], [0, 0, 1, 1, 1])
    assert result['target_active'][1, 0] and not result['target_active'][2, 0]
    assert result['target_position_w'][1, 0, 0] == 2
    assert not result['target_position_w'][2, 0].any()


def test_changed_contact_roles_and_noncontiguous_authored_clock_are_rejected():
    data = timeline()
    events, _ = materialize(data, [dict(start_frame=0, end_frame=4)])
    events[0]['target_contact_parts'] = ['left_hand']
    with pytest.raises(ValueError, match='differs'):
        dataset.interval_targets(data, events)
    with pytest.raises(ValueError, match='continuously'):
        dataset.interval_targets(data, [dict(start_frame=0, end_frame=1), dict(start_frame=2, end_frame=4)])


@pytest.mark.parametrize('height,expected',[(.90,'validation'),(.95,'train'),(1.,'train'),(1.05,'train'),(1.1,'test')])
def test_split_matches_frozen_predictor_height_holdouts(height, expected):
    assert dataset.split_name(height) == expected


def test_contact_command_is_yaw_local_translation_invariant_and_occupancy_explicit(monkeypatch):
    active = torch.tensor([[True, False, False, False, False, False]])
    positions = torch.zeros(1, 6, 3)
    positions[0, 0] = torch.tensor([1., 0., 0.])
    cmd = State(time_steps=torch.tensor([0]), _reference_env_origins=torch.tensor([[10., 20., 0.]]),
                contact_data={'target_active':active,'target_position_w':positions})
    monkeypatch.setattr(terms, 'command', lambda _: cmd)
    root = torch.zeros(1, 13)
    root[0, :3] = torch.tensor([10., 20., 0.])
    root[0, 5] = 1.  # 180 degrees yaw, xyzw.
    env = State(simulator=State(robot_root_states=root))
    value = terms.future_contacts(env)
    torch.testing.assert_close(value[0, :3], torch.tensor([-1., 0., 0.]), atol=1e-6, rtol=0)
    assert value.shape == (1, 24) and not value[0, 3:18].any()
    assert value[0, 18] == 1
    root[0, :3] += 100
    cmd._reference_env_origins += 100
    torch.testing.assert_close(terms.future_contacts(env), value)


def test_contact_reward_uses_allocated_native_activation_and_masks_reset_unknowns(monkeypatch):
    reference = torch.tensor([[True, False, False, False, False, False]])
    cmd = State(time_steps=torch.tensor([0]), contact_data={'reference_active':reference})
    monkeypatch.setattr(terms, 'command', lambda _: cmd)
    actual = {'active':reference.clone(), 'known':torch.ones(1,6,dtype=torch.bool),
              'force_w':torch.zeros(1,6,3)}
    monkeypatch.setattr(terms, 'measured_contacts', lambda _: actual)
    env = State(log_dict={})
    assert terms.contact_match(env).item() == 1  # Zero force does not erase activation.
    actual['active'][0,0] = False
    assert terms.contact_match(env).item() == pytest.approx(5/6)
    actual['known'].zero_()
    assert terms.contact_match(env).item() == 0


def test_php_reuses_native_engine_and_budget_without_reference_kinematics_in_actor(monkeypatch):
    monkeypatch.setattr(dataset, 'load_prepared', lambda _: {
        'split':'train', 'motion_files':[{},{}], 'terrain_spacing':[8.,8.]})
    cfg = build_config(Options(manifest='prepared.json'))
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_climb00_php_expert as base
    assert cfg.env_class == base.env_class and cfg.algo._target_ == base.algo._target_
    assert cfg.robot == base.robot and cfg.action == base.action
    assert cfg.training.num_envs == 4096 and cfg.algo.config.num_learning_iterations == 20000
    assert cfg.algo.config.num_steps_per_env == 24 and cfg.algo.config.save_interval == 100
    assert cfg.observation.groups['critic_obs'] == base.observation.groups['critic_obs']
    for group in cfg.algo.config.module_dict.actor.input_dim:
        for term in cfg.observation.groups[group].terms.values():
            assert all(token not in term.func for token in ('motion_ref', 'motion_command', 'reference_force', 'sparse_reference'))
    assert cfg.observation.groups['actor_proprio_history'].history_length == 5
    monkeypatch.setattr(dataset, 'load_prepared', lambda _: {'split':'test'})
    with pytest.raises(ValueError, match='train split'):
        build_config(Options(manifest='test.json'))


def test_prepared_dataset_rejects_modified_motion_before_initializing_simulation(tmp_path):
    import json
    source = tmp_path/'motion.npz'
    source.write_bytes(b'changed')
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    manifest = tmp_path/'train.json'
    manifest.write_text(json.dumps(dict(schema=dataset.SCHEMA, parts=dataset.PARTS,
        newton_runtime={'runtime_id':NEWTON_RUNTIME_ID}, reference_sampling='fixed_q_forward_no_integration', terrains=[],
        motion_files=[dict(motion_file=str(source), motion_sha256='0'*64)])))
    with pytest.raises(ValueError, match='motion_file hash mismatch'):
        dataset.load_prepared(manifest)


def test_prepared_dataset_rejects_changed_terrain_geometry(tmp_path):
    import json
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    terrain = tmp_path/'terrain.obj'
    terrain.write_text('v 0 0 0\n')
    manifest = tmp_path/'train.json'
    manifest.write_text(json.dumps(dict(schema=dataset.SCHEMA, parts=dataset.PARTS,
        newton_runtime={'runtime_id':NEWTON_RUNTIME_ID}, reference_sampling='fixed_q_forward_no_integration',
        terrains=[dict(terrain_file=str(terrain), terrain_sha256='0'*64)], motion_files=[])))
    with pytest.raises(ValueError, match='terrain hash mismatch'):
        dataset.load_prepared(manifest)
