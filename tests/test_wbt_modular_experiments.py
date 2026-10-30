"""Contracts at the boundary between expert targets and measured student inputs."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_values.wbt.g1.modular import wbt_expert_sparse, wbt_student
from holosoma.utils.eval_utils import _experiment_config_from_serialized
from holosoma.utils.wbt_experiment import PARTS, REFERENCE_SCHEMA, assert_student_observations, load_contact_reference
from somaforge_core.contact_face_selection import FACE_POLICY
from somaforge_core.robot_assets import canonical_g1_asset_metadata


def test_manager_blocks_and_routed_settings_survive_checkpoint_roundtrip():
    from holosoma.config_types.algo import TeacherActorConfig
    cfg = replace(wbt_student, algo=replace(wbt_student.algo,
        config=replace(wbt_student.algo.config,
            distill=replace(wbt_student.algo.config.distill,
                teachers=(TeacherActorConfig("expert.pt", "manifest.json", ("expert_0_actor_obs",)),)))))
    restored = _experiment_config_from_serialized(cfg.to_serializable_dict())
    assert restored.algo.config.distill.teachers[0].observation_keys == ("expert_0_actor_obs",)
    assert restored.algo.config.distill.probes_per_motion == 1
    # Parameter dictionaries serialize tuples as JSON arrays; compare the
    # persisted representation rather than incidental Python container types.
    assert json.dumps(restored.to_serializable_dict()["observation"], sort_keys=True) == json.dumps(cfg.to_serializable_dict()["observation"], sort_keys=True)


def test_target_leak_is_rejected_even_under_an_innocent_term_name():
    assert_student_observations(wbt_student)
    group = wbt_student.observation.groups["actor_obs"]
    group = replace(group, terms={**group.terms, "measured": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b")})
    cfg = replace(wbt_student, observation=replace(wbt_student.observation,
        groups={**wbt_student.observation.groups, "actor_obs": group}))
    with pytest.raises(ValueError, match="not an approved measured observation"):
        assert_student_observations(cfg)


def test_reference_and_reward_blocks_can_be_composed_independently():
    from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_native_root_contact_reward
    cfg = replace(wbt_expert_sparse, reward=g1_29dof_wbt_native_root_contact_reward)
    assert cfg.algo.config.module_dict.actor.input_dim == ["actor_obs", "actor_proprio_history"]
    assert "motion_relative_body_position_error_exp" not in cfg.reward.terms
    assert cfg.reward.terms["native_contact_match"].weight == 10.


def test_student_evaluation_needs_no_teacher_groups_or_checkpoint_files():
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.config_types.observation import ObsGroupCfg
    group = ObsGroupCfg(terms={"forbidden": ObsTermCfg(func="missing.expert:function")})
    cfg = replace(wbt_student, observation=replace(wbt_student.observation,
        groups={**wbt_student.observation.groups, "expert_0_actor_obs": group}))
    evaluation = cfg.get_eval_config(EvaluationConfig(num_envs=1))
    assert not evaluation.algo.config.distill.enable_kl
    assert evaluation.algo.config.distill.teacher_obs_keys is None
    assert not any(key.startswith("expert_") for key in evaluation.observation.groups)
    assert_student_observations(evaluation)


def _metadata():
    return dict(schema=REFERENCE_SCHEMA, face_selection=FACE_POLICY, parts=PARTS,
        sampling="last_native_solve_before_policy_observation",
        activation="CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc",
        robot_asset=canonical_g1_asset_metadata())


def test_contact_activation_does_not_require_positive_force(tmp_path):
    file = tmp_path / "contact.npz"
    np.savez(file, observed_contact_active=np.ones((2, 6), bool),
        observed_contact_force_w=np.zeros((2, 6, 3), np.float32),
        observed_contact_provenance_json=np.asarray(json.dumps(_metadata())))
    active, force, _ = load_contact_reference(file)
    assert active.all() and not force.any()


def test_legacy_force_threshold_labels_are_not_accepted(tmp_path):
    file = tmp_path / "contact.npz"
    meta = {**_metadata(), "activation": "force_gt_10N"}
    np.savez(file, observed_contact_active=np.ones((2, 6), bool),
        observed_contact_force_w=np.zeros((2, 6, 3), np.float32),
        observed_contact_provenance_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match="not native constraint truth"):
        load_contact_reference(file)


def test_reset_pose_does_not_reuse_previous_episode_contact_force():
    from holosoma.utils.wbt_contacts import MeasuredContacts
    from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
    reader = MeasuredContacts.__new__(MeasuredContacts)
    reader.model = object()
    root_state, joint_pos = torch.zeros(2, 13), torch.zeros(2, 29)
    cache = {("wbt_measured_contacts", id(reader.model)): {
        "active": torch.ones(2, 6, dtype=torch.bool),
        "force_w": torch.ones(2, 6, 3), "snapshot": {}, "excluded_rows": [],
        "solved_root_state": root_state.clone(), "solved_dof_pos": joint_pos.clone(),
    }}
    reader.env = SimpleNamespace(_holosoma_step_cache=cache,
        simulator=SimpleNamespace(robot_root_states=RootStatesProxy(root_state), dof_pos=joint_pos),
        episode_length_buf=torch.tensor([0, 1]))
    data = reader.read()
    assert not data["known"][0].any() and data["known"][1].all()
    assert not data["active"][0].any() and not data["force_w"][0].any()
    assert data["active"][1].all() and data["force_w"][1].all()

    # Local window resets keep episode_length_buf > 0. Both root-only and
    # joint-only teleports must invalidate the cached solve for that environment.
    for tensor in (root_state, joint_pos):
        tensor[1, 0] = 1.
        stale = reader.read()
        assert not stale["known"][1].any()
        assert not stale["active"][1].any() and not stale["force_w"][1].any()
        tensor[1, 0] = 0.
    assert reader.read()["known"][1].all()


def test_wbt_samples_requested_contacts_before_reset_callbacks():
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager

    calls = []
    env = SimpleNamespace(base_quat=torch.zeros(2, 4),
        simulator=SimpleNamespace(base_quat=torch.ones(2, 4)))
    WholeBodyTrackingManager._pre_compute_observations_callback(env)
    assert env.base_quat.all()
    env._wbt_measured_contact_reader = SimpleNamespace(read=lambda: calls.append("solve"))
    WholeBodyTrackingManager._pre_compute_observations_callback(env)
    assert calls == ["solve"]


def test_contact_force_rotation_handles_all_environment_part_axes():
    from holosoma.managers.observation.terms.wbt_experiment import force_in_torso_yaw_frame
    force = torch.zeros(2, 6, 3)
    force[..., 0] = 1.
    angle = torch.tensor(torch.pi / 4)
    quaternion = torch.tensor([[0., 0., 0., 1.], [0., 0., angle.sin(), angle.cos()]])
    result = force_in_torso_yaw_frame(force, quaternion)
    assert torch.allclose(result[0], force[0])
    assert torch.allclose(result[1, :, 0], torch.zeros(6), atol=1e-6)
    assert torch.allclose(result[1, :, 1], -torch.ones(6), atol=1e-6)


def test_spatial_index_preserves_native_finite_face_choice_including_edges():
    from somaforge_core.native_surface_index import NativeSurfaceIndex
    from somaforge_core.newton_contacts import actual_primary_face_indices
    faces = [
        dict(surface=10, normal_w=[0., 0., 1.], triangles_w=[[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]),
        dict(surface=2, normal_w=[1., 0., 0.], triangles_w=[[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),
        dict(surface=30, normal_w=[0., 0., 1.], triangles_w=[[[10., 0., 2.], [11., 0., 2.], [10., 1., 2.]]]),
    ]
    rng = np.random.default_rng(42)
    points = np.vstack((rng.uniform(-2, 12, (128, 3)), [[0., 0., 0.], [0., .2, 0.], [.5, .5, 0.]]))
    normals = np.tile([0., 0., 1.], (len(points), 1))
    expected = actual_primary_face_indices(points, normals, faces)
    assert np.array_equal(NativeSurfaceIndex(faces).query(points, normals), expected)


def test_mixed_teacher_widths_route_without_padding_or_shared_normalization(tmp_path):
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    from holosoma.agents.ppo.ppo import EmpiricalNormalization
    from holosoma.agents.ppo.routed_distill_ppo import RoutedTeacherBank
    from holosoma.config_types.algo import TeacherActorConfig
    from holosoma.config_values.wbt.g1.modular import wbt_expert_dense

    specs = []
    expected = []
    dimensions = {}
    histories = {}
    for index, width in enumerate([3, 5]):
        cfg = wbt_expert_dense
        actor_cfg = replace(cfg.algo.config.module_dict.actor,
            layer_config=replace(cfg.algo.config.module_dict.actor.layer_config, hidden_dims=[4]))
        cfg = replace(cfg, algo=replace(cfg.algo, config=replace(cfg.algo.config,
            empirical_normalization=True, module_dict=replace(cfg.algo.config.module_dict, actor=actor_cfg))))
        actor = setup_ppo_actor_module({"actor_obs": width}, actor_cfg, 29, .2, "cpu", {"actor_obs": 1})
        normalizer = EmpiricalNormalization(shape=width, device="cpu")
        normalizer._mean.fill_(index + 1)
        normalizer.eval()
        filename = tmp_path / f"expert_{index}.pt"
        torch.save(dict(experiment_config=cfg.to_serializable_dict(),
            actor_model_state_dict=actor.state_dict(), actor_obs_normalizer_state_dict=normalizer.state_dict()), filename)
        key = f"expert_{index}_actor_obs"
        specs.append(TeacherActorConfig(str(filename), "unused.json", (key,)))
        dimensions[key] = width
        histories[key] = 1
        with torch.no_grad():
            expected.append(actor.act_inference({"actor_obs": normalizer(torch.ones(1, width), update=False)}))
    bank = RoutedTeacherBank(specs, dimensions, histories, 29, "cpu")
    bank.act({"actor_obs": torch.cat((torch.ones(2, 8), torch.tensor([[0.], [1.]])), dim=1)})
    assert torch.allclose(bank.action_mean, torch.cat(expected))
    assert not any(p.requires_grad for p in bank.parameters())
