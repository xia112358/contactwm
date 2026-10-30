from collections import deque
from dataclasses import dataclass
from enum import Enum
import io
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp

from holosoma.utils import training_state as state
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.experiment import TrainingConfig
from holosoma.agents.modules.logging_utils import LoggingHelper


def test_state_roundtrip_restores_mutable_buffers_history_counters_and_generators():
    value=SimpleNamespace(tensor=torch.arange(6).reshape(2,3),array=np.arange(4),cursor=3,
        history=deque([torch.ones(2),torch.ones(2)*2],maxlen=3),
        pending={'next':['collect',8]},generator=torch.Generator().manual_seed(41))
    snapshot=state.freeze(value)
    expected=torch.rand(3,generator=value.generator)
    pointer=value.tensor.data_ptr()
    value.tensor.fill_(99);value.array.fill(99);value.cursor=100
    value.history.clear();value.pending['next']=['wrong',9]
    state.restore(snapshot,value)
    assert value.tensor.data_ptr()==pointer
    assert value.tensor.tolist()==[[0,1,2],[3,4,5]] and value.array.tolist()==[0,1,2,3]
    assert value.cursor==3 and value.history.maxlen==3 and len(value.history)==2
    assert value.pending['next']==['collect',8]
    torch.testing.assert_close(torch.rand(3,generator=value.generator),expected)


def test_rng_roundtrip_restores_python_numpy_and_torch():
    original=state.capture_rng()
    try:
        random.seed(14);np.random.seed(14);torch.manual_seed(14)
        saved=state.capture_rng()
        expected=(random.random(),np.random.random(),torch.rand(3))
        state.restore_rng(saved)
        actual=(random.random(),np.random.random(),torch.rand(3))
        assert actual[:2]==expected[:2]
        torch.testing.assert_close(actual[2],expected[2])
    finally:state.restore_rng(original)


def test_changed_tensor_layout_fails_closed():
    with pytest.raises(ValueError,match='layout'):
        state.restore(state.freeze(torch.zeros(4)),torch.zeros(3))


def test_frozen_training_config_restores_from_serialized_checkpoint_without_mutation():
    source = TrainingConfig(headless=True, num_envs=512, seed=42)
    snapshot = state.freeze(SimpleNamespace(training_config=source, cursor=2))
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    current = TrainingConfig(headless=False, num_envs=512, seed=99)
    target = SimpleNamespace(training_config=current, cursor=0)
    state.restore(saved, target)
    assert target.training_config == source
    assert target.cursor == 2
    assert current.headless is False and current.seed == 99


def test_frozen_object_keeps_mutable_buffer_addresses():
    @dataclass(frozen=True)
    class FrozenRuntime:
        buffer: torch.Tensor
        cursor: int

    FrozenRuntime.__module__ = "holosoma.tests"
    value = FrozenRuntime(torch.arange(3), 7)
    snapshot = state.freeze(value)
    current = FrozenRuntime(torch.zeros(3, dtype=torch.int64), 0)
    pointer = current.buffer.data_ptr()
    restored = state.restore(snapshot, current)
    assert restored.buffer.data_ptr() == pointer
    assert restored.buffer.tolist() == [0, 1, 2]
    assert restored.cursor == 7 and current.cursor == 0


def test_frozen_config_restore_rejects_different_runtime_type():
    with pytest.raises(ValueError, match="Missing or different runtime object"):
        state.restore(state.freeze(TrainingConfig()), SimpleNamespace())


def test_warp_struct_roundtrip_uses_stable_type_and_layout():
    @wp.struct
    class Sample:
        value: wp.float32

    value = wp.array(np.array([(3.0,)], dtype=np.dtype(Sample.numpy_dtype())), dtype=Sample, device="cpu")
    snapshot = state.freeze(value)
    current = wp.zeros(1, dtype=Sample, device="cpu")
    pointer = current.ptr
    state.restore(snapshot, current)
    assert current.ptr == pointer and current.numpy()["value"].tolist() == [3.0]
    # The original checkpoint format stored a different process address.
    legacy = {key: child for key, child in snapshot.items() if key != "dtype_descriptor"}
    legacy["dtype"] = "<warp._src.codegen.Struct object at 0x1234>"
    state.restore(legacy, current)
    assert current.numpy()["value"].tolist() == [3.0]


def test_warp_struct_rejects_different_layout_and_different_type():
    @wp.struct
    class FloatSample:
        value: wp.float32

    @wp.struct
    class OtherFloatSample:
        value: wp.float32

    @wp.struct
    class IntSample:
        value: wp.int32

    snapshot = state.freeze(wp.zeros(1, dtype=FloatSample, device="cpu"))
    with pytest.raises(ValueError, match="Warp layout differs"):
        state.restore(snapshot, wp.zeros(1, dtype=OtherFloatSample, device="cpu"))
    legacy = {key: child for key, child in snapshot.items() if key != "dtype_descriptor"}
    legacy["dtype"] = "<warp._src.codegen.Struct object at 0x1234>"
    with pytest.raises(ValueError, match="Warp layout differs"):
        state.restore(legacy, wp.zeros(1, dtype=IntSample, device="cpu"))


@pytest.mark.parametrize("cache_populated", [True, False])
def test_immutable_proxy_array_restores_without_breaking_torch_warp_alias(cache_populated):
    location = Path(next(iter(importlib.util.find_spec("isaaclab").submodule_search_locations)))
    spec = importlib.util.spec_from_file_location(
        "isaaclab.utils.warp.proxy_array", location / "utils/warp/proxy_array.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.ProxyArray(wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"))
    if cache_populated:
        _ = source.torch
    snapshot = state.freeze(source)
    current = module.ProxyArray(wp.zeros(3, dtype=wp.float32, device="cpu"))
    pointer = current.warp.ptr
    assert state.restore(snapshot, current) is current
    assert current.warp.ptr == pointer
    assert current.torch.data_ptr() == wp.to_torch(current.warp).data_ptr()
    assert current.torch.tolist() == [1.0, 2.0, 3.0]
    current.torch[0] = 7.0
    assert current.warp.numpy()[0] == 7.0


def test_episode_report_batch_can_change_without_resizing_physics_buffers():
    class Environment:
        pass

    Environment.__module__ = "holosoma.envs.tests"
    original = Environment()
    original.positions = torch.arange(6).reshape(2, 3)
    original.extras = {"episode": {"reward": torch.ones(1)}}
    current = Environment()
    current.positions = torch.zeros((2, 3), dtype=torch.int64)
    current.extras = {"episode": {"reward": torch.zeros(2)}}
    pointer = current.positions.data_ptr()
    snapshot = state.freeze(original)
    state.restore(snapshot, current)
    assert current.extras["episode"]["reward"].shape == (1,)
    assert current.positions.data_ptr() == pointer
    assert current.positions.tolist() == [[0, 1, 2], [3, 4, 5]]
    current.positions = torch.zeros((4, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="tensor layout differs"):
        state.restore(snapshot, current)


def test_logging_resume_keeps_new_output_directory_and_restores_progress():
    original = LoggingHelper.__new__(LoggingHelper)
    original.log_dir = "original_run"
    original.tot_timesteps = 24576
    original.rewbuffer = deque([1.0, 2.0], maxlen=100)
    current = LoggingHelper.__new__(LoggingHelper)
    current.log_dir = "resumed_run"
    current.tot_timesteps = 0
    current.rewbuffer = deque(maxlen=100)
    state.restore(state.freeze(original), current)
    assert current.log_dir == "resumed_run"
    assert current.tot_timesteps == 24576
    assert list(current.rewbuffer) == [1.0, 2.0]


def test_native_mesh_handles_stay_in_current_process_while_physics_state_restores():
    import newton

    original = newton.Model.__new__(newton.Model)
    original.shape_source_ptr = wp.array([1234], dtype=wp.uint64, device="cpu")
    original.body_mass = wp.array([3.0], dtype=wp.float32, device="cpu")
    current = newton.Model.__new__(newton.Model)
    current.shape_source_ptr = wp.array([9876], dtype=wp.uint64, device="cpu")
    current.body_mass = wp.zeros(1, dtype=wp.float32, device="cpu")
    pointer = current.shape_source_ptr.ptr
    snapshot = state.freeze(original)
    state.restore(snapshot, current)
    assert current.shape_source_ptr.ptr == pointer
    assert current.shape_source_ptr.numpy().tolist() == [9876]
    assert current.body_mass.numpy().tolist() == [3.0]
    current.shape_source_ptr = wp.zeros(2, dtype=wp.uint64, device="cpu")
    with pytest.raises(ValueError, match="Geometry handle layout differs"):
        state.restore(snapshot, current)


def test_resource_mapping_keys_survive_actual_torch_serialization_and_restore():
    class Frequency(Enum):
        JOINT=1
    module=ModuleType('native_kernel_module')
    value={module:torch.arange(3),(module,'cursor'):7,'ordinary':torch.ones(2),Frequency.JOINT:3}
    snapshot=state.freeze(value)
    stream=io.BytesIO();torch.save(snapshot,stream);stream.seek(0)
    saved=torch.load(stream,weights_only=False)
    recreated=ModuleType('native_kernel_module')
    target={recreated:torch.zeros(3,dtype=torch.int64),(recreated,'cursor'):0,'ordinary':torch.zeros(2),Frequency.JOINT:0}
    pointer=target[recreated].data_ptr()
    state.restore(saved,target)
    assert target[recreated].data_ptr()==pointer
    assert target[recreated].tolist()==[0,1,2] and target[(recreated,'cursor')]==7
    assert target['ordinary'].tolist()==[1.,1.]
    assert target[Frequency.JOINT]==3
    with pytest.raises(ValueError,match='Missing initialized resource key'):
        state.restore(saved,{})


def test_unsupported_mapping_keys_fail_before_checkpoint_publication():
    with pytest.raises(ValueError,match='Unsupported runtime mapping key'):
        state.freeze({object():torch.ones(2)})


def test_project_checkpoint_loader_accepts_numpy_runtime_and_rng_state():
    stream=io.BytesIO()
    torch.save(dict(training_state=dict(rng=state.capture_rng(),buffer=np.arange(3))),stream)
    stream.seek(0)
    ppo=PPO.__new__(PPO);ppo.device='cpu';ppo._uses_g1_asset=Mock(return_value=False)
    loaded=ppo._load_checked_checkpoint(stream)
    assert loaded['training_state']['buffer'].tolist()==[0,1,2]


@pytest.mark.parametrize("full_checkpoints", [False, True])
@pytest.mark.parametrize("historical_runtime", [False, True])
def test_ordinary_checkpoint_roundtrip_restores_training_without_native_runtime(
        monkeypatch, full_checkpoints, historical_runtime):
    from holosoma.agents.ppo.ppo import EmpiricalNormalization

    def make_agent():
        ppo = PPO.__new__(PPO)
        ppo.config = SimpleNamespace(full_checkpoints=full_checkpoints, warm_start=False, load_optimizer=True)
        ppo.actor = torch.nn.Linear(3, 2)
        ppo.critic = torch.nn.Linear(3, 1)
        ppo.actor_optimizer = torch.optim.Adam(ppo.actor.parameters(), lr=.001)
        ppo.critic_optimizer = torch.optim.Adam(ppo.critic.parameters(), lr=.002)
        ppo.empirical_normalization = True
        ppo.actor_obs_normalizer = EmpiricalNormalization((3,), "cpu")
        ppo.critic_obs_normalizer = EmpiricalNormalization((3,), "cpu")
        return ppo

    monkeypatch.setattr(state, "capture", Mock(side_effect=AssertionError("Must not capture native runtime")))
    source = make_agent()
    observations = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    for network, optimizer in ((source.actor, source.actor_optimizer), (source.critic, source.critic_optimizer)):
        network(observations).square().mean().backward()
        optimizer.step()
    source.actor_obs_normalizer(observations)
    source.critic_obs_normalizer(observations + 1)
    source.current_learning_iteration = 8800
    source._checkpoint_metadata = lambda **kwargs: {"iteration": kwargs["iteration"]}
    source._collect_env_state = lambda: {"motion_sampler": {"failure_counts": torch.tensor([2., 4.])}}
    stream = io.BytesIO()
    source.logging_helper = SimpleNamespace(save_checkpoint_artifact=lambda payload, path: torch.save(payload, stream))
    source.save("ordinary.pt", infos={"ok": True})
    stream.seek(0)
    payload = torch.load(stream, weights_only=False)
    assert "training_state" not in payload and "checkpoint_phase" not in payload
    assert payload["iter"] == payload["iteration"] == 8800
    if historical_runtime:
        payload["training_state"] = {"obsolete_native_runtime": True}
    target = make_agent()
    target._load_checked_checkpoint = Mock(return_value=payload)
    assert target.load("ordinary.pt") == {"ok": True}
    assert target.current_learning_iteration == 8801
    assert target._pending_full_runtime_state is None
    assert target.actor_learning_rate == .001 and target.critic_learning_rate == .002
    assert target._pending_env_state["motion_sampler"]["failure_counts"].tolist() == [2., 4.]
    for name in ("actor", "critic", "actor_obs_normalizer", "critic_obs_normalizer"):
        for key, tensor in getattr(source, name).state_dict().items():
            torch.testing.assert_close(tensor, getattr(target, name).state_dict()[key])
    for name in ("actor_optimizer", "critic_optimizer"):
        expected = getattr(source, name).state_dict()
        actual = getattr(target, name).state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        for parameter, fields in expected["state"].items():
            for key, value in fields.items():
                torch.testing.assert_close(actual["state"][parameter][key], value)


def test_periodic_and_final_checkpoints_keep_original_cadence_without_initial_snapshot(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(state, "restore_runtime", Mock(side_effect=AssertionError("Must not restore native runtime")))
    ppo = PPO.__new__(PPO)
    ppo.config = SimpleNamespace(full_checkpoints=True, init_at_random_ep_len=False,
        num_learning_iterations=3, save_interval=2, export_onnx=False)
    ppo.env = SimpleNamespace(reset_all=lambda: {"actor_obs": torch.ones(2, 3)})
    ppo._train_mode = Mock()
    ppo._restore_pending_env_state_after_reset = Mock()
    ppo._pending_full_runtime_state = {"obsolete": True}
    ppo.device = "cpu"
    ppo.is_main_process = True
    ppo.is_multi_gpu = False
    ppo.current_learning_iteration = 5
    ppo.log_dir = "run"
    ppo.logging_helper = SimpleNamespace(record_collection_time=nullcontext, record_learn_time=nullcontext)
    ppo._rollout_step = lambda observations: observations
    ppo._training_step = lambda: {}
    ppo._post_epoch_logging = Mock()
    ppo.save = Mock()
    ppo.learn()
    assert [Path(call.args[0]).name for call in ppo.save.call_args_list] == ["model_00006.pt", "model_00007.pt"]


def test_explicit_warm_start_does_not_inherit_old_sampler_or_rng():
    ppo=PPO.__new__(PPO)
    ppo.config=SimpleNamespace(full_checkpoints=True,warm_start=True,load_optimizer=False)
    ppo.actor=Mock();ppo.critic=Mock();ppo.empirical_normalization=False
    ppo._load_checked_checkpoint=Mock(return_value=dict(iter=8800,actor_model_state_dict={},
        critic_model_state_dict={},env_state={'obsolete_sampler':True},training_state={'obsolete_rng':True}))
    ppo.load('base.pt')
    assert ppo.current_learning_iteration==8801
    assert ppo._pending_env_state is None and ppo._pending_full_runtime_state is None


def test_resume_state_retains_empty_rollout_pool_and_next_update(monkeypatch):
    from holosoma.agents.modules.data_utils import RolloutStorage
    storage=RolloutStorage(2,3);storage.register('actions',(2,));storage._buffers['actions'].fill_(4)
    env=SimpleNamespace(num_envs=2,command_manager=SimpleNamespace(get_state=lambda name:object()),
        episode_length_buf=torch.tensor([2,4]),simulator=SimpleNamespace(positions=torch.ones(2,3)))
    roots={'state':SimpleNamespace(joint_q=torch.arange(4))}
    monkeypatch.setattr(state,'native_roots',lambda:roots)
    ppo=SimpleNamespace(storage=storage,_unwrap_env=lambda:env,device='cpu',
        actor_learning_rate=.001,critic_learning_rate=.002,logging_helper=SimpleNamespace(tot_timesteps=48))
    snapshot=state.capture(ppo,{'actor_obs':torch.ones(2,4)},8801)
    assert snapshot['next_iteration']==8801 and snapshot['pending_minibatches']==[]
    assert snapshot['pending_plan']['storage_step']==0
    assert snapshot['storage']['fields']['_buffers']['items']['actions']['data'].shape==(3,2,2)
    storage.step=1
    with pytest.raises(ValueError,match='empty rollout cursor'):state.capture(ppo,{},8801)
