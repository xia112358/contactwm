"""Same-iteration PPO runtime snapshots for the pinned Newton WBT backend.

Runtime objects are recreated from the pinned experiment; mutable arrays,
histories, counters and queues are restored in place to retain CUDA graph views.
Opaque resources (functions, loggers, USD handles and compiled graphs) are never
pickled. Missing required Newton roots or changed tensor layouts fail closed.
"""
from collections import deque
from dataclasses import is_dataclass, replace
from enum import Enum
import json
import random
from types import ModuleType

import numpy as np
import torch

SCHEMA = "holosoma_newton_ppo_training_state_v2"
_EXCLUDED = {"env", "unwrapped_env", "cfg", "config", "logger", "writer"}
_OBJECT_MODULES = ("holosoma.", "isaaclab.", "isaaclab_newton.", "newton.", "mujoco_warp.", "types")


def _literal_key(value):
    return type(value) in (type(None), bool, int, float, str, bytes) or (
        type(value) is tuple and all(_literal_key(v) for v in value))


def _key_descriptor(value):
    """Identify runtime resource keys without pickling their module/handle."""
    if type(value) is tuple:
        return dict(kind='tuple',items=[_key_descriptor(v) for v in value])
    if _literal_key(value):
        if isinstance(value,bytes):return dict(kind='bytes',data=value.hex())
        return dict(kind='literal',type=type(value).__name__,data=value)
    if isinstance(value,np.generic):return _key_descriptor(value.item())
    if isinstance(value,Enum):return dict(kind='enum',module=type(value).__module__,
                                         type=type(value).__qualname__,name=value.name)
    if isinstance(value,ModuleType):return dict(kind='module',name=value.__name__)
    if isinstance(value,type):return dict(kind='class',module=value.__module__,name=value.__qualname__)
    if type(value).__module__.startswith('warp.'):
        for attribute in ('key','name'):
            name=getattr(value,attribute,None)
            if isinstance(name,str) and name:
                return dict(kind='resource',type=f'{type(value).__module__}.{type(value).__qualname__}',
                            attribute=attribute,name=name)
    raise ValueError(f'Unsupported runtime mapping key: {type(value).__module__}.{type(value).__qualname__}')


def _key_token(descriptor):
    return json.dumps(descriptor,sort_keys=True)


def _restore_literal_key(descriptor):
    if descriptor['kind']=='literal':return descriptor['data']
    if descriptor['kind']=='bytes':return bytes.fromhex(descriptor['data'])
    if descriptor['kind']=='tuple':return tuple(_restore_literal_key(v) for v in descriptor['items'])
    raise ValueError('Missing initialized resource key in runtime mapping')


def _warp_dtype_descriptor(dtype):
    cls = getattr(dtype, "cls", None)
    if isinstance(cls, type) and callable(getattr(dtype, "numpy_dtype", None)):
        return dict(struct=f"{cls.__module__}.{cls.__qualname__}", layout=np.dtype(dtype.numpy_dtype()))
    return dict(name=str(dtype))


def _warp_dtype_matches(node, dtype):
    descriptor = _warp_dtype_descriptor(dtype)
    if "dtype_descriptor" in node:
        return node["dtype_descriptor"] == descriptor
    if node["dtype"].startswith("<warp.") and ".Struct object at " in node["dtype"]:
        # Legacy snapshots contain the process address rather than the struct
        # identity. Their NumPy record dtype still retains the complete layout.
        return "struct" in descriptor and node["data"].dtype == descriptor["layout"]
    return node["dtype"] == str(dtype)


def freeze(value, seen=None, *, path='runtime'):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        return dict(kind="tensor", data=value.detach().cpu().clone(), device=str(value.device))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject: raise ValueError("Object arrays cannot be checkpointed as mutable training state")
        return dict(kind="numpy", data=value.copy())
    if isinstance(value, torch.Generator):
        return dict(kind="torch_generator",data=value.get_state().cpu())
    if isinstance(value, np.random.Generator):
        return dict(kind="numpy_generator",data=value.bit_generator.state)
    if isinstance(value, random.Random):
        return dict(kind="python_generator",data=value.getstate())
    if type(value).__module__.startswith("warp") and callable(getattr(value, "numpy", None)):
        return dict(kind="warp", data=value.numpy().copy(), dtype=str(value.dtype),
                    dtype_descriptor=_warp_dtype_descriptor(value.dtype))
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        return dict(kind="scalar", data=value)
    if id(value) in seen: return dict(kind="resource", type="existing_alias")
    seen.add(id(value))
    if isinstance(value, dict):
        if all(_literal_key(k) for k in value):
            return dict(kind="dict", items={k: freeze(v, seen,path=f'{path}[{k!r}]') for k,v in value.items()})
        entries=[]
        for index,(key,child) in enumerate(value.items()):
            try:descriptor=_key_descriptor(key)
            except ValueError as error:raise ValueError(f'{error} at {path}, key {index}') from error
            entries.append(dict(key=descriptor,value=freeze(child,seen,path=f'{path}.value[{index}]')))
        tokens=[_key_token(e['key']) for e in entries]
        if len(set(tokens))!=len(tokens):raise ValueError('Ambiguous runtime resource mapping keys')
        return dict(kind='resource_dict',entries=entries)
    if isinstance(value, (list, tuple, deque)):
        return dict(kind=type(value).__name__, items=[freeze(v, seen,path=f'{path}[{i}]') for i,v in enumerate(value)],
                    maxlen=value.maxlen if isinstance(value, deque) else None)
    if hasattr(value, "__dict__") and type(value).__module__.startswith(_OBJECT_MODULES):
        return dict(kind="object", type=f"{type(value).__module__}.{type(value).__qualname__}",
                    fields={k: freeze(v, seen,path=f'{path}.{k}') for k,v in vars(value).items() if k not in _EXCLUDED})
    return dict(kind="resource", type=f"{type(value).__module__}.{type(value).__qualname__}")


def restore(node, target, *, device="cpu", path="runtime"):
    kind = node["kind"]
    if kind == "resource": return target
    if kind == "scalar": return node["data"]
    if kind == "torch_generator":
        if target is None: raise ValueError("Missing initialized Torch generator")
        target.set_state(node["data"].cpu()); return target
    if kind == "numpy_generator":
        if target is None: raise ValueError("Missing initialized NumPy generator")
        target.bit_generator.state=node["data"]; return target
    if kind == "python_generator":
        if target is None: raise ValueError("Missing initialized Python generator")
        target.setstate(node["data"]); return target
    if kind == "tensor":
        data = node["data"]
        if target is None: return data.to(device).clone()
        if not isinstance(target, torch.Tensor) or target.shape != data.shape or target.dtype != data.dtype:
            raise ValueError(
                f"Checkpoint tensor layout differs at {path}: saved={data.shape}, {data.dtype}; "
                f"initialized={getattr(target, 'shape', None)}, {getattr(target, 'dtype', None)}")
        with torch.no_grad(): target.copy_(data.to(target.device))
        return target
    if kind == "numpy":
        if target is None: return node["data"].copy()
        if target.shape != node["data"].shape or target.dtype != node["data"].dtype:
            raise ValueError(f"Checkpoint NumPy layout differs at {path}")
        np.copyto(target, node["data"]); return target
    if kind == "warp":
        if target is None or tuple(target.shape) != tuple(node["data"].shape[:len(target.shape)]) or not _warp_dtype_matches(node, target.dtype):
            raise ValueError(
                f"Checkpoint Warp layout differs at {path}: saved={node['data'].shape}, {node['dtype']}; "
                f"initialized={getattr(target, 'shape', None)}, {getattr(target, 'dtype', None)}")
        target.assign(node["data"]); return target
    if kind == "dict":
        target = {} if target is None else target
        for key, child in node["items"].items(): target[key] = restore(child, target.get(key), device=device, path=f'{path}[{key!r}]')
        return target
    if kind == 'resource_dict':
        target={} if target is None else target
        keys={}
        for key in target:
            token=_key_token(_key_descriptor(key))
            if token in keys:raise ValueError('Ambiguous initialized runtime resource mapping keys')
            keys[token]=key
        for entry in node['entries']:
            token=_key_token(entry['key'])
            key=keys[token] if token in keys else _restore_literal_key(entry['key'])
            target[key]=restore(entry['value'],target.get(key),device=device,path=f'{path}.mapping_value')
        return target
    if kind in ("list", "tuple", "deque"):
        previous = list(target) if target is not None else []
        values = [restore(child, previous[i] if i<len(previous) else None, device=device, path=f'{path}[{i}]')
                  for i,child in enumerate(node["items"])]
        if kind == "tuple": return tuple(values)
        if kind == "deque": return deque(values,maxlen=node["maxlen"])
        return values
    if kind == "object":
        if target is None or f"{type(target).__module__}.{type(target).__qualname__}" != node["type"]:
            raise ValueError(f"Missing or different runtime object: {node['type']}")
        if node["type"] == "isaaclab.utils.warp.proxy_array.ProxyArray":
            # The wrapper is immutable and its Torch cache must remain a view
            # of the recreated Warp allocation, never an independent clone.
            restore(node["fields"]["_warp"], target.warp, device=device, path=f'{path}._warp')
            cached = node["fields"].get("_torch_cache", {})
            if cached.get("kind") == "tensor":
                restore(cached, target.torch, device=device, path=f'{path}._torch_cache')
            return target
        if is_dataclass(target) and target.__dataclass_params__.frozen:
            # Configs are immutable. Reconstruct their saved values rather than
            # assigning fields, while mutable child buffers still restore in place.
            return replace(target, **{
                key: restore(child, getattr(target, key, None), device=device, path=f'{path}.{key}')
                for key, child in node["fields"].items()
            })
        for key, child in node["fields"].items():
            if node["type"] == "holosoma.agents.modules.logging_utils.LoggingHelper" and key == "log_dir":
                # A resumed run writes to its new directory; counters and reward
                # histories resume, but filesystem destinations are live resources.
                continue
            if node["type"] == "newton._src.sim.model.Model" and key == "shape_source_ptr":
                # These uint64 values are Warp mesh handles allocated by this
                # process, not simulation state. Old handles cause device faults.
                current = getattr(target, key, None)
                if child["kind"] == "warp" and (
                    current is None or tuple(current.shape) != tuple(child["data"].shape)
                    or not _warp_dtype_matches(child, current.dtype)
                ):
                    raise ValueError(f"Geometry handle layout differs at {path}.{key}")
                continue
            # Episode reports contain only the envs reset in the latest step;
            # their batch size is unrelated to fixed simulation allocations.
            episode_report = key in {"extras", "log_dict"} and type(target).__module__.startswith("holosoma.envs.")
            previous = None if episode_report else getattr(target, key, None)
            restored = restore(child, previous, device=device, path=f'{path}.{key}')
            if restored is not previous:
                setattr(target, key, restored)
        return target
    raise ValueError(f"Unknown training-state node {kind}")


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count(): raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def native_roots():
    from isaaclab_newton.physics.newton_manager import NewtonManager
    required = ("_state_0", "_state_1", "_control", "_model", "_solver", "_contacts")
    roots = {name: getattr(NewtonManager,name,None) for name in required}
    if any(v is None for v in roots.values()): raise ValueError("Incomplete native Newton runtime for checkpoint")
    solver = roots["_solver"]
    if getattr(solver,"mjw_data",None) is None or getattr(solver,"mjw_model",None) is None:
        raise ValueError("Recoverable training requires the actual MJWarp solver data and model")
    return roots


def capture(ppo, observations, next_iteration):
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    roots = native_roots()
    if ppo.storage.step != 0:
        raise ValueError("Milestone snapshots require a completed PPO update and empty rollout cursor")
    env = ppo._unwrap_env()
    if not hasattr(env,"command_manager") or env.command_manager.get_state("motion_command") is None:
        raise ValueError("Recoverable training snapshot requires the WBT command state")
    state = dict(schema=SCHEMA, runtime_id=NEWTON_RUNTIME_ID, num_envs=env.num_envs,
        next_iteration=int(next_iteration), phase="collect_next_rollout", pending_minibatches=[],
        environment=freeze(env), simulator=freeze(env.simulator),
        native={k: freeze(v) for k,v in roots.items()},
        observations=freeze(observations), storage=freeze(ppo.storage), logging=freeze(ppo.logging_helper),
        schedule=dict(actor_learning_rate=ppo.actor_learning_rate, critic_learning_rate=ppo.critic_learning_rate),
        pending_plan=dict(next_iteration=int(next_iteration), storage_step=int(ppo.storage.step)),
        rng=capture_rng())
    return state


def restore_runtime(ppo, state):
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    if state.get("schema") != SCHEMA or state["runtime_id"] != NEWTON_RUNTIME_ID:
        raise ValueError("Obsolete or different Newton training snapshot")
    env = ppo._unwrap_env()
    if state["num_envs"] != env.num_envs: raise ValueError("Exact training resume requires the same environment count")
    roots = native_roots()
    restore(state["environment"],env,device=ppo.device)
    restore(state["simulator"],env.simulator,device=ppo.device)
    for name,value in roots.items(): restore(state["native"][name],value,device=ppo.device)
    restore(state["storage"],ppo.storage,device=ppo.device)
    restore(state["logging"],ppo.logging_helper,device=ppo.device)
    ppo.actor_learning_rate=state["schedule"]["actor_learning_rate"]
    ppo.critic_learning_rate=state["schedule"]["critic_learning_rate"]
    ppo.current_learning_iteration=state["next_iteration"]
    observations=restore(state["observations"],None,device=ppo.device)
    import warp as wp
    wp.synchronize()
    env.simulator.refresh_sim_tensors()
    # No observations are recomputed: that would consume noise or shift history.
    restore_rng(state["rng"])
    return observations
