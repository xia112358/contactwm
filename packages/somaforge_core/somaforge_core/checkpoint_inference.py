"""Read inference fields without importing saved simulator resource caches.

Full recovery snapshots remain untouched. Opaque Lab resources are tolerated
only in discarded runtime state, never in the inference tensors/configuration.
"""
import pickle
from types import SimpleNamespace


class _RuntimeResource:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass


class _RuntimeString(str):
    pass


class _InferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "isaaclab.utils.string" and name == "ResolvableString":
            return _RuntimeString
        if module.startswith(("isaaclab.", "isaaclab_newton.")):
            return _RuntimeResource
        return super().find_class(module, name)


def load_inference_fields(path):
    import torch
    loader = SimpleNamespace(__name__="inference_pickle", Unpickler=_InferenceUnpickler, load=pickle.load)
    value = torch.load(path, map_location="cpu", weights_only=False, pickle_module=loader)
    names = ("actor_model_state_dict", "critic_model_state_dict", "actor_obs_normalizer_state_dict",
             "critic_obs_normalizer_state_dict", "experiment_config", "robot_asset", "iter",
             "iteration", "infos", "wandb_run_path")
    def check(item):
        if isinstance(item, _RuntimeResource) or item is _RuntimeResource or isinstance(item, _RuntimeString):
            raise ValueError("Simulator resource found in required inference fields; cannot discard it")
        if isinstance(item, dict):
            for key, child in item.items():
                check(key); check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    output = {key: value[key] for key in names if key in value}
    check(output)
    return output
