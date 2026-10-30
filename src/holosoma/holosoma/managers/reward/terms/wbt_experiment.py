"""Contact and force rewards using the exact same measured-contact view as the actor."""
import torch
from holosoma.managers.observation.terms.wbt_experiment import reference_contacts
from holosoma.utils.wbt_contacts import measured_contacts


def contact_match(env):
    target, _ = reference_contacts(env)
    actual = measured_contacts(env)
    score = (actual["active"] == target[:, 0]).float()
    return (score * actual["known"]).sum(-1) / actual["known"].sum(-1).clamp_min(1)


def force_tracking(env, sigma: float = 100.):
    if sigma <= 0:
        raise ValueError("Force tracking sigma must be positive")
    active, target = reference_contacts(env)
    actual = measured_contacts(env)
    valid = active[:, 0] & actual["known"]
    error = (target[:, 0] - actual["force_w"]).square().sum(-1)
    score = torch.exp(-error / sigma**2)
    return (score * valid).sum(-1) / valid.sum(-1).clamp_min(1)
