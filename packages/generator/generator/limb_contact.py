"""Four-limb contact intent; regional contact subsets remain independent.

Leg states choose either foot or knee/shin, never two independent targets.
Scores are joint potentials from the existing part heads, normalized and
supervised over legal limb states rather than clipped after prediction.
"""
import torch
import torch.nn.functional as F

SCHEMA = 'four_limb_single_contact_mode_v1'
GROUPS = ((0, 4), (1, 5), (2,), (3,))


def limb_logits(role_logits):
    scores = []
    for group in GROUPS:
        if len(group) == 1:
            scores.append(F.pad(role_logits[:, group[0]], (0, 3), value=-torch.inf))
        else:
            a, b = (role_logits[:, i] for i in group)
            scores.append(torch.cat((a[:, :1]+b[:, :1],
                                     a[:, 1:]+b[:, :1], b[:, 1:]+a[:, :1]), -1))
    return torch.stack(scores, 1)


def decode_limb_roles(logits):
    state = logits.argmax(-1)
    roles = torch.zeros((len(logits), 6), dtype=torch.long, device=logits.device)
    for limb, group in enumerate(GROUPS):
        s = state[:, limb]
        roles[:, group[0]] = torch.where(s <= 3, s, 0)
        if len(group) == 2:
            roles[:, group[1]] = torch.where(s > 3, s-3, 0)
    return roles


def limb_targets(roles):
    if roles.ndim != 2 or roles.shape[1] != 6 or roles.dtype != torch.long:
        raise ValueError('Limb roles require long [B,6] labels')
    if bool(((roles < 0) | (roles > 3)).any()):
        raise ValueError('Unknown limb role')
    targets = []
    for group in GROUPS:
        a = roles[:, group[0]]
        if len(group) == 2:
            b = roles[:, group[1]]
            if bool(((a != 0) & (b != 0)).any()):
                raise ValueError('One limb cannot request independent foot and knee targets')
            a = torch.where(b != 0, b+3, a)
        targets.append(a)
    return torch.stack(targets, 1)


def role_objective(prediction, roles):
    joint = getattr(prediction, 'limb_role_logits', None)
    logits, target = (prediction.role_logits, roles) if joint is None else (joint, limb_targets(roles))
    if not bool(torch.isfinite(logits.gather(-1, target[..., None])).all()):
        raise ValueError('Supervised role is outside the actual-contact action space')
    return F.cross_entropy(logits.transpose(1, 2), target, reduction='none').mean(-1)
