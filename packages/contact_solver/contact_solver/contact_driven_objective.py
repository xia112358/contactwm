"""Predictor objective: demonstrations guide choices; own contacts constrain FK.

Generated states have no relabeled expert successor. Do not pretend that their
historical demonstration successor is a valid supervised target.
"""
import torch
import torch.nn.functional as F
from contact_solver.contact_validity import surface_gap


def masked_mean(value, mask):
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def contact_geometry(prediction, current_contact, current_anchor, center, rotation, half, ground):
    point = prediction.contact_position
    local = torch.einsum('bij,bpj->bpi', rotation.transpose(-1, -2), point-center[:, None])
    outside = (local[..., :2].abs()-half[:, None, :2]).relu()
    box2 = (local[..., 2]-half[:, None, 2]).square()+outside.square().sum(-1)
    ground2 = (point[..., 2]-ground[:, None]).square()
    gap2 = torch.where(prediction.contact_surface == 1, box2, ground2)
    active = prediction.active_contact.bool()
    hold = active & prediction.persistent_support.bool() & (current_contact > .5)
    anchor2 = (point-current_anchor).square().sum(-1)
    def mean_peak(value, mask):
        return masked_mean(value, mask)+(value*mask).amax(-1).mean()
    return {
        'hold': mean_peak(anchor2/.005**2, hold),
        'surface': mean_peak(gap2/.005**2, active),
        'hold_max_cm': (anchor2.sqrt()*hold).amax()*100,
        'surface_max_cm': (gap2.sqrt()*active).amax()*100,
    }


def contact_driven_terms(prediction, values, current_q, current_contact, current_anchor, generated, canonical_offsets, supervised_point=None):
    nominal = ~generated
    # Only unchanged expert inputs have valid expert next-event labels.
    bce = lambda logits, target: masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(-1), nominal)
    choice = (bce(prediction.contact_logits, values[6])
              + bce(prediction.touchdown_logits, values[7])
              + bce(prediction.persistent_logits, values[8]))
    valid = nominal[:, None] & (values[6]>.5) & (values[9]>=0)
    surface_class = masked_mean(F.cross_entropy(prediction.surface_logits.transpose(1, 2), values[9].clamp(0, 1).long(), reduction='none'), valid)
    pose = (((prediction.qpos[:, :3]-values[5][:, :3])/.02).square().mean(-1)
            + (1-(prediction.qpos[:, 3:7]*values[5][:, 3:7]).sum(-1).abs())
            + ((prediction.qpos[:, 7:]-values[5][:, 7:])/.1).square().mean(-1))
    pose = masked_mean(pose, nominal)
    landing = masked_mean(((prediction.contact_position-values[10])/.01).square().sum(-1), valid & (values[7]>.5))
    duration = masked_mean(((prediction.duration.log()-values[11].log())/.1).square(), nominal)
    geometry = contact_geometry(prediction, current_contact, current_anchor, values[13], values[14], values[15], values[16])
    # A wrong predicted release must not remove expert-required support from
    # supervised loss. Self-consistency complements, never replaces, labels.
    expert_point = prediction.contact_position if supervised_point is None else supervised_point
    expert_hold=valid & (values[8]>.5) & (current_contact>.5)
    hold2=((expert_point-current_anchor)/.005).square().sum(-1)
    expert_hold_loss=masked_mean(hold2,expert_hold)+(hold2*expert_hold).amax(-1).mean()
    # Exactly one geometry contract per row: teacher topology for labeled
    # endpoints, own intention only for generated rows without expert labels.
    target = torch.where((values[8]>.5)[...,None], current_anchor, values[10])
    target2 = ((expert_point-target)/.005).square().sum(-1)
    surface2 = surface_gap(expert_point, values[9], values[13], values[14], values[15], values[16], squared=True)/.005**2
    surface2 = surface2.masked_fill(~valid, 0)
    supervised_geometry = masked_mean(target2, valid)+(target2*valid).amax(-1).mean()
    supervised_geometry += masked_mean(surface2, valid)+surface2.amax(-1).mean()
    from types import SimpleNamespace
    generated_prediction = SimpleNamespace(**{**vars(prediction), 'active_contact': prediction.active_contact & generated[:,None]})
    generated_geometry = contact_geometry(generated_prediction, current_contact, current_anchor, values[13], values[14], values[15], values[16])
    # A bounded weak regularizer, not a command to remain motionless or copy q.
    step = ((prediction.qpos[:, 7:]-current_q[:, 7:])/1.).square().mean()
    new = prediction.active_contact & ~prediction.persistent_support
    offset = masked_mean(((prediction.contact_local_offset-canonical_offsets)/.01).square().sum(-1), new)
    total = choice+surface_class+duration+.05*pose+10*(generated_geometry['hold']+generated_geometry['surface']+supervised_geometry)+.01*step+.1*offset
    return dict(total=total, pose=pose, choice=choice, landing=landing, expert_hold=expert_hold_loss, **geometry)
