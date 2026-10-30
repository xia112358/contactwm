"""Give each output its own contact/safety gate for secondary objectives."""

import torch


def contact_priority_loss(primary, secondary, accepted):
    if primary.shape != secondary.shape or primary.shape != accepted.shape:
        raise ValueError("contact priority requires one acceptance value per output")
    if accepted.dtype != torch.bool:
        raise ValueError("contact acceptance must be an explicit boolean Newton result")
    # Avoid NaN * 0 for unavailable layout evidence on rejected outputs.
    return primary + torch.where(accepted.detach(), secondary, torch.zeros_like(secondary))
