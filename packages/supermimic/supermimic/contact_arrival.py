"""Measured arrival patch within a short, contiguous first-contact episode.

No point averaging, geometric activation, later recontact, or target-face
substitution. The chosen witness is an actually observed solver contact.
"""
import torch


class ArrivalWindow:
    def __init__(self, shape, device, frames=2):
        if frames < 1:
            raise ValueError("Arrival window must include at least the onset frame")
        self.frames = frames
        self.age = torch.full(shape, -1, device=device, dtype=torch.long)
        self.closed = torch.zeros(shape, device=device, dtype=torch.bool)
        self.distance = torch.full(shape, float('inf'), device=device)
        self.index = torch.full(shape, -1, device=device, dtype=torch.long)

    def reset(self, slots):
        self.age[slots] = -1
        self.closed[slots] = False
        self.distance[slots] = float('inf')
        self.index[slots] = -1

    def update(self, first, observed, eligible, distance, step, terminal):
        start = first & (self.age < 0) & ~self.closed
        self.age = torch.where(start, 0, self.age)
        opened = (self.age >= 0) & ~self.closed
        improved = opened & observed & eligible & (distance < self.distance)
        self.distance = torch.where(improved, distance, self.distance)
        self.index = torch.where(improved, torch.as_tensor(step, device=distance.device), self.index)
        self.closed |= opened & (~observed | (self.age + 1 >= self.frames) | terminal[..., None])
        self.age = torch.where(opened & ~self.closed, self.age + 1, self.age)
        return improved


def arrival_indices(data, target, *, frames=2, eligible=None, live=None):
    """Same state machine as online completion, returning real witness indices."""
    T, B = data['done'].shape
    if live is None:
        live = (data['done'].long().cumsum(0) - data['done'].long()) == 0
    observed = data['known'] & data['active']
    if eligible is None:
        eligible = torch.ones_like(observed)
    window = ArrivalWindow(observed.shape[1:], observed.device, frames)
    distance = (data['point'] - target[None]).norm(dim=-1)
    for t in range(T):
        terminal = data['done'][t] | (t == T-1)
        if t+1 < T:
            terminal = terminal | ~live[t+1]
        window.update(data['contact_first'][t] & live[t,:,None],
                      observed[t] & live[t,:,None], eligible[t], distance[t], t, terminal)
    return window.index, window.distance
