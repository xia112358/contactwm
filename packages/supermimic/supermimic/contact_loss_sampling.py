"""Choose trustworthy loss witnesses, without deciding task completion."""
import torch


def stable_loss_samples(stable,live):
    eligible=stable&live[...,None]
    times=torch.arange(len(stable),device=stable.device)[:,None,None]
    last=torch.where(eligible,times,-1).amax(0)
    return eligible&(times==last[None])


class SubstepLossWitness:
    """Keep the last qualified native point within each control interval.

    NativeContactStability already checks consecutive solves, row identity,
    load and material speed. This does not require a new support event or
    change the control-end observations used by task completion.
    """
    def __call__(self,tick,substep,observation,features,support):
        qualified=observation['known']&observation['active']&features['contact_stable']
        if substep==0:
            self.qualified=torch.zeros_like(qualified)
            self.point=torch.zeros_like(observation['point'])
        self.point=torch.where(qualified[...,None],observation['point'],self.point)
        self.qualified|=qualified
        self.features=features

    def summary(self):
        return dict(self.features,contact_stable=self.qualified,contact_loss_point=self.point)


def completed_stable_samples(stable, live, error_sq, tolerance):
    """First qualified completion per part; latest qualified evidence until completion."""
    eligible = stable & live[..., None]
    times = torch.arange(len(stable), device=stable.device)[:, None, None]
    reached = eligible & (error_sq <= tolerance.square())
    first = torch.where(reached, times, len(stable)).amin(0)
    last = torch.where(eligible, times, -1).amax(0)
    chosen = torch.where(first < len(stable), first, last)
    return eligible & (times == chosen[None])


@torch.no_grad()
def material_witness_errors(data, clock):
    """Each actual solver witness is compared only with its own paired target."""
    from .contact_material import world_points, mixed_error_sq
    event = data['goal_index']
    target = clock.data['point'][clock.event_ends[event] - 1]
    actual = world_points(data['contact_loss_position'], data['contact_loss_rotation'], clock.material_local[event])
    error = mixed_error_sq(actual, clock.material_targets[event], data['contact_loss_point'], target)
    if hasattr(clock, 'paired_material_local'):
        from .contact_paired_material import paired_material_error_sq
        paired, _ = paired_material_error_sq(data['contact_loss_position'], data['contact_loss_rotation'],
            data['contact_loss_point'], data['contact_stable'], clock.paired_material_local[event],
            clock.paired_material_targets[event], clock.paired_material_valid[event], data['contact_loss_candidate'])
        error = torch.cat((error[..., :4], paired[..., 4:]), -1)
    return error


def stable_duration_stops(done, stable, required, cutoffs):
    """Keep native duration only when every target has a stable witness."""
    native = done.long().argmax(0) + 1
    live = torch.arange(len(done), device=done.device)[:, None] < native[None]
    witnessed = (stable & live[..., None]).any(0)
    complete = required.any(-1) & (witnessed | ~required).all(-1)
    return torch.where(complete, native, cutoffs)
