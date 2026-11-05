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
