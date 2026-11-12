"""Explicit-noise Gaussian bottleneck actor for contact policy experiments."""
import torch
from torch import nn

from supermimic.contact_control import OBS_DIM, GOAL_DIM
from supermimic.networks import RunningNorm, mlp


class BottleneckContactActor(nn.Module):
    """Default inference uses mu; sampling/replay must be requested explicitly.

    ``sample=True`` generates epsilon and exposes it in ``last_epsilon``.
    Passing ``epsilon`` reuses that exact draw without touching the RNG.
    ``return_aux=True`` returns (action, aux), with a per-sample standard-normal
    KL and its batch mean. Neither sampling nor KL depends on train/eval mode.
    """
    def __init__(self, hist=5, include_time=False):
        super().__init__()
        if hist != 5 or include_time:
            raise ValueError('This architecture requires hist=5 and no time input')
        self.hist = hist
        self.include_time = include_time
        self.latent_dim = 32
        self.norm = RunningNorm(OBS_DIM)
        self.state_encoder = nn.Sequential(mlp(hist * OBS_DIM, [512, 256], 128), nn.ELU())
        self.part_embedding = nn.Parameter(torch.randn(1, 6, 64) * .01)
        self.part_encoder = nn.Sequential(nn.Linear(5, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.posterior_encoder = nn.Sequential(mlp(128 + 393, [256], 128), nn.ELU())
        self.mu_head = nn.Linear(128, self.latent_dim)
        self.log_std_head = nn.Linear(128, self.latent_dim)
        self.decoder = mlp(128 + self.latent_dim, [256, 128], 29)
        self.architecture = dict(type='contact_gaussian_bottleneck', version=1,
            hist=hist, obs_dim=OBS_DIM, include_time=include_time,
            state_hidden=[512, 256, 128], part_width=64, goal_dim=393,
            posterior_hidden=[256, 128], latent_dim=self.latent_dim,
            decoder_hidden=[256, 128], activation='ELU', default_latent='mu',
            kl='sum_over_latents_mean_over_batch_to_standard_normal',
            sampling='explicit_sample_or_epsilon', log_std_bounds=[-5., 2.])
        with torch.no_grad():
            self.decoder[-1].weight.mul_(.01)
            self.decoder[-1].bias.zero_()
        self.last_epsilon = None
        self.last_kl = None

    def forward(self, history, condition, epsilon=None, *, sample=False, return_aux=False):
        state = self.state_encoder(self.norm(history).flatten(-2))
        parts = torch.cat((condition[:, :18].reshape(-1, 6, 3),
                           condition[:, 18:24, None], condition[:, 33:39, None]), dim=-1)
        goal = torch.cat(((self.part_encoder(parts) + self.part_embedding).flatten(1),
                          condition[:, 24:33]), dim=-1)
        encoded = self.posterior_encoder(torch.cat((state, goal), dim=-1))
        mu = self.mu_head(encoded)
        log_std = self.log_std_head(encoded).clamp(-5., 2.)
        if epsilon is not None:
            if sample:
                raise ValueError('Explicit epsilon and fresh sampling are mutually exclusive')
        else:
            epsilon = torch.randn_like(mu) if sample else torch.zeros_like(mu)
        z = mu + log_std.exp() * epsilon
        kl_per_sample = .5 * (mu.square() + torch.expm1(2 * log_std) - 2 * log_std).sum(-1)
        self.last_epsilon = epsilon.detach()
        self.last_kl = kl_per_sample.mean()
        action = self.decoder(torch.cat((state, z), dim=-1)).clamp(-20, 20)
        if return_aux:
            return action, dict(epsilon=epsilon, mu=mu, log_std=log_std, z=z,
                                kl_per_sample=kl_per_sample, kl=self.last_kl)
        return action

    def forward_mean(self, history, condition):
        return self(history, condition)
