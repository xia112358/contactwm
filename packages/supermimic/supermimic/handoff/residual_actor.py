"""State-conditioned residual decoders for controlled contact-policy diagnostics."""
import copy
import torch
from torch import nn
from supermimic.networks import mlp


class ResidualContactActor(nn.Module):
    """Keep the accepted policy fixed; learn a dense or four-expert correction.

    Routing uses physical state and goal features only. The original stochastic
    policy and its input derivatives remain intact for WM-backed BPTT.
    """
    def __init__(self, base, *, mixture):
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.hist, self.include_time = base.hist, base.include_time
        self.mixture = mixture
        self.feature_dim = 128 + 32 + 39
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim))
        self.register_buffer('feature_scale', torch.ones(self.feature_dim))
        if mixture:
            self.experts = nn.ModuleList(mlp(self.feature_dim, [64], 29) for _ in range(4))
            self.router = nn.Linear(self.feature_dim, 4)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
            heads = [expert[-1] for expert in self.experts]
        else:
            self.residual = mlp(self.feature_dim, [260], 29)
            heads = [self.residual[-1]]
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.architecture = dict(type='contact_residual_moe' if mixture else 'contact_residual_dense',
            version=1, base=copy.deepcopy(base.architecture), feature_dim=self.feature_dim,
            experts=4 if mixture else 1, hidden=64 if mixture else 260,
            routing='softmax_state_mu_goal', base_frozen=True)
        self.last_epsilon = self.last_kl = None

    @property
    def norm(self):
        return self.base.norm

    def features(self, aux, condition):
        raw = torch.cat((aux['state'], aux['mu'], condition), -1)
        return (raw - self.feature_mean) / self.feature_scale

    @torch.no_grad()
    def initialize_features(self, history, condition):
        """Fit fixed scaling and state-only k-means routing on actual inputs."""
        _, aux = self.base(history, condition, return_aux=True)
        raw = torch.cat((aux['state'], aux['mu'], condition), -1)
        self.feature_mean.copy_(raw.mean(0))
        self.feature_scale.copy_(raw.std(0).clamp_min(1e-3))
        if self.mixture:
            x = self.features(aux, condition)
            # Farthest-point seeds give distinct initial routes without labels.
            centers = [x[0]]
            for _ in range(3):
                distance = torch.cdist(x, torch.stack(centers)).square().min(-1).values
                centers.append(x[distance.argmax()])
            centers = torch.stack(centers)
            for _ in range(12):
                labels = torch.cdist(x, centers).argmin(-1)
                sums = torch.zeros_like(centers).index_add_(0, labels, x)
                counts = torch.bincount(labels, minlength=4)
                assert bool((counts > 0).all()), 'Empty routing initialization cluster'
                centers = sums / counts[:, None]
            temperature = 0.25 * self.feature_dim
            self.router.weight.copy_(2 * centers / temperature)
            self.router.bias.copy_(-centers.square().sum(-1) / temperature)

    def forward(self, history, condition, epsilon=None, *, sample=False, return_aux=False):
        original, aux = self.base(history, condition, epsilon=epsilon, sample=sample, return_aux=True)
        x = self.features(aux, condition)
        if self.mixture:
            routing = self.router(x).softmax(-1)
            correction = (torch.stack([expert(x) for expert in self.experts], -2) * routing[..., None]).sum(-2)
            aux['routing'] = routing
        else:
            correction = self.residual(x)
        action = (original + correction).clamp(-20, 20)
        self.last_epsilon, self.last_kl = aux['epsilon'].detach(), aux['kl']
        if return_aux:
            return action, aux
        return action

    def forward_mean(self, history, condition):
        return self(history, condition)
