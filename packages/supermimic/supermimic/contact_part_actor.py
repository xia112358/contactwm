"""Per-part spatial goal encoding for the reference-free contact actor."""
import torch
from torch import nn

from .contact_control import ContactActor, OBS_DIM
from .networks import mlp


class PartContactActor(ContactActor):
    def __init__(self, hist=5, hidden=(512, 256, 128), include_time=False):
        # Keep the part encoder's seeded initialization identical across head sizes.
        super().__init__(hist=hist, include_time=include_time)
        self.part_embedding = nn.Parameter(torch.randn(1, 6, 64) * .01)
        self.part_encoder = nn.Sequential(nn.Linear(5, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.net = mlp(hist * OBS_DIM + 6 * 64 + 9 + int(include_time), list(hidden), 29)
        self.architecture = dict(type='part_mlp', part_width=64, hidden=list(hidden),
                                 hist=hist, include_time=include_time)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.zero_()

    def forward(self, history, condition):
        observed = self.norm(history).flatten(-2)
        parts = torch.cat((condition[:, :18].reshape(-1, 6, 3),
                           condition[:, 18:24, None], condition[:, 33:39, None]), dim=-1)
        encoded = self.part_encoder(parts) + self.part_embedding
        goal = torch.cat((encoded.flatten(1), condition[:, 24:33], condition[:, 39:]), dim=-1)
        return self.net(torch.cat((observed, goal), dim=-1)).clamp(-20, 20)
