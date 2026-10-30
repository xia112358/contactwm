"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, extra_fields: dict | None = None):
        self.T, self.N, self.device = T, N, device
        self.extra_fields = extra_fields or {}
        self.fields = {**FIELDS, **self.extra_fields}
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def recent(self, count: int, fields):
        """Chronological read-only views of the last collection where contiguous.

        Consumers finish before collection overwrites the buffer. A wrapped
        interval needs one concatenation per field, never a GPU index tensor.
        """
        if not 0 <= count <= self.size:
            raise ValueError('Requested recent interval is outside replay')
        start = (self.ptr - count) % self.T
        if count == 0:
            return {key: self.data[key][:0] for key in fields}
        if start + count <= self.T:
            return {key: self.data[key][start:start+count] for key in fields}
        return {key: torch.cat((self.data[key][start:], self.data[key][:self.ptr]), 0)
                for key in fields}

    def sample_windows(self, B: int, k: int, *, fields=None):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        keys = dict.fromkeys(("s", "s1", "a", "xi", "done", *self.extra_fields)
                             if fields is None else (*fields, "done"))
        out = {key: self.data[key][rows, e] for key in keys}
        if fields is None and "contact_next" in self.data:
            out["contact_next"] = self.data["contact_next"][rows, e]
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in dict.fromkeys(("s", "a_prev", "xi", "f", *self.extra_fields))}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
