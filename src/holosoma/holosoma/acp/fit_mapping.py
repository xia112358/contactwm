"""Fit the local three-mode mapper on paired noisy/ground-truth grids (CPU by default).

NPZ arrays: height, valid, target, target_valid, all [samples, rows, columns].
Grids use metres relative to the gravity-aligned sensor frame, at 3 cm spacing.
This offline optimizer has no simulator or AppLauncher dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.nn import functional as F

from holosoma.acp.mapping import MAPPING_SCHEMA, MappingNetwork, mode_labels


@dataclass(frozen=True)
class Options:
    dataset: Path
    output: Path
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-3
    label_tolerance: float = 0.03
    seed: int = 42
    device: str = "cpu"
    resume: Path | None = None


def fit(options):
    if min(options.steps, options.batch_size, options.learning_rate, options.label_tolerance) <= 0:
        raise ValueError("Training steps, batch, learning rate and tolerance must be positive")
    if options.output.exists():
        raise FileExistsError(options.output)
    with np.load(options.dataset, allow_pickle=False) as data:
        arrays = {key: torch.from_numpy(data[key].copy()) for key in ("height", "valid", "target", "target_valid")}
    height, valid = arrays["height"].float(), arrays["valid"].bool()
    target, target_valid = arrays["target"].float(), arrays["target_valid"].bool()
    if height.ndim != 3 or not len(height) or any(v.shape != height.shape for v in arrays.values()):
        raise ValueError("Mapping dataset requires matching nonempty [N,H,W] arrays")
    if not torch.isfinite(height[valid]).all() or not torch.isfinite(target[target_valid]).all():
        raise ValueError("Known mapping heights and targets must be finite")
    height = height.masked_fill(~valid, 0)
    target = target.masked_fill(~target_valid, 0)
    usable = target_valid.flatten(1).any(-1)
    if not usable.any():
        raise ValueError("Mapping dataset has no known ground-truth supervision")
    height, valid, target, target_valid = (x[usable] for x in (height, valid, target, target_valid))
    torch.manual_seed(options.seed)
    model = MappingNetwork().to(options.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    generator = torch.Generator().manual_seed(options.seed)
    trained_steps = 0
    if options.resume:
        saved = torch.load(options.resume, map_location=options.device, weights_only=True)
        if saved.get("schema") != MAPPING_SCHEMA:
            raise ValueError("Not an ACP mapping checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["rng"].cpu())
        trained_steps = saved["trained_steps"]
    first_loss = None
    for _ in range(options.steps):
        indices = torch.randint(len(height), (options.batch_size,), generator=generator)
        h, v, t, tv = (x[indices].to(options.device) for x in (height, valid, target, target_valid))
        labels = mode_labels(h, v, t, tv, options.label_tolerance)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(h, v), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite mapping objective")
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
    bundle = {"schema": MAPPING_SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "trained_steps": trained_steps+options.steps, "rng": generator.get_state(),
        "dataset": str(options.dataset.resolve()), "label_tolerance": options.label_tolerance,
        "first_loss": first_loss, "last_loss": float(loss.detach()),
        "implementation": "local_three_mode_cnn_not_author_weights"}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("xb") as handle:
        torch.save(bundle, handle)
    return {key: bundle[key] for key in ("trained_steps", "first_loss", "last_loss")}


if __name__ == "__main__":
    print(fit(tyro.cli(Options)))
