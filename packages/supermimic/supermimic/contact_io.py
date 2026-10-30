"""Lazy CPU reads for trusted local training checkpoints.

Only tensors copied into a model or optimizer need to reach the accelerator.
Memory mapping also avoids eagerly reading unused optimizer/replay tensors.
Legacy non-zip checkpoints retain PyTorch's ordinary CPU loading path.
"""
from pathlib import Path

import torch


def load_checkpoint_cpu(path):
    path = Path(path)
    with path.open('rb') as stream:
        is_zip = stream.read(4) == b'PK\x03\x04'
    return torch.load(path, map_location='cpu', weights_only=False, mmap=is_zip)
