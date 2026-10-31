"""Checkpoint scheduling without importing torch or initializing a simulator."""
from pathlib import Path
import csv
import time


def completed_iterations(run_dir):
    path = Path(run_dir) / "eval.csv"
    if not path.exists():
        return set()
    with path.open() as stream:
        return {int(row["iteration"]) for row in csv.DictReader(stream)}


def pending_checkpoints(run_dir, done, *, iteration_interval=None, every=1, stop_after=None):
    paths = []
    for path in sorted(Path(run_dir).glob("ckpt_*.pt")):
        iteration = int(path.stem[5:])
        if iteration in done or (stop_after is not None and iteration > stop_after):
            continue
        selected = (iteration % iteration_interval == 0 if iteration_interval is not None
                    else (iteration // 2000) % every == 0)
        if selected:
            paths.append(str(path))
    return paths


def wait_for_checkpoint(run_dir, *, iteration_interval=None, every=1, stop_after=None, poll_seconds=60):
    """Wait before loading assets/CUDA; stopped or completed runs need no device."""
    root = Path(run_dir)
    while True:
        if (root / "invalid_transition.pt").exists() or (root / "STOPPED.json").exists():
            return False
        done = completed_iterations(root)
        if stop_after is not None and any(it >= stop_after for it in done):
            return False
        if pending_checkpoints(root, done, iteration_interval=iteration_interval,
                               every=every, stop_after=stop_after):
            return True
        time.sleep(poll_seconds)
