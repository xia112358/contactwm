"""Successful-episode hindsight replay, balanced by motion/terrain tile."""
from __future__ import annotations

import random

import numpy as np
import torch

from holosoma.acp.labels import hindsight_labels


class HindsightReplay:
    def __init__(self, capacity=100_000, seed=42):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.episodes = []
        self.size = 0

    def admit(self, frames, *, success, initial_phase, final_phase, tile):
        if not success or initial_phase > 0.1 or final_phase < 0.9 or len(frames) < 11:
            return False
        arrays = {key: np.stack([row[key] for row in frames]) for key in frames[0]}
        labels = hindsight_labels(arrays["active"], arrays["force"], arrays["position"], arrays["known"])
        target = labels["target_w"]-arrays["root"][:, None, :3]
        angle = arrays["yaw"][:, None]
        x, y = target[..., 0].copy(), target[..., 1].copy()
        target[..., 0] = np.cos(angle)*x+np.sin(angle)*y
        target[..., 1] = -np.sin(angle)*x+np.cos(angle)*y
        inside = (np.abs(target[..., 0]) <= 0.8) & (np.abs(target[..., 1]) <= 0.5)
        rows = {"history": arrays["history"], "goal": arrays["goal"], "terrain": arrays["terrain"],
                "valid": arrays["valid"], "target": target,
                "occupancy": labels["occupancy"] & inside, "known": labels["known"]}
        usable = rows["known"].any(-1) & rows["valid"].any(-1)
        rows = {key: torch.from_numpy(value[usable].copy()) for key, value in rows.items()}
        count = int(usable.sum())
        if count == 0:
            return False
        if count > self.capacity:
            rows = {key: value[:self.capacity] for key, value in rows.items()}
            count = self.capacity
        self.episodes.append((tile, rows))
        self.size += count
        while self.size > self.capacity and len(self.episodes) > 1:
            _, old = self.episodes.pop(0)
            self.size -= len(old["goal"])
        return True

    def sample(self, size, device):
        if not self.episodes:
            raise ValueError("No successful hindsight episodes")
        tiles = {}
        for tile, episode in self.episodes:
            tiles.setdefault(tile, []).append(episode)
        keys = list(tiles)
        selected = []
        for _ in range(size):
            episodes = tiles[self.rng.choice(keys)]
            lengths = [len(e["goal"]) for e in episodes]
            offset = self.rng.randrange(sum(lengths))
            for episode, length in zip(episodes, lengths):
                if offset < length:
                    selected.append((episode, offset))
                    break
                offset -= length
        return {key: torch.stack([row[key][index] for row, index in selected]).to(device) for key in selected[0][0]}

    def state_dict(self):
        return {"episodes": self.episodes, "size": self.size, "rng": self.rng.getstate(), "capacity": self.capacity}

    def load_state_dict(self, state):
        self.episodes, self.size, self.capacity = state["episodes"], state["size"], state["capacity"]
        self.rng.setstate(state["rng"])
