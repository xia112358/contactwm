"""CPU checks of recorded planner observations; this is not a controller evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tyro

from holosoma.acp.labels import PARTS, hindsight_labels
from holosoma.acp.models import ContactPlanner, planner_loss
from holosoma.acp.observations import yaw
from holosoma.acp.replay import HindsightReplay


@dataclass(frozen=True)
class Options:
    reference: Path
    output: Path
    updates: int = 20


def validate(options):
    torch.manual_seed(42)
    with np.load(options.reference, allow_pickle=False) as data:
        meta = json.loads(str(data["observed_contact_provenance_json"].item()))
        if not meta.get("complete"):
            raise ValueError("A complete, actually executed reference is required")
        labels = hindsight_labels(data["observed_contact_active"], data["observed_contact_force_w"],
                                   data["observed_contact_position_w"], data["observed_contact_known"])
        angles = yaw(torch.as_tensor(data["joint_pos"][:, [4, 5, 6, 3]])).numpy()
        frames = [{"active": data["observed_contact_active"][i], "force": data["observed_contact_force_w"][i],
            "position": data["observed_contact_position_w"][i], "known": data["observed_contact_known"][i],
            "root": data["joint_pos"][i, :3], "yaw": angles[i], "history": data["acp_history"][i],
            "goal": data["acp_goal"][i], "terrain": data["acp_terrain"][i],
            "valid": data["acp_terrain_valid"][i]} for i in range(min(500, len(data["joint_pos"])))]
        active_counts = dict(zip(PARTS, data["observed_contact_active"].sum(0).tolist()))
        target_counts = dict(zip(PARTS, labels["occupancy"].sum(0).tolist()))
    replay = HindsightReplay(seed=42)
    if not replay.admit(frames, success=True, initial_phase=0, final_phase=1, tile=0):
        raise RuntimeError("Successful native rollout produced no hindsight supervision")
    model = ContactPlanner(29)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = replay.sample(64, "cpu")
    def objective():
        return planner_loss(model(batch["history"], batch["goal"], batch["terrain"], batch["valid"]),
                            batch["terrain"], batch["target"], batch["occupancy"], batch["known"])["loss"]
    before = float(objective().detach())
    for _ in range(options.updates):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
    after = float(objective().detach())
    if not np.isfinite([before, after]).all() or after >= before:
        raise RuntimeError(f"Planner optimization did not decrease loss: {before} -> {after}")
    # Exercise the actual ten-update / eight-subbatch training hook as well.
    from holosoma.acp.ppo import ACPPPO
    runner = object.__new__(ACPPPO)
    runner.device = "cpu"
    runner.env = SimpleNamespace(acp_planner=model, acp_replay=replay,
        acp_settings={"minimum_replay": 1, "planner_batch": 64})
    runner.planner_optimizer = optimizer
    hook_metrics = runner._planner_training_step()
    report = {"schema": "somaforge_acp_validation_v1", "reference": str(options.reference.resolve()),
        "paper_planner_parameters": sum(p.numel() for p in ContactPlanner().parameters()),
        "canonical_29_joint_parameters": sum(p.numel() for p in model.parameters()),
        "native_active_frames": active_counts, "future_contact_targets": target_counts,
        "replay_samples": replay.size, "diagnostic_optimizer_updates": options.updates,
        "fixed_batch_loss_before": before, "fixed_batch_loss_after": after, "ppo_planner_hook": hook_metrics,
        "interpretation": "Optimizer/real-data wiring check using a dense expert rollout; not ACP convergence or held-out task success"}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    validate(tyro.cli(Options))


if __name__ == "__main__":
    main()
