"""Preserved experiment helpers for regression tests; no training CLI."""
import numpy as np
import torch
from generator.next_interaction import NextInteraction
from generator.next_interaction_surface import objective, SCHEMA
from somaforge_core.newton_contact_query import query_contacts
from somaforge_core.contact_face_selection import select_contact_pairs
from generator.research.archived_experiments.paired_pose_transform import transform_q
from generator.research.archived_experiments.contact_geometry import COLLISION_TOLERANCE_M
GATE_SCHEMA = 'replay_endpoint_feasibility_v1'
HOLD_DRIFT_BUDGET_M = 0.005
JOINT_NUMERICAL_TOLERANCE_RAD = 1e-06

def passes_gate(exact_topology, penetration_m, hold_drift_m, joint_violation_rad):
    return bool(exact_topology and penetration_m <= COLLISION_TOLERANCE_M and (hold_drift_m <= HOLD_DRIFT_BUDGET_M) and (joint_violation_rad <= JOINT_NUMERICAL_TOLERANCE_RAD))

def verify_candidate(model, candidate, origin, basis, steps=150):
    (x, t, scene) = candidate
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pred = model(**x)
        seeds = torch.cat((pred.qpos, t['q'], x['current_q']), 0)
        target = {k: v.expand(3, *v.shape[1:]) for (k, v) in t.items()}
        local = {k: v.expand(3, *v.shape[1:]) for (k, v) in scene.items()}
        roles = torch.nn.functional.one_hot(target['role'], 4).float() * 20
        surfaces = torch.nn.functional.one_hot(target['surface'].clamp_min(0), 2).float() * 20
        raw = seeds.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([raw], lr=0.008)
        best = seeds.detach().clone()
        score = torch.full((3,), float('inf'))

        def evaluate(value):
            q = torch.cat((value[:, :3], torch.nn.functional.normalize(value[:, 3:7], dim=-1), value[:, 7:]), -1)
            return (q, objective(model, NextInteraction(q, roles, surfaces, target['duration']), target, local))
        with torch.enable_grad():
            for step in range(steps + 1):
                (q, (loss, _)) = evaluate(raw)
                if not torch.isfinite(loss).all():
                    raise ValueError('Nonfinite replay feasibility objective')
                with torch.no_grad():
                    better = loss < score
                    best[better] = q.detach()[better]
                    score = torch.minimum(score, loss.detach())
                if step == steps:
                    break
                optimizer.zero_grad()
                raw.grad = torch.autograd.grad(loss.mean(), raw)[0]
                optimizer.step()
        with torch.no_grad():
            (_, (_, metrics)) = evaluate(best)
        world = np.stack([transform_q(q.numpy(), origin, basis) for q in best])
        actual = query_contacts(world, None)
        selected = select_contact_pairs(actual['pairs'], actual['surface_catalog'])
        rows = []
        for (i, seed) in enumerate(('predictor', 'demonstration', 'current_state')):
            mask = np.asarray(selected['contact_part_mask'][i])
            surface = np.asarray(selected['contact_surface'][i])
            desired = target['role'][i].numpy() != 0
            ds = target['surface'][i].numpy()
            exact = bool(np.array_equal(mask, desired) and np.array_equal(surface[desired], ds[desired]))
            pen = float(metrics['penetration_cm'][i]) / 100
            hold = float(metrics['persistent_drift_max_cm'][i]) / 100
            violation = float(metrics['joint_violation_rad'][i])
            rows.append(dict(seed=seed, exact_newton_topology=exact, penetration_m=pen, persistent_drift_m=hold, joint_violation_rad=violation, passed=passes_gate(exact, pen, hold, violation), diagnostic_world_q=world[i].tolist()))
        return dict(schema=GATE_SCHEMA, objective_schema=SCHEMA, accepted=any((r['passed'] for r in rows)), iterations=steps, collision_budget_m=COLLISION_TOLERANCE_M, hold_drift_budget_m=HOLD_DRIFT_BUDGET_M, model_fingerprint=actual['provenance']['model_fingerprint'], witnesses=rows, optimized_pose_used_as_label=False, trajectory_reachability_verified=False)
    finally:
        model.train(training)
