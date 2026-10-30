"""Cache equivalence and real trainer commit/rollback integration (CPU)."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contact_solver.keep_patch_cache import KeepPatchCache
from contact_solver.keep_patch_loss import bind_keep_patch, keep_patch_loss
from generator.parallel_rollout_pool import ParallelRolloutPool


class RigidFK:
    def link_poses(self, q, names):
        return (q[:, None, :3].expand(-1, len(names), -1),
                torch.eye(3, dtype=q.dtype, device=q.device).expand(len(q), len(names), 3, 3))


def observed(q, scenes):
    # Different patch sizes including empty, with rows deliberately interleaved.
    sample = torch.arange(len(q), device=q.device).repeat(3)
    site = torch.arange(3, device=q.device).repeat_interleave(len(q))
    active = site <= scenes[sample].remainder(3)
    active &= q[sample, 2] < 1
    point = q[sample, :3].detach().clone()
    point[:, 0] += .03*(site-1)
    point[:, 1] += .002*scenes[sample]
    pairs = dict(sample=sample, part=torch.zeros_like(sample), body_link1=torch.zeros_like(sample),
        body_link0=torch.full_like(sample, -1), geometry_point1_w=point,
        primary_surface=torch.zeros_like(sample), active=active,
        constraint_allocated=torch.ones_like(active), eligible=active)
    return dict(schema='newton_device_witness_batch_v1', pairs=pairs, link_names=('foot',),
        face_surface=torch.zeros(len(q), 1, dtype=torch.long, device=q.device),
        face_normal=q.new_tensor([0, 0, 1]).expand(len(q), 1, 3),
        contact_part_mask=(q[:, 2:3] < 1))


class Query:
    def __init__(self):
        self.samples = 0

    def __call__(self, q, scenes):
        self.samples += len(q)
        return observed(q, scenes)


def assert_loss_gradient_equal(q, a, b):
    outputs = []
    for patch in (a, b):
        prediction = (q+q.new_tensor([.07, .02, .01])).requires_grad_()
        loss, metrics = keep_patch_loss(RigidFK(), prediction,
            torch.full((len(q), 6), 2, device=q.device), patch, q.new_full((6,), .01))
        loss.sum().backward()
        outputs.append((loss.detach(), prediction.grad, metrics))
    for x, y in zip(outputs[0][:2], outputs[1][:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for name in outputs[0][2]:
        torch.testing.assert_close(outputs[0][2][name], outputs[1][2][name], rtol=0, atol=0)


def test_repeated_reordered_duplicate_reads_preserve_loss_and_gradient():
    q = torch.tensor([[0., 0., 0.], [1., 2., 0.], [0., 0., 2.]], dtype=torch.float64)
    scenes = torch.tensor([0, 2, 1])
    cache, query = KeepPatchCache(3, q, ('foot',)), Query()
    for keys in (torch.arange(3), torch.tensor([2, 0, 1, 0]), torch.tensor([1, 2])):
        cached = cache.get(keys, q[keys], scenes[keys], RigidFK(), query)
        fresh = bind_keep_patch(RigidFK(), q[keys], observed(q[keys], scenes[keys]))
        assert_loss_gradient_equal(q[keys], fresh, cached)
        assert not cached.local.requires_grad and not cached.position.requires_grad
    assert query.samples == 3 and cache.hit_samples == 6


def test_pose_scene_changes_and_smaller_empty_patches_invalidate_correctly():
    q = torch.zeros(2, 3, dtype=torch.float64)
    scenes = torch.tensor([2, 2]); keys = torch.arange(2)
    cache, query = KeepPatchCache(2, q, ('foot',)), Query()
    cache.get(keys, q, scenes, RigidFK(), query)
    q[0, 0] = 1e-12  # No rounded pose hashing.
    cache.get(keys, q, scenes, RigidFK(), query)
    assert query.samples == 3
    scenes[1] = 0
    patch = cache.get(keys, q, scenes, RigidFK(), query)
    assert query.samples == 4 and cache.count.tolist() == [3, 1]
    q[:, 2] = 2
    patch = cache.get(keys, q, scenes, RigidFK(), query)
    assert query.samples == 6 and len(patch.sample) == 0
    cache.get(keys, q, scenes, RigidFK(), query)
    assert query.samples == 6


def test_unallocated_active_contact_still_raises_and_does_not_populate_cache():
    q = torch.zeros(1, 3)
    cache = KeepPatchCache(1, q, ('foot',))
    def bad_query(q, scenes):
        result = observed(q, scenes)
        result['pairs']['constraint_allocated'][:] = False
        return result
    with pytest.raises(ValueError, match='Unallocated'):
        cache.get(torch.tensor([0]), q, torch.tensor([0]), RigidFK(), bad_query)
    assert not cache.ready.any()


def test_actual_trainer_feedback_reuses_prediction_and_keeps_rollback_reset_semantics():
    # Execute the trainer's actual feedback block, including cache population.
    import generator.parallel_rollout_pool as module
    path = Path(module.__file__).with_name('train_full1000_position.py')
    tree = ast.parse(path.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and node.items[0].context_expr.args
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == 'pool_feedback')
    code = compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec')
    bank = {'current_q': torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])}
    pool = ParallelRolloutPool(bank, torch.arange(3), torch.full((3,), -1), 3, max_episode_steps=2)
    pool.indices[:] = torch.arange(3)
    pool.observation = {k: v.clone() for k, v in bank.items()}
    pool.attempts[:] = torch.tensor([0, 0, 1])
    pool.draw_references = lambda n: torch.full((n,), 2, dtype=torch.long)
    scenes = torch.tensor([0, 1, 2]); slots = torch.arange(3)
    query = Query(); cache = KeepPatchCache(3, bank['current_q'], ('foot',))
    cache.get(slots, pool.observation['current_q'], scenes, RigidFK(), query)
    output = pool.observation['current_q']+.1
    prediction = SimpleNamespace(qpos=output.requires_grad_())
    raw = observed(output, scenes)  # Already performed prediction query.
    env = dict(torch=torch, timed=lambda _: nullcontext(), pool=pool, slots=slots,
        indices=pool.indices.clone(), prediction=prediction, queried=(None, raw),
        safe=torch.tensor([True, False, True]), recoverable=torch.ones(3, dtype=torch.bool),
        fresh_observation=lambda q, ids, obs, chosen: {'current_q': q}, reset_reasons={},
        pool_patch_cache=cache, tensor_scene_ids=scenes, model=SimpleNamespace(fk=RigidFK()),
        bind_keep_patch=bind_keep_patch, pending=None,
        cfg=SimpleNamespace(training_stage='legacy_mixed'))
    exec(code, env)
    torch.testing.assert_close(pool.observation['current_q'][0], output[0])
    torch.testing.assert_close(pool.observation['current_q'][1:], bank['current_q'][1:])
    # Accepted state seeded from raw prediction, rejected state retained, reset
    # happens to return to its exact original pose/scene: all three are hits.
    patch = cache.get(slots, pool.observation['current_q'], scenes[pool.indices], RigidFK(), query)
    assert query.samples == 3
    fresh = bind_keep_patch(RigidFK(), pool.observation['current_q'], observed(pool.observation['current_q'], scenes[pool.indices]))
    assert_loss_gradient_equal(pool.observation['current_q'], fresh, patch)
    assert not cache.q.requires_grad


def test_duplicate_missing_reference_queries_only_once_and_grows_storage():
    q = torch.zeros(3, 3)
    cache, query = KeepPatchCache(3, q, ('foot',)), Query()
    keys = torch.tensor([1, 1, 0]); scenes = torch.tensor([0, 0, 0])
    cache.get(keys, q, scenes, RigidFK(), query)
    assert query.samples == 2 and cache.width == 1
    cache.get(torch.tensor([2]), q[:1], torch.tensor([2]), RigidFK(), query)
    assert query.samples == 3 and cache.width == 3
    assert cache.count.tolist() == [1, 1, 3]


def test_resampled_pool_can_reuse_reference_without_new_query():
    q = torch.zeros(3, 3); q[:, 0] = torch.arange(3)
    scenes = torch.arange(3)
    source, query = KeepPatchCache(3, q, ('foot',)), Query()
    source.get(torch.arange(3), q, scenes, RigidFK(), query)
    pool_cache = KeepPatchCache(2, q, ('foot',))
    for ids in (torch.tensor([0, 1]), torch.tensor([2, 0])):
        patch = pool_cache.get(torch.arange(2), q[ids], scenes[ids], RigidFK(), query,
            source=source, source_keys=ids)
        assert_loss_gradient_equal(q[ids], patch, bind_keep_patch(RigidFK(), q[ids], observed(q[ids], scenes[ids])))
    assert query.samples == 3 and pool_cache.query_samples == 0 and pool_cache.source_samples == 4
    # A generated state with the same demonstration ID must not use its patch.
    changed = q[ids].clone(); changed[0, 0] += .3
    pool_cache.get(torch.arange(2), changed, scenes[ids], RigidFK(), query,
        source=source, source_keys=ids)
    assert query.samples == 4
