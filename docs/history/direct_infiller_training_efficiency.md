> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Infiller training efficiency

> Historical experiment: the Infiller implementation and entrypoints were retired
> on 2026-10-23. Data, checkpoints and the original results below remain archived.


The first boundary-velocity experiment was stopped by explicit user request
during initial validation, before trajectory optimization. Its logs and selected
timing artifact remain in place; pipeline status is `stopped_by_user`.

## Measured result

The benchmark fixes the same weights and nine training batches (78 segments:
demonstration, accepted and recursive sources, each at three reference lengths).
It performs forward/backward without optimizer updates. Two measured passes
follow warmup. Test examples are excluded. Wall timing synchronizes only at
the beginning/end of a pass; component measurements are separate.

| Configuration | Seconds / 9 batches | Batches / second |
| --- | ---: | ---: |
| Original, 128-frame query chunks | 11.578 | 0.777 |
| Cached/batched decoder, 1024-frame query chunks | 6.224 | 1.446 |
| Plus lossless packed CUDA IPC | 6.033 | 1.492 |

Selected configuration is **1.919x** faster on this cohort, reducing measured
time **47.9%**. This is fixed-batch forward/backward throughput; complete-run
speedup, validation overhead and cold cache construction are not included.
The optimizer, learning rate, sampling/exposure schedule, loss weights and
checkpoint selection remain the same. Profiling uses a smoke checkpoint to
hold weights fixed; it does not establish trained trajectory quality.

## Changes and preserved contracts

- Use the existing exact-tensor FK cache across the native witness, target
  interval and contact-region computations. Complete-solid float64 rigid FK
  stays separate, with its original orthonormalization and GJK/EPA backend.
  Cache lifetime is one physical chunk and releases in a `finally` block.
- Cache only immutable boundaries, scenes, contracts and independently frozen
  frame-head outputs. Frozen counts are computed with the legacy scalar head
  path, preserving native rounding near half frames. Generated placeholder
  interiors never enter this cache or supply velocity targets/history.
- Decode independent predecessor chains by depth in batches; recurrence
  dependencies and per-update history detachment stay intact. Decode current
  seam pairs in batches, retaining gradients through both sides.
- Increase trainer query chunks from 128 to 1024 frames. The actual initialized
  Newton world count stays **16**, so scene fingerprints/assets/margins remain
  unchanged. No physics samples or raw contact fields are dropped.
- Audit native activation/allocation directly on the field tensors, downloading
  four error flags rather than full contact arrays. The same CONSTRAINT bit,
  actual distance/include-margin and all required constraint addresses apply.
  Missing, invalid and unallocated fields still fail explicitly.
- Optionally pack every exported CUDA field into buffers by exact dtype, then
  clone owned buffers and unpack shaped views. This reduces IPC allocation
  handles without changing dtype, precision, fields or solver computation.
  Consumer synchronization still precedes the producer lifetime acknowledgement.
  Legacy ungrouped responses remain supported.

Enable with `--efficient-training --query-batch-size 1024`. Default behavior
remains unchanged for old recipes. Frame parameters must be frozen and timing
decoupled before duration caching can be enabled for training.

## Validation

All nine actual Newton batch comparisons passed for predicted poses, unified
losses and complete parameter-gradient vectors. Maximum absolute differences
were 2.623e-6 for poses, 3.815e-6 for losses and 4.197e-5 for gradients. These
are floating-point batching differences, not a claim of bitwise identical
trajectory training. The comparisons do not use test data or select weights.

Ninety-one related unit tests passed. They include scalar/batched prefix and
seam gradients, frozen scalar frame counts, numpy/tensor activation parity,
lossless dtype/shape/value packing, empty contacts, owned-copy lifetime and
acknowledgement ordering. The four-update actual Newton optimizer smoke passed
with finite gradients and bitwise unchanged independent timing parameters.
Its selected validation score was 2.102242726 versus 2.102242875 in the original
equivalent smoke. This verifies the short optimizer path, not full training
outcome equality.

The formal optimized restart retains the same 1500-update trajectory recipe
and all eight planned train/test/rollout stages, in fresh directories:
`tmp/infiller_boundary_velocity_efficient_v1_20261011/run_pipeline.py` and
`runtime/current/models/generator/direct_infiller_boundary_velocity_efficient_v1_20261011`.
The interrupted original run remains retained. All eight stages completed.
The trajectory training timer reports **2114.2 seconds (35.2 minutes)**,
including its initial/intermediate/final validation; setup and velocity warmup
precede this timer. Three intervals excluding validation measured
**1.35–1.36 updates/second**. Because the original run stopped before trajectory
updates, there is no original complete-run timing to establish a full-run
speedup.

The new boundary-velocity architecture regressed against its decoupled-time
parent: the three 30-event recursive runs passed 30/90 segments under the
original endpoint-and-interior criterion versus 45/90, with worst interior
depth 131.5 mm versus 58.4 mm. Fixed-one-second runs passed 13/90 versus 45/90.
These are architecture/trajectory-training comparisons, distinct from the
same-weights efficiency equivalence checks above. The new checkpoint remains
experimental and does not replace its parent. Complete quality results are in
[direct_infiller_boundary_velocity.md](direct_infiller_boundary_velocity.md).

Evidence: `tmp/infiller_efficiency_20261011/comparison_summary.json` and its
referenced complete profiling reports. Profiling entrypoint uses the regular
AppLauncher-first CLI:

```bash
python -m generator.train_frozen_predictor_infiller --headless --device cuda:0 \
  --mode profile --objective unified --corpus <corpus.pt> --checkpoint <best.pt> \
  --report <new-report-directory> --batch-size 16 --frame-budget 512 \
  --query-batch-size 1024 --seam-weight 1 --profile-repeats 1
```
