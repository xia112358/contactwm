# Ten-stage bottleneck handoff training

This is the maintained form of the #82–#91 climb_58 experiment, including the
Gaussian bottleneck actor, source/depth-balanced handoff pool, grouped batched
backward, stable native contact sampling, native collision truncation and batched
reset optimization. It does not require scripts under `tmp/`.

All required contact parts now use the same obstacle-aware distance to their
own material goals. Feet retain their four material targets; hands retain their
native stable contact-point targets, using their collision surface for guidance
when no stable witness exists; knee candidates retain individual local/world
pairs and stable native candidate selection. Missing-witness guidance may choose
any complete valid pair. Targets are never raised to a box top or swapped across
materials. The geometry supports multiple oriented boxes, floor paths and
floating-box underpasses. Native contact/stability/completion logic is unchanged.
Body terms, regularization, sampling and grouped gradient processing retain the
existing recipe; equal-norm balancing and projection remain disabled.

The WM now shares its three 2048-wide hidden layers between state prediction
and an 18-component force projection. Both supervised fitting and DMO backward
compute the common trunk once. Force labels, heading coordinates, the 200 N
scale, SmoothL1 weight, native stable-contact sampling and success criteria are
unchanged. On first transfer from an independent-head checkpoint, state weights
and normalization are preserved exactly, while the shared force projection is
initialized to zero and the WM optimizer is rebuilt. The new force projection
needs fitting before its prediction accuracy can be evaluated; the transfer
itself does not preserve the old force predictions. Actor and actor-optimizer
state are unaffected. Subsequent shared-head resumes restore the complete WM
and optimizer normally. The checkpoint records the architecture and transfer.

## Install and run

First install the configured SuperMimic/Newton workflow environment as described
in [the current workflow](current-workflow.md). This requires CUDA and the
project's canonical G1 assets; a Python-only installation is insufficient.

The three starting models are published in the
[checkpoint release](checkpoints.md). The companion
[40 MB task-data release](https://github.com/xia112358/contactwm/releases/tag/handoff-ten-stage-20261112-v1)
contains the exact recording, cuts, collision model and task assets.
The preparation command downloads and SHA256-verifies all required inputs.

From the repository root:

```bash
python3 scripts/prepare_handoff.py --assets /external/contactwm/handoff
python3 scripts/run_handoff.py \
  --assets /external/contactwm/handoff \
  --output /external/contactwm/runs/handoff-100 \
  --gpu 0 --iterations 100
```

Use `--dry-run` to inspect the command without starting a simulator; `--python`
and `--resources` select the existing workflow runtime. The default starts from
the intact single-stage iteration-154 checkpoint and creates a fresh handoff
pool. This is 100 additional iterations, not random policy initialization.
Original checkpoint/recording/slice bytes are unchanged. Installed explicit path
aliases relocate provenance paths; the original metadata, slice hashes and
physics/robot checks remain enforced. A separate `--single_stage_anchor` is
passed explicitly to the backend instead of loading a hidden experiment path.

Output includes ordinary resumable model/optimizer checkpoints, handoff pools,
`metrics.jsonl`, `evaluation.jsonl` and `status.json`. Initial source/depth pool
validation precedes training. Complete-chain evaluation occurs every ten
iterations; metric rows are written after that evaluation finishes.

## Verification

The batch reset implementation was checked against the original reset on 4,085
student worlds (4,096 total environments): all 13 reset-boundary fields matched
exactly. The 24-frame collection benchmark decreased from 36.36 to 15.00 seconds.
These are collection measurements, not an end-to-end training speed guarantee.
The old implementation's full repeated rollouts differ after integration, so
reset-boundary equivalence is not a claim of bitwise full-trajectory equivalence.

The download code validates archive and checkpoint hashes and rejects archive
path traversal and non-file/non-directory members. Those checks are at the
external artifact boundary; no new contact thresholds or fallback rules are
introduced.

Contact loss reuses one FK result per evaluated state for surface guidance,
stable witnesses and material targets. The standard recipe enables
`--compile_contact_fk` (FP32, dynamic batches, no CUDA graphs); omit this backend
flag for eager execution. No optimizer or sampling settings change.
An isolated old-loss benchmark on the same fixed 1024-sample replay measured
3.131 s baseline, 1.924 s with FK reuse, and 1.969 s with FK compilation alone.
Compilation requires a one-time warmup; these timings exclude it.

The repaired navigation distance compares complete paths, including continuous
solid traversal with a smooth depth-dependent density: `1 + 9 * (depth / (depth + 0.05))**2`. The extra density and its first derivative vanish at the surface, while deep interior travel remains expensive. The 0.05 m scale matches the existing objective distance scale; it is not a contact threshold. It
does not project queries to a single nearest exit. This is a finite weighted
optimization metric, not a guarantee that every returned route is collision
free. Overlapping solids are integrated as a union. Each foot additionally
uses oriented bounds derived from its canonical collision mesh: corner detour
cost and separating-axis overlap cost catch body overlap missed by material
points alone. The bound's overlap at the original target pose is retained,
so neither the material targets nor native solver contact criteria change.

Navigation query efficiency (fixed 1024-sample replay, no updates): evaluate only
the existing per-frame contact-loss mask, skip overwritten legacy surface loss,
batch material and bound-corner queries, and compile segment costs only. Route
selection stays eager because whole-query compilation produced nonfinite
gradients. The final replay benchmark measured 7.694 s versus 2.295 s; loss
differed by 1.91e-6 and merged gradient relative L2 by 1.28e-5. This is BPTT
compute timing, not a new end-to-end training result.


With `--project_entry_updates`, the actual Adam step is projected onto the same
per-entrance half-space constraints. CUDA solves the complete nonnegative dual QP on the GPU. An incremental
active-set implementation is retained for comparison and checks every entrance
after each restricted solve, adding newly violated constraints. GPU numerical KKT
tolerance is `5e-8` for normalized rows; no task or success-class exemptions are
used. The CPU reference solver is retained for validation. No training restart
is performed by this optimization.
