# Ten-stage bottleneck handoff training

This is the maintained form of the #82–#91 climb_58 experiment, including the
Gaussian bottleneck actor, source/depth-balanced handoff pool, grouped batched
backward, stable native contact sampling, native collision truncation and batched
reset optimization. It does not require scripts under `tmp/`.

The obstacle-aware whole-foot distance is used for the box-top foot transfers
(#90 right foot, #91 left foot). Ground foot goals, hand/knee targets, body terms,
regularization and completion criteria retain the existing recipe. Equal-norm
balancing and gradient projection remain disabled.

## Install and run

First install the configured SuperMimic/Newton workflow environment as described
in [the current workflow](current-workflow.md). This requires CUDA and the
project's canonical G1 assets; a Python-only installation is insufficient.

The three starting models already exist in the
[checkpoint release](checkpoints.md). **The companion 40 MB task-data release is
pending explicit publication approval.** Until that release is published, a new
machine must obtain `handoff-data.tar.gz` from its owner and place it in the asset
directory; the installer verifies exactly the same SHA256 before extraction.

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
