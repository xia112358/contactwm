# Ten-stage bottleneck handoff training

Handoff allocates every requested environment to selected motion tasks; it no
longer reserves teacher-driven background worlds. A 4096-world run therefore
uses 4096 student trajectories. Original motion IDs are preserved while physical
terrain prototypes are compacted, with live collider validation after creation.
When resuming a run that included background worlds, pass its fingerprinted
`solver-collision.mjb` with `--collision_model` to preserve the checkpoint's
surface-objective contract. A fresh export has a different whole-model hash
after removing unused terrain prototypes; the contract guard remains enabled.

This is the maintained form of the #82–#91 climb_58 experiment, including the
Gaussian bottleneck actor, source/depth-balanced handoff pool, grouped batched
backward, stable native contact sampling, native collision truncation and batched
reset optimization. It does not require scripts under `tmp/`.

All required foot, hand and knee contacts use the same point-navigation distance
to their original world targets. A cached box visibility roadmap approximates
obstructed routes; unobstructed routes use the exact Euclidean distance. Targets
are not raised to a box top, and material-point heights are not replaced by the
lowest mesh height. This is point navigation, not a whole-limb clearance guarantee.
Stable native witnesses and paired material/goal identities are retained; missing
hand contacts use area-normalized soft minimum over surface samples. Penetrating
endpoints use an exit-distance extension without changing stored targets.
Interrupted trajectories retain endpoint body loss. Native completion and
collision termination remain unchanged. Equal-norm balancing and gradient
projection remain disabled.

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

After the initial reference bootstrap and pool validation, successful training
terminals refill the successor pools directly, retaining their physical state,
previous action, sensor context, history and source/depth. Training iterations
do not run a separate reference-producer rollout. This changes the entry-source
distribution from the historical reference-producer experiment; the run config
records `pool_population`, and `producer_seconds` is zero for these iterations.

The recipe enables `--compile_contact_fk` to fuse canonical contact FK forward
and input-gradient operations with Torch Inductor. It retains FP32 and supports
varying microbatch sizes; it does not alter physics or control-step counts.
The first backward can spend tens of seconds compiling, with caches retained
under the configured runtime. Checkpoint tensor formats are unchanged. For an
eager comparison, omit this flag from the backend arguments.

The recipe uses 1024-trajectory microbatches and one combined actor update per
iteration. It leaves `--check_dmo` off: its diagnostic backward currently runs
inside grouped-gradient hooks and would contaminate the first update. This
workaround does not repair that optional diagnostic path.

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
