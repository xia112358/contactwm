# PHP Stage 1 checkpoint and native contact references

Release: [php-stage1-20261025-v1](https://github.com/xia112358/somaforge/releases/tag/php-stage1-20261025-v1).
Use the matching Git tag for the PHP student implementation. This is the
contact-conditioned PHP tracker, not a completed three-stage ACP reproduction.

## Download

From a checkout of this tag, prepare the runtime layout using the repository
installation instructions, then run:

```bash
python3 scripts/bootstrap_runtime.py
python3 scripts/download_data.py fetch php-stage1-checkpoint \
  --catalog configs/php_stage1_20261025_release.json
python3 scripts/download_data.py verify php-stage1-checkpoint \
  --catalog configs/php_stage1_20261025_release.json
```

The checkpoint component automatically includes the reference and teacher input
dependencies. To download only the reference data, select
`php-stage1-native-references`. This release contains runtime-required evidence;
bulk extraction intermediates and source execution recordings are not included.

Archives and every installed file have SHA256 checksums. Existing files with
different bytes are rejected. The installer records old-machine path aliases in
`runtime/current/manifests/published_data_paths.json`; embedded evidence,
checkpoint tensors, manifests and their original hashes remain unchanged.
Use the loaders from this tag, rather than opening embedded absolute paths
directly. Do not rewrite the original manifests to relocate data.

## Entry points

| Artifact | Installed path |
| --- | --- |
| Student checkpoint | `runtime/current/models/php_continuous/low_climb_20261024_112654/PHPSegmentDistill/20261024_112701-low_climb_continuous_20261024_112654-locomotion/model_07000.pt` |
| Student native configuration | `holosoma_config.yaml` beside the checkpoint |
| 40-task / 775-segment manifest | `runtime/current/motions/php_stage1_20261025_v1/prepared/train.json` |
| Explicit teacher specification | `runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json` |
| Native reference motions and labels | `runtime/current/motions/php_stage1_20261025_v1/native/motion_000000/` through `motion_000039/` |
| Original extraction manifest | `runtime/current/motions/php_stage1_20261025_v1/extraction/dataset_manifest.json` |
| Evaluation report | `runtime/current/motions/php_stage1_20261025_v1/evaluation/stage1_7000_acceptance_20261025.json` |

Each native motion directory contains the canonical motion, `contacts.npz`,
clock alignment and provenance. Native labels retain complete point-pair and
abnormal-face evidence, part/surface mappings, sampling semantics and robot asset
fingerprints. The prepared directory contains the student targets and the
unchanged authored event segments.

With the main simulation environment installed, validate inputs without starting
training:

```bash
python3 scripts/manage_environment.py run --profile main -- python -B \
  -m holosoma.php_contact.distill \
  --manifest runtime/current/motions/php_stage1_20261025_v1/prepared/train.json \
  --teacher-spec runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json \
  --segment-mode continuous --pure-distill-iters 200 --regularizer-mse-coef 0.1 \
  --validate-only --headless
```

Evaluate the frozen student from all 40 trajectory heads:

```bash
python3 scripts/manage_environment.py run --profile main -- python -B \
  -m holosoma.php_contact.evaluate_trajectories \
  --manifest runtime/current/motions/php_stage1_20261025_v1/prepared/train.json \
  --teacher-spec runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json \
  --checkpoint runtime/current/models/php_continuous/low_climb_20261024_112654/PHPSegmentDistill/20261024_112701-low_climb_continuous_20261024_112654-locomotion/model_07000.pt \
  --output-dir tmp/php_stage1_evaluation --headless --device cuda:0
```

Choose a new output directory for each evaluation. This loads inference weights
and normalizers without optimizer updates. The published checkpoint itself is
the original training checkpoint, retaining actor/critic, optimizer, normalizer,
configuration, iteration and sampler/statistics state; it is not a simulator
snapshot or a weights-only export. For continuation use the same manifest,
teacher specification, continuous mode, 200-iteration distillation schedule and
0.1 MSE regularizer with the `distill` entrypoint and `--checkpoint` above.

## Scope and contact meaning

The student was trained on all 40 low-climb trajectories, with 775 action
segments and 4096 parallel environments. Its first 3500 updates used independent
segments; the selected checkpoint is iteration 7000 after continuous training.
The explicit original teacher is low-climb checkpoint 03600, also included with
the 40 inputs selected by its saved manifest (the dataset directory is named
`motions-low-climb-76`).

Reference labels were measured by Newton/MJWarp at fixed poses without
integration. They represent contact intentions on the fitted reference, not
verified dynamic support, load or slip. Activation requires the actual
`CONSTRAINT` flag, `dist < includemargin` and allocated constraint rows. The
shared main-face selector retains ground and upward horizontal tops and records
side-face contacts as abnormal. Publication does not change `training_ready`,
`physical_execution_verified`, collision margins, contact definitions or cuts.

The live student uses actual solver contacts under the same eligibility rule.
Two deterministic 40-task evaluations completed 33 and 34 trajectories. These
are training-task evaluations, not held-out generalization or paper-equivalent
success rates. The second run matched contact flags at 333/699 reached segment
ends, or 619/699 within an offline +/-100ms diagnostic window. Time windows do
not change execution or define a new contact success condition.

The saved evaluation report documents a pre-existing indexing issue in
`limb_error_m`; that field is excluded from acceptance conclusions. Contact-point
placement errors in the separate diagnostic report use actual native points.
Canonical G1 source and converted assets come from the tagged code's pinned
robot asset contract. Upstream PHP LICENSE/NOTICE are included under `upstream/`.
