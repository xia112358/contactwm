# Repository maintenance

## Maintained code and entrypoints

This workstation's large runtime data is held outside the checkout under
`/perception/workspace/zhouxia/<project>/`, reached through the `runtime/current`
directory symlink. A fresh clone can use a local directory or an explicitly
selected external data root through `scripts/bootstrap_runtime.py`. See
[installation](install.md) and [repository data](repository-data.md) for setup,
the code/data boundary and backup requirements. Moving data does not change
baseline weights, contact labels or losses.

| Area | Implementation | Entrypoint / tests |
| --- | --- | --- |
| Assets, FK, contact semantics | `packages/somaforge_core` | `scripts/check_asset_manifest.py`, package tests |
| Read-only motion/research playback | `packages/motion_edit` | `motion-edit review --input PATH`, package tests |
| Predictor | `packages/generator` | `python -m generator.train_full1000_position`, package tests |
| Contact optimization | `packages/contact_solver` | package API; diagnostics under `research/`; package tests |
| Newton WBT | `src/holosoma` | `train_agent.py`, `eval_agent.py`, `replay.py`, `run_sim.py`; `scripts/train_motion_edit_policy_finetune.py` for explicitly supplied references |
| PHP grouped experts | `scripts/train_php_wbt.py` | Explicit group training; use `--help` for current arguments |
| Modular experts and distillation | Native WBT managers and `RoutedDistillPPO` | [Configuration and invocation guide](wbt-modular-experiments.md); dense/sparse/force experts and student |
| Native contact-reference preparation | `scripts/record_wbt_contact_reference.py`, `scripts/combine_wbt_manifests.py` | Independent recording/manifest commands; manually supplied expert checkpoints |

Shared direct Newton collision queries and canonical self-collision filters live
in `somaforge_core.newton_collision` and `somaforge_core.newton_collision_filter`.
Tracking imports the filter directly from core; it does not initialize Motion Edit.
`contact_solver` and Predictor retain their shared core contracts. Canonical FK
and native contact-query helpers are owned by core, so Predictor, WBT preparation
and contact processing do not import the retired trajectory generator. Shared
implementations include `somaforge_core.newton_direct_fk`,
`somaforge_core.canonical_input`, `somaforge_core.native_geometry_release`,
`somaforge_core.rollout_source` and `somaforge_core.event_acceptance`.
The playback package retains motion reading and historical diagnostic rendering.

The old trajectory editing/generation and Infiller entrypoints are retired.
Their implementation is not part of the current training pipeline. Keep next-contact
Predictor, WBT/distillation, contact-control code and shared physical-evidence
contracts intact. In particular, a script name containing `motion_edit` does not
make a retained WBT fine-tuning or dataset utility obsolete.

The historical 2026-10-10 follow-up acceptance record is
`runtime/current/archives/package_simplification_followup_20261010/acceptance.json`.
It records 1093 package tests (plus 3 subtests), 58 entrypoint checks, and bounded
real Tracking training/evaluation, edited-reference policy fine-tuning/evaluation,
and Generator scratch training/evaluation. Training now initializes the existing
Newton API bridge after AppLauncher, matching evaluation; no solver parameters
or Predictor implementation were changed. Smoke checkpoints are temporary
outputs, not new baselines or required inputs. These checks validate working
entrypoints and finite updates, not convergence or full-trajectory acceptance.

Current implementations and ordinary tests import the owning package directly.
`climb00_pipeline` contains aliases for historical imports and module CLI calls;
its module-identity tests and frozen reproduction dependencies remain supported.
It contains no independent model, solver or dataset implementation. It is excluded
from the default workspace, installation and subprocess search paths. Historical
reproduction opts in through its [package instructions](../packages/climb00_pipeline/README.md).

The maintained full1000 trainer uses `generator.training_data`,
`generator.scene_workers` and `generator.research.isolated_gradients`, rather
than importing training scripts from `tmp/`. Query checkpoint, source scene
manifest and model inspection paths are explicit configuration fields.

The current baseline is selected by [`baselines/current.json`](../baselines/current.json):
[`predictor.v1_latest.20261010`](../baselines/predictor_v1_latest_20261010/README.md).
It records v1, loaded-material207_joint_v12 data, keep-patch weight 1 and the fixed
step1000 comparison checkpoint. New scratch runs use a fresh output directory
and do not load that checkpoint. Cleanup does not approve historical data variants.

## Visualization

The supported browser entry is `motion-edit review --input PATH`. It opens a
canonical NPZ motion, JSON diagnostic report or case collection in a read-only
Three.js page. Add `--terrain PATH` when the input does not specify its terrain.
See the [playback documentation](../packages/motion_edit/README.md#read-only-playback).
The packaged page references the current JS/CSS bundle in `web_dist/index.html`.
Historical diagnostic contact flags remain unverified; playback does not infer
contact truth from geometry or force thresholds.

The contact editor and trajectory generation routes are retired. There is no
maintained standalone Viser, workbench server or force-retarget CLI.
Simulation Kit rendering is controlled by AppLauncher.
Remote simulation playback uses the same evaluator with `--visualizer kit` and
`--livestream 1`; see the [modular playback guide](wbt-modular-experiments.md#evaluate-or-replay-separately)
and [installation guide](install.md). TensorBoard reads the selected training run
directory independently of Kit. Browser review does not run the policy or
certify native contact execution.

## Data and retention

| Content | Location |
| --- | --- |
| Formal motion/dataset archives | `runtime/current/motions/` |
| Formal checkpoints | `runtime/current/models/` |
| Local asset, data and acceptance indexes | `runtime/current/manifests/` |
| Tracked asset and training contracts | `configs/` |
| Diagnostics and temporary work | project `tmp/` |
| Frozen experiment source/configuration | `baselines/` |

Root `data`, `logs` and `tmp` are symlinks into `runtime/current/holosoma`.
Required inputs must never live under `tmp/`. Persistent native build caches
live under `runtime/current/cache/`; retained experiment inputs and launch
snapshots live under `runtime/current/archives/`. Temporary outputs may be
removed only after checking references and confirming the itemized list.
Do not rewrite frozen source snapshots to look like current code.

Use [`baselines/current.json`](../baselines/current.json) and
[`baselines/archives.json`](../baselines/archives.json) for current and archived
Predictor records. scratch_recipe.20261002 and control500 were archived in place
on 2026-10-10; their historical weights, data, source snapshots and launch records
remain available. The historical direct-infiller corpus is described in
[its recipe](history/climb00_direct_infiller_baseline.md), not a current training admission.

`runtime/current/manifests/repository_cleanup_20260928.json` is a historical
retention inventory, not the current baseline or a training admission registry.
It predates the explicitly approved 2026-10-02 retirement of corrected1000 and
therefore includes paths that no longer exist. Consult current baseline records
and actual references before any cleanup. Tracked upstream motion/examples also
require dependency-specific review; the audit lists them rather than removing them.

## Cleanup rules and record

Follow the user's approved scope and root `AGENTS.md` for file changes. The
approved retirement removes old trajectory editing/generation and Infiller while
retaining next-contact Predictor. Preserve historical datasets, weights and
research artifacts. Never blanket-clear `tmp/`, discard unrelated untracked work,
stop training, relabel datasets or promote a baseline as a side effect of cleanup.
Keep changes reviewable until a separate commit/push request.

The cleanup records below describe earlier repository states and their checks;
they are not current entrypoint lists or verification results.

Earlier cleanup moved 35 tests to their implementation owners and retired old
Motion Edit handlers, viewers and the LTE adapter. The 2026-09-28 movement and
verification are recorded in
`runtime/current/manifests/repository_cleanup_20260928.json`; the former
`tmp/repository_cleanup_20260928/` diagnostics were later recycled.
That pass verified 755 tests and 3 subtests plus 870 statically parsed Python
files; these counts are historical, not current verification results.

The 2026-10-10 maintenance pass replaces legacy imports in 16 ordinary root tests
with canonical package imports. It removes only the four approved unreferenced
web JS bundles, empty `source/`, root Python/pytest caches and the cache-only
`packages/climb00_pipeline/tests/` directory. Compatibility tests, Predictor
implementation, current baseline, formal data and checkpoints remain intact.
Caches may be recreated by normal tooling. Movement and verification records
for this pass are historical; the former `tmp/repository_cleanup_20261010/`
diagnostics were recycled in the separately confirmed cleanup. Verification passed
85 affected/compatibility tests, statically parsed 959 Python files without errors,
and validated both the asset and training manifests.

The historical 2026-10-10 reorganization kept Tracking, Motion Edit and
Generator interfaces. Motion Edit generation was subsequently retired. Historical reports now live under
[docs/history](history/README.md). The old direct-infiller launcher and unused
editor-cut forwarding module were moved to the recycle bin after an itemized
confirmation. CI runs `scripts/check_repository.py` instead of the removed
GMVQ checker; it covers package, test, app and baseline changes. The audit also
rejects retired entrypoints, default legacy search paths and missing workflow
scripts. The former `tmp/repository_reorganization_20261010/` diagnostics were recycled;
current retained verification is indexed in
`runtime/current/manifests/repository_tmp_cleanup_acceptance_20261010.json`.

## Verification

Current regression covers maintained implementations, shared data contracts,
checkpoint compatibility and read-only playback. Tests dedicated solely to
`generator.research.archived_experiments` and the preserved dropout207 training
corpus were retired on 2026-10-30. Historical source, data and acceptance reports
remain archives. Shared dropout handling lives in the core test suite, historical
contact-plan schemas retain their round-trip/validation checks, and one centralized
command-retirement test prevents removed editing/generation entrypoints from
returning. A historical test name alone is not a reason to drop active coverage.
The follow-up pass also drops old editor jitter/cut and pinned step14/recovery
experiment tests. Production pose-coordinate checks import their owning module
directly; current projection, native evidence, segmentation, and storage tests
remain in the maintained suite.

Static source/config/import audit and manifest checks:

```bash
python3 scripts/check_repository.py --output tmp/repository_audit.json
source scripts/source_somaforge.sh
python scripts/check_asset_manifest.py
python scripts/check_training_manifest.py
```

Package and contact-pipeline tests require the project's Newton environment:

```bash
source scripts/source_isaaclab3_newton_setup.sh
OMP_NUM_THREADS=2 PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -p no:cacheprovider \
  packages tests/test_newton_contact_pipeline.py -q
```

Root regression tests import their owning packages directly. Preserved experimental
helpers live under `generator.research.archived_experiments` and
`contact_solver.research`; they do not expose old training CLIs. The teacher
loader test creates its own synthetic fixture, and does not pass silently when
an external dataset is missing. Local historical corpus integration tests read
explicit formal archive paths, not `tmp/`.

For the complete CPU regression batch, use importlib collection to avoid same-name
test modules and expose historical fixture modules only to the test process:

```bash
source scripts/source_isaaclab3_newton_setup.sh
test_run="tmp/cpu-regression-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
mkdir -p "$test_run"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/scripts:$PWD/tests:$PWD/packages/generator/tests:$PWD/packages/motion_edit/tests:$PYTHONPATH" \
python -m pytest packages tests src/holosoma/holosoma --ignore=tests/e2e \
  --import-mode=importlib -m 'not isaacsim and not multi_gpu and not requires_inference' \
  -p no:cacheprovider -o tmp_path_retention_policy=all --basetemp="$test_run/pytest" -q
```

Some historical corpus tests require separately restored datasets. Loopback/thread
access is needed for browser-service tests. GPU checks and end-to-end training are
separate from this batch.

The temporary-input migration preserves motion/checkpoint bytes and frozen
source snapshots. Two registered Tracking manifests now live under
`runtime/current/motions/wbt/`; their relative and absolute input closure is
preserved. JSON path edits also refresh associated file hashes. Archived labels
and preprocessing caches are not re-certified by this migration.
The complete local source/target hash ledger is
`runtime/current/archives/preserved_inputs_20261010/migration.json`.

The repository audit rejects temporary-source imports, temporary data/model
reads and temporary persistent caches. Temporary diagnostic output is allowed.
Tests are also checked with workspace `tmp/` reads explicitly denied.

After the separately confirmed cleanup, 2807 migrated originals (about 3.8 GB)
were moved with `gio trash`; each has a verified replacement and system-trash
receipt. Verification passed 1091 package tests, 247 root unit tests and 19
Tracking preset tests, plus 3 subtests, with workspace temporary inputs denied.
Eight maintained CLI help checks, native Coal loading and both asset/training
manifest validators passed. Training E2E workflows are outside this cleanup
verification. The persistent verification and receipt ledger is
`runtime/current/manifests/repository_tmp_input_migration_20261010.json`.

IsaacLab/CUDA tests require the host environment. Static/unit checks do not
establish trajectory acceptance or policy execution quality.

For modular WBT changes, the focused CPU regression set is:

```bash
python -m pytest tests/test_wbt_modular_experiments.py \
  src/holosoma/holosoma/config_values/tests/test_experiment_presets.py \
  src/holosoma/holosoma/utils/tests/test_eval_utils.py -q
```

These cover configuration, student observation boundaries, teacher routing,
contact freshness and checkpoint reconstruction. Short GPU smoke runs additionally
checked dense/sparse/force updates, distillation, independent evaluation and exact
model/optimizer/probe-history restoration. They do not establish convergence or
full-size throughput. New native contact labels use constraint activation plus
allocation and shared surface selection; older eight-part threshold/hysteresis
masks remain historical force evidence and are not silently certified.


The historical 2026-10-10 Motion Edit detail cleanup unified editor and CLI generation through
`motion_edit.generation.jobs.generate_candidate`, removed the retired segment
output setting and obsolete solver kwargs, and rebuilt saved plans from verified
source contacts without copying labels to the candidate. Historical object-typed
text arrays were normalized without changing any numeric motion arrays; the
original is archived under
`runtime/current/archives/motion_edit_interface_cleanup_20261010/` and was moved
with `gio trash`. Generated versions remain playback candidates until their own
native validation. That historical cleanup did not change Predictor data, loss or weights.
The generation entrypoint described in this record is now retired.
