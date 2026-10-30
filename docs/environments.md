# Native SomaForge runtime compatibility

For the existing local server, the validated additive venv is described in
[current workflow](current-workflow.md#本机运行环境与启动验证). It currently
serves both workflow interpreter entries. The fully pinned native installation
below is an alternative recipe, with its own version catalog.

For the merged project's main flow and runtime selection, see
[current workflow](current-workflow.md): SuperMimic tracking → collect → DAgger
→ contact uses its own interpreter. Original PPO tracking remains independent;
its downstream integration is deferred. Predictor remains available.

This page describes the native `main` environment at
`runtime/environments/conda/somaforge`: WBT, distillation, Predictor,
read-only motion playback and native contact workers.
[Installation and dependency ownership](python-environments.md) are defined by
`configs/environments/profiles.json` and the main lock.

The current local source checkout is `somaforge-unified`; `main` here names a
runtime profile. The version table below applies to that native profile, not
the separate SuperMimic environment. Old trajectory editing/generation, its IK
workers and Infiller are retired. Shared Newton FK, contact processing and
Predictor remain maintained.

| Component | Pinned version |
| --- | --- |
| Python | 3.12 |
| Isaac Lab | 3.0.0, `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` |
| Isaac Sim / Torch | 6.2 / 2.11 |
| Newton / Warp | 1.7.0 / 1.18.0 |
| MuJoCo / MuJoCo Warp | 3.15.0 / 3.15.0 |

```bash
python3 scripts/bootstrap_runtime.py
python3 scripts/manage_environment.py create --profile main
python3 scripts/manage_environment.py check --profile main
source scripts/source_isaaclab3_newton_setup.sh
python scripts/train_php_wbt.py --help
packages/motion_edit/motion-edit --help
```

`setup_all.sh` and `setup_isaaclab3_newton.sh` are compatibility entrypoints to
this same installer. The activation script uses main. Bare
`source_somaforge.sh` sets workspace
paths only. Native worker launchers resolve the managed main interpreter and
its pinned Lab snapshot without inheriting the retired overlays.

Use `--visualizer kit` for a window. With no visualizer argument the launch is
headless; explicit `--headless` takes precedence. Official AppLauncher arguments
such as `--device` and `--enable_cameras` are parsed before project configuration.
Read-only browser playback uses `motion-edit review --input PATH` in the same
managed environment. Canonicalization and native contact queries use retained
core helpers; there is no trajectory-generation worker to launch. The standalone
Retargeting package and its bundled models/examples were retired on 2026-10-23;
current input data is supplied separately.

## Checkpoints and rotations

Root/body states, motion tensors and checkpoint observations remain XYZW.
Explicit WXYZ asset configuration is converted only at USD/Lab boundaries.
Canonical serialized motion files use their declared quaternion convention;
do not infer NPZ ordering from the simulator's internal tensor convention.
The historically named depth-camera `offset_rot_wxyz` field was previously passed
directly to an XYZW sensor. Its default `rotation_convention="legacy_xyzw"`
preserves that physical camera frame for existing checkpoints. New camera
configurations can explicitly select `rotation_convention="wxyz"`; do not change
this on resume without intending to change the observation frame.

Normal model, optimizer, normalization, progress and sampler checkpoint formats
are retained. No simulator snapshot is added. Resume into a separate output
folder when validating an existing run. The upgrade does not restart running
processes; new processes use the environment selected at launch.

The new [modular expert/student workflow](wbt-modular-experiments.md) uses the
native `DistillPPO` checkpoint format. Its student networks/layout differ from the
historical PHP adapter under local `tmp/`; those student checkpoints still require
their original entrypoint. Expert checkpoints must match their canonical asset,
action controls and saved observation/network configuration.

The canonical USD bundle is installed from `assets/g1_spherehand_usd/` with its
existing hashes. `scripts/convert_g1_spherehand_usd.py` is a deliberate asset-build
tool, not a fresh-install prerequisite. Re-running an importer may change USD
bytes even with the same URDF; do not relax fingerprint checks or relabel existing
datasets/checkpoints to accept an unverified conversion.

Newton's native collision guard and canonical G1 asset rules remain active.
The shared adapters handle renamed Lab/Newton APIs; it does not change collision margins,
contact activation or the robot asset. Historical experiment reports retain
their original environment paths and are not current setup instructions.

## Main consolidation validation (2026-10-22/23)

This section records historical validation before the editor/generator was
retired. Its test counts and component results do not describe the current
cleanup. At that time, installers, activation scripts, the editor and native query
workers resolved the managed `main` profile. On the validation host, all 12 confirmed
legacy runtime directories were moved to system trash on October 23. The project
runtime directory now contains only `conda/somaforge`; the global Miniforge named
environment directory contains no runtimes. Stale Conda registry entries were
backed up and removed; its remaining project prefix is main. Recovery receipts are retained under
`tmp/main_unification_20261023/trash_receipts.json`. Other installations must
inventory their own prefixes before an itemized trash operation.
The Miniforge base installation supplies the environment manager and is not a
second SomaForge runtime.

Container builds use `docker/Dockerfile` for all maintained image names.
The old Isaac Sim Dockerfile path retains a matching compatibility recipe
and also installs only main; it does not create a separate environment. The container recipe and build-command dry run were checked, but
the images were not built during this consolidation. Local execution is the
acceptance scope; Docker is not a required validation target.

| Component | Verification in main |
| --- | --- |
| Dependencies and sources | Managed ownership, exact versions, source snapshot and isolation audit passed |
| CPU regression | After legacy cleanup, 2,186 tests and three subtests passed with five skipped; 79 focused tests also passed on October 23 |
| WBT and distillation | Three expert short updates; student optimizer updates and fresh-process resume changed actor and critic parameters |
| Historical Motion Edit generation (retired) | Original six-task batch reproduced and compared with Torch/Newton; the then-current UI used main |
| Historical Retargeting (retired) | Three-frame solver run; canonical MuJoCo FK matched Torch FK on nonzero poses |
| Predictor | Fresh current-Newton labels/cache; one optimizer update and single-step diagnostic evaluation completed |
| Historical Infiller (retired) | Full 7,271-segment corpus rebuilt; small-corpus update (eight optimizer states) and evaluation completed; all parameters finite, frozen Predictor unchanged with no gradients |
| Kit | Final-source windowed evaluation loaded the resumed student checkpoint, completed four steps with one environment and exited normally; repeated successfully after legacy environment cleanup |

These are execution and compatibility checks. The one-step Predictor was not
trained to acceptance, and its diagnostic rollout did not complete a chain.
The six-task Torch/Newton run took 227.195 seconds versus 223.456 seconds for the
frozen JAX run; this migration does not establish a speed improvement. Long
training and policy convergence remain unqualified. Docker builds are outside
the local execution acceptance scope.

Local evidence is under `tmp/main_component_validation_20261022/`,
`tmp/main_unification_20261022/` and `tmp/main_unification_20261023/`.
The archive catalog registration was reconciled with its preserved October 17
pre-registration copy and all 621 registered motion/contact/event file hashes.
The asset manifest records both catalog hashes and retains the original
training-admission status.

## Generator inputs for main

`configs/generator/main_runtime.json` binds the existing frozen v1 step-1000
Predictor to explicitly rematerialized Newton 1.7 inputs. It records the parent
baseline and hashes of the weights, manifest, data cache, pose cache and query
dependencies. Historical baseline records and results retain their original
inputs.

Predictor defaults select
`runtime/current/motions/climb00_contact_dataset/training/predictor_main_20261022/`:
207 unchanged source motions, 144,900 native frames and 3,105 events (1,470 train,
780 validation, 855 test). Contact labels, event evidence and cache certificates
come from actual Newton queries and the shared primary-face selector. Their
static-query sampling does not certify executed support or dynamic stability.

To reproduce these inputs, invoke `scripts/rebuild_predictor_contact_data.py`
through `manage_environment.py run --profile main`, supplying the source
manifest, query checkpoint/source manifest/source model, a new `--output` and a
new `--worker-output` under project `tmp/`. The existing build refuses to
overwrite an output directory. Its completed invocation and source-integrity
audit are retained with the local validation evidence.

Infiller was retired on 2026-10-23. Its corpus, checkpoints and immutable
release records remain archived. The former build and validation evidence is
preserved in [the historical record](history/infiller-main-validation-20261023.md).

## Earlier migration validation (2026-10-20)

The production environment passed old checkpoint `03913` → `03914` → fresh-process
`03915` resume checks with exact model/normalization, optimizer, sampler and gate
restoration. The original training entry then resumed `03915` and saved `03916`.
These checks used 8 environments and short updates; full-size long-running
training and policy-quality equivalence have not been qualified.

The full regression run passed 2035 tests, skipped 3 and retained 6 Generator
failures also reproduced in the old environment (missing mock `workers`,
undefined `cfg`, and a gradient audit treating a boolean as a tensor).
The final focused state/camera/worker regression passed 42 tests; launcher parsing
passed 6 cases. G1 GPU validation checked 29 joints, XYZW round trips and 30 active
contacts with no unallocated constraints. Actual RGB recording and a 50-step
Kit student playback passed. The Docker recipe was updated but not built.

Local machine-readable evidence and logs (not distributed with a Git clone):
[`tmp/isaaclab_ga_20261020/production_status.json`](../tmp/isaaclab_ga_20261020/production_status.json).
