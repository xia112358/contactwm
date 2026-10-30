# ContactWM

ContactWM brings the current G1 tracking and contact workflow together: **SuperMimic tracking → native contact collection → DAgger
initialization → contact training** using imagination or DMO updates.

The original optimized PPO tracking code remains available as an independent
baseline. Connecting PPO to this downstream flow is deferred; its checkpoints
are not SuperMimic teachers or world models. Next-contact Predictor remains
available through its existing training entrypoint. The old trajectory
editor/generator and Infiller are retired; shared Newton FK, contact processing
and read-only playback remain.

Runtime profiles select interpreters and external data roots through
`configs/workflow_resources.json`. The validated local setup uses an additive
`runtime/environments/venv/workflow` environment over the existing SuperMimic
simulation libraries; create it with `python3 scripts/setup_workflow_environment.py`.
The native catalog's `main` profile remains a separate fully pinned installation
recipe. See [startup validation](docs/startup-validation.md) for no-training
GPU checks. Original checkouts and Git histories are archived; their old workspace names
are resource compatibility links. External artifacts retain their original history.

## Start here

- [Current workflow and commands](docs/current-workflow.md): tracking, collect, initialize, contact and Predictor
- [Native PPO/Predictor installation](docs/install.md), [Python environment setup](docs/python-environments.md) and [compatibility](docs/environments.md)
- [SuperMimic package and original runtime recipe](packages/supermimic/README.md)
- [Download training inputs, reference models and extracted demonstrations](docs/data-downloads.md)
- [Read-only browser playback](packages/motion_edit/README.md)
- [Predictor baseline and reproduction records](baselines/README.md)
- [Current main-environment Predictor inputs](docs/environments.md#generator-inputs-for-main)
- [Local data layout](docs/repository-data.md), [data policy](docs/data-policy.md),
  and [repository maintenance](docs/repository-maintenance.md)

From this checkout, inspect the configured commands without launching simulation:

```bash
python3 scripts/run_workflow.py --help
python3 scripts/run_workflow.py tracking --dry-run -- --help
python3 scripts/run_workflow.py collect --dry-run -- --help
python3 scripts/run_workflow.py initialize --dry-run -- --help
python3 scripts/run_workflow.py contact --dry-run -- --help
python3 scripts/run_workflow.py tracking --backend ppo --dry-run -- --help
python3 scripts/run_workflow.py predictor --dry-run -- --help
```

The launcher does not install dependencies or start subsequent stages
automatically. Follow the [current workflow guide](docs/current-workflow.md) to
select matching task manifests, checkpoints and physics settings. Training
inputs and checkpoints remain external; the [component downloader](docs/data-downloads.md)
documents the separately published native SomaForge data.

## Layout

```text
src/holosoma/               WBT training, evaluation, and simulator integration
packages/supermimic/        Current tracking, native collection, DAgger and contact control
packages/motion_edit/       Read-only motion/research playback and data utilities
packages/generator/         Predictor, datasets and prediction training
packages/contact_solver/    Contact objectives, collision gradients and projection
packages/climb00_pipeline/  Optional historical import/CLI compatibility
packages/somaforge_core/    Shared assets, FK, contracts and Newton contact semantics
OmniRetarget_Dataset/       Imported preprocessing tools
scripts/run_workflow.py     Unified stage selector; original backend arguments pass through
runtime/current/           Native SomaForge runtime alias
runtime/supermimic/         SuperMimic external runtime alias
runtime/stage2/             Retained Stage 2 artifacts
configs/workflow_resources.json Per-runtime interpreter and external resource registry
configs/assets_manifest.json Manifest-first registry for every runtime asset
configs/training_pipeline_manifest.json Shared training data contract
```

See [repository maintenance](docs/repository-maintenance.md) for active entrypoints,
historical compatibility, data retention and validation commands.
Past experiment reports are separated under [docs/history](docs/history/README.md).

## Canonical assets

Every G1 path uses:

```text
src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf
```

Native SomaForge inputs use `configs/assets_manifest.json` and
`configs/training_pipeline_manifest.json`; SuperMimic uses its verified task
manifest and checkpoint metadata. The workflow resource registry selects runtime
locations, not artifact compatibility. Each backend retains its canonical robot
identity and input checks; missing or mismatched identity is rejected.

```bash
source scripts/source_somaforge.sh
python scripts/check_asset_manifest.py --allow-missing
python scripts/check_training_manifest.py
```

Programs can resolve an asset by manifest ID instead of joining paths:

```python
from somaforge_core import AssetManifest

assets = AssetManifest.load()
terrain_dir = assets.resolve("terrain.climb.cache", "root")
robot_urdf = assets.resolve("robot.g1.spherehand", "urdf")
```

## Retained native WBT and reproduction recipes

These independent recipes remain available for reproduction; they are separate
from the SuperMimic main flow above. See the [modular WBT guide](docs/wbt-modular-experiments.md)
and [ACP single-task reproduction](docs/acp-reproduction.md). They use the existing `ExperimentConfig` and
observation, reward, command, termination, curriculum and algorithm managers:

| Preset | Purpose |
| --- | --- |
| `exp:wbt-expert-dense` | Dense-reference expert |
| `exp:wbt-expert-sparse` | Sparse-reference expert with native contact labels |
| `exp:wbt-expert-force` | Expert with reference and measured contact force |
| `exp:wbt-student` | Native PPO/DAgger distillation from explicitly supplied experts |

Each command is independent: manually select manifests and checkpoints, train
experts, record contact-reference data when needed, combine manifests, and distill.
The [modular workflow guide](docs/wbt-modular-experiments.md) lists the commands,
replaceable configuration blocks and resume rules. Student actor inputs contain
measured observations and a high-level task command; reference poses, joint
targets, desired contact/force and teacher routing remain privileged.

The CLI also retains named baseline, future-reference, sparse-climbing and PHP
experiment presets. `exp:g1-29dof-wbt-contact-force` is the older eight-part
force-data experiment; its threshold/hysteresis masks do not establish native
contact truth. New sparse/force experiments require the six-part native-contact
reference contract described in the modular guide.

Compatible checkpoints retain the normal model, optimizer, normalization,
progress and sampler/curriculum state. Compatibility depends on the canonical
robot identity, algorithm and observation/network layout. The native student
cannot load the historical `tmp/` PHP adapter's different checkpoint schema.
No simulator snapshot is saved.

The canonical USD bundle is distributed with its existing fingerprints. Do not
regenerate it as a routine installation step: a new conversion can change bytes
and invalidate the recorded bundle identity. See the environment guide for asset
checks and deliberate conversion.

## Next-contact prediction and control

Predictor lives in `packages/generator/` and produces next-contact plans and
endpoint poses from robot state, actual contact observations and terrain.
Its implementation, training and evaluation entrypoints, baseline and data
contracts remain maintained. Candidates require independent Newton validation.

Contact-conditioned control learns actions for the supplied contact goal;
world-model training and policy optimization are separate from next-contact
prediction. WBT and distillation remain available for tracking and policy priors.
This architecture does not imply that combined prediction/control generalization
or executed support/load/slip has been validated.

The old contact-editor trajectory generation, IK generation workers and Infiller
are retired. Shared canonical FK, contact evidence processing and read-only
playback remain. Historical edited motions, datasets, checkpoints, published
release records and design documents are retained as archives; their old commands
are not current entrypoints. Historical eight-part threshold masks are not
recertified as the six-part native-contact contract by this cleanup.

The current Predictor pointer is [baselines/current.json](baselines/current.json):
[v1 latest](baselines/predictor_v1_latest_20261010/README.md), using
loaded-material207_joint_v12 data, coherent interval loss and keep-patch weight 1.
Its retained step1000 checkpoint is the latest saved result of the 2026-10-09
scratch run, not a selection by a single validation peak. The run reached
1002 of 1100 planned intervals and has no completion report. Static endpoints and
training-case recursion do not establish held-out recursive generalization or
executed support/load/slip, and no combined Predictor + WBT deployment acceptance is asserted.

[Archived baselines](baselines/README.md) retain scratch_recipe.20261002,
source550 and control500 recipes and comparisons. The structured v2 replacement
was [not adopted](baselines/predictor_control500_20261009/architecture_decision.json).
Historical reports live under [docs/history](docs/history/README.md); they are
not current installation instructions.

## Evaluation and visualization

The evaluator reconstructs a compatible experiment from its checkpoint:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m holosoma.eval_agent \
  --checkpoint /path/to/model.pt \
  --num-envs 1 --max-steps 2000 --visualizer kit
```

Omit `--visualizer` for headless operation; explicit `--headless` takes precedence.
Evaluation does not write persistent TensorBoard or W&B logs by default. Video,
recording, acceptance output and ONNX export are explicit options. Remote Kit
uses livestreaming; TensorBoard independently reads the selected training run.
See [evaluation and remote playback](docs/wbt-modular-experiments.md#evaluate-or-replay-separately).

Read-only browser playback uses:

```bash
packages/motion_edit/motion-edit review --input /path/to/canonical.npz
```

The input can also be a diagnostic JSON report or case collection. Add
`--terrain /path/to/terrain.obj` when needed. See the
[playback guide](packages/motion_edit/README.md#read-only-playback).
Browser review does not run the tracking policy or certify actual contact.

Contact intent, realized load and support slip are distinguished in
[the support semantics guide](docs/support-semantics.md).
