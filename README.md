# ContactWM

ContactWM brings the current G1 tracking and contact workflow together: **SuperMimic tracking → native contact collection → DAgger
initialization → contact training** using imagination or DMO updates.

Original optimized PPO tracking, next-contact Predictor and read-only motion
playback remain available as independent entrypoints. Shared assets, FK and
native contact processing live in `somaforge_core`.

## Quick start

Linux x86_64 (Ubuntu 22.04 or newer), Python 3, Git, a C++ build toolchain and
an NVIDIA GPU/driver compatible with CUDA 13 are required for simulation.
Source checks need no GPU. Native Windows simulation has not been qualified.

```bash
git clone https://github.com/xia112358/contactwm.git
cd contactwm
python3 -m pip install uv==0.13.1
python3 scripts/check_workflow.py --source-only
python3 scripts/setup_workflow_environment.py
python3 scripts/check_workflow.py --check-cli
```

The default installer builds Python 3.12 from the pinned Isaac Lab kit-less
lockfile and installs the project packages. No existing SuperMimic environment,
workstation paths or second project checkout is required. It does not start
training. Large downloads require access to GitHub, PyPI, NVIDIA and PyTorch.

The tracked resource registry uses checkout-relative paths. Training data, logs
and caches default to a sibling `contactwm-data/` directory; the environment and
pinned dependencies are ignored under `runtime/`. No GPU index is forced:
`CUDA_VISIBLE_DEVICES` is optional and respected when set.

For the exact original12 experiment, `python3 scripts/original12.py prepare`
installs the environment and downloads its data plus full initialization/resume
checkpoints. See [the exact original12 recipe](docs/original12.md).

Read [fresh installation and first run](docs/quickstart.md) for custom storage,
data downloads and a one-step simulation check. Installation does not include
trained policies or certify historical contact labels. The optional native
PPO/Predictor environment has its own [installation recipe](docs/install.md).

Latest contact segmentation and supervision status: [2026-11-05 progress](docs/research/contact-20261105/README.md), including loss-aligned native witnesses, handoff-chain experiments, and the 82→83 entry diagnosis.

## Start here

- [Current workflow and commands](docs/current-workflow.md): tracking, collect, initialize, contact and Predictor
- [Native PPO/Predictor installation](docs/install.md), [Python environment setup](docs/python-environments.md) and [compatibility](docs/environments.md)
- [SuperMimic package and original runtime recipe](packages/supermimic/README.md)
- [original12 complete original training setup: source, exact recipe, data and full checkpoints](docs/original12.md)
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

The original optimized PPO tracking recipe remains available through
`scripts/run_workflow.py tracking --backend ppo`. Its DAgger integration is deferred.
See [modular WBT](docs/wbt-modular-experiments.md) for expert/student recipes and
[ACP reproduction](docs/acp-reproduction.md) for the retained experiments.

## Next-contact prediction and control

Predictor in `packages/generator` maps robot state, actual contact observations
and terrain to next-contact plans and endpoint poses. Contact control learns
actions for the supplied goal using the world model and imagination or DMO.

The [current Predictor baseline](baselines/current.json),
[baseline records](baselines/README.md) and [support semantics](docs/support-semantics.md)
contain the data contracts and validation scope. Historical experiments and
retired trajectory generation/Infiller records live under [docs/history](docs/history/README.md).

## Evaluation and visualization

For native WBT evaluation, use the [evaluation guide](docs/wbt-modular-experiments.md#evaluate-or-replay-separately).
CLI simulation defaults to headless; `--visualizer kit` enables Kit and
explicit `--headless` takes precedence.

Read-only playback accepts canonical motions, diagnostic reports and case collections:

```bash
packages/motion_edit/motion-edit review --input /path/to/canonical.npz
```

See the [playback guide](packages/motion_edit/README.md#read-only-playback)
for terrain and comparison options.
