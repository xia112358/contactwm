# Install ContactWM from a fresh clone

The repository defaults are portable. You do not need the author's directories,
an existing SuperMimic installation, or a sibling SomaForge checkout.
Run the commands below from the repository root. If the repository is private,
Git must be authenticated with an account that has repository access.

## Requirements

Simulation targets Linux x86_64, Ubuntu 22.04 or newer, Python 3.12 and an NVIDIA
GPU with a driver compatible with CUDA 13. Git, Python 3 with pip/venv and a C++
build toolchain are needed to install. On Ubuntu the system prerequisites include
`python3-venv`, `build-essential`, `cmake`, `libgl1`, `libglib2.0-0` and `libegl1`.
A native Windows simulator installation is not qualified; Windows can be a remote
client. Source checks and installation-plan inspection need no GPU.

Allow space for the Python/CUDA environment, download cache and selected datasets.
Installation downloads several GB; dataset sizes are listed in the
[data catalog](data-downloads.md). GitHub, PyPI, NVIDIA's package index and the
PyTorch wheel index must be reachable. Keep the upstream NVIDIA license terms.
The default workflow is kit-less; optional native Kit/PPO setup is separate.

## Default installation

```bash
git clone https://github.com/xia112358/contactwm.git
cd contactwm
python3 -m pip install uv==0.13.1

# These two commands inspect the checkout without installing or launching physics.
python3 scripts/check_workflow.py --source-only
python3 scripts/setup_workflow_environment.py --dry-run

# Install the pinned simulator, workspace packages and runtime directories.
python3 scripts/setup_workflow_environment.py
python3 scripts/check_workflow.py --check-cli
```

`uv` supplies Python 3.12 if necessary. The installer clones Isaac Lab at
`46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`, uses its committed `uv.lock`
(`all`, `importers`, `test` extras; no `isaacsim` extra), then installs the
tracked `workflow-extras.lock.txt` and the local packages. The upstream lock
pins Torch/CUDA, Newton, Warp and MuJoCo-Warp together. It is not substituted
with the optional native PPO profile's different physics versions.

This follows the existing workflow's dependency boundary: `holosoma` is installed
with `--no-deps` for shared source utilities. Its full native PPO development
requirements are installed only by the [native profile](install.md). A workflow
installation is not a full native environment and `pip check` can report those
native-only requirements and upstream explicitly overridden pins. Do not fix that
by upgrading Warp/Newton ad hoc; use the intended backend's installer.

The default layout is:

```text
contactwm/                                  Git source checkout
  runtime/environments/venv/workflow/        ignored Python environment
  runtime/dependencies/isaaclab-workflow/    ignored pinned upstream source
  runtime/current -> ../../contactwm-data/native
  runtime/supermimic -> ../../contactwm-data/supermimic
../contactwm-data/
  native/                                   published native motion/terrain inputs
  supermimic/                                prepared workflow data, logs and caches
```

The installer restores the tracked canonical G1 USD bundle, preserves existing
files/links, and refuses conflicting runtime paths or an unowned target environment.
Retry an interrupted installation with the same options. It does not replace or
activate your current environment, choose a GPU, download all datasets, or train.
No GPU index is forced. To select a GPU explicitly, set `CUDA_VISIBLE_DEVICES`.

## First data and one physics step

The Git checkout contains source and canonical robot assets. Download the published
PHP motion/terrain component, verify it, then prepare one paired task using the
installed workflow interpreter:

```bash
python3 scripts/download_data.py fetch php-wbt-296-data
python3 scripts/download_data.py verify php-wbt-296-data
runtime/environments/venv/workflow/bin/python \
  packages/supermimic/tools/prepare_lowstep.py \
  --component motions-low-step --count 1 --contact_model force6

python3 scripts/run_workflow.py tracking -- \
  --task_manifest ../contactwm-data/supermimic/data/lowstep_boxes/task.json \
  --num_envs 1 --startup_check --headless --amp none \
  --contact_model force6 --collision_profile surface_v2 \
  --contact_response manifold_v2 \
  --log_dir ../contactwm-data/supermimic/logs/first-startup-check
```

The last command creates physics, runs one policy step without optimization, and
writes `startup-check.json`. It needs an available GPU; it is distinct from a
successful source check, CLI help or dependency installation. First-time kernel
compilation can take several minutes. This small startup check does not establish
policy quality, accepted demonstrations or training convergence.

The PHP release contains source motions and terrains. Its native PPO teachers
are not SuperMimic tracking teachers. For the full tracking → collection → DAgger
→ contact workflow, train a tracking teacher or supply an explicitly compatible
SuperMimic checkpoint, then use [the stage commands](current-workflow.md).
Current original-clip contact datasets, fresh Newton contact recordings and
experimental local checkpoints are not bundled with the historical data release.
Build/record those inputs with their corresponding workflow tools; downloading
an old data component does not make historical contact labels current.

## Optional machine configuration

The committed `configs/workflow_resources.json` contains only generic paths
relative to this checkout. The commands above use it unchanged.
For external storage or an existing pinned Isaac Lab source, copy it to the
ignored local file and edit the relevant paths **before** installation:

```bash
cp configs/workflow_resources.json configs/workflow_resources.local.json
# Edit runtime_root, python and isaaclab_root as needed; runtime_root must be outside the checkout.
python3 scripts/setup_workflow_environment.py --dry-run
python3 scripts/setup_workflow_environment.py
```

Registry precedence is explicit `--resources`, then `SOMAFORGE_RESOURCES`, then
`configs/workflow_resources.local.json` if present, then the committed defaults.
Relative registry paths are resolved against the checkout root. Environment
variables `SOMAFORGE_SUPERMIMIC_{PYTHON,RUNTIME,ISAACLAB}` and the corresponding
`SOMAFORGE_PPO_*` values override individual launcher settings. Setup honors the
SuperMimic settings and the native runtime root. For a custom `--prefix` or
`--isaaclab`, put the corresponding path in the registry as well so subsequent
launches use it. The launcher does not inherit arbitrary `PYTHONPATH` trees.

`--base-python /path/to/python` is an optional compatibility route for someone
who already has the exact simulation stack; it is never the default prerequisite.
The explicit overlay uses that interpreter's installed simulation libraries.

For native PPO tracking or Predictor, install [the `main` profile](install.md).
Its generic registry entry points to its own interpreter and native runtime.

## Troubleshooting and validation scope

- Missing interpreter/modules: rerun the matching installer, then
  `python3 scripts/check_workflow.py --check-cli`. Dry runs prove routing only.
- Runtime/link conflict: inspect the reported path and your local registry.
  The installer preserves existing data instead of silently relocating it.
- Incompatible Isaac Lab revision: use a clean checkout at the printed commit;
  setup will not switch another existing source checkout behind your back.
- Download failures: check connectivity to the named upstream index/repository.
  Dataset downloads support retained archives; see [data downloads](data-downloads.md).
- No compatible GPU/driver: CPU checks still work; GPU physics does not.
- Missing recording, slices or teacher: those are stage inputs, generated or
  supplied explicitly. There is no automatic model substitution.

GitHub CI checks source assets, portable installation plans and routing on a clean
CPU runner. GPU startup/physics and complete installation must be verified on a
suitable host; CI does not claim convergence or re-certify archived checkpoints.
