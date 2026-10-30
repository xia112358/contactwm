# Native PPO/Predictor environment setup

The unified repository keeps separate SuperMimic and native SomaForge runtime
profiles. [The current workflow](current-workflow.md) selects them through
`configs/workflow_resources.json`: SuperMimic serves tracking → collect → DAgger
→ contact; the `ppo` entry selects the native environment for original PPO
tracking and Predictor. PPO's downstream integration is deferred.

This page describes the existing native managed Miniforge + pip environment:
`runtime/environments/conda/somaforge` (`main`). It includes WBT, distillation,
Predictor, read-only motion playback and native Newton queries. The old trajectory
editor/generator, its IK workers and Infiller are retired. Shared Newton FK and
contact processing remain in core. The separate PyRoki/JAX IK profile and unused
standalone retarget profile have also been retired.

Use the new local `somaforge-unified` checkout for these instructions. The
runtime profile name `main` selects installed dependencies and pinned simulator
sources; it does not select a Git branch or the SuperMimic environment.

The catalog is [`configs/environments/profiles.json`](../configs/environments/profiles.json).
It declares Python 3.12, the external dependency lock, editable local packages,
the pinned Isaac Lab source commit and explicit dependency exceptions. Conda
supplies Python and pip; pip installs the locked artifacts. This workflow does
not use the root `uv.lock` or inherit another environment through `.pth` files.

## Create, inspect and run

Use Linux x86_64 and Miniforge. Main builds `evdev` from source; on Ubuntu install
`build-essential` and `linux-libc-dev` for the C compiler and Linux input/uinput
headers. Run these commands from the checkout root:

```bash
python3 scripts/bootstrap_runtime.py
python3 scripts/manage_environment.py plan --profile main
python3 scripts/manage_environment.py create --profile main
python3 scripts/manage_environment.py check --profile main
python3 scripts/manage_environment.py run --profile main -- motion-edit --help
packages/motion_edit/motion-edit review --help
```

`plan` is read-only. `create` installs the declared lock and local packages and
can resume a prefix owned by the manager with identical inputs. It refuses
foreign or incompatible prefixes. `check` audits ownership, versions, sources,
dependency exceptions and isolation. A passing CPU check alone does not validate
Kit, CUDA, policy execution or training.

To select Miniforge, use `--conda /path/to/miniforge3/bin/conda`. Discovery also
accepts `CONDA_EXE`, `MINIFORGE_ROOT`, a `conda` on `PATH`, or
`~/miniforge3/bin/conda`. Use `--isaaclab /path/to/pinned/IsaacLab` to select the
source checkout. Machine-local paths are not embedded in dependency locks.

`run` supplies the prefix interpreter, runtime markers, pinned Lab source
snapshot and subprocess routing. A bare `conda activate` does not reproduce
that context. The `motion-edit` wrapper selects managed main automatically.
The setup/source compatibility scripts also select main. For an interactive
shell, run `source scripts/source_isaaclab3_newton_setup.sh`; it uses the same
catalog and sanitized environment as `run --profile main`. Native workers
resolve main even when started from an unactivated shell.

For example:

```bash
python3 scripts/manage_environment.py run --profile main -- python -m holosoma.train_agent --help
```

Isaac Sim can bootstrap Kit and request the NVIDIA Omniverse license even for a
help command. The manager does not accept the license automatically. Environment
setup does not restart existing processes or migrate an active training run.

## Read-only motion playback and shared helpers

Use `packages/motion_edit/motion-edit review --input PATH` for canonical NPZ
motion or a supported JSON diagnostic report. Add `--terrain PATH` when needed.
The wrapper selects the same managed `main` environment. See the
[playback guide](../packages/motion_edit/README.md#read-only-playback).

There is no current `generate-ref`, editor generation or trajectory IK worker
entrypoint, and no separate IK environment to install. Canonicalization and
native contact-query tools remain available through shared `somaforge_core`
helpers. Current Newton/MJWarp contact semantics and canonical robot provenance
still apply; archived source masks do not establish current contact truth.

## Lock and source provenance

The main lock contains 382 external packages, exported from the pinned official
Isaac Lab lock plus SomaForge additions and validated compatibility overrides.
Exact artifact URLs and SHA-256 hashes are recorded in the lock; its
`.provenance.json` records their origins. Pip consumes these artifacts without
resolving new simulator versions. The exporter is
`scripts/build_environment_lock.py`; it verifies both the Lab Git commit and
committed upstream lock and refuses changes to retained upstream versions.

The catalog records exact Newton overlay exceptions and ONNX GPU / official
OpenCV provider substitutions. Only one distribution may supply each shared
ONNX Runtime or OpenCV API. The manager rejects duplicate providers even if
version requirements appear satisfied.

Retiring trajectory generation does not uninstall shared upstream packages or
rewrite the validated simulator overlay. The main lock and its ownership
fingerprint remain unchanged. JAX/JAXlib packages retained from the official Lab
lock are independent of the removed editor/generator.

## Historical validation

The following checks describe the 2026-10-22/23 implementation, before trajectory
generation was retired; they are not new cleanup verification results.

On 2026-10-22, main was created with Miniforge and Python 3.12.15 and passed the
manager's dependency, source and isolation audit with nine explicit compatibility/
provider exceptions. Its initial CPU repository suite passed 2,159 tests and
three subtests, with three tests skipped. Motion Edit subsequently switched to
Torch/Newton after reproducing the original six-task batch and comparing its
outputs with the frozen JAX baseline. Test results and comparison artifacts are
kept under project `tmp/`. The subsequent [component verification and current
Generator inputs](environments.md#main-consolidation-validation-2026-10-2223)
describe actual training updates, resume checks, historical retargeting and native queries.
These checks do not certify policy quality or convergence. Training restarts
remain explicit tasks.
