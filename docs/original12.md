# original12: task_max76_with_walk.json

`original12` contains twelve original PHP reference clips: step and 58/76 cm climb
with their mirrors, plus three eight-second walking windows and their mirrors.
The complete task component is distributed as
[original12-20261031-v1](https://github.com/xia112358/contactwm/releases/tag/original12-20261031-v1),
not as an assumed local training directory.

## Complete original training setup

From a fresh clone with GitHub access and `uv==0.13.1` installed:

```bash
python3 scripts/original12.py prepare
```

This installs the pinned workflow environment, downloads the task's 54-file data
component and both full training checkpoints, then verifies every distributed
file. It needs no pre-existing author environment, directory, temporary script or
uncommitted source. For a private repository, authenticate GitHub CLI first with
`gh auth login`. No credentials are stored in this repository.

If the pinned environment is already installed, use `prepare --skip-environment`.
For offline files, `--assets /path/to/assets` accepts a directory containing the
index and archive files from **both** original12 releases. The original data-only
command remains available:

```bash
python3 scripts/download_data.py fetch original12 --catalog configs/original12_release.json
```

Inspect the complete exact launch, then run it when an appropriate GPU is available:

```bash
python3 scripts/original12.py tracking --dry-run
python3 scripts/original12.py tracking
```

The frozen [recipe](../configs/experiments/original12_newton2ms.json) retains every
recorded training parameter, including `newton_contacts_v2`, `--no_reset_noise`,
4096 environments, force6, surface_v2, the completion sampler and the original
WM refit schedule. Newton collision generation runs at every 2 ms physics
substep while MJWarp performs dynamics. There is no `manifold_v2` substitution.
The current source includes the collision-mask handling, response contract and
runtime guards from the original training tree. Geometry, margins and gaps are
unchanged. GPU selection uses the destination host's available devices.

By default tracking starts from the original full iteration-1000 initialization,
which records its explicit transfer from manifold_v3 to newton_contacts_v2.
To continue the saved iteration-1500 training state instead:

```bash
python3 scripts/original12.py tracking --resume-state latest
```

The [training release](https://github.com/xia112358/contactwm/releases/tag/original12-training-20261031-v1)
contains `initialization.pt`, `tracking/last.pt`, original arguments, transfer
metadata and two historical audit reports: approximately 476 MB compressed,
517 MB uncompressed. Both full checkpoints retain actor, WM, critic, optimizers,
progress and reset sampler. Their ordinary checkpoint boundary creates fresh
replay and simulator state on resume; it is not a bitwise simulator restart.
Historical source-model/audit paths remain provenance and are not required on
the destination host.

The run is **paused at iteration 1500 and is not a converged/accepted policy**.
Publication does not establish task success. No GPU training or physics check
was run as part of this synchronization.

For file/checkpoint-contract checks on the destination host, without simulation:

```bash
python3 scripts/original12.py check
```

To evaluate the supplied iteration-1500 checkpoint under its exact physics:

```bash
python3 scripts/original12.py evaluate --dry-run
python3 scripts/original12.py evaluate
```

Evaluation runs only when explicitly requested. `tracking --startup-check` is
also an explicit one-step physics check without optimization. The output path
defaults to the configured external runtime; `--output` can select another
external directory. Original checkpoint/data files are not rewritten.

The task manifest path remains:

```text
runtime/current/workflow/original12/task_max76_with_walk.json
```

The 54-file component includes the unchanged task JSON, all twelve canonical NPZ
references, both referenced/selected CSV directories, paired USD/OBJ terrain,
visualization XML, source catalogue and upstream license/notice. It contains no
tracking policy, world-model weights, contact recording or training checkpoint.
The catalogue is retained as provenance; unrelated clips in that inventory are
not included in this selected component.

The installer verifies archive and file hashes and writes host-local path aliases
under `runtime/current/manifests/published_data_paths.json`. Task readers apply
those aliases in memory. Original files, manifest SHA, robot identity, sampling
weights, mirror relationships, frame ranges, collision surfaces and contact
semantics are preserved. No directory with the author's username is needed.
Do not rewrite the JSON to make its old absolute paths look local: changing its
bytes also changes the identity used by recordings/checkpoints.

A task manifest supplies reference inputs; it does not choose compatible physics
or certify physical execution. Keep the intended experiment's explicit collision
profile/contact response and matching checkpoint contracts. Source contact hints
are not actual Newton contacts. Fresh native contact collection remains required.
The auxiliary renderer uses the destination checkout's canonical mesh directory;
it does not load a second robot asset from the historical machine.

Maintainers can rebuild a release without simulation using
`scripts/build_original12_release.py --help`. Supply the original manifest and
upstream LICENSE/NOTICE explicitly. The builder checks every declared input hash,
preserves original bytes and stream-verifies the resulting archive.
