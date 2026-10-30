# Motion and research playback

`motion_edit` now provides read-only browser playback and retained motion-data
utilities. The old trajectory editor/generator, trajectory IK workers and Infiller
are retired. Next-contact Predictor remains in [`packages/generator`](../generator/README.md).
Shared FK and native contact processing belong to
[`somaforge_core`](../somaforge_core/README.md).

Use the unified Miniforge + pip `main` environment; see
[Python environment setup](../../docs/python-environments.md). The
`packages/motion_edit/motion-edit` launcher selects this environment automatically.

## Read-only playback

Run from the repository root:

```bash
packages/motion_edit/motion-edit review --input /path/to/canonical.npz
```

JSON diagnostic reports and case collections use the same command:

```bash
packages/motion_edit/motion-edit review --input /path/to/report.json \
  --terrain /path/to/terrain.obj
```

Use `review --help` for host and port options. Bind local browser services to
`127.0.0.1` and choose an explicit free port. A remote browser needs the existing
forwarding/service arrangement; launching a browser process on a headless server
does not open the user's laptop browser.

Supported inputs include canonical NPZ motion, pose comparison reports, Predictor
case manifests, and supported contact/support diagnostic reports. A case collection
can refer to relative NPZ paths and explicitly paired reference cases. Historical
Infiller case collections remain readable as archived motion data.

Playback checks the canonical G1 identity, joint names, finite pose values,
normalized root quaternions and valid frame rate. It does not execute a policy,
edit a motion, generate a trajectory, update a checkpoint or certify task success.

## Contact and support evidence

Recorded native contact evidence retains its original timing and semantics.
Actual contact, normal load, support and material slip are separate quantities.
A historical `actual_contact` flag, geometric proximity or force threshold does
not establish current Newton contact. Inputs without native evidence remain
unknown. Eligible contact visualization uses the shared primary-face selection;
raw solver observations remain available for audit.

See the [support semantics guide](../../docs/support-semantics.md) and
[contact-label contract](../../docs/contact_label_contract.md).

## Data and compatibility

Motion storage, contact-record readers and segmentation/data utilities needed by
retained workflows remain available. Runtime metadata stays under the configured
`data/` links; large motions, models, logs and archived experiment outputs stay
outside the checkout as documented in [repository data](../../docs/repository-data.md).

Historical editing plans, generated candidates, datasets, checkpoints and design
notes are retained as records. The design notes under this package's `docs/`
directory describe the retired editor/generator and are archival, even where
older prose calls a route production. Do not use their commands as current setup
or restore retired entrypoints. Archived motions keep their original evidence
status and need independent current validation before use in a physical task.

`generate-ref` and generation worker/batch commands are no longer supported.
`contact-editor` remains a read-only alias for `review`; it cannot edit or generate
trajectories. Use review for existing results. Predictor training and
prediction remain independent of the removed trajectory generator.

## Development and verification

The browser server is `motion_edit/web/server.py`; the frontend source is `web/`.
Build the frontend before packaging playback:

```bash
npm --prefix packages/motion_edit/web run build
```

Run retained package tests in the managed environment:

```bash
source scripts/source_somaforge.sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/manage_environment.py run --profile main -- python \
  -m pytest -p no:cacheprovider packages/motion_edit/tests -q
```

Playback tests do not run tracking training or establish contact-control quality.
For WBT execution and Kit rendering, use the
[modular WBT guide](../../docs/wbt-modular-experiments.md).
