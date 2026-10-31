# SuperMimic in SomaForge

This package preserves the `supermimic` module namespace and current tracking,
contact initialization, full-segment WM/DMO control, and diagnostic source.
`README_CONTACT.md` records the inherited research contracts; `README.md` is the
upstream reproduction reference. `SOURCE_PROVENANCE.json` records original
source hashes. Historical experiment JSON is preserved byte-for-byte.

Use the unified workflow launcher or an environment where this package,
`somaforge_core`, and `holosoma` are available. Simulation still needs the
registered compatible Isaac Lab / Newton / MuJoCo-Warp environment; it is not
vendored or silently upgraded by installing this package.

For a new clone, start with the root [quick start](../../docs/quickstart.md).
The default installer creates the pinned simulation environment from scratch.

Runtime location is configurable: set `SOMAFORGE_SUPERMIMIC_RUNTIME`, or register
`supermimic.runtime_root` in `configs/workflow_resources.local.json` (ignored), or use the portable defaults
in `configs/workflow_resources.json`. `SOMAFORGE_RESOURCES`
can select another resource file. The root contains existing `data`, `logs`,
`tmp`, `results`, `models`, etc. Run tools from that runtime directory (the
compatibility `tools/run_native.sh` does this); packaged `tools/` and `experiments/`
are source assets. No data/checkpoints/Isaac Lab tree are copied into this package.

Current entrypoints remain `python -m supermimic.train`,
`python -m supermimic.contact_collect`, `python -m supermimic.contact_initialize`
and `python -m supermimic.contact_train`. Their action, observation, physics,
normalizer, recording and checkpoint contracts are unchanged. Existing
SuperMimic source/actor checkpoints remain explicit inputs; PPO weights are a
different backend format and are not implicitly converted to `pi`/`wm`.

Inherited old-checkout paths in archived manifests/checkpoints are provenance.
Pass current resource paths explicitly; do not rewrite frozen artifact hashes or
pretend different training routes are equivalent. Installed SomaForge packages
own robot/contact/perception code; manifest `somaforge_root` no longer changes
Python import paths or selects a sibling checkout.
