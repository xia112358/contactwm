# Runtime data

Runtime data is physically separate from the code checkout:

```text
/home/zhouxia/workspace/somaforge-unified/            code checkout
/perception/workspace/zhouxia/somaforge/runtime/current/  external runtime
runtime/current                                      -> external runtime above

runtime/supermimic -> /perception/workspace/zhouxia/supermimic_vibed/
runtime/stage2     -> /perception/workspace/zhouxia/somaforge-stage2/runtime/current/
```

The 2026-10-26 migration keeps the existing logical paths. Models, motions,
training data, logs, generated results, downloads, caches and temporary outputs
must remain on `/perception`, including future runs. Do not replace the runtime
symlink with a local directory. New checkouts can initialize this arrangement
with `python3 scripts/bootstrap_runtime.py --data-root /perception/workspace/zhouxia/<checkout-name>/runtime/current`.

Source-controlled canonical robot resources stay with the code. Python
installation environments are outside this project-data migration. Keep
`runtime/current` declared in `configs/assets_manifest.json` so asset and browser
resource resolution retain the same logical paths. Directory location alone is
not certification: training inputs still require canonical assets, current
contact semantics and acceptance records. Use logical project paths in configs.

Current categories are `current/motions`, `current/terrains`,
`current/models`, `current/manifests`, `current/generated`, persistent
`current/cache`, and historical `current/archives`. Required inputs and caches
are never kept in `tmp/`; that directory is for temporary outputs only.

Canonical identity is resolved by `somaforge_core.robot_assets` and recorded in
`configs/assets_manifest.json`. Do not copy fingerprints into documentation;
they become stale when the authoritative robot/mesh bundle changes.

`data`, `logs` and `tmp` at the repository root are symlinks into
`current/holosoma/`. Cleaning their contents changes the actual runtime files.

## Unified workflow resource aliases

`configs/workflow_resources.json` selects separate `supermimic` and `ppo`
interpreters and runtime roots. The default workflow is SuperMimic tracking →
native collection → DAgger initialization → contact training. `ppo` selects the
native environment for the retained original PPO tracking and Predictor; PPO's
connection to the downstream SuperMimic workflow is deferred.

The new source checkout is `somaforge-unified`. Its runtime aliases retain the
existing stores: merging source does not copy, rename or validate experimental
data. Original SomaForge, Stage 2 and standalone SuperMimic checkouts are
preserved. An external IsaacLab dependency can be selected explicitly in the
resource registry; project source must come from this checkout.

Use `scripts/run_workflow.py --help` and [the current workflow guide](../docs/current-workflow.md).
The launcher leaves checkpoint formats intact. Old trajectory editing and
Infiller results remain historical data; their generation entrypoints are retired.
