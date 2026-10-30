# Climb00 compatibility package

Implementations now live in [`generator`](../generator/README.md),
[`contact_solver`](../contact_solver/README.md), and
[`somaforge_core`](../somaforge_core/README.md).
Old imports and `python -m climb00_pipeline.<module>` remain available for
explicit historical reproduction. The package is not added to the default
workspace, setup installation or sourced environment.
New code should use the canonical packages; WBT remains in `src/holosoma`.
Current expert/distillation entrypoints are documented in
[modular WBT experiments](../../docs/wbt-modular-experiments.md); the
[documentation index](../../docs/README.md) lists the maintained interfaces.

Tests now live beside their canonical implementations in `generator/tests`,
`contact_solver/tests`, and `somaforge_core/tests`. This package contains no
independent model, solver or dataset implementation. Its aliases preserve module
identity for historical scripts and saved experiment reproduction.

Historical experiments and runtime contracts are indexed in
[repository maintenance](../../docs/repository-maintenance.md).
The [direct-infiller recipe](../../docs/history/climb00_direct_infiller_baseline.md) is a
historical experiment with protected external dependencies; it is not the current
training-data approval contract.

For an archived script that requires these aliases, opt in from the repository
root after activating the current environment:

```bash
export PYTHONPATH="$PWD/packages/climb00_pipeline${PYTHONPATH:+:$PYTHONPATH}"
```

This exposes the same canonical implementation modules; it does not restore
retired training initializations or make old data valid. Existing environments
that previously installed this package are not uninstalled by repository cleanup.
