# Holosoma Core Package

This package is currently maintained in this repository as part of the
IsaacLab 3 / Newton WBT climbing research workspace.

Use the repository root README for the active project scope. The active setup
entrypoint is:

```bash
source scripts/source_isaaclab3_newton_setup.sh
```

Useful local checks:

```bash
python scripts/check_motion_manifest.py \
  runtime/current/manifests/omniretarget_baseline_29.json
python -m holosoma.train_agent exp:g1-29dof-wbt-baseline-29 simulator:isaaclab3-newton logger:disabled
```

Only IsaacLab3/Newton is supported by the SomaForge runtime. Legacy simulator
and real-robot bridge code has been removed.
