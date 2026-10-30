# Historical experiments

These records describe earlier experiments, data and commands. They are not
current training recipes or approval to load old assets.

- [Pairwise48 / GMVQ](climb00-pairwise48-production.md): retired generation pipeline.
- [Direct Infiller](climb00_direct_infiller_baseline.md): archived recipe and results;
  its `scripts/train_climb00_direct_baseline.py` launcher has been retired.
- [Paired-surface training](paired_surface_training_v1.md): 2026-09-13 training protocol.
- [Newton contact-pipeline migration](../newton_contact_pipeline_migration.md):
  original28 relabeling and the migration's then-open surface-vocabulary work;
  retained at its original path for existing links.

Current interfaces are listed in the [documentation index](../README.md),
[environment guide](../environments.md) and
[modular WBT expert/distillation guide](../wbt-modular-experiments.md).
The registered Predictor comparison recipe is selected by [baselines/current.json](../../baselines/current.json);
archived baseline records remain in [baselines/archives.json](../../baselines/archives.json).
Current scratch defaults are described separately in [Generator](../../packages/generator/README.md).
