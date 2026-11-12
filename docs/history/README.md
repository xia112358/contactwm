# Historical experiments

These records describe earlier experiments, data and commands. They are not
current training recipes or approval to load old assets.

- [Retired Contact WM](contact-wm-legacy/README.md): old Holosoma WM adapters, stream trainers and tests.
- [Pairwise48 / GMVQ](climb00-pairwise48-production.md): retired generation pipeline.
- [Direct Infiller](climb00_direct_infiller_baseline.md): archived recipe and results;
  its `scripts/train_climb00_direct_baseline.py` launcher has been retired.
- [direct_infiller_boundary_velocity](direct_infiller_boundary_velocity.md): retired Infiller experiment.
- [direct_infiller_decoupled_time](direct_infiller_decoupled_time.md): retired Infiller experiment.
- [direct_infiller_frames](direct_infiller_frames.md): retired Infiller experiment.
- [direct_infiller_latest_predictor](direct_infiller_latest_predictor.md): retired Infiller experiment.
- [direct_infiller_training_efficiency](direct_infiller_training_efficiency.md): retired Infiller experiment.
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
