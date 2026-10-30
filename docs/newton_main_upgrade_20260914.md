> Historical migration report. Current setup: [environments.md](environments.md).

# Newton main runtime — 2026-09-14

The default `scripts/source_isaaclab3_newton_setup.sh` profile is now `main`.
It activates the original `env_somaforge` base and then the isolated overlay
`runtime/environments/newton-main-20260914`. Existing running processes are
not restarted and still use their original loaded engine. Merely sourcing a
new shell does not upgrade a running contact server.

Pinned upstream commit: `01381081fa0de0bac8f85f2669777782504787d0`.
Dependencies: Newton 1.7.0.dev0, Warp 1.17.0, MuJoCo/MJWarp 3.12.0,
NumPy 2.3.1. Reproducible package requirements are in
`configs/newton-main-requirements.txt`. The overlay inherits other dependencies
from the original conda environment; it is not a standalone portable environment.

## Compatibility

- Native main collision factories are left untouched. Reviewed source hashes
  guard the collision implementation, including the reduced analytic sphere path.
- The old retry/SAT patch remains available only on the legacy runtime.
- Local Isaac Lab API compatibility maps SolverNotifyFlags to ModelFlags and
  the old joint target field names to joint_target_q/joint_target_qd.
- Upstream's supported `use_coord_layout_targets=False` retains the trained
  policy's DOF-shaped actuator layout.
- Source-key capture reads actual dynamic bit widths. Main's reduced analytic
  mesh/sphere sub-key is losslessly converted to the existing canonical stored
  triangle/manifold format. No geometry-distance face fallback is introduced.
- Robot asset, shape configuration, margin/gap and contact activation criteria
  remain unchanged. Contact query outputs record a distinct numeric schema:
  `newton_native_main_01381081_v1`.

## Validation

Evidence: `tmp/predictor_loss_diagnosis/1789389916609700526/`.

- Nine exact captured MPR core cases compared across old/new engines; the
  old-Newton/new-Warp control rules out Warp alone as the fix.
- New runtime: 28 project collision/contact tests passed.
- Legacy compatibility: 10 tests passed, one main-only test skipped.
- Real G1/terrain scene created through the official AppLauncher.
- Full fixed-q queries passed for samples 1993, 321, 3079, 27, including
  sample 3079's original extreme-error perturbation (epsilon 0.01).
- Required body/shape, scale, transform, margin and gap fields compare exactly
  to the original scene. Active part constraints are allocated and source
  attribution is checked against actual triangle IDs.
- Three independent solver steps produced finite joint positions/velocities.

This is a migration smoke test, not a full policy rollout or training-convergence
certification. Real penetration remains in the frozen predictor outputs.

## Bindings, labels, running jobs

The native model fingerprint differs. Do not replace old binding fingerprints
by hand or silently reuse old verified-label status. The freshly generated
scene-1 binding is
`tmp/predictor_loss_diagnosis/1789389916609700526/upstream_upgrade_binding_final_smoke.json`.
Its companion model and `upstream_upgrade_full_query.json` retain provenance.
Generate bindings from each actual upgraded scene and revalidate cached contact
labels before a new training baseline. Existing old-engine jobs remain labelled
as legacy; do not mix their query results into a main-engine run.

## Explicit rollback (no deletion)

```bash
export SOMAFORGE_NEWTON_PROFILE=legacy
source scripts/source_isaaclab3_newton_setup.sh
```

Return to main with `export SOMAFORGE_NEWTON_PROFILE=main` and source again.
Keep the original conda environment while the overlay depends on it.
