# Surface contact system, phases 1 and 2

The shared implementation is `somaforge_core.contact_system`, schema
`somaforge_surface_contact_v1`. Both SomaForge checkouts carry the same package
implementation. SuperMimic consumes the core selected by its existing validated
task manifest; no extra checkout/namespace loader is introduced.

Phase 1 changes evidence handling and geometric objectives. Phase 2 adds a
versioned collision revision derived from the same authoritative G1 source.
Native contact activation and success criteria keep their existing semantics.
Hydroelastic contact, continuous substep acquisition, spatial WM heads and
simulator-Jacobian acceptance remain later migration steps. No training is
automatically launched by the migration.

## Evidence and timing

`ContactEvidence` owns all valid solver rows, including inactive, self and
task-excluded contacts. It records actual type/margin/allocation, world/shape/
body mapping, position, frame and world force on body1. Optional torque, geometry
witnesses, local application points and velocities remain missing when not
available; metadata lists available fields and units. Active unallocated
constraints raise. Zero-force active contacts remain active.

`BodyBinding` and `terrain_view` construct explicit robot-side views and correct
force/torque signs. The existing primary-face eligibility mask can select the
SomaForge upward-face task. Omitting that mask explicitly selects all terrain
contacts. Known masks never rewrite raw activation.

`Sampling` records control/physics dt, the sampled substep and distinct contact/
pose timing. One last-solve snapshot cannot claim interval-complete coverage.
`control_step` is an environment-local episode step, not a persistent solver
contact identity. Local application points use constraint-evaluation transforms.

WBT and nine-body ACP readers expose `read_evidence()`. Reference recording and
Stage 2 collection/audit exports use this schema. Existing policy observations
retain their shape; raw per-contact force/torque is separate from sensor force6.

## Body-surface representation

`SurfaceAtlas` contains fixed part-local points, unit normals and area weights.
It accepts realized collision surfaces rather than adding colliders. SuperMimic
uses the compiled MuJoCo convex hulls; helper spheres exist only in legacy_v1.
Sampling is deterministic
and area-stratified, with minimum allocation to small primitives. The binary
model hash and atlas fingerprint identify the representation and node count.

`project(view, solve_positions, solve_rotations, bandwidth=...)` independently
maps each real point-force pair. Node forces sum to the selected resultant; an
explicit moment residual retains wrench accounting, including opposing forces
and contact torque. Values are forces in N and evidence mass, not calibrated
pressure or a physical occupancy predicate. Unknown and known-unloaded differ.
Grid refinement does not create new physical measurements.

## Geometric objective

`smooth_region_loss` computes scalar normal/tangent costs to an explicit bounded
disk, then an area-normalized log-sum-exp. Radius zero denotes a point task.
Selection/scoring use the same metric; candidate coordinates are never averaged.
Duplicating nodes while dividing their areas leaves the cost unchanged. Surface
quadrature accuracy still requires a node-density study. The loss includes an
entropy term; diagnostic normal/tangent expectations alone do not sum to it.

SuperMimic DMO and imagination default to `--contact_position_loss surface_region`:

- `--surface_samples_per_part 128`: fixed nodes per part.
- `--surface_temperature .02`: dimensionless scalar-cost temperature.
- `--contact_region_radius 0`: explicit geometric target radius in metres.
- `--contact_target_normal X Y Z`: required unit world normal for a disk target.

Radius cannot exceed a native completion radius; it does not redefine native
success. Distance normalization retains 0.05 m. `collision_surface` and
`measured_material` are explicit legacy comparisons. Run configuration saves the
model/atlas/target contract; resume rejects a different existing surface
contract. An old checkpoint has no such contract and is recorded as a null
source contract when initializing a new surface run.

Actual success requires native contact. Geometry is not gated by force,
predicted activation or transient measured points. DMO event/terminal scoring and
early-action derivatives are retained. `temporal_region_loss` provides a valid
window objective for diagnostics/future adapters; it does not silently change
the stopping rule or add dwell.

SuperMimic replay adds `paired_force_w` and `contact_count`; `force` and `xi`
force6 retain their body-sensor meaning. `--contact_evidence_every 1000` exports
owned raw rows and, in surface runs, projected feature samples and moment
residuals. Zero disables exports. Projection uses retained solve poses even if a
task reset followed capture. Dense surface fields are not added to every replay
slot or checkpoint.

## Validation and audit

Run from a configured SomaForge environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_contact_system.py packages/somaforge_core/tests/test_newton_support.py \
  tests/test_contact_record_storage.py tests/test_contact_face_selection.py
```

From SuperMimic with the same core package available:

```bash
python tools/check_surface_contact_system.py
python tools/check_contact_surface.py
python tools/audit_surface_contact.py \
  --recording tmp/contact-tracker-20261027-v1/demo0-roles-v1/rollout.pt \
  --trajectory tmp/contact-tracker-20261027-v1/isolated-surface-normal4-v1/rollout_050.pt \
  --collision-model tmp/contact-tracker-20261027-v1/initialization-fast-v3-solver.mjb \
  --segment 7 --normal-weight 4
```

The audit prints JSON. `--output tmp/<new-report>.json` explicitly saves a new
file with exclusive creation. It measures fixed-trajectory cost variation,
native witness reselection and geometry finite differences, excluding reset/
unknown boundaries. These do not certify executed control or WM derivatives.

On the retained segment-7 example (44 valid samples), adjacent-loss p95 was
9.419 for old Euclidean selection then weighted scoring, 7.446 for the
metric-consistent hard atlas objective, and 7.451 for the smooth objective.
Maxima were 14.721, 9.382 and 9.382. The improvement here comes primarily from
representation/metric, not extra softening. Maximum absolute geometric root
gradient finite-difference error was 2.8e-5 at a 1e-5 m perturbation. These are
CPU replay diagnostics, not online simulation or training acceptance.

A 512-node repeat gave smooth adjacent-loss p95 8.245, maximum 11.196 and
geometric derivative finite-difference error 1.1e-5. Node-density convergence
is not established; 128 is an initial configurable quadrature, not a certified
physical resolution. Production adoption needs a density/performance study and
matched native executions. Dense geometric evaluation in DMO is restricted to
the worlds/parts participating in each scoring event to bound BPTT memory.


## Physical collision revision (phase 2)

`somaforge_core.robot_assets` resolves `collision_profile="surface_v2"` to the
pinned overlay `assets/g1_surface_v2_usd/g1_29dof_surface_v2.usda`. It references
the validated legacy USD bundle, disables 14 auxiliary sphere collider/visual
geometries and 29 duplicate mesh colliders. The enabled robot colliders are
**29 meshes, 4 cylinders, 0 spheres**. The hand cap remains its continuous mesh.
The overlay preserves all 53 body frames, joint properties, authored mass and
inertia attributes and the canonical source URDF/mesh fingerprint. Names such as
`sphere_hand_link` remain frame names, not evidence that a helper sphere exists.
Newton may infer mass for bodies whose source mass is unspecified; preservation
of authored attributes alone does not certify identical realized inertias.

Fresh SomaForge and Stage 2 G1 configurations select surface_v2. Historical
serialized configurations missing the new field resolve to legacy_v1; existing
jobs continue their original physics. SuperMimic terrain training and contact
collection/training select surface_v2; generic evaluation inherits the saved run
profile and continuous contact evaluation inherits the recording profile.
`canonical_g1_asset_metadata()` keeps its historical default for consumers that
need source/legacy identity; physical execution callers pass the profile.

Part maps include only the main ankle and hand bodies in surface_v2. Historical
heel/toe ball identities have no physical counterpart in that revision. Stage 2
hand-plane geometry omits the fingertip sphere when the recording uses surface_v2.
These proxies remain geometric objectives, never native contact detectors.

Evidence and physical checkpoints carry the profile, profile hash, base bundle
hash and selected USD hash. Physical resume, native contact reference loading,
Stage 2 WM/stream initialization, and SuperMimic recording/resume checks reject
mixing legacy and surface evidence. Old kinematic trajectories can supply intent;
old active/force/witness labels require new native execution. Nothing relabels an
old recording in place or silently imports its optimizer as a surface run.

SuperMimic contact training no longer defaults to an old hardcoded MJB. By
default it exports the actual running solver to `solver-collision.mjb` plus a
hash/profile sidecar, then builds the surface objective from that model. An
explicit surface-v2 `--collision_model` needs a matching sidecar and the expected
realized collider topology. Collection exports the same model for audit. An old
tracking policy may be used only through explicit `--allow_collision_transfer`
as an unvalidated initialization; this does not exempt recordings, contact
policy resumes or priors from the physical profile check.

Validation:

```bash
python scripts/check_g1_surface_colliders.py
# Optional CPU Newton import/solver geometry inspection, not contact labeling:
python scripts/check_g1_surface_colliders.py --realize
python -m pytest -q -p no:cacheprovider \
  packages/somaforge_core/tests/test_collision_profiles.py \
  packages/somaforge_core/tests/test_robot_assets.py tests/test_contact_system.py
```

The composed physx/mujoco/physics variants have the same 29/0/4 colliders; tests
compare all authored body/joint physical properties and relationship targets.
A CPU Newton import with a ground collider compiled to 36 generalized positions
and 34 total geoms (33 robot + ground), confirming the same 29/0/4 robot topology.
This is a geometry check, not Newton/MJWarp contact truth or control acceptance.
The migration has not run GPU trajectories or trained a surface-v2 policy. Fresh
native collection, force/support checks and matched gradient/control evaluation
remain required before claiming reduced online loss jumps or policy success.


## Contact WM stable-contact scoring (2026-11-02)

The Contact WM stream defaults to DMO and `single_segment`. This objective now
scores stable contact rather than first touchdown. Native activation still uses
the allocated Newton constraint and upward-primary-face contract above.

The scoring gate requires known, active contact in consecutive control frames,
normal load above 30 N in both frames, load change at most 10 N per control frame,
and unchanged loaded rows. A loaded row has normal force above 1 N. Row matching
uses part-local solver points: nearest assignments must form a bijection within
1 cm. This is conservative: ambiguous nearest assignments are excluded even if
another possible matching exists. Reordering rows alone does not change the gate.
Three consecutive passing comparisons are required before scoring or resuming.
These are initial scoring thresholds, not new contact activation thresholds.

At the first qualifying frame of a contact episode, latch the normal-force-weighted
local centroid of loaded rows. Keep that material anchor until contact is lost,
unknown or reset. Row switching and load transients pause scoring and reset the
streak, but do not move the anchor. FK continues to transform the fixed local
anchor, so body movement and slip remain visible after stable scoring resumes.
Solver points are from the last solve; poses are post integration. Local matching
uses that recorded convention and does not claim strict same-state sampling.

For each required part, average squared position error divided by (5 cm)^2 over
qualifying frames, then average required parts and trajectories equally. New parts
use the fixed task target; maintained parts use their entry support target. If a
part has no qualifying frame in the entire episode, use terminal collision-surface
region guidance. This separate term supplies a learning signal without scoring
unstable raw contact coordinates. Root endpoint and action regularizers retain
their existing definitions. Logs expose `stable_contact_loss`,
`terminal_guidance_loss`, `stable_contact_frames` and `missing_stable_parts`.

Raw rows are processed during collection with a 32-row-per-part capacity contract;
overflow fails instead of truncating. Replay stores only the scoring mask and local
anchor for DMO, not the padded raw rows in the WM buffer. `objective.json` and stream
checkpoints record `stable_material_contact_v1`; full resume rejects older loss
contracts. Historical first-touch monitor metrics remain explicitly first-touch
diagnostics and are not the stable-contact training loss.

Validation: synthetic CPU tests cover permutation, row appearance/disappearance,
load jumps, unknown/reset boundaries, anchor retention and zero position-loss
gradients on excluded transition frames. This change has not run GPU training.
