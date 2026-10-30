# Realized shape and finite surface interval experiment

As of 2026-10-10, the user-designated current recipe and fixed checkpoint are
registered in [predictor.v1_latest.20261010](../../../baselines/predictor_v1_latest_20261010/README.md).
The control500 checkpoint below is an archived experimental reference. The
matched-update evidence preserves that reference's behavior; it is not an
independent performance-gain claim for the new scratch training run.

The final coherent material-query interval is connected to the maintained
trainer and enabled by default for its unified-contact objective. The pinned
control500 checkpoint is unchanged. Assets, Newton activation/allocation,
primary-face selection, margins, architecture and other loss weights are
unchanged. Earlier cap and broad witness-merging variants below remain failed
experiments; they are not enabled in production.

The final implementation and matched evidence are described in the last
section. Historical results below must not be read as its current status.

`shape_face_interval.py` measures collision skin support clipped to the actual
finite face prism. `shape_surface_band.py` measures regional convex mesh and
analytic sphere distance to the finite shape/face contact band. Mesh regions
are cut before constructing their Minkowski cap. Sphere features retain their
actual radii, regional cuts, and rounded face edges/corners. Scalar derivatives
use ordinary autograd through moving geometry and current closest features;
there is no pose teacher, custom backward, QP layer, or inference projector.

The optional `surface_interval` input of `unified_region_objective` exercises
this geometry. It replaces the contact-side material surrogate only. Complete
terrain and self-collision clearance loss and raw depth reporting are retained.
Consequently this path does **not yet** unify the recovery of different
material regions or establish that their joint-space gradients cannot oppose.
It must not be advertised as a completed conflict-free training objective.

## Evidence on 2026-10-09

- All 3,105 cached current-Newton demonstration endpoints were checked using
  their unchanged native poses, current regional labels, and initialized
  geometry. All 15,426 labeled regions have finite face overlap and normal
  gaps within the actual 0–20 mm interval. These are geometric checks, not
  a replacement contact criterion. No held-out pose is used as a teacher.
- The first cap implementation clipped anatomical regions after choosing
  Minkowski hull point pairs. It falsely penalized 475 valid foot regions,
  by up to 52.041 mm. A Minkowski point has multiple possible terrain/body
  witness pairs; choosing one pair loses valid simultaneous contacts.
- Cutting source skin into anatomical regions **before** constructing caps
  removes this false rejection: all 15,426 labeled region band distances are
  zero to numerical precision (maximum 3.3e-7 mm).
- 44 related tests pass, including true fresh-query derivatives, flat soles
  with simultaneous regional contacts, rounded sphere edges/corners,
  side-to-top continuity, duplicate geometry, and batched gradient equality.
- Mesh edge/corner margins use Euclidean clearance, rather than a purely
  normal extrusion. Actual native candidate upper bounds remain in the loss;
  a smaller geometric band residual must not erase them. Enabled pair entries
  index compact exported shape records, not sparse native shape IDs. An audit
  caught and corrected this indexing mistake before any training used it.
  With all three corrections, every labeled source region has exactly zero
  band residual (`native_pair_indices/source_report.json`).
- Existing reverse-gradient cases remain. At equal250 indices 341, 416,
  and 1421, old contact/clearance q-space cosines are approximately -0.105,
  -0.124, -0.112; the regional band replacement gives -0.164, -0.182, -0.170.
  These comparisons use exactly the same frozen no-grad predicted poses.
  Actual physical depths remain unchanged at approximately 1.077, 4.960,
  and 9.622 mm. This is not evidence of resolved gradient competition.
- The colliding shape in all three examples is native shape 54,
  `left_ankle_roll_sphere_1_link`, radius 5 mm. Its center belongs to rear
  region 1. The predicted plan requires front region 3 on the target face.
  Opposing joint-space derivatives therefore involve different material
  regions of one rigid foot, rather than opposite bounds of one point.
- A negative cosine between two component gradients does not establish that
  their actual combined descent increases either cost. For these three
  equal250 cases, the original foot-only mean+max contribution has negative
  directional derivatives for both contact and clearance. This excludes
  other bodies, other losses and the network parameter update; it is not a
  global convergence claim (`gradient_direction_readout.json`).
- Fresh complete-solid queries confirm simultaneous descent for both original
  contact material and corrected shape-band geometry in these three poses,
  at all tested normalized pose steps from 1e-6 through 1e-3. At the largest
  step, original terrain depths change 1.077→0, 4.960→2.079 and
  9.622→6.709 mm; corrected-band depths change 1.077→0, 4.960→2.064 and
  9.622→6.688 mm. Contact geometry cost also decreases in every case.
  This checks foot-only descent with fresh Coal geometry; native candidate
  updates, other bodies/losses and Adam's network update are excluded
  (`fresh_direction_report.json`).

Audits: `tmp/shape_face_interval_20261009/source_audit.json`,
`tmp/shape_surface_band_20261009/source_report.json` (failed first construction),
`tmp/shape_surface_band_20261009/region_first/source_report.json` (corrected),
and `tmp/shape_surface_band_20261009/gradient_audit/setup/report.json`.

A matched 50-update comparison starts both arms from the exact control500
checkpoint and independent deep-copied Adam state at step 11600. It uses the
same frozen minibatches, teacher masks, dropout, LR 1e-5, original anatomical
mean+max aggregation, and other losses. Full train/validation/test endpoints
are evaluated, followed by recursive comparison. The experiment records the
unresolved reverse gradients; its purpose is to measure actual effects rather
than assume that changing distance geometry automatically resolves them.
No baseline promotion is authorized by a lower total loss alone.

Matched output: `tmp/shape_surface_band_20261009/matched50/setup/`;
live progress: `tmp/shape_surface_band_20261009/matched50/training_status.json`.

The first matched run omitted the native candidate maximum and used the
normal-only mesh edge margin. It failed effect preservation: at height 1.10,
mean penetration improved from 1.203 to 1.105 mm versus matched control,
but actual contact acceptance fell from 89.94% to 86.43%. In motion 4 raw30,
accepted prefix fell from nine to three and maximum terrain penetration rose
from 0.966 to 9.721 cm. Motion 8's prefix remained three. These are failed
prototype results, not evidence for adopting the path. All raw diagnostics
continue past rejection; only the first 15 events have demonstration targets.

The corrected paired run is stored separately under
`tmp/shape_surface_band_20261009/matched50_native_bounds/`. It retains native
candidate maxima and Euclidean edge margins; it does not alter Newton labels,
checkpoint architecture, margin, aggregation or any other loss weight.

## Corrected matched run

Both arms finished 50 updates with independent Adam states advancing
11600→11650. Initial train/validation/test physical rows were identical and
all recorded gradient norms were finite. The new path is **not promoted**.

| Split | Control contact accepted | Band contact accepted | Control mean penetration mm | Band mean penetration mm |
| --- | ---: | ---: | ---: | ---: |
| Train (1470) | 99.25% | 99.66% | 0.00022 | 0.00089 |
| Height 0.90 (780) | 75.64% | 72.18% | 0.529 | 0.583 |
| Height 1.10 (855) | 89.82% | 90.76% | 1.316 | 1.196 |

At height 0.90, 27 samples lose actual contact acceptance and none gain it.
Fifteen of these losses are in phase 260→281. At height 1.10, eight samples
gain and none lose contact acceptance. This is a measured asymmetric
generalization result; changing the geometry does not preserve every split.

| Raw30 diagnostic | Control50 | Corrected band50 |
| --- | ---: | ---: |
| Motion 4 accepted prefix | 9 | 9 |
| Motion 4 own-plan safe prefix | 11 | 15 |
| Motion 4 mean terrain depth, events 1–15 (cm) | 0.0652 | 0.0239 |
| Motion 4 maximum terrain depth, extra 15 events (cm) | 0.9148 | 13.8628 |
| Motion 8 accepted prefix | 3 | 3 |
| Motion 8 mean terrain depth, events 1–15 (cm) | 0.3176 | 0.2710 |
| Motion 8 maximum terrain depth, extra 15 events (cm) | 1.0710 | 0.9511 |

The large corrected-band motion 4 failure occurs outside the demonstration
chain. Extra steps 17–18 first fail own-plan realization; step 18 also requests
persistence without current contact. Step 19 then embeds; step 20's worst
complete-solid witness is native shape 81 (`right_ankle_roll_link`), with
138.628 mm depth and a valid upward normal. Newton separately reports
83.953 mm right-knee embedding. No invalid penetrating witness is reported
at this step, so the failure cannot be attributed to the previously corrected
Coal invalid-result issue. These are failure locations and observed ordering,
not proof of which network update caused the divergence.

Evidence: corrected `setup/matched_report.json`, `setup/provenance.json`,
`setup/updates.json`, `recursive/rollout_status.json`, and
`recursive/recursive_region_audit.json`. Raw diagnostics deliberately continue
after rejection; extra events have no demonstration pose teacher. Complete
terrain/self clearance remains active, and no geometric residual is relabeled
as actual contact. The path is an experimental regional contact geometry
replacement; global clearance coupling and effect preservation are unresolved.

## Shared interval bounds

The next experimental path returns `SurfaceIntervalBounds(lower, upper)`
from one selected shape/face feature, retaining the same anatomical region
and target-surface identity. Alternatives minimize their combined geometric
cost; the lower/upper minima must not come from different shapes/features.
Tie derivatives agree with the original scalar minimum. Bounds still never
establish actual Newton activation.

In the first shared-bound prototype, each region's contact-field recovery
merged with complete-solid and release lower bounds by maximum. This was an
overbroad merge: retaining a raw collision report did not retain its gradient.
The corrected identity handling below supersedes that merge. The prototype
attempted to avoid charging the same penetration
once as contact attraction and again as physical separation. Native candidate
upper evidence remains. The regional objective is one scalar:

`log1p(max(physical_lower, field_lower) + max(native_upper, field_upper))`.

Its ordinary autograd derivative applies one robust tail to the combined
interval violation. Complete terrain/self queries and depth reports, native
labels, margins, architecture, prior/layout/keep weights and fixed anatomical
mean+max aggregation stay intact. Different bodies/regions can still have
coupled kinematics; this formula does not prove that all network gradients
align or that a learned predictor remains safe outside its training inputs.

All 3,105 source endpoints / 15,426 labeled regions have zero shared-field
residual (`shared_bounds/source_report.json`). Regression checks cover
duplicate lower-bound accounting, retained physical terrain/self gradients,
one tail for simultaneous bounds, both derivative contributions, coherent
geometry choice and true derivatives of the original geometric scalar.
The complete old/new execution objective was also evaluated on all 3,105
native source target poses with identical cached Newton evidence and fresh
complete-solid queries. Every loss value, realized-region mask and raw depth
is identical; no source target acquires an extra penalty
(`source_objective/report.json`). 49 related tests pass.

Independent one-update audits of original/summed and then split-bound paths
are preserved under `tmp/contact_interval_update_audit_20261009/` and
`tmp/contact_interval_update_audit_shared_bounds_20261009/`. The second audit
preceded the final shared-tail change. Mean+max additive attribution can jump
when its dominant region changes: its component value increase alone is not
proof that raw contact violation increased. Raw-bound audits were added to
distinguish this allocation change from a genuine residual change. No audit
checkpoint was promoted or saved as a training result.

The shared-tail matched comparison is separate:
`tmp/shape_surface_band_20261009/matched50_shared_bounds/`. Adoption still
requires effect preservation on the full endpoint splits and raw30 comparisons.

The first shared-tail comparison does not pass: contact acceptance improves
versus its matched control (0.90: 70.51→72.18%; 1.10: 89.12→89.94%), but
1.10 mean penetration increases 0.978→1.180 mm. Motion 4 raw30 keeps a
nine-event accepted prefix but reaches 14.255 cm embedding outside the
demonstration chain. No baseline is promoted (`matched50_shared_bounds/outcome.json`).

Repeated controls with the same source, frozen stream and random seed vary
after the first update. A reset-at-control500 diagnostic isolates part of the
cause: in three ordinary-backward trials, forward pose, native geometry,
complete-solid geometry and loss hashes are identical, while gradient hashes
differ (maximum 9.54e-7) and Adam parameter updates differ (maximum 1.79e-7).
Three deterministic-backward trials are bitwise identical, including the
updated parameters. This demonstrates a numerical update discrepancy; it
does not by itself prove the cause of every 50-update metric difference.
Evidence: `tmp/contact_interval_update_audit_repro_20261009/setup/repro_report.json`.

The next paired comparison enables deterministic torch with its required
cuBLAS workspace configuration and includes an independent duplicate control.
Model and Adam tensors must match at every update, and both controls' final
endpoint rows must match. Its independent output is
`matched50_shared_bounds_deterministic/`; these numerical verification settings
do not change the objective, native simulation rules or checkpoint architecture.

An additional conditioning audit evaluates the local derivative coefficient
of the combined robust tail on saved, identical pre-update region costs.
Compared with the original lower-bound penalty its ratio is
`(1 + lower)/(1 + lower + upper)`. Across five frozen batches (640 rows,
including repeat samples), 46 of 117 positive lower-bound groups also have
positive upper error. The smallest ratio is 0.018325 at row 1829, region group
2. Thus the combined tail can substantially weaken a physical lower-bound
derivative when another witness in the same region has a large upper error.
This is a mathematical conditioning effect, not proof of the cause of any
recursive failure (`shared_tail_slope_audit.json`).

The research-only `shared_bounds_original_tails` ablation preserves the shared
geometry, coherent feature selection and recovery deduplication, but restores
the original separate bound penalties. It modifies only a captured research
function; production defaults are unchanged. A fixture with distinct upper
and physical lower witnesses verifies exact equality of its loss and gradient
to the original function when the added field bounds are zero. This isolates
the tail change without introducing a new penalty, weight or safety gate.

## Deterministic matched results and isolated tail ablation

Both new runs completed 50 updates per arm, independent Adam 11600→11650,
all endpoint splits, and raw30 motions 4 and 8. Each run's two controls match
bitwise in model/Adam state at every update and in all final endpoint records.
Across fresh worker processes the controls are not wholly bitwise identical:
maximum final endpoint depth differences are 0.0091, 0.0109 and 0.0266 mm
on train/0.90/1.10; one training contact verdict changes. Both held-out contact
counts are unchanged. Compare each variant with its own matched control;
the tail ablation is not proof of complete cross-process reproducibility.

| Variant | 0.90 contact, control → variant | 1.10 mean penetration mm, control → variant | Motion 4 extra15 maximum depth cm, control → variant |
| --- | ---: | ---: | ---: |
| Shared geometry, deduplication, combined tail | 74.74% → 71.92% | 1.197 → 1.169 | 0.958 → 14.831 |
| Shared geometry, deduplication, original bound tails | 74.74% → 68.33% | 1.195 → 1.091 | 0.925 → 6.943 |

The original-tail variant improves motion 4's demonstrated accepted prefix
9→10, and demonstrated maximum depth 0.499→0.224 cm. Motion 8's prefix
remains three. Its extra15 maximum depth worsens 0.896→2.664 cm.
Neither variant passes effect preservation; neither checkpoint is promoted.
Restoring the bound tails therefore does not fix the entire regression.
Shared geometry and recovery deduplication remain confounded in this ablation.

A further CPU forward audit uses the training pipeline's local-to-native
chart and compares reconstructed input q/contact/anchor/heightmap against
three saved training observations; all four have exactly zero error. On the
combined-tail variant's 23 lost 0.90 contacts (and two lost 1.10 contacts),
coarse contact/surface, regional intent, roles, target raster cells and target
points are unchanged. The original-tail variant's 51 lost 0.90 contacts also
retain all these intentions. No affected sample loses its imitation gate.
These observations locate the regressions in the emitted pose rather than a
changed intended contact plan. CPU/GPU forward arithmetic is not asserted to
match bitwise; paired Newton endpoint evaluations remain the contact truth.

On the 23 lost 0.90 rows, root displacement between control and combined-tail
variant averages 2.678 mm (maximum 6.490 mm); the maximum joint angle change
per row averages 0.626° (maximum 1.150°). For original tails' 51 losses these
are 3.359 mm / 6.398 mm and 0.910° / 2.062°. This is not a root-only change.
It does not establish which parameter-gradient term caused the pose shift.

Evidence in `matched50_shared_bounds_deterministic/` and
`matched50_shared_bounds_original_tails_deterministic/`:
`outcome.json`, `setup/updates.json`, `setup/matched_report.json`,
`recursive/rollout_status.json`, and `plan_change_audit.json`.
The first plan audit accidentally used the box chart instead of the training
local-to-native chart. Its retained `plan_change_audit_wrong_frame_INVALID.json`
is invalid and must not be used. The intermediate `*_without_cells.json`
audits are superseded by the complete target-cell/point comparisons.

The implementation establishes a tested shared interval interface and removes
duplicate recovery accounting on the experimental path. It does not establish
that the original reverse-gradient examples were erroneous constraints, that
changing geometry preserves learned generalization, or that all body/parameter
updates become mutually compatible. Formal training remains on control500's
existing geometry and objective pending a successful comparison.

## Original-material interval ablation

The next ablation preserves the exact control geometry rather than expanding
its regional material support to a shape cap. `material_interval.py` uses the
same fixed training Newton material bank and the same explicitly reported
asset proxy for unknown regions, unchanged link FK, surface chart, native
margin and closest combined feature. Lower and upper components share that
feature; exact ties average their derivatives as the original scalar minimum.
This is a diagnostic adapter, not a newly adopted contact definition or a
replacement for complete-solid coverage.

Four new tests verify scalar/derivative equality, coherent alternative
selection, exact tie derivatives, and actual control500 material-bank/FK
gradients on three saved reverse cases with original, lowered and raised root
poses. All pass. The complete unchanged-Newton source objective audit checks
all 3,105 source endpoints: maximum increase and mean difference are zero,
with unchanged native region masks and physical depths
(`material_bounds_source/report.json`).

`matched50_material_bounds_original_tails_deterministic/` isolates splitting
and recovery deduplication with the original robust penalties, original
geometry and all other objectives unchanged. Independent duplicate controls,
the frozen 50-update stream and subsequent full endpoint/raw30 checks are
retained. No checkpoint is promoted based on the source-only checks.

That comparison also fails effect preservation. Contact acceptance changes
74.23%→71.79% on the 0.90 split and 89.94%→90.99% on the 1.10 split.
Their mean penetration changes 0.545→0.649 mm and 1.202→1.208 mm.
Motion 4's demonstrated accepted prefix falls 9→6; extra15 maximum terrain
depth rises 0.910→5.956 cm. Motion 8 remains at prefix three, with extra15
maximum depth 0.900→2.244 cm. These are paired within-run comparisons;
restoring the original geometry alone does not make the merge acceptable.

## Collision identity and independent derivatives

A regression fixture proves a bug in the experimental regional maximum:
a large target recovery along z suppressed the x derivative of a smaller
independent self collision. The self penetration was still reported.
This is direct evidence of an implementation error, not yet evidence that
the same mechanism caused the measured held-out regressions above.

`SurfaceSolidOwners` binds each actual primary target face to its initialized
static shape/component, with model fingerprint and shape/component/link
checks. Component identity matters: ground and box can share native shape ID
zero while belonging to different convex components. Unknown ownership fails
explicitly. This geometry identity check does not define Newton contact.

Only a target-component match in the desired part-region may now share the
interval recovery bound. Self collisions, other static components and release
remain independent physical constraints; their penalty is not hidden behind
the target recovery maximum. With no identity binding, physical lower bounds
remain independent. This adds no new task constraint, asset, label criterion,
margin, model change or tuning weight. It corrects which existing constraints
may be treated as duplicate evidence. Within a part-region, distinct robot
features can still interact; static ownership alone is not proof of identical
material witnesses or absence of kinematic competition.

All 62 focused tests pass, including the retained self-collision derivative,
actual static component ownership, coherent bounds and geometry derivatives.
The identity-bound source check again covers all 3,105 endpoints: no objective
increase or mean change, and unchanged native masks and physical depths
(`material_bounds_source_identity/report.json`).

The independent paired comparison is
`matched50_material_bounds_identity_deterministic/`. It retains the original
material geometry and robust bound penalties, complete physical queries,
all other losses and the frozen 50-update stream. Its source snapshots and
SHA-256 hashes record the exact experimental implementation. Formal training
still uses the original default until full endpoint and recursive comparisons
establish effect preservation.

The original-tail identity comparison fails as well. On 0.90, contact
acceptance changes 74.23%→71.79% and mean depth 0.544→0.631 mm; on 1.10,
contact changes 89.94%→90.53% while mean depth increases 1.203→1.275 mm.
Motion 4's accepted prefix falls 9→6 and extra15 maximum depth increases
0.910→6.983 cm. Motion 8 retains prefix three but extra15 maximum depth
increases 0.902→2.386 cm. The two duplicate controls match at every update
and in all final endpoint records. No checkpoint is promoted.

All 25 lost 0.90 contact samples retain their coarse/regional intent, role,
target cell and target point. The four lost 1.10 samples retain contact,
surface, role and regions, but two change their target cells/points.
This audit must not be generalized to unchanged target positions on every
held-out failure (`plan_change_audit.json`).

## Material robustness ordering

The term "original bound tails" in the ablation names is not scalar
equivalence to control. Control robustifies its complete material residual
as `log1p(field_lower + field_upper)`. Splitting that material query and then
robustifying both components independently as
`log1p(field_lower) + log1p(field_upper)` changes their relative derivative
coefficients. Geometry, minima, weights and source-zero residuals can all
be unchanged while the gradient at a predicted pose changes substantially.

A frozen control500 CPU diagnostic on the first five saved minibatches
(640 rows including repeats) finds 80 rows with simultaneous material bounds;
all 80 change scalar cost under the separate-tail identity path. The smallest
cosine with the original material/solid pose gradient is -0.812. Restoring a
common material tail reduces cost changes to ten rows, representing two
repeated sample IDs; 630 rows have maximum q-gradient difference below 1e-4,
and the smallest nonzero-gradient cosine is 0.999985. No independently
suppressed collision is observed in this frozen subset. The earlier synthetic
self-collision regression is still valid; it is not a measured explanation
for these 80 changes.

This diagnostic includes actual initialized complete solids and unchanged
material geometry, but excludes native upper/release evidence, layout/prior/
keep losses and Adam. It is not complete training-gradient attribution or
contact truth (`recovery_accounting_joint.json`). A permanent regression
checks scalar and gradient equality to control when both material bounds are
positive and independent physical violations are zero.

`matched50_material_joint_identity_deterministic/` therefore tests original
material geometry with common material robustness and the corrected collision
identity guard. It changes no label, margin, asset, model or tuning weight.
Its own paired controls and all endpoint/raw30 comparisons remain required
before any baseline adoption.

Shared-tail component attribution is not a pair of independently defined
task objectives. The reported additive components distribute `log1p(L+U)`
using `L/(L+U)` and `U/(L+U)`. Differentiating those allocations introduces
opposite ratio-derivative terms that cancel in the total. Consequently a
negative cosine between their gradients, including the experimental
`setup/report.json` diagnostics after identity changes, is not evidence
of a wrong combined recovery direction. Use raw residual adjoints and the
complete objective/update instead. The frozen accounting audit above compares
the complete scoped scalar gradients; it does not compare these allocations.

## Common-material-tail matched outcome

The common-tail identity run completed 50 updates, independent Adam
11600→11650, and all endpoint/raw30 comparisons. Within-run duplicate
controls match at every update and in all final endpoint rows.

| Split | Contact accepted, control → interval | Mean depth mm, control → interval |
| --- | ---: | ---: |
| Train | 99.18% → 99.59% | 0.00041 → 0.00114 |
| 0.90 | 75.00% → 75.13% | 0.545 → 0.611 |
| 1.10 | 89.82% → 91.93% | 1.224 → 1.155 |

Train gains six accepted contacts and loses none; 1.10 gains 18 and loses
none. The 0.90 split gains seven and loses six. Its losses are now three
183→260 rows and three 573→621 rows, not the earlier 260→281 cluster.
All six lost rows retain contact/surface/role/region/cell/point intentions.
These are exact paired counts, not proof of complete effect preservation.

Motion 4 retains accepted prefix nine and own-plan safe prefix eleven.
Its first15 maximum terrain depth changes 0.521→0.528 cm, but extra15
maximum depth worsens 0.949→11.692 cm. Motion 8 retains both prefixes at
three, with first15 maximum depth 2.759→2.817 cm and extra15 maximum
depth 0.939→2.058 cm. Both diagnostics continue after rejection, and
extra events have no demonstration endpoint teacher.

In interval motion 4, event 18 intends both feet and both hands but only
the hands realize contact; terrain depth is still zero. Event 19 then
reports 8.367 cm maximum terrain depth, and event 20 reaches 11.692 cm.
The primary-foot material residuals at event 18 are nonzero (left-foot
regions approximately 0.961–2.459 mm), so this is not a demonstrated
zero-loss acceptance of every missing foot. Extra-event native candidate
arrays are absent from these serialized diagnostics; their normals and
allocation cannot be inferred from missing audit fields.

The scalar/gradient ordering correction removes an experimentally verified
distortion and restores much of the endpoint contact behavior, but does not
preserve 0.90 depth or outside-chain recursion. The interval path therefore
remains experimental and control500 is unchanged. Evidence:
`matched50_material_joint_identity_deterministic/outcome.json`,
`setup/source_snapshot/`, `plan_change_audit.json`, and
`recursive/recursive_region_audit.json`.

## Final coherent query identity and production integration

The broad entity/region merge above was incorrect. A native collision witness
and a regional material query can refer to different points even when their
body and target component agree. Combining a physical lower bound from one
query with a material upper bound from another, or changing the order of
query selection and robustification, changes the optimization problem.

The complete frozen-objective audit covers 6,268 actual training rows from
50 saved minibatches, including demonstration and saved-pool inputs, original
teacher/dropout decisions, native candidates, and all other losses. Before
the correction, only three saved-pool states (3059, 734 and 1874), each
repeated 26 times, changed the objective. The corrected audit has exactly
zero scalar, execution-gradient and total-gradient differences for all
6,268 rows. This is a measured scope, not a universal equivalence proof.

`material_surface_interval.py` now supplies coherent lower/upper bounds,
the original ordinary minimum scalar and its tie derivative, and selected
material-query identity. `shape_target_interval.py` binds them to initialized
static components. Its geometry adapter protocol is reusable; the default
adapter represents the current ground/box scene and rejects unsupported
target faces rather than guessing their geometry.

The objective compares complete native/material query costs before applying
the original robustness. It never mixes bounds from different queries.
A physical row can be covered by the material interval only when static
component, selected unique interior material, link, local point, normal,
signed gap and lower residue identify the same affine constraint. The upper
residue must be inactive. Numerical identity tolerances are not new contact
or penetration thresholds. Different witnesses, self collisions, other
static components and release constraints retain their independent coverage.
No QP teacher, custom backward, projection, loss weight or root cost was added.

This removes contradictory bookkeeping of the same geometric query. It
does not require gradients from different rigid-body features to align:
one feature can legitimately require lifting while another requires rotation.
The earlier three opposite-cosine examples involve different front/rear foot
features and are not duplicate point constraints. Component-gradient cosine
alone remains insufficient evidence of a wrong combined update.

The corrected research run and the separate production-entry run each
complete 50 matched updates from control500 with independent Adam states,
unchanged frozen inputs and LR 1e-5. In both runs the final model and Adam
tensors are bitwise equal, and all 3,105 endpoint rows are exactly equal.
The production test actually uses `Workers.target_interval` and the canonical
provider, with the control flag false and the interval flag true.

Production endpoint results are identical in the two arms:

| Split | Accepted contacts | Mean penetration mm |
| --- | ---: | ---: |
| Train | 99.2517% | 0.0004135 |
| 0.90 | 74.8718% | 0.546385 |
| 1.10 | 90.0585% | 1.212424 |

Both motions run 30 recursive steps, including continuation after rejection.
Motion 4 outputs, contacts, depths and acceptance are exactly equal. Motion 8
has a maximum control/new depth difference of 0.00004908 mm; an independent
duplicate-control run reproduces the new path's outputs, depths and acceptance
exactly at all 30 steps. This identifies worker/runtime numerical variation,
not a training-loss change. Successful prefixes remain 9 and 3 respectively,
and own-plan safe prefixes remain 11 and 3. Effect preservation therefore
passes; the baseline's existing recursive failures are not claimed repaired.

`Config.shared_shape_target_interval=True` enables this path for maintained
unified-contact training. Independent research objectives remain unchanged.
The dataset contract records
`coherent_shape_target_material_query_interval_v2`, scene fingerprints and
adapter identity. New source modules contain no dependency on temporary
experiment scripts. The control500 checkpoint hash remains
`95c72146267da4ed47554192243076d0e62d61bb07db98e52cc1166f52397b0b`;
neither matched checkpoint has replaced it.

Validation: 102 contact-solver, geometry, initialization and training-entry
tests pass, including edge/rotation geometry, mismatched witnesses,
independent/self-collision coverage and equal scalar/derivative regressions.

Evidence under `tmp/shape_surface_band_20261009/`:

- `fixed_pose_full_objective_query_identity/setup/report_full.json`
- `material_bounds_source_query_identity/report.json`
- `matched50_material_query_identity_deterministic/outcome.json`
- `matched50_production_query_identity_deterministic/outcome.json`
- `matched50_production_query_identity_deterministic/equivalence.json`
- `matched50_production_query_identity_deterministic/setup/source_snapshot/`
- `production_tests_run2.log`
