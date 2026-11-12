> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Frozen latest Predictor → Direct Infiller experiment

> Historical experiment: the Infiller implementation and entrypoints were retired
> on 2026-10-23. Data, checkpoints and the original results below remain archived.


This experiment freezes the checkpoint pinned by
`baselines/predictor_v1_latest_20261010/baseline.json`. It trains a new Direct
Infiller with the historical 182 → 256 → 256 → 256 → 36 tanh network, cubic
endpoint interpolation and a residual envelope that vanishes at both ends.
The old checkpoint is not used as initialization. Both caller-supplied endpoint
poses, including their quaternion representations, are preserved exactly.

The corpus contains native 50 Hz demonstration segments and frozen Predictor
segments admitted by current Newton contact, native regional position, joint
limit and complete terrain/self collision checks. Demonstration contact pairs
are explicitly rematerialized through the shared primary upward face selector.
Raw source pair files are referenced by path and SHA256; new endpoint and dense
evaluation queries retain raw solver pair tensors in audit shards. Active
contacts without allocated constraints fail the run.

Predictor contacts describe intent. Their early leaving / late arriving windows
are declared optimization windows, not demonstration timing or achieved
contact labels. This Predictor has no duration head; segment durations come
from the source event templates and are recorded. Infiller inference sees the
two boundaries, boundary contact contract, scene, phase and duration. It does
not receive the source intermediate trajectory. Demonstration intermediate
poses and material witnesses are training supervision only.

By default, training alternates the two boundary sources. With
`--training-source demonstration_only`, both optimizer batches and checkpoint
selection use demonstration records only. Generated records remain in the
immutable corpus for evaluation, including recursive chains. The control uses
the same losses and duration templates as the mixed experiment, so it isolates
the boundary-source change; it does not repair the missing inactive-contact
loss or historical budget/duration conflicts.

Training uses the historical 18
derivative budgets, weak pose prior, Adam and cosine learning rate, with dense
terrain penalties and current material face intervals. Random interior complete
solid checks also penalize terrain and self collision. The 2 mm optimization
allowance and 5 mm residual normalization do not define solver contact.
`best.pt` is selected by the balanced validation optimization loss; final
Newton evaluation supplies independent contact and geometry observations.
`last.pt` records the final training step.

Run from the repository root in the pinned Newton environment:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m generator.train_frozen_predictor_infiller --headless --device cuda:0 \
  --steps 5000 --evaluation-every 250
```

Official AppLauncher arguments are parsed before the project configuration.
The existing query worker launches the actual Newton runtime headlessly.
`--evaluation-splits test` evaluates the complete test split without using it
for model selection; omitted split selection evaluates every corpus split.
Training refuses an existing output directory. `--mode build`, `--mode train`
and `--mode evaluate --checkpoint PATH` can run stages separately. A dataset
cap is allowed only with an explicit `--smoke` flag and separate output paths.

Evaluation reports demonstration and predicted boundaries separately for train
heights 0.95/1.00/1.05, validation 0.90 and test 1.10. Penetration checks run at
500 Hz on nlerp between native output samples; contact episodes use unfiltered
actual 50 Hz Newton activation/allocation and primary face observations.
Reports include establishment times, demonstration timing errors where a
reference exists, same-surface contact interruptions, per-segment derivatives,
and seam jumps between genuine adjacent segments in the same world frame.
Failed predicted endpoints terminate their generated chain without projection
or retry during corpus construction. The separate `rollout` mode continues
ordinary endpoint rejections, records each failure, and feeds the Infiller's
exact last pose and fresh actual Newton contacts into the next Predictor call.
Only the requested demonstration's first pose and scene initialize recursion;
future demonstration poses and duration templates do not enter inference.
Missing solver fields or unallocated active contacts still fail explicitly.
With the current Predictor's lack of a duration head, rollout uses a configured
fixed duration, rounded to native 50 Hz sampling. For 30 one-second segments:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m generator.train_frozen_predictor_infiller --headless --device cuda:0 \
  --mode rollout \
  --checkpoint runtime/current/models/generator/direct_infiller_latest_predictor_20261010/best.pt \
  --report tmp/direct_infiller_recursive30_20261010 \
  --rollout-events 30 --rollout-duration 1.0 --rollout-motion-id 0
```

The resulting `rollout/predictor_infiller_recursive30.npz` contains all 1,501
native frames, including rejected segments, and loads in the existing Contact
Editor alongside `rollout/terrain.obj`. Per-event records, raw endpoint pairs,
dense fixed-q query shards, and contact semantic/mapping metadata are retained
beside the motion. Frozen checkpoints are verified unchanged.

Fixed-pose observations do not establish dynamic support or successful
policy execution. Height holdouts use the same original motion source and are
not independent-source generalization tests.

Artifacts are under:

- `runtime/current/motions/infiller_latest_predictor_20261010/corpus.pt`
- `runtime/current/models/generator/direct_infiller_latest_predictor_20261010/`
- `tmp/direct_infiller_latest_predictor_20261010/`

The smoke run under the corresponding `smoke_r2` paths completed two training
steps and evaluated 91 segments. It checks pipeline operation and is not an
accepted Infiller baseline. The formal experiment remains experimental until
its completed report is reviewed.

## Unified trajectory objective (2026-10-11)

The `--objective unified` experiment keeps the 182-input decoder and exact
boundaries. It replaces the historical sampled collision penalty and fixed
2000-weight derivative budget with one objective applied to every training
record:

- Full expert pose trajectories supervise demonstration interiors, with
  quaternion signs aligned and coordinates normalized by the decoder scales.
  Their pose weight is 10, rather than the historical 0.01 weak prior.
- Native observations define demonstration contact/release intentions at
  50 Hz. Generated records specify boundaries and same-surface persistent
  contacts only; their historical 20/80 percent timing and cubic intermediate
  poses are not expert targets. Unspecified generated timing stays unknown.
- Every native frame and every inter-frame midpoint uses fresh same-state
  Newton activation/allocation queries plus initialized complete-solid
  distances. The shared Predictor `unified_region_objective` and coherent
  shape/material target intervals supply contact, explicit release and full
  terrain/self separation. Midpoint intentions require consistent neighboring
  native annotations; they are not interpolated contact truth. Temporal mean
  plus per-segment maximum prevents a small set of violations being diluted.
- The pinned Predictor contact-material bank is verified against its SHA256,
  canonical asset, current contact/face semantics, link mapping and training
  sample IDs. No validation/test materials enter it. Unknown material regions
  retain explicitly reported loss-only asset approach geometry.
- Smoothness is a soft `log1p` excess penalty with weight 1. Each effective
  limit is the maximum of the historical limit and 1.5 times the derivative
  of that record's expert trajectory or generated cubic reference at its
  actual duration. That reference witnesses derivative-budget feasibility;
  this does not establish joint feasibility with all contact/collision
  constraints. Original historical budgets remain unchanged for reporting.

Immutable boundaries are queried and audited, but their constant defects do
not dominate the interior loss or checkpoint selection. The physical weight
is 1. These are global recipe settings, not case-specific repairs. Inference
does not project, filter, retry, or reset generated trajectories.

The first run initializes decoder weights from the completed mixed baseline,
starts a fresh Adam optimizer, freezes the same Predictor, and uses all 4111
training records and all 1501 validation records. Initial full validation is
included in checkpoint selection so the starting model remains best if
updates do not improve it. Test trajectories never enter optimization or
checkpoint selection. A separate smoke check completed two optimizer steps
with actual Newton and finite gradients; it is not an accepted model.

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m generator.train_frozen_predictor_infiller --headless --device cuda:0 \
  --mode train --objective unified \
  --initialization-checkpoint runtime/current/models/generator/direct_infiller_latest_predictor_20261010/best.pt \
  --output runtime/current/models/generator/direct_infiller_unified_v1_20261011 \
  --report tmp/direct_infiller_unified_v1_20261011 \
  --steps 1500 --batch-size 16 --frame-budget 512 \
  --learning-rate 0.0002 --evaluation-every 500
```

After training, evaluate the complete test split using the existing unchanged
500 Hz native/complete-solid penetration and 50 Hz actual contact protocol,
then compare multiple 30-event recursively generated trajectories. The unified
run remains experimental; its completed comparisons are recorded below.
Predictor endpoint failures remain a separate part of the joint optimization task.

### Completed unified comparison

The 1500-step run completed normally. Validation selected step 1500 with a
balanced loss of 1.130948, versus 1.831771 for the initial old model. All 1659
test records and three 30-event recursive comparisons (initial motion IDs
0, 2, 7, selected before evaluation) completed. Predictor and original Infiller
checkpoint hashes were unchanged. The implementation checks passed 46 tests.

| Test metric | Old | Unified |
| --- | ---: | ---: |
| Demonstration boundaries: mean segment maximum terrain depth | 10.68 mm | 3.70 mm |
| Predictor boundaries: mean segment maximum terrain depth | 7.29 mm | 2.17 mm |
| Demonstration boundaries: mean persistent-contact coverage | 91.07% | 95.22% |
| Predictor boundaries: mean persistent-contact coverage | 98.93% | 99.44% |
| Predictor boundaries: mean joint seam velocity jump | 0.545 rad/s | 0.876 rad/s |
| Demonstration root position RMSE, mean over segments | 44.70 mm | 45.76 mm |
| Demonstration joint RMSE, mean over segments | 0.1309 rad | 0.1282 rad |

Long recursion did not show a consistent improvement in combined acceptance:

| Initial motion | Old / new accepted endpoints | Old / new mean segment peak depth | Old / new worst depth | Old / new accepted endpoint and interior depth <= 5 mm |
| --- | ---: | ---: | ---: | ---: |
| 0 | 20 / 20 | 22.88 / 15.46 mm | 149.46 / 143.08 mm | 11 / 12 |
| 2 | 16 / 16 | 22.89 / 16.15 mm | 149.01 / 142.40 mm | 8 / 6 |
| 7 | 19 / 19 | 22.26 / 13.61 mm | 151.16 / 144.70 mm | 11 / 11 |

The endpoint arrays differed by at most 1.61e-6 in absolute coordinates across
matched runs; endpoint acceptance counts were unchanged. Depths use the
unchanged maximum of native Newton and complete-solid diagnostics. Their
signed-distance conventions differ: for example, at the new seed-0 worst
sample the complete-solid depth is 43.57 mm while the reported native/solid
maximum is 143.08 mm. Neither value defines contact activation.

Historical derivative-budget violations also increased: 488 to 515 of 855
demonstration-boundary test segments, and 400 to 425 of 804 Predictor-boundary
segments. Those original budgets include known endpoint/duration conflicts;
the counts are retained without redefining the old metric.

Conclusion: the unified objective improves average test penetration and
contact coverage, but does not resolve long-recursive failures and worsens
some continuity metrics. Keep this checkpoint as an experimental candidate,
not a replacement accepted baseline. These are fixed-q observations of
same-source height holdouts, not independent-source or dynamic execution
validation. No test trajectory was used for training or checkpoint selection.

Artifacts:

- Model: `runtime/current/models/generator/direct_infiller_unified_v1_20261011/best.pt`
- Paired report: `tmp/direct_infiller_unified_v1_20261011/comparison_summary.json`
- Per-test-segment comparison: `tmp/direct_infiller_unified_v1_20261011/paired_test.csv`
- Recursive curves: `tmp/direct_infiller_unified_v1_20261011/recursive_penetration_comparison.png`
- Recursive motions: `tmp/direct_infiller_unified_compare_{old,new}_seed{0,2,7}_20261011/rollout/`
  (old seed 0 remains at `tmp/direct_infiller_recursive30_20261010/rollout/`).

## Recursive coverage and continuity refinement

The next authorized experiment adds full 30-event chains for all 98 training
and 52 validation initial states. Every ordinary rejected endpoint is retained
and fed into the next prediction. Missing solver fields and unallocated active
constraints still fail explicitly. No test chain is collected for training.
The original 7271 records are preserved; 2940 train and 1560 validation records
are appended as `predictor_recursive`, including raw same-state Newton pairs,
shape/surface mappings, exact boundary contracts, and individual failure audits.
These generated intermediates are cubic references for feasible smoothness,
not expert pose or contact-timing supervision. Their absent material witnesses
stay unknown; unified training obtains fresh full geometry constraints.

Infiller refinement initializes from the completed unified checkpoint, keeps
the original Predictor frozen, and adds sampled seam loss with weight 1.
Only genuinely adjacent records in one split, chain, and Newton scene form a
pair. Both neighboring predictions receive gradients. Each uses its native
sample spacing and actual duration. Root and joint velocity differences and
sign-invariant rotation-matrix velocity differences use the existing decoder
scales and a log1p penalty. The architecture already has zero analytic endpoint
velocity; this added term constrains the finite differences reported by the
unchanged 50 Hz evaluator. Position boundaries remain exact.

Demo and generated sources retain equal aggregate selection weight. Within
the generated half, historical accepted records and new unfiltered recursion
receive equal weight. Model selection includes all 3061 validation records
and their genuine adjacent pairs; the initial checkpoint is also a candidate.
The proposed formal schedule is 1500 steps, learning rate 0.0001, batch size
16, and native frame budget 512. This experiment is called recursive v2 only
to identify its training iteration; the Predictor architecture remains v1.

`generator.refine_predictor_roles` separately freezes the shared encoder,
location/region heads, and pose executor, and trains only the role head.
Fresh Newton observations of train/validation inputs supply current contact.
The role loss penalizes probability assigned to keep when current contact is
absent, alongside demonstrated event-role cross entropy. Generated states
have no future-event teacher labels. Distillation preserves the parent's
relative probabilities among semantically possible roles with weight 0.1.
There is no inference mask, pose projection, or manual role replacement.

The role candidate must improve the validation role objective without reducing
actual Newton endpoint acceptance in either the demonstration or generated
validation group. Demonstration gates also require the demonstrated target
contact layout. The unchanged parent is retained if no candidate passes.
Every non-role state tensor is checked unchanged at checkpoint save. Training
uses a fresh optimizer; its provenance is recorded separately from the parent
scratch-training configuration. No test record enters feature collection,
optimization, or selection.

Final evaluation uses the unchanged 1659 original test records to isolate the
Infiller change, followed by the predefined initial states 0, 2, and 7 with
30 segments each. Infiller-only, role-only, and combined candidates are
compared separately. An explicit rollout-only Predictor override requires the
same recorded parent checkpoint and reports both the runtime Predictor and
the Infiller's training Predictor. An unchanged selected role parent skips
redundant comparisons.

The pipeline and per-stage logs are saved under
`tmp/recursive_refinement_v2_20261011/`. Extended corpus, Infiller, and role
checkpoints use new directories; original datasets and checkpoints remain
immutable. These candidates remain experimental pending the completed
matched comparisons.

### Completed recursive refinement comparison (2026-10-11)

All stages completed: 400 role-head updates, 1500 Infiller updates, the unchanged
1659-record test set, and nine 30-event test rollouts. The 52 targeted unit tests
and four-update Newton/seam-gradient smoke passed before formal training.
The extended corpus contains 2940 new training and 1560 new validation records;
167 and 505 rejected endpoints respectively remain in those records. Every
original test-record tensor was checked unchanged. No test trajectory was used
for optimization or checkpoint selection.

The selected Infiller is
`runtime/current/models/generator/direct_infiller_recursive_v2_20261011/best.pt`
at step 1500. Training completed normally in 3708.5 seconds, with the original
Predictor unchanged. Under this round's expanded validation objective, balanced
loss decreased from 2.169833 to 1.532963. Demonstration / historical accepted /
new recursive losses changed from 3.195248 / 0.259427 / 2.029408 to
2.706477 / 0.165715 / 0.553183. These scores include the added seam objective and
must not be compared directly with the previous round's 1.130948 score.

The complete original test set reveals a tradeoff relative to unified v1:

| Metric | Unified v1 | Recursive refinement |
| --- | ---: | ---: |
| Demo mean segment peak terrain penetration (mm) | 3.698 | 4.607 |
| Demo worst terrain penetration (mm) | 30.572 | 52.388 |
| Generated mean segment peak terrain penetration (mm) | 2.175 | 2.091 |
| Demo mean joint seam velocity jump (rad/s) | 1.102 | 0.673 |
| Generated mean joint seam velocity jump (rad/s) | 0.876 | 0.276 |
| Demo mean root seam velocity jump (m/s) | 0.1323 | 0.0990 |
| Generated mean root seam velocity jump (m/s) | 0.1300 | 0.0730 |
| Demo contact-establishment failures | 239 / 1069 | 182 / 1069 |
| Generated contact-establishment failures | 294 / 817 | 471 / 817 |
| Demo keep coverage | 0.95224 | 0.94737 |
| Generated keep coverage | 0.99437 | 0.99381 |
| Demo root / joint RMSE | 45.757 mm / 0.12825 rad | 42.883 mm / 0.12333 rad |
| Original smoothness-budget violations, demo / generated | 515 / 425 | 467 / 399 |

All establishment failures in both evaluated models are cases already active
on the target surface with no later reactivation; none is an inactive endpoint.
Thus the increased generated-group failures concern missing release/recontact,
not newly missing endpoint contacts. The original metric remains unchanged.
This distinction does not erase the regression or prove correct event intent.

Thirty-event rollouts use the same predefined test initial states 0, 2, and 7.
The pass count below means endpoint acceptance AND interior terrain/self depth
at most 5 mm; it is not a full contact-transition or dynamic-support certificate.

| Variant | Pass counts for seeds 0 / 2 / 7 | Total / 90 | Worst interior depth (mm) |
| --- | --- | ---: | ---: |
| Historical original | 11 / 8 / 11 | 30 | 151.164 |
| Unified v1 | 12 / 6 / 11 | 29 | 144.698 |
| New Infiller only | 17 / 12 / 17 | 46 | 71.427 |
| New role head only | 12 / 6 / 11 | 29 | 160.152 |
| Both refinements | 18 / 12 / 17 | 47 | 66.293 |

Every variant has endpoint acceptance 20 / 16 / 19, totaling 55 / 90. With the
unchanged Predictor, seeds 0 and 7 have exactly identical endpoint arrays;
seed 2 has maximum absolute q-component numerical difference 9.02e-5 between
the previous and new run. No projection, teacher reset, or failure filtering
was introduced. The new Infiller has a new worst failure at seed 2's final
segment, despite reducing the previously dominant mid-chain peaks.

Role refinement selected step 400 at
`runtime/current/models/generator/predictor_role_refinement_20261011/best.pt`.
Validation keep-precondition failures (absent current contact OR different
surface) changed from 10 / 150 to 10 / 105 in demo / generated groups. Actual
endpoint acceptance stayed 542 / 780 and 1776 / 2281. All non-role parameters
were verified unchanged. The rollout comparison provides no substantial
endpoint improvement from this head-only change.

The candidates remain experimental: long-recursion penetration and sampled
continuity improve, but demonstrated penetration and generated contact
transitions regress. No baseline is promoted or overwritten. The next issue
to resolve is the shared event contract for current contact, intended surface,
release, and recontact, together with the continuity/contact tradeoff; changing
an acceptance threshold would conceal the problem.

Full metrics and pairwise test rows are in
`tmp/recursive_refinement_v2_20261011/comparison_summary.json` and
`paired_test.csv`, with `recursive_comparison.png` showing all five variants.
All rollouts are exported as canonical 1501-frame, 50 Hz NPZ trajectories.
These remain same-source height holdouts and fixed-q Newton observations;
they establish neither demonstration-only natural generalization nor dynamic
load-bearing execution.
