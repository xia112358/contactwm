> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Time-conditioned Infiller with generated boundary-velocity history

> Historical experiment: the Infiller implementation and entrypoints were retired
> on 2026-10-23. Data, checkpoints and the original results below remain archived.


This experiment follows independent timing training. The frozen time head
already predicts demonstration frame counts accurately (0.149 test-frame MAE),
but the prior decoder forces zero analytical endpoint velocities. At the
predicted short durations, its native 50 Hz seams and persistent contact
coverage regressed. The experiment tests a velocity-aware trajectory
parameterization without updating timing, Predictor, scene geometry, contacts,
or acceptance thresholds.

## Architecture and supervision

The 172-feature frame head is copied from
`direct_infiller_frames_decoupled_v2_20261011/best.pt`. No new frame warmup runs.
The selected frame parameters are frozen and checked bitwise after every
trajectory update. Predicted duration conditions the trajectory; trajectory
gradients cannot update the time head.

The trajectory decoder retains its 182/256/256/256/36 residual network and
adds two 128/128 velocity heads plus a zero-initialized 70-to-182 velocity
adapter. Physical velocities have 35 components: world root translation (3),
world angular velocity (3), and joints (29). Heads predict unbounded corrections
to endpoint secant velocities, normalized using existing decoder pose scales
and the predicted duration. These scales are normalization, not velocity caps.

Root position and joints use cubic Hermite interpolation with initial and
terminal velocity conditions. Rotation uses a world-frame SO(3) logarithm and
exponential, with the inverse left Jacobian converting terminal angular
velocity to the terminal Lie-coordinate derivative. A squared endpoint gate
preserves Hermite endpoint derivatives for the learned interior residual.
Exact caller poses, including quaternion sign representations, remain fixed
at both endpoints. Existing interior joint-limit clamping remains: it can
alter derivatives where a curve hits a limit, so the model does not guarantee
unconditional C1 continuity of its clamped, finite-rate output.

Only real demonstrations supply velocity labels. Labels are their native
first/last 50 Hz finite differences; they are estimates of boundary velocity,
not exact instantaneous derivatives or integrated Newton state. Generated
accepted and recursive corpus interiors are placeholder interpolations and
never supply velocity labels or velocity history.

An expert-only 3000-update velocity warmup trains both velocity heads while
leaving the time head, residual network, and adapter unchanged. Validation
demonstration normalized velocity MAE selects the warmup. The terminal head
uses the model's estimated initial velocity as context, with history marked
unknown; expert velocity labels are never fed as its input. Velocity heads
then remain trainable under demonstration velocity supervision and the full
unified trajectory objective.

## Self-generated history and evaluation

True predecessor relationships require the same split, kind, chain and scene,
exact shared poses and adjacent source indices. Training contexts decode the
current model's predecessor prefixes at their own rounded native durations,
then compute incoming velocity from the last two generated frames. Prefix
history is detached per update; current seam pairs retain gradients through
both trajectories. Isolated records and initial chain segments have unknown
history and use the initial velocity head.

Complete test evaluation and recursive rollout use the same generated native
velocity history. They never consume demonstration interior poses as context.
Ordinary Predictor rejection continues the chain without a reset; the next
Infiller receives the previous generated segment's terminal finite difference.
The actual incoming source and predicted boundary velocities are recorded.

Formal training uses the unchanged recursive corpus, the prior decoupled
checkpoint as initialization, 1500 trajectory updates, learning rate 1e-4,
16 maximum segments per batch and 512 reference-frame budget. Both velocity
supervision and seam weights are 1. Validation selects the trajectory model.
The test cohort remains the complete 1659 held-out records (855 demonstrations
and 804 Predictor-accepted segments). Three test initial states (0, 2, 7) run
30 events each, followed by the same-model fixed-one-second counterfactual.
No test trajectories or test metrics select model weights.

Current Newton constraint activation/allocation and shared upward main-face
selection remain contact truth. Full collision geometry is evaluated at the
original native/dense sampling rates. These are fixed-pose Newton observations;
dynamic support and execution are unknown. Original pass criteria are unchanged.

## Run status

The original formal attempt was stopped by user request during initial
validation to optimize training efficiency. It completed expert-only velocity
warmup, selecting step 2750 (normalized validation MAE 0.193861, initial joint
MAE 0.123119 rad/s, terminal joint MAE 0.079217 rad/s). No formal trajectory
weights were selected before the stop. All old artifacts are retained.

The four-update actual Newton smoke passed, with finite gradients and bitwise
unchanged time weights. Unit checks cover physical Hermite velocities,
quaternion signs and identity/near-pi gradients, generated-only history,
expert-only velocity labels, legacy loading, and velocity carry even after
rejected recursive endpoints. The optimized formal restart completed all eight
train/test/rollout stages, selecting trajectory step 1500 by validation.

Run recipe and progress:
`tmp/infiller_boundary_velocity_v1_20261011/run_pipeline.py` and
`tmp/infiller_boundary_velocity_v1_20261011/pipeline_progress.json`.

The efficiency update and throughput comparison are documented in
[direct_infiller_training_efficiency.md](direct_infiller_training_efficiency.md).
The optimized restart uses a new directory, preserving the interrupted run.
Its recipe/progress are under
`tmp/infiller_boundary_velocity_efficient_v1_20261011/`; models are under
`runtime/current/models/generator/direct_infiller_boundary_velocity_efficient_v1_20261011/`.

## Completed results

Balanced validation loss declined from 8.10765 at initialization to 3.33090 at
step 1500. This composite loss includes the new velocity/seam terms; its drop
does not establish trajectory improvement against the parent architecture.
All 1659 held-out records retain exactly their parent's rounded frame counts.
The independent timing tensors are bitwise unchanged, with demonstration frame
MAE 0.148538. Predictor, canonical robot asset and contact rules remain unchanged.

| Test metric | Decoupled-time parent | Boundary-velocity model |
| --- | ---: | ---: |
| Demonstration mean segment terrain peak | 5.115 mm | 14.055 mm |
| Demonstration worst terrain penetration | 57.676 mm | 168.779 mm |
| Demonstration worst self penetration | 9.902 mm | 158.350 mm |
| Demonstration joint trajectory RMSE | 0.121833 rad | 0.150404 rad |
| Demonstration keep coverage | 89.64% | 75.41% |
| Demonstration failed contact establishment | 153/1069 | 25/1069 |
| Demonstration mean joint seam velocity jump | 0.562226 rad/s | 0.448509 rad/s |
| Accepted mean segment terrain peak | 2.091 mm | 2.510 mm |
| Accepted worst terrain penetration | 25.748 mm | 65.478 mm |
| Accepted keep coverage | 98.42% | 91.69% |
| Accepted failed contact establishment | 370/817 | 111/817 |

The three model-duration recursive runs (initial states 0, 2, 7; 30 events
each) pass **30/90** segments versus the parent's **45/90**, using the original
accepted-endpoint plus at-most-5-mm interior-depth criterion. Per-state counts
are 12, 7 and 11, versus 17, 12 and 16. Mean segment peak depth rises from
7.145 to 23.193 mm; worst depth rises from 58.441 to 131.520 mm. Mean joint
seam velocity jump improves from 0.591679 to 0.529560 rad/s and failed contact
establishment falls from 37/100 to 27/100. The fixed-one-second counterfactual
passes 13/90 versus 45/90, with worst depth 158.848 versus 59.347 mm. The new
checkpoint is not promoted: better transition/seam metrics accompany worse
penetration and persistent-contact behavior.

Native counts, exact endpoints, canonical asset metadata, frozen timing and
actual preceding-generated-frame velocity history were verified for all six
rollouts. Complete comparison, paired test rows and the plot are in
`tmp/infiller_boundary_velocity_efficient_v1_20261011/comparison_summary.json`,
`paired_test.csv` and `recursive_boundary_velocity_comparison.png`.

## Validation diagnosis and proposed next experiment

On 64 evenly spaced validation demonstrations, final-step joint trajectory
RMSE with generated history is 0.147449 rad versus the parent's 0.118171 rad;
unknown-history decoding gives 0.127014 rad. The incoming generated finite
difference differs from the real first joint finite difference by 0.905767
rad/s on average. These pose/velocity diagnostics use no test data and do not
redefine contact or select additional model weights.

The current model hard-copies incoming velocity as its analytical initial
velocity whenever history is known. Warmup covers only unknown history, and
the seam objective targets zero native finite-difference change. Real adjacent
demonstrations have identical shared poses but nonzero native finite-difference
changes: mean maximum-joint change is 3.685825 rad/s over 1372 training pairs
and 3.684646 rad/s over 728 validation pairs. These one-sided estimates do not
prove an instantaneous physical discontinuity or contact impulse.

The proposed next experiment conditions the initial-velocity predictor on
generated incoming velocity instead of hard-copying it, trains with generated
known-history contexts, and supervises demonstration seams against their real
native changes. Independent timing stays frozen; generated records retain
physical objectives without invented velocity labels. This architecture change
requires user confirmation and has **not been implemented**. Diagnosis evidence:
`velocity_diagnosis_step1500.json` and `demonstration_seam_reference.json` in the
same experiment directory.
