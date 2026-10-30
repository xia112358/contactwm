> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Independently supervised timing, time-conditioned trajectories

> Historical experiment: the Infiller implementation and entrypoints were retired
> on 2026-10-23. Data, checkpoints and the original results below remain archived.


This experiment addresses systematic lengthening in the first learned-frame
Infiller. On the same 780 validation demonstrations, its independent frame
warmup had 0.518 frames MAE, rising to 8.576 after joint time/trajectory training.
On 855 test demonstrations, the 9–29-frame group had 12.531 frames MAE and all
456 records were lengthened. The >=60-frame group had 2.110 frames MAE.

The frame module and trajectory module already have separate parameters. The
change separates their optimization as well. `--frame-training decoupled` is
the default for new learned-frame training; recorded old frame checkpoints
without this field load with `joint` semantics for reproducibility.

1. Train the frame head only on training demonstration lengths, using SmoothL1
   in absolute frames. Validation demonstration continuous-frame MAE selects
   its weights. Generated records have no timing targets or efficiency loss.
2. Save `timing_head_selected.pt` with canonical robot asset and corpus/Predictor
   provenance. Freeze the selected frame parameters for trajectory training.
3. Train the trajectory decoder conditioned on this predicted time, keeping
   exact caller endpoints and the existing cubic/residual parameterization.
   Detach predicted durations and one-native-step seam phase locations.
   Physics, pose, smoothness, and seam losses optimize only trajectory weights.
4. After every trajectory update assert that every selected frame parameter
   is bitwise unchanged. Report frame accuracy separately; trajectory model
   selection uses trajectory validation objectives without a timing penalty.

The decoder continues to respond to its duration input; detaching a gradient
does not remove the condition. Full current Newton contact semantics, upward
main-face selection, actual collision assets, immutable boundaries, and
original evaluation thresholds remain in effect. No generated timing labels
or future demonstration frames enter inference.

The formal comparison initializes from the same recursive Infiller as the
coupled experiment, uses its unchanged corpus, 3000 frame-head updates, then
1500 trajectory updates, and evaluates the complete original test set and
three 30-event self-fed rollouts. Same-model fixed-one-second rollouts isolate
timing effects. Timing accuracy and physical trajectory quality are both
reported; timing success alone does not establish a successful trajectory.

## Independent timing results

The selected frame head is warmup step 3000: validation rounded-frame MAE is
0.236 over 780 demonstrations, continuous-frame MAE 0.318, and p95 rounded error
1 frame. The selected weights are frozen throughout subsequent trajectory
training.

An independent CPU timing check on all 855 test demonstrations gives rounded
MAE **0.149 frames**, bias -0.074, p95 1 frame, and maximum 6 frames. By expert
length, the 456 short intervals (9–29 frames) have 0.088 MAE, 171 medium intervals
(30–59) have 0.146 MAE, and 228 long intervals have 0.272 MAE. This fixes the
previous systematic short-interval lengthening. These test metrics are not
used to select either timing or trajectory weights.

Sixty-four relevant unit tests and a four-update actual Newton training smoke
passed. The smoke verifies bitwise unchanged timing weights during trajectory
updates. Gradient tests cover direct motion and both sides of seam phase
sampling, independently of parameter freezing, and verify that trajectories
still respond numerically to their time condition.

## Completed trajectory comparison

All 1500 trajectory updates completed normally in 3760 seconds. Validation
selected trajectory step **500**, with trajectory-only balanced loss 1.803137
versus 1.922664 at initialization. Step 1000 scored 1.895351 and step 1500
1.811732, so neither replaced step 500. These losses exclude time supervision
and are not directly comparable to the previous joint composite criterion.

The selected checkpoint is
`runtime/current/models/generator/direct_infiller_frames_decoupled_v2_20261011/best.pt`,
SHA256 `f751a82e1e34b9e0dc43821af014f4949b7c1d3a699fdb76352985bc19fc0a1d`.
Its time weights are bitwise identical to `timing_head_selected.pt`; Predictor
is unchanged. Complete native-output test evaluation confirms the same
0.148538 rounded-frame MAE reported by the independent timing check.

| Complete test metric | Coupled timing model | Decoupled timing model |
| --- | ---: | ---: |
| Demo frame MAE | 8.577 | 0.149 |
| Demo mean segment terrain peak (mm) | 4.858 | 5.115 |
| Demo worst terrain peak (mm) | 53.798 | 57.676 |
| Demo worst self penetration (mm) | 0.000 | 9.902 |
| Generated mean segment terrain peak (mm) | 2.404 | 2.091 |
| Generated worst terrain peak (mm) | 27.517 | 25.748 |
| Demo mean joint velocity seam (rad/s) | 0.210 | 0.562 |
| Generated mean joint velocity seam (rad/s) | 0.161 | 0.242 |
| Demo persistent-contact coverage | 0.9680 | 0.8964 |
| Generated persistent-contact coverage | 0.9981 | 0.9842 |
| Demo contact-establishment failures | 193 / 1069 | 153 / 1069 |
| Generated contact-establishment failures | 411 / 817 | 370 / 817 |
| Demo phase-aligned root RMSE (mm) | 41.901 | 42.391 |
| Demo phase-aligned joint RMSE (rad) | 0.1215 | 0.1218 |
| Demo original smoothness-budget violations | 358 / 855 | 507 / 855 |
| Generated original smoothness-budget violations | 322 / 804 | 472 / 804 |

Frame accuracy and contact-establishment improved, while seams, persistent
contact, and demonstration penetration regressed. Timing decoupling and the
absolute-frame loss changed together, so this experiment does not separately
attribute the full timing gain to either change. Neither condition guarantees
that an arbitrary Predictor endpoint can be reached safely within predicted
time.

| Three 30-event rollouts | Coupled model, learned time | Decoupled model, learned time | Decoupled model, fixed time |
| --- | ---: | ---: | ---: |
| Accepted endpoint and interior depth <=5 mm | 50 / 90 | 45 / 90 | 45 / 90 |
| Accepted Predictor endpoints | 55 / 90 | 55 / 90 | 55 / 90 |
| Worst terrain/self depth (mm) | 51.352 | 58.441 | 59.347 |
| Mean joint velocity seam (rad/s) | 0.1948 | 0.5917 | 0.1535 |
| Mean root velocity seam (m/s) | 0.0334 | 0.1011 | 0.0302 |
| Contact-transition failures | 33 / 100 | 37 / 100 | 38 / 100 |

Learned durations for initial states 0/2/7 are 24.44/26.00/25.40 seconds, with
1223/1301/1271 native frames. Segment counts range from 9 to 149. All six new
rollout native/dense lengths and cumulative boundary indices passed checks.
Pose boundaries remain exact, without projection or resets on rejected events.

The same new decoder has substantially larger velocity seams at learned timing
than at fixed one-second timing. The retained cubic/residual parameterization
forces zero analytical velocity at both ends of every segment. Training expert
clips are generally moving at their cuts: the median per-segment maximum joint
speed estimated by a 50 Hz one-sided difference is 2.923 rad/s at the start and
3.113 rad/s at the end (1470 demonstrations). This is a structural mismatch
that motivates a boundary-velocity-conditioned trajectory comparison; it is
not proof that this one issue explains every observed regression. No such
parameterization change was included in this decoupling experiment.

Keep this as an experimental candidate. Timing prediction is accurate on this
same-source height-held-out test; overall trajectory quality has not surpassed
the coupled baseline. Generated events have no timing ground truth. Actual
Newton observations are fixed-q geometry/contact queries, and dynamic execution
success remains unknown. Test data and rollout outcomes were not used to train
or select weights. Existing corpora, checkpoints, and baseline remain available.

Artifacts:

- `tmp/infiller_frames_decoupled_v2_20261011/comparison_summary.json`
- `tmp/infiller_frames_decoupled_v2_20261011/paired_test.csv`
- `tmp/infiller_frames_decoupled_v2_20261011/expert_frame_prediction.png`
- `tmp/infiller_frames_decoupled_v2_20261011/recursive_frames_comparison.png`
- `tmp/infiller_frames_decoupled_model_seed0_20261011/rollout/predictor_infiller_recursive30.npz`
