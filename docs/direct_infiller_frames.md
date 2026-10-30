> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Infiller predicts total segment frames

> Historical experiment: the Infiller implementation and entrypoints were retired
> on 2026-10-23. Data, checkpoints and the original results below remain archived.


This experiment moves segment timing into Infiller. The frozen v1 Predictor
still supplies the next endpoint and contact intent. Infiller receives the two
poses, contact contract, and current scene and predicts total frames, including
both endpoints. Native inference is 50 Hz, so elapsed seconds are `(N-1)/50`.

## Architecture and supervision

`DirectInfiller(predict_frames=True)` adds a 172/128/128/1 frame head to the
existing endpoint-preserving trajectory decoder. Boundary features exclude
external duration. The continuous count is `4 + 50*softplus(raw)`; 50 is a
count normalization scale, not a fixed output count. Inference rounds this
count and decodes exactly that many phases. There is no upper modeling bound.
Predictions outside the explicit 4096-frame resource budget fail with an error
rather than silently clip or fall back to a fixed duration. Four is the minimum
count needed by the existing third-difference trajectory metrics.

Frame targets come exclusively from actual demonstration segment lengths.
The 1470 training and 780 validation demonstrations have lengths from 9 to 148
frames. Generated accepted/recursive records have no expert timing labels;
their old frame counts only define reference phase sampling grids. Historical
fixed 51-frame recursion is never a supervised frame target.

Frame-head warmup uses only training demonstration boundary features and a
SmoothL1 loss on log count. Validation demonstrations select the warmup state.
The trajectory decoder is verified unchanged during warmup. The formal warmup
schedule is 3000 updates with Adam at 0.001, evaluated every 250 updates.

Joint training predicts continuous count without rounding. Its elapsed time
conditions the decoder and all derivative objectives. The expert log-count
loss has weight 10. Generated duration has a positive efficiency cost with
weight 0.02 per second, preventing unbounded slowing. Demo smoothness budgets
use the expert's own duration as their feasibility witness; generated segments
use the original absolute derivative limits, not a cubic with invented timing.

Full demonstrated poses and contact episodes remain supervised on the expert
phase grid. Generated trajectories have no expert intermediate pose/timing.
Complete current Newton/solid geometry is queried on the training phase grids
and their midpoints; these queries are not described as fixed native 50/100 Hz
when predicted time differs. Native inference/evaluation samples at 50 Hz and
uses 500 Hz densified complete geometry checks. All contact observations retain
actual Newton constraint activation/allocation and the shared primary upward
face selector, canonical robot asset, mappings, and same-state sampling audit.

Sampled seam loss decodes both sides at phases exactly one 50 Hz step from the
boundary, using each predicted continuous count. Gradients reach both frame
and trajectory heads. All boundaries preserve the caller's exact poses and
quaternion representations.

## Reproduction and evaluation

The pipeline is `tmp/infiller_frames_v1_20261011/run_pipeline.py`. It initializes
the decoder from the completed recursive refinement model, preserves its
corpus/Predictor provenance, adds the learned head, and trains for 1500 updates
with batch size 16, reference frame budget 512, learning rate 0.0001, seam
weight 1, and full validation every 500 updates. The selected candidate is
saved under `runtime/current/models/generator/direct_infiller_frames_v1_20261011/`.

`--predict-frames` opts training into this architecture. Checkpoints explicitly
record frame-head architecture, native rate, and inclusive count convention;
evaluation and rollout load the recorded architecture automatically. Legacy
checkpoints retain externally supplied durations for reproducibility.

The complete original 1659-record test set is evaluated with predicted frame
counts. Frame error is reported only for demonstrations. Pose RMSE is explicitly
aligned by phase, so it must be read alongside frame/time error. Generated
reference length is recorded for provenance, not treated as timing truth.
The previous fixed-duration test reports used the externally supplied reference
lengths. Those timing conditions differ from this model-predicted test, so the
same-model fixed-duration rollouts are the direct timing comparison. Generated
frame accuracy is unknown and is logged as null. In this first formal run,
generated training batches logged a zero placeholder for `frame_mae`; that
placeholder is not a measured accuracy and is excluded from evaluation.

Three predefined test initial states (0, 2, 7) each recurse for 30 events.
The same selected checkpoint also runs with `--rollout-timing fixed` and one
second per event as an explicit timing counterfactual. Ordinary endpoint
rejections remain in the chain; no reset, projection, or future demonstration
is used. NPZ exports contain per-segment counts, durations, and cumulative
first/last native indices. Dense depth arrays have their own 500 Hz indexing.

Sixty targeted unit tests passed, including legacy loading, expert-only frame
labels, learned duration gradients, generated budget independence from fake
reference time, and variable-length self-fed recursion. The four-update actual
Newton training smoke passed. A three-event rollout with external duration set
to seven seconds produced 51/50/50 frames and correctly exported 149 native
frames, verifying that the model's count controls inference.

## Completed results (2026-10-11)

The 3000-update frame warmup and 1500 joint updates completed normally. Full
validation selected joint step 1500. Its composite validation score was
1.652212, compared with 2.015423 at joint step zero. Warmup alone achieved
0.518 validation frames MAE; this is not the final jointly trained accuracy.
The selected checkpoint SHA256 is
`bf6441a52664750ebe85e8e1bf7ba44e1e7bba58ce36cb4a8fffea27861469e7`.
Predictor remained frozen and unchanged throughout.

All 1659 original test records completed. On the 855 demonstrations, rounded
frame MAE is **8.577 frames / 0.172 s**, with p95/max 22 frames / 0.44 s.
Mean predicted count is 55.704 versus 47.600 expert frames, a positive bias of
8.104 frames. The short demonstrated intervals are generally lengthened while
long intervals remain close to their demonstrated lengths. The 804 generated
test records have no frame ground truth and no claimed frame accuracy.

| Complete test metric | Previous recursive Infiller | Learned-frame Infiller |
| --- | ---: | ---: |
| Demo mean segment terrain peak (mm) | 4.607 | 4.858 |
| Demo worst terrain peak (mm) | 52.388 | 53.798 |
| Generated mean segment terrain peak (mm) | 2.091 | 2.404 |
| Generated worst terrain peak (mm) | 24.300 | 27.517 |
| Demo mean joint velocity seam (rad/s) | 0.673 | 0.210 |
| Generated mean joint velocity seam (rad/s) | 0.276 | 0.161 |
| Demo persistent-contact coverage | 0.9474 | 0.9680 |
| Generated persistent-contact coverage | 0.9938 | 0.9981 |
| Demo contact-establishment failures | 182 / 1069 | 193 / 1069 |
| Generated contact-establishment failures | 471 / 817 | 411 / 817 |
| Demo phase-aligned root RMSE (mm) | 42.883 | 41.901 |
| Demo phase-aligned joint RMSE (rad) | 0.1233 | 0.1215 |
| Demo original smoothness-budget violations | 467 / 855 | 358 / 855 |
| Generated original smoothness-budget violations | 399 / 804 | 322 / 804 |

Previous complete-test timing was supplied from reference records; learned
timing is predicted. These rows therefore compare whole systems under different
timing conditions. They show better test seams and contact retention alongside
slightly worse penetration, rather than a uniform quality improvement.

Three complete self-fed rollouts and all three same-checkpoint fixed-timing
counterfactuals completed. Counts varied from **19 to 149**, equivalent to
0.36–2.96 elapsed seconds at 50 Hz. Learned rollout totals for initial states
0/2/7 were respectively 31.06/33.12/31.40 seconds and 1554/1657/1571 native
frames. Every exported cumulative boundary index and native/dense array length
passed consistency checks. Shared pose boundaries have zero jumps.

| Three 30-event rollouts | Previous model, fixed timing | New model, fixed timing | New model, learned timing |
| --- | ---: | ---: | ---: |
| Endpoint accepted and interior depth <= 5 mm | 46 / 90 | 48 / 90 | 50 / 90 |
| Accepted Predictor endpoints | 55 / 90 | 55 / 90 | 55 / 90 |
| Worst terrain/self depth (mm) | 71.427 | 46.524 | 51.352 |
| Mean joint velocity seam (rad/s) | 0.1640 | 0.1568 | 0.1948 |
| Contact-transition failures | 33 / 100 | 32 / 100 | 33 / 100 |

Learning timing added two passing segments against the same-model fixed
counterfactual, but increased its worst penetration and recursive velocity
seams. Thus the observed improvement over the previous model cannot be
attributed wholly to timing. Predictor endpoint acceptance did not change.
Frame prediction works end to end, while timing accuracy, penetration, and
recursive seams still trade off. Keep this checkpoint as an experimental
candidate; original corpora/checkpoints and the baseline remain available.

This is a same-source height-held-out test using fixed-q actual Newton queries.
It does not establish independent-source generalization or dynamic execution
success. Test records and these rollout results were not used for training or
checkpoint selection.

Artifacts:

- `tmp/infiller_frames_v1_20261011/comparison_summary.json`
- `tmp/infiller_frames_v1_20261011/paired_test.csv`
- `tmp/infiller_frames_v1_20261011/expert_frame_prediction.png`
- `tmp/infiller_frames_v1_20261011/recursive_frames_comparison.png`
- `tmp/infiller_frames_model_seed0_20261011/rollout/predictor_infiller_recursive30.npz`

The trajectory viewer on port 8094 was verified with this last artifact:
1554 frames at 50 Hz, read-only, with the matching terrain.
