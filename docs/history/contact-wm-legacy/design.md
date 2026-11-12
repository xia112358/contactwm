> Historical implementation, archived 2026-11-08. These commands are retired.
> Current training: `supermimic.contact_train`; see [current workflow](../../current-workflow.md).

# Contact-conditioned control through WM-BPTT

## Repository findings and minimal design

The release-tag worktree already contains a direct six-contact PHP policy:
567 inputs, MLP 512/256/128, 29 actions. Contact order is feet, hands, knees.
It consumes a 24-dimensional XYZ/mask target, high-level goal, 63 terrain
samples, measured contact/known flags, and five-step proprioception. The
observation manager packs alphabetical terms with **per-term history**, not
frame-major history. The student has no future-motion window. This is the
smallest useful prior; a new Velocity Encoder, shared latent, motion decoder
or downstream tracker is unnecessary.

Existing WM-BPTT lives in the sibling `supermimic_vibed` project. Its learned
world model predicts local state increments and integrates position/rotation/
velocity/joints. The force6 variant also predicts six heading-frame force
vectors. Its existing policy consumes future motion frames; that policy is
not reused here. Its recursive dynamics/gradient pattern is reusable.

Use the explicit low-climb force6 milestone
`supermimic_vibed/logs/climb_low_force6_resume1000/ckpt_006000.pt` as the WM
initialization. `contact_wm.supermimic.SuperMimicForceWM` strictly loads its
272→70 dynamics, both running normalizers and 272→18 force head, preserving
source scaling and manifold integration. It translates xyzw state quaternions
to the source's wxyz helpers. It does not import or instantiate a source policy.
A six-logit native activation head is new and initially uncalibrated. Source
force measurements and current eligible native contacts must be reconciled by
calibration, not treated as automatically identical labels.

`ContactPolicy` extends the PHP first layer with zero-initialized columns for
six surface normals, six maintained-support flags and arrival time. Initial
actions equal the clipped prior action. All existing layers remain trainable;
no decoder is forcibly frozen. `load_php_policy` validates the canonical asset,
architecture and observation layout. Motion regularization is a soft penalty
against prior actions on current imagined observations, not a hard reference
trajectory constraint.

## Implemented interfaces

- `core.py`: direct policy initialization; diagnostic residual WM (not the chosen
  production initialization); native state/contact/force regression; recurrent
  BPTT; contact/load/hold/slip/support and action/prior losses; external simulator
  directional-gradient comparison.
- `supermimic.py`: strict inherited force6 WM. It requires explicit verified
  action/physics contracts and an environment context provider. The context
  includes 33 normalized physical parameters, 121 relative source-layout
  heights, and previous world forces. PHP's 63 heights cannot simply replace
  those 121 inputs. Force predictions feed the next step's context without
  detaching. The source weights and running statistics are reused exactly.
- `observations.py`: differentiable current-state PHP observation and history
  reconstruction; target XYZ/normals are transformed into base yaw coordinates.
  A height reader is mandatory and must query available perception at imagined
  poses. There is no implicit ground-truth filling of unknown observations.
- `geometry.py`: differentiable canonical G1 robot-side witness points; no free
  learned XYZ contact-point head. Local witnesses must be calibrated from the
  native geometry correspondence for the selected transition.
- `validation.py`: extract onset candidates with known preceding absence and
  sustained activation/support; separately check real native activation,
  force, target placement, slip, retained support and failure over a hold window.
- `collect.py`: frozen native PHP rollout capture with raw solver evidence,
  pre/post states, action, observation, label and termination timestamps.
- `fit.py`: small one-step WM/data integration diagnostic only; its report
  explicitly disables policy optimization. The inherited SuperMimic WM is the
  intended next-stage model, not this from-scratch diagnostic.

The `unroll` interface requires observation, environment-context, geometry and
whole-body stability callbacks. Context receives previous prediction so force
feedback remains differentiable. An approach of “reuse true future observations”
or “detach after every WM step” is not a valid adapter. During BPTT only WM
parameters are frozen; state/action derivatives remain connected. Do not update
WM weights before consuming a pending policy backward graph.

Contact loss is evaluated over arrival **and a hold interval** and separately
retains pre-existing supports throughout. It combines geometry, native activation
surrogate, normal load, tangential velocity and regularization. Stability and
force predictions remain learned surrogates; the actual success function accepts
independent native observations. Unknown/reset windows cannot create negative
labels or successful support. Geometric closeness alone cannot create contact.

## Validation performed, 2026-10-26

Fourteen unit tests pass: exact prior preservation/learnability, early-action BPTT
and WM freezing, native activation/load/hold checks, onset/support extraction,
unknown force/reset masks, directional-gradient reporting, manager history layout
and differentiable canonical FK, analytic world-height query gradients, and
material velocity independent of contact-centroid movement, and per-trajectory
WM selection including rejection of nonfinite errors, runtime goal replacement,
proxy-safe reference-independent failure checking, and separate fixed-point vs
sustained-support contracts, plus detection of a policy ignoring target changes.
Analytic dynamics in a unit test are explicitly
not a robot model or evidence of physical success.

The actual inherited milestone loads 26 WM tensors (about 9.25M trainable source
parameters including the force head). Numerical state and force outputs match
the original SuperMimic implementation; action derivatives are finite and
nonzero. See `tmp/contact-wm/supermimic-transfer.json` and its reproduction script.
The corresponding static control audit matches joint order, default pose, effort
scaling, PD gains, effort limits, armature and 20-ms control period. Runtime
actuator/solver equivalence and gradient validity are not yet certified.

Frozen PHP motion 0 produced 241 native steps and then terminated. Two native
onset candidates were found: left foot at sample 153, left hand at 210, each
with a pre-existing support and at least ten known active samples. Neither
passes the provisional stable-placement/no-slip check: the foot candidate fails
slip, the hand candidate fails placement and slip. The hand's mean placement
error was about 14 cm. Do not call these a verified stable-contact baseline.
The slip diagnostic uses a fixed body witness calibrated at onset and retains
the recorded last-solve/post-integration timing distinction; these limitations
must be considered before acceptance thresholds are finalized.

Data and reports: `tmp/contact-wm/native-baseline-v1/`. The scratch diagnostic
WM used 167 training and 71 held-out transitions for 100 optimizer steps. Its
training loss fell, but switching errors exceeded steady-state errors and no
simulator gradient agreement was measured. This checks wiring, not learned
physical accuracy, and is superseded as initialization by the inherited WM.

## Native multi-motion calibration

`tmp/contact-wm/native-baselines-8-v2/` records motions 0–7 with the frozen
released policy. Five completed and three terminated. This is episode completion,
not stable target-contact success: none of 58 extracted events passed all
placement (5 cm), slip (5 cm/s), activation, load and retained-support checks.
The updated audit uses each current solver point's material velocity under FK,
avoiding false slip from a changing contact centroid. Last-solve versus
post-integration timing remains a limitation. Motion 5's left-hand event near
sample 286 passes the other checks but has about 9.8 cm maximum placement error.

The inherited SuperMimic WM was calibrated for 200 updates on motions 1–6,
holding out entire motions 0 and 7. Source normalizers were frozen, contact
switches oversampled, and the new activation head trained against native labels.
The 1,688 held-out valid samples improved in joint RMSE from 0.00769 to
0.00424 rad and force RMSE from 34.5 to 17.7 N. On 146 switch samples, force
RMSE improved from 73.5 to 41.7 N. Activation recall/specificity reached
0.820/0.715 overall. Checkpoint and report are in
`tmp/contact-wm/supermimic-calibrated-v1/`. These are one-step errors; they do
not establish useful BPTT gradients. Policy weights were not updated during this calibration stage.

Calibration currently uses nominal physics context and oracle terrain for
diagnostics. Source implicit PD and PHP IdealPD differ despite matching static
control parameters, so native adaptation and independent validation are needed.
`height_context.py` reads a recorded world raster differentiably at imagined
poses. `probe.py` compares recurrent WM Jacobians against paired native action
pulses, including recursively predicted forces rather than future true forces.

The initial paired experiment exposed reset-time motion reassignment, now fixed
by assigning within command buffer initialization and asserting the final IDs.
After that correction, all nine copies used motion 5, but their states diverged
before the requested sample-280 pulse. The strict matching check rejected the
experiment (`paired-probe-v3/paired-start.json`); it is not gradient evidence.

A separate pulse from the normal reset state had exactly equal initial states.
Nine native steps with ±0.02 action pulses in four joint directions yielded
position-Jacobian cosine 0.984 for inherited WM and 0.979 after calibration;
relative errors were 21.9% and 22.9%. Thus one-step calibration did not improve
this local sensitivity metric. This probes the left wrist origin during free
motion, not successful contact. The hand had zero native force throughout the
terminal comparison, so force cosine/relative error are undefined, reported as
null. There is not yet an unperturbed replica noise-floor test. Full report:
`tmp/contact-wm/paired-probe-initial-v1/gradient-report-v2.json`.

## Iterative native data aggregation

`collect.py --action-noise-std ... --noise-seed ...` adds reproducible Gaussian
action exploration to the frozen prior, clips to its action range and records
the executed action with native state/force labels. Episode resets are masked.
`calibrate.py --resume ... --additional-recordings ...` restores WM and Adam
state, retains old data and keeps motions 0 and 7 entirely outside training
across all collection rounds. Half the training batch samples contact switches;
by default one quarter samples the top 20% prediction-error cases, refreshed
every 50 updates. Priority scoring uses only the training split. Normalizers
remain frozen. Reports include each held-out recording separately.

`learn_loop.py` runs a finite number of collection/calibration rounds, writes
process identities and logs, saves every candidate, and selects the next WM
only if no held-out trajectory's overall or switching joint/force error grows
by more than 10%. This is a prediction gate, not BPTT gradient certification.
No policy is trained by this loop, and it is not a permanent background service.

Completed first aggregation: `tmp/contact-wm/exploration-round1/` (noise std
0.03) plus original data; resume initial calibration for 400 updates, ending at
iteration 600 in `supermimic-online-round1/model.pt`. On 3,376 held-out samples,
switch force RMSE improved 38.53→35.05 N and switch activation recall/specificity
0.803/0.704→0.857/0.821. All per-recording prediction gates passed. However,
the independent free-motion position Jacobian cosine decreased to 0.967 and
relative error increased to 27.9%; one-step improvement is not gradient proof.

The next round (`learning-loop-v1/round_00/`, noise std 0.05, seed 37) was
rejected: original motion 7 force RMSE increased 19.02→23.48 N and the new
motion 7 switch force RMSE increased 59.24→69.52 N. Its data and candidate
checkpoint remain available; selection retained round 1. A conservative retry
from round 1, 100 updates at 0.25 times the saved learning rate and 0.125 hard
replay fraction, is saved separately in `online-round2-retry/`. On the same
4,505 validation samples its switch force error improved 40.43→39.39 N,
but original motion 7's overall force error still increased 19.02→21.18 N,
so the same gate rejected this retry. The selected model remains round 1.
These repeated motions are validation data, not a new independent test set.

An additional frozen-policy native collection on motions 8–15 is retained in
`tmp/contact-wm/native-baselines-8to15-v1/`. Its 79 audited contact events still
yield no event satisfying every placement/load/slip/support check. This batch
has not been used to fit the WM. The search for a verified target-execution
baseline and matched contact-switch sensitivity measurements remains open.

Example bounded loop (use the pinned runtime launcher and configured GPU):

```sh
python -m holosoma.contact_wm.learn_loop \
  --policy-checkpoint PATH_TO_PHP_07000 \
  --wm-checkpoint tmp/contact-wm/supermimic-online-round1/model.pt \
  --recording tmp/contact-wm/native-baselines-8-v2 \
  --additional-recordings tmp/contact-wm/exploration-round1 \
  --output-dir tmp/contact-wm/NEW_UNIQUE_RUN --rounds 2
```

## Direct policy training and native execution

`train_policy.py` now performs actual policy updates through recurrent calibrated
WM dynamics. The first run uses motion 5's state at sample 266, a left-hand
target due at sample 286, retained right-hand/left-knee support, and ten hold
steps: a 30-step (0.6 s) optimization horizon. The target remains the commanded
target from the recording; it is not replaced by the achieved contact point.
Robot-side witnesses are calibrated from native contact labels. The policy
receives current proprioception/history, current contact evidence, terrain and
explicit contact conditions; no future reference sequence enters the rollout.

The first successful training run, `tmp/contact-wm/policy-contact-v3/`, completed
100 Adam updates of all policy layers with WM weights frozen. Imagined total
loss decreased 8.969→0.602 and geometry loss 8.597→0.225. It saves
`policy_0000.pt`, `policy_0100.pt`, optimizer/RNG/provenance and corresponding
imagined state/action trajectories. These losses are model predictions and do
not establish native success. Earlier v1/v2 attempts failed before updates due
to insufficient terrain coverage and missing legacy origin metadata; their
logs remain preserved.

`runtime.ContactExecutor` loads the direct policy without a WM. It replaces the
reference command with the specified contact goal, anchors maintained supports
to current native contact points, and outputs 29 joint actions per step. The
collector's `--contact-policy` option runs this controller from the requested
initial time after a frozen-prior warmup, recording the executed state/action
trajectory. Reference tracking termination is replaced for this experiment by
an explicit gross-fall check (nonfinite root, up-vector dot < 0.2, or world root
z < 0.15); these limits are not a general stability certificate.
`evaluate_policy.py` checks actual contact activation, loading, placement, slip,
retained supports and failure over the requested arrival/hold interval.

Native execution results for v3: the unoptimized conditional policy established
contact and maintained load, but failed placement (maximum 8.66 cm) and slip
(maximum 7.27 cm/s). The 100-update policy terminated after 16 executed task
steps, before arrival/hold completed. Initial states varied between native runs,
so this is an independent execution comparison, not an exactly paired causal
estimate. Imagined loss improvement did not transfer to physical success.

The failed execution was added to WM replay (`policy-feedback-wm-v1/`). Held-out
switch force error improved 37.99→34.72 N, but one held-out switch joint metric
regressed beyond the predeclared 10% gate; this candidate was retained without
promotion. Policy v4 therefore used the previously selected WM, replayed three
actual initial states, increased the soft motion penalty to 5 and strengthened
the tilt penalty. Its 50-step candidate still terminated before arrival in
both the original task and a target shifted 2 cm sideways on the same platform.
The current proprioception reconstruction agrees with a recorded manager
observation to 5.96e-7 maximum absolute error at the training start.

All candidate policies and actual trajectories remain saved. The latest
bounded experiment scales the v4 parameter update to 0.25, clears stale Adam
moments, and specifies an arrival deadline 0.1 s later *before* independent
execution. Original and 5 cm forward goals are tested separately. This is a
different timed task, not a retrospective change to the failed earlier trials.

Under the fixed-point support contract, the quarter-update policy completed
the timed rollout and passed native activation, load and gross-fall checks.
The forward-shifted goal's left hand was within 3.86 cm, with maximum required
material slip 4.02 cm/s, but the maintained knee contact centroid moved 7.88 cm,
so the fixed-point task failed. Preserving a supporting contact does not
necessarily pin the centroid of a rolling knee contact to its original XYZ.

The collector therefore also supports an explicitly selected, predeclared
`--support-contract sustained_contact`: the new target still requires placement
within 5 cm; existing supports require continuous known activation, load and
material slip ≤5 cm/s, while centroid displacement is reported separately.
The default remains `fixed_points` and old reports are not re-scored as success.
The first independent sustained-support trial reached the new target within
4.12 cm but failed the slip bound (5.24 cm/s). Repeated trials and an unoptimized
policy comparison are retained separately; this remains a task reliability
question, not evidence from imagined losses.

The second independent sustained-support trial passed all native checks: target
position error 3.31 cm, maximum material slip 4.60 cm/s, continuous activation
and loading, no gross fall. The unoptimized policy on the same declared task
failed slip (6.44 cm/s). Native initial states are not identical, and the trained
policy passed only 1 of 2 independent trials, so neither robust execution nor
a statistically established improvement is claimed. The original fixed-point
support task remains unsolved.

The successful task is exported in `tmp/contact-wm/contact-task-demo-v1/`:
`policy.pt` has the +5 cm target baked into its explicit goal;
`executed_trajectory.npz` contains the actual 35 actions, 36 states, native
contacts/forces and initial observation for the 0.7 s task, excluding warmup;
`report.json` includes both trials and the unoptimized comparison. The task
deadline is 0.52 s and the hold spans ten 20-ms samples. Inference requires no
WM or future reference sequence. Terrain remains simulator oracle data in
this experiment; this is not a verified real-robot deployment.

## Matched-state multi-target training

The requested objective is now explicit in `train_policy.py --multi-target`:
clone one recorded initial state, history, contact evidence and prior action
across a 3×3 grid of target offsets (±4 cm in platform X/Y). Check sampled
targets lie on the same recorded platform height. Only the new hand target
changes; no edited reference trajectory is supplied. Retained support, current
body state and high-level hint are identical across targets. The current run
centers this grid at the original command plus 5 cm in X, with the declared
0.52 s arrival deadline and ten-sample hold.

`target_response.py` adds a loss on the difference between achieved end positions
and the center-target end position versus the corresponding commanded offsets.
It cannot be satisfied merely by producing different actions or the same mean
trajectory for every goal. Absolute placement, load/activation, slip, stability
and soft motion penalties remain. Eight inner ±2 cm goals are excluded from
training and evaluated separately; the shared center is excluded from their
reported validation metrics. The reported 2×2 target-to-endpoint gain should
approach the identity matrix; near-zero gain means target ignoring.

The first 150-update run (`policy-multitarget-v1/`) reduced absolute imagined
error but left target gain near zero, confirming that the previous success
criterion was insufficient. The active second run (`policy-multitarget-v2/`)
has 400 updates, response weight 20, base Adam learning rate 1e-6 and a 250×
actual update multiplier on the input columns carrying contact targets. All
layers remain trainable. Scaling Adam's input gradient alone would largely be
cancelled by its second moment; this code scales the parameter update itself.
Checkpoints, optimizer/RNG and response diagnostics save every 100 updates.

`collect.py --task-initial-recording ...` now initializes a single native world
from the same recorded physical state, followed by five identical replayed
actions before enabling goal-conditioned control. This avoids 266 steps of
prefix divergence between target trials. A normal native warmup test completed
40 steps; its task-start root closely matches the training state. Native
constraint truth is re-measured after warmup, not imported from the recording.

`evaluate_targets.py` is a finite detached evaluator waiting for the second
training process's completion report. It runs prior and trained policies on
five targets each (center and unseen ±2 cm axis shifts), saves every trajectory
and contact result, and computes native target-response gain only as a valid
matched-state comparison when observed initial states, contact flags and forces
meet the declared tolerances. Terminated or unestablished-contact runs cannot
be counted as a complete response sweep. Output destination:
`tmp/contact-wm/multitarget-native-evaluation-v2/`. Neither this scheduled check
nor imagined target gain is a claim that real target-conditioned control has
already been learned.

## Alternating long run (SuperMimic training pattern)

The sibling `supermimic/train.py` collects current-policy transitions into replay,
updates WM on contiguous multi-step windows, then freezes WM for policy BPTT.
`online_train.py` now implements this pattern in finite blocks for the existing
direct contact policy. No SuperMimic velocity encoder or motion-goal policy is
imported. Each round collects nine training-target episodes with the current
policy, updates WM for 200 steps, then updates the multi-target policy for 200
steps. Every fifth round and the final round run the independent target sweep.

The active run is `tmp/contact-wm/online-contact-long-v1/`, 20 rounds on GPU 1,
starting from policy update 400 and aiming at update 4400. Each native episode
uses the recorded initial-state reset and five fixed warmup actions. Gaussian
action noise std 0.02 is applied **after** current-policy inference; a recorded
first episode was checked against replayed policy outputs and had residual std
0.01888 across 35 task actions. No validation target is added to the fit replay.

WM fitting retains old replay, samples 50% of each batch from the new round,
and combines one-step supervision with a five-step recurrent loss. Windows are
built within a single recording and excluded if any transition is reset/invalid.
State and previous-force predictions recur without detaching. Terrain heights
are recorded exogenous contexts in this WM fit (as in the source's replayed xi
windows); these are conditional prediction diagnostics, not a self-contained
sensor forecast. Held-out final-window state error is reported alongside
one-step contact/force errors. A two-update GPU smoke run completed successfully.

Candidate WMs are retained; the next policy block uses a candidate only when
the per-trajectory one-step prediction checks and multi-step joint error stay
within the predeclared 20% regression allowance. Rejection is recorded in
`status.json`, rather than silently presenting the candidate as the active WM.
Policy checkpoints retain Adam state and RNG; all round data, candidate WMs and
policy milestones remain available. The finite coordinator records child PIDs,
process start identities, logs and phase changes and stops visibly on failure.

The first actual round collected all nine goals and 351 valid new transitions,
completed 200 WM updates and passed the selection gate. Five-step held-out joint
RMSE decreased 0.01734→0.01617 rad and final position error 6.36→5.43 mm. Policy
updates then resumed from iteration 401 using that newly fitted WM. These are
model-fit diagnostics; native goal-response success remains to be measured.

The existing dashboard now follows the long-run collection, WM and policy
phases, shows the selected WM decision and five-step prediction errors, and
retains periodic native evaluation results. This run tests whether continued
on-policy model adaptation enables genuine target response; it is not yet a
claim of successful contact generalization.

## Remaining experimental work

First select and independently verify a dynamically stable single transition
under the chosen support/placement/slip contract. Existing fixed-q labels locate
candidates but cannot supply dynamic success. Adapt the inherited WM's 121-sample
perception context, source force semantics, and runtime physics to the PHP task;
fit activation and recalibrate dynamics using real transitions and action
perturbations around contact switches. Test paired positive/negative action
perturbations from matched initial conditions, comparing actual simulator loss
changes with WM directional derivatives before policy optimization.

Then compare the original contact-conditioned prior, motion-prior distillation,
and prior + WM-BPTT on identical starts. Target perturbations must be attached
to an actual reachable surface. Moving the commanded height while leaving no
surface there tests an impossible task. Vertical/horizontal terrain edits require
new sensor observations and independent native execution evidence; no edited
reference motion is supplied to the policy. Report dynamic activation, sustained
load, placement, slip, retained supports and falls separately from WM loss.

No generalization or robust task-success claim is supported by the current runs.
No existing training process was stopped or modified. Prior checkpoints remain
unchanged. WM remains training-only; the extended direct policy produces actions
without a runtime WM, infiller or extra tracker.

### Persistent 4096-environment restart (2026-10-26)

`holosoma.contact_wm.stream_train` replaces the coarse subprocess-per-trajectory
loop. It initializes the direct policy from Stage 1 iteration 7000 and the WM
from original SuperMimic iteration 6000, with fresh optimizers. One native Newton
process owns 4096 environments. After 64 warmup steps, each iteration collects
alternately 2 and 3 native steps, performs 10 recurrent WM updates (2048 samples,
8 steps, gradient accumulation in chunks of 256), and one policy update (8 groups
of 9 counterfactual goals, 72 imagined trajectories total). Native environment
count and policy BPTT batch are intentionally reported separately. The bounded
replay retains 256 steps per environment and rejects reset-crossing or unknown
contact windows. Early approach states for policy BPTT come from this live replay.

Only the original trajectory defines task targets/witnesses and the five-step
reset warmup. No modified reference motion is supplied. Terrain remains an oracle
atlas in this experiment, not deployed perception. The inherited WM is adapted to
Newton online; this does not establish physical-gradient validity. Native contact
rates and geometric error are diagnostics, not a stable-contact success claim.

Checkpoints contain policy/WM/optimizers/normalizers/RNG/configuration/progress,
not solver snapshots. `--resume` resets native physics and warms replay again.
The new run writes `tmp/contact-wm/stream-4096-v1`; the requested deletion of old
contact-training results is recorded in `deleted-old-results.json`. Native task
baselines, terrain atlas, and both pretrained source checkpoints are retained.

Nine of the 4096 environments are reserved for frozen-policy, noise-free native
execution. Their transitions are excluded from replay. Each 40-step evaluation
contains five reset warmup steps and a full 35-step contact task. Evaluation
requires actual allocated native contacts, normal load, new-contact placement
within 5 cm, material-point slip at most 5 cm/s, maintained supports, no failure,
and ten hold steps. Reset/failure invalidates the episode. The monitor exports
canonical G1 link poses from actual integrated states, raw native contact/force
observations, a target marker, and the terrain atlas. The website shows these
executed trajectories separately from imagined WM trajectories; iteration zero
is retained as the native baseline. Solver contacts and post-integration poses
retain their documented timing difference.

### Batched compiled revision (v2)

The v1 stream was stopped at iteration 30. Its ~19 s iteration time came with
serial WM gradient accumulation and serial groups of nine policy trajectories.
V2 uses one 2048-sample, eight-step compiled WM graph per optimizer update (ten
updates per iteration). Policy BPTT uses 455 matched-start groups × nine targets
= 4095 trajectories in one batch and one backward pass. Tensor deadlines and
masks preserve each start's remaining horizon; loss normalization weights each
start equally. FK and contact objectives are evaluated over the stacked batch,
not through Python FK calls at every step. Pure tensor step/loss functions use
`torch.compile(fullgraph=True)`; compilation failure is not silently bypassed.
TF32 matches the source SuperMimic configuration. Stage wall times synchronize at
collection, WM, and policy boundaries and distinguish startup compilation from
steady-state timings. CPU parity tests cover mixed deadlines and native WM loss;
the GPU benchmark compares original and batched policy losses/gradients.

Native evaluation v1 also had a variable-overwrite bug: diagnostic FK witnesses
replaced aggregated solver contact points before monitor export. V2 passes the
actual selected solver points to evaluation and versions exports as
`solver_points_v2`; the browser rejects old unversioned contact statistics. Old
numeric success/slip/placement summaries must not be treated as valid evidence.

Measured validation on physical GPU 1 (RTX PRO 6000 Blackwell, 4096 native envs):
40 post-warmup iterations averaged 0.702 s total: 0.230 s collection/monitoring,
0.213 s for ten WM updates, and 0.259 s for one 4095-trajectory policy update.
The isolated compiled learner averaged ~0.50 s. These exclude first compilation
and periodic checkpoint writes. The old v1 loop averaged ~18.8 s. Final training
starts afresh from Stage 1/SuperMimic source checkpoints after this validation;
validation artifacts are under `tmp/contact-wm/stream-4096-v2-validation`.
Post-reset age-zero transitions are excluded because their previous solve does
not correspond to the freshly reset pose. Missing native contact points are not
reported as geometric errors against a zero-coordinate placeholder.

### Historical panorama input

`stream_train --panorama` replaces the actor's 63 oracle height samples with
features encoded from one 128 × 64 panoramic depth channel. Age and confidence
belong to the upstream fusion stage and are not policy input channels. Valid
depth is divided by the 2.5 m reading range; unknown pixels are zero, distinct
from a valid far return. The reading origin is root position plus 0.73 m along
world Z, with yaw only. Its relative position and yaw accompany a held scan.
Proprioception, contact evidence and target conditions remain policy inputs.

Each native environment has an isolated 3 cm surface-voxel history. The synthetic
torso-mounted LiDAR supplies 20,000 rays at 10 Hz. Task resets seed history only
from sensor scans along the recorded past prefix (frames before 261); future
motion and full terrain occupancy never populate that history. Reset seeding is
an experimental initialization, not a claim about an uninitialized real robot.
The emulator currently uses exact odometry and terrain-only instantaneous scans;
self-occlusion, scan distortion and odometry noise remain limitations.

Sensor frames are stored once in a separate replay ring; policy samples retrieve
the matching historical frame, never a newer frame. During a 35-step imagined
rollout, the starting depth features are held fixed and the reading pose is
expressed relative to each predicted state. Live execution updates depth at
10 Hz. The frozen motion prior and WM retain their training-only height inputs.
The new image encoder initially emits a fixed prior feature bias to preserve
initial behavior; this initialization does not read runtime terrain. Checkpoints
record the panorama contract, and the direct executor requires depth plus its
reading pose when loading such a policy.

### Experimental surface-based hand placement

`BatchedLearner(..., hand_surface=True)` replaces only newly established hand
placement points with `HandPlaneContact`. Legacy training remains unchanged by
default. The module reads the canonical hand collision mesh and fingertip sphere
through `somaforge_core.robot_assets`, samples 512 support directions, and uses a
0.2 mm soft support selection. It computes the midpoint between the robot-side
support witness and its projection onto the locally planar target surface,
matching the native solver's midpoint convention approximately. This is not a
contact detector, a force model, or an exact multiple-contact solver.

The batched integration currently requires upward horizontal hand targets and
uses the existing atlas height at each target XY for the surface plane. Planned
goal coordinates are not changed. Fixed material points continue to determine
slip and maintained support anchors; motion of a geometric contact point over a
curved hand must not be counted as material sliding. Foot/knee placement and
nonhorizontal surface integration are not implemented by this option.

The opt-in diagnostic checkpoints record `contact_position_model` separately;
the policy's deployment input/action interface is unchanged. The standalone
serial rollout/export remains the legacy fixed-witness diagnostic: experiments
using the new placement model must report their surface-aware batched metrics
and native solver evaluation, not relabel legacy serial errors as surface errors.
On five saved nine-goal climbing executions, the implemented surface proxy had
5.8–8.4 mm mean and 10.3–14.6 mm P95 error against native solver points during
hold. This validates a reduction in observation/target mismatch, not 1 cm
policy execution accuracy or generalization to other contact geometries.

### Continuous WM / policy update ratios

The collector continues throughout training. `--wm-batches N` performs N
dynamics updates per attempted policy update; `--wm-schedule frozen` is a
zero-WM-update experimental control. There is no requirement to finish a policy
stage before fitting dynamics. Desired contact losses train only the policy,
through frozen WM parameters; WM supervision remains measured native transitions.
`--freeze-policy` instead collects with fixed policy weights and performs only
WM fitting. It records a pre-fit dynamics audit, preserves the policy optimizer,
and verifies the entire policy state is unchanged before writing the final
`policy-frozen-verification.json`. This supports isolated model calibration
before testing further policy optimization; it does not imply that calibration
passed the dynamics/contact audit or improved native task success.
This follows the alternating structure described by
[SuperMimic](https://supermimic.github.io/#questions), without assuming its update
ratio is optimal for this contact-conditioned objective.

`--policy-learning-rate` explicitly overrides the restored Adam learning rate
after loading optimizer moments. Omitting it preserves the checkpoint rate.
Logs record the effective rates and actual WM/policy update counts, including
policy updates skipped by the dynamics gate. New stream checkpoints preserve
the actual cumulative WM update count rather than inferring it from iteration.

`--fixed-probe-every 10` evaluates identical height-conditioned reference starts
and all nine target offsets before WM fitting, after WM fitting, and after the
policy update. `fixed-probes.jsonl` separates changes caused by the two learners.
It does not add a loss or supply fitting targets to the WM. Fixed reference
states remain diagnostics, not independent native generalization evidence.
`--sampling-seed` decouples collection/policy sampling randomness from the number
of WM minibatches, for controlled ratio comparisons.

The initial comparison uses the same checkpoint, policy LR 1e-5, 4096 native
environments, and 150 outer iterations for frozen, 10:1 and 2:1 WM:policy ratios.
Actual policy updates can differ when the native dynamics gate blocks them;
compare both iteration and actual update counts. Native frozen-policy episodes,
contact coverage, support/slip success and dynamics audits determine whether a
ratio helps. A smaller imagined loss alone is not evidence of better execution.

### Native dynamics validation before policy updates

The stream reserves nine frozen-policy native evaluation environments and a
separate default 64 environments for WM validation. Neither set enters training
replay. The remaining environments supply actual transitions for WM fitting.
Half of the WM validation environments and 128 training environments replay the
native recorded Motion actions, while the other environments execute the policy.
This supplies full contact transitions even when the current policy fails before
the task deadline. Both groups still use actual native physics and measured
contact labels. The audit reports Motion and policy executions separately and
checks both observed groups, preventing good Motion predictions from averaging
away policy-distribution errors. Missing long policy rollouts remain explicitly
unobserved even when Motion supplies the long-horizon check. The nine task-success
evaluation environments always execute the policy, never reference actions.
Finite pre-reset terminal transitions are retained; sequence sampling rejects
reset crossings using age continuity and terminal flags. Half of the WM sampling
mixture preferentially selects windows containing measured contact switches or
failures when such windows are present. The other half samples ordinary valid
windows. Desired contacts never replace native WM labels.
WM batches reserve 25% of their samples for native Motion executions, rather than
letting the small number of Motion environments be overwhelmed by failed policy
episodes. Actual activation BCE balances observed positive and negative samples
per body part. The newly initialized activation head uses a separate 2e-4 Adam
learning rate; inherited dynamics and force heads retain 2e-5. Legacy one-group
optimizer states are split in original parameter order while preserving moments.
These adjustments fit measured contact state, not requested task contact.
The original native reference's task frames 266..300 provide an additional fixed
validation set, never used for WM fitting. Both live and fixed-reference audits
must pass before policy updates. The fixed 35-step window ends at the task
deadline/hold boundary; validation must not silently extend into later reference
actions that the current task never executes. At normal completion the collector
finishes the outstanding evaluation and then a complete episode of the final
frozen policy; a user interruption instead saves promptly without extra rollout.

Every ten iterations, held-out action sequences are replayed in WM for 1, 8, 16
and 35 steps. The audit compares both recorded height context and height queried
from predicted states. Metrics include root and six canonical contact-link
position errors, joint/rotation error, force error, and contact false-positive/
false-negative rates, with separate switch/steady statistics. Link origins are
used for the dynamics audit: task witnesses are undefined on inactive task parts
and must not be interpreted as six valid material points. No target-contact
position enters these metrics.

Policy updates require the predicted-context audit to pass explicit engineering
tolerances (default 3 cm P95 root/link error, 0.1 rad joint RMSE, 0.15 rad P95
rotation error, 100 N force RMSE and 10% contact classification error). Missing
valid horizons or nonfinite predictions block updates; unobserved classes/modes
are reported as unobserved, not perfect. These are configurable provisional
tolerances, not evidence of task success or Jacobian reliability. Collection and
ten WM updates continue while policy updates are blocked. Target-input parameter
update amplification defaults to 1 instead of 250. The contact-task objective
still updates only the policy through a parameter-frozen WM.

`wm-audit-latest.json` and `wm-audits.jsonl` contain measured errors, coverage and
gate decisions. Iteration metrics distinguish actual policy updates and audit
time from WM optimization and collection. A benchmark without accepted policy
updates is explicitly marked as such and does not establish full policy-training
throughput. Resume warms a fresh replay and revalidates WM before updating policy.

The recovered native motion-5 baseline observed the left-hand transition at
frame 287, one frame later than the previous recording. Use
`--reference-contact-frame 287 --contact-deadline-delay 4` for this recording to
retain the existing 35-step control task; do not relabel the observed event.
Initial inherited-WM diagnostics are under
`tmp/contact-wm/inherited-wm-audit-v2/report.json`. They are a single-trajectory
diagnostic, not validation of an already adapted WM or a generalization result.

## Experimental world-fixed spatial force field

`world_contact_field.py` represents contact on **external surface nodes fixed in
the reference world**, with separate spatial distributions for the six parts.
The requested goal never enters WM or its field. Native labels retain every
eligible allocated Newton point/force pair, including the robot-side force sign;
packing overflow raises an error. `--spatial-contact-slots 32
--export-native-replay --freeze-policy --wm-schedule frozen` can collect a
fixed-policy dataset without updating either model. Clone environment
translations are removed; queries do not move with the robot.

The experimental field distributes each part's inherited resultant force over
its nodes. Its moment about the pelvis feeds a zero-initialized residual in the
inherited normalized dynamics increments. Multi-step WM fitting adds spatial KL,
contact-position error and vector-field error against native evidence. These
are actual-vs-predicted dynamics losses, not planned-contact losses. Policy BPTT
instead minimizes expected spatial error around its goal; optimizing the
centroid alone could accept two incorrect peaks on opposite sides of the goal.
Material slip and existing support constraints still use body motion.

Current experiments use 21x21 nodes per part at 2cm spacing, selected only from
training contact coverage and the terrain atlas. Soft labels support continuous
centroids; the grid spacing is not a guarantee of 1cm accuracy. Field queries
are on external atlas surfaces, while raw solver positions follow Newton's
contact-point convention. Audits must distinguish XY placement, 3D placement,
body prediction and actual native task error. Contact labels are paired with
post-integration state and retain the existing last-solve timing limitation.

This first field shares one force direction within each part and cannot model
opposing contact forces on the same part. Parts without positive native contact
evidence have unvalidated spatial predictions. Free spatial distributions can
also disagree with the body; neither spatial fit nor predicted task success is
evidence of actual execution success. New checkpoints preserve ordinary model
and optimizer state, auto-restore the optional field, and require spatial native
labels to continue WM fitting. Legacy checkpoints retain their original model.
