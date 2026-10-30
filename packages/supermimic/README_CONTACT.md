# Single-segment contact learning

## Native solver verification (2026-10-30)

The sphere-free `surface_v2` geometry and contact response are separate versioned
contracts. `manifold_v2` enables native mesh multi-contact CCD, uses ten 2 ms
substeps per 20 ms control step, and preserves the original EPA normal when
clipping a contact manifold. The patch does not replace native contact positions,
distances or activation. It checks the complete upstream factory source and must
be installed in a fresh process before any CCD kernel is created. Switching
between patched and upstream physics inside one process is rejected.

Run the frozen-teacher stress audit without training:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_native.sh tools/check_contact_solver_sweep.py \
  --output logs/contact_solver_sweep_NEW --headless
```

The audit retains all 296 original motion/terrain pairs (including the selected
109 low-speed tasks). It checks every native solve for malformed contact frames,
missing constraint allocation, capacity overflow and nonfinite values; checks
post-integration state; and compares CPU/device contact reads, reset unknown
states and observation-scope changes. Later phases use cold reference-frame
restarts and small action perturbations. Those phases are stress tests, not
claims of successful trajectory tracking. Reports and failure snapshots are
diagnostic-only and cannot be used as production training inputs. A policy
tracking termination is recorded separately from a solver failure.

Collectors preserve all native rows. Unobserved or reset worlds have
`known=False`; zero summary values are sentinels, not measured absence of contact.
Collision-model exports synchronize live response and friction parameters and
identify world 0 as their representative world, recording per-world differences.

`supermimic.contact_train` is the sole current WM training entrypoint. It uses
SuperMimic's existing `TrackingSim`, GPU `ReplayBuffer`, episode/history masks,
and collect → WM fit → policy BPTT scheduling (`utd`). There is no separate
native evaluation loop or custom episode buffer. Runtime contact checks read
allocated simulator contacts directly, without face classification.

## Task and initialization

- Use the original tracking `WorldModel` directly, including its existing force
  predictor and normalization. The extra native-contact activation head has been
  removed. DAgger initialization loads the source tracking WM exactly; current
  continuation checkpoints store the original WM keys without a wrapper.
- Actor optimization unrolls each selected segment from its exact reset state.
  Predicted limb placement and root endpoint errors occur only at the terminal
  state,
  with the frozen DAgger prior and action regularization along the trajectory.
  The terminal step is the latest actual stop observed in the first native world
  of each segment (target contact, physical failure, or deadline). Until observed,
  it falls back to demonstration length plus the allowed grace. No gradients are
  detached within the retained trajectory, and segments have equal weight. There is no imagined
  contact-classifier gate or absolute upright penalty. Real contacts decide task
  completion. The first transition from DAgger starts a fresh actor optimizer;
  later original-WM continuations restore its optimizer normally.
- Resume the accepted DAgger checkpoint, or a later contact WM checkpoint, and
  retain the DAgger actor as an explicit frozen training prior.
- All 19 original segments are sampled across matching native worlds. Each
  resets to its exact executed demonstration start, with reset sensor history.
  There are no randomized start frames, target offsets, or arrival-time inputs.
- The student acts alone in contact-task worlds. The original tracker maintains
  the other task terrains required by the existing paired scene allocation.
- Completion uses actual allocated simulator contact by the requested body part
  within 5 cm of its goal, with the root endpoint within 25 cm / 0.4 rad.
  `experiments/contact_completion.json` explicitly relaxes demo 0 segment 12
  (frames 478–513, two-foot landing from the box) to 8 cm. Matching uses the
  task manifest, motion, segment bounds and requested parts; other segments
  retain 5 cm. `--contact_criteria` selects this completion profile. Resolved
  radii are recorded in each new run's `contact_tolerances_m`; historical runs
  without that field keep their recorded 5 cm criterion. The continuous
  contact loss and its 0.05 m normalization are unchanged.
  There is no face matching. When several native contact points exist for a part,
  the closest actual witness to the goal is selected, not their centroid.
  A previous brief support loss does not veto a completed transfer. Stages without
  a new contact require their commanded contacts at completion, without a force
  threshold or universal dwell.
- With `--completion_mode accumulated_target`, each requested part can reach its
  contact goal at a different frame; achievements accumulate within the attempt.
  Success, physical failure, or demonstration length plus 10 control frames ends
  the attempt. An inaccurate first touch does not end it. Success on the last
  allowed frame counts. No arrival-time input or tracking-specific failure is added.
- The completion/failure signals from training produce the dashboard statistics.
  Rates describe completed training attempts, not an independent test set.

## Direct native contact reads

The training entrypoint uses `NativeContacts(env, direct=True)`. Its shared
`somaforge_core.contact_system` evidence owns all native solver rows, decodes
per-contact force/torque and maps actual robot bodies to six parts. Activation
uses no mesh/BVH or face/normal filter. Replay stores `paired_force_w` and
`contact_count` separately from existing body-sensor force6; neither force
channel defines activation. Reset states are unknown and solve/pose timing is
explicit.
`--audit_contacts` compares a retained solve against an independent CPU reduction
of the same raw contacts. `--profile_collection --freeze_actor --freeze_wm`
measures collection without changing weights. Legacy face readers remain for old
recordings; old face-based reports must not be compared as identical criteria.

New runs default to `--contact_position_loss surface_region`: fixed local
collision-surface nodes with area-weighted smooth scalar costs.
`--surface_samples_per_part 128` and `--surface_temperature .02` configure it.
The default target radius is zero; `--contact_region_radius` needs an explicit
unit world `--contact_target_normal X Y Z`. The model/atlas/target contract is
saved and checked on resume. `collision_surface` and `measured_material` remain
explicit legacy objectives. Native success and event/terminal timing remain
independent of geometric guidance.

`--contact_evidence_every 1000` saves owned point-force evidence and sampled
surface features; zero disables exports. `surface-atlas.npz` and
`contact_binding.json` identify nodes and body mapping. Run
`python tools/check_surface_contact_system.py` for CPU DMO integration checks,
and `python tools/audit_surface_contact.py --help` for the fixed-trajectory
audit. Shared contracts and migration limits are documented in SomaForge's
`docs/surface-contact-system.md`.

## Efficiency checks without training

The 2026-10-30 efficiency pass preserves contact criteria, duration selection,
terminal targets, regularization, and BPTT boundaries. Checkpoints are loaded
through CPU memory mapping and only required model/optimizer tensors reach the
GPU. DAgger reuses its archive CPU snapshot for aggregation and concatenates
restored rollout chunks once per field. WM fitting reads only required replay
fields and shares normalized state/context inputs between state and force heads.
DMO resolves immutable event indices before BPTT and evaluates root terms only
at their supervised endpoints. Per-segment reports are reduced on the GPU and
read back together. `--progress_interval_seconds 1` limits progress snapshots
to roughly one per second; final status and normal iteration logs remain.

`tools/profile_contact_pipeline.py` accepts the normal contact training CLI plus
`--profile_collection --freeze_actor --freeze_wm --policy_gradient dmo`.
It collects complete real first attempts, computes WM/actor gradients, checks
that weights remain unchanged, and exits before checkpoint/replay saving.
Its Adam step is explicitly disabled. This measures initialization, native
collection, replay preparation, metrics, WM fitting and full-batch BPTT without
a formal training run. Use a fresh output directory and enough replay capacity.
The probe currently supports independent single stages; it does not benchmark
continuous sequence collection or full multi-motion DAgger training.

Paired checks are `tools/check_contact_pipeline_efficiency.py` (pre-change
source directory required), `tools/check_contact_dmo_efficiency.py` (pre-change
DMO source required), `tools/check_contact_collection_efficiency.py`, and
`tools/check_contact_io_efficiency.py`. Reports and comparison sources are in
`tmp/contact-tracker-20261027-v1/efficiency-20261030/`.
At 4096 physical worlds / 4057 contact worlds, the frozen legacy-baseline probe
collected 95 frames in 2.89–2.90 s and ran warmed surface-region DMO backward in
2.74 s, using the original baseline's 30-frame grace. These component timings exclude optimizer steps and periodic checkpoint
writes, and are not a direct before/after comparison with historical training.

## Running

Use `tools/run_native.sh -m supermimic.contact_train` with `--task_manifest`,
`--checkpoint` (original tracker), `--recording`, `--resume`,
`--prior_checkpoint` (accepted DAgger initialization), and `--output`.
Defaults are 4096 physical worlds, 16-step WM fitting, full-segment policy BPTT,
512 WM samples and one exact start for each selected segment per policy update.
The 4096 physical environments remain unchanged; repeated identical imagined
starts are unnecessary for this deterministic full-segment objective. Native allocation, state
history and contact semantics are validated before production training.

`tools/contact_monitor.py --kind wm` serves only JSON/HTML under the existing
visualization server. The current browser URL is
http://127.0.0.1:18767/contact-segment-wm/ . It displays training success rates,
actual updates, collection/WM/policy timings, and terminal state.

## Preserved artifacts and removed flows

The accepted initialization is
`tmp/contact-tracker-20261027-v1/demo0-dagger-mse-v2/checkpoint_dagger.pt`
(best DAgger round 28, 1400 updates, action MSE 0.05343).
All weights, logs and data remain in the externally stored experiment directory.
Superseded source was archived there under `retired-flow-source/` for provenance.
The custom `contact_stream.py`, `FastNativeContacts`, random-start sampling,
serial segment-evaluation scheduler and mixed offline-BC/DAgger/WM CLI were
removed from the current training flow. DAgger helpers and native collection /
replay diagnostics remain available; no unrelated project changes were reverted.

Contact termination tracks inactive-to-active transitions of the requested parts, independent of face and placement. Initial contacts do not count. For multiple requested parts, each must make a new contact during the segment; they need not coexist. Placement uses each first onset, and root uses the final onset. Root success tolerance defaults to 0.25 m; the terminal loss scale is unchanged (0.1 m). Historical run reports retain their original thresholds.


DMO mode (`--policy_gradient dmo`) collects fresh complete native first attempts
with fixed actor parameters, then reconstructs the BPTT graph on those exact
states before a single optimizer step. All selected physical worlds participate;
`--dmo_microbatch 4096` is the default and handles all 4057 task worlds in one batch. Smaller values can bound graph memory without dropping trajectories. State,
sensor context, and action forward values are copied from simulation; the original
tracking WM supplies state/context derivatives. TF32 is disabled for reproducible
actor replay across batch sizes. `--check_dmo` checks exact state equality, action
replay consistency, and terminal-to-first-action gradients on the first update.
Measured per-part onset points supply forward contact errors and their body-local
FK derivatives; missing contacts retain the existing terminal FK proxy, reported
separately. Root/pose loss uses the native endpoint. This is a DMO-BPTT adaptation
(arXiv:2509.00215v2 A.1.3), without a critic; `--freeze_wm` keeps the original WM
fixed for the initial experiment. It does not guarantee correct local derivatives.

Frozen-policy comparisons can explicitly use `--freeze_actor --freeze_wm
--completion_mode accumulated_target --root_tolerance 0.25`: each part's actual
same-face contact within 5 cm is accumulated during the episode, and success
requires the root/pose test at the current frame. A first inaccurate touch does
not end the attempt. The original demonstration-plus-10-frame deadline and
physical failure rule still apply. DMO training supports this mode: a part contributes once at its first valid goal contact; parts still missing at the native endpoint use the endpoint contact/FK error. Root loss is evaluated at that native endpoint. Imagination-mode learning does not support this completion mode.

## Accepted restart baseline

Use `tmp/contact-tracker-20261027-v1/demo0-dagger-mse-v2/checkpoint_dagger.pt`
for the next actor initialization, with the original tracking WM. The canonical
artifact pointer is `tmp/contact-tracker-20261027-v1/accepted-baseline.json`.
The frozen initialization recheck passed 14/19 first attempts under the explicit
25 cm root / accumulated-target criterion. Degraded full-segment optimization
runs and their DMO descendants were deleted at the user's request; their weights
are not restart candidates. The DMO implementation itself is retained.

Single-task DMO verification uses `--segment 7 --completion_mode accumulated_target
--policy_gradient dmo --dmo_microbatch 4096 --root_tolerance .25 --freeze_wm`,
restarting from the accepted DAgger actor. `native_batch` in each metrics row
contains first-attempt success, target-face contact coverage, closest measured
contact error, and endpoint root/pose error for the very trajectories used in
that update. `collected_policy_updates` identifies the actor that generated them;
the optimizer update logged on the same row happens afterward. Missing contacts
and post-reset frames are excluded from measured contact-error means.

## Adam momentum diagnostic

`--adam_beta1` selects the actor Adam first-moment decay (default 0.9); the
second-moment decay remains 0.95. `--update_diagnostics` records the raw gradient
norm, actual parameter step norm, adjacent clipped-gradient cosine, and step
cosine with the current negative gradient. A negative last cosine means the
actual step is locally uphill for the surrogate loss, not proof of simulator
loss increase. Diagnostics do not change the gradient or optimizer step.

## Online WM and adaptive actor LR experiments

DMO online WM fitting follows paper §3.2 / Algorithm 1: draw real one-step
transitions from replay, fit the model before collecting the next actor rollout,
then hold its parameters fixed during actor backpropagation. The original
tracking deterministic residual network uses the deterministic MSE branch also
available in the authors' `algorithms/dmo_shac.py` (`no_stoch_dyn_model`).
Tracking coordinate normalization and existing force6 supervision are retained.
This is not a new stochastic variance head or a 16-step autoregressive fit.
Source: https://arxiv.org/html/2509.00215v2 and
https://github.com/machines-in-motion/DMO_code/blob/master/algorithms/dmo_shac.py .

`--adaptive_lr` is our requested experiment, not the paper's linear schedule:
use mean success over the last three native training batches; after three
non-improving observations (absolute improvement threshold 0.01), halve actor
LR, down to `--min_learning_rate` (3e-7). No additional simulator evaluation.
Scheduler state is checkpointed. WM pre/post metrics measure the same training
batch and are fitting diagnostics, not held-out accuracy or Jacobian validation.

### Adjacent two-stage DMO windows

`supermimic.contact_train --sequence_length 2` trains all adjacent windows
(1→2 through 18→19), using `accumulated_target` completion and DMO. Reset occurs
only at the recorded start of a window. Successful stage handoff changes the
goal and stage-local deadline without resetting physical state, action history,
or the differentiable replay chain. Failure in stage 1 ends that window.

Contact/root endpoint losses are averaged over retained stages within each
window. Duration selection uses local stage durations and the existing rolling
prior. An earlier supervision cutoff in stage 1 does not detach the chain to
stage 2. Short excluded stages contribute no endpoint loss; retained later
stages can still differentiate through their actions. Native success requires
both stages and is measured before filtering. WM fits only active native
transitions, excluding inactive tails after window completion.

When switching window length, actor Adam moments are reset; actor/WM weights
and duration history are loaded from the supplied checkpoint. Provide a replay
capacity at least the largest two-stage duration plus two grace periods
(current task: 190 frames; use `--buffer_steps 256 --warmup_steps 190`).
Regression check: `tools/run_native.sh tools/check_contact_pairs.py`.
The first actual update with `--check_dmo` checks exact simulator forward states,
action replay consistency and the second-stage-to-first-action gradient.

### Continuous collection with independent stage gradients

`--continuous_random` samples a uniform random **segment start** for each world
at each collection. It then executes successive goals without physical or
history resets until a physical failure, stage timeout, or the final goal.
There is no fixed two-stage execution window. Completion uses the existing
accumulated target-contact/root criterion, not the diagnostic contact-only gate.
The current task's worst-case collection horizon is 1369 frames (799 + 19×30);
use `--buffer_steps 1408 --warmup_steps 1369`. Collection stops early once all
worlds finish their attempts.

`contact_sequence.stage_batches` extracts each retained stage from native replay,
including its exact five-observation history from the preceding stage. Each
stage gets a separate BPTT graph and its own endpoint loss; the shared actor is
updated once after accumulating all stage gradients. No later-stage loss reaches
an earlier stage's actions. Native collection uses 4096 worlds; replayed stages
are processed in batches of up to `--dmo_microbatch` for bounded GPU graph memory.
Existing local duration filtering remains actor-supervision-only; it never
advances the simulator's goal. WM samples all active collected transitions.
The monitor reports completed-stage counts, per-stage contact/root losses,
native success rates and the count of training entries inherited from a prior
stage. Checks: `tools/run_native.sh tools/check_contact_continuous.py`.

Continuous stage replay now sorts retained samples by effective BPTT length
before forming batches (disable only for profiling with
`stage_batches(..., sort_by_duration=False)`). Native rollout, selected stages,
loss weights and gradient boundaries are unchanged. Continuous status reporting
also copies its four counter arrays to CPU together rather than reading every
scalar separately.

Matched-data benchmark: `tmp/contact-tracker-20261027-v1/dmo-stage-batching-benchmark-v3/batching-benchmark.json`.
On 9591 retained stages, steady BF16 backward time changed from 8.56s to 5.29s;
padded stage-steps from 795750 to 411433; additional peak allocation from 36.84GB
to 16.37GB. Loss difference was 2.5e-7. Relative gradient differences were 6.56e-4
with the existing BF16 WM and 2.26e-7 in the FP32 control. No actor optimizer
updates were performed by this benchmark. These numbers measure replay+BPTT,
not the whole training iteration. The collector still waits for all active
attempts; asynchronous collection is separate work because unfinished segments
can span policy updates and require an explicit policy-consistency design.

### Consistent collision-surface contact objective

Fresh contact collection/training now defaults to `--collision_profile surface_v2`
and training defaults to `--contact_position_loss surface_region`. The versioned
canonical USD removes all 14 auxiliary sphere colliders and duplicate meshes:
29 robot meshes, 4 cylinders, zero spheres. Continuous hand meshes remain.
`legacy_v1` is explicit for reproducing old runs; existing jobs are unaffected.

Contact training exports its actual running solver to `solver-collision.mjb`
with a profile/hash sidecar instead of choosing a historical MJB by default.
The atlas reads this geometry and uses area-normalized scalar region costs.
Physical recordings, contact policy resumes and priors must match the selected
revision; old contact labels need fresh native execution. The optional
`--allow_collision_transfer` accepts old tracker weights only as an unvalidated
initialization, and never exempts recordings/resumes from these checks.
New collection exports the same solver model for audit. Full architecture and
verification notes are in SomaForge `docs/surface-contact-system.md`.

The following exact closest-point mode is retained as a legacy comparison.

`contact_position_loss=collision_surface` uses one geometry definition before
and after sensed contact: the closest point on the requested robot part's actual
convex collision mesh/helper spheres to the target. `contact_surface.py` reads
the canonical MuJoCo model's mesh vertices, geom transforms and fixed helper-body
transforms; it builds convex hull triangles once, then uses differentiable
triangle/sphere projection through the existing canonical FK. It does not switch
between a calibrated material point and a simulator witness when sensors toggle.
Sensors still determine actual target-contact success and native stopping.

The legacy `measured_material` mode remains available for controlled comparisons.
Contact-event timing can still change the supervised frame; this change removes
the witness-source discontinuity, not all contact or stopping discontinuities.
The independent six-case refinement uses normal weight 4 and tangent weight 1,
with the original fixed deadlines and separate policy/optimizer per case.
Verification: `tools/run_native.sh tools/check_contact_surface.py` checks analytic
triangle projections, canonical FK against MuJoCo, finite-difference gradients
away from feature boundaries, and identical loss/gradient when sensor activation
changes while state and supervision frame remain fixed.


CPU compatibility checks for the physical revision:

```bash
PYTHONPATH=/path/to/somaforge/packages/somaforge_core python tools/check_collision_profile_contract.py
# After fresh surface-v2 collection, inspect its actual objective geometry:
python tools/check_contact_surface.py --collision_profile surface_v2 \
  --recording tmp/<surface-run>/rollout.pt \
  --collision_model tmp/<surface-run>/solver-collision.mjb
```

Native GPU comparisons have verified the surface-v2 solver geometry and contact
reader. The frozen legacy tracker does not complete the tested surface-v2 motion;
it is not an accepted teacher. A fresh surface-v2 tracking run is in progress.

The staged flow in `tools/run_surface_training_flow.py` scales the number and
coverage of **complete contact-transfer episodes**, rather than simulator steps.
After tracking validation, `contact_collect --all_motions` records each paired
motion in its own native world. `tools/build_contact_transfer_library.py` accepts
only complete successful source trajectories and builds each contact clock before
pooling. Source boundaries cannot create contact events or borrow initial contact
evidence from another motion. The catalogue records source motion/frame bounds,
initial support, new/maintained contacts, target heights and displacements;
support-only tails are counted separately from contact transfers.

The first library must contain at least 64 transfers from at least 4 motions.
These are acceptance thresholds, not a claim that the dataset already exists.
Every accepted case enters balanced BC/DAgger and independent contact episodes
using its matching terrain. The current staged policy is one goal-conditioned
actor with separate episode starts, not separate model weights per case.
Force6 remains enabled. A failed coverage or physical-contract gate stops the
queue before downstream training. CPU regression commands:

```bash
python -B tools/check_contact_transfer_library.py
python -B tools/check_surface_contact_system.py
```


### Default slicing rule: remove idle ends

Future slices remove long standing/waiting intervals at the beginning and end,
including the initial settling/waiting prefix. Keep actual transition dynamics
and two frames of boundary context. The current recording's accepted proposal is
`experiments/contact_trim_demo0.json`: first slice [44,82), final slice [690,732),
with interior boundaries unchanged and original contact/reference goals retained.
The original tracking teacher succeeds 107/107 from both starts; mean first-action
execution changes from 82.24 to 38.30 frames. Restore original reference clock and
sensor history at warm starts. Existing baselines remain unchanged.
