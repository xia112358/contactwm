# Joint policy refinement and executed trajectory collection

Explicitly supplied motions are kinematic references. This workflow jointly fine-tunes one
existing WBT policy against a reference collection and collects complete
executions from one fixed refined checkpoint in one balanced parallel batch.
Aggregation retains event alignment and whole-body/joint smoothing. Final
aggregate poses receive fresh Newton labels. Evaluation is an optional separate
diagnostic and is not a collection prerequisite.

```text
explicit canonical reference collection
  -> reference manifest + matched terrains + base checkpoint
  -> prepare and fingerprint inputs
  -> joint PPO refinement (one policy, all references)
  -> frozen-checkpoint collection: N references x K environments in one run
  -> lossless bounded recording chunks + complete first-episode extraction
  -> shared event alignment + robust pose/contact fitting + temporal smoothing
  -> canonical Newton FK + fresh Newton contact labels + aggregate manifest
```

The controller is `scripts/run_trajectory_refinement.py`. Its implementation
lives in `somaforge_core.trajectory_refinement` and `somaforge_core.rollout_dataset`.
It has no motion-name, collection-size, height or task-specific checkpoint
defaults. The current backend deliberately retains SomaForge's authoritative
G1 asset and Newton/MJWarp contact contract.

The old trajectory editor/generator and augmentation dispatcher are retired.
This retained workflow starts from an explicitly supplied canonical reference
manifest; it does not author edit plans or run trajectory IK. Existing edited
reference collections remain historical inputs and must pass the same current
identity and evidence checks. They are not recertified by retaining their files.

The former `generate-ref` and `scripts/run_augmentation_refinement.py` routes,
source-prefix generation workers and `climb00_source_prefix_v14.json` recipe are
historical records, not supported preparation entrypoints. Original contact/load
observations remain attached to each actual policy rollout's native solver
record. Those unchanged executions stay distinct from fitted aggregate
kinematics and their subsequent fixed-q contact labels.

## Direct parallel collection from an existing milestone

`scripts/collect_refined_rollouts.py` accepts an immutable training plan and an
explicit milestone without resuming training or rewriting historical plans.
It verifies the checkpoint's training manifest, reference/terrain hashes and
current implementation. All N references receive K fixed environments, using
`environment_id % N`; independent clips do not enable segment chaining. Default
K is 16. The recorder flushes every 16 control samples, preserving all original
raw/solver contacts and same-solve load channels. No initial pose noise is added
unless it was explicitly configured; replicas are not claimed independent.

```bash
source scripts/source_isaaclab3_newton_setup.sh
python scripts/collect_refined_rollouts.py \
  --training-plan /path/to/refinement/plan.json \
  --checkpoint /path/to/model_11400.pt \
  --output-dir /path/to/new_collection \
  --envs-per-motion 16 --chunk-frames 16 --headless --execute
```

Before aggregation, only complete uninterrupted first episodes contribute.
Recorded terrain translations are removed consistently from root, body and
contact positions; quaternions retain their original orientation. A separate
regression verifies heading and per-motion local clocks. Fewer than three
complete executions produce an explicit insufficient-data result.

Contact-task extraction uses `somaforge_consensus_extraction_v1`, defined in
`somaforge_core.extraction_standard`. Its current generation engine is
`parallel_continuous_population_consensus_v12`; admission requires certified continuous transport, actual population events,
continuity relative to actual sources transported onto the identical output
clock, and canonical joint limits. Original-clock comparisons remain diagnostics.

Complete executions provide observations, including their disagreements. Group
incompatible support strategies before fusion, with at least three contributors.
Derive shared events from the new executions, never the unedited reference.
Those events initialize a joint registration of one continuous latent motion and
one monotone whole-body clock per execution. Both participate in a robust fit;
soft event residuals, transported motion derivatives, latent smoothness and a
population time gauge are optimized together. Missing events do not acquire
invented anchors. Nearby independent events need not acquire a total order.

Translation and joints use C2 cubics. Root orientation uses a continuous unit
quaternion lift of the same cubic representation; its first/second derivatives
include normalization and the clock chain rule. Heading and path share the same
clock, including under a rigid world rotation. Insufficient quaternion sampling
fails explicitly. Exact source samples remain unchanged at integer source times.
The quintic clock displacement fixes endpoints, endpoint rate (one) and clock
acceleration (zero). Bernstein derivative coefficients certify the 0.25–4 rate
bounds throughout each polynomial span, not just at sampled frames.

All transported poses and the jointly estimated consensus receive actual Newton
queries before they can contribute to the final result. Preserve the original
and transported observations. Compare both original-clock and transported-source
envelopes to separate retiming from fusion variation. The transported-source
envelope is the fusion admission baseline; raw-clock peaks do not identify
new fusion jumps after retiming. Continuous transport certification and actual
source-event preservation are independently required. The established robust whole-body marker
fit, event guide, staging, and incumbent retention remain active.

The fitter retains the robust marker loss, a maximum of 600 Adam/cosine steps, marker
second-difference weight .05, joint second-difference weight .2 and joint prior
.02. Fresh actual Newton primary-face witnesses guard contact during fitting;
between queries frozen witnesses provide an optimization guide only. Explicit
matched release events supply release intent. A fixed median material point is
no longer used to certify retention. Final poses still receive full fresh Newton
labels, and these must agree with the optimizer's native validation.
Fitting stops early only after fresh successful native contact and continuity
audits and sustained objective convergence (default relative improvement .001,
patience 50 iterations, at least three fresh audits). A failed audit resets the
streak. Frozen geometry guides between queries cannot establish convergence.

Reprocessing uses a previous smoothed aggregate as an optimization initial guess.
Correct actual contact events and local continuity are both required for admission.
Global marker/joint acceleration RMS remains an optimization/comparison metric;
it is not capped by the old aggregate. Candidate selection prioritizes actual
contact failures, then continuity admission, worst local excess, violation count
and the fitting objective. The exact existing aggregate is evaluated before any
optimization step and retained if no candidate improves that ordering.
All fits still use multiple source executions; copying
an accepted source is not aggregation. Completion-only inputs retain the legacy
pose-alignment adapter and do not acquire event verification implicitly. Contact
acceptance is kinematic evidence, not a proof of load capacity or dynamic feasibility.

The local audit checks each frame's root translation, shortest root rotation,
29 joints and four markers on every canonical URDF link. Output phases come from
that output's freshly matched Newton events, including collapsed repeated events;
source intervals are matched by event identity, allowing independent reversals.
Within each phase, per-channel maximum source peaks on the identical output
clock plus float32 roundoff define the fusion envelope. The original-clock
envelope is retained separately for retiming diagnostics. Median/MAD deviations remain denoising diagnostics. A stencil crossing an event uses adjacent phase
envelopes. First differences are speed diagnostics; sustained fast motion alone
is not called a discontinuity. Second differences (linear/angular velocity bends)
are hard acceptance checks and catch isolated pose jumps. The optimization guide
targets 85% of the bound to leave room for other fitting objectives. Bounds refresh
only with actual output contact observations; intermediate gradients are guides.
The historical fixed-clock audit is retained for interpreting older artifacts,
but is no longer used for admission.
Reports retain every offending frame, channel, value, limit and phase, as well
as the full calibration bounds. This is a source-relative anomaly test, not a
universal guarantee of physical feasibility. The existing robust denoising and
smoothness terms remain active even inside the accepted envelope.

Existing collections can be reprocessed without new policy inference or training:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python scripts/reaggregate_rollouts.py \
  --training-plan /path/to/refinement/plan.json \
  --source-collection /path/to/retained_collection \
  --output-dir /path/to/new_consensus_version \
  --failed-manifest /path/to/reassessed_dataset_manifest.json
```

Use `--case-keys` for explicit diagnostic cases, or omit both selectors to process
all references. Cases remain serial; `--query-worlds` batches independent fixed-q
contact queries within one case, without changing collision assets or margins.
The new receipt pins upstream inputs and implementation; original motions,
labels, collection plans and checkpoints are not overwritten.

To change acceptance on an existing run, pass `--reuse-results /path/to/old_run`.
The controller first verifies and reassesses eligible completed poses and their
exact Newton labels on CPU, then processes pending/rejected cases in the new
output. `--reuse-manifest` also imports indexed `validation_result` receipts from
a combined manifest, verifying their upstream collection and artifact hashes.
`--reassess-only` stops after that audit. Reusing unchanged labels requires their
source hash, runtime and shared face semantics to validate; changed poses receive
fresh Newton labels. The old RMS cap is retained only in historical reports.

The collection retains `collection_plan.json`, `recording.npz.json` and its
lossless parts, per-reference candidate chunks, `alignment.npz`,
`optimization.json`, fitted `motion.npz`, new `contacts.npz`,
`collection_summary.json` and `dataset_manifest.json`. Raw source states and
loads remain recoverable through the original recording. Inference-only policy
copies are marked as such and reference the unchanged full training checkpoint.

The retired augmentation workflow previously generated references with separate
IK/FK workers before dispatching refinement. Its worker counts, device settings,
queue/socket protocol and `--stage all` command are no longer current interfaces.
Use the retained refinement and collection commands with explicit existing
reference manifests; preserve their fingerprint checks and immutable source
records. Native reference refinement and population-consensus processing below
remain maintained and are distinct from that retired trajectory generator.

Checkpoints use the original WBT format: networks, optimizers (including learning
rates), normalizers, configuration, completed iteration and existing sampler /
curriculum state. Resume continues at the next iteration with a fresh simulator.
Periodic milestones and the final checkpoint are retained. The deprecated
`train.full_checkpoints` option remains accepted for saved-config compatibility
but no longer enables native Newton / MJWarp runtime snapshots or initialization
snapshots. Existing artifacts remain untouched. Starting from the base policy
uses `--warm-start` to avoid inheriting a sampler for a different reference set;
subsequent resumes restore the current reference sampler.

Initial-state diagnosis (2026-10-14): an unstable augmented start can force the
policy to recover balance before following the task. Those recovery adjustments
can move the robot away from the reference, trigger tracking termination or
disrupt the intended contact sequence, making the task fail even when the
kinematic reference looks plausible. Check the initial stance, root attitude
and velocities together before attributing early failures to insufficient
training; a rotated reference must rotate its root attitude consistently.

With the same iteration-8800 policy, the diagnosed -10-degree case failed at
frame 51 before repair; three repaired trials completed all 932 frames. The
diagnosed +20-degree case failed at frame 15 in all three original trials;
all three repaired trials completed 932 frames, and two passed all 34 contact
event checks. These comparisons support an initial-state cause for these cases,
not a claim that every failure has this cause or that initial recovery alone
guarantees task acceptance. Abnormal side contacts still prevented dataset
admission. Evidence: project `tmp/trajectory_refinement_diagnostics/` under
`20261014_minus10_source_prefix_v1/validation_summary.json` and
`20261014_plus20_source_prefix/validation_summary.json`.

## Configuration and input adapters

Paths in a workflow configuration are relative to the repository root. Paths
inside input manifests, collection indexes and edit plans are relative to their
declaring file. Absolute paths are accepted for external inputs.

`input.kind` supports:

- `motion_manifest`: `motion_files` with `motion_file`, `terrain_id`, unique
  `motion_name` or `motion_id`, and optional positive `weight`; `terrains` with
  `terrain_id` and `terrain_file`. For contact-event acceptance, each motion also
  supplies `contact_catalog_file`, a Newton label NPZ from which only the explicit
  scene surface catalog is read. An entry's optional `task_contract` supplies its
  own intent contract, supporting different trajectory lengths and tasks in one
  joint training manifest; it overrides the shared `task.contract` default.
- `native_edit_collection`: an index with `motion_files` containing `motion_id`,
  full `motion`, `plan`, and `contacts`. The plan binds its target terrain through
  `metadata.target_terrain_mesh`. Cropped/soft-joined fields and old acceptance
  flags are intentionally not substituted for the full reference.

Both adapters validate the motion's robot and kinematics provenance, finite
state arrays and timing, and build the same canonical motion/terrain manifest.
Reference weights affect joint training; evaluation covers every reference.
The trainer currently reserves three start probes per reference, so
`train.num_envs` must be at least three times the collection size.

All settings are explicit in the configuration: `base_checkpoint`, `output_dir`,
training environment count/additional iterations/save interval, evaluation
attempt count/minimum success rate/seeds/reset noise, collection attempt count/
seeds/reset noise, and the task acceptance contract. Unknown configuration keys
fail validation rather than silently changing behavior.

The example [climb00_joint_v12.json](../configs/trajectory_refinement/climb00_joint_v12.json)
binds the existing full v12 collection (207 references, five terrains) to the
original motion's iteration-6000 WBT checkpoint. Its proposed budget is 3,000
additional PPO iterations, 4,096 environments and a checkpoint every 100
iterations. Evaluation requires 3/3 successful trials per reference; collection
requests ten further trials. These are configurable starting settings, not
measured sufficiency or an instruction to start training. Zero reset noise gives
repeatability checks, not perturbation robustness; use a separate configured
run with explicit reset noise to assess robustness.

## Prepare without GPU work

Select the normal project Python environment. Preparation needs NumPy and the
installed project packages, but does not import or initialize Isaac/Kit/CUDA.
For offline contact extraction/tests use the pinned Newton environment so that
runtime metadata can be checked without launching physics.

```bash
source scripts/source_somaforge.sh
python3 scripts/manage_environment.py run --profile main -- python -B \
  scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json
```

The default stage is `prepare`, without writes or process launches. Adding
`--stage prepare --execute` writes the pinned `plan.json`, the joint
`references.json`, and per-motion recording manifests. This stage still does
no GPU work. Preparation can be repeated with identical inputs; changed files,
configuration or workflow implementation require a new output directory.
Checkpoint metadata is checked on CPU when planning/executing training or
evaluation; prepare only checks its existence and content hash.

```bash
python3 scripts/manage_environment.py run --profile main -- python -B \
  scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json \
  --stage prepare --execute
```

## Explicit execution stages

Run the following stages only after GPU work is authorized and resources are
available. The controller does not wait for GPUs, schedule future jobs, stop
other processes, or automatically start the next stage.

Activate `scripts/source_isaaclab3_newton_setup.sh` before running simulator
stages. Each command below is a **dry run** until `--execute` is explicitly
added. Replace the example configuration for a different reference collection.

```bash
python scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json --stage train

python scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json \
  --stage evaluate --checkpoint /path/to/refined/model_09000.pt

python scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json \
  --stage collect --checkpoint /path/to/refined/model_09000.pt

python scripts/run_trajectory_refinement.py \
  --config configs/trajectory_refinement/climb00_joint_v12.json \
  --stage export --checkpoint /path/to/refined/model_09000.pt
```

Training reuses `train_motion_edit_policy_finetune.py`: one joint manifest, one
policy, existing PPO/optimizer configuration, self-collision enabled and
failure-window sampling. `train.iterations` is an additional budget relative to
the base checkpoint. A training-stage `--checkpoint` resumes from a checkpoint
inside this workflow's training directory and uses the remaining budget.
The controller does not claim an exhausted budget means the references are
feasible. Unsuccessful motions remain explicit failures.

Simulator arguments `--device`, `--visualizer kit` and `--headless` are forwarded
to existing AppLauncher-first entrypoints. No visualizer means headless;
`--headless` takes precedence over `--visualizer kit`. The controller itself is
a CPU command dispatcher and does not consume a GPU to parse its options.

Optional evaluation retains its historical single-motion diagnostic queue.
Collection uses one balanced parallel run and bounded recording chunks. The
`collection.envs_per_motion` and `collection.chunk_frames` settings default to 16.

## Acceptance and evidence

Two independent results are retained for each trial:

1. Complete execution: the first episode has a contiguous native clock from
   reference frame zero through its final frame, finite recorded states, no
   early reset/termination, and the expected motion identity. Only the recorder's
   duplicate initial/bootstrap sample may be skipped; later episodes cannot
   hide an earlier failure.
2. Configured task acceptance: `contact_events` applies the existing
   `somaforge_core.event_acceptance.audit_events` to freshly recorded
   active/allocated contacts, with explicit intended surface mapping.
   Establish/keep/release requirements and timing tolerances come from the
   supplied contract. `completion_only` is an explicit weaker mode: task intent
   remains `not_requested`, and the dataset's scope must not be called task
   acceptance. A different task metric requires a new explicit adapter.

Contact identity comes from actual solver `type`, `dist`, `includemargin`, and
all required `efc_address` rows. `newton_support.select_support_contacts` resolves
native primary faces and uses `contact_face_selection`; sides and other
non-upward faces remain abnormal evidence. `max_abnormal_contacts` is an
acceptance count budget, never a contact distance threshold. Self/cross-robot
contacts do not become terrain contact. No margins, gaps, collision assets or
policy physics are modified to fit reference labels.

Missing fields, stale runtime provenance, unknown surface mappings and
unallocated active contacts stop the queue with an `unknown` report. They cannot
silently pass, fall back to force thresholds, or be treated as observed task
failure. Normal task failures are recorded and the queue proceeds.

The pipeline retains same-solve force and relative tangent-velocity observations.
Contact establishment alone is not evidence of stationary or load-bearing
support. This adapter does not impose an undeclared load/slip budget and does
not claim unrecorded substep motion is verified.

An optional evaluation reports its own success-rate gate. Collection does not
read that gate: it records every requested reference directly from the fixed
checkpoint. Failed first episodes remain in the audit. A different checkpoint
creates a separate result tree. Aggregate task checks use fresh final-pose
Newton labels; insufficient samples remain explicit.

## Outputs and recovery

```text
output_dir/
  plan.json, references.json, references/motion_*.json
  training/<attempt>/command.json, process.log, models/, completed.json
  policies/<checkpoint-sha256>/
    evaluate/<motion-key>/trial_*/<attempt>/recording.npz, report.json
    evaluation_summary.json
    collect/<motion-key>/trial_*/<attempt>/recording.npz, report.json
    collection_summary.json
    dataset/<motion-key>/trial_*/motion.npz, observations.npz, export.json
    dataset/manifest.json
```

Resume by repeating the same stage/config/checkpoint. Completed trials are reused
only after checking their context and all artifact hashes. Interrupted or unknown
attempts get a new attempt directory; old evidence is preserved. A controller
file lock prevents concurrent execute commands against the same workflow output.
There are no cleanup or permanent deletion operations.

Export stores recorded root/joint poses, velocities and body states without
averaging, temporal interpolation, FK replacement or pose correction. Each trial
remains an independent trajectory. `observations.npz` contains fresh effective
contact labels, abnormal pairs, load/slip observations, mapping and sampling
provenance. The complete original raw and solver contact record remains linked
by SHA256. Post-integration poses and solver geometry are explicitly distinct.

The observation schema is `somaforge_executed_contact_observations_v1`.
Existing fixed-q label loaders intentionally cannot treat this sidecar as
strictly same-state fixed-q contact truth. Downstream Predictor/Infiller data
preparation must explicitly consume this native recording clock, or separately
materialize fixed-q diagnostics while retaining the original execution evidence.
This workflow does not silently replace current training datasets or baselines.

`--stage status` reports prepared references, completed training runs,
evaluations and exported manifests. GPU behavior remains unverified until the
explicit simulator stages have actually been run.

## CPU regression tests

```bash
python3 scripts/manage_environment.py run --profile main -- python -B -m pytest \
  tests/test_trajectory_refinement.py -q -p no:cacheprovider
```

Tests retain their generated artifacts under project `tmp/trajectory_refinement_tests/`.
They exercise actual CPU contact activation/face selection, task failure,
unallocated/missing solver evidence, unchanged pose/velocity export, altered
recording detection, fixed-policy stage gating and no-process dry runs. They do
not launch Isaac, train a policy or run a GPU smoke test.
# Standard extraction and export

Formal admission requires all three checks: population event information retained,
source-relative local continuity, and canonical nominal joint limits. Root
translation, SO(3) attitude, all joints and canonical whole-body markers receive
local jump checks. Overall RMS remains a denoising/comparison metric. A valid
single-source fallback stays diagnostic and is never exported as a formal
multi-execution aggregate. At least three distinct executions must contribute.

`summary.json` and `progress.json` state the complete requested case scope,
processed count, pending count, `legacy_unverified`, `needs_refinement`,
`insufficient_data`, `source_baseline`, accepted cases and exported variants.
`complete` means every requested case finished processing; `all_passed` means
every case passed formal admission. Multiple valid strategies can produce more
variants than accepted cases. A subset run never claims to cover the whole plan.
Old `accepted` flags without current evidence verification remain unverified.

The formal `dataset_manifest.json` contains accepted multi-execution variants
only, with source IDs, recipe, current checks and hashes. Export verifies those
checks and files; it does not call the authored-reference task contract again.
Every receipt pins the implementation. After code/contact-semantic changes,
create a new version and revalidate evidence rather than reuse old pass flags.
Retained old poses may pass fresh verification without being regenerated.
Intermediate results are not training-ready or dynamic-feasibility certificates.

The old `scripts/reassess_rollout_contacts.py` command and
`contact_partial_order_acceptance_v1` reports are historical authored-contract
diagnostics. They are not admission/export entrypoints for this standard.
Completion-only inputs retain an explicitly `legacy_unverified` adapter; mixed
contact/completion collections must be separated instead of silently routing
contact tasks into it.

## Staged aggregation of retained rollouts

Historical v9 results use `population_information_consensus_v9`; current collections
use the v10 joint continuous registration described above. Event structure comes
from the retained new executions, not the original unedited reference contract.
The existing full-body fit remains in the staged pipeline; Adam adjusts motion
variables only and never trains the policy.

The goal is to extract consistent, denoised information from valid execution
observations. Population disagreement does not make an input trajectory wrong.
Do not use consensus reports to blacklist sources: check transformed outputs for
lost shared information and newly introduced variation instead.

1. Retain complete executions without recorded non-timeout termination. This is
   execution evidence, not an independently verified task-success label. Group
   different per-part surface itineraries separately; groups with fewer than
   three executions remain explicitly unaggregated and their sources retained.
2. Extract episodes from actual Newton primary-top-face contacts, using the
   existing three-sample stable-evidence/dropout policy without altering labels.
   Duration-scaled soft sequence correspondence avoids pairing unrelated climb
   and descent occurrences merely to maximize matched counts. Ongoing contact
   can realize repeated confirmations unless shared release evidence requires
   reacquisition.
   Strict-majority episodes form the common structure. Equal execution votes
   describe endpoint ordering and overlap. Cross-part onset/onset and end/end
   variations remain diagnostics, not task rules. Shared overlap and per-part
   sequence information are checked; missing events are counted once, without
   multiplying failures through every incident relation. Original action IDs, support
   owners, and reference deadlines do not participate.
3. Align motions to the consensus clock, then reuse local ordered pose matching.
   A representative supplies pose initialization, not event authority. Use one
   bounded continuous time map for every body channel and requery Newton after
   transport. Smooth clock rates at the existing evidence resolution, retaining
   endpoints and monotonic rate bounds; record the resulting event residuals.
   Check each source's observed consensus events for transport damage; minority
   incidental contacts are not promoted to hard requirements. Do not reject
   sources merely because they differ from population majority events. If a warp
   damages stable information, progressively reduce it toward identity and retain
   the source; record the applied clock and residual explicitly.
4. Reuse SO(3) residual fusion, smoothly interpolated whole-body robust weights,
   second-difference smoothing, and the established full-body FK fit. Whole-signal
   smoothing also removes representative noise left in residual-only smoothing:
   use a continuous quaternion lift, normalize rotations, preserve endpoints,
   and requery each candidate. Source-baseline fallbacks are explicitly separate
   from accepted multi-execution consensus outputs. Native
   contact guidance is compiled from those same population episodes, with equal
   execution votes on measured native event depth. A common episode remains an
   optimization target even when timing variation prevents a per-frame majority.
   Release guides come only from shared observed release evidence, never from a
   majority noncontact timestamp. Guide intervals are soft fitting intent, not
   new contact truth or acceptance deadlines. Observed native penetration is a baseline for
   excess-penetration guidance, not automatically treated as an error. Actual
   activation/allocation and margins remain unchanged. The full-body fit uses
   C2 spline corrections of a denoised seed, with fixed endpoints, to avoid
   reintroducing independent per-frame noise; the original per-frame fitter
   remains available through its unchanged default API. Unresolved outputs receive
   bounded local denoising and spline correction through compact C2 windows
   around measured violations or missing events. Poses outside these windows
   remain exact, and every candidate is audited again with actual Newton contacts.
   Retain the previous candidate and
   representative fallback, with the selected stage explicitly recorded. Separately
   audit joint-bounded versions of candidates with joint-limit overshoot; never
   silently replace poses or relax native contact/continuity checks.
5. Requery each fusion/smoothing stage, checking realization of population
   information, joint limits, and introduced local variation. Native executions
   define the corresponding-phase envelope. Median/MAD deviations are noise
   diagnostics, never errors in those source observations. Envelope exceedance
   identifies new variation for processing, not a physical infeasibility proof.
   RMS remains a comparative metric. Relabel the selected output again.

`event_consensus.json` records episode voters, median intervals, endpoint
relations, minority episodes and disputed relations. `alignment_checks` records
source-consensus differences separately from each source's own event preservation.
`stages/`, `source_continuity.json` and `optimization.json` locate failures by
processing stage. Distinct surface-itinerary strategies are processed separately
under `strategies/`; the primary result records all variant locations.

Old contract-based admission results cannot be reused as new admission decisions;
exact pose/provenance-matching native observations can be reused. Prior stage
poses are also retained as candidates and freshly queried with Newton.
All outputs are separate and remain `training_ready=false`. Event consensus and
fixed-q contact checks do not establish task success or dynamic trackability.

The 2026-10-16 v9 validation reprocessed all six remaining continuity cases:
all retained their population events and passed the unchanged source-envelope
checks. Five selected the full-body fit; one selected the compact local smoothing
stage. Six previously accepted cases also passed a complete pipeline regression,
and all 53 previously accepted outputs passed a separate unchanged-pose audit.
The combined 59-case manifest is
`tmp/native_information_consensus_v9_validated59_20261016/dataset_manifest.json`.
It records each output's own recipe and verification receipt: 12 v9 outputs and
47 retained v8 outputs, rather than claiming that every pose was regenerated.
The related suite passed 103 tests. Copied receipts and multiple surface-strategy
results resolve event and alignment evidence from the selected motion's artifact
records, rather than assuming evidence was copied beside the new receipt.

## Continuous boundary conditions and full generation regression

Global and local denoising solve the second-difference regularized signal with
fixed outside samples, endpoint positions and first sampled velocities. Local
support extends into the motion at boundaries instead of shrinking to the nearest
endpoint. A smooth support envelope bounds subsequent spline corrections; it is
not multiplied into a separately filtered signal. All body channels share support.
New poses still require fresh Newton labels and event/continuity/joint audits.

Use `scripts/reaggregate_rollouts.py --rebuild --reuse-results OLD_OUTPUT ...`
to exercise the current generator for every selected case. Historical evidence
is re-audited and old poses remain eligible incumbents; a previous accepted flag
does not skip the new generation path in this mode. Without `--rebuild`, checked
completed outputs can be admitted without rerunning generation. Selectors always
limit the declared output scope.

Incremental export writes `dataset_manifest.pending.json`; only a completed
processing scope writes the immutable `dataset_manifest.json`. This avoids the
old finalization error where a mutable intermediate occupied the final name.
A finished manifest cannot be replaced by an incomplete run.

Continuity schema `transported_source_information_continuity_v3` compares every
output phase with the actual, freshly Newton-observed source motions on that
same clock. The derivative chain rule separates legitimate clock-induced
acceleration from additional fusion variation. Original-clock diagnostics are
retained and can still flag a change; they do not redefine a smooth time change
as a pose jump. An inserted spike still fails the same per-channel envelope.
A legacy clock cannot acquire continuous certification merely by rechecking its
final pose. Full rebuilding provides that evidence; historical poses may compete
as candidates only inside a newly certified generation run.

The default reprocessing policy audits an existing multi-execution consensus
before generating new poses. If its clocks are already certified, independently
reconstruct and certify them from the stored controls, recheck the actual event
evidence, continuity and joint limits, and retain the unchanged output on success.
Legacy results lacking continuous-clock evidence first receive new population
registration and actual transported-source observations. Their final pose is
then audited on that clock; a prior pass flag cannot supply missing evidence.
Both reuse paths require at least three distinct contributors from the same
source group. A representative execution remains diagnostic only.

Historical candidate reuse is bounded to the final consensus and at most one
useful intermediate, deduplicated by pose. Do not recursively replay stages
accumulated across earlier runs. An accepted unchanged consensus skips new
denoising and pose fitting, preserving its existing alignment and smoothness.
Otherwise try the established residual fusion and smoothing candidates; only
unresolved cases need the full-body fitter and local refinement. Every newly
materialized final motion still receives complete fresh Newton labels and the
same event, continuity and joint-limit gates. `--rebuild` explicitly exercises
the generation and fitting path even if an incumbent already passes.

Reprocessing defaults to `--case-workers 4`: independent cases on the same terrain
overlap source reads, continuous registration, audits, denoising and fitting. They
share one initialized Newton model. An owner queue routes every solver pass and
CPU audit to the main thread that initialized Newton, with explicit CUDA event
handoffs; raw solver storage is cloned before it can be overwritten. Worker
failure drains queued queries before joining other workers, preventing a shutdown
deadlock. Each case keeps its own routing/FK/optimizer state. CPU Direct Newton
FK export also runs through this owner queue: it shares Warp's native runtime
with the GPU solver, so a separate export mutex is insufficient. Its mutable
process-wide model cache is never entered concurrently.
Progress is published by the coordinator from completed futures, never by reading
another case's unfinished receipt. A worker error propagates and does not become
a pass. `--case-workers 1` uses the same path without case concurrency.

Repeat `--reuse-results` in priority order to retain a partial newer run and use
older completed evidence for remaining cases. Certified completed results are
independently rechecked and admitted without regenerating poses. Missing legacy
clock evidence routes directly to population registration instead of a futile
CPU-only admission attempt. Source observation caches are resolved from pinned
artifact records, including historical directories and strategy groups; exact
source poses, actual-model provenance and file hashes must match. A mismatched
pose/model is a cache miss, while a changed pinned file is an error. Candidate
reading loads only requested pose/progress fields, leaving unused raw contact
arrays compressed. Output receipts record source-cache hit counts.

The final marker/contact fitter uses the same boundary contract as denoising:
quintic correction splines with clamped endpoint derivatives and null-space
constraints on first/last sample velocities. It no longer introduces a natural
spline endpoint velocity after earlier stages have preserved the initial state.

Accepted executions can be published as a separate segmentation version with
`scripts/publish_refined_contact_dataset.py`. Supply `--source-manifest`,
`--dataset-root`, `--baseline-index`, `--version`, and `--display-name`.
CPU publication defaults to four workers and starts no training or simulator.
It independently copies full canonical motions and terrains, rematerializes
current shared Newton primary-face task labels while preserving raw solver
evidence, then cuts the complete trajectory at its own observed touchdown
groups. It retains the entire prefix and tail; adjacent inclusive intervals
share an endpoint sample. Stability/grouping defaults reuse the existing
touchdown policy and impose no minimum segment length.

The new version stores `manifest.json`, `index.json`, event JSONL, scene geometry,
per-motion receipts, and `comparison.json` matching whole motions to the direct
generation baseline. Original frame boundaries and contact roles are not reused.
All source acceptance and implementation fingerprints are checked. Explicit
user acceptance remains distinct from automatic passage and cannot waive the
Newton event, clock, asset, or joint-limit gates. Only a fully verified release
is registered in `catalog.json`; `--resume` verifies completed files against the
same immutable publication plan. Publishing does not switch training manifests.
