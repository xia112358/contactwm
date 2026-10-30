# ACP reproduction on SomaForge

Implementation reference: [Humanoid Mobility via Attention-based Contacts](https://arxiv.org/html/2610.31608v1),
especially Eq. (2), Eq. (3), Table 3 and Tables S2--S6. The project page still
marks its code release as TBD when inspected on 2026-10-23.

This is an independent **single-task implementation of stages 1 and 2** using
the maintained SomaForge simulation stack. It is not the authors' implementation,
a completed reproduction of their multi-skill results, or a deployment-ready policy.
Stage 3, procedural goal-reaching terrains, separate reference/goal critics, and
the authors' trained LiDAR mapping model are not included in this slice. The train
entrypoint rejects unsupported stage numbers rather than silently substituting
reference tracking for reference-free fine-tuning.

## Reused assets and explicit differences

- The only robot is the canonical spherehand G1, resolved through the existing
  asset contract. URDF, collisions, margins and existing policy assets are preserved.
- The local robot has 29 joints; the paper controls 23. `ContactPlanner(23)` has
  exactly **786,993 parameters**, matching Table S3. The running 29-joint variant
  has **798,513**: only the first proprioception encoder's input width changes
  from 442 to 532. It does not claim exact 23-joint morphology/control equivalence.
- The first task uses the canonical `climb_00` terrain and an actual rollout of
  the explicitly selected existing dense expert. Its executed trajectory becomes
  the reference. This differs from using the paper's original retargeted dataset.
- Simulation runs at 200 Hz and control at 50 Hz, using the existing Isaac Lab /
  Isaac Sim / Newton-MJWarp environment. It does not claim the same unspecified
  simulator revisions or solver settings as the paper.
- Local training scales are intentionally small. Paper batch sizes and training
  durations are not established by these validation runs.

## Components

`src/holosoma/holosoma/acp/` contains:

| File | Responsibility |
| --- | --- |
| `models.py` | Table S3 CNN, FiLM, 32-head attention, occupancy classifier, terrain-point decoder, Eq. (2) and schedules |
| `contacts.py` | Nine-body aggregation of actual allocated native constraints and decoded world forces |
| `labels.py` | Episode-bounded, known-mask-aware future contact-onset labels |
| `command.py` | Validated reference labels and backward state-initialization curriculum |
| `observations.py` | Five-step proprioception, body-yaw scans, goal and privileged critic inputs |
| `env.py` | Native WBT integration, issued-plan storage and successful-rollout collection |
| `replay.py` | Bounded successful-episode replay balanced by motion/terrain tile |
| `rewards.py` | Reference kinematic/contact task terms and planner-contact transition |
| `ppo.py` | Existing PPO plus separate planner updates, save/load and stage transitions |
| `record.py` | Explicitly selected expert execution and raw contact/observation recording |
| `train.py` | AppLauncher-first independent stage-1/2 CLI |
| `actions.py`, `randomization.py`, `sensing.py` | Table S2 startup dynamics, substep actuator delay and delayed odometry |
| `evaluate.py` | Independent Stage-1 evaluation from frame zero with explicit completion accounting |
| `validate.py` | CPU real-data optimizer diagnostic, explicitly separate from policy evaluation |
| `mapping.py` / `perception.py` | Three-mode mapping, timestamped history fusion and synthetic LiDAR adapter |
| `fit_mapping.py` / `map_recording.py` | Offline mapping training, reconstruction and point-cloud/height-map export |

Shared code changes are narrow: face selection accepts an optional part count
(default six), native support selection accepts an optional body/part mapping,
and `MotionCommand` exposes an overridable reset-phase sampler (default uniform).
Existing six-body calls and ordinary WBT presets keep their defaults.

## Contact and observation contracts

Part order is `left_foot, right_foot, left_knee, right_knee, left_hand,
right_hand, left_elbow, right_elbow, pelvis`. Pelvis includes its contour collision
body. Hands and feet reuse canonical collision-body aliases.

Raw contact truth remains `CONSTRAINT && dist < includemargin` with allocated
`efc` rows. Unallocated active constraints fail explicitly. Self/cross-robot
collisions do not contribute. Shared primary horizontal upward face selection
continues to apply, including to elbow/pelvis contacts; raw excluded contacts are
preserved in the recorder. Force never creates activation. The paper's vertical
force magnitude >1 N and minimum ten control steps select supervision events
within those native contacts. They do not rewrite the saved activation stream.

Labels look ahead 75 control steps (1.5 s). Already-active left-censored segments
cannot create an onset. Unknown frames and insufficient future coverage mask
occupancy supervision instead of manufacturing negative examples. A target must
be inside the 1.6 x 1.0 m body-yaw scan footprint. The reference loader rejects
missing channels, incomplete recordings, wrong robot identity, wrong contact
semantics and non-50-Hz recordings.

The recorder preserves the established timing distinction: contact evidence is
from the last native solve and poses are post-integration. Metadata explicitly
sets `strict_pose_contact_same_state=false`; these samples are not represented
as a strictly same-state contact/pose query.

Tracker actor inputs are measured five-step proprioception, a three-component
goal, 9x7 heights and nine contact commands. Reference joint states/body poses
stay in rewards and the privileged critic. The critic uses current proprioception
without history and a 17x11 scan. The planner takes five-step proprioception, the
goal and 17x11 XYZ terrain points. New training defaults to `fused_raw`, sampling
actor/planner observations from synthetic LiDAR history. `fused_neural` requires
trained mapping weights; `ground_truth` retains direct initialized-terrain
queries and remains the implicit mode for old checkpoints. The privileged
critic always uses terrain truth. See [history-fused mapping](acp-mapping.md)
for implementation choices, limitations, training and export commands.

The aligned interface (version 2) has 558 actor inputs and 383 reference-critic
inputs for the canonical 29-joint robot. The critic receives noiseless current
proprioception, base linear velocity, 17x11 heights, reference kinematics and
nine reference contact flags/positions; it does not additionally receive the
actor's goal and issued contact commands. Actor and critic both use
512/256/128-unit ELU networks. The paper's 23-joint critic has 353 inputs.

Table S2 startup randomization covers leg mass scaling, added torso mass,
torso CoM offsets, armature, PD gains, joint zero offsets and 0--20 ms actuator
delay. Observation perturbations include proprioception, contact positions,
tracker/planner heights, goal direction and heading, and 50--100 ms odometry
delay. Odometry drift is a per-episode world-frame XYZ offset; poses are
interpolated at the control rate with wrapped yaw. The 5 ms physics grid uses
interval-averaged actuator targets when a delay crosses a substep. These are
explicit simulation approximations, not a reproduced hardware estimator.
Newton supports one friction coefficient per shape, sampled in [0.3, 1.2],
and restitution in [0, 0.5]; independent static/dynamic coefficients from the
paper cannot be represented by this backend. Ground friction remains 1.
Terrain queries use the delayed/drifted pose but still query the actual mesh;
the paper's LiDAR mapper remains absent. Startup dynamics are fixed during a
run; episode resets resample odometry drift and reference-state perturbations.

Stage 1 includes the Table 3 action rate, second action difference, squared
torque, torque-clipping gap, capped soft position/velocity limits and summed
nine-body acceleration penalties. Acceleration history is cleared on reset.
Contact-body reference positions are tracked even when not currently active;
planner targets alone are occupancy gated. Torque clipping is measured from
the actual PD command before and after the existing limiter. These changes
preserve the shared controller's applied torque and all collision assets.

## Explicit commands

Activate the existing environment from the repository root:

```bash
source scripts/source_isaaclab3_newton_setup.sh
```

Record a named expert and source motion into a new output directory:

```bash
python -m holosoma.acp.record \
  --checkpoint /path/to/canonical_dense_expert/model.pt \
  --manifest runtime/current/manifests/wbt_single_climb_00.json \
  --output-dir runtime/current/motions/acp/my_reference --headless
```

Recording never selects a policy automatically. `--allow-partial` produces an
explicit diagnostic recording; incomplete recordings cannot be stage-1 inputs.
New reference recordings also contain noiseless measured planner history,
goal and actual terrain scans for optimizer diagnostics. Source force labels are
not copied into a changed trajectory.

Train the first controller:

```bash
python -m holosoma.acp.train \
  --manifest runtime/current/motions/acp/climb00_native_reference_v5/manifest.json \
  --stage 1 --num-envs 8 --iterations 200 --save-interval 50 \
  --terrain-observation ground_truth --name climb00_pilot --headless
```

Stage 1 starts each environment's initialization phase at 0.9 and resets at
that phase. A completed, non-failed episode reaching at least 0.99 lowers its
phase by 0.05; failure raises it by 0.05, clipped to [0, 0.99]. Empty initial
resets do not advance the curriculum. Each environment has independent state,
even when environments share a motion. The state is checkpointed and restored
after the normal reset. Evaluation always starts at frame zero and freezes CSI.
Interface-1 checkpoints used a different critic and motion-level curriculum;
aligned training rejects them and must start from scratch. Same-version resume
requires the same environment count to restore per-environment curriculum.
Resume rebuilds the simulator and resamples startup dynamics; it restores
learning state and curriculum, not an identical physical trajectory.

The PPO defaults follow Table S6: 24-step rollouts, five epochs, eight minibatches,
initial learning rate 1e-4, target KL 0.02 and entropy coefficient 0.006 in Stage 1.
The paper uses 12,000 environments per worker. Local single-GPU runs are smaller
and do not establish equivalence to the paper's training scale or success rate.

Start joint planning from an explicit ACP controller checkpoint:

```bash
python -m holosoma.acp.train \
  --manifest runtime/current/motions/acp/climb00_native_reference_v5/manifest.json \
  --stage 2 --checkpoint /path/to/acp_stage1/model.pt \
  --terrain-observation ground_truth \
  --num-envs 8 --iterations 1000 --name climb00_planner --headless
```

Stage transitions reset the schedule counter. Same-stage resume continues at
the saved iteration + 1 and restores controller/critic, normalizers, optimizers,
curriculum, planner, planner optimizer and replay. Simulator/native solver state
is never bundled. Checkpoints retain milestones and the final checkpoint.

Stage-2 default transition starts are input iteration 2000 and reward iteration
4000, with duration 2000. These are explicit local choices because the paper
provides the schedule form without concrete start/duration values. Short runs
will not reach them. Override `--input-start`, `--reward-start`, `--anneal-steps`
deliberately; observation selection uses complete commands, not interpolated XYZ.

Only non-failed rollouts starting within the first 10% and reaching at least 90%
of a reference enter planner replay. At most the first 500 steps are recorded.
Up to `--record-envs` environments contribute recordings; this bounded resource
adaptation does not select actors' actions. Stage 2 uses ten Adam updates per
PPO iteration, 4096 sampled entries and eight accumulated sub-batches by default,
after 1000 replay entries. Global masked-loss denominators are shared across
sub-batches. Default capacity is 100,000 entries with episode FIFO eviction.
This single-task replay balances tiles; it does not implement Stage-3 quality
eviction or reference/goal partitions. Distributed ACP training is not validated.

Evaluate aligned Stage 1 independently, using deterministic policy means and
nominal dynamics/noiseless observations by default:

```bash
python -m holosoma.acp.evaluate --checkpoint /path/to/acp_stage1/model.pt \
  --episodes 10 --seed 42 --output-dir tmp/acp/my_stage1_evaluation --headless
```

The report records completion, failure reasons, root/contact errors and missed
or unexpected native contacts. It saves per-frame raw Newton pairs, post-physics
poses, the native shape/body binding and contact semantics for audit. Excluded
faces remain in the raw pairs. A budget cutoff never counts as success. With
`--randomization`, dynamics are sampled once for the physical evaluation
environment and episode drift is resampled; this is not a multi-seed dynamics
generalization estimate. All episodes still start at reference frame zero.

Use the standard native evaluator for interactive inspection:

```bash
python -m holosoma.eval_agent --checkpoint /path/to/acp/model.pt \
  --num-envs 1 --max-steps 1100 --visualizer kit
```

Stage-2 evaluation always selects planner contact commands, regardless of the
saved annealing counter. It still receives a reference-derived high-level goal
on the reference terrain; it is not reference-free Stage-3 deployment. Omit
`--visualizer` for headless use.

## Aligned Stage-1 validation, 2026-10-24

- Interface-2 training on actual Newton completed two PPO updates with eight
  randomized environments. Native mass writes were checked against the samples;
  the local pinned Warp build needs explicit Warp-array arguments for these
  property setters. No dependency or collision-asset changes were needed.
- Same-version resume continued at updates 2 and 3. Actor optimizer steps
  advanced from 80 to 120 and 160; observation counts advanced from 384 to
  576 and 768. Checkpoints retain finite model parameters, normalizers and
  independent eight-environment CSI state using the existing checkpoint format.
- Three independent nominal frame-zero trials of the four-update diagnostic
  policy all failed with `bad_tracking` after 33 control frames (0.66 s).
  This validates failure accounting; the diagnostic policy has not learned
  the complete 1005-frame reference.
- A 256-environment, eight-update profiling run completed and retained its
  milestones. It used approximately 1.1 GB GPU memory and about 10 seconds per
  update on the local RTX 4060 Ti. This scale is smaller than the paper's.
- Two randomized frame-zero evaluation trials cut off after five steps were
  correctly reported as truncated and unsuccessful. The selected ACP/shared
  CPU regression and existing PPO learning-state tests passed all 1353 cases.
- The aligned single-task training budget is 5000 PPO updates with 256
  environments (30,720,000 transitions), saving every 100 updates and at the end.
  It starts from scratch; the earlier interface-1 pilot is incompatible with
  aligned training. Completion of this budget does not itself establish task
  success: a later independent frame-zero evaluation is still required.
- Reports, exact checkpoint checks and runtime logs are under
  `tmp/acp_stage1_alignment_20261024/`; earlier failed diagnostics are retained.

## Historical interface-1 validation, 2026-10-23

- Complete `climb_00` dense-expert rollout: 1005 frames at 50 Hz, nine native
  contact channels and per-frame raw solver snapshots.
- The latest reference is `runtime/current/motions/acp/climb00_native_reference_v5/`.
  V1--V3 are retained failed diagnostics; V4 is the complete earlier reference
  without planner observation arrays. None were deleted or overwritten.
- There are native foot/hand/left-knee positives in this motion. No elbow,
  pelvis or right-knee positive examples occur, so this motion does not establish
  nine-body skill coverage.
- Stage-1 two-update PPO smoke test saved normal checkpoints; CSI moved from
  0.90 to 0.85 in that test.
- Stage-2 three-update smoke test loaded the stage-1 controller and exercised
  full planner-input and contact-reward transitions, using shortened diagnostic
  schedules. These runs do not establish learned successful planner rollouts.
- Same-stage Stage-2 resume restored the checkpoint and advanced from iteration
  2 to 3.
- Recorded-observation optimizer diagnostic: fixed-batch loss decreased from
  5.91823 to 5.82682 in twenty CPU Adam updates. The actual ten-update,
  eight-sub-batch planner training hook also ran. This uses a dense expert
  rollout to test wiring, not to claim successful ACP closed-loop behavior.
- The final targeted test suite passed 37 tests, including the exact Table S3
  parameter count, observed-surface decoder, finite gradients, masked-loss
  accumulation equivalence, censored labels, replay restoration, config
  round-trip, default six-body behavior and existing sampler checkpoint tests.
- Standard `holosoma.eval_agent` loaded the Stage-2 checkpoint, rebuilt the ACP
  environment and completed a four-step headless inference check.
- The first Stage-1 pilot completed 200 PPO updates with 8 environments and
  38,400 transitions. Milestones 0/50/100/150 and final iteration 199 were saved
  in `runtime/current/models/acp/ACP/20261023_050057-climb00_pilot_v1_stage1-locomotion/`.
  CSI expanded the allowed initialization range to phase zero; that is not
  evidence of full-motion execution.
- Independent noiseless, frame-zero evaluations failed at 33 frames for update
  100 and 32 frames (0.64 s) for the final checkpoint. The complete task was not
  learned in this pilot. These are individual diagnostic rollouts, not an
  estimated multi-trial success rate. No successful ACP-generated episode or
  trained closed-loop planner is claimed.
- The pilot summary and curves are `tmp/acp/stage1_pilot_report.json`,
  `tmp/acp/stage1_pilot_curves.png` and `.pdf`. Independent rollout reports and
  all native snapshots are under `tmp/acp/pilot100_evaluation/` and
  `tmp/acp/pilot199_evaluation/`.

Full logs, the optimizer report and subsequent pilot/evaluation reports are
under `tmp/acp/`. Ordinary model checkpoints are under
`runtime/current/models/acp/ACP/`. A short pilot is evidence of integration and
learning behavior, not convergence, held-out generalization or paper-level success.
