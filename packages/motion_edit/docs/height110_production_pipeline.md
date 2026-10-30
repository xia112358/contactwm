> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Climb00 Height 1.10 Production Pipeline

This is the retained production route for the first height-edited policy:

```text
original force-free canonical motion
→ self-collision-enabled base WBT policy
→ parallel Newton policy rollout recording
→ one rollout source (median pose, measured force, raw contacts)
→ contact layer extracted from that same source
→ validated ContactEditPlan
→ task-variant surface translation
→ contact-Laplacian semantic curves
→ Newton local contact patches
→ PyRoki trajectory IK with Newton full-body penetration barriers
  and active foot/hand/knee signed-distance similarity
→ direct Newton FK canonical motion
→ fine-tune the same self-collision policy on the augmented motion
→ Newton execution produces the augmented motion's real force
→ full-start acceptance evaluation
```

Self-collision is enabled in the base policy, rollout collection, augmentation
IK, fine-tuning, and acceptance evaluation. It is never introduced as a later
repair stage. The collision contract preserves the source rollout's
per-body/per-surface soft signed distance instead of enforcing an arbitrary
penetration depth:

```text
similarity weight       25
deeper-than-reference   100
maximum refinements       1
```

The current reproduction solved all 1005 frames and had no unresolved semantic
or contact targets. All 75 rigid patches were bound from Newton raw contacts,
with no fallback patches. Maximum environment penetration was 1.371 mm, of
which 0.165 mm was deeper than the source reference. Contact target error was
0.389 mm mean / 17.712 mm max; semantic target error was 6.891 mm mean /
61.337 mm max. Filtered self-collision penetration was zero.

The collision objective is layered. Newton checks every unfiltered robot
collision body, including all hip-pitch/roll/yaw geometry. Every body receives
a one-sided no-new-penetration barrier. Physical foot, hand, and knee bodies
receive an additional priority multiplier when they are not the active
reference body. Active contact tracking remains restricted to the mature
six-part contract:

```text
LF / RF  -> left/right ankle_roll_link
LH / RH  -> left/right sphere_hand_link
LK / RK  -> left/right knee_link
```

For those six active reference bodies, the contact patch target, signed-depth
similarity, and deeper-than-reference weight are unchanged. Other bodies never
become contact targets: they are only prevented from penetrating. No separate
hip-protrusion Laplacian node or IK target is used because the full
`hip_pitch_link` collision mesh is already covered by Newton.

## 1. Build the rollout source

Record several complete executions of the same motion with the accepted
self-collision-aware policy. Merge them once into `source_motion.npz`.
Kinematics use the timestep-aligned median, force uses the measured world-vector
median, and external raw contacts are grouped by stable Newton labels. No
dynamic replay or selected environment becomes an authority.

```bash
cd /home/xiaz/somaforge

RUN_DIR="$PWD/tmp/climb00"
SOURCE="$RUN_DIR/source_motion.npz"

python scripts/build_rollout_source.py \
  --recording "$RUN_DIR/rollouts.npz" \
  --reference-motion runtime/current/motions/climb_00_z_scale_1.0.npz \
  --checkpoint runtime/current/holosoma/logs/WholeBodyTracking/<run>/model_06000.pt \
  --output "$SOURCE" \
  --work-dir "$RUN_DIR/work" \
  --device cpu

python -m motion_edit.cli import-force-proto \
  --motion-dir "$RUN_DIR" \
  --pattern source_motion.npz \
  --motion-id climb_00 \
  --layer-name climb00_source \
  --source rollout_source \
  --surface-catalog \
    packages/motion_edit/data/layers/contact/climb00_source/surfaces/climb_00.jsonl \
  --max-surface-distance 0.08
```

The plan's `source_motion_path` must point to `SOURCE`. The legacy
`metadata.contact_force_source_path` may be omitted; when present it must point
to exactly the same file. Every augmentation starts from this source, never
from another generated variant. The contact layer must also be extracted from
this file so its phases and stable Newton shape IDs remain aligned.
Surface-aware import is mandatory for a surface-follow task. It assigns anchors
from raw Newton contact points, filters edge/outside fragments, and fails unless
every retained anchor is bound. An unbound hand contact is never kept as a
fixed world target, because raising the obstacle would turn that stale target
into a side/interior contact.

## 2. Generate edited kinematics

```bash
cd /home/xiaz/somaforge
export SOMAFORGE_ROOT=/home/xiaz/somaforge
export PYTHONPATH="$PWD/packages/motion_edit:$PWD/packages/somaforge_core:$PYTHONPATH"

RUN_DIR="$PWD/tmp/climb00"
PLAN="$RUN_DIR/height110.json"

source scripts/source_isaaclab3_newton_setup.sh
python \
  scripts/generate_contact_aware_edited_motion.py \
  --plan "$PLAN" \
  --output "$RUN_DIR/final_motion.npz" \
  --intermediate-dir "$RUN_DIR/work" \
  --ik-collision-reference-cache \
    "$RUN_DIR/source_collision_reference.npz" \
  --ik-conda-env env_somaforge \
  --newton-device cpu \
  --overwrite
```

The generator fails closed if pose, force, or raw contacts do not come from the
same source file.

For the first variant of one source motion, the linear semantic proxy now uses
three unconstrained convergence iterations. This produces the same converged
dual-Laplacian solution as the former eight-iteration, 5 cm trust-step
configuration while avoiding repeated solves that only advanced toward the
same linear optimum.

Subsequent variants with the same source motion, anchor set, contact frames,
objective weights, and proportional edit displacement can reuse that solved
deformation:

```bash
python scripts/generate_contact_aware_edited_motion.py \
  --plan "$NEXT_DERIVED_PLAN" \
  --output "$NEXT_RUN_DIR/final_motion.npz" \
  --intermediate-dir "$NEXT_RUN_DIR/work" \
  --semantic-proxy-basis \
  "$RUN_DIR/work/final_motion.pyroki_fk_preview.semantic_task_proxy.npz" \
  --ik-collision-reference-cache \
  "$RUN_DIR/source_collision_reference.npz" \
  --ik-conda-env env_somaforge \
  --newton-device cpu \
  --overwrite
```

Reuse is fail-closed: a different source, objective, anchor set, frame range,
non-proportional displacement, unconverged basis, or non-empty pose edit raises
an error instead of silently approximating the result. In the measured
climb00 height pair, height110 served as the basis for height090 with scale
`-1`. The shared collision cache is separately guarded by the source-terrain,
source-qpos, and canonical-URDF hashes. On the current 1005-frame acceptance
run, basis reuse plus a warm collision cache took 21.69 seconds; cache miss and
hit final joint/body arrays were identical.

For multiple independent variants, use the bounded batch wrapper rather than
raising the inner solver's thread count:

```bash
python scripts/generate_contact_aware_batch.py \
  --plan-manifest "$RUN_DIR/plans.json" \
  --output-dir "$RUN_DIR/motions" \
  --work-dir "$RUN_DIR/batch_work" \
  --semantic-proxy-basis \
    "$RUN_DIR/work/final_motion.pyroki_fk_preview.semantic_task_proxy.npz" \
  --collision-reference-cache \
    "$RUN_DIR/source_collision_reference.npz" \
  --workers 2 \
  --overwrite
```

The manifest is a JSON list of plan paths or objects with `plan_path`,
optional `output`, and optional per-plan `semantic_proxy_basis`. Two workers
are the measured default for the 16-core workstation: two identical
1005-frame jobs completed in 27.86 seconds and matched the serial result
exactly. Four workers caused severe JAX/Newton thread oversubscription and are
not the production default. When the shared collision cache is absent, the
wrapper completes one job first to populate it before starting parallel jobs.

The final motion must report:

```text
environment_collision_backend = newton_soft_signed_distance_integrated_frame_ik
environment_collision_contract = source_rollout_soft_signed_distance_similarity
environment_collision_fullbody_deeper_weight_multiplier = 4
environment_collision_contact_capable_deeper_weight_multiplier = 4
environment_collision_active_contact_deeper_weight_multiplier = 1
least_squares_failure_count = 0
unresolved_semantics = []
unresolved_contacts = []
```

The canonical output contract is:

```text
joint_pos        [1005, 36]
joint_vel        [1005, 35]
body_pos_w       [1005, 53, 3]
body_quat_w      [1005, 53, 4]
body_lin_vel_w   [1005, 53, 3]
body_ang_vel_w   [1005, 53, 3]
joint_names      [29]
body_names       [53]
```

All numeric arrays must be finite. The body poses must equal direct Newton FK
of `joint_pos`. The file must carry the canonical `robot_asset_json` and
direct-Newton kinematics provenance, and must not carry stale source
`contact_force_part_w`, `contact_force_provenance_json`, or `raw_contact_*`.
Files in `work/` are diagnostic intermediates; they are not the final motion.

## 3. Fine-tune and evaluate

Fine-tuning has two authorities:

```text
canonical augmented motion manifest ─┐
                                     ├→ self-collision policy fine-tune
accepted self-collision checkpoint ──┘
```

The checkpoint is already capable of producing contact force in Newton. The
augmented NPZ does not contain copied or replay-generated force. Its new force
is produced by the fine-tuned policy during Newton execution.

```bash
python scripts/train_motion_edit_policy_finetune.py \
  --headless \
  --device cuda:0 \
  --checkpoint \
    runtime/current/holosoma/logs/WholeBodyTracking/\
20260727_085520-g1_29dof_wbt_single_climb00_completionema_horizon50_ncon160_from4k_to10k-locomotion/\
model_06000.pt \
  --motion-manifest "$POLICY_DIR/manifest.json" \
  --output-dir "$POLICY_DIR/train_logs" \
  --project Height110PolicyFinetune \
  --name climb00_height110_selfcollision_completionema \
  --num-envs 4096 \
  --iterations 10000 \
  --save-interval 500
```

The retained sampler parameters are:

```text
reset_sampler                          hotspot_failure_window
failure_window_pre_frames              50
failure_window_post_frames             20
failure_window_before_prob             0.7
failure_window_success_horizon_frames  50
hotspot_failure_uniform_mix            0.3
hotspot_failure_decay                  0.995
probe_completion_alpha                 0.02
probe_env_per_motion                   10
probe_uniform_mix                      0.4
nconmax_per_env                        >= 160
njmax_per_env                          >= 1024
```

The training entry point fails closed if the checkpoint does not have
`robot.asset.enable_self_collisions=True` or lacks the retained Newton contact
capacity.

Run `scripts/eval_motion_edit_acceptance.py` from frame zero against the
resulting checkpoint. File-level acceptance is not sufficient: also inspect
the policy execution's phase-specific root height, knee flexion, contact masks,
force timing, and force magnitude.

## 4. Regression and UI checks

Run the source regression in the pinned GA environment:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m pytest -q \
  packages/motion_edit/tests \
  packages/somaforge_core/tests \
  src/holosoma/holosoma/agents/modules/tests/test_censored_normal.py \
  src/holosoma/holosoma/agents/ppo/tests/test_kl_early_stop_ppo.py
```

The complete command above must pass before the route is admitted. Launch the
current single-port editor and use its recent list for registered motions:

```bash
packages/motion_edit/motion-edit contact-editor --port 8094
```

Before admitting a generated reference, inspect the complete motion in the
editor/player: the root and joint order must be correct, the timeline must
advance through all frames, and no quaternion-order, body-mapping, foot-pose,
or contact-phase discontinuity may be visible.
