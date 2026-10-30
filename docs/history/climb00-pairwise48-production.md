> Historical record: the GMVQ/HyAR code and online entrypoints below have been retired.
> Predictor + Infiller now provide the generation architecture. Old artifacts remain
> preserved locally; commands below are not current runnable instructions.
> Current entrypoints: [repository README](../../README.md); current Predictor:
> [baseline pointer](../../baselines/current.json). Historical recipes remain archived.

# Climb00 Pairwise48 Production Result

This document is the single path registry and runbook for the accepted
`climb_00` contact-aware augmentation result. Files below use the canonical G1
asset:

```text
src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf
```

## Stable artifacts

```text
runtime/current/manifests/climb00_pairwise48.json
runtime/current/motions/climb00_pairwise48/                 48 final Newton-FK motions
runtime/current/motions/climb00_pairwise48_sources/         48 PyRoki source previews
runtime/current/motion_edit/plans/climb00_pairwise48/       48 validated edit plans
runtime/current/motion_edit/surfaces/climb00_pairwise48/    per-plan surface catalogs
runtime/current/motion_edit/sources/climb00_pairwise48/     four incidence-adjusted source motions
runtime/current/terrains/climb00_pairwise48/                5 top-surface terrains
runtime/current/models/climb00_pairwise48/wbt.pt             accepted WBT checkpoint
runtime/current/models/climb00_pairwise48/gmvq.pt            self-contained GMVQ runtime bundle
runtime/current/logs/climb00_pairwise48/                     future run output
```

Accepted WBT provenance:

```text
checkpoint iteration: 9000
source: runtime/current/logs/climb00_pairwise48_continue_from8k/Climb00Pairwise48Finetune/20260802_103601-pairwise48_from_8k_1000iter-locomotion/model_09000.pt
sha256: 0578de376bb2d464303c07d9bb0e73b80b337821ab01b2edb816d236c2c24b1a
fixed acceptance: 480/480 (48 motions x 10 repeats)
previous iteration-6500 checkpoint: runtime/current/models/climb00_pairwise48/wbt_model06500.pt
```

The manifest binds every final motion to its source preview, edit plan, and
terrain with SHA256 digests. The promoted WBT checkpoint points to this stable
manifest. The GMVQ bundle embeds the accepted GMVQ codec, code selector, theta
selector, start-conditioned decoder, scan grid, duration prior, and transition
mask; its recorded component paths use `embedded://` identifiers and are not
runtime dependencies.

The bootstrap motion is only the reset pose and schema source:

```text
runtime/current/motions/climb00_pairwise48/pair48_01_h090_dp50p000_ap10_lu_m050.npz
```

## Production flow

```text
canonical force-free climb00 motion
-> self-collision-enabled WBT policy
-> parallel Newton policy rollouts with measured raw contacts and force
-> one fused rollout source
-> surface-constrained ContactEditPlans
-> contact-Laplacian semantic curves
-> Newton robot-local rigid contact patches
-> PyRoki trajectory IK with penetration and self-collision costs
-> direct Newton FK canonical edited motions
-> WBT fine-tune on all 48 motion/terrain pairs
-> contact/support event atom segmentation
-> GMVQ code + theta + start-conditioned decoder
-> online atom-boundary decoding from terrain scan and current robot state
-> WBT policy executes every decoded reference frame in Newton
```

## Stage contracts

| Stage | Authoritative input | Operation | Authoritative output |
| --- | --- | --- | --- |
| 0. Asset | canonical sphere-hand URDF | asset fingerprint validation | one robot joint/body contract |
| 1. Reference | force-free `climb_00` motion | direct Newton FK canonicalization | `[T,36]` qpos, `[T,35]` qvel and complete body FK |
| 2. Base policy | canonical reference and terrain | self-collision-enabled WBT training | policy able to execute `climb_00` |
| 3. Force source | repeated policy rollouts | align, reject failures, and fuse pose/contact/force samples | one rollout authority with raw Newton contacts and measured 8-part force |
| 4. Augmentation | rollout authority, surface catalog, edit plan | contact-Laplacian curves, rigid patch binding, PyRoki IK, Newton collision costs, direct Newton FK | 48 canonical edited kinematic motions |
| 5. Tracking policy | 48 motions and five matched terrains | WBT fine-tune through iteration 9000 with self-collision still enabled; fixed acceptance 480/480 | `wbt.pt` |
| 6. Skill model | support/contact-transition atom segments from the 48 motions | GMVQ codec, code selector, theta selector, start-conditioned decoder | self-contained `gmvq.pt` |
| 7. Runtime | `gmvq.pt`, current terrain scan and current robot state | decode at atom boundaries; WBT tracks every decoded frame | Newton execution state and newly computed physical contact force |

The four stable incidence source motions are under:

```text
runtime/current/motion_edit/sources/climb00_pairwise48/
├── incidence_m10/source_motion_approach_front.npz
├── incidence_p0/source_motion_approach_front.npz
├── incidence_p10/source_motion_approach_front.npz
└── incidence_p20/source_motion_approach_front.npz
```

Each contains the complete pose trajectory, body FK, `contact_force_part_w`,
contact masks, Newton raw body/shape/point/normal/force channels, and canonical
robot identity. These files are the contact/force authority for rebuilding the
48 edits. The corresponding final files under
`runtime/current/motions/climb00_pairwise48/` contain kinematics only.

## Runtime boundary

At reset, the bootstrap motion contributes only the initial robot pose and the
motion schema. It is not played as the future reference. At each atomic-action
boundary, the GMVQ command consumes:

```text
158-point local terrain-height scan
+ root position/quaternion/linear velocity/angular velocity
+ joint_pos [36]
+ joint_vel [35]
+ previous discrete code through the transition mask
```

The code selector chooses the support/contact transition. The theta selector
chooses continuous geometry, posture, and amplitude. The duration prior chooses
the atom length. The start-conditioned decoder produces an absolute-shape
`joint_pos/joint_vel` buffer whose first frame is conditioned on the current
robot state. Torch FK converts that buffer into WBT body targets. Neural
decoding runs only at atom boundaries; WBT control and Newton physics run every
frame.

No future source trajectory, edit-plan label, target-height scalar, or copied
contact force is supplied to runtime inference. The current height scan and
proprioception must determine the next action. Contact force is recomputed by
Newton from the executed state and velocity; it is never transplanted from the
source rollout into an edited motion.

## Production boundaries

- `semantic_task_proxy.npz`, task-space specs, PyRoki previews, and collision
  caches are intermediate generation artifacts, not policy references.
- The edited motion NPZ is a kinematic target, not a force recording.
- The embedded GMVQ models generate references; they do not replace WBT.
- WBT produces actions every control frame; Newton/MJWarp produces the actual
  motion, contacts, and force.
- Offline decoded NPZ files and historical selector experiments are diagnostics,
  not the online runtime route.
- No production config or command may depend on `tmp/`.

The 48 edited NPZ files are kinematic references and intentionally contain no
copied force channels. Real force exists only when the WBT policy executes the
reference in Newton. Self-collision remains enabled from the base policy through
rollout collection, fine-tuning, and evaluation.

## Validate stable assets

```bash
cd /home/xiaz/somaforge
export PYTHONPATH="$PWD/src/holosoma:$PWD/packages/gmvq:$PWD/packages/somaforge_core:$PYTHONPATH"

conda run --no-capture-output -n env_somaforge python - <<'PY'
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest

manifest = load_motion_terrain_manifest(
    "runtime/current/manifests/climb00_pairwise48.json"
)
assert len(manifest["motion_files"]) == 48
assert len(manifest["terrains"]) == 5
print(manifest["path"])
PY
```

## Headless smoke test

The main entry point first lets Isaac Lab AppLauncher parse official simulator
arguments, then parses the GMVQ runtime arguments with tyro:

```bash
cd /home/xiaz/somaforge
source scripts/source_isaaclab3_newton_setup.sh

python -m holosoma.eval_gmvq_reference \
  --checkpoint runtime/current/models/climb00_pairwise48/wbt.pt \
  --bundle runtime/current/models/climb00_pairwise48/gmvq.pt \
  --bootstrap-motion \
    runtime/current/motions/climb00_pairwise48/pair48_01_h090_dp50p000_ap10_lu_m050.npz \
  --max-steps 500
```

## Interactive Kit run

Visualization uses the single supported switch `--visualizer kit`:

```bash
cd /home/xiaz/somaforge
source scripts/source_isaaclab3_newton_setup.sh

python -m holosoma.eval_gmvq_reference \
  --checkpoint runtime/current/models/climb00_pairwise48/wbt.pt \
  --bundle runtime/current/models/climb00_pairwise48/gmvq.pt \
  --bootstrap-motion \
    runtime/current/motions/climb00_pairwise48/pair48_01_h090_dp50p000_ap10_lu_m050.npz \
  --max-steps 2000 \
  --visualizer kit
```

Do not substitute an old G1 URDF, a `tmp/` checkpoint, an intermediate
`semantic_task_proxy.npz`, or an offline decoded motion for these production
artifacts.
