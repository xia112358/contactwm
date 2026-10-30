# Policy Ref V1

`policy_ref_v1` is the single cross-repository reference-motion contract.
It means the reference object consumed by the Holosoma WBT policy. All contact
editing, IK, augmentation, Predictor training, and Infiller generation steps serve this
format.

## Required Fields

Every `policy_ref_v1` NPZ must contain:

```text
fps
joint_names
body_names
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

Expected shapes:

```text
joint_pos       [T, 7 + J]
joint_vel       [T, 6 + J]
body_pos_w      [T, B, 3]
body_quat_w     [T, B, 4]
body_lin_vel_w  [T, B, 3]
body_ang_vel_w  [T, B, 3]
joint_names     [J]
body_names      [B]
fps             scalar
```

For current G1 WBT data, `joint_pos` is `[T, 36]` and `joint_vel` is `[T, 35]`.
The first seven `joint_pos` dimensions are root xyz and root quaternion, followed
by 29 actuated joints. The first six `joint_vel` dimensions are root linear and
angular velocity, followed by 29 actuated joint velocities.

`body_quat_w` is kept in the same convention used by the policy motion loader.
Canonicalization normalizes quaternions but does not reinterpret their ordering.

## Optional Fields

The canonicalizer preserves optional fields when they can be read safely:

```text
contact_force_part_*
raw_contact_*
contact metadata
motion_edit_generation_metadata
source_* metadata
```

Optional fields are for contact analysis, segmentation, augmentation, debugging,
and manifests. They are not the trajectory generator target interface.

## Pipeline Rule

The only artifact passed between repositories as a motion reference is
`policy_ref_v1`:

```text
motion_edit generated raw npz
-> canonical Newton FK
-> policy_ref_v1 npz
-> Predictor + Infiller generation and Newton validation
-> policy eval
```

Intermediate formats are allowed inside one tool, but they must be canonicalized
before entering the next stage.

## Canonicalization

Use the canonical-asset Newton FK path documented in the repository README
(`scripts/canonicalize_omniretarget_newton.py`) and validate the motion manifest
with `scripts/check_motion_manifest.py`. The retired codec-specific
canonicalizer is no longer an entrypoint.

References must retain the fields and quaternion conventions above. Never
reuse source force or raw-contact labels for a newly generated trajectory;
query Newton again and preserve its provenance.
