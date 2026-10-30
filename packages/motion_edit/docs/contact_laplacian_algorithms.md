> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Contact Laplacian Editing Algorithms

This document describes the two Laplacian-style residual families that should
coexist in the geometry stage of Motion Edit's contact-centered `generate-ref`
pipeline:

1. task-space contact-handle LTE over semantic keypoints;
2. fullbody interaction-mesh / q-space Laplacian refinement.

The goal is not to choose one family and delete the other. The long-term target
is a unified full-trajectory optimization where both temporal/task-space and
spatial/fullbody terms affect the same solve.

The target variable is the whole robot trajectory:

```text
Q = [q_0, q_1, ..., q_{T-1}]
```

The target solver optimizes `Q` jointly. Per-frame QP is insufficient as the
final algorithm because temporal Laplacian and q-smoothness terms couple
adjacent frames.

## Problem

Contact-anchor editing starts from a simple user operation:

```text
move this bound contact anchor 10 cm along its original surface
```

The edit is stored as a `ContactAnchorEditRecord` inside a
`ContactEditPlan`. That record is only an intent:

- anchor id;
- body;
- old and new contact position;
- tangent delta on the same surface;
- affected frames;
- surface id/object id;
- clamp/reject provenance.

It should not directly mutate the source motion. Generation is a separate
step.

The current failure mode is caused by under-constrained generation. If the user
moves one foot anchor, task-space LTE can propagate that displacement through
the semantic body graph into torso and hands. If a hand is also in contact but
is not explicitly constrained, it can bend or drift even though the user did not
edit that contact.

The correct principle is:

```text
edited contact anchors move to their targets;
unedited contact anchors stay fixed unless explicitly edited.
```

## Residual Family A: Task-Space Contact Handle LTE

### Role

Task-space LTE is the fast semantic trajectory editor and a useful production
path / warm-start source. It operates on a small semantic keypoint set, not
directly on robot joint angles.

Current implementation source:

```text
/home/xiaz/lte/contact_handle_lte.py
```

Current semantic keypoints:

```text
root
torso
left_hand
right_hand
left_foot
right_foot
```

In `motion_edit`, these semantic keypoints are derived from `body_pos_w` using
known body-name aliases such as wrist links and ankle/contact links.

### Inputs

- source motion npz;
- `ContactGraph`;
- `ContactEditPlan`;
- semantic keypoint trajectories `X_demo[name][frame, xyz]`;
- edited contact anchor handles;
- fixed contact anchor handles for unedited contacts.

### Variables

The optimization solves for edited semantic keypoint trajectories:

```text
X_edit[t, keypoint, xyz]
```

The existing solver uses an offset form:

```text
X_edit = X_demo + offset
```

### Objective Terms

#### Temporal Smooth Offset

Preserve smooth temporal motion by penalizing the Laplacian of offsets:

```text
|| L_time offset ||^2
```

This prevents a contact edit from creating abrupt frame-to-frame jumps.

#### Body-Relative Edges

Preserve semantic body structure using edges such as:

```text
root -> torso
torso -> left_hand
torso -> right_hand
root -> left_foot
root -> right_foot
```

In offset form this means nearby semantic body parts prefer similar offsets:

```text
offset[parent] - offset[child] ~= 0
```

This is useful for whole-body coherence, but it is also why a foot edit can
pull hands if hand contacts are not pinned.

#### Edited Contact Handles

For every nonzero `ContactAnchorEditRecord`, add a high-weight handle:

```text
X_edit[body, affected_frames] ~= X_demo[body, affected_frames] + delta_world
```

The body must be one of:

```text
left_hand
right_hand
left_foot
right_foot
```

or a resolvable semantic alias.

#### Fixed Contact Handles

For every unedited contact anchor, add a high-weight fixed handle:

```text
X_edit[body, contact_frames] ~= X_demo[body, contact_frames]
```

This is the production-path constraint that prevents unedited hand/foot
contacts from being dragged by another edit.

Important detail: fixed handles should generally target the original semantic
keypoint trajectory, not the anchor representative point. The anchor
`world_position` may be a representative contact point on the surface, while the
semantic keypoint may be wrist/ankle/contact-link position. Pinning to the
original keypoint trajectory avoids creating an artificial snap.

### Output

Task-space LTE returns edited semantic keypoints. `motion_edit` then maps their
offsets back to dense `body_pos_w` and optionally invokes a fullbody IK/backend
to produce a generated motion npz.

### Strengths

- fast;
- deterministic;
- independent of MuJoCo/CVX setup;
- good for interactive preview and batch generation;
- directly compatible with `ContactEditPlan`.

### Weaknesses

- not a physics solver;
- no joint limits by itself;
- no non-penetration by itself;
- only sees a small semantic skeleton;
- can produce artifacts if contact pins are incomplete.

### Current Production Path

`motion-edit generate-ref` uses this ContactEditPlan-driven path with both
edited and fixed contact handles, then passes the dense task-space target to
fullbody IK. Production contact forces are collected later by a Newton policy
rollout; this geometry stage does not solve contact dynamics. The obsolete LTE and proxy-acceptance queue CLI entries have been removed. The generation metadata should record:

- number of moving contact handles;
- number of fixed contact handles;
- bodies and frame intervals pinned;
- zero-delta edits ignored or treated as fixed contacts;
- the selected fullbody solver backend.

This path is the practical geometry backend for current real source motions.

## Residual Family B: Fullbody Interaction Mesh / q-Space Laplacian

### Role

The fullbody interaction-mesh method constrains robot motion in joint space. It
is closer to OmniRetarget/Holosoma retargeting than the task-space LTE layer and
is the direction for the experimental batch contact-Laplacian backend.

Relevant implementation source:

```text
/home/xiaz/somaforge/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py
```

That code is not a direct drop-in replacement for `motion_edit` generation. It
depends on robot models, MuJoCo Jacobians, CVXPY, object geometry, and per-frame
QP solves. But it contains the right fullbody constraints.

### Inputs

- robot q trajectory or nominal q;
- source/demo body or human keypoints;
- object/terrain points;
- edited task-space keypoints or contact targets;
- contact anchors and contact states;
- surface bindings;
- robot/object geometry.

### Variables

The solver optimizes robot configuration increments:

```text
dq
```

and auxiliary Laplacian variables:

```text
lap_var
```

per frame.

### Interaction Mesh Laplacian

Holosoma builds an interaction mesh from robot/human match points and object
points:

```text
vertices = [robot_or_human_points, object_points]
```

It computes adjacency/tetrahedra and target Laplacian coordinates:

```text
target_lap = L_source vertices_source
```

At solve time, it linearizes robot point motion through Jacobians and penalizes:

```text
|| lap_var - target_lap ||^2
```

with the relationship:

```text
J_L dq - lap_var = -lap0
```

This preserves spatial relationships between robot links and terrain/object
points while allowing the robot configuration to change.

### Contact / Sticking Constraints

The Holosoma solver also supports hard or near-hard constraints:

- foot sticking XY constraints against previous frame;
- foot lock Z constraints over configured windows;
- non-penetration constraints;
- self-collision constraints;
- joint limits;
- trust-region constraints;
- q smoothness and nominal tracking.

For contact-anchor editing, the analogous constraints should be:

```text
edited anchor body reaches edited target on same surface
unedited contact bodies stay fixed on their contact surfaces
normal displacement is not allowed
surface_id/object_id is preserved
joint limits are respected
penetration is avoided when geometry is available
```

### Output

The output is a full robot motion, usually q or qpos. In `motion_edit`, that
would eventually become a generated `MotionVersion` with:

- generated motion npz;
- edited ContactGraph;
- canonical segmentation;
- token catalog built from canonical segments.

### Strengths

- fullbody consistency;
- joint-limit aware;
- contact/sticking constraints can be hard constraints;
- can use terrain/object geometry;
- reduces artifacts after task-space editing.

### Weaknesses

- heavier dependencies;
- slower;
- needs robot model and geometry;
- harder to unit test;
- not ideal as the first interactive preview path.

## Full-Trajectory Contact-Laplacian Optimization

The target algorithm is not:

```text
task-space LTE first, then unrelated fullbody cleanup
```

That cascade can remain as a production path and warm start, but the target
solver should assemble one global sparse least-squares problem over the whole
trajectory:

```text
q'_t = q_t + dq_t
dq = [dq_0, dq_1, ..., dq_{T-1}]
min_dq || A dq - b ||^2 + damping ||dq||^2
```

The matrix structure should be:

```text
spatial/contact terms: block diagonal over time
temporal terms: banded across adjacent frames
```

The combined objective is:

```text
min
  w_move_contact   * edited contact anchor target error
+ w_fixed_contact  * unedited contact anchor drift
+ w_temporal       * temporal Laplacian smoothness
+ w_body_rel       * semantic body-relative Laplacian
+ w_mesh_lap       * interaction mesh Laplacian error
+ w_q_prior        * q / pose prior
+ w_q_smooth       * q smoothness

subject to
  same-surface contact constraints
  no normal-direction contact displacement
  preserved surface_id/object_id
  contact sticking for pinned contacts
  joint limits
  optional non-penetration
```

Both residual families then influence the same `dq`, instead of one stage
permanently baking artifacts for a later stage to clean up.

## Recommended motion_edit Roadmap

### Stage 1: Stabilize Existing Production Path

Add fixed handles for unedited contacts.

This directly addresses the current artifact where a hand contact bends after
only a foot anchor was edited.

Expected behavior:

- moving `left_foot` should not move `left_hand` during left-hand contact
  windows unless the left-hand anchor is also edited;
- moving one anchor creates exactly one moving handle plus many fixed contact
  handles;
- body-relative Laplacian still contributes, but cannot overpower fixed
  contact pins.

This is still valuable even after the batch solver lands because the existing
`lte_fullbody` subprocess path is the current production path and can provide a
warm start/proposal.

### Stage 2: Add Batch Contact-Laplacian Backend

Keep the public generation mode as:

```text
--mode lte_fullbody
```

Add internal solver selection:

```text
--fullbody-solver ik_subprocess|batch_contact_laplacian
```

`batch_contact_laplacian` should solve the full `[T, nq]` trajectory in one
global problem. Task-space LTE output may be used as a warm start or proposal,
but the final target algorithm is the unified batch solve.

The current interaction-mesh term is a lightweight extraction of the Holosoma
`interaction_mesh_retargeter.py` Laplacian idea: robot semantic points and
fixed object/terrain points form an interaction mesh, and the solver penalizes
changes in uniform Laplacian coordinates `L * V`. Only robot vertices write
Jacobian columns; object vertices are fixed. Real robot/object geometry adapters
and richer mesh construction remain future work.

### Stage 3: Unify Solver Interface

Create a backend-neutral interface:

```text
ContactEditPlan
ContactGraph
source MotionVersion
solver config
  -> generated motion npz
  -> generated ContactGraph
  -> generation report
```

Backends can include:

- `lte_fullbody` with `ik_subprocess`;
- `lte_fullbody` with `batch_contact_laplacian`;
- future physics/IK refiners.

## Key Invariant

Contact anchor dragging remains a planning operation. It creates or updates a
`ContactEditPlan`; it does not generate motion automatically.

Force-reference generation remains explicit:

```text
motion-edit generate-ref --plan ...
```

The generated force reference must be a new motion version. The archived source
motion, source contact layer, and canonical segmentation must not be mutated by
default.
