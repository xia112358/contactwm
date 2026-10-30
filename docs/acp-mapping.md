# ACP history-fused point clouds and height maps

The implementation follows the mapping section and Figure 8 of
[ACP](https://arxiv.org/html/2610.31608v1): retain the highest hit per 3-cm cell
in a gravity-aligned sensor grid covering `[-3,3) x [-1.5,1.5)` metres; select
measured/neighbor-average/empty modes; transform predictions with scan-time
odometry and fuse nonempty predictions into a world-axis map. CNN widths,
mode-disagreement uncertainty, precision-weighted fusion, process variance and
10-second expiry are **local implementation choices**, not claimed to be the
unpublished author's exact implementation.

## Geometry and history

The rolling map is 256x256 cells per environment (7.68 m square). It preserves
world coordinates across robot translation/rotation and never wraps discarded
edges. Duplicate scans do not reduce uncertainty; out-of-order scans require
an explicit episode reset. Reset clears only the affected environment. No map
or simulator snapshot is included in policy checkpoints; resumed episodes start
with empty perception history.

The overhang prior follows the paper's thresholds: if an existing cell has
standard deviation below 0.15 m, reject a new elevation more than 0.15 m lower
unless the new observation's elevation angle is greater than -40 degrees.
Unknown/high-uncertainty query cells use local minimum inpainting followed by a
masked median; confident measured cells are retained. This selective filtering
is a local choice that avoids smoothing away thin observed structures. Large
unobserved regions remain unknown. Exported maps retain the fusion mask before
query-time inpainting, variance, timestamps and perception mode. Neural-mode
cells can include the network's neighborhood-average predictions.

Policy scans use the robot's **current body-yaw frame**, with 0.1-m spacing:
187 XYZ points for the planner and 63 heights for the tracker. The planner masks
unknown points; an entirely unknown scan issues zero contact commands and is
excluded from hindsight training. Unknown tracker heights are zero placeholders
without added observation noise; the existing 63-value tracker interface has no
separate validity channel. Training with this observation mode is therefore
required; old ground-truth controllers are not established as robust to it.
Training resume rejects a perception-mode mismatch to avoid mixing replay data.
The critic retains privileged terrain-height queries.

## Synthetic sensor and its limits

The simulation adapter runs at 10 Hz with 20,000 rays and exact simulated
odometry. It ray traces the initialized terrain mesh, including terrain
occlusion, in chunks of eight environments. This is an **instantaneous synthetic
panoramic scan**, not MID-360's exact pattern, DLIO, motion deskew, robot
self-occlusion or hardware calibration. It currently has no measurement noise.
The configurable mount defaults to `torso_link`, offset `(0,0,0.35)` m and xyzw
rotation `(1,0,0,0)` (downward-facing synthetic mount); rays cover -7 to 52 degrees
in the sensor frame. Robot assets and native contact truth are unchanged.
External scans must already be deskewed and carry their scan-time pose.
Poor initial coverage stays unknown until observed.

## G1 sensor-interface closed-loop recording

`holosoma.acp.robot_io.G1TrackerInput` builds the existing dense WBT actor
observation from a synchronized `RobotSample`: encoder positions/velocities,
base-frame IMU angular velocity, and estimated world pose/linear velocity.
Tracked link transforms are reconstructed with canonical URDF forward
kinematics. Desired joint/anchor trajectories are commands, not measured robot
poses. No simulator link poses or ground-truth height queries enter this
observation builder.

The dense WBT checkpoint has a different terrain contract from the ACP tracker:
49 samples in a torso-yaw 7x7 grid, x varying fastest, with values
`torso_z - mapped_terrain_z - 0.5`. The map query returns an explicit mask;
unknown entries are zero placeholders because this checkpoint has no mask input.
Robustness to missing cells is not established by the interface conversion.

To run the native Newton recorder with that actor input:

```bash
python -m holosoma.acp.record --checkpoint /path/to/canonical/model.pt \
  --manifest /path/to/canonical/manifest.json --motion-index 0 \
  --output-dir tmp/g1-sensor-rollout --terrain-observation fused_raw \
  --tracker-inputs sensor_interface --allow-partial --headless --device cuda:1
```

This selects a single motion and evaluates the existing policy; it starts no
training. The recorder saves actual executed articulation poses, native solver
contact audit, LiDAR packets, accumulated map, and `control_inputs.npz` with
pre-normalization actor inputs, actions and height validity. A partial or failed
motion remains marked `complete=false`. Poses are post-physics; contacts retain
the native solver sampling semantics described in the recording provenance.

The simulator adapter supplies **ideal** IMU/encoders/odometry and the synthetic
LiDAR described above. Hardware still needs calibrated sensor acquisition,
deskew/odometry, timing and actuation integration. This mode is a WBT deployment
interface experiment, not a trained ACP planner or a hardware deployment claim.

Options in `holosoma.acp.train` include `--lidar-hz`, `--lidar-points`,
`--lidar-max-range`, `--lidar-body`, `--lidar-offset`,
`--lidar-rotation-xyzw` and `--mapping-max-age`.

## Train with fused observations

Activate the managed SomaForge environment using
`source scripts/source_isaaclab3_newton_setup.sh` before running these commands.

New ACP training runs default to the raw historical fusion baseline. This is
explicitly a baseline with no learned denoising:

```bash
python -m holosoma.acp.train --manifest /path/to/reference/manifest.json \
  --stage 1 --num-envs 8 --terrain-observation fused_raw --headless
```

Use `--terrain-observation ground_truth` to reproduce the original ACP pilot or
resume an old ground-truth policy. Old checkpoint configurations without a mode
keep their previous behavior.

To fit the neural mapper, prepare an NPZ with `height`, `valid`, `target`, and
`target_valid` arrays of shape `[N,200,100]`. Heights are in metres relative to
the gravity-aligned sensor frame at 3-cm spacing. `height` is the highest noisy
hit per cell; `target` is aligned ground-truth elevation. Unknown ground truth
is ignored. Labels choose the closest available measured/neighbor-mean mode
within 3 cm, or empty when neither is suitable. No author's data/weights are
bundled; diverse noisy scene meshes and a held-out validation set are still
needed for a useful model. An optimizer check is not evidence of real LiDAR
mapping quality.

```bash
python -m holosoma.acp.fit_mapping --dataset /path/to/mapping_pairs.npz \
  --output runtime/current/models/acp_mapping/model.pt --steps 1000
python -m holosoma.acp.train --manifest /path/to/reference/manifest.json \
  --stage 1 --num-envs 8 --terrain-observation fused_neural \
  --mapping-checkpoint runtime/current/models/acp_mapping/model.pt --headless
```

`fit_mapping` is a CPU-default offline utility; `--resume` restores its model,
optimizer and sample RNG into a **new** output checkpoint. Neural inference
rejects checkpoints without a positive training-step count. That check detects
untrained weights, not convergence. Policy configurations retain the mapping
checkpoint path, so that file must accompany a neural-mode policy.

## Record and export

The expert recorder can save synthetic scans while retaining the selected
dense expert's original policy inputs:

```bash
python -m holosoma.acp.record --checkpoint /path/to/dense_expert/model.pt \
  --manifest runtime/current/manifests/wbt_single_climb_00.json \
  --output-dir runtime/current/motions/acp/mapped_reference \
  --terrain-observation fused_raw --headless
```

This adds `lidar_scans.npz` (10-Hz scan-time poses/points) and
`perception_map.npz` (accumulated surface points and the raw height map).
The reference's 50-Hz `acp_terrain`/`acp_terrain_valid` arrays contain sampled
historical-map observations; their perception mode is recorded in metadata.
Recording retains `ground_truth` as its default for compatibility.

For offline reconstruction, use that `lidar_scans.npz` or an external NPZ with
`points_sensor [T,P,3]`, `valid [T,P]`, `position_w [T,3]`,
`quaternion_xyzw [T,4]`, `timestamps [T]`, `root_position_w [T,3]`, and
`root_quaternion_xyzw [T,4]`. Optional `episode_start [T]` clears history between
episodes. Poses use metres, timestamps seconds and quaternions xyzw.

```bash
python -m holosoma.acp.map_recording --recording /path/to/lidar_scans.npz \
  --output-dir tmp/acp/mapped_cloud
```

Add `--checkpoint /path/to/mapping/model.pt` for neural reconstruction.
Outputs are `scans.npz` (planner points/masks, tracker height maps/masks),
`map_XXXX.npz` per episode, and `report.json` with observed coverage. Exported
`points_w` are accumulated **2.5-D surface-cell centers**, not a full 3-D map of
every historical return; original scans retain those raw returns. All output
paths refuse overwriting existing artifacts.

## Validation scope

`tests/test_acp_mapping.py` exercises coordinate transforms, highest-return
rasterization, world-map rolling, temporal fusion, overhang preservation,
unknown-region handling, selective reset, mode supervision/optimization,
checkpoint resume, offline reconstruction and a real CPU Warp terrain-ray scan.
These tests do not establish closed-loop robot control, hardware performance,
the paper's 30-ms timing, or trained mapping quality on real observations.

## Historical 3-D near-field panorama (alternative observation)

`holosoma.acp.panorama.SurfacePanorama` implements the proposed alternative to
vertical height sampling. It integrates deskewed sensor-frame XYZ returns with
scan-time world poses into bounded 3-D occupied voxels. It retains different
heights at the same XY, including walls and overhangs. A virtual reader above the robot, aligned with gravity and robot yaw, casts spherical rays to the
nearest occupied voxel and returns a 64 × 128 full-sphere depth image within
0.12–2.5 m. Resolution is 3 cm; history expires after 15 s and is bounded by
5 m retention radius and 200,000 voxels. These are configurable defaults.

```python
from holosoma.acp.panorama import SurfacePanorama

memory = SurfacePanorama()  # numpy reference; backend="warp" for compiled CPU
memory.integrate(points_sensor, valid, sensor_position_w, sensor_xyzw, scan_time)
observation = memory.render(base_position_w, gravity_aligned_yaw_xyzw, query_time)
# depth [H,W] metres; valid; age seconds; confidence; points_w [H,W,3]
```

History can supply previously observed surfaces behind or below the robot.
Never-observed and expired rays remain unknown (depth zero, validity false),
not inferred flat ground or certified free space. Depth is distance to the
occupied voxel entry, with voxel-scale geometric error. Confidence is a
heuristic observation-count/freshness score, not a calibrated probability.
The static-scene implementation does not remove old surfaces by free-space
carving. Real deployment requires synchronized poses, deskewing, self-return
filtering and handling localization drift and dynamic objects.

This is a new observation representation, not a claim about the ACP paper.
Its depth discontinuities at true silhouette/occlusion boundaries remain;
it avoids collapsing 3-D surfaces to the highest Z for each vertical column.
The existing ACP height-map policy/checkpoint is not wired to this new tensor;
a compatible encoder and training would be needed to use it for control.

The G1 climbing playback uses 160 recorded synthetic terrain-only scans and
401 kinematic frames, ideal odometry, and no scan self-occlusion. It shows the
final panorama, observation age and selected pixel's world-space ray alongside
the robot and obstacle. This demonstrates perception, not closed-loop control.
`tests/test_acp_panorama.py` checks 3-D retention, history, unknowns, expiry,
pose transforms, bounds and optional compiled/reference numerical agreement.

Local artifacts are in `tmp/g1-panorama-player/`; the export script is
`tmp/export_g1_panorama_player.py`. A warmed server CPU read of 8,192 rays from
83,376 voxels took median 67.3 ms across 20 reads with the optional Warp CPU
backend (reference export median 204 ms). This excludes fusion and initialization
and does not establish real-robot hardware latency. Warp here performs array
computation only; no physics simulator is used.

The playback reader now uses a fixed offset along world Z above the base. The
offset is the initial canonical head mesh top relative to the base plus 20 cm;
it is not rotated by body roll or pitch. Each frame exports `panoramaOrigin`,
and both depth generation and browser backprojection use that exact origin.
Only translation and yaw follow the robot. This virtual viewpoint does not
create observations: it reads only previously observed occupied surfaces.
