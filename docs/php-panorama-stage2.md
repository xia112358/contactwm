# Six-contact panoramic Stage 2 bootstrap

This experimental pilot connects `PanoramaContactPlanner` to the published PHP
Stage 1 tracker at tag `php-stage1-20261025-v1`. It is **not** the complete ACP
successful-rollout hindsight algorithm. Planner labels are the released fixed-q
native contact intentions at the currently commanded segment endpoint. They do
not certify dynamic load/support. A bounded 512-sample buffer collects online
observations and these labels, rather than successful-episode hindsight labels.

The original 567-dimensional student observation/action contract is preserved.
Only the 24-dimensional contact command is substituted as a complete command
per environment. Planner order (feet, knees, hands) is explicitly converted to
PHP order (feet, hands, knees); coordinates are base-yaw-relative. The lower
tracker still uses its established privileged terrain-height input. Thus this
pilot is not an end-to-end real-sensor-only controller.

The upper model reads causal synthetic LiDAR history, with 20,000 rays at 10 Hz,
from the torso mount. It reads a 64 × 128 full-sphere panorama within 2.5 m at
base world Z + 0.73 m, with yaw but no body roll/pitch. Sensor scans are terrain
only with exact odometry. History is cleared per environment reset. No ground
truth terrain is used to fill unknown panorama pixels.

PPO/DAgger continues on the tracker while the planner receives four independent
Adam updates per PPO iteration, minibatch four, learning rate 1e-4. Default pilot
uses four environments and 200 PPO updates. The first 50 updates use reference
commands; the following 100 ramp predicted-command probability to 0.2. This is
a conservative interface/optimization trial, not evidence of learned autonomy.
Reference rewards and goal generation remain active during the pilot.

Checkpoints retain native actor/critic, optimizers, normalizers and curriculum
state, plus planner config/weights/optimizer, its update count, contact schema
and bounded sample replay. Stage 1 initializes the tracker and preserves its global iteration/curriculum;
the independent panorama mixing schedule starts at zero; same-run resume restores the panorama checkpoint and schedule.
No simulator/native solver snapshot is serialized. Original release files are
never overwritten. `panorama_metrics.jsonl` explicitly reports supervised label
support, updates, loss and reference/predicted command counts.

Entrypoint: `python -m holosoma.acp.php_stage2 --checkpoint <model_07000.pt>`.
It uses AppLauncher-first parsing and the normal native simulation stack. The
current machine has an isolated pinned launcher at `tmp/run_stage2_native.py`;
it reuses installed simulator dependencies from the original checkout without
altering that environment. Process/log records are under this checkout's `tmp`.
