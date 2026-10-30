# Modular WBT expert and student experiments

Use the existing `ExperimentConfig`, manager blocks and native train/eval entrypoints.
Each invocation is independent. Data preparation, training and evaluation never launch
one another or choose a checkpoint. Set paths and adjust configurations manually.

```bash
source scripts/source_isaaclab3_newton_setup.sh
```

## Configuration blocks

| Module | Native configuration |
|---|---|
| Dense expert | `exp:wbt-expert-dense` |
| Sparse-reference expert | `exp:wbt-expert-sparse` |
| Force-reference expert | `exp:wbt-expert-force` |
| Student | `exp:wbt-student` |
| Sparse reference observation | `observation:wbt-sparse-native` |
| Sparse reference + force expert observation | `observation:wbt-sparse-native-force` |
| Future reference without qdot | `observation:g1-29dof-wbt-future-ref-no-qdot` |
| Measured and reference force expert observation | `observation:wbt-native-force` |
| Student measured contact state | `observation:wbt-student-contact-state` |
| Student measured contact state and force | `observation:wbt-student-contact-force` |
| Student without contact observations | `observation:wbt-student-proprio-depth` |
| Dense reward | `reward:g1-29dof-wbt` |
| Dense + native contact reward | `reward:wbt-native-sparse` |
| Root + native contact reward | `reward:wbt-native-root-contact` |
| Dense + native force reward | `reward:wbt-native-force` |

These are defaults of the same manager configuration types, not separate environments
or a second configuration format. Reward weights, observation term parameters,
termination boundaries, curriculum, network, optimizer and sampler parameters retain
their standard Tyro overrides. Sparse actors consume both `actor_obs` and
`actor_proprio_history`; retain these module input keys when replacing their blocks.

Student actors have a checked allowlist of measured observation functions. Their
inputs may include proprioception, depth, an externally specified task command and
actual contacts. Reference q/qdot, reference poses, desired force/contact states,
teacher routing and phase are never student actor inputs. Teacher/critic groups are
separate. The task command is independently configurable using
`--observation.groups.actor-obs.terms.velocity-command.params.source constant`
and `--observation.groups.actor-obs.terms.velocity-command.params.command '(1.0, 0.0)'`.
By default it replays the dataset's explicit 15-way `vel_cmd` task command stream,
as in existing PHP training. No phase index or reference pose is an actor input.

## Train an expert

Pass the SAME manifest to the motion command and terrain manager:

```bash
python -m holosoma.train_agent exp:wbt-expert-dense \
  --command.setup-terms.motion-command.params.motion-config.motion-manifest /path/group/manifest.json \
  --terrain.terrain-term.motion-matched-manifest /path/group/manifest.json \
  --logger.base-dir runtime/current/models/my_dense_experts \
  --training.num-envs 4096 \
  --algo.config.num-learning-iterations 3000 --headless
```

For sparse-reference training replace the experiment with `exp:wbt-expert-sparse`.
For a force expert use `exp:wbt-expert-force`. Both require validated native-contact
reference data. To try reduced rewards independently, append
`reward:wbt-native-root-contact`. Set termination conditions explicitly: changing
the reward does not silently remove limb-tracking termination checks. For example,
the historical root/contact experiment used
`--termination.terms.bad-tracking.params.bad-motion-body-pos-body-names '[]'`.

Use `--training.checkpoint /path/model_03000.pt` to resume a compatible experiment.
The native PPO loop interprets `num-learning-iterations` as additional updates on
resume. Checkpoints contain networks, optimizers, normalizers, configuration,
progress and learned sampler/curriculum state; no simulator snapshot is saved.

## Prepare contact-reference data independently

The released PHP motions contain no contact/force labels. Do not substitute old
force-threshold or geometry-distance labels. Record one explicitly supplied expert
and motion at a time:

```bash
python scripts/record_wbt_contact_reference.py \
  --checkpoint /path/expert/model_03000.pt \
  --manifest /path/group/manifest.json --motion-index 0 \
  --output-dir runtime/current/motions/my_contact_reference/motion_0000 --headless
```

The output contains the executed canonical trajectory, actual contact states and
per-constraint force, a reusable manifest and every raw solver contact snapshot.
The reference actor targets are torso plus six endpoint poses and contact bits;
there are no reference joint targets. Forces use the same six-part order:
left/right foot, left/right hand, left/right knee.

Activation is `CONSTRAINT && dist < includemargin`, with allocated `efc` rows.
Primary finite mesh faces are resolved before shared `contact_face_selection`
filtering; only ground and horizontal upward top faces contribute. Self/cross-robot
contacts do not contribute. Force is decoded from the actual MJWarp constraints,
not the raw Newton force buffer or body net-force sensor. The saved contract records
that last-solve contacts precede post-integration poses; they are not certified as
strictly same-state samples. Reset poses expose an explicit unknown mask rather
than stale previous-episode contacts. Contacts are captured before reset callbacks;
local window teleports also invalidate them even when the episode counter stays
positive. The next completed physics solve supplies fresh contacts.

The recorder requires a completed rollout by default. `--max-steps` and
`--allow-partial` are explicit diagnostic options. Output directories must be new.

Combine chosen per-motion or per-group manifests separately:

```bash
python scripts/combine_wbt_manifests.py \
  --manifests /path/a/manifest.json /path/b/manifest.json \
  --output /path/combined/manifest.json
```

## Distill manually supplied experts

Create a teacher routing table (paths are relative to this JSON file or absolute):

```json
{
  "teachers": [
    {"checkpoint": "/path/locomotion/model_11000.pt", "manifest": "/path/locomotion/manifest.json"},
    {"checkpoint": "/path/climb/model_03600.pt", "manifest": "/path/climb/manifest.json"}
  ]
}
```

The file is an explicit list of inputs, not a scheduler. Supply each final checkpoint
yourself. Student motions must be covered exactly once by the teacher manifests;
routing uses canonical motion-file identity. Every expert retains its checkpoint's
own network, observation layout, history and normalization. Different expert input
widths are supported. Action controls and canonical robot assets must agree.

```bash
python -m holosoma.train_agent exp:wbt-student \
  --command.setup-terms.motion-command.params.motion-config.motion-manifest /path/combined/manifest.json \
  --terrain.terrain-term.motion-matched-manifest /path/combined/manifest.json \
  --algo.config.distill.teacher-spec-file /path/teachers.json \
  --logger.base-dir runtime/current/models/my_student \
  --training.num-envs 2048 --headless
```

Defaults: measured contact state + force, ray depth and proprioception; global
adaptive motion allocation begins at iteration 1000, with one full-motion probe per
motion. For 296 motions this leaves 1752 normal environments. Before that iteration
the sampler remains uniform. Override
`--algo.config.distill.motion-schedule-start` and
`--algo.config.distill.probes-per-motion` directly. To remove force from student
inputs choose `observation:wbt-student-contact-state`.

The student uses the repository's native `DistillPPO` PPO/DAgger implementation and
native checkpoint format. It does not load the historical `tmp/` PHP adapter's
student checkpoints; those have a different network/checkpoint schema. Resume a
new student with `--training.checkpoint`; expert hashes and the complete student
observation layout are checked before restoring optimization.

## Evaluate or replay separately

The standard evaluator reconstructs modules from the checkpoint configuration:

```bash
python -m holosoma.eval_agent --checkpoint /path/student/model_01000.pt \
  --num-envs 1 --max-episode-length-s 40 --max-steps 1500 --visualizer kit
```

For remote Kit use the same invocation with `--livestream 1` and
`PUBLIC_IP=<tailscale-ip>` in its environment. Evaluation is never invoked by
training. TensorBoard uses the normal run directory; adaptive probe statistics and
normal sampling weight extrema are logged under `MotionSchedule/`.
