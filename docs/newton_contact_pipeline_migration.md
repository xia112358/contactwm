# Newton contact pipeline migration

Historical migration record. The status, remaining work, test counts and local
artifact paths below describe the original28 migration at that time; they are
not the current setup or a training admission. The file stays at its original
path to preserve existing references.

Current interfaces are listed in the [documentation index](README.md).
Use the [contact-label contract](contact_label_contract.md),
[support evidence](support-semantics.md) and shared
[`contact_face_selection`](../packages/somaforge_core/README.md) rules for new
work: effective task contacts are ground or horizontal upward top faces;
side contacts remain raw audit evidence, not a reason to promote a side face to
top. Current [Generator defaults](../packages/generator/README.md) and
[modular WBT experiments](wbt-modular-experiments.md) are documented separately.

## Status at the time of this record

The shared contact implementation and original28 relabeling are implemented and
tested. **The whole predictor → infiller → online Newton loop is not yet
end-to-end accepted.** The old two-surface predictor cannot represent real side
contacts; unsupported surfaces now raise instead of being clamped into ground
or top. Expanding surface inputs/outputs and their geometry/loss mappings is the
next architecture decision. Do not start training or call this a finished new
training baseline yet.

## Implemented paths

| Path | Change |
| --- | --- |
| `somaforge_core.newton_contacts` | One activation predicate, explicit constraint allocation, exact body/shape/environment mapping, self/cross-robot exclusion. |
| Full evaluation recording | Saves actual solver fields including EFC addresses, expected row counts and contact frames. Missing interface fails explicitly. |
| `extract_solver_contact_labels.py` | Uses the shared recording reducer. |
| `extract_rollout_ref_contact_force_demo.py` | Contact bits come from solver activation/allocation. Force-threshold masks remain separately named force-support diagnostics. Solver/raw buffers are not accidentally sliced as environment arrays. |
| `serve_newton_contact_queries.py` | Creates a dedicated scene from a local policy checkpoint/config and optional matching motion/terrain manifest. No training, actions or integration. |
| `relabel_newton_motion.py`, `relabel_original28_newton.py` | Fixed-q relabeling with immutable source fingerprints, explicit geometry-point semantics and exclusive output creation. |
| Predictor demonstration loaders | Load labels for the actual edited motion, not its unmodified source; do not debounce raw truth. Current/target points use recorded geometry points, including their normal coordinate. |
| DAgger teacher | Queries current and candidate endpoint contacts; geometric proximity no longer admits labels. Geometry losses/penetration metrics remain separate. |
| Both closed-loop generators | Query realized contacts; the direct generator's visual schedule and the unified generator's probability-derived schedule are replaced by per-frame queries. |
| Corpus builder/trainer | Re-query corpus trajectories, rebuild contact contracts, and reject unverified old corpora. Infiller network architecture is unchanged. |
| Original28 segmenter | Segments verified part/surface states; movement/debounce selects events without editing raw labels. |

Historical diagnostic functions/scripts are retained, not deleted. Not every
archived experiment is a migrated production entry point. Geometry proximity
helpers are explicitly diagnostic and cannot stand in for Newton observation.

## Actual verification

- 55 targeted tests passed: shared predicate, missing data, corrupt saved
  activation, partially allocated friction constraints, part/environment
  filtering, unplanned observed contacts, edge-face classification, surface
  transitions, and relevant existing extraction/predictor tests.
- Actual G1 model loaded through IsaacLab/Newton from
  `runtime/current/models/climb00_pairwise48/wbt.pt`, using the canonical robot
  asset. Three fixed-q probes completed with unchanged q. See
  `tmp/newton_contact_model_probe_v7.json`.
- All **28 motions / 17,839 frames** were relabeled against each motion's own
  terrain, using the policy physics configuration. Output:
  `tmp/original28_newton_relabel_v2/`.
- All labels passed the loader's asset, source-hash, timeline, allocation and
  geometry-point-semantics checks.
- New cuts: **393 segments**, 17 shorter than seven frames, from 444 selected
  part/surface events. Output:
  `tmp/original28_newton_interaction_segments_v1/`. Its `training_ready` remains
  false pending the surface-vocabulary/data-contract migration and inspection.
- `climb_00` has 96 frames containing side contacts. Side-contact part-frame
  counts, in left foot/right foot/left hand/right hand/left knee/right knee
  order: `[26, 2, 0, 0, 6, 68]`. These are not ground/top labels.

The fixed-q probes' candidate/active counts include self contacts and are not
support counts. Reduction excludes self contacts before producing part masks.

## Important correctness fixes

1. The dedicated query process disables CUDA Graph capture. Initialization with
   graph capture left lazy inverse-map buffers unusable for ad-hoc graph-external
   queries, causing a real CUDA access error in contact conversion. This is an
   execution-mode change in the query process, not a physics-parameter change.
2. Query runs Newton collision, the solver's own contact conversion and MJWarp
   `fwd_position(..., factorize=False)`, including constraint construction.
   There is no integration, q correction or full dynamics solve.
3. Contact geometry points are recovered through the solver's exact
   raw-contact-to-solver index mapping and the same offset/margin stripping as
   the conversion kernel. Midpoint ± half-distance is not reliable at edges.
4. Surface identity uses the actual terrain contact point and the actual mesh
   face planes. Edge normals need not point along the face normal. Candidate
   faces are retained, with a deterministic representative; surface labeling
   does not redefine whether contact is active.
5. Some nominal poses exceeded the checkpoint's 160-contact query capacity.
   Failed jobs were retried with explicitly larger **query-only** buffers
   (`nconmax=2048`, `njmax=16384`). Original/query capacities are recorded where
   overridden. Policy weights, margin/gap, collision filters and dynamics
   parameters were not changed. Earlier successful jobs needed no expansion.
6. Live recordings retain their last-solver/post-integration timing distinction.
   They cannot simply be declared fixed-q training labels; the loader requires
   same-state re-evaluation for that contract.

## Remaining work recorded at the time

- Extend the predictor's surface vocabulary and associated input, target,
  candidate-surface, geometry-loss and cache formats. Never map side contacts to
  top just to load an old checkpoint.
- Complete authoritative online scene routing/adaptation for varying known box
  scenes. Current native workers handle their actual initialized terrain;
  frame-bound queries reject changed dimensions. There is no silent approximate
  terrain rebuild.
- Rebuild the predictor manifest/q cache and direct-infiller corpus from the
  accepted new surface contract; run a small full closed-loop acceptance test.
- The infiller architecture and policy checkpoint remain unchanged. No training
  was started or stopped, and no old data/checkpoints/logs were deleted.

## Reproduction commands

From project root, with the local Newton environment available:

```bash
source scripts/source_isaaclab3_newton_setup.sh
python scripts/relabel_original28_newton.py \
  --checkpoint runtime/current/models/climb00_pairwise48/wbt.pt \
  --output tmp/original28_newton_relabel_v2 \
  --workers 2 --query-nconmax 2048 --query-njmax 16384
```

Already verified files are checked and skipped. Failed attempts get new
numbered directories; nothing is overwritten or deleted. For a fresh
reproduction, use a new output directory. Fixed-q label files may contain older
cached convenience touchdown indices; the segmenter and predictor loader
recompute events from raw contact/surface arrays, which are the authoritative
inputs.
