# Shared contact-label contract

Raw Newton activation/allocation is immutable evidence. Task contact labels may
be filtered, but all stages must use `somaforge_core.contact_labels`, not local
force thresholds, morphological closing, or geometric contact classifiers.

`DEFAULT_POLICY` is currently explicit identity (one sample for activation,
release, and surface changes). This preserves the verified behavior without
choosing new smoothing hyperparameters silently. `ContactLabelStream` also
supports causal confirmation filtering. `process_contact_sequence` uses exactly
that stream implementation. Positions held by filtering are task memory, not
fresh solver observations. A changed sample rate or policy changes the contract.

Offline loaders process raw labels through the shared policy and expose both
`contact_*` task fields and `raw_contact_*` evidence fields. New relabel archives
save raw fields, `task_contact_*` fields and `contact_label_contract_json`.
Persisted task fields must equal recomputation; incomplete or incompatible
archives fail. Existing raw-only fixed-q archives remain valid raw inputs.

Online independent-q queries use the same identity policy. If a temporal policy
is enabled, callers without history fail rather than pretending independent
poses are consecutive frames. Stateful online callers must retain one stream
across segments and reset only on episode boundaries. State/filter policy must
not be changed independently by dataset, DAgger, and evaluation callers.

Augmentation generation entry points (`contact_aware_preview`, `lte_fullbody`,
`rollout_authority`) require verified source layers. Provide
`plan.metadata.newton_contact_file`, or a layer-local
`contact_labels/<motion_id>.npz`. Source hashes, task labels, part/face mapping,
anchor intervals and real face polygons are checked before solving. The plain
graph reader remains diagnostic. An edited target is still intent, not observed
contact; generated q must be queried again, including q created by aggregation.

## Remaining acceptance work

The consensus16 base now has a rebuilt shape/face source layer at
`tmp/climb00_consensus16_contact_layer_v2/`. It preserves all 10,947 observed
contact-point samples in 1,620 fixed-manifold-size intervals (not action
segments). Heel/toe are mapped by canonical sphere identity. Side contacts and
multiple points are retained. Manifold-size changes do not generate fake
touchdowns. All patches compile into the task-space builder with zero skips;
source point reconstruction against saved canonical FK is below 0.00084 mm.
No plane projection or contact-point median was used. The source-point semantic
tag is `newton_robot_geometry_point_trajectory`, not counterpart-plane targets.
Existing edit plans reference old anchor IDs and have not been silently rebound.

Coverage207 migration output: `tmp/coverage207_task_labels_v1/` contains 187
processed label archives and an inspection-only manifest. All 20 missing raw
query outputs are explicitly excluded with reasons, not converted to negatives.
The 48 trajectories with unknown historical target provenance remain only
verified observations; this migration does not certify their target contracts.

Boundary-state caches now fingerprint motion bytes, terrain bytes, label bytes,
sample rates and shared policy. Old caches without this provenance fail rather
than being assigned a new version string. Shared corpus records separate raw
from task masks and save their contract. Sequence query callers retain state and
avoid counting a shared boundary twice. Full generators have been wired to this
sequence path, but a fresh simulator-backed full-generation acceptance run is
still pending.

- Rebuild legacy augmentation anchor/patch layers from the shared labels; old
  layers may now correctly fail source validation. Do not approve them by merely
  adding a contract string.
- Complete checks on historical cache/checkpoint entry points and regenerate
  applicable caches. Archived experiments have not all been migrated.
- Temporal filtering is implemented and tested, but a non-identity deployment
  policy is not enabled until every online caller carries history correctly.
- Resolve the known 20 Newton face-mapping failures and the 48 missing historical
  target provenance cases before declaring coverage207 fully accepted.

Touchdown grouping, movement selection and hold trimming are segmentation rules,
not permission to rewrite contact labels. Current augmentation validation checks
part/representative-face intervals; it is not proof of material-patch force,
support, or absence of penetration.
