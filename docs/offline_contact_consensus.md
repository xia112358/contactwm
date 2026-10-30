# Offline contact consensus experiment

This is a separate contact-aggregation product, **not** a replacement for Newton
activation labels or the existing training baseline. No averaged pose is emitted.

Implementation: `somaforge_core.contact_aggregation`. Experiment driver:
`tmp/run_contact_consensus_v1.py`; regression tests: `tests/test_contact_aggregation.py`.

## Contract

- Inputs are source-hash-validated, same-state Newton contact queries, including
  verified allocation and the shared normal-ranked primary geometry-face map.
- One vote per demonstration, frame, robot collision shape and mapped face.
  No-contact demonstrations remain in the denominator. Contact manifold point
  count cannot increase a demonstration's vote.
- Reference is a whole-trajectory pose medoid among the fitting demonstrations.
  Bounded monotone pose-only DTW aligns complete repeated demonstrations. It uses
  no contact labels, interpolates no pose/label, and preserves source-frame paths.
  This is offline alignment, not a causal online observation algorithm.
- A binary time dynamic program minimizes mean vote disagreement plus a switch
  penalty. Independent face groups are never collapsed into a single mesh ID.
  Consensus and raw support are both exported. Low-support/unsupported held states
  must not be interpreted as measured contact.
- Contact-point sets remain intact. The representative patch is an observed
  point-cloud medoid; all contributing clouds and per-demonstration-balanced point
  weights are retained. These are robot geometry points, not projected surface
  targets. Point-cloud centers are diagnostic summaries, not contact constraints.
- Multiple primary-face alternatives at mesh edges remain explicitly recorded.
  Aggregation does not claim to resolve ambiguity in the underlying face mapper.

## Experiment and validation

The 12 successful recordings share one underlying motion. Fixed held-out sequence
indices are 3, 7 and 11 (environment IDs 3, 9, 13). The other nine determine the
reference and consensus. Three-fold validation within those nine selects switch
cost from 0, .25, .5, 1, 2, 4, 8. No held-out contact labels select parameters.
The group vocabulary is the union of observed physical shape/face identities,
including held-out-only identities so their missed contacts are counted.

Comparators are all-observation union, equal-demo majority, temporal consensus,
and temporal consensus without alignment. Union is an ablation of occurrence
selection, **not** an exact rerun of the legacy force-label aggregation pipeline.
All held-out metrics use the same pose-aligned evaluation timeline.

The selected switch cost was .5. On the three held-out recordings, shape/face F1
was .93829 (majority .93639); part F1 was .96344 (majority .96237). Part switches
fell from 84 to 40 and one/two-frame shape-face runs from 95 to 3. This supports
temporal contact denoising without a measured loss in held-out agreement. It does
not establish new-terrain generalization or physical realizability of a new pose.

Authoritative successful output: `tmp/contact_consensus_v3/`. Earlier v1/v2 runs
contain partial or pre-final outputs from serialization diagnostics; retained,
not designated baselines.

## Remaining boundary

The consensus is an interaction hypothesis. A subsequent trajectory generator
must satisfy it together with continuous full-body kinematics and collision
requirements, then re-query the generated poses through Newton. This experiment
does not generate that trajectory or establish improved sliding/position jitter.
There is no change to the global raw-contact or task-label filtering policy.
