# Training readiness of constraint-aware directions

The network arm performs real parameter updates and recomputes its predicted
pose; it does not replace that output with an inference-time pose projection.
Nevertheless, this is currently a static-case research training loop, not a
production loss module or a demonstrated generalizing training recipe.

## What is being optimized

The scalar AL objective contains geometric/contact residuals and a deviation
prior. A separate optimizer operation projects the proposed parameter update
against active protected residual derivatives. Its affine halfspaces include a
small inward reserve, and trial updates must pass the nonlinear geometric guard
and actual Newton contact evidence.

Ordinary `loss.backward(); Adam.step()` does not implement this method. Momentum,
adaptive scaling and weight decay can change an already projected gradient's
direction. Integrating an optimizer requires constraining its actual proposed
update and validating the resulting prediction. A minibatch also requires
consistent sample/context identities for multipliers and joint handling of
constraints across examples; per-example parameter projections are not
independent because examples share network parameters.

The 480-step static experiment repeatedly fits one input to diagnose convergence.
It does not imply that production training should solve every sample to completion
inside every minibatch; that would be a different and substantially more expensive
training procedure. Whether one or a few constrained updates per batch generalize
remains untested. Inference can be a normal network forward pass, but feasibility
on unseen inputs has not been established.

## Cost and engineering limits

The current implementation materializes one full parameter-gradient row per
active constraint. With P parameters and M active constraints, this uses O(MP)
storage, additional reverse passes, and a small Gram solve. Backtracking can
perform up to 24 candidate Newton queries per outer iteration. Research traces
save raw query tensors and JSON, so measured wall times include diagnostic I/O
and cannot be quoted as optimized production throughput.

The extended experiment records per-method wall time, step duration, query
count, total model parameters, peak Torch CUDA allocation and maximum active
rows. CUDA memory excludes allocations owned directly by Warp/Isaac. Setup and
checkpoint loading are excluded from the per-method timer. These measurements
are for one static example, not batched training throughput.

## Explicit evidence for disappearing self pairs

A missing Newton candidate is not automatically safe. The optional research
adapter reads convex-mesh vertices from the actual initialized Newton model's
mesh pointers and transforms them using its current body and shape transforms.
For a previously observed self pair, it tests separation along the previous
normal, conservatively padded by both model shape margins and 1 micrometre of
numerical reserve. A positive support gap proves geometric separation of those
convex shapes. This supplements an inactive optimization residual; it never
creates or changes a Newton contact, constraint activation, or allocation.

Unsupported geometry or inconclusive separation remains missing/unknown and
still fails the existing protection guard. Existing raw pairs always take
precedence. Saved certificates include actual query IDs, shape IDs, axis, body
transforms, padding and support-gap bound; per-shape NPZ files preserve the
actual body-local collision vertices and model fingerprint for offline audit.

This is a research adapter for verified convex meshes, not a generic silent
fallback for missing solver fields. Its first attempted run exposed that Newton
uses deduplicated collision meshes; the corrected implementation resolves the
actual pointer against the model's retained mesh objects.

## Evidence needed before production use

- Convergence and feasibility across motions, scenes and contact transitions.
- Held-out generalization of a shared trained network, not independent frame fits.
- Batch-compatible constraint/multiplier state and actual optimizer integration.
- Profiling without research export, then bounded memory and query costs.
- Explicit treatment of infeasible tasks and incomplete collision evidence.

No production checkpoint or ongoing training job is changed by these experiments.

## Extended-run protocol

The nominal pair of projected arms is extended from 120 to 480 outer iterations,
with unchanged loss, tolerances, root cost and no separation-certificate change.
This isolates additional optimization budget. The root-input perturbation is
rerun for the same 120-iteration budget with explicit separation certificates,
so its comparison isolates missing-pair handling rather than extra iterations.
Both record runtime cost. The certificate rerun saves a research-only network
state plus its input and asset identity, reloads the learned parameters and
checks the plain forward prediction against the accepted endpoint. This does
not evaluate unseen examples and does not replace a production checkpoint.

## Nominal extended result

Both 480-iteration nominal arms passed the existing static acceptance. The
network endpoint had zero audited foot-vertex interior depth and zero queried
Newton penetration; old hand/knee contacts remained, and the left foot was
activated. Root displacement was 5.082 cm and maximum joint change 20.896 degrees
(relative to that run's initial prediction). Joint-limit error was 8.64e-7 rad,
within the existing tolerance. This is static feasibility, not a proof of
support forces or trajectory validity.

The 2,970,792-parameter network arm accepted 478 updates in 540.24 seconds,
including diagnostic overhead: median measured step 0.929 seconds, p95 1.591
seconds, 5,141 queries, and peak Torch allocation 0.926 GiB (excluding Warp/Isaac).
The pose arm accepted 475 updates in 460.44 seconds, using 5,626 queries. One
network and four pose local direction solves did not reach their projection
convergence tolerance; every accepted step still passed the nonlinear guards.
The first saved passing network checkpoint was iteration 221; the first saved
passing pose checkpoint was iteration 320. Saved checkpoints here mean recorded
pose diagnostics; the nominal run did not export learned network weights.

## Certified missing-pair rerun

At the same 120-iteration root-perturbation budget, pose/network foot depth
improved from 4.133/22.431 mm to 2.980/8.692 mm. Both now accepted 120 updates
rather than stopping at 99/73. Neither passed complete static acceptance; the
network still had 0.001942 rad joint violation. All old actual contacts remained.
Offline replay verified all 4,036 separation certificates (minimum padded gap
9.003 mm), including absence of the corresponding raw pair in each query.
The saved network state reproduced its final output after disk reload with
maximum q error 0.0. It is a research fit, not a production checkpoint.

Report: `tmp/constraint_direction_extension_20260926/RESULTS.md`.
33 tests passed, with additional runtime certificate and checkpoint audits.
