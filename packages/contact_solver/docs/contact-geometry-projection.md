# Acceptance-aligned geometry and row-space projection

This opt-in continuation of the contact-band experiment retains the actual
Newton activation/allocation/primary-surface contact definition and the same
1 mm penetration acceptance. It changes neither policy physics nor production
predictor training. No QP pose teacher is used.

## Geometry agreement

`acceptance_geometry_residuals` adds the complete collision-shape support-plane
violation used by the static acceptance check. This is the existing conservative
box-face diagnostic, not a claim that a box-plane bound is an exact mesh SDF.
The previous final pose had zero sampled excess but 0.0477995 mm full-shape excess;
the new row has a nonzero derivative there (recorded in the geometry regression).

A second row takes the worst signed clearance among Newton's reported full-body
witnesses, complete shape-plane distances, ground distances, and sampled SDF.
It matches the maximum tested by penetration acceptance. Raw Newton measurements
remain separate in the diagnostics and are never re-labelled as geometry truth.
An empty Newton candidate set is zero *reported* penetration, not a separation
certificate; complete-shape geometry remains active to detect deep embedding.

Protecting the Newton reported maximum independently was a failed intermediate
ablation: with about 82 mm geometric embedding, newly discovered pairs changed
reported depth from about 19 to 33 mm, blocking otherwise improving updates.
Protecting the joint worst-case measure avoids treating increased query coverage
as increased physical violation. The separate complete-shape rows also remain.

## Projection solver

`project_direction(..., algorithm='rowspace')` normalizes constraint rows and
uses a double-precision Gram eigendecomposition to work in their small row space.
Nonhomogeneous projections use a primal SLSQP solve. The result must satisfy
primal feasibility, stationarity and complementarity checks in addition to the
solver status. The experiment refuses to apply a nonconverged direction.

Failed interior-reserve solves receive a separate linear feasibility check.
Only when the reserve halfspaces are reported infeasible may the opt-in
`relax_infeasible_reserve` path retry with homogeneous non-worsening halfspaces.
It does not relax the nonlinear guard, actual contact evidence, or final limits.
The failure and retry are both logged; this is not proof the nonlinear task is
infeasible.

Homogeneous cones use polar-cone NNLS. Degenerate cones can have no strict
interior and very large cancelling multipliers. When the first fit fails primal
feasibility, bounded LPs identify implicit equality faces, those faces are
eliminated, and the reduced NNLS is checked again against the original normals.
The facial reduction is numerical (LP tolerances and rank cutoffs are recorded
in the implementation), not a symbolic exact-arithmetic certificate. A recorded
30-row failure is retained as a standalone numerical regression fixture.

## Experiments and precision

- `step14_contact_geometry.json`: complete geometry, original coordinate solver,
  pose only. This isolates the geometry change.
- `step14_contact_rowspace.json`: complete geometry, checked row-space projection,
  pose and network. Network parameters and floating inputs use float64; the live
  Newton engine is unchanged. Initial float64 output drift must remain below
  1e-4 and is reported, and learned weights are reloaded in the same precision.
- Old coordinate projection remains available for reproducible comparisons.

A successful single-frame fit is still a custom constrained training update,
not a standard Adam scalar loss, minibatch validation, or proof of dynamic
support/no-slip behavior. The final report must distinguish contact established,
penetration accepted, static acceptance, and deployment readiness.

## Activation boundary reserve

The fused-geometry pose arm completed 480 steps; auditing every accepted query
and its full-body residual places first contact satisfaction at step 108, and
all later accepted steps remain satisfactory. Its final maximum penetration is
0.371800 mm (root correction 3.135103 cm).

The first fused-geometry float64 network arm stopped at 35 accepted steps with
converged projections: a knee upper-gap guard crossed its numerical boundary
by about 0.000020 mm. `step14_contact_rowspace_interior.json` separately tests
`zero_tolerance_interior_reserve`: within the existing 0.5 mm anticipation band,
the projected direction requests at most a 0.05 mm inward reserve at the full
trial step. Backtracking scales that displacement. This is a numerical update
rule, not a new contact definition or an attraction target at 1 mm. Actual
activation, full-shape penetration acceptance, and nonlinear guards are unchanged.

## Final single-frame result

The network interior-reserve arm accepts all 480 updates, satisfies actual
contact plus 1 mm penetration from accepted step 193 onward, and ends with zero
reported/diagnostic penetration and 4.435356 cm root correction. All later
accepted steps remain satisfactory; all applied projections converge without
reserve fallback, old contacts and the plan remain unchanged, and checkpoint
reload reproduces q exactly in float64. The successful pose arm first satisfies
the same criterion at step 108, remains satisfactory thereafter, and ends at
0.371800 mm penetration with 3.135103 cm root correction.

The complete report and per-step audits are in
`tmp/contact_geometry_projection_report_20260926/RESULTS.md` and `summary.json`.
This does not establish float32 training, shared minibatch generalization, or
policy rollout feasibility. The network run takes about 615 seconds and 3788
queries including diagnostic export; no production throughput claim is made.
