# Solver contact activation and penetration-band experiment

This opt-in research change retains the projected AL updates in pose and network
parameter space. It is not a drop-in scalar loss for an ordinary optimizer and
has not changed production predictor training or any Newton collision setting.

`configs/contact_learning/step14_direction_contact_band.json` compares the same
baseline1000 frame 14 and 480-step budget against `step14_direction_long.json`.
The root prior, tangential retention, foot alignment and coverage remain present.
They are explicit task constraints, not definitions of contact.

## Contact evidence and residuals

Actual contact comes from the live Newton reader's `eligible` mask: task pair,
actual activation/allocation, and selected upward primary surface. Active but
unallocated rows raise an error. Distances alone never establish contact.

For the configured body/surface, an eligible contact switches off the upper-gap
attraction residual. Otherwise the approach residual uses the minimum matching
live witness distance; absent witnesses use the support-plane distance only as
geometric guidance, never as evidence. The boundary is the fixture's verified
uniform actual margin minus a 1 micrometre numerical crossing reserve. This
reserve is not an alternative contact threshold. The fixture requires an
unambiguous actual configured margin; heterogeneous adapters must use pair margins.

The independent lower-gap and full-body collision residuals allow the existing
`InteractionAcceptance.shallow_penetration_m` (1 mm). They still push out deep
penetrations even when contact constraints are already active. The nonlinear
collision guard uses raw penetration depth with a 1 mm bound. Unlike the loss,
its residual is not shifted: this preserves the existing projection algorithm's
interior reserve (10% of the bound) without changing physical acceptance.
Positive separation certificates stay conservative and do not assert contact.

## Loss is not the protection boundary

Gating attraction must not erase the normal used to protect existing contact.
The experiment evaluates a second set of residuals for guards and direction
projection. Its upper-gap residual remains differentiable after activation,
with zero violation tolerance. It does not contribute to the AL objective.
Rows are considered for projection within a 0.5 mm anticipation band; this only
selects protective normals and does not redefine activation or add attraction.
Actual old-contact evidence is still required at every accepted fresh query.

## Acceptance and comparison

Reports retain `legacy_static_recovery_pass` and separately expose:

- `required_contact_established`: actual eligible contact for each task.
- `penetration_accepted`: both full Newton penetration and geometric diagnostics
  are at most the same 1 mm tolerance.
- `contact_satisfied`: both conditions above.
- `static_recovery_pass`: contact satisfaction plus the existing joint and anchor
  checks; it is not a claim of load bearing, no slip, or dynamic stability.

The previous run must be rescored with the same criterion. In its saved records,
pose/network first satisfy contact plus 1 mm penetration at iterations 200/181,
not the legacy passing inspections at 320/221. Saved inspections are sparse and
these are not necessarily the exact first passing iterations.

The first gated-only trial stopped after 22 pose updates / 12 network updates:
actual old contacts blocked line search after their protective boundary gradient
had been erased. Its failed outputs remain in
`tmp/constraint_direction_contact_band_20260926`. The corrected guarded run is
`tmp/constraint_direction_contact_band_guard_20260926`; compare all three with
`tmp/summarize_contact_band_20260926.py`.

The first boundary-guard run used shifted collision residuals with nearly zero
guard tolerance. Pose optimization stopped at 7.67 mm penetration with a
nonconverged local projection. The physical-depth guard variant retains the same
1 mm physical bound while preserving a nonzero interior reserve. Its artifacts
are in `tmp/constraint_direction_contact_band_physical_guard_20260926`.

## Completed comparison

See `tmp/contact_band_comparison_20260926/RESULTS.md` and `summary.json`.
The physical-depth guard pose arm first has a saved satisfactory pose at step
260 (root 2.430 cm), but ends at 1.048 mm full-shape penetration after 480 steps.
Its sampled collision residual is zero: sampling and acceptance are still
inconsistent near the boundary. The network arm accepts 84 steps and stops at
24.720 mm penetration with a nonconverged direction projection. Neither final
endpoint passes the new criterion. All accepted steps retain old contacts, and
saved network weights reproduce the final q exactly on reload. This is a failed
replacement experiment with corrected attraction semantics, not a validated
production loss or proof that widening contact acceptance solves optimization.
