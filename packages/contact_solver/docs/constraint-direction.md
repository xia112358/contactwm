# Constraint-aware update directions

This experiment separates pose-space coordination from network parameter coupling.
It uses the existing residuals, AL objective, normalized root cost (1000 relative to
joints), per-component feasibility filter, and authoritative Newton contact audit.
No QP endpoint is loaded or used as a target.

Three arms start from the same network prediction:

- `pose_al`: AL descent in the normalized floating-base pose chart, with protection.
- `pose_projected`: the same descent, projected onto local protected halfspaces.
- `network_projected`: pull back the objective and protected residual derivatives
  through the actual network output, then project in network parameter space.

For protected signed residual r <= t, activate an outward halfspace when r is
within the configured fraction of t or already violates t. Equality residuals
have two signed sides. The local direction satisfies a d <= 0 for activated rows.
This permits inward motion; it does not lock all protected contacts as equalities.
If a rejected trial exposes an inactive blocker, activate that row and recompute
the direction at the unchanged accepted state. This is an optimizer operation,
not a new scalar loss or proof that an ordinary loss inherits these guarantees.

`constraint_direction.project_direction` uses dual coordinate descent on the
small row Gram matrix to project the proposed direction. The solve uses double
precision and scales by the proposed direction norm; coordinate updates run on
the CPU. It reports convergence
and residual violation. Every actual candidate must still decrease the AL
objective under Armijo and pass the nonlinear filter and actual Newton evidence.
Missing contact evidence or missing protected residuals are never accepted.

The trace distinguishes three quantities for active rows:

1. `variable_space_linear_change`: derivative prediction for the proposed pose or
   parameter update.
2. `linear_change`: residual Jacobian applied to the *actual* pose chart change.
3. `actual_change`: freshly evaluated residual difference.

The first versus second isolates output-map nonlinearity; the second versus third
isolates residual/FK nonlinearity and refreshed witness changes. These are local
finite-step diagnostics, not a proof of global feasibility or generalization.

Configuration: `configs/contact_learning/step14_direction.json`.
Run using the existing `step14_loss_direction` fixture with `CONSTRAINT_LEARNING=1`,
`DIRECTION_EXPERIMENT=1` and `CONSTRAINT_CASE_CONFIG` pointing to this config.
The MHA fast path is disabled in this experiment to make gradient-enabled and
inference forward evaluations agree; the mismatch is recorded in the report.

## Interior-reserve variant

`step14_direction_interior.json` adds an inward target for active rows:

    A_i d <= -min(0.1 t_i, max(0, r_i - 0.8 t_i)) / initial_step_size

The same formula applies to all configured protected constraints. It leaves a
small interior buffer rather than insisting on a purely tangent direction at a
finite-step boundary. The acceptance tolerances and Newton evidence rules are
unchanged. Already violated constraints request only a bounded inward increment,
not immediate feasibility. Both projected arms use the same rule; the AL control
arm is unchanged. The affine local problem may be infeasible, and convergence
flags must be inspected; the nonlinear guard remains the acceptance authority.

The two perturbation configs use the existing +1 cm input-root X and +0.02 rad
input-hip-pitch cases. They remain static, independently optimized instances,
not a trained model's held-out generalization test.

## Nominal-frame findings (2026-09-26)

The valid double-precision baseline (`tmp/constraint_direction_20260926_v2`)
ended at 89.17 mm foot interior depth for protected pose AL, 35.87 mm for
pose-space tangent projection, and 33.25 mm for parameter-space tangent
projection. All retained the original actual contacts. Pose tangent projection
stalled when the knee upper-gap residual sat exactly on its guard: a full step
violated it by a nonlinear amount, while the smallest trial exceeded the cap by
about 1.1e-14 in normalized units. No acceptance epsilon was added to hide this.

With the generic inward-reserve rule (`tmp/constraint_direction_20260926_interior`),
the corresponding projected endpoints were 1.946 mm and 3.838 mm after a
120-iteration budget. The network accepted 119 updates; one iteration activated
an additional blocker. Maximum joint changes were 20.38 and 17.65 degrees;
root translations were 24.44 and 36.70 mm. Both retained hand/knee contacts and
activated the left-foot contact according to Newton. Joint-bound violations were
below the existing 1e-6 rad acceptance tolerance. Neither passed full static
recovery: penetration remains above the existing thresholds. AL, contact
activation and task residual satisfaction are distinct metrics.

The original float32 projection pilot (`..._v1`) contained unconverged local
projections and is kept as an audit artifact, not the main comparative result.
The v2 and inward-reserve nominal runs began from the same prediction; the v1
initial prediction differs slightly. All arm comparisons within a run share the
same initial prediction. The numerical fix alone did not solve the problem.

This supports constraint-aware updates with interior reserve as a useful next
research direction. It does not establish ordinary scalar-loss sufficiency,
full feasibility, or held-out generalization. The parameter-space operation is a
small constrained direction solve; it must not be marketed as removing all
optimization machinery merely because no QP pose teacher is used.

## Perturbation results

With the same inward-reserve rule, input hip-pitch +0.02 rad reached 1.219 mm
(pose) and 2.661 mm (network) foot interior depth. Input root X +1 cm was harder:
4.133 mm (pose) versus 22.431 mm (network), with network joint-limit violation
still 0.01495 rad (initially 0.02650 rad). Old actual contacts were retained in
both cases, but neither case passed full static recovery.

The root case exposed missing protected self-collision residuals when Newton
query pairs disappeared. The guard correctly refuses to infer safety from
missing rows. Stable pair-distance evidence is a separate adapter problem that
must be addressed before attributing this stopping condition solely to gradient
coordination. The hip-network arm had one approximate projection at step 33
(relative violation 1.38e-8 after 4000 sweeps); its nonlinear checks passed.

Detailed results and curves: `tmp/constraint_direction_summary_20260926/RESULTS.md`.
All 31 relevant tests passed. Per-case corrections in the summary are relative
to that case's initial network output, unlike legacy record displacement fields
which reference the original nominal fixture.
