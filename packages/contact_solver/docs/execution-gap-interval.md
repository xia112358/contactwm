# Predictor execution gap interval

The current full1000 unified training objective uses a band from zero to the
actual Newton `includemargin`, not a target distance or the interval midpoint.
Its contact residual and gradient vanish inside that band. Complete-solid
penetration, endpoint location and keep-material motion are separate residuals;
they must not be reported as a failure of the contact-distance band.

The predictor's ordinary CPU and GPU plan-realization loss uses
`-DEFAULT_ACCEPTANCE.shallow_penetration_m <= d <= actual includemargin`.
This is an optimization interval, not a new contact predicate. Newton's strict
activation test and allocation/primary-face checks remain authoritative;
zero loss at the upper boundary is not proof of contact.

The intended part/surface term penalizes `relu(d - includemargin)^2`, taking
the minimum over matching witnesses (one witness can establish a part's contact).
It no longer attracts inactive contacts toward `0.05 * includemargin`, and
does not switch based on frozen activation flags. Each witness uses its own margin.

The full-body term penalizes depth beyond the existing accepted penetration
tolerance, including other witnesses on the intended part and self collisions.
This is the lower interval bound; it is not added again to the intended-contact
term. Existing loss normalizations and aggregation settings are retained.
The optional body_mean_plus_max mode still provides its explicitly configured
additional per-body aggregation; the current training uses max.

Absent witnesses remain missing, not realized. Finite-face geometric guidance
approaches the configured margin's upper bound, clips negative normal residuals,
and retains tangential region guidance. Newton must re-query to establish truth.
The approach weight, pose prior, planner, acceptance criteria, physics margins,
assets, and standalone projector's legacy buffered mode are unchanged.

Regression tests cover upper/lower gradient signs, zero gradients inside the
interval, activation-flag independence, per-pair margins, missing-pair guidance,
and the scalar right-foot distance recorded at recursive step 2 (24.049 mm).
These are frozen-witness tests, not an end-to-end network recovery experiment.

The shared regional objective and Motion Edit's missing-region proxies also use
the actual `includemargin` upper bound (`native_activation_upper_gap_v1`). Their
penetration lower bound remains zero; this does not change the predictor's
existing shallow-penetration allowance. There is no regional 1 mm target mode.

The unified predictor objective also retains a fixed, training-only Newton
material field after native candidates appear. A candidate cannot switch off
the intended upward recovery of an embedded region. Its loss-only interval is
`native_margin_finite_upward_face_material_band_v1`: the native-width
neighborhood of the finite face, restricted to its upward side. Edge and corner
distance combines normal and tangential separation before applying the actual
margin; it does not require the contact point's projection to lie strictly
inside a rectangle. Newton activation, allocation and primary-face truth are
unchanged. Complete solids still supply physical penetration independently.

Source-face ownership uses `newton_source_triangle_witness_normal_fan_v2` on
both CPU and GPU. The actual terrain witness identifies a feature of its
encoded source triangle; only incident faces containing that feature compete
by normal alignment. A point inside a box side cannot be assigned to its top
merely because the source triangle has a top vertex. Projection is metadata
for this ownership check: physical witnesses, poses and Newton activation or
allocation are preserved. Device metadata records the ownership version, and
old or missing versions fail explicitly instead of silently reusing old workers.

The full1000 position term now uses
`native_part_surface_to_normal_interval_rms_v1`, shared with endpoint position
checks. It measures native part mesh triangles and analytic sphere surfaces
against the vertical segment at the issued XY location, from the actual face
plane to its configured margin. Anatomical regions constrain the part geometry.
The existing global RMS aggregation and 4 cm tolerance remain. Unlike the old
arbitrary-witness XY error, normal excess contributes outside the interval;
this metric change is explicit in new dataset/checkpoint metadata. It has no
point or midpoint attraction inside its allowed range and remains defined when
Newton has no candidate. Actual activation, allocation and primary-face/region
coverage still independently determine contact completeness. Complete-solid
penetration and material keep constraints remain necessary.

The geometry scalar runs in CPU float64, with ordinary Torch transfers carrying
its derivative to the predicted pose. Sphere rotation alone cannot generate a
location gradient. No QP correction, target pose, or custom backward is used.
Initialized scene metadata is exported alongside the solids for both HTTP and
tensor consumers; missing metadata is an error, never a witness-loss fallback.

Position geometry is batched over poses. Whole spheres use an exact skin-to-
segment formula (including segments inside the sphere); clipped spheres retain
the same circle/intersection candidates and ordinary autograd. Scenes with
identical initialized robot shapes share a geometry/FK batch, while each sample
retains its own scene plane and configured margin. Different robot shapes stay
in separate batches. This changes execution cost, not the scalar definition,
contact predicate, or precision. The scalar sphere routine remains a test oracle.

Mesh distances select all minimizing triangles without a graph, then recompute
their exact distances with ordinary autograd. Tied minima share derivatives
equally; selected triangles remain differentiable geometry, not frozen witness
points. This reduces backward graph size without changing the distance scalar.
