# Whole-foot navigation distance

`supermimic.contact_navigation` owns the retained online distance implementation.
It is a research metric for top contacts on one grounded, upright box. It is not
the default contact-training loss or a complete online contact planner.

## Interface and ownership

```python
from supermimic.contact_navigation import WholeFootBoxNavigation

metric = WholeFootBoxNavigation(boxes, goals, vertices, material_local, device)
# boxes: one [xyz, wxyz, size] box extending to the z=0 floor.
# goals / material_local: four corresponding world/body-local sole points.
# vertices: the realized canonical collision mesh in the same foot frame.
distance = metric.distance(points, position, rotation)  # [batch, 4], metres
loss = distance.square().mean() / 0.05**2

metric.set_goal(next_goals)  # reuse terrain graph; no voxel reconstruction
```

Supply the canonical G1 collision mesh through the existing geometry pipeline.
Do not infer solver contact, stability, support or success from this distance.
Those decisions continue to use allocated native Newton contacts and the shared
contact semantics. The metric does not change margins or physics.

Each material XY reference has a conservative horizontal radius enclosing every
mesh vertex under rotation. All references use the actual mesh minimum Z as the
foot base. Inflating the obstacle by these envelopes reduces position navigation
to four three-dimensional queries. Cached visibility graphs contain 96 nodes per
reference; source and goal connections update online. Paths are graph
approximations, not exact configuration-space shortest paths.

Overlapping envelopes use a nearest allowed side/top exit extension. This supplies a
correction for a penetrated or conservatively overlapping query without treating
the solid interior as free space. The box bottom is not an exit because it lies
below the floor; this prevents nonfinite routes for low overlapping queries. Box underpasses are rejected because vertical
occupancy below a floating obstacle is not represented. Other terrain layouts
and contact surfaces require a separate representation; no fallback is selected.

## Verified scope and remaining work

On the retained `climb_58` case, 256 translated top-contact targets produced 1,024
clear reference paths, including 512 routed paths. Position and goal gradients
passed finite-difference checks. A single target update/query took about 2.67 ms
end to end in that run; this measures the query module, not a complete controller.

In a fixed-goal comparison from the same checkpoint and Adam state, both the
voxel and online versions reached zero outside-region collisions for all 53
samples at update 26. At update 60, fixed material-point error was about 13.07 cm,
foot tilt about 73.6 degrees, and final stable-contact success remained 0/53.

The navigation references retain material XY and whole-foot minimum Z, rather
than every material point's height. Final pose completion remains unresolved.
The online query is validated for the stated scene and translated targets;
policy execution with changing targets has not been trained in this comparison.

Local models, optimizer states, source receipts, reports and replay assets are
indexed by `tmp/contact-navigation-current/scheme.json`. They are diagnostic
outputs, not distributed installation prerequisites. The maintained module and
its unit tests do not import or load temporary artifacts.

## Verification

Run from the repository root in the configured workflow environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/somaforge_core:packages/supermimic \
  runtime/environments/venv/workflow/bin/python -m pytest -q \
  packages/supermimic/tests/test_contact_navigation.py -p no:cacheprovider
```

The tests exercise online goal updates, blocked routes, finite-difference
agreement, whole-mesh envelope coverage, pose derivatives and scene scope.
