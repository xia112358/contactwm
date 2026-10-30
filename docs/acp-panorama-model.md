# Panoramic contact planner

`holosoma.acp.panorama_model.PanoramaContactPlanner` is an experimental ACP-style
planner for the historical full-sphere surface observation. It is separate from
`ContactPlanner`, which retains the paper's 17 × 11 height-map architecture and
checkpoint compatibility. This model has 294,225 parameters with 29 joints.
It is implemented and tested, but not trained or wired into the policy loop.

The inputs are five steps of measured proprioception `[B,465]`, a base-yaw-frame
goal `[B,3]`, metric depth, validity, age and optional confidence `[B,64,128]`,
base and virtual-view world positions `[B,3]`, and base yaw `[B]`. Age is in
seconds; confidence is a heuristic score in [0,1]. Pixel centers use longitude
−π to π and elevation π/2 to −π/2, matching `SurfacePanorama`. The virtual view
is gravity aligned: roll and pitch are intentionally absent from this interface.
The sensor's full scan-time pose is used upstream for fusion; it is distinct
from both the virtual view and the robot base.

Two spherical convolution layers encode depth, validity, age and confidence.
Longitude padding wraps; elevation padding does not join the poles. Local XYZ
embeddings are concatenated with image features to form 96-dimensional tokens.
Masked pooled context, robot history and goal condition six body queries via
FiLM. Eight-head cross-attention generates per-body surface affordances and an
occupancy head. The default six-body order is left/right foot, knee, hand. `num_contacts=6` is
saved in the model config; an explicit nine-body option exists for compatibility
experiments, and weights with a different body count must not be loaded silently.
This architecture is a proposed adaptation, not a Table S3 parameter match.

Decoding selects one observed pixel per body. It never averages separate
surfaces or regresses free XYZ. Outputs include contact probability, occupancy,
selected pixel index, base-yaw and world-space targets, and ungated candidates.
The discrete argmax is trained via attention supervision, not through a
coordinate-regression gradient. Targets are intended contacts, not physical
contact truth, and selection alone does not establish reachability or support.

Unknown pixels are masked before encoding and attention. Stale observations
beyond `max_age` are unknown. All-empty observations return zero probability,
no contact and selected index −1 without NaNs. Invalid pixels may carry NaN/inf
placeholders; known pixels require finite, in-range data. Zero output positions
must always be interpreted with the occupancy/availability masks.

```python
from holosoma.acp.panorama_model import (
    PanoramaContactPlanner, panorama_planner_loss,
)

model = PanoramaContactPlanner(num_joints=29)
prediction = model(history, goal, depth, valid, age,
                   base_position_w, view_origin_w, yaw, confidence)
losses = panorama_planner_loss(prediction, target_w, occupancy, known)
losses['loss'].backward()
```

The loss combines Gaussian attention cross-entropy with occupancy BCE. Positive
labels whose world targets have no observed point within 12 cm are masked from
both losses; unknown labels and empty scans are masked too. The support radius
and Gaussian bandwidth are configurable observation-association parameters,
not definitions of physical contact. Supervision must come from the existing
native-contact label pipeline; the new model does not fabricate contact labels.
A full labelled panorama training dataset and policy integration remain to do.

A model-only checkpoint can store `config` and `state_dict`; reconstruct with
`PanoramaContactPlanner(**checkpoint['config'])` before loading weights. Actual
training checkpoints additionally need optimizer and progress state. Old ACP
height-map checkpoints are not interchangeable with this model.

Validation: seven tests cover exact surface selection and world backprojection,
translation consistency, unknown/empty scans and finite gradients, unseen
positive-label masking, longitude wrap, age expiry, checkpoint round-trip and
fixed-batch optimizer loss reduction. A separate diagnostic used the actual
0, 7.2 and 15.98-second climbing panoramas at full 64 × 128 resolution, with
zero history/goal solely to check the interface. Its random-initialized targets
are not trained predictions. The report and script are in
`tmp/acp-panorama-model/`; no successful robot control is implied.
