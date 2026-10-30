> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# Canonical 8-Part Contact Force

Production contact-force references are collected from successful Isaac Lab
3/Newton policy rollouts of the canonical spherehand USD. Historical force
references from other robot assets or backends are invalid.

The canonical part order is:

```text
LHEE, LTOE, RHEE, RTOE, LH, RH, LK, RK
```

The required arrays are:

```text
contact_force_part_w:          [T, 8, 3]
contact_force_part_mask:       [T, 8]
contact_force_part_position_w: [T, 8, 3]
contact_force_part_order:      [8]
```

Heel and toe are assigned from the spherehand collision bodies:

```text
left_heel:  left_ankle_roll_sphere_1_link, left_ankle_roll_sphere_2_link
left_toe:   left_ankle_roll_sphere_3_link, left_ankle_roll_sphere_4_link,
            left_ankle_roll_sphere_5_link
right_heel: right_ankle_roll_sphere_1_link, right_ankle_roll_sphere_2_link
right_toe:  right_ankle_roll_sphere_3_link, right_ankle_roll_sphere_4_link,
            right_ankle_roll_sphere_5_link
```

Masks are independent. A frame may contain heel-only, toe-only, simultaneous
heel-and-toe contact, or no foot contact. Only robot-to-world Newton contacts
contribute to the production reference.

Motion Edit does not solve contact dynamics. It may retarget an existing Newton
force reference, but a production force target must be recollected by an Isaac
Lab/Newton rollout of the edited kinematic reference.
