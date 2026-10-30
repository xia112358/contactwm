"""Self-collision filters for the canonical G1 Newton model."""
from __future__ import annotations

from collections import deque

# Body pairs to exclude from self-collision (label_a, label_b).
# Covers: cosmetic links + kinematically adjacent mesh overlaps.
_EXCLUDED_BODY_PAIRS: frozenset[tuple[str, str]] = frozenset({
    # Cosmetic shells
    ("pelvis_contour_link", "left_hip_pitch_link"),
    ("pelvis_contour_link", "left_hip_roll_link"),
    ("pelvis_contour_link", "right_hip_pitch_link"),
    ("waist_support_link", "left_hip_roll_link"),
    ("logo_link", None),  # logo_link with anything — filtered via label check
    # Same-leg adjacent overlap (knee ↔ ankle mesh on same limb)
    ("left_knee_link", "left_ankle_roll_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    # Same-arm adjacent (hand tip ↔ wrist)
    ("left_sphere_hand_tip_link", "left_wrist_yaw_link"),
    ("right_sphere_hand_tip_link", "right_wrist_yaw_link"),
})

# Labels that should never collide with anything.
_ALWAYS_EXCLUDE: frozenset[str] = frozenset({
    "pelvis_contour_link",
    "logo_link",
    "waist_support_link",
    # Semantic/helper collision markers, not physical self-contact bodies.
    "left_ankle_intermediate_1_link",
    "right_ankle_intermediate_1_link",
})

_BODY_ORDER: tuple[str, ...] = (
    "pelvis", "pelvis_contour_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_intermediate_1_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "left_ankle_roll_sphere_3_link", "left_ankle_roll_sphere_4_link",
    "left_ankle_roll_sphere_5_link", "left_ankle_roll_sphere_1_link",
    "left_ankle_roll_sphere_2_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_intermediate_1_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "right_ankle_roll_sphere_3_link", "right_ankle_roll_sphere_4_link",
    "right_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_1_link",
    "right_ankle_roll_sphere_2_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "logo_link", "head_link", "waist_support_link",
    "imu_link", "d435_link", "mid360_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "left_sphere_hand_link",
    "left_sphere_hand_tip_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_sphere_hand_link",
    "right_sphere_hand_tip_link",
)

assert len(_BODY_ORDER) == 53


def _body_instance(label: object) -> tuple[str, str] | None:
    text = str(label)
    marker = "/Robot/"
    if marker in text:
        instance, _, suffix = text.partition(marker)
        return instance, suffix.rsplit("/", 1)[-1]
    leaf = text.rsplit("/", 1)[-1]
    if leaf in _BODY_ORDER:
        return "canonical_g1", leaf
    return None


def _zero_one_distances(adjacency: dict[int, list[tuple[int, int]]], start: int) -> dict[int, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        body = queue.popleft()
        base = distances[body]
        for neighbor, weight in adjacency.get(body, ()):  # fixed joints have zero kinematic distance
            candidate = base + weight
            if candidate >= distances.get(neighbor, 1 << 30):
                continue
            distances[neighbor] = candidate
            if weight:
                queue.append(neighbor)
            else:
                queue.appendleft(neighbor)
    return distances


def apply_self_collision_filters_to_builder(builder, *, exclude_kinematic_distance: int = 3) -> tuple[int, int]:
    """Rebuild same-robot filters before Newton model finalization.

    Kinematic distance counts movable joints. Fixed joints have zero cost, so
    cosmetic/fixed sub-links do not make adjacent collision geometry appear
    artificially far apart.  The USD importer may have already added every
    same-articulation shape pair when it observed stale self-collision
    metadata.  Remove those same-robot pairs first; otherwise adding the
    intended near-neighbour filters can never re-enable distant contacts such
    as hand against leg.
    """
    import newton

    body_info = [_body_instance(label) for label in builder.body_label]
    shape_body = [int(body) for body in builder.shape_body]
    retained_filters: list[tuple[int, int]] = []
    for shape_a, shape_b in builder.shape_collision_filter_pairs:
        body_a = shape_body[int(shape_a)]
        body_b = shape_body[int(shape_b)]
        if body_a == body_b:
            retained_filters.append((int(shape_a), int(shape_b)))
            continue
        first = body_info[body_a] if 0 <= body_a < len(body_info) else None
        second = body_info[body_b] if 0 <= body_b < len(body_info) else None
        if first is not None and second is not None and first[0] == second[0]:
            continue
        retained_filters.append((int(shape_a), int(shape_b)))
    builder.shape_collision_filter_pairs = retained_filters

    adjacency: dict[int, list[tuple[int, int]]] = {index: [] for index in range(builder.body_count)}
    children: dict[int, list[int]] = {index: [] for index in range(builder.body_count)}
    fixed_type = int(newton.JointType.FIXED)
    for parent, child, joint_type in zip(builder.joint_parent, builder.joint_child, builder.joint_type):
        parent = int(parent)
        child = int(child)
        if parent < 0 or child < 0:
            continue
        first = body_info[parent]
        second = body_info[child]
        if first is None or second is None or first[0] != second[0]:
            continue
        weight = 0 if int(joint_type) == fixed_type else 1
        adjacency[parent].append((child, weight))
        adjacency[child].append((parent, weight))
        children[parent].append(child)

    descendants: dict[int, set[int]] = {}
    for start in range(builder.body_count):
        seen: set[int] = set()
        stack = list(children[start])
        while stack:
            body = stack.pop()
            if body in seen:
                continue
            seen.add(body)
            stack.extend(children.get(body, ()))
        descendants[start] = seen

    excluded_body_pairs: set[tuple[int, int]] = set()
    instances: dict[str, list[int]] = {}
    for body, info in enumerate(body_info):
        if info is not None:
            instances.setdefault(info[0], []).append(body)

    for bodies in instances.values():
        body_set = set(bodies)
        for body in bodies:
            info = body_info[body]
            assert info is not None
            if info[1] in _ALWAYS_EXCLUDE:
                excluded_body_pairs.update((min(body, other), max(body, other)) for other in bodies if other != body)
                continue
            distances = _zero_one_distances(adjacency, body)
            for other, distance in distances.items():
                same_serial_chain = (
                    other in descendants[body] or body in descendants[other]
                )
                if (
                    other in body_set
                    and other > body
                    and (
                        distance == 0
                        or (
                            same_serial_chain
                            and distance <= exclude_kinematic_distance
                        )
                    )
                ):
                    excluded_body_pairs.add((body, other))
            for other in bodies:
                if other <= body:
                    continue
                other_info = body_info[other]
                assert other_info is not None
                if (info[1], other_info[1]) in _EXCLUDED_BODY_PAIRS or (
                    other_info[1], info[1]
                ) in _EXCLUDED_BODY_PAIRS:
                    excluded_body_pairs.add((body, other))

    shape_pair_count = 0
    for first, second in sorted(excluded_body_pairs):
        for shape_a in builder.body_shapes.get(first, ()):
            for shape_b in builder.body_shapes.get(second, ()):
                builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))
                shape_pair_count += 1
    return len(excluded_body_pairs), shape_pair_count
