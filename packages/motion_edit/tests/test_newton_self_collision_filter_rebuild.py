from __future__ import annotations

import sys
from types import SimpleNamespace

from somaforge_core.newton_collision_filter import (
    apply_self_collision_filters_to_builder,
)


class _Builder:
    def __init__(self) -> None:
        root = "/World/envs/env_0/Robot/Geometry/pelvis"
        self.body_label = [
            root,
            f"{root}/left_hip_pitch_link",
            f"{root}/left_hip_pitch_link/left_knee_link",
            f"{root}/torso_link/left_sphere_hand_link",
        ]
        self.body_count = len(self.body_label)
        self.shape_body = [0, 1, 2, 3, -1]
        self.body_shapes = {index: [index] for index in range(4)}
        self.joint_parent = [0, 1, 0]
        self.joint_child = [1, 2, 3]
        self.joint_type = [1, 1, 1]
        # Simulate the USD importer disabling all articulation self-collision.
        self.shape_collision_filter_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 4),  # robot/environment filter must survive
        ]

    def add_shape_collision_filter_pair(self, first: int, second: int) -> None:
        self.shape_collision_filter_pairs.append(
            (min(first, second), max(first, second))
        )


def test_rebuild_removes_global_self_filter_but_keeps_neighbours(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "newton",
        SimpleNamespace(JointType=SimpleNamespace(FIXED=0)),
    )
    builder = _Builder()

    apply_self_collision_filters_to_builder(
        builder,
        exclude_kinematic_distance=3,
    )

    filters = set(builder.shape_collision_filter_pairs)
    assert (0, 4) in filters
    assert (0, 1) in filters
    assert (1, 2) in filters
    assert (0, 3) in filters
    # Hip/knee and hand are separate branches below the root, so even a short
    # undirected path must not suppress their real cross-branch collision.
    assert (1, 3) not in filters
    assert (2, 3) not in filters


def test_rebuild_keeps_filters_between_shapes_on_same_body(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "newton",
        SimpleNamespace(JointType=SimpleNamespace(FIXED=0)),
    )
    root = "/World/envs/env_0/Robot/Geometry/pelvis"
    builder = _Builder()
    builder.body_label = [root, f"{root}/left_sphere_hand_link"]
    builder.body_count = 2
    builder.shape_body = [0, 0, 1]
    builder.body_shapes = {0: [0, 1], 1: [2]}
    builder.joint_parent = [0]
    builder.joint_child = [1]
    builder.joint_type = [1]
    builder.shape_collision_filter_pairs = [(0, 1), (0, 2), (1, 2)]

    apply_self_collision_filters_to_builder(
        builder,
        exclude_kinematic_distance=-1,
    )

    assert set(builder.shape_collision_filter_pairs) == {(0, 1)}


def test_rebuild_filters_fixed_joint_siblings_as_one_component(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "newton",
        SimpleNamespace(JointType=SimpleNamespace(FIXED=0)),
    )
    root = "/World/envs/env_0/Robot/Geometry/ankle_roll"
    builder = _Builder()
    builder.body_label = [
        root,
        f"{root}/sphere_1",
        f"{root}/sphere_2",
    ]
    builder.body_count = 3
    builder.shape_body = [0, 1, 2]
    builder.body_shapes = {0: [0], 1: [1], 2: [2]}
    builder.joint_parent = [0, 0]
    builder.joint_child = [1, 2]
    builder.joint_type = [0, 0]
    builder.shape_collision_filter_pairs = []

    apply_self_collision_filters_to_builder(
        builder,
        exclude_kinematic_distance=3,
    )

    assert set(builder.shape_collision_filter_pairs) == {
        (0, 1),
        (0, 2),
        (1, 2),
    }


def test_rebuild_supports_direct_urdf_body_labels(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "newton",
        SimpleNamespace(JointType=SimpleNamespace(FIXED=0)),
    )
    builder = _Builder()
    builder.body_label = [
        "pelvis",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_sphere_hand_link",
    ]

    apply_self_collision_filters_to_builder(
        builder,
        exclude_kinematic_distance=3,
    )

    filters = set(builder.shape_collision_filter_pairs)
    assert (0, 1) in filters
    assert (1, 2) in filters
    assert (1, 3) not in filters
    assert (2, 3) not in filters
