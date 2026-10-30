"""CPU checks for read-only binding of the already initialized Newton model."""
from types import SimpleNamespace

import numpy as np
import pytest

from somaforge_core.native_model_binding import inspect_model, native_binding


class ModelArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


def realized_model():
    triangle = SimpleNamespace(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        indices=np.array([0, 1, 2], dtype=np.int32),
    )
    model = SimpleNamespace(
        body_label=["/World/envs/env_0/Robot/left_ankle_roll_link"],
        shape_label=["robot", "ground", "top"], joint_label=["root"],
        shape_collision_filter_pairs={(0, 1)},
        shape_source=[None, triangle, triangle], world_count=1,
    )
    arrays = dict(
        shape_body=[0, -1, -1], shape_type=[1, 7, 7],
        shape_transform=[[0., 0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 0., 0., 1.],
                         [4., 5., 1., 0., 0., 2**-.5, 2**-.5]],
        shape_scale=[[1., 1., 1.], [1., 1., 1.], [2., 3., 1.]],
        shape_flags=[1, 1, 1], shape_margin=[.01, .01, .01],
        shape_gap=[0., 0., 0.], shape_collision_group=[1, 1, 1],
        joint_q_start=[0], joint_type=[4], joint_parent=[-1], joint_child=[0],
        joint_X_p=[[0., 0., 0., 0., 0., 0., 1.]],
        joint_X_c=[[0., 0., 0., 0., 0., 0., 1.]],
        body_world=[0], shape_world=[0, -1, -1], joint_world=[0],
    )
    for name, value in arrays.items():
        setattr(model, name, ModelArray(value))
    return model


def test_realized_mesh_transform_preserves_ground_and_top_attribution():
    model = realized_model()
    info = inspect_model(model)
    binding = native_binding(model, info)
    assert binding["model_fingerprint"] == info["model_fingerprint"]
    assert binding["body_env"] == {0: 0}
    assert binding["surface_catalog"] == [
        dict(surface=0, normal_w=(0., 0., 1.), plane_offset=0.),
        dict(surface=1, normal_w=(0., 0., 1.), plane_offset=1.),
    ]
    top = binding["shape_surface"][2][0]
    assert top["surface"] == 1 and top["triangle_indices"] == [0]
    assert top["attribution_schema"] == "native_triangle_nearest_v1"
    np.testing.assert_allclose(top["triangles_w"], [[[4, 5, 1], [4, 7, 1], [1, 5, 1]]], atol=1e-14)
    assert info["terrain_mesh_bounds"][1]["z_levels"] == [1.]
    # The model is consumed read-only: transformations never overwrite vertices.
    np.testing.assert_array_equal(model.shape_source[2].vertices[1], [1, 0, 0])


def test_fingerprint_tracks_realized_collision_settings_and_geometry():
    model = realized_model()
    original = inspect_model(model)["model_fingerprint"]
    assert original == inspect_model(model)["model_fingerprint"]
    model.shape_margin.value[0] += .001
    changed_settings = inspect_model(model)["model_fingerprint"]
    assert changed_settings != original
    model.shape_source[2].vertices[1, 0] += .1
    assert inspect_model(model)["model_fingerprint"] != changed_settings


def test_inspection_rejects_missing_realized_field():
    model = realized_model()
    model.shape_gap = None
    with pytest.raises(ValueError, match="missing Newton model field shape_gap"):
        inspect_model(model)


def test_binding_rejects_unknown_world_and_non_mesh_terrain():
    model = realized_model()
    info = inspect_model(model)
    info["body_world"] = [2]
    with pytest.raises(ValueError, match="one canonical robot"):
        native_binding(model, info)
    info["body_world"] = [0]
    model.shape_source[1] = None
    with pytest.raises(ValueError, match="static triangle-mesh terrain"):
        native_binding(model, info)


def test_binding_can_use_environment_labels_without_optional_world_array():
    model = realized_model()
    info = inspect_model(model)
    info.pop("body_world")
    assert native_binding(model, info)["body_env"] == {0: 0}
