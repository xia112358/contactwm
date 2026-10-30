import json

import numpy as np
import pytest
import torch

from generator.fullbody_dataset import _local_box
from generator.newton_query_supervision import validate_native_box_scene
from somaforge_core.contact_dataset import contact_dataset_fingerprint, validate_contact_cache


def native_scene():
    corners = [[-1., -2., .7], [1., -2., .7], [1., 2., .7], [-1., 2., .7]]
    return dict(surface_catalog=[dict(surface=0, normal_w=[0., 0., 1.], plane_offset=0.),
            dict(surface=1, normal_w=[0., 0., 1.], plane_offset=.7)],
        shape_surface={'0': [dict(surface=1, triangles_w=[corners[:3], [corners[0], corners[2], corners[3]]])]})


def scene_values():
    return (torch.tensor([[0., 0., .35]]), torch.eye(3)[None],
            torch.tensor([[1., 2., .35]]), torch.tensor([0.]))


def test_native_scene_validates_outline_and_both_height_planes():
    assert validate_native_box_scene(*scene_values(), native_scene()) < 1e-6


@pytest.mark.parametrize('field,delta', [(0, .1), (2, .1), (3, .1)])
def test_wrong_cached_center_size_or_ground_is_rejected(field, delta):
    values = list(scene_values()); values[field] = values[field]+delta
    with pytest.raises(ValueError, match='differs from actual Newton'):
        validate_native_box_scene(*values, native_scene())


def test_missing_native_triangles_cannot_silently_skip_geometry_validation():
    scene = native_scene(); scene['shape_surface'] = {}
    with pytest.raises(ValueError, match='Missing actual Newton top triangles'):
        validate_native_box_scene(*scene_values(), scene)


def test_source_loader_ignores_side_before_top_and_catalog_order(tmp_path):
    top = dict(surface_id='box_top', normal=[0., 0., 1.], origin=[0., 0., .7],
        metadata=dict(polygon_world=[[-1., -2., .7], [1., -2., .7], [1., 2., .7], [-1., 2., .7]]))
    side = dict(surface_id='box_side', normal=[1., 0., 0.], origin=[1., 0., .35])
    ground = dict(surface_id='terrain_ground_z0', normal=[0., 0., 1.], origin=[0., 0., 0.], metadata=dict(ground_z=0.))
    path = tmp_path/'surfaces.jsonl'
    expected = None
    for records in ([side, ground, top], [top, side, ground], [ground, side, top]):
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        result = _local_box(dict(metadata=dict(target_surface_catalog=str(path))), np.zeros(3), np.eye(3))
        if expected is None: expected = result
        for a, b in zip(result, expected, strict=True): np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(result[0], [0., 0., .35])
        np.testing.assert_allclose(result[1][:, 2], [0., 0., 1.])


def test_catalog_change_invalidates_cache_even_when_labels_and_clock_do_not_change(tmp_path):
    motion = tmp_path/'motion.npz'; np.savez(motion, fps=50.)
    labels = tmp_path/'labels.npz'; labels.write_bytes(b'Newton labels unchanged')
    terrain = tmp_path/'terrain.obj'; terrain.write_text('unchanged terrain')
    catalog = tmp_path/'surfaces.jsonl'; catalog.write_text('catalog before correction')
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps(dict(metadata=dict(target_surface_catalog=str(catalog)))))
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(
        motion_files=[dict(motion_id='demo', motion_file=str(motion), newton_contact_file=str(labels),
            terrain_id='scene', edit_plan_file=str(plan))], terrains=[dict(terrain_id='scene', terrain_file=str(terrain))])))
    fingerprint = contact_dataset_fingerprint(manifest)
    catalog.write_text('corrected catalog')
    with pytest.raises(ValueError, match='rebuild cache'):
        validate_contact_cache(dict(contact_dataset_fingerprint=fingerprint), manifest)
