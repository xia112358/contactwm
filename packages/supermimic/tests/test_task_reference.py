import copy,hashlib,json
import pytest
from supermimic.task_reference import replace_motion_references


def fixture(tmp_path):
    csv=tmp_path/'new.csv';csv.write_text('0,1,2\n')
    task=dict(name='old',terrain_usd_sha256='terrain',source_terrain_sha256='terrain',boxes=[[0,0,0,1,0,0,0,1,1,1]],csv=str(csv),csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
    spec=dict(tasks=[task],robot_asset={'identity':'same'},motion_dir=str(tmp_path))
    refs=copy.deepcopy(spec);refs['tasks'][0]['name']='new_short_window'
    manifest=tmp_path/'references.json';manifest.write_text(json.dumps(refs))
    return spec,refs,manifest


def test_new_window_name_preserves_physical_slot(tmp_path):
    spec,refs,manifest=fixture(tmp_path)
    provenance=replace_motion_references(spec,manifest)
    assert spec['tasks'][0]['name']=='new_short_window'
    assert provenance['reference_tasks'][0]['csv_sha256']==refs['tasks'][0]['csv_sha256']


def test_changed_terrain_is_not_a_reference_replacement(tmp_path):
    spec,refs,manifest=fixture(tmp_path);refs['tasks'][0]['boxes'][0][-1]=2;manifest.write_text(json.dumps(refs))
    with pytest.raises(ValueError,match='terrain'):replace_motion_references(spec,manifest)


def test_modified_reference_csv_is_rejected(tmp_path):
    spec,refs,manifest=fixture(tmp_path);(tmp_path/'new.csv').write_text('3,4,5\n')
    with pytest.raises(ValueError,match='CSV changed'):replace_motion_references(spec,manifest)
