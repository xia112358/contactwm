import pytest
from somaforge_core.contact_face_selection import ground_top_catalog


def records():
    return [dict(surface_id='box_top',normal=[0,0,1],origin=[0,0,.7]),
            dict(surface_id='box_side',normal=[1,0,0],origin=[0,0,.35]),
            dict(surface_id='box_bottom',normal=[0,0,-1],origin=[0,0,0]),
            dict(surface_id='terrain_ground_z0',normal=[0,0,1],origin=[0,0,0])]


def test_side_cannot_overwrite_top_regardless_of_order():
    for rows in (records(),list(reversed(records()))):
        catalog=ground_top_catalog(rows)
        assert catalog[1]['surface_id']=='box_top'
        assert catalog[1]['normal']==[0,0,1]
        assert catalog[1]['origin'][2]==.7
        assert catalog[0]['surface_id']=='terrain_ground_z0'


def test_missing_ambiguous_or_duplicate_top_fails_closed():
    with pytest.raises(ValueError):ground_top_catalog(records()[1:])
    with pytest.raises(ValueError):ground_top_catalog(records()+[dict(surface_id='another_top',normal=[0,0,1],origin=[0,0,.8])])
    with pytest.raises(ValueError):ground_top_catalog(records()+[records()[0]])
