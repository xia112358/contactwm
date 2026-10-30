import copy
import numpy as np
import pytest
from somaforge_core.contact_face_selection import select_contact_pairs, process_face_contacts

CATALOG = [dict(surface=8, normal_w=[0,0,1]), dict(surface=2, normal_w=[1,0,0])]


def pair(surface, depth=-.001):
    return dict(part=0, surface=surface, allocated=True, dist=.02+depth,
                includemargin=.02, position_w=[1,2,3], surface_candidates=[2,8])


def test_mixed_part_keeps_top_and_preserves_raw():
    raw = [[pair(2, -.01), pair(8)]]
    before = copy.deepcopy(raw)
    result = select_contact_pairs(raw, CATALOG)
    assert result['contact_surface'][0,0] == 8
    assert result['contact_pairs'] == [[raw[0][1]]]
    assert result['abnormal_contact_pairs'] == [[raw[0][0]]]
    assert raw == before


def test_side_candidate_is_not_promoted_and_breaks_effective_run():
    result = process_face_contacts([[pair(8)], [pair(2)], [pair(8)]], CATALOG, fps=50)
    np.testing.assert_array_equal(result['contact_part_mask'][:,0], [True,False,True])
    assert result['contact_surface'][1,0] == -1
    assert not result['contact_position_w'][1].any()
    assert result['contact_label_contract']['face_selection'] == 'primary_horizontal_upward_faces_v1'


@pytest.mark.parametrize('change', [dict(surface=99), dict(allocated=False), dict(dist=.021)])
def test_invalid_evidence_fails_closed(change):
    p = pair(8); p.update(change)
    with pytest.raises(ValueError):
        select_contact_pairs([[p]], CATALOG)
