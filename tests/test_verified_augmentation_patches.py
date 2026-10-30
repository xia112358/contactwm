from types import SimpleNamespace
import pytest
from motion_edit.contact.layers import verified_source_patches

def graph():
    a=SimpleNamespace(anchor_id='a',start_frame=0,end_frame=2,metadata={'newton_shape_label':'foot'})
    p=SimpleNamespace(anchor_id='a',metadata={'source_target_contract':'newton_robot_geometry_point_trajectory'},
        source_target_frames=[0,1],source_points_local_by_frame=[[[0,0,0]],[[0,0,0]]],
        source_target_points_w=[[[1,2,3]],[[4,5,6]]])
    return SimpleNamespace(anchors=[a],patches=[p])

def test_verified_patches_not_reconstructed():
    g=graph();patches,summary=verified_source_patches(g,source_motion_path='motion.npz')
    assert patches[0] is g.patches[0]
    assert summary['contact_point_samples']==2 and not summary['legacy_raw_contacts_used']

def test_no_force_gate_or_legacy_patch():
    with pytest.raises(ValueError):verified_source_patches(graph(),source_motion_path='motion.npz',min_force_norm=1)
    g=graph();g.patches[0].metadata={}
    with pytest.raises(ValueError):verified_source_patches(g,source_motion_path='motion.npz')
