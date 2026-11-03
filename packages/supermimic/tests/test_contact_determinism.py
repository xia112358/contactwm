from types import SimpleNamespace
import numpy as np
import warp as wp
import pytest
from supermimic.contact_determinism import CanonicalContacts


def fixture(device,capacity=4):
    dev=wp.get_device(device);n=8
    contact=SimpleNamespace(worldid=wp.array([1,0,1,0,0,0,0,0],dtype=int,device=dev),
        geom=wp.array([[2,3],[0,1],[2,3],[0,1]]+[[0,0]]*4,dtype=wp.vec2i,device=dev),
        pos=wp.array([[2,0,0],[1,0,0],[1,0,0],[0,0,0]]+[[0,0,0]]*4,dtype=wp.vec3,device=dev),
        dist=wp.zeros(n,device=dev),frame=wp.array(np.tile(np.eye(3),(n,1,1)),dtype=wp.mat33,device=dev),
        efc_address=wp.array(np.arange(n*4,dtype=np.int32).reshape(n,4),dtype=int,device=dev))
    d=SimpleNamespace(nworld=2,naconmax=n,nacon=wp.array([4],dtype=int,device=dev),contact=contact)
    solver=SimpleNamespace(mjw_data=d,device=dev,_contact_tid_to_cid=wp.array([0,1,2,3,-1,-1,-1,-1],dtype=int,device=dev),_invalidate_contact_fast_path=lambda:None,_convert_contacts_to_mjwarp=lambda:None)
    return solver,CanonicalContacts(solver,per_world_capacity=capacity)


def test_canonical_order_and_mapping():
    s,c=fixture('cpu');c.convert();c.check()
    np.testing.assert_array_equal(s.mjw_data.contact.worldid.numpy()[:4],[0,0,1,1])
    np.testing.assert_array_equal(s.mjw_data.contact.pos.numpy()[:4,0],[0,1,1,2])
    np.testing.assert_array_equal(s._contact_tid_to_cid.numpy()[:4],[3,1,2,0])
    np.testing.assert_array_equal(s.mjw_data.contact.efc_address.numpy()[:4,0],[12,4,8,0])


def test_overflow_is_explicit():
    s,c=fixture('cpu',capacity=1);c.convert()
    with pytest.raises(RuntimeError,match='capacity'):c.check()


def test_cuda_graph_capture():
    if not wp.is_cuda_available():pytest.skip('CUDA required')
    s,c=fixture('cuda:0');c.convert();wp.synchronize()
    with wp.ScopedCapture(device=s.device) as cap:c.convert()
    wp.capture_launch(cap.graph);wp.synchronize();c.check()
    np.testing.assert_array_equal(s.mjw_data.contact.pos.numpy()[:4,0],[0,1,1,2])
