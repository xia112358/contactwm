import numpy as np
import pytest
from holosoma.acp.panorama import PanoramaConfig, SurfacePanorama

Q=np.array([0.,0.,0.,1.])

def mapper(**kwargs):
    return SurfacePanorama(PanoramaConfig(height=8,width=16,resolution=.03,**kwargs))

def test_same_xy_retains_floor_and_overhang_and_wall():
    m=mapper()
    pts=np.array([[1.,0.,-.7],[1.,0.,.4],[1.,0.,0.]])
    m.integrate(pts,np.ones(3,bool),np.zeros(3),Q,0.)
    p=m.surface_points(0.)
    assert len(p)==3
    assert np.ptp(p[:,2])>1.0
    assert len(np.unique(m.cells[:,:2],axis=0))==1

def test_nearest_ray_hit_and_metric_backprojection_without_height_query():
    m=mapper();direction=m.directions[4,8]
    m.integrate(np.array([direction*1.2,direction*2.0]),[True,True],[0,0,0],Q,0.)
    out=m.render([0,0,0],Q,0.)
    assert out['valid'][4,8]
    assert 1.2-np.sqrt(3)*.03<=out['depth'][4,8]<=1.2+1e-6
    np.testing.assert_allclose(out['points_w'][4,8],direction*out['depth'][4,8],atol=1e-7)
    assert np.isnan(out['points_w'][~out['valid']]).all()
    assert (out['depth'][~out['valid']]==0).all()

def test_history_remains_behind_after_turning_and_underfoot():
    m=mapper();p=np.array([m.directions[4,8]*1.2,m.directions[7,8]*.8])
    m.integrate(p,[True,True],[0,0,0],Q,0.)
    m.integrate(np.empty((0,3)),np.empty(0,bool),[0,0,0],Q,.1)
    out=m.render([0,0,0],[0,0,1,0],1.)
    assert out['valid'][4,0] and out['valid'][7,0]
    np.testing.assert_allclose(out['age'][[4,7],0],1.)
    assert (out['confidence'][[4,7],0]>0).all()

def test_unseen_expired_and_out_of_range_are_unknown():
    m=mapper(max_age=1.)
    assert not m.render([0,0,0],Q,0.)['valid'].any()
    m.integrate([m.directions[4,8]*3.0],[True],[0,0,0],Q,0.)
    assert not m.render([0,0,0],Q,0.)['valid'].any()
    m.integrate([m.directions[4,8]*1.0],[True],[0,0,0],Q,.1)
    assert m.render([0,0,0],Q,.2)['valid'].any()
    assert not m.render([0,0,0],Q,1.2)['valid'].any()

def test_pose_alignment_duplicate_time_and_reset():
    m=mapper();d=m.directions[4,8]
    m.integrate([d],[True],[1,2,3],Q,1.)
    out=m.render([1,2,3],Q,1.)
    assert out['valid'][4,8]
    np.testing.assert_allclose(out['points_w'][4,8],[1,2,3]+d*out['depth'][4,8],atol=1e-6)
    m.integrate([d],[True],[1,2,3],Q,1.)
    assert m.count.max()==1
    with pytest.raises(ValueError,match='backwards'):m.integrate([d],[True],[1,2,3],Q,0.)
    with pytest.raises(ValueError):m.render([0,0,0],[0,0,0,0],1.)
    m.reset();assert not m.render([0,0,0],Q,0.)['valid'].any()

def test_capacity_and_retention_are_bounded():
    m=mapper(max_voxels=5)
    pts=np.stack([np.arange(20)*.1,np.zeros(20),np.zeros(20)],-1)
    m.integrate(pts,np.ones(20,bool),[0,0,0],Q,0.)
    assert len(m.keys)==5
    m.integrate(np.empty((0,3)),[],[20,0,0],Q,1.)
    assert len(m.keys)==0


def test_compiled_reader_matches_numpy_reference():
    pytest.importorskip('warp')
    ref=mapper();fast=SurfacePanorama(ref.config,backend='warp')
    rng=np.random.default_rng(7)
    points=np.concatenate([ref.directions.reshape(-1,3)*r for r in [.5,1.2,2.]])
    mask=rng.random(len(points))>.2
    for m in [ref,fast]:m.integrate(points,mask,[0,0,0],Q,0.)
    a=ref.render([.01,.02,.03],Q,.2);b=fast.render([.01,.02,.03],Q,.2)
    np.testing.assert_array_equal(a['valid'],b['valid'])
    for k in ['depth','age','confidence','points_w']:np.testing.assert_allclose(a[k],b[k],atol=1e-6)
