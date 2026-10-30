"""Geometry identity fixtures are not Newton contact ground truth."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contact_solver.interval_identity import SurfaceSolidOwners


def fixture():
    vertices=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*[1.,1.,.5]
    records=[dict(shape=0,component=i,link=-1,geometry=SimpleNamespace(points=lambda:vertices),
        rotation=np.eye(3),position=np.array([0.,0.,height])) for i,height in enumerate((-.5,.5))]
    records.append(dict(shape=54,component=0,link=0))
    faces=[]
    for surface,z in enumerate((0.,1.)):
        corners=[[-1.,-1.,z],[1.,-1.,z],[1.,1.,z],[-1.,1.,z]]
        faces.append(dict(surface=surface,normal_w=[0.,0.,1.],
            triangles_w=[[corners[0],corners[1],corners[2]],[corners[0],corners[2],corners[3]]]))
    metadata=dict(solid_geometry=dict(model_fingerprint='fixture',shapes=[dict(shape=0,kind='convex_mesh')]),
        surface_catalog=faces)
    scene=SimpleNamespace(fingerprint='fixture',link_names=('left_ankle_roll_link',),records=records)
    return metadata,scene


def test_equal_static_shape_ids_do_not_merge_distinct_components():
    metadata,scene=fixture();owners=SurfaceSolidOwners(metadata,scene)
    pair=dict(sample=torch.zeros(2,dtype=torch.long),shape0=torch.zeros(2,dtype=torch.long),
        component0=torch.tensor([0,1]),body_link0=torch.full((2,),-1),
        shape1=torch.full((2,),54),component1=torch.zeros(2,dtype=torch.long),
        body_link1=torch.zeros(2,dtype=torch.long),full_kind=torch.zeros(2,dtype=torch.long))
    surface=torch.zeros(1,6,dtype=torch.long);active=torch.zeros(1,6,dtype=torch.bool);active[0,0]=True
    assert owners.pair_sides(pair,surface,active).tolist()==[[False,True],[False,False]]
    pair['component0'][0]=999
    with pytest.raises(ValueError,match='identity mismatch'):owners.pair_sides(pair,surface,active)


def test_owner_fingerprint_mismatch_is_an_error():
    metadata,scene=fixture();metadata['solid_geometry']['model_fingerprint']='other'
    with pytest.raises(ValueError,match='identities differ'):SurfaceSolidOwners(metadata,scene)


def test_unknown_target_face_does_not_fall_back_to_geometry_threshold():
    metadata,scene=fixture();owners=SurfaceSolidOwners(metadata,scene)
    pair={key:torch.empty(0,dtype=torch.long) for key in (
        'sample','shape0','shape1','component0','component1','body_link0','body_link1','full_kind')}
    surface=torch.full((1,6),99);active=torch.zeros(1,6,dtype=torch.bool);active[0,0]=True
    with pytest.raises(ValueError,match='Unknown or nonprimary'):owners.pair_sides(pair,surface,active)


def test_provider_routes_arbitrary_geometry_queries_by_actual_component():
    from contact_solver.shape_target_interval import ShapeTargetIntervalProvider
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    metadata,scene=fixture()
    router=SimpleNamespace(scenes={0:scene},link_names=scene.link_names)
    def adapter(fk,regions,q,surface,scene,margin):
        zero=q.sum(-1)[:,None,None].expand(-1,6,4)*0
        return SurfaceIntervalBounds(zero,zero)
    provider=ShapeTargetIntervalProvider({0:metadata},router,geometry_query=adapter)
    pair=dict(sample=torch.zeros(2,dtype=torch.long),shape0=torch.zeros(2,dtype=torch.long),
        component0=torch.tensor([0,1]),body_link0=torch.full((2,),-1),
        shape1=torch.full((2,),54),component1=torch.zeros(2,dtype=torch.long),
        body_link1=torch.zeros(2,dtype=torch.long),full_kind=torch.zeros(2,dtype=torch.long))
    rows=SimpleNamespace(q=torch.zeros(1,36),solid=SimpleNamespace(pair=pair),
        observed=dict(link_names=scene.link_names,configured_margin=torch.tensor([.02])))
    model=SimpleNamespace(fk=None,region_geometry=None)
    active=torch.zeros(1,6,dtype=torch.bool);active[0,0]=True;surface=torch.zeros(1,6,dtype=torch.long)
    value=provider(model,rows,active,surface,{},torch.zeros(1,dtype=torch.long))
    assert value.solid_target_mask.tolist()==[[False,True],[False,False]]
    with pytest.raises(ValueError,match='Unknown initialized'):
        provider(model,rows,active,surface,{},torch.ones(1,dtype=torch.long))
    assert provider.contract()['newton_assets_margin_weights_architecture_changed'] is False
