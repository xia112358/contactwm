import numpy as np
import pytest
from somaforge_core.newton_scene_router import NewtonSceneRouter


def fake_query(q, scene, *, endpoint):
    assert scene is None
    fp = {'one':'11'*32,'one-replica':'11'*32,'two':'22'*32}[endpoint]
    return dict(schema='s',parts=['p'],sampling='fixed',provenance=dict(model_fingerprint=fp),
        configured_terrain_includemargins=[0.02],
        configured_margin_source='realized_mjwarp_contact_params_margin_sum_v2',
        surface_catalog=[endpoint],candidate_pairs=[[{'body_name':endpoint}] for _ in q],
        pairs=[[] for _ in q],active=q[:,0,None],unallocated=np.zeros((len(q),1)),
        position_w=np.zeros((len(q),1,3)),surface=np.zeros((len(q),1)))


def test_mixed_routes_restore_order_and_keep_scene_catalogs_separate():
    router=NewtonSceneRouter({'11'*32:'one','22'*32:'two'},query=fake_query)
    q=np.zeros((3,36));q[:,0]=[3,5,7]
    r=router(q,['22'*32,'11'*32,'22'*32])
    assert r['active'][:,0].tolist()==[3,5,7]
    assert r['surface_catalog_by_sample']==[['two'],['one'],['two']]
    assert r['configured_terrain_includemargins_by_sample']==[[0.02],[0.02],[0.02]]


def test_wrong_fingerprint_and_unknown_scene_fail_closed():
    q=np.zeros((1,36))
    with pytest.raises(ValueError,match='different actual'):
        NewtonSceneRouter({'11'*32:'two'},query=fake_query)(q,['11'*32])
    with pytest.raises(ValueError,match='No authoritative'):
        NewtonSceneRouter({'11'*32:'one'},query=fake_query)(q,['22'*32])


def test_same_scene_is_sharded_across_replicas_and_order_is_restored():
    calls = []
    def recorded_query(q, scene, *, endpoint):
        calls.append((endpoint, q[:, 0].tolist()))
        return fake_query(q, scene, endpoint=endpoint)
    router = NewtonSceneRouter(
        {'11'*32: ['one', 'one-replica']}, query=recorded_query, workers=2,
    )
    q = np.zeros((5, 36)); q[:, 0] = [9, 1, 7, 3, 5]
    result = router(q, ['11'*32] * len(q))
    assert result['active'][:, 0].tolist() == [9, 1, 7, 3, 5]
    assert {endpoint for endpoint, _ in calls} == {'one', 'one-replica'}
    assert sorted(value for _, values in calls for value in values) == [1, 3, 5, 7, 9]
