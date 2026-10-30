"""Exact finite-face and solver selection semantics after CPU batching."""
import numpy as np
import pytest

from somaforge_core.native_surface_index import NativeSurfaceIndex
from somaforge_core.newton_contacts import actual_primary_face_indices, constraint_decision, triangle_distance_squared
from somaforge_core.newton_support import SupportContactSelector, select_support_contacts


def faces():
    return [
        dict(surface=10, normal_w=[0.,0.,1.], triangles_w=[
            [[-3.,-3.,0.],[3.,-3.,0.],[3.,3.,0.]],
            [[-3.,-3.,0.],[3.,3.,0.],[-3.,3.,0.]]]),
        dict(surface=2, normal_w=[1.,0.,0.], triangles_w=[
            [[0.,-3.,-3.],[0.,3.,-3.],[0.,3.,3.]],
            [[0.,-3.,-3.],[0.,3.,3.],[0.,-3.,3.]]]),
    ]


@pytest.mark.parametrize('batch_size', [1, 7, 64, 4096])
def test_face_batches_match_original_exhaustive_finite_geometry_and_ties(batch_size):
    rng = np.random.default_rng(17)
    geometry = faces()
    # Off-axis normals and shared-edge ties exercise scalar-score roundoff routing.
    for i in range(6):
        angle = i*.2
        normal = np.asarray([np.sin(angle), 0., np.cos(angle)])
        u = np.asarray([np.cos(angle), 0., -np.sin(angle)])
        v = np.asarray([0., 1., 0.])
        geometry.append(dict(surface=40-i, normal_w=normal.tolist(),
                             triangles_w=[np.stack((-u-v, u-v, v)).tolist()]))
    points = np.vstack((rng.normal(size=(160,3)), [[0.,0.,0.],[0.,.2,0.],[0.,0.,1e-12]],
                        [[1e6,0.,0.],[-1e6,0.,1.]]))
    normals = rng.normal(size=points.shape)
    normals /= np.linalg.norm(normals, axis=1)[:,None]
    normals[160] = np.asarray([1.,0.,1.])/np.sqrt(2)
    expected = actual_primary_face_indices(points, normals, geometry)
    actual = NativeSurfaceIndex(geometry).query(points, normals, batch_size=batch_size)
    assert np.array_equal(actual, expected)


def test_paired_point_triangle_distance_keeps_original_arithmetic():
    rng = np.random.default_rng(5)
    triangles = rng.normal(size=(100,3,3)); points = rng.normal(size=(100,3))
    expected = np.asarray([triangle_distance_squared(point, triangle[None])[0]
                           for point, triangle in zip(points, triangles)])
    assert np.array_equal(triangle_distance_squared(points, triangles), expected)
    index = NativeSurfaceIndex(faces())
    assert index.query(np.empty((0,3)), np.empty((0,3))).dtype == np.int64
    with pytest.raises(ValueError, match='batch size'):
        index.query(points, points, batch_size=0)


def snapshot_fixture():
    geometry = faces()
    binding = dict(body_env={0:0,1:0,2:1}, shape_surface={100:geometry}, surface_catalog=geometry)
    labels = ['env0/left_ankle_roll_link','env0/torso_link','env1/right_sphere_hand_link']
    n = 10
    snapshot = dict(count=n, dist=np.zeros(n), includemargin=np.full(n,.01),
        type=np.ones(n, np.int64), efc_address=np.arange(n)[:,None], constraint_rows=np.ones(n,np.int64),
        body0=np.asarray([0,0,-1,1,0,0,0,0,0,0]), body1=np.asarray([-1,-1,2,-1,1,2,-1,-1,-1,-1]),
        worldid=np.asarray([0,0,1,0,0,0,1,0,0,0]),
        position_w=np.tile([.2,.2,0.],(n,1)), frame_w=np.tile(np.eye(3),(n,1,1)),
        force_on_body1_w=np.zeros((n,3)))
    snapshot['position_w'][1] = [0.,.2,.5]  # Side remains primary even with an upward normal.
    snapshot['frame_w'][:,0] = [0.,0.,-1.]; snapshot['frame_w'][2,0] = [0.,0.,1.]
    snapshot['shape0'] = np.where(snapshot['body0'] >= 0, 5, 100)
    snapshot['shape1'] = np.where(snapshot['body1'] >= 0, 5, 100)
    snapshot['shape1'][3] = 999  # Untracked torso must be excluded before terrain attribution.
    snapshot['type'][7] = 0
    snapshot['dist'][8] = snapshot['includemargin'][8]  # Strict native activation boundary.
    snapshot['type'][9] = 0; snapshot['force_on_body1_w'][9] = 1e6  # Force cannot activate contact.
    refresh_decision(snapshot)
    return snapshot, binding, labels


def refresh_decision(snapshot):
    snapshot['active'], snapshot['constraint_allocated'] = constraint_decision(
        snapshot['dist'], snapshot['includemargin'], snapshot['type'], snapshot['efc_address'],
        snapshot['count'], snapshot['constraint_rows'])


def test_compiled_selection_excludes_sides_self_cross_robot_wrong_world_and_inactive():
    snapshot, binding, labels = snapshot_fixture()
    expected = select_support_contacts(snapshot, binding, labels)
    actual = SupportContactSelector(binding, labels)(snapshot)
    for key in expected:
        assert np.array_equal(actual[key], expected[key]), key
    assert np.flatnonzero(actual['eligible']).tolist() == [0,2]
    assert actual['excluded_rows'] == [1]
    assert actual['groups'][2] == 9


def test_compiled_selection_rejects_unallocated_and_inconsistent_solver_evidence():
    snapshot, binding, labels = snapshot_fixture()
    selector = SupportContactSelector(binding, labels)
    snapshot['efc_address'][0,0] = -1; refresh_decision(snapshot)
    with pytest.raises(ValueError, match='Unallocated'):
        selector(snapshot)
    snapshot['constraint_allocated'][0] = True
    with pytest.raises(ValueError, match='Inconsistent'):
        selector(snapshot)


def test_empty_solver_snapshot_preserves_output_shapes():
    snapshot, binding, labels = snapshot_fixture()
    empty = {key:value[:0] if isinstance(value,np.ndarray) else 0 for key,value in snapshot.items()}
    expected = select_support_contacts(empty,binding,labels)
    actual = SupportContactSelector(binding,labels)(empty)
    for key in expected:
        assert np.array_equal(actual[key],expected[key]),key


def device_selector(binding, labels):
    torch = pytest.importorskip('torch')
    pytest.importorskip('warp')
    if not torch.cuda.is_available():
        pytest.skip('CUDA contact equivalence test')
    from somaforge_core.live_contact_device import DeviceSupportContactSelector
    return DeviceSupportContactSelector(binding,labels,'cuda:0')


def device_snapshot(snapshot):
    import torch
    return {key:torch.as_tensor(value,device='cuda:0') if isinstance(value,np.ndarray) else value
            for key,value in snapshot.items()}


def assert_device_selection(selector, snapshot, expected):
    import torch
    actual = selector(device_snapshot(snapshot)); selector.validate(actual)
    for key in ('eligible','robot_side','groups','surfaces'):
        assert np.array_equal(actual[key].cpu().numpy(),expected[key]),key
    assert np.array_equal(torch.where(actual['excluded'])[0].cpu().numpy(),expected['excluded_rows'])


def test_gpu_force_point_selection_keeps_native_activation_and_primary_faces():
    snapshot, binding, labels = snapshot_fixture()
    selector = device_selector(binding,labels)
    assert_device_selection(selector,snapshot,select_support_contacts(snapshot,binding,labels))
    empty = {key:value[:0] if isinstance(value,np.ndarray) else 0 for key,value in snapshot.items()}
    assert_device_selection(selector,empty,select_support_contacts(empty,binding,labels))


def test_gpu_rejects_unallocated_unknown_and_nonfinite_contact_evidence():
    snapshot, binding, labels = snapshot_fixture(); selector = device_selector(binding,labels)
    snapshot['efc_address'][0,0] = -1; refresh_decision(snapshot)
    with pytest.raises(ValueError,match='unallocated'):
        selector.validate(selector(device_snapshot(snapshot)))
    snapshot['efc_address'][0,0] = 0; refresh_decision(snapshot)
    snapshot['shape1'][0] = 10000
    with pytest.raises(ValueError,match='external'):
        selector.validate(selector(device_snapshot(snapshot)))
    snapshot['shape1'][0] = 100; snapshot['position_w'][0,0] = np.nan
    with pytest.raises(ValueError,match='geometry'):
        selector.validate(selector(device_snapshot(snapshot)))


def test_gpu_double_precision_edges_scores_and_far_force_points_match_cpu():
    rng = np.random.default_rng(25)
    geometry = faces()
    for i in range(6):
        angle = .2*(i+1); u=np.asarray([np.cos(angle),0.,-np.sin(angle)])
        v=np.asarray([0.,1.,0.]); normal=np.asarray([np.sin(angle),0.,np.cos(angle)])
        geometry.append(dict(surface=40-i,normal_w=normal.tolist(),
            triangles_w=[np.stack((-u-v,u-v,v)).tolist()]))
    points=np.vstack((rng.normal(size=(160,3)),[[0.,0.,0.],[0.,.2,0.],[0.,0.,1e-12]],
                      [[1e6,0.,0.],[-1e6,0.,1.]]))
    normals=rng.normal(size=points.shape);normals/=np.linalg.norm(normals,axis=1)[:,None]
    normals[160]=np.asarray([1.,0.,1.])/np.sqrt(2)
    n=len(points)
    binding=dict(body_env={0:0},shape_surface={100:geometry},surface_catalog=geometry)
    labels=['robot/left_ankle_roll_link']
    snapshot=dict(count=n,dist=np.zeros(n),includemargin=np.full(n,.01),type=np.ones(n,np.int64),
        efc_address=np.arange(n)[:,None],constraint_rows=np.ones(n,np.int64),
        body0=np.zeros(n,np.int64),body1=np.full(n,-1,np.int64),worldid=np.zeros(n,np.int64),
        shape0=np.zeros(n,np.int64),shape1=np.full(n,100,np.int64),position_w=points,
        frame_w=np.zeros((n,3,3)))
    snapshot['frame_w'][:,0]=-normals;refresh_decision(snapshot)
    assert_device_selection(device_selector(binding,labels),snapshot,
                            select_support_contacts(snapshot,binding,labels))


def test_gpu_capacity_padding_does_not_create_contacts_or_reject_unused_nan_fields():
    snapshot,binding,labels=snapshot_fixture();selector=device_selector(binding,labels)
    expected=select_support_contacts(snapshot,binding,labels);count=snapshot['count']
    padded={key:np.concatenate((value,value[:5])) if isinstance(value,np.ndarray) else value
            for key,value in snapshot.items()}
    padded['dist'][count:]=np.nan;padded['position_w'][count:]=np.nan
    # Unused solver capacity is never active, regardless of its stale contents.
    padded['active'][count:]=False;padded['constraint_allocated'][count:]=False
    actual=selector(device_snapshot(padded));selector.validate(actual)
    for key in ('eligible','robot_side','groups','surfaces'):
        assert np.array_equal(actual[key][:count].cpu().numpy(),expected[key]),key
    assert not actual['active'][count:].any()


def test_gpu_reader_respects_the_callers_nondefault_cuda_stream():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA stream equivalence test')
    snapshot,binding,labels=snapshot_fixture()
    with torch.cuda.stream(torch.cuda.Stream()):
        selector=device_selector(binding,labels)
        result=selector(device_snapshot(snapshot));selector.validate(result)
        assert result['eligible'].is_cuda
        expected=select_support_contacts(snapshot,binding,labels)
        assert np.array_equal(result['eligible'].cpu().numpy(),expected['eligible'])
