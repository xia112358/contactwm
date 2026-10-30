from types import SimpleNamespace
import numpy as np
import pytest
import torch

from somaforge_core.native_rollout_consensus import ContactGuard
from somaforge_core.observed_event_consensus import build_consensus, audit_consensus


class TranslationFK:
    def link_poses(self, q, names):
        return q[:, None, :3].expand(-1, len(names), -1), torch.eye(3).expand(len(q), len(names), 3, 3)


def fixture():
    catalog = [dict(surface=0, normal_w=[0, 0, 1], plane_offset=0)]
    case = dict(fps=50, task=dict(kind='contact_events'), expected_surface_catalog=catalog,
        contract=dict(policy=dict(stable_frames=1, max_dropout_frames=0), retained_blocks=[[0, 4]],
                      actions=[dict(id='support', start=0, end=3, requirements=[dict(part=0, surface=0, kind='establish')])]))
    masks = np.zeros((3, 4, 6), bool); masks[:, :, 0] = True
    surfaces = np.full((3, 4, 6), -1); surfaces[:, :, 0] = 0
    case['observed_consensus'] = build_consensus(masks, surfaces, stable_frames=1, max_dropout_frames=0)
    report = audit_consensus(case['observed_consensus'], masks[0], surfaces[0])

    class Scene:
        metadata = dict(surface_catalog=catalog)
        allocated = True
        calls = 0

        def __call__(self, q):
            self.calls += 1
            n = len(q); active = q[:, 2] < .02
            mask = torch.zeros(n, 6, dtype=torch.bool); mask[:, 0] = active
            surface = torch.full((n, 6), -1); surface[active, 0] = 0
            return dict(link_names=('pelvis',), pairs=dict(sample=torch.arange(n), part=torch.zeros(n, dtype=torch.long),
                type=torch.ones(n, dtype=torch.long), dist=q[:, 2].detach().clone(), includemargin=torch.full((n,), .02),
                active=active, constraint_allocated=torch.full((n,), self.allocated),
                geometry_point0_w=torch.zeros(n, 3), geometry_point1_w=q[:, :3].detach().clone(),
                body_link0=torch.full((n,), -1), body_link1=torch.zeros(n, dtype=torch.long),
                normal_w=torch.tensor([[0., 0., 1.]]).expand(n, -1), task_pair=torch.ones(n, dtype=torch.bool),
                upward=torch.ones(n, dtype=torch.bool), primary_surface=torch.zeros(n, dtype=torch.long),
                full_kind=torch.zeros(n, dtype=torch.long), eligible=active & self.allocated),
                contact_part_mask=mask, contact_surface=surface, contact_position_w=torch.zeros(n, 6, 3))
    scene = Scene()
    guard = ContactGuard(scene, case, [report]*3, masks, surfaces)
    q = torch.zeros(4, 36); q[:, 2] = .01; q[:, 3] = 1
    guard.prepare(0, q, TranslationFK())
    return scene, guard, q


def test_frozen_witness_guide_tracks_pose_change_between_actual_queries():
    scene, guard, q = fixture()
    q[:, 2] = .021; q.requires_grad_()
    cost, report = guard(0, q, TranslationFK())
    assert report['failed_checks'] == 1
    cost.backward()
    assert (q.grad[:, 2] > 0).all()  # Descent approaches the actual plane.
    moved = q.detach().clone(); moved[:, 2] = .019
    lower, report = guard(1, moved, TranslationFK())
    assert lower < cost and report is None and scene.calls == 1
    _, report = guard(10, moved, TranslationFK())
    assert report['passed'] and scene.calls == 2


def test_unallocated_active_contact_is_an_error():
    scene, guard, q = fixture(); scene.allocated = False
    with pytest.raises(ValueError, match='unallocated'):
        guard(0, q, TranslationFK())


def test_native_fit_preserves_shared_clocks_and_reduces_multi_execution_noise():
    from somaforge_core.rollout_consensus import fit_consensus
    n = 24
    q = np.zeros((3, n, 36), np.float32); q[:, :, 2] = .8; q[:, :, 3] = 1
    q[:, :, 7] = np.asarray([.01, -.01, .015])[:, None]*np.where(np.arange(n)%2, 1., -1.)

    class VerifiedGuard:
        masks = np.ones((3, n, 6), bool)
        history = []
        def prepare(self, medoid, template, fk): self.medoid = medoid
        def __call__(self, step, pose, fk, final=False):
            return pose.sum()*0, dict(failed_checks=0)
    guard = VerifiedGuard()
    maps = np.tile(np.arange(n), (3, 1))
    result, actual_maps, report = fit_consensus(q, q[0], None, None, None,
        mappings=maps, contact_guard=guard, steps=12, representative_index=1)
    np.testing.assert_array_equal(actual_maps, maps)
    assert report['best_step'] > 0 and report['validated_failed_checks'] == 0
    assert guard.medoid == 1 and report['initialization_replica'] == 1
    assert 'marker acceleration' in report['objective_components']
    assert np.diff(result[:, 7:], n=2, axis=0).var() < np.diff(q[guard.medoid, :, 7:], n=2, axis=0).var()


def test_candidate_selection_prioritizes_contact_then_local_continuity():
    from somaforge_core.rollout_consensus import fit_consensus
    n = 12
    q = np.zeros((3, n, 36), np.float32); q[:, :, 2] = .8; q[:, :, 3] = 1

    class NativeGuard:
        masks = np.ones((3, n, 6), bool)
        history = []
        def prepare(self, *args): pass
        def __call__(self, step, pose, fk, final=False):
            # A later contact failure cannot win just by smoothing better.
            return pose.sum()*0, dict(failed_checks=int(step > 2))

    class LocalGuard:
        step = -1
        def objective(self, pose, fk):
            self.step += 1
            return pose.sum()*0, int(self.step < 2), float(self.step < 2)

    _, _, report = fit_consensus(q, q[0], None, None, None, steps=4,
        mappings=np.tile(np.arange(n), (3, 1)), contact_guard=NativeGuard(), continuity_guard=LocalGuard())
    assert report['best_step'] == 2
    assert report['validated_failed_checks'] == report['local_continuity_violations'] == 0
    assert report['rms_is_acceptance_condition'] is False


def test_exact_existing_aggregate_survives_inferior_optimizer_steps():
    from somaforge_core.rollout_consensus import fit_consensus
    q = np.zeros((3, 12, 36), np.float32); q[:, :, 2] = .8; q[:, :, 3] = 1
    class NativeGuard:
        masks = np.ones((3, 12, 6), bool)
        history = []
        def prepare(self, *args): pass
        def __call__(self, step, pose, fk, final=False): return pose.sum()*0, dict(failed_checks=0)
    class LocalGuard:
        calls = 0
        def objective(self, pose, fk):
            count = int(self.calls > 0); self.calls += 1
            return pose.sum()*0, count, float(count)
    result, _, report = fit_consensus(q, q[0], None, None, None, steps=3,
        mappings=np.tile(np.arange(12), (3, 1)), contact_guard=NativeGuard(),
        continuity_guard=LocalGuard(), initial_pose=q[0])
    np.testing.assert_array_equal(result, q[0])
    assert report['selected_existing_aggregate'] and report['best_step'] == -1


def test_measured_native_depth_does_not_push_valid_contact_to_arbitrary_margin_fraction():
    scene,guard,q=fixture()
    guard.source_depths=np.full((3,4,6),np.inf,np.float32)
    guard.source_depths[:,:,0]=.019
    q[:,2]=.019
    guard.prepare(0,q,TranslationFK())
    cost,report=guard(0,q,TranslationFK())
    assert report['passed'] and float(cost)<1e-8


def test_spline_correction_keeps_initial_final_pose_and_shared_clock():
    from somaforge_core.rollout_consensus import fit_consensus
    n=24;q=np.zeros((3,n,36),np.float32);q[:,:,2]=.8;q[:,:,3]=1
    q[:,:,7]=np.asarray([.03,-.02,.01])[:,None]*np.where(np.arange(n)%2,1.,-1.)
    initial=q.mean(0);initial[:,7]=0
    class Guard:
        masks=np.ones((3,n,6),bool)
        history=[]
        def prepare(self,*args):pass
        def __call__(self,step,pose,fk,final=False):return pose.sum()*0,dict(failed_checks=0)
    maps=np.tile(np.arange(n),(3,1))
    result,actual,report=fit_consensus(q,initial,None,None,None,mappings=maps,contact_guard=Guard(),
        initial_pose=initial,correction_time_scale=.12,fps=50,learning_rate=.0003,steps=20)
    np.testing.assert_array_equal(result[[0,1,-2,-1]],initial[[0,1,-2,-1]])
    np.testing.assert_array_equal(actual,maps)
    assert report['endpoints_preserved'] and report['correction_control_count']<n
    assert np.diff(result[:,7],n=2).std()<np.diff(q[0,:,7],n=2).std()/10


def test_local_spline_fit_preserves_poses_outside_correction_window():
    from somaforge_core.rollout_consensus import fit_consensus
    from somaforge_core.motion_signal import local_smooth_motion_signal
    n=40;q=np.zeros((3,n,36),np.float32);q[:,:,2]=.8;q[:,:,3]=1
    q[:,:,7]=.02*np.where(np.arange(n)%2,1.,-1.)
    seed,window=local_smooth_motion_signal(q[0],[20],fps=50,support_time_scale=.04)
    class Guard:
        masks=np.ones((3,n,6),bool)
        history=[]
        def prepare(self,*args):pass
        def __call__(self,step,pose,fk,final=False):return pose.sum()*0,dict(failed_checks=0)
    result,_,report=fit_consensus(q,seed,None,None,None,mappings=np.tile(np.arange(n),(3,1)),
        contact_guard=Guard(),initial_pose=seed,correction_time_scale=.12,
        correction_envelope=window,steps=12)
    np.testing.assert_array_equal(result[window==0],seed[window==0])
    assert report['local_correction']


def test_retained_candidates_have_matching_multi_execution_provenance_and_bounded_history():
    from pathlib import Path
    from somaforge_core.native_rollout_consensus import retained_consensus_candidates
    directory = Path('/results/aggregated/case')
    stages = [dict(stage=f'retained_{i}_whole_body_fit', failed_checks=0,
        local_violations=0, metrics=dict(marker_acceleration_rms_m_per_sample2=1/(i+1))) for i in range(60)]
    saved = dict(aggregation_written=True, contributing_env_ids=[4, 5, 6],
        motion=str(directory/'motion.npz'), optimization=dict(representative_fallback=False, stages=stages),
        artifacts=[dict(path=str(directory/'stages'/r['stage']/'observations.npz')) for r in stages])
    paths = retained_consensus_candidates(saved, [6, 5, 4])
    assert len(paths) == 2 and paths[0][1] == directory/'motion.npz'
    assert paths[1][1].parent.name == 'retained_59_whole_body_fit'
    assert not retained_consensus_candidates(saved, [7, 8, 9])
    saved['optimization']['representative_fallback'] = True
    assert not retained_consensus_candidates(saved, [4, 5, 6])


def test_verified_consensus_excludes_failed_contact_and_continuity_candidates():
    from somaforge_core.native_rollout_consensus import has_verified_consensus
    assert not has_verified_consensus([((1, False), 'contact_failure', None, None),
                                       ((0, True), 'jump', None, None)])
    assert has_verified_consensus([((0, False), 'denoised_multi_source', None, None)])


def test_fit_convergence_requires_sustained_successful_actual_audits():
    from somaforge_core.rollout_consensus import FitConvergence
    convergence = FitConvergence(50, .001)
    assert not convergence.update(0, 10., 0, 0)
    assert not convergence.update(25, 9., 0, 0)  # Significant progress resets patience.
    assert not convergence.update(50, 9., 0, 0)
    assert not convergence.update(75, 9., 1, 0)  # Failed contact resets the streak.
    assert not convergence.update(100, 9., 0, 0)
    assert not convergence.update(125, 9., 0, 1)  # A jump also resets it.
    assert not convergence.update(150, 9., 0, 0)
    assert not convergence.update(175, 9., 0, 0)
    assert convergence.update(200, 9., 0, 0)


@pytest.mark.parametrize('contact_failures,local_violations', [(0, 0), (1, 0), (0, 1)])
def test_fitter_stops_only_when_native_audits_pass_and_objective_converges(contact_failures, local_violations):
    from somaforge_core.rollout_consensus import fit_consensus
    from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
    fk = CanonicalG1ForwardKinematics()
    n = 12
    q = np.zeros((3, n, 36), np.float32); q[:, :, 2] = .8; q[:, :, 3] = 1
    q[:, :, 7:] = ((fk.joint_lower+fk.joint_upper)/2).numpy()
    class Guard:
        masks = np.ones((3, n, 6), bool)
        history = []
        def prepare(self, *args): pass
        def __call__(self, step, pose, fk, final=False):
            return pose.sum()*0, (dict(failed_checks=contact_failures) if step % 5 == 0 or final else None)
    class Continuity:
        def objective(self, pose, fk): return pose.sum()*0, local_violations, float(local_violations)
    maps = np.tile(np.arange(n), (3, 1))
    result, actual_maps, report = fit_consensus(q, q[0], None, None, None, mappings=maps,
        initial_pose=q[0], contact_guard=Guard(), continuity_guard=Continuity(),
        correction_time_scale=.12, steps=35, convergence_patience=10)
    np.testing.assert_array_equal(actual_maps, maps)
    np.testing.assert_array_equal(result[[0, 1, -2, -1]], q[0, [0, 1, -2, -1]])
    if contact_failures or local_violations:
        assert report['stop_reason'] == 'step_budget' and report['steps'] == 35
    else:
        assert report['stop_reason'] == 'verified_convergence' and report['steps'] < 35


def test_source_cache_follows_pinned_historical_artifacts_and_checks_pose_and_scene():
    import json
    import uuid
    from pathlib import Path
    from somaforge_core import trajectory_refinement as workflow
    from somaforge_core.native_rollout_consensus import source_observation_index, cached_source_observation
    root = Path(__file__).resolve().parents[1]/'tmp'/'consensus_parallel_tests'/uuid.uuid4().hex
    directory = root/'historical'/'source_observations'; directory.mkdir(parents=True)
    path = directory/'env_000042.npz'
    q = np.zeros((4, 36), np.float32); q[:, 3] = 1
    provenance = dict(model_fingerprint='actual-model', contact_semantics='current')
    np.savez(path, joint_pos=q, provenance_json=json.dumps(provenance))
    pinned = workflow.fingerprint(path)
    saved = dict(artifacts=[pinned])
    index = source_observation_index([(root/'copied'/'result.json', saved)])
    assert cached_source_observation(index[path.name], q, provenance) == path.resolve()
    assert cached_source_observation(index[path.name], q+1, provenance) is None
    assert cached_source_observation(index[path.name], q, dict(provenance, model_fingerprint='different')) is None
    np.savez(path, joint_pos=q+1, provenance_json=json.dumps(provenance))
    with pytest.raises(ValueError, match='changed'):
        cached_source_observation(index[path.name], q, provenance)


def test_parallel_cases_overlap_and_keep_results_with_their_own_case():
    import threading
    from somaforge_core.native_rollout_consensus import parallel_case_results
    rendezvous = threading.Barrier(4, timeout=5)
    threads = set(); lock = threading.Lock()
    def worker(case):
        with lock: threads.add(threading.get_ident())
        rendezvous.wait()
        return dict(key=case, squared=case**2)
    results = list(parallel_case_results(range(8), worker, 4))
    assert len(threads) == 4
    assert sorted((r['key'], r['squared']) for r in results) == [(i, i*i) for i in range(8)]


def test_parallel_cases_propagate_failure_without_publishing_a_pass():
    from somaforge_core.native_rollout_consensus import parallel_case_results
    def fail(case): raise ValueError('native contact failure')
    with pytest.raises(ValueError, match='native contact failure'):
        list(parallel_case_results(range(20), fail, 4))


def test_shared_solver_serializes_device_and_cpu_audits_and_keeps_result_storage():
    import threading
    import time
    from somaforge_core.native_rollout_consensus import SharedNativeQueries, parallel_case_results
    owner = threading.get_ident()
    class Backend:
        model = SimpleNamespace(device='cpu')
        active = 0
        max_active = 0
        buffer = torch.zeros(1)
        def __call__(self, q, scene):
            assert threading.get_ident() == owner
            self.active += 1; self.max_active = max(self.max_active, self.active)
            self.buffer[:] = q
            time.sleep(.002)  # Exercise overlapping callers, not a performance test.
            result = self.buffer.clone()
            self.active -= 1
            return result
    backend = Backend()
    class Scene:
        native = backend
        metadata = {}
        def query(self, q): return backend(q, None)
    shared = SharedNativeQueries(Scene())
    start = threading.Barrier(4, timeout=5)
    def request(i):
        start.wait()
        if i % 3 == 0:
            return shared.run_native(lambda: backend(i, None))
        return shared.query(i) if i % 2 else shared.cpu_query(i, None)
    results = list(parallel_case_results(range(8), request, 4, poll=shared.service_one, abort=shared.close))
    assert backend.max_active == 1
    assert sorted(float(r) for r in results) == list(range(8))


def test_native_query_failure_unblocks_other_workers_during_shutdown():
    from somaforge_core.native_rollout_consensus import SharedNativeQueries, parallel_case_results
    class Backend:
        model = SimpleNamespace(device='cpu')
    class Scene:
        native = Backend()
        metadata = {}
        def query(self, q): raise ValueError('native solver failure')
    shared = SharedNativeQueries(Scene())
    with pytest.raises(ValueError, match='native solver failure'):
        list(parallel_case_results(range(8), shared.query, 4, poll=shared.service_one, abort=shared.close))
    with pytest.raises(RuntimeError, match='stopped'):
        shared.query(1)
