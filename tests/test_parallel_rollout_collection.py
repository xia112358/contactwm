"""Batch identity, bounded lossless recording and preserved alignment/smoothing."""
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np

from somaforge_core.parallel_rollout_collection import batch_command
from somaforge_core.rollout_consensus import event_boundaries, fit_consensus


def test_one_balanced_batch_uses_joint_manifest_and_all_env_recording():
    plan = dict(output_dir="/project/output", cases=[dict(key="a",frames=40),dict(key="b",frames=70)],
                config=dict(collection=dict(seed=5100,initial_pose_noise=0)))
    command = batch_command(plan,"/policy.pt","/recording.npz",envs_per_motion=16)
    assert command[command.index("--num-envs")+1] == "32"
    assert command[command.index("--recording.config.env-id")+1] == "-1"
    assert command[command.index("--max-steps")+1] == "80"
    assert command.count("/project/output/references.json") == 2
    assert not any("/references/a.json" in s for s in command)
    assert "--command.setup-terms.motion-command.params.motion-config.fixed-motion-assignment-in-eval" in command
    assert "--headless" in command and "--visualizer" not in command


def test_chunk_storage_preserves_ragged_contacts_and_publishes_complete_manifest():
    from holosoma.agents.callbacks.recording import EvalRecordingCallback
    from holosoma.config_types.eval_callback import RecordingConfig
    from somaforge_core.trajectory_refinement import read_json,verify_files
    directory=Path(__file__).resolve().parents[1]/"tmp"/"parallel_collection_tests"/uuid.uuid4().hex
    directory.mkdir(parents=True)
    callback=EvalRecordingCallback(RecordingConfig(output_path=str(directory/"recording.npz"),chunk_frames=2),SimpleNamespace(device="cpu"))
    callback._step_count=2
    callback._buffers={key:[np.array([0]),np.array([1])] for key in
                      ("root_pos","motion_id","motion_time_step","solver_contact_count")}
    callback._buffers["solver_contact_dist"]=[np.array([.1]),np.array([.2,.3])]
    callback._save()
    receipt=read_json(directory/"recording.npz.json")
    assert receipt["complete"] and receipt["frames"]==2
    verify_files(receipt["chunks"])
    with np.load(receipt["chunks"][0]["path"]) as z:
        np.testing.assert_array_equal(z["solver_contact_dist"],[[.1,0],[.2,.3]])
    assert all(not values for values in callback._buffers.values())


def test_consensus_keeps_shared_event_alignment_and_removes_execution_jitter():
    n=24
    q=np.zeros((3,n,36),np.float32);q[:,:,2]=.8;q[:,:,3]=1
    q[:,:,0]=np.linspace(0,.1,n)
    q[:,:,7]=np.array([.01,-.01,.015])[:,None]*np.where(np.arange(n)%2,1.,-1.)
    pairs=[[[dict(part=0,shape="same_heel",surface=0,eligible=True,point=[0,0,0])] for _ in range(n)] for _ in range(3)]
    reference=q[0].copy(); reference[:,7]=0
    result,maps,report=fit_consensus(q,reference,[0,8,16,n-1],pairs,
        [dict(surface=0,normal_w=[0,0,1])],steps=12,band=2)
    assert np.all(np.diff(maps,axis=1)>=0)
    assert np.all(maps[:,0]==0) and np.all(maps[:,-1]==n-1)
    assert report["marker_smooth_weight"]==.05 and report["joint_smooth_weight"]==.2
    assert np.mean(np.diff(result[:,7:],n=2,axis=0)**2)<np.mean(np.diff(q[report['initialization_replica'],:,7:],n=2,axis=0)**2)
    np.testing.assert_allclose(np.linalg.norm(result[:,3:7],axis=1),1,atol=1e-6)


def test_event_clock_keeps_repeated_action_boundaries():
    case=dict(frames=24,contract=dict(actions=[dict(start=0,end=7),dict(start=7,end=15),dict(start=15,end=23)]))
    assert event_boundaries(case)==[0,7,15,23]


def test_candidate_pose_read_skips_unused_ragged_contact_arrays(monkeypatch):
    from somaforge_core.parallel_rollout_collection import load_candidates
    directory=Path(__file__).resolve().parents[1]/'tmp'/'parallel_collection_tests'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    for i in range(2):
        np.savez(directory/f'chunk_{i:05d}.npz', env_ids=[4, 8], recording_start=i*2,
            root_pos=np.full((2, 2, 3), i), unused_pairs=np.zeros((200, 20)))
    original = np.lib.npyio.NpzFile.__getitem__
    accessed = []
    def read(self, key):
        accessed.append(key)
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', read)
    result = load_candidates(directory, fields={'root_pos'})
    assert 'unused_pairs' not in accessed and set(result) == {'root_pos', 'env_ids'}
    np.testing.assert_array_equal(result['env_ids'], [4, 8])
    np.testing.assert_array_equal(result['root_pos'][:, 0, 0], [0, 0, 1, 1])


def test_inference_read_discards_only_runtime_resources(monkeypatch):
    import sys
    import types
    import torch
    import pytest
    from somaforge_core.checkpoint_inference import load_inference_fields
    name="isaaclab_newton._checkpoint_read_test"
    module=types.ModuleType(name)
    resource=type("Runtime",(),{"__module__":name})
    module.Runtime=resource
    monkeypatch.setitem(sys.modules,name,module)
    directory=Path(__file__).resolve().parents[1]/"tmp"/"parallel_collection_tests"/uuid.uuid4().hex
    directory.mkdir(parents=True)
    path=directory/"checkpoint.pt"
    weights={"layer":torch.arange(4.)}
    torch.save(dict(actor_model_state_dict=weights,experiment_config={"seed":7},runtime={"resource":resource()}),path)
    result=load_inference_fields(path)
    assert "runtime" not in result and result["experiment_config"]=={"seed":7}
    torch.testing.assert_close(result["actor_model_state_dict"]["layer"],weights["layer"],rtol=0,atol=0)
    forbidden=directory/"forbidden.pt"
    torch.save(dict(experiment_config={"resource":resource()}),forbidden)
    with pytest.raises(ValueError,match="required inference"):
        load_inference_fields(forbidden)


def test_batch_primary_faces_equal_existing_finite_face_attribution():
    from somaforge_core.newton_contacts import actual_contact_faces,actual_primary_face_indices
    binding=[dict(surface=8,normal_w=[0,0,1],triangles_w=[[[-1,-1,0],[1,-1,0],[1,1,0]],[[-1,-1,0],[1,1,0],[-1,1,0]]]),
             dict(surface=2,normal_w=[1,0,0],triangles_w=[[[0,-1,-1],[0,1,-1],[0,1,1]],[[0,-1,-1],[0,1,1],[0,-1,1]]]),
             dict(surface=5,normal_w=[0,1,0],triangles_w=[[[-1,0,-1],[1,0,-1],[1,0,1]],[[-1,0,-1],[1,0,1],[-1,0,1]]])]
    rng=np.random.default_rng(19)
    points=np.r_[rng.normal(size=(300,3)),[[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,1]]]
    normals=rng.normal(size=points.shape);normals[-5:]=[[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,1,1]]
    expected=[actual_contact_faces(p,n,binding)[0]['surface'] for p,n in zip(points,normals)]
    indices=actual_primary_face_indices(points,normals,binding,batch_size=19)
    assert [binding[i]['surface'] for i in indices]==expected
