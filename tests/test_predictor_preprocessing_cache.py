import json
import numpy as np
import pytest
from generator.research.archived_experiments import preprocessing_cache as cache
from generator.research.archived_experiments.contact_event_sample import ContactEventSample, action_labels, observed_action_vocabulary


def test_settled_pose_is_a_real_zero_touchdown_action():
    common = dict(
        motion_id=0,
        source="motion",
        height=1.0,
        current_frame=0,
        target_frame=7,
        x=np.zeros(6),
        end_contact=np.asarray([1, 1, 0, 0, 0, 0], dtype=np.float32),
        surface_class=np.zeros(6),
        surface_uv=np.zeros((6, 2)),
        duration_s=0.14,
    )
    settled = ContactEventSample(touchdown=np.zeros(6), **common)
    landing = ContactEventSample(touchdown=np.asarray([1, 0, 0, 0, 0, 0]), **common)
    vocabulary = observed_action_vocabulary([settled, landing])
    assert vocabulary.shape == (2, 6)
    assert (~vocabulary.any(axis=1)).sum() == 1
    np.testing.assert_array_equal(action_labels([settled, landing], vocabulary), [0, 1])


def test_cache_roundtrip_and_corruption(tmp_path, monkeypatch):
    identity = {'dataset_fingerprint':'verified'}
    monkeypatch.setattr(cache, 'cache_identity', lambda *a:('test', identity))
    sample = ContactEventSample(0, 'motion', 1., 0, 7, np.zeros(6), np.ones(6),
                                np.ones(6), np.zeros(6), np.zeros((6,2)), .14)
    calls = []
    def build(*a):
        calls.append(1)
        return {'positions':np.arange(9).reshape(1,3,3)}, [sample], {'samples':1}
    a = cache.load_or_build_dataset('ignored', 64, build_dataset=build, cache_directory=tmp_path)
    b = cache.load_or_build_dataset('ignored', 64, build_dataset=build, cache_directory=tmp_path)
    assert len(calls) == 1
    assert not a[3]['cache_hit'] and b[3]['cache_hit']
    np.testing.assert_array_equal(a[0]['positions'], b[0]['positions'])
    np.testing.assert_array_equal(a[1][0].x, b[1][0].x)
    assert a[2] == b[2]
    with (tmp_path/'test.pt').open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        cache.load_or_build_dataset('ignored', 64, build_dataset=build, cache_directory=tmp_path)


def test_inputs_changed_during_build_are_not_published(tmp_path, monkeypatch):
    identities = iter([('one',{'dataset_fingerprint':'a'}),('two',{'dataset_fingerprint':'b'})])
    monkeypatch.setattr(cache,'cache_identity',lambda *a:next(identities))
    with pytest.raises(ValueError,match='changed during'):
        cache.load_or_build_dataset('ignored',64,build_dataset=lambda *a:({},[],{}),cache_directory=tmp_path)
    assert not list(tmp_path.glob('*.pt'))


def test_identity_includes_plan_catalog_manifest_and_frames(tmp_path, monkeypatch):
    import somaforge_core.contact_dataset as dataset
    monkeypatch.setattr(dataset,'contact_dataset_fingerprint',lambda p:'verified_motion_labels_terrain')
    catalog = tmp_path/'surface.jsonl'; catalog.write_text('original surface')
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps({'metadata':{'target_surface_catalog':str(catalog)}}))
    manifest = tmp_path/'manifest.json'
    from somaforge_core.newton_runtime_compat import NEWTON_RUNTIME_ID
    m = {'newton_runtime': {'runtime_id': NEWTON_RUNTIME_ID}, 'event_contract':{'v':1},'training_ready':True,'motion_files':[{'edit_plan_file':str(plan)}]}
    manifest.write_text(json.dumps(m))
    original = cache.cache_identity(manifest,64)[0]
    assert original != cache.cache_identity(manifest,32)[0]
    catalog.write_text('changed actual surface')
    assert original != cache.cache_identity(manifest,64)[0]
    catalog.write_text('original surface')
    assert original == cache.cache_identity(manifest,64)[0]
    plan.write_text(json.dumps({'metadata':{'target_surface_catalog':str(catalog)},'changed':True}))
    assert original != cache.cache_identity(manifest,64)[0]
    m['event_contract']={'v':2};manifest.write_text(json.dumps(m))
    assert original != cache.cache_identity(manifest,64)[0]
