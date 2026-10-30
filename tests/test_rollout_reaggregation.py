from pathlib import Path
import pytest
import uuid

from somaforge_core.rollout_reaggregation import _motion_artifact
from somaforge_core.rollout_reaggregation import _import_receipts, _verify_context
from somaforge_core import trajectory_refinement as workflow


def test_reuse_must_match_collection_and_native_event_contact_implementations():
    current = dict(source_collection='/collection', source_plan={}, training_plan={}, recording={})
    old = dict(current, implementation=[])
    with pytest.raises(ValueError, match='native event/contact'):
        _verify_context(old, current)
    old['source_collection'] = '/different'
    with pytest.raises(ValueError, match='source_collection'):
        _verify_context(old, current)


def test_combined_manifest_resolves_receipts_and_rejects_wrong_reference(monkeypatch):
    root = Path(__file__).resolve().parents[1]/'tmp'/'extraction_standard_tests'/uuid.uuid4().hex
    root.mkdir(parents=True)
    workflow.write_json(root/'origin.json', {})
    workflow.write_json(root/'source_manifest.json', dict(reaggregation_plan=workflow.fingerprint(root/'origin.json')))
    workflow.write_json(root/'result.json', dict(key='a', motion='new.npz', contacts='new_contacts.npz'))
    row = dict(source_case_key='a', reference_motion_file='ref.npz', motion_file='new.npz',
               contacts='new_contacts.npz', validation_result=workflow.fingerprint(root/'result.json'))
    path = root/'combined.json'
    manifest = dict(source_manifests=[workflow.fingerprint(root/'source_manifest.json')], motion_files=[row])
    workflow.write_json(path, manifest)
    monkeypatch.setattr('somaforge_core.rollout_reaggregation._verify_context', lambda old, current: None)
    cases = [dict(key='a', reference=dict(motion_file='ref.npz'))]
    assert _import_receipts(path, cases, {}) == {'a': str((root/'result.json').resolve())}
    row['reference_motion_file'] = 'different.npz'; workflow.write_json(path, manifest)
    with pytest.raises(ValueError, match='reference differs'):
        _import_receipts(path, cases, {})
    row['reference_motion_file'] = 'ref.npz'; row['contacts'] = 'different_contacts.npz'
    workflow.write_json(path, manifest)
    with pytest.raises(ValueError, match='different trajectory'):
        _import_receipts(path, cases, {})


def test_copied_receipt_resolves_original_evidence_beside_its_motion():
    original=Path('/old/pilot/aggregated/motion_000014')
    result=dict(motion=str(original/'motion.npz'),artifacts=[
        dict(path=str(original/'event_consensus.json')),
        dict(path=str(original/'alignment.npz'))])
    assert _motion_artifact(result,'event_consensus.json')==original/'event_consensus.json'
    assert _motion_artifact(result,'alignment.npz')==original/'alignment.npz'


def test_primary_strategy_does_not_use_another_strategys_first_artifact():
    first=Path('/run/strategies/group_000');primary=Path('/run/strategies/group_001')
    result=dict(motion=str(primary/'motion.npz'),artifacts=[
        dict(path=str(first/'event_consensus.json')),
        dict(path=str(primary/'event_consensus.json'))])
    assert _motion_artifact(result,'event_consensus.json')==primary/'event_consensus.json'
    result['artifacts'].pop()
    with pytest.raises(ValueError,match='primary motion artifact'):
        _motion_artifact(result,'event_consensus.json')
