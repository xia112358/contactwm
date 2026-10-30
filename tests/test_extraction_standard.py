"""Admission, provenance, scope and export must share one extraction contract."""
from copy import deepcopy
from pathlib import Path
import uuid

import pytest

from somaforge_core import extraction_standard as standard
from somaforge_core import trajectory_refinement as workflow


def accepted(key='a'):
    return dict(key=key, status='accepted', processing_standard=standard.STANDARD,
        aggregation_written=True, complete_rollouts=4, native_qualified=4,
        contributing_env_ids=[0, 1, 2, 3], optimization=dict(representative_fallback=False),
        task_acceptance=dict(schema=standard.EVENT_SCHEMA, passed=True),
        continuity=dict(schema=standard.CONTINUITY_SCHEMA, passed=True),
        joint_limits=dict(passed=True), clock_validation=dict(passed=True), standard_validation=dict(schema=standard.STANDARD,
            event_schema=standard.EVENT_SCHEMA, continuity_schema=standard.CONTINUITY_SCHEMA))


@pytest.mark.parametrize('options', [dict(stpes=100), dict(steps=True), dict(band=2.5),
    dict(steps=1), dict(continuity_weight=float('nan')), dict(learning_rate=0),
    dict(convergence_patience=2.5), dict(convergence_patience=0), dict(convergence_rtol=0)])
def test_fit_recipe_rejects_silent_mistakes(options):
    with pytest.raises(ValueError):
        standard.fit_options(options)


def test_legacy_pass_flags_and_single_source_fallback_do_not_mean_formal_pass():
    result = accepted()
    assert standard.classification(result) == 'accepted'
    old = deepcopy(result); old.pop('standard_validation')
    assert standard.classification(old) == 'legacy_unverified'
    result['optimization']['representative_fallback'] = True
    assert standard.classification(result) == 'source_baseline'
    result['optimization']['representative_fallback'] = False
    result['contributing_env_ids'] = [0, 0, 1]
    assert standard.classification(result) == 'source_baseline'


@pytest.mark.parametrize('gate', ['task_acceptance', 'continuity', 'joint_limits'])
def test_all_three_hard_gates_are_required(gate):
    result = accepted(); result[gate]['passed'] = False
    assert standard.classification(result) == 'needs_refinement'


def test_legacy_clock_does_not_acquire_a_continuous_certificate_from_pose_checks():
    result=accepted();result['clock_validation']['passed']=False
    assert standard.classification(result)=='legacy_unverified'


def test_summary_keeps_pending_scope_and_separates_variants():
    row = accepted(); row['strategies'] = [dict(extraction_status='accepted')]*2
    summary = standard.summarize([dict(key='a'), dict(key='b')], [row])
    assert (summary['total'], summary['processed'], summary['accepted'], summary['pending']) == (2, 1, 1, 1)
    assert summary['exported_variant_count'] == 2 and not summary['complete'] and not summary['all_passed']
    with pytest.raises(ValueError, match='Duplicate'):
        standard.summarize([dict(key='a')], [row, row])


@pytest.fixture
def formal_manifest():
    root = Path(__file__).resolve().parents[1]/'tmp'/'extraction_standard_tests'/uuid.uuid4().hex
    root.mkdir(parents=True)
    motion, contacts = root/'motion.npz', root/'contacts.npz'
    motion.write_bytes(b'motion'); contacts.write_bytes(b'contacts')
    row = accepted(); row.update(motion=str(motion), contacts=str(contacts), artifacts=[],
        standard_validation=standard.validation_stamp(motion, contacts))
    row['optimization']['schema'] = standard.ENGINE
    workflow.write_json(root/'aggregated/a/result.json', row)
    workflow.write_json(root/'reaggregation_plan.json', dict(schema='test plan'))
    cases = [dict(key='a', reference=dict(motion_file='original-reference.npz'))]
    plan = dict(config={}, manifest=dict(terrains=[]))
    standard.publish_manifest(plan, root, cases, [row])
    return root


def test_current_export_verifies_outputs_and_never_calls_authored_task_check(formal_manifest, monkeypatch):
    from somaforge_core.rollout_consensus import export_manifest
    import somaforge_core.rollout_dataset as dataset
    def forbidden(*args, **kwargs):
        pytest.fail('Authored reference must not rejudge execution consensus')
    monkeypatch.setattr(dataset, 'task_check', forbidden)
    plan = dict(cases=[dict(task=dict(kind='contact_events'))])
    manifest = export_manifest(plan, formal_manifest, [])
    assert len(manifest['motion_files']) == 1
    (formal_manifest/'motion.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed|mismatch',):
        export_manifest(plan, formal_manifest, [])


@pytest.mark.parametrize('tamper', ['motion_file', 'gate', 'contributors', 'scope'])
def test_manifest_cannot_retarget_validated_files_or_bypass_gates(formal_manifest, tamper):
    path = formal_manifest/'dataset_manifest.json'; m = workflow.read_json(path)
    row = m['motion_files'][0]
    if tamper == 'motion_file': row['motion_file'] = 'different-output.npz'
    elif tamper == 'gate': row['joint_limits']['passed'] = False
    elif tamper == 'contributors': row['source_env_ids'] = [0]
    else: m['expected_case_keys'].append('missing-case')
    workflow.write_json(path, m)
    with pytest.raises(ValueError):
        standard.verified_manifest(path)


def test_legacy_contact_and_mixed_collections_are_not_silently_exported(formal_manifest):
    from somaforge_core.rollout_consensus import export_manifest, aggregate_collection
    workflow.write_json(formal_manifest/'dataset_manifest.json', dict(motion_files=[]))
    with pytest.raises(ValueError, match='Legacy/incomplete'):
        export_manifest(dict(cases=[dict(task=dict(kind='contact_events'))]), formal_manifest, [])
    mixed = dict(cases=[dict(task=dict(kind=k)) for k in ('contact_events', 'completion_only')])
    with pytest.raises(ValueError, match='Split'):
        aggregate_collection(mixed, formal_manifest)


def test_incremental_publication_finishes_without_overwriting_an_intermediate_final(formal_manifest):
    root=formal_manifest/'incremental';root.mkdir()
    row=workflow.read_json(formal_manifest/'aggregated/a/result.json')
    workflow.write_json(root/'aggregated/a/result.json',row)
    workflow.write_json(root/'reaggregation_plan.json',dict(schema='incremental test'))
    cases=[dict(key='a',reference=dict(motion_file='ref.npz'))]
    plan=dict(config={},manifest=dict(terrains=[]))
    standard.publish_manifest(plan,root,cases,[])
    assert not (root/'dataset_manifest.json').exists()
    assert (root/'dataset_manifest.pending.json').exists()
    standard.publish_manifest(plan,root,cases,[row])
    assert len(standard.verified_manifest(root/'dataset_manifest.json')['motion_files'])==1
    standard.publish_manifest(plan,root,cases,[row])
    with pytest.raises(ValueError,match='immutable'):
        standard.publish_manifest(plan,root,cases,[])
