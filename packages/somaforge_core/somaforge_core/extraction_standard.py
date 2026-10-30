"""One extraction contract for collection, reprocessing, export and progress.

Generation recipe and output verification are distinct: unchanged historical
poses may qualify after their native evidence is audited under this contract.
"""
import math
from pathlib import Path

STANDARD = 'somaforge_consensus_extraction_v1'
EVENT_SCHEMA = 'native_population_information_consensus_v2'
CONTINUITY_SCHEMA = 'transported_source_information_continuity_v3'
ENGINE = 'parallel_continuous_population_consensus_v12'
QUERY_WORLDS = 32
CASE_WORKERS = 4
ENVS_PER_MOTION = 16
CHUNK_FRAMES = 16
DEFAULT_FIT = dict(steps=600, band=10, marker_smooth_weight=.05,
                   joint_smooth_weight=.2, correction_time_scale=.12,
                   learning_rate=.0003, continuity_weight=10.,
                   convergence_patience=50, convergence_rtol=.001)


def fit_options(options=None):
    options = {} if options is None else dict(options)
    unknown = set(options)-set(DEFAULT_FIT)
    if unknown:
        raise ValueError(f'Unknown aggregation options: {sorted(unknown)}')
    result = dict(DEFAULT_FIT, **options)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid aggregation.{key}')
        minimum = 0 if key in ('marker_smooth_weight', 'joint_smooth_weight') else 1e-15
        if value < minimum:
            raise ValueError(f'Invalid aggregation.{key}')
    if type(result['steps']) is not int or result['steps'] < 2:
        raise ValueError('aggregation.steps must be an integer >= 2')
    if type(result['band']) is not int:
        raise ValueError('aggregation.band must be a positive integer')
    if type(result['convergence_patience']) is not int:
        raise ValueError('aggregation.convergence_patience must be a positive integer')
    return result


def definition(options=None):
    return dict(schema=STANDARD, generation_engine=ENGINE, fit=fit_options(options),
        source_selection='complete uninterrupted executions; minimum three per strategy',
        event_acceptance_schema=EVENT_SCHEMA, continuity_schema=CONTINUITY_SCHEMA,
        original_reference_used_for_acceptance=False,
        contact_truth='actual Newton/MJWarp activation, allocation and shared primary upward face selection',
        formal_output='verified multi-execution consensus only; source baselines are diagnostic',
        rms_is_acceptance_condition=False, dynamic_feasibility_verified=False)


def implementation_paths():
    from .robot_assets import somaforge_root
    directory = Path(__file__).parent
    names = ('extraction_standard.py', 'rollout_reaggregation.py', 'trajectory_refinement.py',
        'consensus_event_targets.py', 'motion_signal.py', 'continuous_motion.py', 'population_event_clock.py',
        'observed_event_consensus.py', 'native_rollout_consensus.py', 'staged_consensus.py',
        'consensus_event_alignment.py', 'consensus_continuity.py', 'event_continuity.py',
        'rollout_consensus.py', 'parallel_rollout_collection.py', 'contact_aggregation.py', 'contact_order_acceptance.py',
        'contact_face_selection.py', 'newton_contact_tensors.py', 'newton_contact_data.py',
        'newton_contacts.py', 'contact_labels.py', 'g1_kinematics.py', 'robot_assets.py')
    return [directory/name for name in names]+[somaforge_root()/'scripts/serve_newton_contact_queries.py']


def check_implementation():
    from . import trajectory_refinement as workflow
    directory = Path(__file__).parent
    names = ('extraction_standard.py', 'continuous_motion.py', 'observed_event_consensus.py', 'event_continuity.py',
             'consensus_continuity.py', 'consensus_event_alignment.py', 'newton_contact_data.py',
             'contact_face_selection.py', 'newton_contacts.py', 'newton_contact_tensors.py',
             'g1_kinematics.py', 'robot_assets.py')
    return [workflow.fingerprint(directory/name) for name in names]


def validation_stamp(motion, contacts):
    from . import trajectory_refinement as workflow
    return dict(schema=STANDARD, event_schema=EVENT_SCHEMA, continuity_schema=CONTINUITY_SCHEMA,
                motion=workflow.fingerprint(motion), contacts=workflow.fingerprint(contacts),
                implementation=check_implementation(),
                scope='observed event retention, source-relative continuity and nominal joint limits',
                task_success_verified=False)


def classification(result):
    if result is None:
        return 'pending'
    stamp = result.get('standard_validation', {})
    if stamp.get('schema') != STANDARD:
        if result.get('processing_standard') == STANDARD and not result.get('aggregation_written'):
            return 'insufficient_data' if result['status'].startswith('insufficient') else 'needs_refinement'
        return 'legacy_unverified'
    if (stamp.get('event_schema') != EVENT_SCHEMA or stamp.get('continuity_schema') != CONTINUITY_SCHEMA
            or result.get('event_acceptance', result.get('task_acceptance', {})).get('schema') != EVENT_SCHEMA
            or result.get('continuity', {}).get('schema') != CONTINUITY_SCHEMA):
        return 'legacy_unverified'
    if not result.get('clock_validation',{}).get('passed'):
        return 'legacy_unverified'
    if (not result.get('aggregation_written') or not result.get('task_acceptance', {}).get('passed')
            or not result.get('continuity', {}).get('passed') or not result.get('joint_limits', {}).get('passed')):
        return 'needs_refinement'
    envs = result.get('contributing_env_ids', [])
    if result.get('optimization', {}).get('representative_fallback', True) or len(set(envs)) < 3:
        return 'source_baseline'
    return 'accepted'


def verify_validation(result):
    from . import trajectory_refinement as workflow
    stamp = result.get('standard_validation', {})
    if stamp.get('schema') != STANDARD or stamp.get('implementation') != check_implementation():
        raise ValueError('Extraction checks changed or were not run; revalidate in a new output')
    if (Path(stamp['motion']['path']).resolve() != Path(result['motion']).resolve()
            or Path(stamp['contacts']['path']).resolve() != Path(result['contacts']).resolve()):
        raise ValueError('Extraction validation belongs to different output files')
    workflow.verify_files([stamp['motion'], stamp['contacts']]+result['artifacts'])


def summarize(cases, results):
    indexed = {r['key']: r for r in results}
    if len(indexed) != len(results) or set(indexed)-{c['key'] for c in cases}:
        raise ValueError('Duplicate or unexpected extraction case results')
    motions = []
    counts = dict.fromkeys(('accepted', 'pending', 'legacy_unverified', 'needs_refinement',
                           'insufficient_data', 'source_baseline'), 0)
    variants = 0
    for case in cases:
        row = indexed.get(case['key']); state = classification(row); counts[state] += 1
        motion = dict(key=case['key'], status=state, result_status=None if row is None else row['status'])
        if row is not None:
            motion.update({k: row[k] for k in ('complete_rollouts', 'native_qualified', 'aggregation_written')})
            if state == 'accepted':
                variants += (sum(s.get('extraction_status') == 'accepted' for s in row['strategies'])
                             if row.get('strategies') else 1)
        motions.append(motion)
    return dict(extraction_standard=STANDARD, total=len(cases), processed=len(results),
        accepted=counts['accepted'], accepted_consensus=counts['accepted'],
        pending=counts['pending'], legacy_unverified=counts['legacy_unverified'],
        needs_refinement=counts['needs_refinement'], insufficient_data=counts['insufficient_data'],
        source_baseline=counts['source_baseline'], exported_variant_count=variants,
        complete=len(results)==len(cases), all_passed=counts['accepted']==len(cases),
        motions=motions, training_ready=False)


def publish_manifest(plan, output, cases, results):
    from . import trajectory_refinement as workflow
    output = Path(output)
    entries = []
    by_key = {r['key']: r for r in results}
    for case in cases:
        parent = by_key.get(case['key'])
        if parent is None:
            continue
        variants = ([workflow.read_json(row['result_path']) for row in parent['strategies']]
                    if parent.get('strategies') else [parent])
        for result in variants:
            if classification(result) != 'accepted':
                continue
            verify_validation(result)
            entries.append(dict(case['reference'], motion_id=len(entries), source_case_key=case['key'],
                motion_file=result['motion'], motion_sha256=workflow.sha256_file(result['motion']),
                contacts=result['contacts'], contacts_sha256=workflow.sha256_file(result['contacts']),
                contact_catalog_file=result['contacts'], source_env_ids=result['contributing_env_ids'],
                aggregation_kind='multi_execution_consensus',
                generation_recipe=result['optimization']['schema'], extraction_standard=STANDARD,
                event_acceptance=result['task_acceptance'], task_acceptance=result['task_acceptance'],
                continuity=result['continuity'], joint_limits=result['joint_limits'],
                clock_validation=result['clock_validation'],
                standard_validation=result['standard_validation'],
                validation_result=workflow.fingerprint(output/'aggregated'/case['key']/'result.json'),
                reference_motion_file=case['reference']['motion_file'], training_ready=False,
                execution_kind='event_aligned_denoised_smoothed_aggregate; fixed_q_Newton_labels'))
    summary = summarize(cases, results)
    if summary['exported_variant_count'] != len(entries):
        raise ValueError('Extraction summary and exported variants disagree')
    final_path = output/'dataset_manifest.json'
    if final_path.exists() and workflow.read_json(final_path).get('complete'):
        if workflow.read_json(output/'summary.json') != summary:
            raise ValueError('Completed extraction is immutable; use a new output directory')
    workflow.write_json(output/'summary.json', summary)
    manifest = dict(workflow.manifest_header('newton_direct_fk'), motion_files=entries,
        terrains=plan['manifest']['terrains'], extraction_standard=definition(plan['config'].get('aggregation')),
        expected_case_keys=[c['key'] for c in cases], complete=summary['complete'], all_passed=summary['all_passed'],
        validation_summary=workflow.fingerprint(output/'summary.json'), training_ready=False,
        reaggregation_plan=workflow.fingerprint(output/'reaggregation_plan.json'))
    # A mutable progress snapshot must never occupy the immutable final name.
    destination = final_path if summary['complete'] else output/'dataset_manifest.pending.json'
    workflow.write_json(destination, manifest, immutable=summary['complete'])
    return summary


def verified_manifest(path):
    from . import trajectory_refinement as workflow
    manifest = workflow.read_json(path)
    if manifest.get('extraction_standard', {}).get('schema') != STANDARD or not manifest.get('complete'):
        raise ValueError('Legacy/incomplete extraction manifest; reprocess under the current extraction standard')
    summary = workflow.read_json(manifest['validation_summary']['path'])
    workflow.verify_files([manifest['validation_summary'], manifest['reaggregation_plan']])
    if summary.get('extraction_standard') != STANDARD or summary['exported_variant_count'] != len(manifest['motion_files']):
        raise ValueError('Extraction manifest and verification summary disagree')
    expected = manifest['expected_case_keys']
    if (len(set(expected)) != len(expected) or summary['total'] != len(expected)
            or not summary['complete'] or summary['processed'] != len(expected)
            or manifest['all_passed'] != summary['all_passed']):
        raise ValueError('Extraction scope and verification summary disagree')
    accepted_keys = {r['key'] for r in summary['motions'] if r['status'] == 'accepted'}
    if {r['source_case_key'] for r in manifest['motion_files']} != accepted_keys:
        raise ValueError('Extraction manifest omits or adds accepted cases')
    for row in manifest['motion_files']:
        if row.get('aggregation_kind') != 'multi_execution_consensus' or len(set(row['source_env_ids'])) < 3:
            raise ValueError('Formal extraction contains a source baseline')
        pseudo = dict(row, motion=row['motion_file'], aggregation_written=True, artifacts=[],
            contributing_env_ids=row['source_env_ids'], optimization=dict(representative_fallback=False))
        if classification(pseudo) != 'accepted':
            raise ValueError('Formal extraction has not passed current event/continuity/joint checks')
        verify_validation(pseudo)
        if (row['motion_sha256'] != row['standard_validation']['motion']['sha256']
                or row['contacts_sha256'] != row['standard_validation']['contacts']['sha256']):
            raise ValueError('Manifest hashes differ from validated outputs')
        workflow.verify_files([row['validation_result']])
        parent = workflow.read_json(row['validation_result']['path'])
        variants = [parent]
        if parent.get('strategies'):
            receipts = [a for a in parent['artifacts'] if any(
                Path(a['path']).resolve() == Path(v['result_path']).resolve() for v in parent['strategies'])]
            workflow.verify_files(receipts)
            variants = [workflow.read_json(v['result_path']) for v in parent['strategies']]
        matches = [v for v in variants if v.get('motion') == row['motion_file'] and v.get('contacts') == row['contacts']]
        if (parent['key'] != row['source_case_key'] or len(matches) != 1
                or classification(matches[0]) != 'accepted'
                or any(row[k] != matches[0][k] for k in ('standard_validation', 'task_acceptance', 'continuity', 'joint_limits','clock_validation'))
                or row['source_env_ids'] != matches[0]['contributing_env_ids']):
            raise ValueError('Manifest does not match its verified result receipt')
    return manifest
