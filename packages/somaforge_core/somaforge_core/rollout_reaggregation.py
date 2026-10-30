"""Versioned event-consensus postprocessing of an immutable rollout collection."""
from pathlib import Path
from . import trajectory_refinement as workflow
from . import extraction_standard as standard


def _motion_artifact(result, name):
    """Resolve a primary motion's evidence, including copied reuse receipts."""
    directory = Path(result['motion']).parent.resolve()
    matches = {Path(a['path']) for a in result['artifacts']
               if Path(a['path']).name == name and Path(a['path']).parent.resolve() == directory}
    if len(matches) != 1:
        raise ValueError(f'Missing/ambiguous primary motion artifact: {name}')
    return matches.pop()


def _reassess_one(case, old_path, directory):
    """Audit actual output evidence; generation version alone never admits it."""
    import numpy as np
    from .consensus_continuity import acceptance_status
    from .event_continuity import EventContinuityGuard
    from .g1_kinematics import CanonicalG1ForwardKinematics
    from .newton_contact_data import load_contact_labels
    from .observed_event_consensus import audit_consensus
    old = workflow.read_json(old_path)
    workflow.verify_files(old['artifacts'])
    result = dict(old, previous_status=old['status'],
                  artifacts=old['artifacts']+[workflow.fingerprint(old_path)])
    if (not old.get('aggregation_written') or old.get('task_acceptance', {}).get('schema') != standard.EVENT_SCHEMA
            or not old.get('source_acceptance')):
        result.update(status='legacy_unverified', processing_standard=standard.STANDARD)
        workflow.write_json(directory/'result.json', result, immutable=True)
        return result
    fk = CanonicalG1ForwardKinematics()
    alignment_path = _motion_artifact(old, 'alignment.npz')
    clock_validation = dict(passed=False, reason='legacy transport lacks continuous clock certification')
    with np.load(alignment_path, allow_pickle=False) as z:
        envs = z['source_env_ids'].tolist()
        clock_fields = {'source_frame_maps', 'clock_control_knots', 'clock_displacement_controls'}
        if clock_fields <= set(z.files):
            from .continuous_motion import certify_clocks
            clock_validation = certify_clocks(z['source_frame_maps'], z['clock_control_knots'],
                z['clock_displacement_controls'], old['alignment']['rate_bounds'])
            clock_validation['source_information_preserved'] = all(
                row.get('own_event_preservation', {}).get('passed', False) for row in old['alignment_checks'])
            clock_validation['passed'] &= clock_validation['source_information_preserved']
    source_paths = {Path(a['path']).name: Path(a['path']) for a in old['artifacts']
                    if Path(a['path']).parent.name == 'aligned_observations'
                    and Path(a['path']).parent.parent.resolve() == Path(old['motion']).parent.resolve()}
    aligned_reports = {row['env_id']:row['acceptance'] for row in old['alignment_checks']}
    sources = []
    for env in envs:
        with np.load(source_paths[f'env_{env:06d}.npz'], allow_pickle=False) as z:
            sources.append(z['joint_pos'].copy())
    guard = EventContinuityGuard(sources,
        [aligned_reports[e] for e in envs], case['fps'], fk, reference_clock='transported')
    observed = load_contact_labels(old['motion'], old['contacts'])
    task = audit_consensus(workflow.read_json(_motion_artifact(old, 'event_consensus.json')),
                          observed['contact_part_mask'], observed['contact_surface'])
    with np.load(old['motion'], allow_pickle=False) as z:
        pose = z['joint_pos'].copy()
    continuity = guard.check(pose, fk, task)
    joint_count = int(((pose[:,7:] < fk.joint_lower.numpy()-1e-6)
                      | (pose[:,7:] > fk.joint_upper.numpy()+1e-6)).sum())
    workflow.write_json(directory/'continuity.json', continuity, immutable=True)
    result.update(status=acceptance_status(task, continuity) if not joint_count else 'joint_limits_unresolved',
        processing_standard=standard.STANDARD, event_acceptance=task, task_acceptance=task,
        continuity=dict(path=str(directory/'continuity.json'), schema=continuity['schema'],
                        passed=continuity['passed'], violation_count=continuity['violation_count']),
        joint_limits=dict(passed=joint_count==0, violation_count=joint_count),
        standard_validation=standard.validation_stamp(old['motion'], old['contacts']),
        clock_validation=clock_validation,
        smoothness=dict({k:v for k,v in old['smoothness'].items() if k not in ('budgets','budget_origin')},
                        rms_is_acceptance_condition=False))
    result['artifacts'].append(workflow.fingerprint(directory/'continuity.json'))
    if not result['clock_validation']['passed']:
        result['status']='legacy_clock_unverified'
    workflow.write_json(directory/'result.json', result, immutable=True)
    return result


def reassess_completed(cases, previous, output, *, admit_existing=True, result_paths=None):
    """Versioned evidence reuse, including externally indexed result receipts."""
    previous, output = Path(previous), Path(output)
    summaries = []
    for case in cases:
        old_path = Path((result_paths or {}).get(case['key'], previous/'aggregated'/case['key']/'result.json'))
        if not old_path.exists():
            continue
        directory = output/'reassessment'/case['key']
        if (directory/'result.json').exists():
            result = workflow.read_json(directory/'result.json')
            if result.get('standard_validation'):
                standard.verify_validation(result)
            else:
                workflow.verify_files(result['artifacts'])
        else:
            old = workflow.read_json(old_path)
            if old.get('strategies'):
                variants = []
                for index, row in enumerate(old['strategies']):
                    target = directory/'strategies'/f'group_{index:03d}'
                    child = _reassess_one(case, Path(row['result_path']), target)
                    variants.append(dict(child, result_path=str(target/'result.json')))
                primary = min(variants, key=lambda r: (standard.classification(r) != 'accepted',
                                                       r.get('task_acceptance',{}).get('failed_checks',float('inf'))))
                result = dict(primary, complete_rollouts=old['complete_rollouts'], strategies=[
                    dict(status=r['status'],extraction_status=standard.classification(r),
                         result_path=r['result_path'],motion=r.get('motion'),contributing_env_ids=r.get('contributing_env_ids',[]))
                    for r in variants])
                result['artifacts'] = [a for r in variants for a in r['artifacts']]+[
                    workflow.fingerprint(r['result_path']) for r in variants]
                workflow.write_json(directory/'result.json', result, immutable=True)
            else:
                result = _reassess_one(case, old_path, directory)
        summaries.append(dict(key=case['key'], previous_status=result['previous_status'], status=result['status'],
                              local_violations=result.get('continuity', {}).get('violation_count')))
        if admit_existing and standard.classification(result) == 'accepted':
            workflow.write_json(output/'aggregated'/case['key']/'result.json', result, immutable=True)
        print(f"Reassessed {case['key']}: {result['previous_status']} -> {result['status']}; "
              f"local violations={result.get('continuity', {}).get('violation_count')}", flush=True)
    summary = dict(processed=len(summaries), accepted=sum(r['status'] == 'accepted' for r in summaries), motions=summaries)
    workflow.write_json(output/'reassessment_summary.json', summary)
    return summary


def _verify_context(old, receipt):
    for key in ('source_collection', 'source_plan', 'training_plan', 'recording'):
        if old[key] != receipt[key]:
            raise ValueError(f'Reuse inputs differ: {key}')
    stable = {'observed_event_consensus.py', 'contact_face_selection.py', 'newton_contact_tensors.py'}
    evidence = [p for p in old['implementation'] if Path(p['path']).name in stable]
    if {Path(p['path']).name for p in evidence} != stable:
        raise ValueError('Reuse lacks current native event/contact implementation evidence')
    workflow.verify_files(evidence)


def _import_receipts(path, cases, receipt):
    """Follow immutable upstream receipts; never trust a manifest's pass flag."""
    manifest = workflow.read_json(path)
    seen = set()
    def verify_origin(value):
        if value.get('reaggregation_plan'):
            pinned = value['reaggregation_plan']
            workflow.verify_files([pinned])
            _verify_context(workflow.read_json(pinned['path']), receipt)
        elif value.get('source_manifests'):
            for pinned in value['source_manifests']:
                workflow.verify_files([pinned])
                if pinned['path'] not in seen:
                    seen.add(pinned['path']); verify_origin(workflow.read_json(pinned['path']))
        else:
            raise ValueError('Reuse manifest lacks upstream collection provenance')
    verify_origin(manifest)
    indexed = {c['key']: c for c in cases}
    paths = {}
    for row in manifest['motion_files']:
        key = row.get('source_case_key')
        if key not in indexed:
            continue
        if row['reference_motion_file'] != indexed[key]['reference']['motion_file']:
            raise ValueError('Reused manifest reference differs from selected case')
        pinned = row.get('validation_result')
        if pinned is None:
            raise ValueError('Reuse manifest requires indexed validation_result receipts')
        workflow.verify_files([pinned])
        old = workflow.read_json(pinned['path'])
        matches = old.get('motion') == row['motion_file'] and old.get('contacts') == row['contacts']
        if not matches:
            for variant in old.get('strategies', []):
                child = workflow.read_json(variant['result_path'])
                matches |= child.get('motion') == row['motion_file'] and child.get('contacts') == row['contacts']
        if old['key'] != key or not matches:
            raise ValueError('Reuse receipt belongs to a different trajectory')
        if key in paths and paths[key] != pinned['path']:
            raise ValueError('Ambiguous result receipts for one case; import its parent strategy receipt')
        paths[key] = pinned['path']
    return paths


def reaggregate(plan, source, output, *, keys=None, device='cuda:0', query_worlds=standard.QUERY_WORLDS, case_workers=standard.CASE_WORKERS,
                reuse_results=None, reuse_manifest=None, reassess_only=False, rebuild=False):
    import sys
    import uuid
    source, output = Path(source).resolve(), Path(output).resolve()
    values = reuse_results if isinstance(reuse_results, (list, tuple)) else ([] if reuse_results is None else [reuse_results])
    previous_roots = list(dict.fromkeys(Path(p).resolve() for p in values))
    previous = previous_roots[0] if previous_roots else None
    if source == output or output in previous_roots:
        raise ValueError('Reaggregation requires a new output directory')
    if type(query_worlds) is not int or query_worlds < 1:
        raise ValueError('query_worlds must be a positive integer')
    if type(case_workers) is not int or case_workers < 1:
        raise ValueError('case_workers must be a positive integer')
    if reassess_only and previous is None and reuse_manifest is None:
        raise ValueError('Reassessment needs --reuse-results or --reuse-manifest')
    source_receipt = workflow.read_json(source/'collection_plan.json')
    workflow.verify_files(source_receipt['inputs'])
    selected = [c for c in plan['cases'] if keys is None or c['key'] in keys]
    if not selected or (keys is not None and set(keys) != {c['key'] for c in selected}):
        raise ValueError('Unknown/empty case selection')
    recipe = standard.definition(plan['config'].get('aggregation'))
    receipt = dict(schema='native_event_consensus_reprocessing_v12', extraction_standard=recipe,
        source_collection=str(source), source_plan=workflow.fingerprint(source/'collection_plan.json'),
        cases=[c['key'] for c in selected], training_plan=workflow.fingerprint(Path(plan['output_dir'])/'plan.json'),
        recording=workflow.fingerprint(source/'recording.npz.json'),
        implementation=[workflow.fingerprint(p) for p in standard.implementation_paths()], query_worlds=query_worlds,
        case_workers=case_workers,
        aggregation=recipe['fit'], training=False, new_policy_rollout=False,
        rebuild=rebuild,
        reuse_results=[str(p) for p in previous_roots],
        reuse_manifest=None if reuse_manifest is None else workflow.fingerprint(reuse_manifest))
    for root in previous_roots:
        old_receipt = workflow.read_json(root/'reaggregation_plan.json')
        _verify_context(old_receipt, receipt)
    receipt['reused_plans'] = [workflow.fingerprint(root/'reaggregation_plan.json') for root in previous_roots]
    # Already certified completed outputs can be rechecked cheaply. Legacy
    # clocks need new registration anyway; do not run a futile CPU audit first.
    imported = {}
    for case in selected:
        for root in previous_roots:
            path = root/'aggregated'/case['key']/'result.json'
            if path.exists() and (reassess_only or workflow.read_json(path).get('clock_validation', {}).get('passed')):
                imported[case['key']] = str(path)
                break
    if reuse_manifest:
        imported.update(_import_receipts(reuse_manifest, selected, receipt))
    receipt['reused_results'] = [workflow.fingerprint(path) for path in imported.values()]
    workflow.write_json(output/'reaggregation_plan.json', receipt, immutable=True)
    if imported:
        reassess_completed([c for c in selected if c['key'] in imported],
                           previous or output, output, result_paths=imported, admit_existing=not rebuild)
    def completed_results():
        return [workflow.read_json(output/'aggregated'/c['key']/'result.json') for c in selected
                if (output/'aggregated'/c['key']/'result.json').exists()]
    standard.publish_manifest(plan, output, selected, completed_results())
    workflow.write_json(output/'progress.json', standard.summarize(selected, completed_results()))
    if reassess_only:
        return workflow.read_json(output/'summary.json')
    policy = workflow.read_json(source/'policy/inference_policy.json')['inference_checkpoint']
    workflow.verify_files([policy])
    by_terrain = {}
    for case in selected:
        by_terrain.setdefault(case['terrain']['terrain_id'], []).append(case)
    for terrain, cases in by_terrain.items():
        pending = [c for c in cases if not (output/'aggregated'/c['key']/'result.json').exists()]
        if not pending:
            continue
        directory = output/'labeling'/f'terrain_{terrain:04d}'
        manifest = directory/'manifest.json'
        workflow.write_json(manifest, workflow.read_json(Path(plan['output_dir'])/'references'/f"{cases[0]['key']}.json"), immutable=True)
        jobs = directory/'jobs.json'
        workflow.write_json(jobs, dict(source_collection=str(source), output_dir=str(output), cases=cases,
            total_cases=len(selected), aggregation=receipt['aggregation'], rebuild=rebuild,
            case_workers=case_workers, reuse_results=[str(p) for p in previous_roots]), immutable=True)
        inspection, binding = directory/'model.json', directory/'binding.json'
        command = [sys.executable, 'scripts/serve_newton_contact_queries.py', '--checkpoint', policy['path'],
            '--motion-manifest', str(manifest), '--inspection-output', str(inspection), '--binding', str(binding),
            '--query-worlds', str(query_worlds), '--capture-contact-sources', '--headless', '--device', device,
            '--consensus-jobs', str(jobs)]
        if not binding.exists(): command.append('--create-native-binding')
        workflow.verify_files(receipt['implementation'])
        workflow.run_command(command, directory/'run'/uuid.uuid4().hex)
        standard.publish_manifest(plan, output, selected, completed_results())
    results = completed_results()
    for result in results:
        workflow.verify_files(result['artifacts'])
    summary = standard.publish_manifest(plan, output, selected, results)
    workflow.write_json(output/'progress.json', summary)
    return summary
