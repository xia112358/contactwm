"""Feed successful native captures into the original full trajectory synthesizer.

No policy or world-model training. Input executions are independently verified;
all pose fusion, event registration and native audits belong to somaforge_core.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(capture_root, manifest, output, population_report):
    import numpy as np
    import torch
    from supermimic.contact_source_pool import successful_source_pool
    from supermimic.task_manifest import load_task_manifest
    from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER
    output.mkdir(parents=True, exist_ok=False)
    pool = successful_source_pool(capture_root)
    spec = load_task_manifest(manifest)
    reports = json.loads(population_report.read_text())
    verified = {entry['source_sha256']: entry for entry in pool['selected']}
    groups = {report['motion']: [verified[entry['source_sha256']] for entry in report['sources']]
              for report in reports}
    reports = {report['motion']: report for report in reports}
    jobs = []
    for motion, entries in sorted(groups.items()):
        if len(entries) < 3:
            raise ValueError(f"{entries[0]['name']}: fewer than three successful executions")
        frames = {entry['frames'] for entry in entries}
        if len(frames) != 1:
            raise ValueError('Source frame clocks disagree')
        frames = frames.pop()
        key = f"motion-{motion:03d}"
        source = output / key / 'sources'
        candidate = source / 'candidates' / key
        candidate.mkdir(parents=True)
        states = []
        for entry in entries:
            saved = torch.load(entry['source'], map_location='cpu', weights_only=False)
            s = saved['data']['s1']
            states.append(torch.cat((s[:, :7], s[:, 13:42]), -1).numpy())
        q = np.stack(states, axis=1)
        np.savez_compressed(candidate / 'chunk_000000.npz',
            root_pos=q[:, :, :3], root_quat_xyzw=q[:, :, [4, 5, 6, 3]], dof_pos=q[:, :, 7:],
            motion_time_step=np.broadcast_to(np.arange(frames)[:, None], q.shape[:2]),
            motion_start_step=np.zeros(q.shape[:2], dtype=np.int64),
            terminated=np.zeros(q.shape[:2], dtype=bool), timeout=np.zeros(q.shape[:2], dtype=bool),
            env_ids=np.arange(len(entries)), recording_start=np.asarray(0))
        metadata = dict(env_done_steps=[None]*len(entries), env_done_is_timeout=[False]*len(entries),
                        dof_names=list(G1_29DOF_JOINT_ORDER), robot_asset=spec['robot_asset'])
        (source / 'recording.npz.json').write_text(json.dumps(dict(metadata=metadata), indent=2))
        (source / 'source-executions.json').write_text(json.dumps(entries, indent=2))
        report = reports[motion]
        consensus = report['consensus']
        contract = json.loads((Path(entries[0]['source']).parent/'substeps/contract.json').read_text())
        world = contract['world_task_ids'].index(motion)
        offset = min(face['surface'] for face in contract['terrain_faces'][world])
        for event in consensus['events'] + consensus['minority_episodes']:
            event['surface'] -= offset
            for observed in event['observations'].values():
                observed['surface'] -= offset
        case = dict(key=key, frames=frames, fps=50., name=entries[0]['name'], original_motion=motion,
                    support_consensus=consensus,
                    support_evidence=dict(path=str(population_report.resolve()),
                        sha256=hashlib.sha256(population_report.read_bytes()).hexdigest(),
                        source_hashes=[entry['source_sha256'] for entry in entries],
                        surface_offset=offset, contract=contract['support_contract']))
        single = dict(spec, tasks=[spec['tasks'][motion]])
        single_path = output / key / 'task.json'
        single_path.write_text(json.dumps(single, indent=2))
        destination = output / key / 'synthesis'
        destination.mkdir()
        (destination / 'reaggregation_plan.json').write_text(json.dumps(dict(cases=[key]), indent=2))
        job = dict(source_collection=str(source.resolve()), output_dir=str(destination.resolve()),
                   cases=[case], aggregation={}, case_workers=1, rebuild=False)
        job_path = output / key / 'jobs.json'
        job_path.write_text(json.dumps(job, indent=2))
        jobs.append(dict(motion=motion, name=case['name'], sources=len(entries), frames=frames,
                         manifest=str(single_path.resolve()), jobs=str(job_path.resolve())))
    receipt = dict(capture_root=str(capture_root.resolve()), manifest=str(manifest.resolve()),
                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), jobs=jobs,
                   engine='somaforge_core.native_rollout_consensus.run_jobs', policy_updated=False,
                   stages=['native_source_query', 'population_event_alignment', 'continuous_pose_fusion',
                           'whole_body_fit', 'native_candidate_audit', 'canonical_motion_export'],
                   state_clock='post-integration s1; original actions/forces are not transferred to synthesized poses')
    (output / 'plan.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def synthesize(manifest, jobs, args, launch_simulation):
    import numpy as np
    import torch
    import newton
    from supermimic.lowstep import install
    spec = install(str(manifest), contact_model='force6', collision_profile='surface_v2', contact_response='newton_contacts_v2')
    from supermimic.env import TrackingSim, make_sim
    from supermimic.motion import MotionLib
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.robot import default_joint_pos_list
    from supermimic.contact_native import NativeContacts
    from somaforge_core.native_model_binding import inspect_model
    from somaforge_core.newton_contact_query import DedicatedNewtonSceneQuery
    from somaforge_core.native_rollout_consensus import run_jobs
    from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER
    torch.set_num_threads(4)
    device = 'cuda:0'
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(spec['motion_dir'], kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    cfg = make_sim(device, args)
    with launch_simulation(cfg, args):
        from somaforge_core.extraction_standard import QUERY_WORLDS
        env = TrackingSim(cfg, QUERY_WORLDS, motion, mdp, eval_mode=True)
        reader = NativeContacts(env, direct=True, primary_faces=True)
        manager = env.sim.backend_manager
        model = manager.get_model()
        info = inspect_model(model)
        info['robot_asset'] = spec['_execution_robot_asset']
        from somaforge_core.newton_runtime_compat import runtime_provenance
        info['newton_runtime'] = runtime_provenance()
        labels = [label.rsplit('/', 1)[-1] for label in model.joint_label]
        indices, roots = [], []
        for world in range(env.N):
            members = [i for i, w in enumerate(info['joint_world']) if w == world]
            indices.append([info['joint_q_start'][next(i for i in members if labels[i] == name)]
                            for name in G1_29DOF_JOINT_ORDER])
            free = [i for i in members if info['joint_type'][i] == int(newton.JointType.FREE)]
            if len(free) != 1:
                raise ValueError('Expected one canonical free robot root per world')
            roots.append(info['joint_q_start'][free[0]])
        # Cloned boxes retain actual shape IDs; identical local surface IDs let
        # independent pose queries share one event clock and one terrain catalog.
        for shape, faces in reader.binding['shape_surface'].items():
            world = info['shape_world'][shape]
            for face in faces:
                face['surface'] -= reader.surface_offsets[world]
        catalog = {face['surface']: face for faces in reader.binding['shape_surface'].values() for face in faces}
        binding = dict(reader.binding, surface_catalog=list(catalog.values()), scene=None,
                       model_fingerprint=info['model_fingerprint'])
        binding_path = jobs.parent / 'native-binding.json'
        binding_path.write_text(json.dumps(binding, indent=2))
        query = DedicatedNewtonSceneQuery(model, manager._solver, manager._collision_pipeline,
            scene=None, body_env=reader.binding['body_env'], shape_surface=reader.binding['shape_surface'],
            joint_q_indices=indices, root_q_start=roots,
            model_provenance=info, capture_contact_sources=True)
        # Verify GPU and CPU reduction on the same actual fixed-q solve before
        # allowing any generated motion to be exported.
        from somaforge_core.contact_face_selection import select_contact_pairs
        job = json.loads(jobs.read_text())
        chunk = Path(job['source_collection'])/'candidates'/job['cases'][0]['key']/'chunk_000000.npz'
        with np.load(chunk) as source:
            q = np.concatenate((source['root_pos'][:8, 0],
                source['root_quat_xyzw'][:8, 0][:, [3, 0, 1, 2]], source['dof_pos'][:8, 0]), -1)
        q = torch.as_tensor(q, device=device, dtype=torch.float32)
        native = query.query_device(q)
        cpu = query.audit_current(q)
        primary = select_contact_pairs(cpu['pairs'], binding['surface_catalog'])
        if not np.array_equal(primary['contact_part_mask'], native['contact_part_mask'].cpu().numpy()):
            raise AssertionError('Same-pass CPU/GPU task contacts disagree')
        if not np.array_equal(primary['contact_surface'], native['contact_surface'].cpu().numpy()):
            raise AssertionError('Same-pass CPU/GPU primary surfaces disagree')
        (jobs.parent/'query-parity.json').write_text(json.dumps(dict(samples=len(q), passed=True,
            active=int(primary['contact_part_mask'].sum()), physics_modified=False), indent=2))
        run_jobs(query, jobs, binding_path)


def export_references(population, output):
    import numpy as np
    from supermimic.task_manifest import load_task_manifest
    plan = json.loads((population/'plan.json').read_text())
    spec = load_task_manifest(plan['manifest'])
    output.mkdir(parents=True, exist_ok=False)
    csv_dir = output/'motions'
    csv_dir.mkdir()
    receipts = []
    for row in plan['jobs']:
        job = json.loads(Path(row['jobs']).read_text())
        result_path = Path(job['output_dir'])/'aggregated'/job['cases'][0]['key']/'result.json'
        result = json.loads(result_path.read_text())
        if not result['aggregation_written']:
            raise ValueError(f"No synthesized trajectory for {row['name']}: {result['status']}")
        path = Path(result['motion'])
        with np.load(path) as data:
            q = data['joint_pos']
        task = spec['tasks'][row['motion']]
        csv = csv_dir/Path(task['csv']).name
        np.savetxt(csv, q[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]], delimiter=',', fmt='%.9g')
        task.update(csv=str(csv.resolve()), csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                    source_motion=str(path), source_motion_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        receipts.append(dict(motion=row['motion'], result=str(result_path),
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
            status=result['status'], sources=len(result['contributing_env_ids'])))
    spec['motion_dir'] = str(csv_dir.resolve())
    spec['trajectory_synthesis'] = dict(engine=plan['engine'], source_manifest=plan['manifest'],
        results=receipts, contacts_requeried=True, dynamic_replay_verified=False)
    (output/'manifest.json').write_text(json.dumps(spec, indent=2))
    print(json.dumps(dict(manifest=str(output/'manifest.json'), motions=len(receipts), training_started=False)))


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--export', type=Path, help='Export completed synthesis population as canonical references')
    parser.add_argument('--capture_root', type=Path)
    parser.add_argument('--population_report', type=Path)
    parser.add_argument('--task_manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=Path)
    parser.add_argument('--headless', action='store_true')
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        args.visualizer = None
    if args.export is not None:
        export_references(args.export, args.output)
    elif args.prepare:
        if args.population_report is None:
            parser.error("--prepare requires the event-aligned --population_report")
        prepare(args.capture_root, args.task_manifest, args.output, args.population_report)
    else:
        synthesize(args.task_manifest, args.jobs, args, launch_simulation)


if __name__ == '__main__':
    main()
