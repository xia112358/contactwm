"""Staged event alignment, representative-residual fusion and native audits."""
import json
import hashlib
import threading
import queue
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import torch

from . import trajectory_refinement as workflow
from . import extraction_standard as standard
from .consensus_event_alignment import align_observed_events, resample_motion
from .consensus_continuity import acceptance_status
from .event_continuity import EventContinuityGuard
from .motion_signal import smooth_motion_signal, local_smooth_motion_signal
from .parallel_rollout_collection import load_candidates, save_arrays
from .rollout_dataset import complete_segment
from .observed_event_consensus import build_consensus, audit_consensus, strategy_groups, observed_consensus_subset
from .rollout_consensus import LINKS, PART_LINKS, fit_consensus
from .staged_consensus import motion_medoid, fusion_stages, apply_residual, localized_transfer

_FK_EXPORT_LOCK = threading.Lock()


def reuse_roots(value):
    if value is None:
        return []
    return [Path(p) for p in (value if isinstance(value, (list, tuple)) else [value])]


def reuse_case_results(roots, key):
    """Resolve receipts, not guessed cache locations beside copied results."""
    results = []
    for root in reuse_roots(roots):
        path = root/'aggregated'/key/'result.json'
        if path.exists():
            saved = workflow.read_json(path)
            if saved['key'] != key:
                raise ValueError('Reused result belongs to a different case')
            results.append((path, saved))
    return results


def source_observation_index(results):
    indexed = {}
    for _, saved in results:
        for artifact in saved['artifacts']:
            path = Path(artifact['path'])
            if path.parent.name == 'source_observations':
                rows = indexed.setdefault(path.name, [])
                if artifact not in rows:
                    rows.append(artifact)
    return indexed


def cached_source_observation(artifacts, q, provenance):
    """A pinned source may live anywhere; pose and actual model must match."""
    for artifact in artifacts:
        workflow.verify_files([artifact])
        path = Path(artifact['path'])
        with np.load(path, allow_pickle=False) as z:
            if (np.array_equal(q, z['joint_pos'])
                    and json.loads(z['provenance_json'].item()) == provenance):
                return path
    return None


def parallel_case_results(cases, worker, workers, *, poll=None, abort=None):
    """Bound in-flight cases, propagate errors, and drain active writes on exit."""
    if type(workers) is not int or workers < 1:
        raise ValueError('case_workers must be a positive integer')
    iterator = iter(cases)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='consensus')
    pending = set()
    try:
        for _ in range(workers):
            case = next(iterator, None)
            if case is not None:
                pending.add(executor.submit(worker, case))
        while pending:
            done, pending = wait(pending, timeout=0 if poll is not None else None, return_when=FIRST_COMPLETED)
            if not done and poll is not None:
                poll()
            results = [future.result() for future in done]
            for result in results:
                yield result
                case = next(iterator, None)
                if case is not None:
                    pending.add(executor.submit(worker, case))
    finally:
        if abort is not None:
            abort()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class SharedNativeQueries:
    """One solver owner; independent case workers overlap CPU, IO and fitting.

    Every fixed-q pass and CPU audit runs on the initializing main thread.
    Workers enqueue requests and retain their own copied results. In particular,
    a CPU audit cannot block inside a worker while holding the solver lock.
    """
    def __init__(self, scene):
        self.scene, self.metadata = scene, scene.metadata
        self.owner = threading.get_ident()
        self.requests = queue.Queue()
        self.state_lock = threading.Lock()
        self.closed = False
        device = torch.device(str(scene.native.model.device))
        self.stream = torch.cuda.default_stream(device) if device.type == 'cuda' else None

    def _call(self, operation):
        ready, caller = None, None
        if self.stream is not None:
            caller = torch.cuda.current_stream(self.stream.device)
            ready = torch.cuda.Event(); ready.record(caller)
        future = Future()
        with self.state_lock:
            if self.closed:
                raise RuntimeError('Native query owner has stopped')
            self.requests.put((operation, ready, future))
        result, finished = future.result()
        if caller is not None:
            caller.wait_event(finished)
            def retain(value):
                if isinstance(value, torch.Tensor) and value.is_cuda: value.record_stream(caller)
                elif isinstance(value, dict):
                    for child in value.values(): retain(child)
            retain(result)
        return result

    def service_one(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Only the initializing thread may run Newton queries')
        try:
            operation, ready, future = self.requests.get(timeout=.01)
        except queue.Empty:
            return
        try:
            finished = None
            if self.stream is None:
                result = operation()
            else:
                import warp as wp
                self.stream.wait_event(ready)
                with torch.cuda.stream(self.stream), wp.ScopedStream(wp.stream_from_torch(self.stream)):
                    result = operation()
                    finished = torch.cuda.Event(); finished.record(self.stream)
            future.set_result((result, finished))
        except BaseException as error:
            future.set_exception(error)

    def close(self):
        with self.state_lock:
            self.closed = True
            while True:
                try: _, _, future = self.requests.get_nowait()
                except queue.Empty: break
                future.set_exception(RuntimeError('Native query owner has stopped'))

    def query(self, q):
        def operation():
            if self.stream is not None:
                q.record_stream(self.stream)
            return self.scene.query(q)
        return self._call(operation)

    def cpu_query(self, q, scenes):
        return self._call(lambda: self.scene.native(q, scenes))

    def run_native(self, operation):
        """CPU Newton FK also launches Warp kernels and shares its runtime."""
        return self._call(operation)


class NativeScene:
    """Same compact native fields as the IPC worker, without a second process."""
    def __init__(self, query, transport=None):
        from .newton_contact_tensors import NewtonContactTensorReader
        from .newton_tensor_transport import native_geometry_metadata, TensorSceneRouter
        self.native = query
        self.transport = transport
        if transport is not None:
            self.metadata = transport.metadata
            self.router = TensorSceneRouter({0: transport})
            return
        query.tensor_reader = NewtonContactTensorReader(query.model, query.solver, query.body_env, query.shape_surface)
        self.metadata = native_geometry_metadata(query, query.tensor_reader.link_names)
        self.router = TensorSceneRouter({0: self})

    def query(self, q):
        result = self.native.query_device(q)
        valid = (torch.arange(len(result['dist']), device=q.device) < result['count']) & (result['worldid'] < len(q))
        fields = ('dist', 'includemargin', 'type', 'worldid', 'active', 'constraint_allocated',
                  'constraint_rows', 'efc_address', 'shape0', 'shape1', 'body0', 'body1', 'source_key',
                  'primary_surface', 'part', 'task_pair', 'eligible', 'upward', 'full_kind',
                  'body_link0', 'body_link1', 'normal_w', 'geometry_point0_w', 'geometry_point1_w')
        copied = {key: result[key][valid].clone() for key in fields}
        copied.update({key: result[key][:len(q)].clone() for key in
                       ('contact_part_mask', 'contact_surface', 'contact_position_w')})
        return copied

    def __call__(self, q):
        return self.router(q.detach(), torch.zeros(len(q), device=q.device, dtype=torch.long))


def labels(observed, fps):
    from .contact_labels import process_contact_sequence
    return process_contact_sequence(*(observed[k].detach().cpu().numpy() for k in
        ('contact_part_mask', 'contact_surface', 'contact_position_w')), fps=fps)


def save_observations(path, observed, q, provenance):
    arrays = {"pair_"+k: v.detach().cpu().numpy() for k, v in observed["pairs"].items()}
    arrays.update({k: observed[k].detach().cpu().numpy() for k in
                   ('contact_part_mask', 'contact_surface', 'contact_position_w')})
    arrays.update(joint_pos=np.asarray(q), provenance_json=np.asarray(json.dumps(provenance)),
                  sampling=np.asarray('fixed_q_forward_no_integration'))
    save_arrays(path, arrays)


class ContactGuard:
    """Refresh actual witnesses; between queries use only a frozen FK guide."""
    def __init__(self, scene, case, reports, masks, surfaces, *, refresh=10, source_depths=None):
        self.scene, self.case = scene, case
        self.reports, self.masks, self.surfaces = reports, masks, surfaces
        self.refresh, self.history = refresh, []
        self.source_depths = source_depths

    def prepare(self, medoid, template, fk):
        self.history = []
        # Equal execution votes on the common clock, never one representative's
        # contact schedule. Minority contact remains audit evidence only.
        masks, surfaces = np.asarray(self.masks, bool), np.asarray(self.surfaces)
        wanted = np.zeros(masks.shape[1:], bool)
        faces = np.full(surfaces.shape[1:], -1, dtype=surfaces.dtype)
        for face in np.unique(surfaces[masks]):
            majority = ((surfaces == face) & masks).sum(0)*2 > len(masks)
            wanted |= majority; faces[majority] = face
        release = (~masks).sum(0)*2 > len(masks)
        self.event_guides = None
        if self.case.get('observed_consensus') and self.reports:
            from .consensus_event_targets import event_targets
            self.event_guides = event_targets(self.case['observed_consensus'], self.reports,
                masks, surfaces, self.source_depths)
            wanted, faces, release = (self.event_guides[k] for k in ('wanted', 'faces', 'release'))
        self.wanted = torch.as_tensor(wanted, device=template.device)
        self.faces = torch.as_tensor(faces, device=template.device)
        self.release = torch.as_tensor(release, device=template.device)
        self.target_depth = self.allowed_penetration = None
        if self.source_depths is not None and self.event_guides is not None:
            self.target_depth = torch.as_tensor(self.event_guides['target_depth'], device=template.device, dtype=template.dtype)
            self.allowed_penetration = torch.as_tensor(self.event_guides['allowed_penetration'], device=template.device, dtype=template.dtype)
        elif self.source_depths is not None:
            depth = torch.as_tensor(np.asarray(self.source_depths), device=template.device, dtype=template.dtype)
            valid = torch.as_tensor(masks & (surfaces == faces[None]), device=template.device) & torch.isfinite(depth)
            if bool((self.wanted & ~valid.any(0)).any()):
                raise ValueError('Missing native pair depth for observed majority contact')
            self.target_depth = torch.nanmedian(torch.where(valid, depth, float('nan')),dim=0).values
            minimum = torch.where(valid, depth, float('inf')).amin(0)
            self.allowed_penetration = torch.where(torch.isfinite(minimum), minimum.clamp_max(0), 0)
        with torch.no_grad():
            self.template_p, self.template_r = fk.link_poses(template, PART_LINKS)
            self.template_p = self.template_p.clone()
            self.template_r = self.template_r.clone()

    def __call__(self, step, q, fk, *, final=False):
        from contact_solver.device_contact_objective import first_minimum, group_any
        validation = None
        if step % self.refresh == 0 or final:
            self.observed = self.scene(q)
            p = self.observed['pairs']
            active = ((p['type'].long() & 1) != 0) & (p['dist'] < p['includemargin'])
            if not torch.equal(active, p['active']) or bool((active & ~p['constraint_allocated']).any()):
                raise ValueError('Inconsistent/unallocated actual Newton contact')
            self.points = torch.stack((p['geometry_point0_w'], p['geometry_point1_w']), 1)
            self.links = torch.stack((p['body_link0'], p['body_link1']), 1)
            with torch.no_grad():
                pos, rot = fk.link_poses(q, self.observed['link_names'])
                pos = pos[p['sample'][:, None], self.links.clamp_min(0)]
                rot = rot[p['sample'][:, None], self.links.clamp_min(0)]
                self.local = torch.matmul(rot.transpose(-1, -2), (self.points-pos)[..., None])
            validation = audit_consensus(self.case["observed_consensus"], self.observed["contact_part_mask"].cpu().numpy(), self.observed["contact_surface"].cpu().numpy())
            self.history.append(dict(step=step, failed_checks=validation['failed_checks']))
        p = self.observed['pairs']; sample = p['sample']; part = p['part'].clamp(0, 5)
        normal = p['normal_w']
        normal_valid = torch.isfinite(normal).all(-1) & torch.isclose(normal.norm(dim=-1), torch.ones_like(normal[:, 0]), atol=1e-4, rtol=0)
        if bool(((p['task_pair'] | ((p['full_kind'] >= 0) & (p['dist'] < 0))) & ~normal_valid).any()):
            raise ValueError('Invalid native contact normal')
        pos, rot = fk.link_poses(q, self.observed['link_names'])
        moved = pos[sample[:, None], self.links.clamp_min(0)] + torch.matmul(
            rot[sample[:, None], self.links.clamp_min(0)], self.local).squeeze(-1)
        delta = torch.where((self.links >= 0)[..., None], moved-self.points, 0)
        distance = p['dist']+(torch.where(normal_valid[:, None], normal, 0)*(delta[:, 1]-delta[:, 0])).sum(-1)
        group = sample*6+part
        match = p['task_pair'] & p['upward'] & self.wanted[sample, part] & (p['primary_surface'] == self.faces[sample, part])
        chosen, present = first_minimum(p['dist'], group, match, len(q)*6)
        rows = chosen[present]
        # Interior goal is a fraction of the ACTUAL pair margin, not a contact
        # definition. Activation/allocation above remains the only evidence.
        target = .9*p['includemargin'][rows]
        if self.target_depth is not None:
            observed_target = self.target_depth[sample[rows],part[rows]]
            strict_margin = torch.nextafter(p['includemargin'][rows], torch.full_like(target, -float('inf')))
            target = torch.minimum(observed_target, strict_margin)
        cost = ((distance[rows]-target).relu()/.001).square()
        retention = cost.sum()/self.wanted.sum().clamp_min(1)
        missing = self.wanted & ~present.reshape(len(q), 6)
        position, rotation = fk.link_poses(q, PART_LINKS)
        approach = ((position-self.template_p)/.01).square().mean(-1)+((rotation-self.template_r)/.1).square().mean((-1, -2))
        retention = retention + (approach*missing).sum()/self.wanted.sum().clamp_min(1)
        release_pairs = p['task_pair'] & p['upward'] & self.release[sample, part]
        release_cost = (((1.1*p['includemargin']-distance).relu()/.001).square()*release_pairs).sum()/self.release.sum().clamp_min(1)
        allowed = torch.zeros_like(distance)
        if self.allowed_penetration is not None:
            same_face = p['task_pair'] & p['upward'] & (p['primary_surface'] == self.faces[sample,part])
            allowed = torch.where(same_face, self.allowed_penetration[sample,part], allowed)
        eligible_collision = (p['full_kind'] >= 0) & normal_valid
        excess_penetration = (allowed-distance)[eligible_collision]
        collision = (excess_penetration.relu()/.001).square().mean() if len(excess_penetration) else q.sum()*0
        self.history[-1].update(missing_candidate_part_frames=int(missing.sum()),
            realized_required_part_frames=int(group_any(group, match & p['eligible'], len(q)*6).sum()))
        return 10*retention+release_cost+collision, validation


def motion_metrics(q, fk):
    with torch.no_grad():
        p, r = fk.link_poses(q, LINKS)
        offsets = q.new_tensor([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
        markers = p[..., None, :]+torch.einsum('...ij,kj->...ki', r, offsets)
        return dict(marker_acceleration_rms_m_per_sample2=float(torch.diff(markers, n=2, dim=0).square().mean().sqrt()),
                    joint_acceleration_rms_rad_per_sample2=float(torch.diff(q[:, 7:], n=2, dim=0).square().mean().sqrt()))


def retained_consensus_candidates(saved, source_envs):
    """Reuse final consensus and at most one useful stage, never replay history."""
    contributing = saved.get('contributing_env_ids', [])
    optimization = saved.get('optimization', {})
    if (not saved.get('aggregation_written') or len(set(contributing)) < 3
            or set(contributing) != set(source_envs)
            or optimization.get('representative_fallback', True)):
        return []
    directory = Path(saved['motion']).parent.resolve()
    indexed = {Path(a['path']).resolve() for a in saved['artifacts']}
    paths = [('reused_final', Path(saved['motion']), 'joint_pos')]
    rows = [r for r in optimization.get('stages', [])
            if not r['stage'].endswith(('representative', 'representative_joint_bounded'))]
    rows.sort(key=lambda r: (r.get('joint_limit_violations', 0) > 0, r['failed_checks'],
        r['local_violations'] > 0, r.get('max_acceleration_ratio', float('inf')),
        r['metrics']['marker_acceleration_rms_m_per_sample2']))
    for row in rows:
        path = directory/'stages'/row['stage']/'observations.npz'
        if path in indexed:
            paths.append(('reused_candidate', path, 'joint_pos'))
            break
    return paths


def has_verified_consensus(candidates):
    """Candidate membership already requires multi-source provenance and limits."""
    return any(rank[0] == 0 for rank, *_ in candidates)


def run_case(scene, case, source, destination, metadata, binding, options, reuse_root=None, env_filter=None, rebuild=False):
    from .g1_kinematics import CanonicalG1ForwardKinematics
    from .robot_assets import encode_robot_asset_json
    from .newton_contact_data import load_contact_labels
    from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk
    from relabel_newton_motion import relabel
    options = standard.fit_options(options)
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination/'result.json'
    if completed.exists():
        result = workflow.read_json(completed)
        if result.get('standard_validation'): standard.verify_validation(result)
        else: workflow.verify_files(result['artifacts'])
        return result
    c = load_candidates(source/'candidates'/case['key'], fields={
        'motion_time_step', 'motion_start_step', 'terminated', 'timeout',
        'root_pos', 'root_quat_xyzw', 'dof_pos'})
    qs, native, envs = [], [], []
    for column, env in enumerate(c['env_ids']):
        if env_filter is not None and int(env) not in env_filter: continue
        ix = complete_segment(c['motion_time_step'][:, column]-c['motion_start_step'][:, column],
                              c['terminated'][:, column], c['timeout'][:, column], case['frames'])
        done = metadata['env_done_steps'][int(env)]
        if ix is None or (done is not None and done <= int(ix[-1]) and not metadata['env_done_is_timeout'][int(env)]):
            continue
        qs.append(np.concatenate((c['root_pos'][ix, column], c['root_quat_xyzw'][ix, column][:, [3, 0, 1, 2]], c['dof_pos'][ix, column]), -1))
        native.append(ix); envs.append(int(env))
    del c
    device = str(scene.native.model.device)
    to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    qs = np.asarray(qs)
    reports, masks, surfaces = [], [], []
    artifacts = [workflow.fingerprint(p) for p in sorted((source/'candidates'/case['key']).glob('chunk_*.npz'))]
    previous_results = reuse_case_results(reuse_root, case['key'])
    cached_sources = source_observation_index(previous_results)
    cache_hits = 0
    for i, (q, env) in enumerate(zip(qs, envs)):
        path = destination/'source_observations'/f'env_{env:06d}.npz'
        if not path.exists():
            reused = cached_source_observation(cached_sources.get(path.name, []), q, scene.native.provenance)
            if reused is not None:
                path = reused
                cache_hits += 1
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(q, z['joint_pos']) or json.loads(z['provenance_json'].item()) != scene.native.provenance:
                    raise ValueError('Source observation provenance/pose changed')
                from .contact_labels import process_contact_sequence
                obs = process_contact_sequence(z['contact_part_mask'], z['contact_surface'], z['contact_position_w'], fps=case['fps'])
        else:
            observed = scene(to_tensor(q))
            save_observations(path, observed, q, scene.native.provenance)
            obs = labels(observed, case['fps'])
        # Admission uses complete non-aborted executions, never the authored contract.
        masks.append(obs['contact_part_mask']); surfaces.append(obs['contact_surface'])
        artifacts.append(workflow.fingerprint(path))
        print(f"{case['key']} source {i+1}/{len(envs)}: native evidence loaded", flush=True)
    print(f"{case['key']} source cache hits={cache_hits}/{len(envs)}", flush=True)
    if len(qs) < 3:
        result = dict(processing_standard=standard.STANDARD, key=case['key'], status='insufficient_complete_sources', complete_rollouts=len(qs),
                      native_qualified=0, aggregation_written=False, training_ready=False, artifacts=artifacts)
        workflow.write_json(completed, result, immutable=True); return result
    groups = [list(range(len(qs)))] if "support_consensus" in case else strategy_groups(masks, surfaces)
    if len(groups) > 1 and env_filter is None:
        variants = []
        for group_index, group in enumerate(groups):
            if len(group) < 3: continue
            variant = run_case(scene, case, source, destination/'strategies'/f'group_{group_index:03d}',
                metadata, binding, options, reuse_root, env_filter={envs[i] for i in group}, rebuild=rebuild)
            variant['result_path'] = str(destination/'strategies'/f'group_{group_index:03d}'/'result.json')
            variants.append(variant)
        if variants:
            primary = min(variants, key=lambda r: (standard.classification(r) != 'accepted', not r['aggregation_written'],
                                                   r.get('task_acceptance', {}).get('failed_checks', float('inf'))))
            result = dict(primary, complete_rollouts=len(qs), strategies=[dict(status=r['status'], extraction_status=standard.classification(r), motion=r.get('motion'), result_path=r['result_path'],
                          contributing_env_ids=r.get('contributing_env_ids', [])) for r in variants],
                          source_strategy_groups=[[envs[i] for i in g] for g in groups],
                          unaggregated_source_groups=[[envs[i] for i in g] for g in groups if len(g)<3])
            result['artifacts'] = artifacts+[a for r in variants for a in r['artifacts']]+[workflow.fingerprint(r['result_path']) for r in variants]
        else:
            result = dict(processing_standard=standard.STANDARD, key=case['key'], status='insufficient_sources_per_strategy', complete_rollouts=len(qs),
                          native_qualified=0, aggregation_written=False, training_ready=False, artifacts=artifacts,
                          unaggregated_source_groups=[[envs[i] for i in g] for g in groups])
        workflow.write_json(completed, result, immutable=True); return result
    consensus = case['support_consensus'] if 'support_consensus' in case else build_consensus(masks, surfaces)
    reports = [audit_consensus(consensus, m, s) for m, s in zip(masks, surfaces)]
    if 'support_consensus' in case:
        # Preserve the measured substep support-event clock. Fixed-q native
        # queries audit synthesized geometry; they do not invent load evidence.
        for i, report in enumerate(reports):
            report['source_pose_contact_audit'] = dict(report)
            report['checks'] = [dict(action=e['id'], kind='establish', part=e['part'], surface=e['surface'],
                passed=str(i) in e['observations'], episode=e['observations'].get(str(i)))
                for e in consensus['events']]
            report['failed_checks'] = sum(not check['passed'] for check in report['checks'])
            report['passed'] = report['failed_checks'] == 0
    case = dict(case, observed_consensus=consensus)
    workflow.write_json(destination/'event_consensus.json', consensus, immutable=True)
    artifacts.append(workflow.fingerprint(destination/'event_consensus.json'))
    native_qualified = sum(r['passed'] for r in reports)
    selected = list(range(len(reports)))
    selection = dict(schema='population_consensus_admission_v1', selected=selected,
                     rejected=[], source_consensus_mismatches=[r['failed_checks'] for r in reports],
                     original_reference_used=False, task_success_verified=False,
                     execution_evidence='complete segment without recorded non-timeout termination')
    result = dict(processing_standard=standard.STANDARD, key=case['key'], complete_rollouts=len(qs), source_env_ids=envs,
                  source_acceptance=reports, native_qualified=len(qs), source_consensus_pass_count=native_qualified, event_candidates=len(selected), artifacts=artifacts,
                  aggregation_written=False, training_ready=False, source_selection=selection)
    result['source_cache'] = dict(hits=cache_hits, sources=len(envs),
        lookup='pinned observation artifacts across prioritized prior result receipts')
    if len(selected) < 3:
        result['status'] = 'insufficient_compatible_successful_sources'
        workflow.write_json(completed, result, immutable=True); return result
    if not consensus['events']:
        result['status'] = 'no_shared_observed_events'
        workflow.write_json(completed, result, immutable=True); return result
    representative, _ = motion_medoid(qs[selected])
    maps, event_report = align_observed_events([reports[i] for i in selected], case['frames'],
        reference_index=representative, motions=qs[selected], pose_band=options.get('band', 10))
    aligned_reports, keep, aligned_masks, aligned_faces, aligned_depths = [], [], [], [], []
    alignment_checks = []
    for j, i in enumerate(selected):
        pose_report = reports[i]['source_pose_contact_audit'] if 'support_consensus' in case else reports[i]
        own_events = observed_consensus_subset(consensus, pose_report)
        proposed_map = maps[j].copy()
        attempts = []
        for amount in (1., .5, .25, 0.):
            mapping = np.arange(case['frames'])+amount*(proposed_map-np.arange(case['frames']))
            transported = resample_motion(qs[i], mapping)
            observed = scene(to_tensor(transported))
            obs = labels(observed, case['fps'])
            mask, surface = obs['contact_part_mask'], obs['contact_surface']
            transport_check = audit_consensus(own_events, mask, surface)
            attempts.append(dict(alignment_amount=amount, information_loss=transport_check['failed_checks']))
            if transport_check['passed'] or amount == 0:
                maps[j] = mapping
                break
        # Source observations are retained. If the warp damages stable evidence,
        # reduce the warp, not the source population. Identity is the baseline.
        if own_events['events'] and not transport_check['passed']:
            raise ValueError('Unchanged source no longer reproduces its observed contact information')
        path = destination/'aligned_observations'/f'env_{envs[i]:06d}.npz'
        save_observations(path, observed, transported, scene.native.provenance)
        result['artifacts'].append(workflow.fingerprint(path))
        report = audit_consensus(consensus, mask, surface)
        alignment_checks.append(dict(env_id=envs[i], acceptance=report,
            own_event_preservation=transport_check, alignment_attempts=attempts,
            selected_alignment_amount=amount, source_discarded=False))
        print(f"{case['key']} source {j+1}/{len(selected)}: alignment={amount:g}, information_loss={transport_check['failed_checks']}", flush=True)
        keep.append(j); aligned_reports.append(report); aligned_masks.append(mask); aligned_faces.append(surface)
        pairs = observed['pairs']
        eligible = pairs['eligible'] & pairs['task_pair'] & pairs['upward'] & (pairs['part'] >= 0)
        eligible &= pairs['primary_surface'] == observed['contact_surface'][pairs['sample'], pairs['part'].clamp(0,5)]
        depth = torch.full((case['frames']*6,), float('inf'), device=device)
        group = pairs['sample']*6+pairs['part'].clamp(0,5)
        depth.scatter_reduce_(0, group[eligible], pairs['dist'][eligible], reduce='amin', include_self=True)
        aligned_depths.append(depth.reshape(case['frames'],6).cpu().numpy())
    result.update(alignment=event_report, alignment_checks=alignment_checks, aligned_qualified=len(keep))
    registration = event_report.get('joint_registration') or {}
    from .continuous_motion import certify_clocks
    applied_controls=np.asarray(registration['clock_displacement_controls'])*np.asarray([
        c['selected_alignment_amount'] for c in alignment_checks])[:,None]
    result['clock_validation'] = certify_clocks(maps,registration['clock_control_knots'],
                                               applied_controls,event_report['rate_bounds'])
    result['clock_validation']['source_information_preserved']=all(
        c['own_event_preservation']['passed'] for c in alignment_checks)
    result['clock_validation']['passed'] &= result['clock_validation']['source_information_preserved']
    if len(keep) < 3:
        result['status'] = 'insufficient_qualified_after_event_alignment'
        workflow.write_json(completed, result, immutable=True); return result
    maps = maps[keep]; selected = [selected[j] for j in keep]
    # Population events define the clock; a failed pose initializer can be
    # replaced without changing that clock or any source's validated mapping.
    original_representative = representative
    if representative in keep:
        representative = keep.index(representative)
    else:
        representative, _ = motion_medoid(np.asarray([
            resample_motion(qs[i], mapping) for i, mapping in zip(selected, maps)]))
    result['pose_initialization'] = dict(original_source_index=original_representative,
        selected_env_id=envs[selected[representative]],
        changed=original_representative not in keep, population_clock_unchanged=True)
    fk = CanonicalG1ForwardKinematics().to(device)
    continuity_guard = EventContinuityGuard(qs[selected], [reports[i] for i in selected], case['fps'], fk)
    calibration = []
    for j, i in enumerate(selected):
        raw_audit = continuity_guard.check(qs[i], fk, reports[i])
        aligned_audit = continuity_guard.check(resample_motion(qs[i], maps[j]), fk, aligned_reports[j])
        calibration.append(dict(env_id=envs[i], native_local_violations=raw_audit['violation_count'],
            aligned_local_violations=aligned_audit['violation_count'],
            native_max_ratio=raw_audit['max_ratio_by_derivative'][1],
            aligned_max_ratio=aligned_audit['max_ratio_by_derivative'][1]))
    workflow.write_json(destination/'source_continuity.json', dict(sources=calibration,
        interpretation='excess over native corresponding-phase envelope; population deviation is not source error',
        calibration_warning=any(r['native_local_violations'] for r in calibration)), immutable=True)
    result['artifacts'].append(workflow.fingerprint(destination/'source_continuity.json'))
    metrics = dict(sources=[motion_metrics(to_tensor(resample_motion(qs[i], m)), fk) for i, m in zip(selected, maps)])
    previous = source/'aggregated'/case['key']/'motion.npz'
    initial_pose = None
    if previous.exists():
        with np.load(previous, allow_pickle=False) as z:
            initial_pose = z['joint_pos'].copy()
            metrics['previous_aggregate'] = motion_metrics(to_tensor(initial_pose), fk)
        result['artifacts'].append(workflow.fingerprint(previous))
    aligned = np.asarray([resample_motion(qs[i], m) for i, m in zip(selected, maps)])
    native_clock_guard = continuity_guard
    continuity_guard = EventContinuityGuard(aligned, aligned_reports, case['fps'], fk,
                                           reference_clock='transported')
    base, delta, smooth, fusion = fusion_stages(aligned, representative, event_report['output_anchors'], fps=case['fps'])
    stages, candidates, stage_observations = [], [], {}
    lower, upper = fk.joint_lower.cpu().numpy(), fk.joint_upper.cpu().numpy()
    # Preserve contact already achieved by the population fusion. Smoothing
    # must not trade away support for a smaller pose/smoothness objective.
    from .consensus_support_preservation import support_reference, support_loss
    fused = apply_residual(base, delta)
    fused_bounded=fused.copy();fused_bounded[:,7:]=np.clip(fused_bounded[:,7:],lower,upper)
    fused_native=labels(scene(to_tensor(fused_bounded)),case['fps'])
    support_intent=support_reference(consensus,aligned_reports,np.asarray(aligned_masks),
        np.asarray(aligned_faces),fused_native['contact_part_mask'],fused_native['contact_surface'],maps,selected)
    def audit_stage(name, pose, *, eligible=True, parent=None):
        observed = scene(to_tensor(pose))
        # Raw actual witnesses are retained for every transformed candidate.
        pairs = observed['pairs']
        active = ((pairs['type'].long() & 1) != 0) & (pairs['dist'] < pairs['includemargin'])
        if not torch.equal(active, pairs['active']) or bool((active & ~pairs['constraint_allocated']).any()):
            raise ValueError('Inconsistent/unallocated actual Newton stage contact')
        observed_labels = labels(observed, case['fps'])
        task = audit_consensus(consensus, observed_labels['contact_part_mask'], observed_labels['contact_surface'])
        preservation=support_loss(support_intent,observed_labels['contact_part_mask'],observed_labels['contact_surface'])
        stage_observations[name] = (task, observed_labels['contact_part_mask'], observed_labels['contact_surface'])
        continuity = continuity_guard.check(pose, fk, task)
        native_clock = native_clock_guard.check(pose, fk, task)
        joint_limit_count = int(((pose[:, 7:] < lower-1e-6) | (pose[:, 7:] > upper+1e-6)).sum())
        path = destination/'stages'/name/'observations.npz'
        save_observations(path, observed, pose, scene.native.provenance)
        audit = dict(stage=name, parent=parent, task_acceptance=task, support_preservation=preservation, continuity=continuity,
                     native_clock_diagnostics=dict(passed=native_clock['passed'],
                        violation_count=native_clock['violation_count'], violations=native_clock['violations'],
                        max_ratio_by_derivative=native_clock['max_ratio_by_derivative'],
                        interpretation='includes legitimate clock-induced acceleration changes; diagnostic only'),
                     joint_limit_violations=joint_limit_count, metrics=motion_metrics(to_tensor(pose), fk))
        workflow.write_json(path.parent/'audit.json', audit, immutable=True)
        result['artifacts'].extend(workflow.fingerprint(p) for p in (path, path.parent/'audit.json'))
        from .rollout_consensus import population_fit_cost
        with torch.no_grad():
            fit_cost=float(population_fit_cost(to_tensor(pose),to_tensor(aligned),
                torch.as_tensor(np.asarray(aligned_masks),device=device),fk,
                marker_smooth_weight=options['marker_smooth_weight'],joint_smooth_weight=options['joint_smooth_weight']))
        row = dict(stage=name, parent=parent, support_preservation=preservation, population_fit_cost=fit_cost, failed_checks=task['failed_checks'],
                   local_violations=continuity['violation_count'], joint_limit_violations=joint_limit_count,
                   max_acceleration_ratio=continuity['max_ratio_by_derivative'][1], metrics=audit['metrics'])
        stages.append(row)
        if eligible and joint_limit_count == 0:
            rank = (task['failed_checks'], preservation['lost_part_frames'], fit_cost,
                    audit['metrics']['marker_acceleration_rms_m_per_sample2'], row['local_violations'],
                    max(0., row['max_acceleration_ratio']-1))
            candidates.append((rank, name, pose.copy(), task))
        print(f"{case['key']} stage {name}: contact={row['failed_checks']}, local={row['local_violations']}", flush=True)
        if eligible and joint_limit_count:
            # Preserve the unmodified stage and audit a separate bounded pose.
            # This prevents a small source overshoot from forcing initialization
            # from an unrelated old aggregate. Contact/continuity still decide.
            bounded = pose.copy()
            bounded[:, 7:] = np.clip(bounded[:, 7:], lower, upper)
            audit_stage(name+'_joint_bounded', bounded, parent=name)
        return task
    audit_stage('representative', base, eligible=False)
    if initial_pose is not None:
        audit_stage('previous_aggregate', initial_pose, eligible=False)
    seen_poses = set()
    for prior_index, (previous_result, saved) in enumerate(previous_results):
        if has_verified_consensus(candidates):
            break
        workflow.verify_files(saved['artifacts'])
        result['artifacts'].append(workflow.fingerprint(previous_result))
        for name, path, field in retained_consensus_candidates(saved, np.asarray(envs)[selected].tolist()):
            if has_verified_consensus(candidates):
                break
            with np.load(path, allow_pickle=False) as z:
                pose = z[field].copy()
            digest = hashlib.sha256(pose.tobytes()).hexdigest()
            if digest in seen_poses: continue
            seen_poses.add(digest)
            audit_stage(f'{name}_{prior_index}', pose)
    def attempt_local(name, pose, task):
        if task['passed']:
            return
        corrected, correction = localized_transfer(pose, base, task, fps=case['fps'],
            representative_report=stage_observations['representative'][0],
            candidate_mask=stage_observations[name][1], candidate_surface=stage_observations[name][2])
        workflow.write_json(destination/'stages'/name/'correction.json', correction, immutable=True)
        result['artifacts'].append(workflow.fingerprint(destination/'stages'/name/'correction.json'))
        if corrected is not None:
            audit_stage(name+'_local', corrected, parent=name)
    reused_verified = has_verified_consensus(candidates) and not rebuild
    if not reused_verified:
        registration = event_report.get('joint_registration')
        if registration and registration.get('consensus_pose'):
            audit_stage('joint_motion_clock_consensus', np.asarray(registration['consensus_pose'],dtype=qs.dtype))
        task = audit_stage('fused', fused, parent='representative')
        attempt_local('fused', fused, task)
        for time_scale in (.04, .08, .16):
            pose = smooth_motion_signal(fused, fps=case['fps'], time_scale=time_scale)
            audit_stage(f'whole_signal_{time_scale:g}s', pose, parent='fused')
        for amount in (1., .5, .25):
            name = f'smoothed_{amount:g}'
            pose = apply_residual(base, amount*smooth)
            task = audit_stage(name, pose, parent='fused')
            attempt_local(name, pose, task)
    if not candidates:
        result.update(status='no_joint_limit_valid_candidate', stages=stages)
        workflow.write_json(completed, result, immutable=True); return result
    # Registration, denoising and audits are required. A new pose optimization
    # is necessary only when no multi-execution candidate passes those audits.
    guard = ContactGuard(scene, case, aligned_reports, np.asarray(aligned_masks), np.asarray(aligned_faces), source_depths=aligned_depths)
    fit_options = dict(options)
    fit_options.setdefault('correction_time_scale', .12)
    fit_options.setdefault('learning_rate', .0003)
    fit_options.setdefault('fps', case['fps'])
    whole_body_parent = None
    fit_report = dict(optimizer='not run; verified multi-execution consensus', steps=0,
        stop_reason='verified_existing_consensus' if reused_verified else 'verified_denoised_consensus')
    if rebuild or not has_verified_consensus(candidates):
        _, parent, warm_start, _ = min(candidates, key=lambda row: row[0])
        whole_body_parent = parent
        fitted, fitted_maps, fit_report = fit_consensus(qs[selected], base, None, None, None,
            device=device, mappings=maps, contact_guard=guard, continuity_guard=continuity_guard,
            initial_pose=warm_start, representative_index=representative, **fit_options)
        if not np.array_equal(fitted_maps, maps):
            raise ValueError('Full-body fitting changed the validated shared clock')
        fit_task = audit_stage('whole_body_fit', fitted, parent=parent)
        if fit_task['failed_checks'] != fit_report['validated_failed_checks']:
            raise ValueError('Full-body fitted candidate disagrees with its native validation')
        fit_report.update(release_supervision='shared native release between consensus episodes only',
                          contact_retention='population episode intent and equal-execution measured event depth; fresh Newton feedback')
        workflow.write_json(destination/'stages'/'whole_body_fit'/'fit.json', fit_report, immutable=True)
        result['artifacts'].append(workflow.fingerprint(destination/'stages'/'whole_body_fit'/'fit.json'))
    if getattr(guard, 'event_guides', None) is not None:
        intent = {k: guard.event_guides[k] for k in ('schema', 'events', 'releases', 'contact_truth')}
        intent.update(scope='soft optimization guide from transported native episodes; no fixed acceptance deadlines')
        workflow.write_json(destination/'event_targets.json', intent, immutable=True)
        result['artifacts'].append(workflow.fingerprint(destination/'event_targets.json'))
    local_refinements = []
    for pass_index, support_time_scale in enumerate((.12, .24)):
        rank, parent, incumbent, incumbent_task = min(candidates, key=lambda row: row[0])
        if rank[0] == 0:
            break
        local_audit = continuity_guard.check(incumbent, fk, incumbent_task)
        frames = set(local_audit['violation_frames'])
        missing = {r['action'] for r in incumbent_task['checks'] if not r['passed']}
        for event in guard.event_guides['events'] if guard.event_guides else []:
            if event['id'] in missing:
                frames.update(range(*event['interval']))
        frames = sorted(f for f in frames if 0 < f < case['frames']-1)
        if not frames:
            break
        seed_pose, envelope = local_smooth_motion_signal(incumbent, frames, fps=case['fps'],
            time_scale=.04, support_time_scale=support_time_scale)
        name = f'local_signal_{pass_index+1}'
        audit_stage(name, seed_pose, parent=parent)
        local_options = dict(fit_options, steps=min(400, fit_options.get('steps', 600)),
            learning_rate=.00015, continuity_weight=100., correction_envelope=envelope)
        refined, refined_maps, report = fit_consensus(qs[selected], base, None, None, None,
            device=device, mappings=maps, contact_guard=guard, continuity_guard=continuity_guard,
            initial_pose=seed_pose, representative_index=representative, **local_options)
        if not np.array_equal(refined_maps, maps):
            raise ValueError('Local correction changed the validated shared clock')
        name = f'local_refit_{pass_index+1}'
        task = audit_stage(name, refined, parent=f'local_signal_{pass_index+1}')
        if task['failed_checks'] != report['validated_failed_checks']:
            raise ValueError('Local correction disagrees with native validation')
        report.update(support_time_scale=support_time_scale, violation_frames=frames,
                      correction_scope='compact C2 window; outside poses exactly preserved')
        workflow.write_json(destination/'stages'/name/'fit.json', report, immutable=True)
        result['artifacts'].append(workflow.fingerprint(destination/'stages'/name/'fit.json'))
        local_refinements.append(dict(stage=name, support_time_scale=support_time_scale,
                                     failed_checks=task['failed_checks']))
    _, chosen, q, chosen_task = min(candidates, key=lambda row: row[0])
    optimization = dict(schema=standard.ENGINE, optimizer=fit_report['optimizer'], fusion=fusion, stages=stages,
        selected_stage=chosen, selected_existing_aggregate=chosen.startswith('reused_'),
        representative_fallback=chosen.endswith(('representative', 'representative_joint_bounded')),
        validated_failed_checks=chosen_task['failed_checks'], rms_is_acceptance_condition=False,
        legacy_fit_options=dict(options), legacy_fit_options_applied=True,
        whole_body_fit=fit_report, whole_body_initial_stage=whole_body_parent,
        generation_policy='retain events and achieved population support before ranking pose fit; derivative continuity remains diagnostic',
        local_refinements=local_refinements,
        dynamics='kinematic aggregation; fixed-q contact evidence is not a dynamic feasibility certificate')
    by_stage = {s['stage']: s for s in stages}
    optimization['introduced_failures'] = [dict(stage=s['stage'], parent=s['parent'],
        contact=s['failed_checks'] > by_stage[s['parent']]['failed_checks'],
        local_continuity=s['local_violations'] > 0 and by_stage[s['parent']]['local_violations'] == 0)
        for s in stages if s['parent'] is not None]
    seed, motion, contacts = (destination/name for name in ('seed.npz', 'motion.npz', 'contacts.npz'))
    save_arrays(seed, dict(joint_pos=q, joint_names=np.asarray(metadata['dof_names']), fps=np.asarray(case['fps']),
        robot_asset_json=np.asarray(encode_robot_asset_json(metadata['robot_asset']))))
    save_arrays(destination/'alignment.npz', dict(source_frame_maps=maps, native_record_frames=np.asarray(native)[selected],
        source_env_ids=np.asarray(envs)[selected],clock_displacement_controls=applied_controls[keep],
        clock_control_knots=np.asarray(registration['clock_control_knots'])))
    workflow.write_json(destination/'optimization.json', optimization, immutable=True)
    if not motion.exists():
        # CPU Newton FK shares Warp's native runtime with the GPU solver.
        # A separate FK mutex does not isolate concurrent Warp kernel launches.
        def export():
            with _FK_EXPORT_LOCK:
                return canonicalize_motion_with_direct_newton_fk(seed, motion, device='cpu')
        if scene.transport is not None:
            scene.transport.run_native(export)
        else:
            export()
    if not contacts.exists(): relabel(motion, binding, contacts)
    final_labels = load_contact_labels(motion, contacts)
    task = audit_consensus(consensus, final_labels['contact_part_mask'], final_labels['contact_surface'])
    final_support=support_loss(support_intent,final_labels['contact_part_mask'],final_labels['contact_surface'])
    if task['failed_checks'] != optimization['validated_failed_checks']:
        raise ValueError('Final full Newton labels disagree with native optimization validation')
    with np.load(motion, allow_pickle=False) as z:
        final_q = z['joint_pos'].copy()
    continuity = continuity_guard.check(final_q, fk, task)
    workflow.write_json(destination/'continuity.json', continuity, immutable=True)
    metrics.update(output=motion_metrics(to_tensor(final_q), fk), rms_is_acceptance_condition=False)
    joint_count = int(((final_q[:,7:] < fk.joint_lower.detach().cpu().numpy()-1e-6)
                      | (final_q[:,7:] > fk.joint_upper.detach().cpu().numpy()+1e-6)).sum())
    status = acceptance_status(task, continuity) if not joint_count else 'joint_limits_unresolved'
    result.update(status=status, aggregation_written=True, support_preservation=final_support,
                  motion=str(motion), contacts=str(contacts), task_acceptance=task, event_acceptance=task,
                  task_success_verified=False, smoothness=metrics,
                  contributing_env_ids=np.asarray(envs)[selected].tolist(), optimization=optimization,
                  standard_validation=standard.validation_stamp(motion, contacts),
                  joint_limits=dict(passed=joint_count==0, violation_count=joint_count),
                  continuity=dict(path=str(destination/'continuity.json'), schema=continuity['schema'], passed=continuity['passed'],
                                  violation_count=continuity['violation_count']))
    result['artifacts'] += [workflow.fingerprint(destination/name) for name in
        ('seed.npz', 'alignment.npz', 'optimization.json', 'motion.npz', 'contacts.npz', 'continuity.json')]
    workflow.write_json(completed, result, immutable=True)
    return result


def run_jobs(query, jobs_path, binding):
    import os
    import faulthandler
    from .newton_contact_query import install_contact_query
    from .robot_assets import validate_g1_asset_metadata, somaforge_root
    jobs = workflow.read_json(jobs_path)
    shared = SharedNativeQueries(NativeScene(query))
    source = Path(jobs['source_collection'])
    metadata = workflow.read_json(source/'recording.npz.json')['metadata']
    validate_g1_asset_metadata(metadata.get('robot_asset'), context=str(source/'recording.npz.json'))
    output = Path(jobs['output_dir'])
    if len({c['key'] for c in jobs['cases']}) != len(jobs['cases']):
        raise ValueError('Concurrent aggregation requires unique case keys')
    all_cases = [dict(key=k) for k in workflow.read_json(output/'reaggregation_plan.json')['cases']]
    completed = {}
    for case in all_cases:
        path = output/'aggregated'/case['key']/'result.json'
        if path.exists(): completed[case['key']] = workflow.read_json(path)
    def worker(case):
        # Routing catalogs and differentiable FK caches belong to each case;
        # only immutable metadata and the synchronized solver are shared.
        scene = NativeScene(query, transport=shared)
        return run_case(scene, case, source, output/'aggregated'/case['key'],
                        metadata, binding, jobs['aggregation'], jobs.get('reuse_results'), rebuild=jobs.get('rebuild', False))
    install_contact_query(shared.cpu_query)
    diagnostic = None
    if os.environ.get('SOMAFORGE_CONSENSUS_STACK_LOG'):
        path = Path(os.environ['SOMAFORGE_CONSENSUS_STACK_LOG']).resolve()
        if not path.is_relative_to((somaforge_root()/'tmp').resolve()):
            raise ValueError('Query diagnostics must be inside project tmp')
        path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = path.open('a')
        faulthandler.dump_traceback_later(90, repeat=True, file=diagnostic)
    try:
        workers = jobs.get('case_workers', standard.CASE_WORKERS)
        pending = [case for case in jobs['cases'] if case['key'] not in completed]
        print(f"Concurrent aggregation: case_workers={workers}, query_worlds={query.worlds}, one synchronized solver", flush=True)
        for result in parallel_case_results(pending, worker, workers, poll=shared.service_one, abort=shared.close):
            completed[result['key']] = result
            print(json.dumps(dict(case=result['key'], status=result['status'], complete=result['complete_rollouts'],
                                  qualified=result['native_qualified'])), flush=True)
            # Do not scan another worker's partially written result receipt.
            workflow.write_json(output/'progress.json', standard.summarize(all_cases, list(completed.values())))
    finally:
        install_contact_query(None)
        if diagnostic is not None:
            faulthandler.cancel_dump_traceback_later()
            diagnostic.close()
