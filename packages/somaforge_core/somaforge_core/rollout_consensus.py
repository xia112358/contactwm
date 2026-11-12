"""Preserved event alignment and robust whole-body consensus with smoothing.

The pose objective and scales follow the retained event-alignment experiment:
robust marker fitting, .02 joint prior, .05 marker acceleration, .2 joint
acceleration, Adam(.001) and cosine decay for 600 steps. Contact observations
guide fitting but never become labels of the fitted pose without Newton query.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import uuid

import numpy as np

from . import trajectory_refinement as workflow
from .contact_aggregation import align_event_anchors, align_pose
from .parallel_rollout_collection import load_candidates, save_arrays
from .rollout_dataset import complete_segment

LINKS = ("pelvis", "torso_link", "left_knee_link", "right_knee_link",
         "left_ankle_roll_link", "right_ankle_roll_link", "left_sphere_hand_link",
         "right_sphere_hand_link", "left_elbow_link", "right_elbow_link")
PART_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link", "left_sphere_hand_link",
              "right_sphere_hand_link", "left_knee_link", "right_knee_link")


def event_boundaries(case):
    n = case["frames"]
    actions = (case.get("contract") or {}).get("actions", [])
    if not actions:
        return None
    edges = {0, n-1}
    for action in actions:
        edges.update((int(action["start"]), int(action["end"])))
    if min(edges) < 0 or max(edges) >= n:
        raise ValueError("Event clock is outside this reference")
    return sorted(edges)


class FitConvergence:
    """Stop only after fresh successful audits and sustained objective convergence."""
    def __init__(self, patience, rtol):
        self.patience, self.rtol = patience, rtol
        self.start, self.anchor, self.samples = None, None, 0

    def update(self, step, loss, failed_checks, violations):
        if failed_checks or violations:
            self.start, self.anchor, self.samples = None, None, 0
            return False
        if self.anchor is None or self.anchor-loss > self.rtol*max(abs(self.anchor), 1e-12):
            self.start, self.anchor, self.samples = step, loss, 1
        else:
            self.samples += 1
        return self.samples >= 3 and step-self.start >= self.patience


def population_fit_cost(pose, sources, masks, fk, *, marker_smooth_weight=.05, joint_smooth_weight=.2):
    """Original shared whole-body fitting objective, without admission gates."""
    import torch
    offsets=pose.new_tensor([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    def markers(q):
        position,rotation=fk.link_poses(q,LINKS)
        return position[...,None,:]+torch.einsum('...ij,kj->...ki',rotation,offsets)
    observed,predicted=markers(sources),markers(pose)
    weights=pose.new_ones(len(pose),len(LINKS))
    for part,name in enumerate(PART_LINKS):
        weights[:,LINKS.index(name)]+=3*masks.float().mean(0)[:,part]
    error=(((predicted[None]-observed)/.01).square().mean((-1,-2))*weights[None]).mean(-1)
    data=(2*(torch.sqrt(1+error)-1)).mean()
    prior=(2*(torch.sqrt(1+((pose[None,:,7:]-sources[:,:,7:])/.1).square().mean(-1))-1)).mean()
    smooth=(torch.diff(predicted,n=2,dim=0)/.001).square().mean()
    joint=(torch.diff(pose[:,7:],n=2,dim=0)/.01).square().mean()
    return data+.02*prior+marker_smooth_weight*smooth+joint_smooth_weight*joint


def fit_consensus(q, reference, boundaries, pairs, catalog, *, device="cpu", steps=600,
                  band=10, marker_smooth_weight=.05, joint_smooth_weight=.2,
                  mappings=None, contact_guard=None, continuity_guard=None, initial_pose=None,
                  representative_index=None, correction_time_scale=None, fps=50., learning_rate=.001,
                  correction_envelope=None, continuity_weight=10.,
                  convergence_patience=50, convergence_rtol=.001):
    """Retain one shared monotone clock; fit a continuous canonical trajectory."""
    import torch
    from .g1_kinematics import CanonicalG1ForwardKinematics
    q = np.asarray(q)
    if q.ndim != 3 or q.shape[2] != 36 or q.shape[0] < 3 or not np.isfinite(q).all():
        raise ValueError("At least three finite complete canonical executions are required")
    e, n, _ = q.shape
    if steps < 2 or n < 3:
        raise ValueError("Consensus needs at least two optimization steps and three frames")
    if (type(convergence_patience) is not int or convergence_patience < 1
            or not np.isfinite(convergence_rtol) or convergence_rtol <= 0):
        raise ValueError('Invalid consensus convergence settings')
    if mappings is not None:
        mappings = np.asarray(mappings)
        if (mappings.shape != (e, n) or not np.isfinite(mappings).all()
                or (mappings < 0).any() or (mappings >= n).any()
                or (np.diff(mappings, axis=1) < 0).any()
                or (mappings[:, 0] != 0).any() or (mappings[:, -1] != n-1).any()):
            raise ValueError("Invalid supplied whole-body event clocks")
        anchors = []
    elif boundaries is None:
        aligned_paths = [align_pose(reference, row, band=band) for row in q]
        mappings = [mapping for mapping, _ in aligned_paths]
        anchors = [[0, n-1] for _ in q]
    else:
        mappings, anchors = zip(*(align_event_anchors(reference, row, boundaries, band=band) for row in q))
    mappings = np.asarray(mappings)
    if np.issubdtype(mappings.dtype, np.integer):
        aligned = q[np.arange(e)[:, None], mappings]
    else:
        if contact_guard is None:
            raise ValueError('Continuous transport requires fresh native contact evidence')
        from .consensus_event_alignment import resample_motion
        aligned = np.asarray([resample_motion(row, mapping) for row, mapping in zip(q, mappings)])
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    fk = CanonicalG1ForwardKinematics().to(device)
    with torch.no_grad():
        p, r = fk.link_poses(tensor(aligned), LINKS)
    offsets = tensor([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
    def markers(position, rotation):
        return position[..., None, :] + torch.einsum("...ij,kj->...ki", rotation, offsets)
    observed = markers(p, r)
    center = observed.mean(0)
    medoid = int(((observed-center)**2).mean((1, 2, 3, 4)).argmin())
    if representative_index is not None:
        if not 0 <= representative_index < e:
            raise ValueError('Invalid contact representative index')
        medoid = int(representative_index)
    if contact_guard is not None:
        contact_guard.prepare(medoid, tensor(aligned[medoid]), fk)
    lower, upper = fk.joint_lower, fk.joint_upper
    mid, radius = (lower+upper)/2, (upper-lower)/2
    if initial_pose is not None and (np.shape(initial_pose) != (n, 36) or not np.isfinite(initial_pose).all()):
        raise ValueError("Invalid consensus warm start")
    endpoint_pose = tensor(aligned[medoid] if initial_pose is None else initial_pose).clone()
    initial = endpoint_pose.clone()
    initial[:, 7:] = torch.atanh(((initial[:, 7:]-mid)/radius).clamp(-.9999, .9999))
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid trajectory learning rate')
    if not np.isfinite(continuity_weight) or continuity_weight <= 0:
        raise ValueError('Invalid continuity optimization weight')
    envelope = None
    if correction_envelope is not None:
        a = np.asarray(correction_envelope)
        if (correction_time_scale is None or a.shape != (n,) or not np.isfinite(a).all()
                or np.any((a < 0) | (a > 1)) or a[0] != 0 or a[-1] != 0):
            raise ValueError('Invalid local spline correction envelope')
        envelope = tensor(a)
    basis = None
    if correction_time_scale is None:
        state = torch.nn.Parameter(initial)
        parameters = [state]
    else:
        from .continuous_motion import pose_correction_basis
        if not np.isfinite([correction_time_scale, fps]).all() or correction_time_scale <= 0 or fps <= 0:
            raise ValueError('Invalid correction time scale')
        stride = max(1, int(round(correction_time_scale*fps)))
        values, knots = pose_correction_basis(n,stride)
        basis = tensor(values)
        if envelope is not None:
            basis = basis*envelope[:,None]
        controls = torch.nn.Parameter(torch.zeros((basis.shape[1],36),dtype=initial.dtype,device=device))
        parameters = [controls]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    p_np, r_np = p.cpu().numpy(), r.cpu().numpy()
    groups = {}
    votes = np.zeros((e, n, 6), bool)
    for env in range(e) if contact_guard is None else []:
        for frame, original in enumerate(mappings[env]):
            for pair in pairs[env][int(original)]:
                if not pair["eligible"]:
                    continue
                key = (pair["part"], pair["shape"], pair["surface"])
                groups.setdefault(key, {}).setdefault((env, frame), []).append(pair["point"])
                votes[env, frame, pair["part"]] = True
    normals = {int(s["surface"]): np.asarray(s["normal_w"]) for s in (catalog or [])}
    local, targets, active, indices, normal_rows = [], [], [], [], []
    for (part, shape, surface), observations in sorted(groups.items()):
        idx = LINKS.index(PART_LINKS[part])
        mask = np.zeros((e, n), bool)
        cloud = []
        for (env, frame), points in observations.items():
            mask[env, frame] = True
            cloud.extend((np.asarray(points)-p_np[env, frame, idx]) @ r_np[env, frame, idx])
        on = mask.sum(0) > e/2
        if not on.any():
            continue
        fixed = np.median(cloud, axis=0)
        height = (p_np[:, :, idx] + np.einsum("etij,j->eti", r_np[:, :, idx], fixed)) @ normals[surface]
        target = np.zeros(n)
        for frame in np.flatnonzero(on):
            target[frame] = np.median(height[mask[:, frame], frame])
        local.append(fixed); targets.append(target); active.append(on)
        indices.append(idx); normal_rows.append(normals[surface])
    if not indices and contact_guard is None:
        raise ValueError("No majority contact evidence; do not silently disable contact retention")
    if contact_guard is None:
        local, target, active, normals_t = tensor(local), tensor(np.asarray(targets).T), tensor(np.asarray(active).T), tensor(normal_rows)
    else:
        votes = np.asarray(contact_guard.masks, bool)
    weights = torch.ones((n, len(LINKS)), device=device)
    for part, name in enumerate(PART_LINKS):
        weights[:, LINKS.index(name)] += 3*tensor(votes.mean(0)[:, part])
    source_joints = tensor(aligned[:, :, 7:])
    best_loss = float("inf")
    best_rank = (float("inf"),)*5
    def continuity_objective(pose, validation):
        if hasattr(continuity_guard, 'update'):
            return continuity_guard.objective(pose, fk, validation)
        return continuity_guard.objective(pose, fk)
    # Keep the exact incumbent, including poses at joint limits. Do not force
    # an inferior first optimizer step to replace an existing fitted aggregate.
    if initial_pose is not None and contact_guard is not None and continuity_guard is not None:
        incumbent = tensor(initial_pose)
        with torch.no_grad():
            retention, validation = contact_guard(-1, incumbent, fk, final=True)
            local_cost, count, excess = continuity_objective(incumbent, validation)
            incumbent_loss=float(population_fit_cost(incumbent,tensor(aligned),tensor(votes),fk,
                marker_smooth_weight=marker_smooth_weight,joint_smooth_weight=joint_smooth_weight)+retention+continuity_weight*local_cost)
        best_rank = (validation['failed_checks'], incumbent_loss, int(count > 0), count, excess)
        best_step, best_loss, result = -1, None, np.asarray(initial_pose).copy()
    history = []
    convergence = FitConvergence(convergence_patience, convergence_rtol)
    stop_reason = 'step_budget'
    for step in range(steps):
        optimizer.zero_grad()
        if basis is not None: state = initial+basis@controls
        pose = torch.cat((state[:, :3], torch.nn.functional.normalize(state[:, 3:7], dim=-1),
                          mid+radius*torch.tanh(state[:, 7:])), -1)
        if basis is not None:
            fixed_boundary = torch.zeros(n,dtype=torch.bool,device=device)
            fixed_boundary[[0,1,n-2,n-1]] = True
            pose = torch.where(fixed_boundary[:,None],endpoint_pose,pose)
            if envelope is not None:
                pose = torch.where((envelope > 0)[:,None], pose, endpoint_pose)
        position, rotation = fk.link_poses(pose, LINKS)
        predicted = markers(position, rotation)
        error = ((((predicted[None]-observed)/.01)**2).mean((-1, -2))*weights[None]).mean(-1)
        data = (2*(torch.sqrt(1+error)-1)).mean()
        prior = (2*(torch.sqrt(1+((pose[None, :, 7:]-source_joints)/.1).square().mean(-1))-1)).mean()
        smooth = ((predicted[2:]-2*predicted[1:-1]+predicted[:-2])/.001).square().mean()
        joint = ((pose[2:, 7:]-2*pose[1:-1, 7:]+pose[:-2, 7:])/.01).square().mean()
        validation = None
        if contact_guard is None:
            world = position[:, indices]+torch.einsum("tgij,gj->tgi", rotation[:, indices], local)
            height = (world*normals_t).sum(-1)
            retention = (((height-target)/.001).relu().square()*active).sum()/active.sum().clamp_min(1)
        else:
            retention, validation = contact_guard(step, pose, fk, final=step == steps-1)
        loss = data+.02*prior+marker_smooth_weight*smooth+joint_smooth_weight*joint+retention
        violations, excess = 0, 0.
        if continuity_guard is not None:
            local_cost, violations, excess = continuity_objective(pose, validation)
            loss = loss+continuity_weight*local_cost
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite consensus objective")
        value = float(loss.detach())
        rank = (0 if contact_guard is None else validation["failed_checks"],
                value, int(violations > 0), violations, excess) if validation is not None or contact_guard is None else None
        if rank is not None and rank < best_rank and (contact_guard is None or step > 0):
            best_rank = rank
            best_loss, best_step, result = value, step, pose.detach().cpu().numpy().copy()
        if step % 50 == 0 or step == steps-1:
            history.append(dict(step=step, loss=value, marker_acceleration=float(smooth.detach()),
                                joint_acceleration=float(joint.detach()), contact_retention=float(retention.detach()),
                                local_continuity_violations=violations, local_continuity_excess=excess))
            if contact_guard is not None:
                print(f"Consensus {step+1}/{steps}: loss={value:.5f}, "
                      f"failed={validation['failed_checks'] if validation else 'unqueried'}, local_violations={violations}, local_excess={excess:.5f}", flush=True)
        # Frozen geometry between native queries is only an optimization guide.
        # It cannot establish the successful-audit streak used for stopping.
        if validation is not None and continuity_guard is not None and convergence.update(
                step, value, validation['failed_checks'], violations):
            stop_reason = 'verified_convergence'
            print(f"Consensus converged at {step+1}/{steps}; fresh contact and continuity audits passed", flush=True)
            break
        loss.backward(); optimizer.step(); schedule.step()
    executed_steps = step+1
    report = dict(schema="event_aligned_smoothed_rollout_consensus_v1", steps=executed_steps, max_steps=steps,
        stop_reason=stop_reason, convergence_patience=convergence_patience, convergence_rtol=convergence_rtol,
        parameterization='per-frame pose' if basis is None else 'C2 boundary-constrained quintic correction of denoised signal',
        correction_time_scale=correction_time_scale, learning_rate=learning_rate,
        correction_control_count=None if basis is None else basis.shape[1],
        endpoint_sample_velocities_preserved=basis is not None,
        local_correction=correction_envelope is not None, continuity_weight=continuity_weight,
        endpoints_preserved=basis is not None,
        initialization_replica=medoid, best_step=best_step, best_loss=best_loss, history=history,
        initialization="aligned medoid" if initial_pose is None else "provided fitted pose; observations remain multi-execution",
        alignment="existing align_event_anchors; shared PCHIP monotone integer-frame clock",
        alignment_band_frames=band, boundaries=boundaries, source_anchor_frames=np.asarray(anchors).tolist(),
        marker_smooth_weight=marker_smooth_weight, joint_smooth_weight=joint_smooth_weight,
        rms_is_acceptance_condition=False,
        local_continuity_violations=best_rank[3] if continuity_guard is not None else None,
        selected_existing_aggregate=best_step == -1,
        candidate_priority=['actual contact failures', 'fit and smoothness objective', 'continuity diagnostics'],
        release_supervision="disabled without independent action evidence",
        contact_retention="majority shape/primary-face source geometry; guide only, not new contact truth",
        source_sampling="recorded solve points and post-integration FK are distinct; geometric fitting guide",
        training_ready=False, dynamics="new fitted kinematics; not an unchanged physical execution")
    if contact_guard is not None:
        report.update(schema="native_guarded_event_consensus_v4",
            optimizer='Adam; trajectory variables only, no policy training',
            objective_components=['robust FK markers', 'robust joint prior', 'marker acceleration',
                                  'joint acceleration', 'native contact retention', 'local continuity'],
            alignment="supplied observed-event shared monotone clocks",
            contact_retention="fresh Newton primary-face geometry witnesses; interval targets are optimization intent",
            source_sampling="fixed_q_forward_no_integration, before and after integer event alignment",
            validated_failed_checks=int(best_rank[0]), native_feedback=contact_guard.history,
            release_supervision="explicit matched release events only")
    return result, mappings, report


def aggregate_collection(plan, root, *, device="cuda:0"):
    # Authored event tasks use observed-event fusion and native feedback. The
    # pose-only legacy adapter remains available for completion-only inputs.
    kinds = {c['task']['kind'] for c in plan['cases']}
    if len(kinds) != 1:
        raise ValueError('Split contact and legacy completion tasks into separate collections')
    if kinds == {'contact_events'}:
        from .rollout_reaggregation import reaggregate
        output = root/'event_consensus'
        summary = reaggregate(plan, root, output, device=device)
        manifest = workflow.read_json(output/'dataset_manifest.json')
        workflow.write_json(root/'dataset_manifest.json', dict(manifest, recording=str(root/'recording.npz.json'),
            collection_plan=workflow.fingerprint(root/'collection_plan.json')), immutable=True)
        workflow.write_json(root/'collection_summary.json', dict(summary, stage='complete',
            total_references=summary['total'], processed_references=summary['processed'],
            aggregate_count=sum(r.get('aggregation_written', False) for r in summary['motions']),
            complete_rollout_count=sum(r['complete_rollouts'] for r in summary['motions']),
            preliminary_evaluation=False, training_ready=False))
        return
    from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk
    from .robot_assets import encode_robot_asset_json
    recording = workflow.read_json(root/"recording.npz.json")
    meta = recording["metadata"]
    shapes = [re.sub(r"/env_\d+/", "/env_*/", label) for label in meta["newton_shape_labels"]]
    rows = []
    for case in plan["cases"]:
        destination = root/"aggregated"/case["key"]
        completed = destination/"aggregate.json"
        if completed.exists():
            saved = workflow.read_json(completed); workflow.verify_files(saved["artifacts"])
            rows.append(saved["summary"]); continue
        candidates = load_candidates(root/"candidates"/case["key"])
        good, native, failed = [], [], []
        for column, env in enumerate(candidates["env_ids"]):
            indices = complete_segment(candidates["motion_time_step"][:, column]-candidates["motion_start_step"][:, column],
                candidates["terminated"][:, column], candidates["timeout"][:, column], case["frames"])
            done = meta["env_done_steps"][int(env)]
            if indices is None or (done is not None and done <= int(indices[-1]) and not meta["env_done_is_timeout"][int(env)]):
                failed.append(int(env)); continue
            good.append(column); native.append(indices)
        row = dict(key=case["key"], identity=case["identity"], complete_rollouts=len(good),
                   failed_env_ids=failed, source_env_ids=candidates["env_ids"][good].tolist())
        if len(good) < 3:
            row.update(status="insufficient_complete_rollouts", aggregation_written=False)
            rows.append(row)
        else:
            qs, pairs = [], []
            for column, indices in zip(good, native):
                qs.append(np.concatenate((candidates["root_pos"][indices, column],
                    candidates["root_quat_xyzw"][indices, column][:, [3, 0, 1, 2]],
                    candidates["dof_pos"][indices, column]), -1))
                frame_map = {int(original): frame for frame, original in enumerate(indices)}
                sequence = [[] for _ in indices]
                env = int(candidates["env_ids"][column])
                for idx in np.flatnonzero(candidates["pair_env"] == env):
                    f = frame_map.get(int(candidates["pair_frame"][idx]))
                    if f is None: continue
                    sequence[f].append(dict(part=int(candidates["pair_part"][idx]),
                        surface=int(candidates["pair_surface"][idx]), eligible=bool(candidates["pair_eligible"][idx]),
                        point=candidates["pair_point"][idx].tolist(), shape=shapes[int(candidates["pair_shape"][idx])]))
                pairs.append(sequence)
            with np.load(case["reference"]["motion_file"], allow_pickle=False) as z:
                reference = z["joint_pos"].copy()
            q, maps, report = fit_consensus(np.asarray(qs), reference, event_boundaries(case), pairs,
                meta["newton_scene_binding"]["surface_catalog"], device=device, **plan["config"].get("aggregation", {}))
            save_arrays(destination/"seed.npz", dict(joint_pos=q, joint_names=np.asarray(meta["dof_names"]),
                fps=np.asarray(case["fps"]), robot_asset_json=np.asarray(encode_robot_asset_json(meta["robot_asset"]))))
            save_arrays(destination/"alignment.npz", dict(source_frame_maps=maps,
                native_record_frames=np.asarray(native), source_env_ids=np.asarray(row["source_env_ids"])))
            workflow.write_json(destination/"optimization.json", report, immutable=True)
            motion = destination/"motion.npz"
            if not motion.exists():
                canonicalize_motion_with_direct_newton_fk(destination/"seed.npz", motion, device="cpu")
            row.update(status="aggregated_pending_newton_labels", aggregation_written=True, motion=str(motion))
            workflow.write_json(completed, dict(summary=row, artifacts=[workflow.fingerprint(destination/name)
                for name in ("seed.npz", "alignment.npz", "optimization.json", "motion.npz")]), immutable=True)
            rows.append(row)
        workflow.write_json(root/"collection_summary.json", dict(motions=rows, total_references=len(plan["cases"]),
            processed_references=len(rows), aggregate_count=sum(r["aggregation_written"] for r in rows),
            complete_rollout_count=sum(r["complete_rollouts"] for r in rows), preliminary_evaluation=False,
            stage="aggregation", training_ready=False))
        print(f"Aggregated {len(rows)}/{len(plan['cases'])}: {case['identity']}, complete={len(good)}", flush=True)
    relabel_collection(plan, root, rows, device=device)


def relabel_collection(plan, root, rows, *, device):
    """Reuse the current native Newton query entrypoint and its batch relabeling."""
    from .newton_contact_data import load_contact_labels
    by_terrain = {}
    for case, row in zip(plan["cases"], rows):
        if row["aggregation_written"]:
            by_terrain.setdefault(case["terrain"]["terrain_id"], []).append((case, row))
    inference = workflow.read_json(root/"policy"/"inference_policy.json")["inference_checkpoint"]
    for terrain_id, members in by_terrain.items():
        directory = root/"labeling"/f"terrain_{terrain_id:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        case = members[0][0]
        source_manifest = workflow.read_json(Path(plan["output_dir"])/"references"/f"{case['key']}.json")
        manifest = directory/"manifest.json"
        workflow.write_json(manifest, source_manifest, immutable=True)
        pending = []
        for case, row in members:
            motion = Path(row["motion"]); labels = motion.with_name("contacts.npz")
            if labels.exists():
                load_contact_labels(motion, labels)
                row["status"] = "labeled"; row["contacts"] = str(labels)
            else:
                pending.append(dict(motion=str(motion), labels_output=str(labels)))
        if not pending: continue
        inspection, binding = directory/"model.json", directory/"binding.json"
        base = [sys.executable, "scripts/serve_newton_contact_queries.py", "--checkpoint", inference["path"],
                "--motion-manifest", str(manifest), "--inspection-output", str(inspection),
                "--binding", str(binding), "--create-native-binding", "--headless", "--device", device]
        if not inspection.exists() or not binding.exists():
            workflow.run_command(base+["--inspect-only"], directory/"inspect"/uuid.uuid4().hex)
        jobs = directory/f"jobs_{uuid.uuid4().hex}.json"
        workflow.write_json(jobs, dict(model_fingerprint=workflow.read_json(inspection)["model_fingerprint"], jobs=pending), immutable=True)
        reuse_binding = [item for item in base if item != "--create-native-binding"]
        workflow.run_command(reuse_binding+["--relabel-jobs", str(jobs)], directory/"relabel"/uuid.uuid4().hex)
        for case, row in members:
            labels = Path(row["motion"]).with_name("contacts.npz")
            load_contact_labels(row["motion"], labels)
            row.update(status="labeled", contacts=str(labels))
    workflow.write_json(root/"collection_summary.json", dict(motions=rows, total_references=len(plan["cases"]),
        processed_references=len(rows), aggregate_count=sum(r["aggregation_written"] for r in rows),
        complete_rollout_count=sum(r["complete_rollouts"] for r in rows), preliminary_evaluation=False,
        stage="complete", training_ready=False,
        scope="aligned smoothed kinematic aggregates with fresh Newton labels; original dynamics retained separately"))
    export_manifest(plan, root, rows)


def export_manifest(plan, root, rows):
    from . import extraction_standard as standard
    kinds = {c['task']['kind'] for c in plan['cases']}
    if kinds == {'contact_events'}:
        return standard.verified_manifest(Path(root)/'dataset_manifest.json')
    if kinds != {'completion_only'}:
        raise ValueError('Split contact and legacy completion tasks into separate collections')
    from .newton_contact_data import load_contact_labels
    entries = []
    for case, row in zip(plan["cases"], rows):
        if row["status"] != "labeled":
            continue
        labels = load_contact_labels(row["motion"], row["contacts"])
        with np.load(row["contacts"], allow_pickle=False) as z:
            catalog = json.loads(z["contact_semantics_json"].item())["scene"]["surface_catalog"]
        from .rollout_dataset import task_check
        task = task_check(case, labels, catalog)
        entries.append(dict(case["reference"], motion_id=len(entries),
            motion_file=row["motion"], motion_sha256=workflow.sha256_file(row["motion"]),
            contacts=row["contacts"], contacts_sha256=workflow.sha256_file(row["contacts"]),
            contact_catalog_file=row["contacts"], source_env_ids=row["source_env_ids"],
            complete_rollouts=row["complete_rollouts"], task_acceptance=task,
            reference_motion_file=case["reference"]["motion_file"],
            training_ready=False, execution_kind="aligned_smoothed_aggregate; fixed_q_Newton_labels"))
    manifest = dict(workflow.manifest_header("newton_direct_fk"), motion_files=entries,
        terrains=plan["manifest"]["terrains"], recording=str(root/"recording.npz.json"),
        collection_plan=workflow.fingerprint(root/"collection_plan.json"),
        training_ready=False, extraction_status="legacy_unverified", source_executions="retained in original lossless recording chunks")
    workflow.write_json(root/"dataset_manifest.json", manifest, immutable=True)
    return manifest
