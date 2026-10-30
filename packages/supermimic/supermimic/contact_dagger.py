"""PHP-style pure DAgger warm start using the task-matched tracking expert.

Port the rollout/valid-teacher/action-MSE contract, not Holosoma's PPO or
critic. Teacher histories contain actual student states, never replay states.
The student's spatial command stays fixed within each sampled contact segment.
"""
from __future__ import annotations

import copy
import json
import torch

from .contact_control import observation, BCPlateau
from .mdp import unflatten_state, flatten_state
from .rotations import mat_to_quat
from .contact_io import load_checkpoint_cpu


class BalancedReplay:
    """Vectorized equal-case sampling; build the packed view once per round."""
    def __init__(self, groups, device=None):
        keys=sorted(groups)
        if not keys or any(not len(groups[k]['action']) for k in keys):raise ValueError('Empty case replay')
        storage_device=groups[keys[0]]['action'].device
        self.output_device=storage_device if device is None else device
        self.lengths=torch.tensor([len(groups[k]['action']) for k in keys],device=storage_device)
        self.offsets=self.lengths.cumsum(0)-self.lengths
        self.data={field:torch.cat([groups[k][field] for k in keys]) for field in groups[keys[0]]}

    def sample(self,batch):
        cases=torch.randint(len(self.lengths),(batch,),device=self.lengths.device)
        index=self.offsets[cases]+(torch.rand(batch,device=self.lengths.device)*self.lengths[cases]).long()
        return {key:value[index].to(self.output_device) for key,value in self.data.items()}


def _load_replay_groups(paths):
    """Concatenate each field once, preserving caller-supplied rollout order."""
    restored = {}
    for path in paths:
        saved = load_checkpoint_cpu(path)
        for i, row in saved['data'].items():
            chunks = restored.setdefault(i, {k: [] for k in ('history', 'condition', 'action')})
            for key in chunks:chunks[key].append(row[key])
    return {i: {key: torch.cat(chunks) if len(chunks)>1 else chunks[0]
                for key,chunks in row.items()} for i,row in restored.items()}


def balanced_indices(groups, batch):
    """Equal segment probability, independent of segment length/survival."""
    keys = sorted(groups)
    if not keys or any(not len(groups[k]['action']) for k in keys):
        raise ValueError('DAgger requires nonempty segment buffers')
    device = groups[keys[0]]['action'].device
    segments = torch.randint(len(keys), (batch,), device=device)
    result = []
    for i, key in enumerate(keys):
        count = int((segments == i).sum())
        if count:
            index = torch.randint(len(groups[key]['action']), (count,), device=device)
            result.append({k: v[index] for k, v in groups[key].items()})
    return {k: torch.cat([r[k] for r in result]) for k in result[0]}


def run_dagger(*, args, env, actor, optimizer, tracker, motion, mdp,
               clock, demo, context, default_q, chosen, teacher_hist,
               goal_frames, status, stopped):
    """Native student rollouts for every segment, equal regression batches.

All rounds are aggregated and saved; no teacher action is executed in a live
student segment. Tracking termination bounds teacher validity. Segment length
is a collection budget and teacher phase only, never a student time input.
"""
    count = len(clock.event_ends)
    rollouts_per_segment = int(getattr(args, 'dagger_rollouts_per_segment', 1))
    action_noise = float(getattr(args, 'dagger_action_noise', 0.))
    if rollouts_per_segment < 1 or action_noise < 0:
        raise ValueError('Invalid DAgger transition collection budget')
    device = chosen.device
    from .contact_library import segment_ranges,history_indices
    all_starts,all_stops = segment_ranges(clock)
    all_ends = clock.event_ends
    all_ids = torch.arange(env.N, device=device)
    output = args.output / 'dagger'
    output.mkdir()
    groups = {}
    old_rates = [g['lr'] for g in optimizer.param_groups]
    for group in optimizer.param_groups:
        group['lr'] = args.bc_learning_rate
    plateau = BCPlateau(args.dagger_min_rounds, args.bc_patience,
                        args.bc_min_improvement, args.bc_relative_improvement)
    best = None
    best_score = float('inf')
    updates = 0
    start_round = 0
    if args.dagger_resume:
        previous = load_checkpoint_cpu(args.dagger_resume)
        parent_config = json.loads((args.dagger_resume.parent.parent/'config.json').read_text())
        if parent_config.get('collision_profile','legacy_v1') != getattr(args,'collision_profile','legacy_v1'):
            raise ValueError('DAgger resume collision profile must match')
        if str(getattr(args,'slices',None)) != str(parent_config.get('slices')):
            raise ValueError('DAgger resume slice selection must match')
        if any(str(getattr(args, key)) != str(parent_config[key])
               for key in ('checkpoint', 'recording', 'task_manifest')):
            raise ValueError('DAgger resume teacher/task/recording must match')
        actor.load_state_dict(previous['actor'])
        optimizer.load_state_dict(previous['optimizer'])
        updates = previous['updates']
        start_round = previous['round']
        best=copy.deepcopy(previous);best_score=float(previous['report']['score'])
        torch.save(best,output/'best.pt')
        groups = _load_replay_groups(sorted(args.dagger_resume.parent.glob('rollout_*.pt')))
        del previous
    stop_reason = 'max_rounds'

    @torch.no_grad()
    def collect_batch(segment_ids, selected_ids=None):
        count = len(segment_ids)
        ids = chosen[:count] if selected_ids is None else selected_ids
        starts, ends = all_starts[segment_ids], all_ends[segment_ids]
        durations = all_stops[segment_ids] - starts
        env.reset(all_ids, motion.clip_start[env.task_ids])
        env.reset(ids, demo['f'][starts].long())
        # Restore the actual tracker rollout state, including actuator offsets
        # and previous action. No teacher warm-up or action mixing.
        p, R, v, w, q, qd = unflatten_state(demo['s'][starts])
        quat = mat_to_quat(R)
        pose = torch.cat((p + env.scene.env_origins[ids], quat[:, 1:], quat[:, :1]), -1)
        env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=ids)
        env.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        env.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
        env.robot.set_joint_position_target_index(target=q, env_ids=ids)
        env.a_prev[ids] = demo['a_prev'][starts]
        env.xi[ids] = demo['xi'][starts]
        env.ep_step[ids] = demo['ep_step'][starts].to(env.ep_step.dtype)
        env.done_eval.zero_(); env.failed_eval.zero_(); env.refresh()
        active = torch.ones(count, dtype=torch.bool, device=device)
        state=env.state();env.xi[ids]=demo['xi'][starts]
        initial_obs=mdp.obs(state,env.a_prev,env.xi,env.f)
        th=initial_obs[:,None].expand(-1,teacher_hist,-1).clone()
        teacher_indices=history_indices(clock,starts,teacher_hist)
        th[ids]=mdp.obs(unflatten_state(demo['s'][teacher_indices].flatten(0,1)),
            demo['a_prev'][teacher_indices].flatten(0,1),demo['xi'][teacher_indices].flatten(0,1),
            demo['f'][teacher_indices].flatten()).reshape(count,teacher_hist,-1)
        actor_indices=history_indices(clock,starts,actor.hist)
        ah=observation(unflatten_state(demo['s'][actor_indices].flatten(0,1)),
            demo['a_prev'][actor_indices].flatten(0,1),demo['xi'][actor_indices].flatten(0,1),
            default_q,context).reshape(count,actor.hist,-1)
        rows=[]
        steps = torch.zeros(count, device=device)
        root_error = torch.zeros(count, device=device)
        action_error = torch.zeros(count, device=device)
        restart_observation_error = 0.
        for step in range(int(durations.max())):
            S = env.state()
            if step==0:env.xi[ids]=demo['xi'][starts]
            obs = mdp.obs(S, env.a_prev, env.xi, env.f)
            if step==0:
                expected=mdp.obs(unflatten_state(demo['s'][starts]),demo['a_prev'][starts],demo['xi'][starts],demo['f'][starts])
                restart_observation_error=float((obs[ids]-expected).abs().max())
                if restart_observation_error>1e-3:raise ValueError('DAgger reset failed to restore recorded observation')
            th = obs[:, None].expand(-1, teacher_hist, -1).clone() if th is None else torch.cat((th[:, 1:], obs[:, None]), 1)
            # Both networks see the same pre-action student state/history.
            labels = tracker(th, motion.goal(env.f, goal_frames))
            current = tuple(x[ids] for x in S)
            student_obs = observation(current, env.a_prev[ids], env.xi[ids], default_q, context)
            ah = student_obs[:, None].expand(-1, actor.hist, -1).clone() if ah is None else torch.cat((ah[:, 1:], student_obs[:, None]), 1)
            condition = clock.condition(current, starts, ends=ends)
            student_mean = actor(ah, condition)
            student = (student_mean+action_noise*torch.randn_like(student_mean)).clamp(-20,20) if action_noise else student_mean
            valid = (~mdp.termination(S, env.f)[ids]
                     & torch.isfinite(student).all(-1) & torch.isfinite(labels[ids]).all(-1))
            active &= valid
            row=dict(history=ah.clone(),condition=condition.clone(),action=labels[ids].clone(),
                student_action=student.clone(),state=flatten_state(current).clone(),frame=env.f[ids].clone(),
                episode_step=env.ep_step[ids].clone(),teacher_history=th[ids].clone(),active=active.clone())
            action_error += ((student_mean-labels[ids]).square().mean(-1) * active)
            actions = labels.clone()
            actions[ids[active]] = student[active]
            out = env.step(actions)
            row.update(next_state=out['s1'][ids].clone(),terminated=out['term'][ids].clone());rows.append(row)
            after = unflatten_state(out['s1'][ids])
            reference = motion.ref_state(env.f)
            error = (after[0] - reference[0][ids]).norm(dim=-1)
            root_error = torch.where(active, error, root_error)
            steps += active
            active &= ~out['term'][ids] & (step + 1 < durations)
            if not bool(active.any()) or stopped():
                break
            # Background worlds must not run beyond their original clips.
            reset = out['done'].clone()
            reset[ids[active]] = False
            reset_ids = reset.nonzero().flatten()
            if len(reset_ids):
                env.reset(reset_ids, motion.clip_start[env.task_ids[reset_ids]])
                env.refresh()
                fresh = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                th[reset_ids] = fresh[reset_ids, None].expand(-1, teacher_hist, -1)
            env.f = torch.minimum(env.f, motion.clip_end[env.task_ids]-1)
        stacked={key:torch.stack([row[key] for row in rows]) for key in rows[0]}
        mask=stacked.pop('active')
        data={i:{key:value[mask[:,i],i] for key,value in stacked.items()}
              for i in range(count) if bool(mask[:,i].any())}
        report = dict(segments=count, covered_segments=len(data),
                      restart_observation_error=restart_observation_error,
                      survival_fraction=(steps / durations).tolist(),
                      root_tracking_error_m=root_error.tolist(),
                      student_state_action_mse=(action_error/steps.clamp_min(1)).tolist(),
                      teacher_actions_in_student_segments=False)
        # Fresh closed-loop imitation score, equal weight for every segment.
        # Includes root tracking and survival, not just in-sample action fit.
        score = float((action_error/steps.clamp_min(1)).mean().sqrt()
                      + 2*(1-steps/durations).mean() + root_error.mean())
        return {int(segment_ids[i]): row for i, row in data.items()}, report, score

    def collect():
        data = {}
        report = dict(segments=count, covered_segments=0, survival_fraction=[],
                      root_tracking_error_m=[], student_state_action_mse=[],
                      max_restart_observation_error=0.,
                      teacher_actions_in_student_segments=False)
        for repeat in range(rollouts_per_segment):
            for offset in range(0, count, len(chosen)):
                segment_ids = torch.arange(offset, min(offset+len(chosen), count), device=device)
                selected = chosen[(torch.arange(len(segment_ids),device=device)+repeat*count+offset)%len(chosen)]
                if hasattr(clock,'segment_motion'):
                    from .contact_library import dagger_case_worlds
                    selected=dagger_case_worlds(clock,env.task_ids,chosen,segment_ids,repeat)
                batch, batch_report, _ = collect_batch(segment_ids,selected)
                report['max_restart_observation_error']=max(report['max_restart_observation_error'],batch_report['restart_observation_error'])
                for key,row in batch.items():
                    data[key]=({name:torch.cat((data[key][name],value)) for name,value in row.items()}
                               if key in data else row)
                for key in ('survival_fraction', 'root_tracking_error_m', 'student_state_action_mse'):
                    report[key].extend(batch_report[key])
                if stopped():break
            if stopped():break
        report['covered_segments'] = len(data)
        report['rollouts_per_segment'] = rollouts_per_segment
        report['student_action_noise_std'] = action_noise
        report['native_student_transitions'] = sum(len(row['action']) for row in data.values())
        mean = lambda key: sum(report[key])/len(report[key])
        score = mean('student_state_action_mse')
        return data, report, score

    try:
        for round_id in range(start_round, args.dagger_rounds + 1):
            # Warp owns its allocations independently of PyTorch's cache.
            # Release dead rollout/optimizer buffers before the next native solve.
            if torch.cuda.is_available():torch.cuda.empty_cache()
            fresh, report, score = collect()
            report.update(round=round_id, updates=updates, score=score, score_metric='equal_segment_action_mse')
            if torch.cuda.is_available():
                report.update(torch_allocated_gib=torch.cuda.memory_allocated()/2**30,
                              torch_reserved_gib=torch.cuda.memory_reserved()/2**30,replay_storage='cpu')
            # One owned CPU snapshot serves both the archive and replay. Copying
            # the same history/action tensors again costs an extra GPU readback.
            fresh_cpu = {i: {k: v.detach().cpu() for k,v in row.items()} for i,row in fresh.items()}
            torch.save(dict(data=fresh_cpu,report=report), output/f'rollout_{round_id:04d}.pt')
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(report)+'\n')
            print(json.dumps(dict(phase='dagger',round=round_id,updates=updates,score=score,
                covered_segments=report['covered_segments'],native_student_transitions=report['native_student_transitions'])),flush=True)
            status(phase='dagger', dagger=report)
            if score < best_score:
                best_score = score
                best = dict(actor=copy.deepcopy(actor.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                            round=round_id, updates=updates, report=report)
                torch.save(best, output/'best.pt')
            plateaued = plateau.update(round_id, score)
            if stopped() or plateaued or round_id == args.dagger_rounds:
                stop_reason = 'interrupted' if stopped() else 'plateau' if plateaued else 'max_rounds'
                break
            if len(fresh) != count:
                raise RuntimeError('Some segments have no valid teacher labels; inspect per-segment resets')
            for i, row in fresh_cpu.items():
                row = {k: row[k] for k in ('history', 'condition', 'action')}
                groups[i] = ({k: torch.cat((groups[i][k], v)) for k, v in row.items()}
                             if i in groups else row)
            del fresh,fresh_cpu
            replay=BalancedReplay(groups,device=device)
            for _ in range(args.dagger_updates_per_round):
                batch = replay.sample(args.bc_batch)
                # PHP pure phase: action MSE only; no PPO, entropy, critic, WM.
                loss = 10 * (actor(batch['history'], batch['condition'])-batch['action']).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
            del replay,batch
        actor.load_state_dict(best['actor'])
        optimizer.load_state_dict(best['optimizer'])
        result = dict(stop_reason=stop_reason, rounds=round_id, updates=updates,
                      best_round=best['round'], best_updates=best['updates'], best_report=best['report'],
                      aggregate_samples={i: len(row['action']) for i, row in groups.items()},
                      fresh_native_student_transitions=sum(json.loads(line).get('native_student_transitions',0)
                          for line in (output/'metrics.jsonl').read_text().splitlines()),
                      replay_files='rollout_*.pt', teacher_checkpoint=str(args.checkpoint),
                      parent_checkpoint=str(args.dagger_resume) if args.dagger_resume else None,
                      plateau_metric='fresh equal-segment action MSE only')
        (output/'summary.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        for group, rate in zip(optimizer.param_groups, old_rates):
            group['lr'] = rate
