"""Line-searched refinement of an existing aggregate with fresh native audits."""
import numpy as np
import torch


def refine_existing(aligned, initial_pose, contact_guard, continuity_guard, *, device='cpu',
                    steps=600, marker_smooth_weight=.05, joint_smooth_weight=.2):
    from .g1_kinematics import CanonicalG1ForwardKinematics
    from .rollout_consensus import LINKS, PART_LINKS
    fk = CanonicalG1ForwardKinematics().to(device)
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    source, incumbent = tensor(aligned), tensor(initial_pose)
    offsets = tensor([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
    def markers(q):
        p, r = fk.link_poses(q, LINKS)
        return p[..., None, :]+torch.einsum('...ij,kj->...ki', r, offsets)
    with torch.no_grad():
        observations, original_markers = markers(source), markers(incumbent)
        medoid = int((observations-observations.mean(0)).square().mean((1, 2, 3, 4)).argmin())
    contact_guard.prepare(medoid, source[medoid], fk)
    weights = torch.ones((len(incumbent), len(LINKS)), device=device)
    for part, name in enumerate(PART_LINKS):
        weights[:, LINKS.index(name)] += 3*tensor(np.asarray(contact_guard.masks).mean(0)[:, part])
    state = torch.nn.Parameter(incumbent.clone())
    def pose():
        return torch.cat((state[:, :3], torch.nn.functional.normalize(state[:, 3:7], dim=-1),
                          state[:, 7:].clamp(fk.joint_lower, fk.joint_upper)), -1)
    def objective(q):
        predicted = markers(q)
        error = ((((predicted[None]-observations)/.01).square().mean((-1, -2)))*weights[None]).mean(-1)
        data = (2*(torch.sqrt(1+error)-1)).mean()
        prior = (2*(torch.sqrt(1+((q[None, :, 7:]-source[:, :, 7:])/.1).square().mean(-1))-1)).mean()
        smooth = (torch.diff(predicted, n=2, dim=0)/.001).square().mean()
        joint = (torch.diff(q[:, 7:], n=2, dim=0)/.01).square().mean()
        contact, _ = contact_guard(1, q, fk)
        local, count, excess = continuity_guard.objective(q, fk)
        # The incumbent already contains multi-rollout denoising. Preserve its
        # shape while correcting constraints, rather than pulling all frames
        # toward a newly shifted population average again.
        stay = ((predicted-original_markers)/.01).square().mean()
        outside = ((state[:, 7:]-q[:, 7:])/.001).square().mean()
        loss = data+.02*prior+marker_smooth_weight*smooth+joint_smooth_weight*joint+contact+10*local+stay+outside
        if not torch.isfinite(loss): raise ValueError('Nonfinite line-search objective')
        return loss, count, excess
    best_rank = (float('inf'),)*5
    history = []
    # Refresh witnesses and actual event phases between short line-search
    # solves. Every closure inside a solve uses the SAME frozen objective.
    for step in range(0, steps+1, 10):
        q = incumbent if step == 0 else pose()
        with torch.no_grad():
            _, validation = contact_guard(step, q, fk, final=True)
            continuity_guard.update(validation)
            loss, count, excess = objective(q)
        rank = (validation['failed_checks'], int(count > 0), excess if count else 0., count, float(loss))
        if rank < best_rank:
            best_rank, best_step, result = rank, step, q.detach().cpu().numpy().copy()
        history.append(dict(step=step, loss=float(loss), failed_checks=validation['failed_checks'],
                            local_continuity_violations=count, local_continuity_excess=excess))
        if step % 50 == 0 or step >= steps:
            print(f'Refine {step}/{steps}: loss={float(loss):.5f}, failed={validation["failed_checks"]}, '
                  f'local_violations={count}, local_excess={excess:.5f}', flush=True)
        if step >= steps: break
        optimizer = torch.optim.LBFGS([state], lr=1., max_iter=min(10, steps-step),
            max_eval=25, history_size=10, line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)
        def closure():
            optimizer.zero_grad()
            loss, _, _ = objective(pose())
            loss.backward()
            return loss
        optimizer.step(closure)
    return result, dict(schema='native_guarded_event_consensus_v5', optimizer='LBFGS strong-Wolfe; fresh witnesses per 10 iterations',
        steps=steps, initialization_replica=medoid, best_step=best_step, best_loss=best_rank[-1], history=history,
        validated_failed_checks=int(best_rank[0]), local_continuity_violations=int(best_rank[3]),
        selected_existing_aggregate=best_step == 0, native_feedback=contact_guard.history,
        candidate_priority=['actual contact failures', 'continuity passed', 'worst local excess', 'violation count', 'fit loss'],
        marker_smooth_weight=marker_smooth_weight, joint_smooth_weight=joint_smooth_weight,
        rms_is_acceptance_condition=False, incumbent_marker_weight=1.,
        source_sampling='native and continuously resampled poses separately queried with Newton',
        dynamics='fitted kinematics, not a certified dynamic execution', training_ready=False)
