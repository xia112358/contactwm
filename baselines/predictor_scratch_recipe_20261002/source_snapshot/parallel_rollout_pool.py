"""Persistent endpoint rollouts: commit accepted predictions, retain inputs on failure."""
import torch


def rollout_transition(indices, successor, supervised, completed, continuable):
    """Keep scene identity after demonstration exhaustion; never fabricate labels."""
    next_row = successor[indices]
    advance = supervised & completed & (next_row >= 0) & continuable
    exhausted = supervised & completed & (next_row < 0) & continuable
    retry = supervised & ~completed & continuable
    return (torch.where(advance, next_row, indices), supervised & ~exhausted,
            advance, retry, exhausted)


class ParallelRolloutPool:
    def __init__(self, reference, training_indices, successor, size, *,
                 recover_failed_states=False, max_episode_steps=64, max_retries=8,
                 allow_post_demo=True,
                 global_directions=None, progress_window=3, progress_minimum_m=.03615079075098038):
        if recover_failed_states:
            raise ValueError('Legacy recovery flag is obsolete; failed predictions are rolled back to their input')
        if size < 1 or max_episode_steps < 1 or max_retries < 1:
            raise ValueError('Pool size, resampling interval and logging limits must be positive')
        if not len(training_indices): raise ValueError('Empty training pool')
        self.reference, self.training_indices, self.size = reference, training_indices, size
        allowed = torch.zeros(len(successor), dtype=torch.bool, device=successor.device)
        allowed[training_indices] = True
        valid = (successor >= 0) & allowed[successor.clamp_min(0)]
        self.successor = torch.where(valid, successor, -1)
        self.indices = self.draw_references(size)
        self.observation = {k:v[self.indices].detach().clone() for k,v in reference.items()}
        self.age = torch.zeros_like(self.indices)
        # Count every prediction, independently of successful chain depth.
        self.attempts = torch.zeros_like(self.indices)
        self.resample_interval = max_episode_steps
        self.allow_post_demo = allow_post_demo
        self.retries = torch.zeros_like(self.indices)
        self.supervised = torch.ones_like(self.indices, dtype=torch.bool)
        self.cursor = 0
        self.progress_directions = None
        self.progress_window, self.progress_minimum = progress_window, progress_minimum_m
        if global_directions is not None:
            if progress_window < 1 or not __import__('math').isfinite(progress_minimum_m) or progress_minimum_m <= 0:
                raise ValueError('Invalid progress gate settings')
            norms = global_directions.norm(dim=-1, keepdim=True)
            if global_directions.shape != (len(successor),2) or not bool(torch.isfinite(norms).all() & (norms > 1e-8).all()):
                raise ValueError('Every sample requires a finite nonzero global task direction')
            self.progress_directions = global_directions / norms
            initial = (self.observation['current_q'][:,:2]*self.progress_directions[self.indices]).sum(-1)
            self.progress_history = initial[:,None].repeat(1,progress_window+1)

    def draw_references(self, count):
        return self.training_indices[torch.randint(len(self.training_indices), (count,), device=self.training_indices.device)]

    def batch(self, count):
        if not 1 <= count <= self.size: raise ValueError('Invalid parallel batch size')
        slots = (torch.arange(count, device=self.indices.device)+self.cursor)%self.size
        self.cursor = (self.cursor+count)%self.size
        return slots, self.indices[slots], {k:v[slots] for k,v in self.observation.items()}

    @torch.no_grad()
    def update(self, slots, q, observed, completed, continuable, make_observation, *, reset_reasons=None):
        indices = self.indices[slots]
        no_progress = torch.zeros_like(continuable)
        if self.progress_directions is not None:
            projection = (q[:,:2]*self.progress_directions[indices]).sum(-1)
            frontier = torch.maximum(self.progress_history[slots,-1], projection)
            history = torch.cat((self.progress_history[slots,1:],frontier[:,None]),dim=1)
            no_progress = continuable & (self.age[slots]+1 >= self.progress_window) & (history[:,-1]-history[:,0] < self.progress_minimum)
            continuable = continuable & ~no_progress
            if reset_reasons is not None:
                reset_reasons['no_forward_progress'] = no_progress
        next_indices, supervised, advance, retry, exhausted = rollout_transition(
            indices, self.successor, self.supervised[slots], completed, continuable)
        accepted = completed & continuable
        retry = ~accepted
        chosen = accepted.nonzero().flatten()
        # A rejected B' never replaces A (which may itself be generated).
        # Loss/backprop is handled by the caller before this detached commit.
        if len(chosen):
            fresh = make_observation(q[chosen].detach(), next_indices[chosen], observed, chosen)
            for key, storage in self.observation.items():
                storage[slots[chosen]] = fresh[key].detach()
            if self.progress_directions is not None:
                self.progress_history[slots[chosen]] = history[chosen]
        self.indices[slots] = next_indices
        self.age[slots] += accepted.long()
        self.retries[slots] = torch.where(retry, self.retries[slots]+1, 0)
        self.supervised[slots] = supervised
        self.attempts[slots] += 1
        terminal = exhausted & (not self.allow_post_demo)
        periodic = (self.attempts[slots] >= self.resample_interval) & ~terminal
        resample = periodic | terminal
        refresh_slots = slots[resample]
        if len(refresh_slots):
            references = self.draw_references(len(refresh_slots))
            self.indices[refresh_slots] = references
            for key, storage in self.observation.items():
                storage[refresh_slots] = self.reference[key][references].detach()
            self.age[refresh_slots] = 0
            self.attempts[refresh_slots] = 0
            self.retries[refresh_slots] = 0
            self.supervised[refresh_slots] = True
            if self.progress_directions is not None:
                initial = (self.reference['current_q'][references,:2]
                           * self.progress_directions[references]).sum(-1)
                self.progress_history[refresh_slots] = initial[:,None]
        zero = accepted.sum()*0
        result = dict(advances=advance.sum(), retries=retry.sum(), rollbacks=(retry & ~resample).sum(),
            accepted_transitions=accepted.sum(), resets=resample.sum(),
            periodic_resamples=periodic.sum(),
            terminal_resets=terminal.sum(), invalid_resets=zero, timeout_resets=periodic.sum(),
            demo_exhausted=exhausted.sum(), autonomous_states=(~self.supervised[slots]).sum(),
            generated_inputs=(self.age[slots] > 0).sum(),
            no_forward_progress_resets=zero,
            no_forward_progress_rejections=no_progress.sum())
        if reset_reasons is not None:
            for key, value in reset_reasons.items():
                result[key+'_resets'] = zero
                result[key+'_rejections'] = (value & retry).sum()
        return result

    def state_dict(self):
        return dict(schema='bounded_attempt_rollout_v5', indices=self.indices.detach().cpu(),
            allow_post_demo=self.allow_post_demo,
            attempts=self.attempts.detach().cpu(), resample_interval=self.resample_interval,
            age=self.age.detach().cpu(), retries=self.retries.detach().cpu(), cursor=self.cursor,
            supervised=self.supervised.detach().cpu(),
            progress=(None if self.progress_directions is None else dict(
                history=self.progress_history.detach().cpu(), directions=self.progress_directions.detach().cpu(),
                window=self.progress_window, minimum_m=self.progress_minimum)),
            observation={k:v.detach().cpu() for k,v in self.observation.items()})
