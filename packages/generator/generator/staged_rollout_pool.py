"""Separate demonstrated chains and post-demonstration endpoint rollouts."""
import torch
from generator.parallel_rollout_pool import ParallelRolloutPool

STAGES = ('single_segment', 'demonstration_rollout', 'post_demonstration')


def validate_stage_transition(saved, current):
    old, new = saved.get('training_stage', 'legacy_mixed'), current.get('training_stage', 'legacy_mixed')
    if old == new:
        return False
    if (old, new) not in tuple(zip(STAGES, STAGES[1:])):
        raise ValueError(f'Unsupported training stage transition: {old} -> {new}')
    return True


class SeedTensor:
    """Index a compact terminal bank by original dataset row, without copying the corpus."""
    def __init__(self, values, lookup):
        self.values, self.lookup = values, lookup

    def __getitem__(self, index):
        row, tail = (index[0], index[1:]) if isinstance(index, tuple) else (index, ())
        mapped = self.lookup[row]
        if bool((mapped < 0).any()):
            raise ValueError('Requested an unverified terminal seed')
        value = self.values[mapped]
        return value[(slice(None), *tail)] if tail else value


class StagedRolloutPool(ParallelRolloutPool):
    def __init__(self, reference, training_indices, successor, size, *, stage,
                 terminal_bank=None, **kwargs):
        if stage not in STAGES[1:]:
            raise ValueError('Staged pool needs a recursive stage')
        self.stage = stage
        self.terminal_bank = terminal_bank or {}
        self.finished_episodes = []
        self.episode_totals = dict(count=0, accepted_steps=0, retries=0, attempts=0,
            accepted_depth_histogram={}, first_failure_depth_histogram={})
        device = training_indices.device
        if stage == 'demonstration_rollout':
            self.seed_indices = training_indices
        else:
            if not self.terminal_bank:
                raise ValueError('Stage 3 requires actual successful stage-2 terminal states; no demonstration fallback')
            self.seed_indices = torch.tensor(sorted(self.terminal_bank), device=device)
            lookup = torch.full_like(successor, -1)
            lookup[self.seed_indices] = torch.arange(len(self.seed_indices), device=device)
            reference = {k: SeedTensor(torch.stack([self.terminal_bank[int(i)][k]
                for i in self.seed_indices]).to(device), lookup) for k in reference}
        kwargs['allow_post_demo'] = False
        super().__init__(reference, training_indices, successor, size, **kwargs)
        self.supervised.fill_(stage == 'demonstration_rollout')
        self.first_failure_depth = torch.full_like(self.age, -1)
        self.episode_retries = torch.zeros_like(self.age)
        self.first_failure_reasons = [[] for _ in range(size)]

    def draw_references(self, count):
        return self.seed_indices[torch.randint(len(self.seed_indices), (count,), device=self.seed_indices.device)]

    @torch.no_grad()
    def update(self, slots, q, observed, completed, continuable, make_observation, *, reset_reasons=None):
        # Apply the same progress gate before recording acceptance/depth evidence.
        effective_continuable = continuable
        if self.progress_directions is not None:
            projection = (q[:, :2]*self.progress_directions[self.indices[slots]]).sum(-1)
            frontier = torch.maximum(self.progress_history[slots, -1], projection)
            no_progress = continuable & (self.age[slots]+1 >= self.progress_window) & (
                frontier-self.progress_history[slots, 1] < self.progress_minimum)
            effective_continuable = continuable & ~no_progress
            if reset_reasons is not None:
                reset_reasons['no_forward_progress'] = no_progress
        accepted = completed & effective_continuable
        terminal = self.supervised[slots] & accepted & (self.successor[self.indices[slots]] < 0)
        age = self.age[slots].clone()
        first = ~accepted & (self.first_failure_depth[slots] < 0)
        self.first_failure_depth[slots[first]] = age[first]
        for j in first.nonzero().flatten().tolist():
            self.first_failure_reasons[int(slots[j])] = [k for k,v in (reset_reasons or {}).items() if bool(v[j])]
        self.episode_retries[slots] += (~accepted).long()
        fresh_terminal = {}
        def observe(q, indices, actual, chosen):
            fresh = make_observation(q, indices, actual, chosen)
            for j in terminal[chosen].nonzero().flatten().tolist():
                fresh_terminal[int(indices[j])] = {k:v[j].detach().cpu().clone() for k,v in fresh.items()}
            return fresh
        rejected_reset = ~accepted & (self.stage == 'post_demonstration')
        reset = terminal | rejected_reset | (self.attempts[slots]+1 >= self.resample_interval)
        for j in reset.nonzero().flatten().tolist():
            slot = int(slots[j]); depth = int(age[j]+accepted[j])
            reasons = [k for k,v in (reset_reasons or {}).items() if bool(v[j])]
            self.finished_episodes.append(dict(stage=self.stage, accepted_steps=depth,
                first_failure_depth=int(self.first_failure_depth[slot]), first_failure_reasons=list(self.first_failure_reasons[slot]), retries=int(self.episode_retries[slot]),
                attempts=int(self.attempts[slot]+1), completed_demonstration=bool(terminal[j]),
                reset_reason=('completed_demonstration' if terminal[j] else
                              'prediction_rejected' if rejected_reset[j] else 'attempt_budget'),
                final_rejection_reasons=reasons))
        for record in self.finished_episodes[-int(reset.sum()):] if bool(reset.any()) else []:
            self.episode_totals['count'] += 1
            for key in ('accepted_steps', 'retries', 'attempts'):
                self.episode_totals[key] += record[key]
            for key, field in [('accepted_depth_histogram', 'accepted_steps'),
                               ('first_failure_depth_histogram', 'first_failure_depth')]:
                depth = str(record[field])
                hist = self.episode_totals[key]
                hist[depth] = hist.get(depth, 0)+1
        feedback = super().update(slots, q, observed, completed, continuable, observe,
            reset_reasons=reset_reasons, reset_on_rejection=self.stage == 'post_demonstration')
        self.terminal_bank.update(fresh_terminal)
        if self.stage == 'post_demonstration':
            self.supervised.fill_(False)
            feedback['autonomous_states'] = (~self.supervised[slots]).sum()
        self.first_failure_depth[slots[reset]] = -1
        self.episode_retries[slots[reset]] = 0
        for slot in slots[reset].tolist():
            self.first_failure_reasons[slot] = []
        return feedback

    def stage_metrics(self):
        import copy
        return dict(stage=self.stage, rejection_policy=self.rejection_policy,
            episode_totals=copy.deepcopy(self.episode_totals), terminal_seed_count=len(self.terminal_bank),
            depth_semantics=('successful steps beyond demonstrated endpoint' if self.stage == 'post_demonstration'
                             else 'successful steps from randomly sampled demonstration state'),
            active_accepted_steps=self.age.cpu().tolist(),
            active_first_failure_depth=self.first_failure_depth.cpu().tolist(),
            active_retries=self.episode_retries.cpu().tolist(),
            active_first_failure_reasons=[list(v) for v in self.first_failure_reasons],
            finished_episodes=list(self.finished_episodes))

    def state_dict(self):
        state = super().state_dict()
        state['staged'] = dict(stage=self.stage, rejection_policy=self.rejection_policy,
            terminal_bank=self.terminal_bank,
            first_failure_depth=self.first_failure_depth.cpu(), episode_retries=self.episode_retries.cpu(),
            finished_episodes=self.finished_episodes, first_failure_reasons=self.first_failure_reasons,
            episode_totals=self.episode_totals)
        return state

    def load_state_dict(self, state):
        stage = state.get('staged', {})
        if stage.get('stage') != self.stage:
            raise ValueError('Resume recursive stage differs')
        if stage.get('rejection_policy', 'rollback') != self.rejection_policy:
            raise ValueError('Resume rejection policy differs; historical stage-3 rollback states require explicit migration')
        super().load_state_dict(state)
        self.terminal_bank = stage['terminal_bank']
        self.first_failure_depth.copy_(stage['first_failure_depth'].to(self.age.device))
        self.episode_retries.copy_(stage['episode_retries'].to(self.age.device))
        self.finished_episodes = list(stage['finished_episodes'])
        self.first_failure_reasons = [list(v) for v in stage['first_failure_reasons']]
        import copy
        self.episode_totals = copy.deepcopy(stage['episode_totals'])

    @property
    def rejection_policy(self):
        return 'reset_episode' if self.stage == 'post_demonstration' else 'rollback'
