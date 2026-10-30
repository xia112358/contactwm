"""Transactional minibatch updates. The closure owns fresh physical evidence.

Project the optimizer's *increment*, including momentum and weight decay.
No teacher pose or inference correction is constructed here.
"""
from dataclasses import dataclass, field
import copy
import math
import torch
from .constraint_learning import AugmentedLagrangian, validate
from .constraint_direction import project_direction
from .feasibility_filter import FeasibilityFilter


@dataclass
class ConstraintBatch:
    prior: torch.Tensor
    residuals: list
    protection: list
    evidence: dict
    batch_size: int
    payload: object = None
    diagnostics: dict = field(default_factory=dict)

    def validate(self):
        validate(self.residuals)
        validate(self.protection)
        if self.batch_size < 1 or self.prior.numel() != 1 or not torch.isfinite(self.prior):
            raise ValueError('Expected finite scalar prior and positive batch size')
        if any(type(v) is not bool for v in self.evidence.values()):
            raise ValueError('Evidence must be explicit bool, never inferred geometry')
        if len({r.sample_id for r in self.residuals}) != self.batch_size:
            raise ValueError('Each batch sample requires a unique stable identity')


@dataclass(frozen=True)
class StepConfig:
    max_trials: int = 6
    active_slack: float = .05
    interior_fraction: float = .1
    max_active_rows: int = 128
    armijo: float = 1e-4

    def __post_init__(self):
        if self.max_trials < 1 or self.max_active_rows < 1 or not (0 < self.armijo < 1):
            raise ValueError('Invalid step limits')
        if not math.isfinite(self.active_slack) or self.active_slack < 0 or not 0 <= self.interior_fraction <= 1:
            raise ValueError('Invalid active constraint reserve')


class ConstrainedOptimizerStep:
    """One optimizer proposal per batch, bounded fresh-query backtracking.

    Closure is deterministic, creates a fresh graph and solver query every call,
    and has no training-state side effects. Mutable module buffers are restored
    before each call; accepted buffers and Adam state are committed once. The
    Adam moments describe the original gradient, while the applied increment is
    projected/scaled. A rejected proposal advances neither moments nor AL state.
    """
    def __init__(self, model, optimizer, rules, *, config=StepConfig(), al=None):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.guard = FeasibilityFilter(rules)
        self.al = al or AugmentedLagrangian(rho=1., max_rho=1e3)
        self.params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Duplicate optimizer parameters')
        if not self.params or len({(p.device, p.dtype) for p in self.params}) != 1:
            raise ValueError('Expected parameters on one device with one dtype')

    def state_dict(self):
        return dict(schema='constrained_optimizer_step_v1', config=self.config.__dict__,
                    rules=[r.__dict__ for r in self.guard.rules], al=self.al.state_dict())

    def load_state_dict(self, state):
        if (state['schema'] != 'constrained_optimizer_step_v1' or state['config'] != self.config.__dict__
                or state['rules'] != [r.__dict__ for r in self.guard.rules]):
            raise ValueError('Constrained step contract changed')
        self.al.load_state_dict(state['al'])

    def _flat_grad(self, value):
        if not value.requires_grad:
            return torch.cat([torch.zeros_like(p).flatten() for p in self.params])
        gradients = torch.autograd.grad(value, self.params, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(self.params, gradients)])

    def step(self, closure):
        weights = [p.detach().clone() for p in self.params]
        buffers = [(b, b.detach().clone()) for b in self.model.buffers()]
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        al_state = self.al.state_dict()
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.params[0].is_cuda else None
        committed = False
        attempts = []
        def restore_buffers_rng():
            with torch.no_grad():
                for b, saved in buffers: b.copy_(saved)
            torch.random.set_rng_state(rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        def set_weights(increment=None, fraction=0.):
            with torch.no_grad():
                chunks = [None]*len(weights) if increment is None else increment.split([p.numel() for p in self.params])
                for p, w, d in zip(self.params, weights, chunks):
                    p.copy_(w if d is None else w + fraction*d.reshape_as(p).to(p))
        try:
            restore_buffers_rng()
            self.optimizer.zero_grad(set_to_none=True)
            before = closure(); before.validate()
            value = self.al.loss(before.prior, before.residuals) / before.batch_size
            grad = self._flat_grad(value).detach()
            normals, bounds = [], []
            for row in before.protection:
                rules = [r for r in self.guard.rules if row.constraint_id.startswith(r.prefix)]
                if not rules: continue
                tolerance = min(r.tolerance for r in rules)
                for component in row.value:
                    for sign in ((1., -1.) if row.kind == 'eq' else (1.,)):
                        v = sign*component
                        scalar = float(v.detach())
                        if scalar < tolerance-self.config.active_slack: continue
                        if len(normals) >= self.config.max_active_rows:
                            raise ValueError('Active-row memory budget exceeded; reduce batch size')
                        normals.append(self._flat_grad(v).detach())
                        bounds.append(-min(self.config.interior_fraction*self.config.active_slack,
                                           max(0., scalar-(tolerance-self.config.active_slack))))
            value.backward()
            self.optimizer.step()
            proposal = torch.cat([(p.detach()-w).flatten() for p, w in zip(self.params, weights)])
            set_weights()
            a = torch.stack(normals) if normals else proposal.new_empty((0, proposal.numel()))
            b = proposal.new_tensor(bounds)
            direction, info = project_direction(proposal, a, bounds=b, algorithm='rowspace')
            if not info['converged'] and (info.get('feasibility') or {}).get('status') == 2:
                failed = info
                direction, info = project_direction(proposal, a, algorithm='rowspace')
                info['infeasible_reserve_attempt'] = failed
            slope = float(grad.double() @ direction)
            report = dict(accepted=False, projection=info, trials=attempts,
                          loss_before=float(value.detach()), slope=slope, gradient_norm=float(grad.norm()), active_rows=len(normals), before=before.diagnostics)
            if not info['converged'] or slope >= 0:
                report['status'] = 'projection_failed' if not info['converged'] else 'not_descent'
                return before, report
            required = {key: True for key, established in before.evidence.items() if established}
            for trial in range(self.config.max_trials):
                fraction = .5**trial
                set_weights(direction, fraction)
                restore_buffers_rng()
                # Keep autograd enabled, matching the baseline forward path.
                after = closure(); after.validate()
                if after.batch_size != before.batch_size or set(after.evidence) != set(before.evidence):
                    raise ValueError('Batch/evidence identity changed within optimizer step')
                number = float(self.al.loss(after.prior, after.residuals).detach()) / after.batch_size
                decision = self.guard.assess(before.protection, after.protection,
                    required_evidence=required, candidate_evidence=after.evidence)
                moved = any(not torch.equal(p.detach(), w) for p, w in zip(self.params, weights))
                accepted = moved and decision['accepted'] and number < report['loss_before'] and number <= report['loss_before']+self.config.armijo*fraction*slope
                attempts.append(dict(fraction=fraction, loss=number, accepted=accepted, reasons=decision['reasons'], diagnostics=after.diagnostics))
                if accepted:
                    self.al.update(after.residuals, adapt_rho=False)
                    committed = True
                    report.update(accepted=True, status='accepted', loss_after=number, fraction=fraction, after=after.diagnostics)
                    return after, report
            report['status'] = 'backtracking_exhausted'
            return before, report
        finally:
            if not committed:
                set_weights()
                restore_buffers_rng()
                self.optimizer.load_state_dict(optimizer_state)
                self.al.load_state_dict(al_state)
            self.optimizer.zero_grad(set_to_none=True)
