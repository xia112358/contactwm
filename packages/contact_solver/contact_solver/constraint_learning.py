"""Robot-independent constraint losses. Geometry guidance is not contact truth.

Residual values are dimensionless and signed: equalities == 0, inequalities
<= 0. Identity includes sample and task revision, not position in a batch.
"""
from dataclasses import dataclass
from typing import Literal, Sequence
import math
import torch


@dataclass(frozen=True)
class Residual:
    sample_id: str
    context_id: str
    constraint_id: str
    kind: Literal['eq', 'le']
    value: torch.Tensor
    component_ids: tuple[str, ...]
    valid: bool = True
    invalid_reason: str = ''

    @property
    def key(self):
        return (self.sample_id, self.context_id, self.constraint_id)

    @classmethod
    def scaled(cls, sample_id, context_id, constraint_id, kind, value, scale, *, component_ids=None):
        scale=torch.as_tensor(scale,device=value.device,dtype=value.dtype)
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):
            raise ValueError('Residual scales must be finite and positive')
        normalized=(value/scale).reshape(-1)
        ids=tuple(str(i) for i in range(normalized.numel())) if component_ids is None else tuple(component_ids)
        return cls(sample_id,context_id,constraint_id,kind,normalized,ids)


def validate(residuals: Sequence[Residual]):
    seen=set()
    for r in residuals:
        if not all(isinstance(k,str) and k for k in r.key):raise ValueError('Nonempty sample, context and constraint IDs required')
        if r.key in seen:raise ValueError(f'Duplicate residual identity: {r.key}')
        seen.add(r.key)
        if not r.valid:raise ValueError(f'Unknown/invalid constraint {r.key}: {r.invalid_reason}')
        if r.kind not in ('eq','le'):raise ValueError(f'Unknown constraint kind: {r.kind}')
        if r.value.ndim!=1 or not r.value.numel():raise ValueError('Residual must be a nonempty vector')
        if len(r.component_ids)!=r.value.numel() or len(set(r.component_ids))!=len(r.component_ids):raise ValueError('Unique stable component IDs required')
        if not bool(torch.isfinite(r.value).all()):raise ValueError(f'Nonfinite constraint {r.key}')


def violation_metrics(residuals):
    validate(residuals)
    return {'/'.join(r.key):dict(max=float((r.value.abs() if r.kind=='eq' else r.value.relu()).detach().max()),
        rms=float((r.value if r.kind=='eq' else r.value.relu()).detach().square().mean().sqrt())) for r in residuals}


def penalty_loss(prior, residuals, *, rho):
    """Fixed quadratic penalty on the same signed constraints used by AL."""
    if not math.isfinite(rho) or rho<=0:raise ValueError('rho must be finite and positive')
    validate(residuals)
    return prior+sum(.5*rho*(r.value if r.kind=='eq' else r.value.relu()).square().sum() for r in residuals)


class AugmentedLagrangian:
    """Explicit outer dual updates; forward evaluation never changes state.

    Dual variables are owned per sample/context/constraint/component. Changing
    context starts a new state. Missing rows are retained until explicitly reset;
    dynamic collision rows must use physical witness IDs, never array indices.
    """
    def __init__(self, *, rho=1., growth=2., contraction=.75, max_rho=1e4):
        if not all(math.isfinite(x) for x in (rho,growth,contraction,max_rho)) or not (0<rho<=max_rho and growth>=1 and 0<contraction<1):
            raise ValueError('Invalid augmented-Lagrangian parameters')
        self.config=dict(rho=rho,growth=growth,contraction=contraction,max_rho=max_rho)
        self._state={}

    def _check(self,residuals):
        validate(residuals)
        for r in residuals:
            old=self._state.get(r.key)
            if old is not None and old['kind']!=r.kind:raise ValueError('Constraint kind changed without a new context ID')

    def _values(self,r):
        old=self._state.get(r.key)
        if old is None:return torch.zeros_like(r.value),self.config['rho']
        return r.value.new_tensor([old['dual'].get(i,0.) for i in r.component_ids]),old['rho']

    def loss(self,prior,residuals):
        self._check(residuals)
        result=prior
        for r in residuals:
            dual,rho=self._values(r)
            if r.kind=='eq':result=result+(dual*r.value+.5*rho*r.value.square()).sum()
            else:result=result+((dual+rho*r.value).relu().square()-dual.square()).sum()/(2*rho)
        return result

    def update(self,residuals, *, adapt_rho=True):
        """Call after a block of primal steps, using freshly evaluated residuals."""
        self._check(residuals)
        pending={}
        for r in residuals:
            dual,rho=self._values(r);value=r.value.detach()
            updated=dual+rho*value
            if r.kind=='le':updated=updated.clamp_min(0)
            magnitude=float((value.abs() if r.kind=='eq' else value.relu()).max())
            old=self._state.get(r.key)
            next_rho=rho
            if adapt_rho and old is not None and magnitude>self.config['contraction']*old['violation']:
                next_rho=min(rho*self.config['growth'],self.config['max_rho'])
            entries={} if old is None else dict(old['dual'])
            entries.update(zip(r.component_ids,updated.cpu().tolist()))
            if not all(math.isfinite(x) for x in entries.values()):raise ValueError('Nonfinite multiplier update')
            pending[r.key]=dict(kind=r.kind,dual=entries,rho=next_rho,violation=magnitude)
        self._state.update(pending)

    def reset(self, *, sample_id, context_id=None):
        self._state={k:v for k,v in self._state.items() if not (k[0]==sample_id and (context_id is None or k[1]==context_id))}

    def state_dict(self):
        import copy
        return copy.deepcopy(dict(schema='constraint_al_v1',config=self.config,
            entries=[dict(key=list(k),**v) for k,v in self._state.items()]))

    def load_state_dict(self,payload):
        if payload['schema']!='constraint_al_v1' or payload['config']!=self.config:raise ValueError('AL state contract mismatch')
        import copy
        pending={}
        for entry in payload['entries']:
            e=copy.deepcopy(entry);key=tuple(e.pop('key'))
            if len(key)!=3 or key in pending or not all(isinstance(x,str) and x for x in key):raise ValueError('Invalid saved identity')
            if e['kind'] not in ('eq','le') or not 0<e['rho']<=self.config['max_rho'] or not math.isfinite(e['violation']) or e['violation']<0:raise ValueError('Invalid saved constraint state')
            if not all(isinstance(k,str) and math.isfinite(v) and (e['kind']=='eq' or v>=0) for k,v in e['dual'].items()):raise ValueError('Invalid saved multiplier')
            pending[key]=e
        self._state=pending
