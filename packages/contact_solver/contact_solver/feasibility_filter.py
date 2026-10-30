"""Candidate acceptance guard; never generates a correction or contact label."""
from dataclasses import dataclass
import math
from .constraint_learning import validate


@dataclass(frozen=True)
class ProtectionRule:
    prefix: str
    tolerance: float

    def __post_init__(self):
        if not self.prefix or not math.isfinite(self.tolerance) or self.tolerance<0:
            raise ValueError('Protection rules require a prefix and finite nonnegative tolerance')


class FeasibilityFilter:
    """Protect individual residual components, with ratcheting restoration.

    Each component is bounded by max(previous violation, configured tolerance).
    Unmatched constraints remain controlled by the optimizer's objective. Actual
    contact evidence comes from the caller's authoritative backend, separately.
    Missing previously protected rows are unknown, never silently satisfied.
    """
    def __init__(self,rules):
        self.rules=tuple(rules)
        if not self.rules:raise ValueError('At least one protection rule required')

    def _extract(self,rows):
        validate(rows);result={}
        for r in rows:
            matches=[rule for rule in self.rules if r.constraint_id.startswith(rule.prefix)]
            if not matches:continue
            tolerance=min(rule.tolerance for rule in matches)
            values=(r.value.abs() if r.kind=='eq' else r.value.relu()).detach().cpu().tolist()
            for component,value in zip(r.component_ids,values):result[(*r.key,component)]=(value,tolerance)
        return result

    def assess(self,previous,candidate,*,required_evidence,candidate_evidence):
        old=self._extract(previous);new=self._extract(candidate);reasons=[]
        for key,expected in required_evidence.items():
            if expected is not True:raise ValueError('Required evidence must explicitly require True')
            if candidate_evidence.get(key) is not True:reasons.append(dict(kind='actual_evidence_missing_or_false',key=key))
        for key,(value,tolerance) in old.items():
            if key not in new:reasons.append(dict(kind='missing_protected_residual',key=list(key)));continue
            next_value,next_tolerance=new[key]
            if next_tolerance!=tolerance:raise ValueError('Protection definition changed')
            cap=max(value,tolerance)
            if next_value>cap:reasons.append(dict(kind='protected_residual_worsened',key=list(key),before=value,after=next_value,cap=cap))
        for key,(value,tolerance) in new.items():
            if key not in old and value>tolerance:reasons.append(dict(kind='new_protected_violation',key=list(key),after=value,cap=tolerance))
        return dict(accepted=not reasons,reasons=reasons)
