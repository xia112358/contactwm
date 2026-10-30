"""One complete-solid penetration contract for endpoint predictors.

Newton activation/allocation remains the sole source of contact truth.
Native triangle depths are diagnostic only; missing solid evidence is an error.
"""
import math

import torch

from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE
from contact_solver.endpoint_task import PENETRATION_LOSS_SCALE_M


PENETRATION_SCHEMA = 'complete_solid_endpoint_penetration_v1'


def penetration_contract(limits, *, penalty='huber', scale_m=PENETRATION_LOSS_SCALE_M):
    if penalty not in ('log1p', 'huber') or not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('Invalid endpoint penetration penalty or scale')
    return dict(schema=PENETRATION_SCHEMA, distance='complete realized solid signed distance',
        scope='enabled robot/terrain and self pairs; same initialized assets and collision filters',
        acceptance_penetration_m=limits.shallow_penetration_m,
        aggregation='maximum pair depth', invalid_query='unknown; reject; no native-depth fallback',
        native_depth='diagnostic only', penalty=penalty, penalty_scale_m=scale_m,
        optimization_target_m=0., severity_threshold_is_acceptance=False)


def solid_depth_m(metrics):
    if 'solid_penetration_cm' not in metrics:
        raise ValueError('Missing complete-solid penetration evidence')
    return metrics['solid_penetration_cm'] / 100


def penetration_checks(metrics, *, limits=DEFAULT_ACCEPTANCE):
    depth = solid_depth_m(metrics)
    known = torch.isfinite(depth) & (depth >= 0)
    exceeded = known & (depth > limits.shallow_penetration_m)
    return dict(penetration_exceeded=exceeded, penetration_unknown=~known,
                penetration_passed=known & ~exceeded)


def solid_generation_mask(metrics, *, limits=DEFAULT_ACCEPTANCE):
    return (metrics['newton_contact_accepted'].bool()
            & penetration_checks(metrics, limits=limits)['penetration_passed']
            & (metrics['newton_invalid_fullbody_witnesses'] == 0))


def huber_squared(squared_residual):
    """Huber(sqrt(x), delta=1), with a finite derivative at x=0."""
    return torch.where(squared_residual <= 1, .5*squared_residual,
                       squared_residual.clamp_min(1).sqrt()-.5)


def solid_separation_record(native, depth, pair, *, schema=PENETRATION_SCHEMA):
    """Scalar evaluation view with the same depth source as tensor training."""
    if len(depth) != 2 or any(not math.isfinite(v) or v < 0 for v in depth):
        raise ValueError('Invalid complete-solid penetration evidence')
    result = dict(native, distance_schema=schema)
    for kind, name in enumerate(('terrain', 'self')):
        result[f'{name}_penetration_m'] = float(depth[kind])
        result[f'worst_{name}'] = None
        # Native closest/worst witnesses must not be presented as solid ones.
        result[f'closest_{name}'] = None
        if depth[kind] > 0:
            indices = (pair['full_kind'] == kind).nonzero()[0]
            if len(indices) == 0:
                raise ValueError('Solid penetration depth has no witness')
            index = indices[pair['dist'][indices].argmin()]
            result[f'worst_{name}'] = {k: v[index].tolist() for k, v in pair.items()}
    return result
