"""Keep optimizer updates within the losses that own each parameter group."""
from typing import Literal

import torch


OptimizerOwnership = Literal['legacy', 'planner_supervised_v1']


def prepare_optimizer_gradients(model, supervised, *, policy: OptimizerOwnership,
                                isolate_unsupervised_execution_gradients=False):
    """Skip planning updates when this batch contains no planning supervision.

    Zero gradients still advance AdamW moments and weight decay. None skips the
    parameter and preserves its optimizer state until a legitimate update.
    Shared parameters in the explicit legacy-gradient experiment remain owned
    by execution unless that execution path is also isolated.
    """
    if policy not in ('legacy', 'planner_supervised_v1'):
        raise ValueError(f'Unknown optimizer ownership policy: {policy}')
    if supervised.dtype != torch.bool or supervised.ndim != 1 or not len(supervised):
        raise ValueError('Planner supervision must be a nonempty boolean batch mask')
    if policy == 'legacy' or bool(supervised.any()):
        return ()
    groups = model.training_parameter_groups()
    skipped = ['planner']
    if isolate_unsupervised_execution_gradients and 'shared' in groups:
        skipped.append('shared')
    for name in skipped:
        for parameter in groups[name]:
            parameter.grad = None
    return tuple(skipped)


def reset_shared_optimizer_state(model, optimizer):
    """Explicit shared-to-isolated migration; preserve planner heads/executor.

    Old shared moments mix planning and execution gradients and cannot be
    separated retrospectively. Reset only those moments, including their
    per-parameter update counts; ordinary same-mode resume never calls this.
    """
    if getattr(model, 'schema', None) != 'full1000_position_predictor_v1':
        raise ValueError('Shared optimizer migration requires the v1 architecture')
    if model.execution_observation_gradients:
        raise ValueError('Shared optimizer migration requires isolated execution')
    reset = []
    for name, parameter in model.named_parameters():
        if not name.startswith(model.EXECUTION_PREFIXES + model.PLAN_HEAD_PREFIXES):
            if parameter in optimizer.state:
                optimizer.state.pop(parameter)
                reset.append(name)
    return reset
