"""Full epoch-boundary state for matched Predictor gradient experiments."""
import copy
import random

import numpy as np
import torch

SCHEMA = 'predictor_epoch_training_state_v2'
EXPERIMENT_FIELDS = {'output', 'resume_training_state', 'stop_after_step',
    'capture_training_state_steps', 'isolate_unsupervised_execution_gradients',
    'resume_gradient_isolation'}


def validate_resume_config(saved, current):
    saved, current = dict(saved), dict(current)
    # Missing fields belong to historical runs, never to new defaults.
    for config in (saved, current):
        config.setdefault('optimizer_gradient_ownership', 'legacy')
        config.setdefault('limb_contact_modes', False)
        config.setdefault('training_stage', 'legacy_mixed')
    migrate = bool(current.get('resume_gradient_isolation', False))
    if migrate:
        if (current.get('architecture') != 'full1000_position_predictor_v1'
                or current.get('execution_observation_gradients', False)
                or current.get('isolate_unsupervised_execution_gradients', False)
                or current['optimizer_gradient_ownership'] != 'planner_supervised_v1'):
            raise ValueError('Gradient-isolation migration requires isolated v1 and planner-owned optimizer updates')
    from generator.staged_rollout_pool import validate_stage_transition
    stage_changed = validate_stage_transition(saved, current)
    stage_fields = ({'training_stage'} if stage_changed else set()) | (
        {'steps'} if current['training_stage'] != 'legacy_mixed' else set())
    allowed = EXPERIMENT_FIELDS | stage_fields | ({'execution_observation_gradients', 'optimizer_gradient_ownership'} if migrate else set())
    differences = {key: (saved.get(key), current.get(key)) for key in saved.keys() | current.keys()
        if key not in allowed and saved.get(key) != current.get(key)}
    if differences:
        raise ValueError(f'Resume training configuration differs: {differences}')
    return migrate and bool(saved.get('execution_observation_gradients', False))


def capture_training_state(*, step, model, optimizer, scheduler, pool, pending, config,
        robot_asset_json, training_predictions, history, best_score, best_step, best_state):
    return dict(schema=SCHEMA, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), pool=pool.state_dict() if pool is not None else None,
        pending_plan=pending.state_dict() if pending is not None else None, config=config,
        robot_asset_json=robot_asset_json, training_predictions=training_predictions,
        history=history, best_score=best_score, best_step=best_step, best_state=best_state,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore_training_state(state, *, model, optimizer, scheduler, pool, pending,
        config, robot_asset_json):
    if state.get('schema') != SCHEMA or state.get('robot_asset_json') != robot_asset_json:
        raise ValueError('Resume requires a complete matching canonical Predictor training state')
    isolate_shared = validate_resume_config(state['config'], config)
    if (pool is None) != (state['pool'] is None) or (pending is None) != (state['pending_plan'] is None):
        raise ValueError('Resume rollout/pending state differs')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    migration = None
    if isolate_shared:
        from generator.optimizer_ownership import reset_shared_optimizer_state
        migration = dict(schema='planner_owned_representation_migration_v1',
            source_step=state['step'], reset_parameters=reset_shared_optimizer_state(model, optimizer),
            reason='historical shared optimizer moments mix planner and executor gradients',
            preserved='model, planner-head and executor optimizer states, scheduler, pool, pending plans, RNG, progress')
    if pool is not None:
        pool.load_state_dict(state['pool'])
    if pending is not None:
        for key, value in state['pending_plan'].items():
            destination = getattr(pending, key)
            if destination.shape != value.shape or destination.dtype != value.dtype:
                raise ValueError(f'Resume pending plan shape/dtype differs: {key}')
            destination.copy_(value.to(destination.device))
    # Restore RNG last, after pool construction (which draws reference indices).
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        if not torch.cuda.is_available() or len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA RNG topology differs')
        torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    restored = copy.deepcopy({key:state[key] for key in
        ('step', 'training_predictions', 'history', 'best_score', 'best_step', 'best_state')})
    restored['optimizer_migration'] = migration
    return restored


def save_epoch_training_state(state, directory):
    """Retain every milestone, including pre-pool warmup and initialization."""
    from pathlib import Path
    directory = Path(directory)
    milestone = directory / f"training_state_{state['step']:04d}.pt"
    if milestone.exists():
        raise FileExistsError(f'Refusing to overwrite retained training state: {milestone}')
    torch.save(state, milestone)
    torch.save(state, directory / 'parallel_training_state.pt')
