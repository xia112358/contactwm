"""Versioned, read-only branch gradient audit; never construct an optimizer."""
import hashlib
import json

import torch




def audit(model, cfg, inputs, validation, labels_for, objective):
    """Audit the explicitly loaded architecture; never import an archived model."""
    from generator.predictor_architecture import load_position_predictor
    from generator.structured_position_predictor import STRUCTURED_SCHEMA
    from generator.predictor_initialization import state_fingerprint
    model.eval()
    original = state_fingerprint(model)
    checkpoint = torch.load(cfg.audit_checkpoint, map_location='cpu', weights_only=False)
    # Checkpoint construction/version checks are the same as rollout evaluation.
    reloaded = load_position_predictor(checkpoint, device=validation.device).eval()
    count = min(cfg.static_audit_samples, len(validation))
    ids = validation[torch.linspace(0, len(validation)-1, count, device=validation.device).long()]
    batches = [('reference_validation', ids, {k: v[ids] for k, v in inputs.items()})]
    if cfg.static_audit_pool is not None:
        if cfg.static_audit_pool.parent.resolve() != cfg.audit_checkpoint.parent.resolve():
            raise ValueError('Pool must accompany the asset-validated checkpoint')
        state = torch.load(cfg.static_audit_pool, map_location=validation.device, weights_only=False)
        if state['schema'] != 'persistent_parallel_rollout_training_state_v1' or state['step'] != checkpoint['step']:
            raise ValueError('Pool/checkpoint mismatch')
        pool = state['pool']; slots = (pool['age'] > 0).nonzero().flatten()[:cfg.static_audit_samples]
        if not len(slots): raise ValueError('No generated pool inputs')
        batches.append(('generated_pool_states', pool['indices'][slots],
            {k: v[slots] for k, v in pool['observation'].items()}))

    def cpu(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu()
        if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return type(value)(cpu(v) for v in value)
        return value

    named = list(model.named_parameters()); params = [p for _, p in named]
    group_ids = {group: {id(p) for p in members} for group, members in model.training_parameter_groups().items()}
    sets = {group: [i for i, (_, p) in enumerate(named) if id(p) in members]
            for group, members in group_ids.items()}
    records = []
    for source, ids, observation in batches:
        prediction = model(**observation)
        # Use the same grad mode/attention kernel as the audited forward.
        # no_grad can select a different fast path and break bitwise comparison.
        repeat = reloaded(**observation)
        for key in prediction.__dataclass_fields__:
            first, second = getattr(prediction, key), getattr(repeat, key)
            if first is not None:
                torch.testing.assert_close(
                    first.detach() if isinstance(first, torch.Tensor) else first,
                    second.detach() if isinstance(second, torch.Tensor) else second, rtol=0, atol=0)
        del repeat
        loss, metrics, queried = objective(prediction, observation, ids,
            labels_for(observation, ids, recursive=source == 'generated_pool_states'))
        torch.save({'schema': 'frozen_actual_newton_gradient_witnesses_v1',
            'sample_indices': ids.cpu(), 'q_world': prediction.qpos.detach().cpu(),
            'q_local': queried[0].q.detach().cpu(), 'observations': cpu(observation),
            'world_frame': cpu(queried[0].world_frame), 'newton': cpu(queried[1])},
            cfg.output/f'witnesses_{source}.pt')
        terms = {'planner': metrics['audit_planner'], 'contact_execution': metrics['unified_region_loss'],
                 'imitation': metrics['audit_imitation'], 'layout': metrics['audit_layout'],
                 'total': loss}
        details = {}
        for term, value in terms.items():
            gradients = torch.autograd.grad(value.mean(), params, retain_graph=True, allow_unused=True)
            norms = {group: sum(float(gradients[i].detach().double().square().sum())
                for i in members if gradients[i] is not None)**.5 for group, members in sets.items()}
            if checkpoint['schema'] == STRUCTURED_SCHEMA and term != 'total':
                forbidden = 'executor' if term == 'planner' else 'planner'
                assert all(gradients[i] is None for i in sets[forbidden]), (source, term, forbidden)
            details[term] = {'mean_loss': float(value.detach().mean()), 'gradient_norms': norms}
        records.append({'source': source, 'architecture': checkpoint['schema'], 'samples': len(ids),
            'sample_indices': ids.cpu().tolist(), 'forward_bitwise_equal_after_reload': True,
            'newton_pairs': len(queried[0].pair['dist']),
            'valid_gradient_samples': int(metrics['region_gradient_valid'].sum()), 'components': details})
        print(json.dumps({'isolation_audit': source, 'architecture': checkpoint['schema'],
            'contact_gradient_norms': details['contact_execution']['gradient_norms']}), flush=True)
    assert state_fingerprint(model) == original
    assert all(p.grad is None for p in params)
    report = {'schema': 'versioned_planning_execution_gradient_audit_v2', 'parameter_updates': 0,
        'optimizer_constructed': False, 'weights_bitwise_unchanged': True,
        'source_checkpoint': str(cfg.audit_checkpoint),
        'source_checkpoint_sha256': hashlib.sha256(cfg.audit_checkpoint.read_bytes()).hexdigest(),
        'records': records,
        'scope': 'Fixed weights, same-version reload, fresh actual Newton witnesses. No training or quality claim.'}
    (cfg.output/'isolation_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    write_witness_manifest(cfg)
    print(json.dumps({'isolation_audit_complete': True, 'parameter_updates': 0}), flush=True)

def write_witness_manifest(cfg):
    from somaforge_core.robot_assets import decode_robot_asset_json
    from somaforge_core.newton_contacts import SCHEMA
    from somaforge_core.contact_face_selection import FACE_POLICY
    checkpoint = torch.load(cfg.audit_checkpoint, map_location='cpu', weights_only=False)
    asset = checkpoint['robot_asset_json']
    decode_robot_asset_json(asset, context='frozen contact audit')
    samples = torch.load(cfg.data_cache, map_location='cpu', weights_only=False)['samples']
    scenes = {}
    for folder in sorted((cfg.output/'newton_workers').glob('h*')):
        model_file, binding_file = folder/'model.json', folder/'binding.json'
        model = json.loads(model_file.read_text())
        scenes[folder.name] = {'model_fingerprint': model['model_fingerprint'],
            'model': str(model_file.resolve()), 'binding': str(binding_file.resolve()),
            'model_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(),
            'binding_sha256': hashlib.sha256(binding_file.read_bytes()).hexdigest()}
    witnesses = []
    for path in sorted(cfg.output.glob('witnesses_*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        indices = payload['sample_indices'].tolist()
        scene_names = [f"h{float(samples[i]['height']):.8f}" for i in indices]
        if not all(name in scenes for name in scene_names): raise ValueError('Unknown witness scene')
        witnesses.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sample_indices': indices, 'scene_by_batch_sample': scene_names})
    (cfg.output/'witness_manifest.json').write_text(json.dumps({
        'schema': 'frozen_actual_newton_gradient_witness_manifest_v1',
        'robot_asset_json': asset, 'contact_semantics': SCHEMA, 'face_selection': FACE_POLICY,
        'sampling': 'static query at saved raw predicted q; no integration; same real witnesses reused across derivative variants',
        'indexing': 'pairs.sample indexes the saved q batch; pairs.worldid is the reused solver slot within its scene',
        'scenes': scenes, 'witnesses': witnesses}, indent=2))
