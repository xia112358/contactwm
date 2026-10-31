"""Check published checkpoint contracts on CPU, without simulation or CUDA setup."""
from __future__ import annotations
import json
from pathlib import Path


def validate_state(state, recipe):
    from .contact_response import metadata, validate_identity
    expected = json.dumps(recipe['contact_response_contract'], sort_keys=True)
    if metadata('newton_contacts_v2') != expected:
        raise ValueError('Installed source physics differs from the frozen original12 recipe')
    validate_identity(state, {'contact_response_contract': expected})
    if state.get('task_manifest_sha256') != recipe['task_manifest_sha256']:
        raise ValueError('Checkpoint task identity differs from original12')
    if not state.get('args', {}).get('no_reset_noise'):
        raise ValueError('Original12 requires no_reset_noise=true in the original checkpoint')
    if state['args'].get('contact_response') != 'newton_contacts_v2':
        raise ValueError('Checkpoint arguments do not select the original collision pipeline')
    required = {'pi', 'wm', 'critic', 'opt_pi', 'opt_wm', 'opt_v', 'reset_sampler_state', 'iteration'}
    if missing := required - state.keys():
        raise ValueError(f'Incomplete full training checkpoint: {sorted(missing)}')
    return state['iteration']


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    recipe = json.loads((root/'configs/experiments/original12_newton2ms.json').read_text())
    from .task_manifest import verify_task_manifest
    report = verify_task_manifest(root/recipe['task_manifest'])
    if report['manifest_sha256'] != recipe['task_manifest_sha256']:
        raise ValueError('Published original12 manifest changed')
    import torch
    checkpoints = {}
    for key, expected in [('initialization', 1000), ('saved_state', 1500)]:
        # The caller verifies the release hashes before this trusted load.
        state = torch.load(root/recipe[key], map_location='cpu', weights_only=False, mmap=True)
        iteration = validate_state(state, recipe)
        if iteration != expected:
            raise ValueError(f'{key} iteration changed: {iteration}')
        checkpoints[key] = iteration
    print(json.dumps(dict(task=report, checkpoint_iterations=checkpoints,
                          scope='File and checkpoint contracts on CPU; no physics, convergence or task-success validation'), indent=2))


if __name__ == '__main__':
    main()
