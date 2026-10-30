"""Explicit predictor weight provenance, independent of simulation workers."""
import hashlib
import torch


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize_predictor(model, *, initialization='scratch', seed=None):
    """Report scratch initialization without reading or migrating any weights."""
    if initialization != 'scratch':
        raise ValueError('Predictor training supports scratch initialization only')
    return {}, [], sorted(model.state_dict()), {
        'schema': 'predictor_initialization_v1', 'initialization': 'scratch',
        'seed': seed, 'source': None, 'source_step': None, 'sha256': None,
        'predictor_checkpoints_loaded': [], 'optimizer_resumed': False,
        'shared_weights_exact': False, 'new_geometry_zero_output': False,
        'random_state_sha256': state_fingerprint(model),
    }


def initialize_training_joint_bias(model, target_q, training_indices):
    """Use training targets only; all other trainable weights keep their initialization."""
    if not len(training_indices):
        raise ValueError('Joint bias initialization requires training samples')
    mean = target_q[training_indices, 7:].mean(0)
    if not bool(torch.isfinite(mean).all()):
        raise ValueError('Nonfinite training joint bias')
    with torch.no_grad():
        if hasattr(model, 'joint_pose_head'):
            model.joint_pose_head.bias.copy_(mean)
        else:
            model.pose_head.bias[7:].copy_(mean)
    return {'schema': 'training_joint_mean_bias_v1', 'training_samples': len(training_indices),
            'joint_bias_rad': mean.detach().cpu().tolist(), 'validation_or_test_used': False}
