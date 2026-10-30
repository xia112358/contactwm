"""The weighted terms actually optimized by the endpoint Predictor."""

OBJECTIVE_SCHEMA = 'endpoint_predictor_single_recovery_objective_v1'
LOSS_GROUPS = ('planner', 'imitation', 'contact', 'endpoint', 'joint_limit')
DIAGNOSTIC_LOSSES = (
    'own_plan_realization_loss', 'missing_pair_approach_loss',
    'contact_establishment_loss', 'interaction_location_loss',
    'interaction_pose_prior_loss', 'nominal_taskspace_teacher_loss',
    'collision_loss', 'newton_collision_loss',
)
POSE_POSITION_SCALE_M = .10
POSE_ROTATION_MATRIX_SCALE = .30
POSE_JOINT_SCALE_RAD = .50
STRUCTURE_WEIGHT = .50


def compose_endpoint_loss(*, planner, imitation, contact, endpoint, joint_limit,
                          legacy_keep=None):
    """Sum each weighted group once; expose those same tensors for logging."""
    values = dict(planner=planner, imitation=imitation, contact=contact,
                  endpoint=endpoint, joint_limit=joint_limit)
    if legacy_keep is not None:
        values['legacy_keep'] = legacy_keep
    if any(value.shape != planner.shape for value in values.values()):
        raise ValueError('Endpoint loss groups require matching per-sample shapes')
    total = sum(values.values())
    return total, {f'loss_contribution_{name}': value for name, value in values.items()}

