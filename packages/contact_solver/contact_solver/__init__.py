"""Contact objectives and pose/trajectory feasibility solvers."""
from importlib import import_module

_EXPORTS = {'G1MechanicalProjector': ('contact_solver.mechanical', 'G1MechanicalProjector'),
 'MechanicalConstraintError': ('contact_solver.mechanical', 'MechanicalConstraintError'),
 'CanonicalG1CollisionPoints': ('contact_solver.collision_geometry', 'CanonicalG1CollisionPoints'),
 'full_geometry_box_collision_penalty': ('contact_solver.collision_geometry',
                                         'full_geometry_box_collision_penalty'),
 'full_geometry_box_ground_penetration': ('contact_solver.collision_geometry',
                                          'full_geometry_box_ground_penetration'),
 'full_geometry_box_penetration': ('contact_solver.collision_geometry', 'full_geometry_box_penetration'),
 'full_geometry_contact_surface_penalty': ('contact_solver.collision_geometry',
                                           'full_geometry_contact_surface_penalty'),
 'InteractionProjectionResult': ('contact_solver.trajectory_projection', 'InteractionProjectionResult'),
 'contact_rotation_losses': ('contact_solver.trajectory_projection', 'contact_rotation_losses'),
 'project_interaction_q_trajectory': ('contact_solver.trajectory_projection',
                                      'project_interaction_q_trajectory'),
 'rollout_dense_collision_loss': ('contact_solver.trajectory_projection', 'rollout_dense_collision_loss'),
 'rollout_ground_collision_loss': ('contact_solver.trajectory_projection', 'rollout_ground_collision_loss')}
_EXPORTS.update({
    'Residual': ('contact_solver.constraint_learning', 'Residual'),
    'AugmentedLagrangian': ('contact_solver.constraint_learning', 'AugmentedLagrangian'),
    'penalty_loss': ('contact_solver.constraint_learning', 'penalty_loss'),
    'ContactTask': ('contact_solver.constraint_residuals', 'ContactTask'),
    'PlaneSurface': ('contact_solver.constraint_residuals', 'PlaneSurface'),
    'FeasibilityFilter': ('contact_solver.feasibility_filter', 'FeasibilityFilter'),
    'ProtectionRule': ('contact_solver.feasibility_filter', 'ProtectionRule'),
})
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
