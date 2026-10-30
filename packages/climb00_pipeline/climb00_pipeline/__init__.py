"""Compatibility exports; implementations live in generator/contact_solver/somaforge_core."""
from importlib import import_module

_EXPORTS = {'ContactQPrediction': ('generator.contact_q_generator', 'ContactQPrediction'),
 'ContactTransitionPlan': ('generator.contact_q_generator', 'ContactTransitionPlan'),
 'SequentialContactQPredictor': ('generator.contact_q_generator', 'SequentialContactQPredictor'),
 'BODY_NAMES': ('somaforge_core.motion_contracts', 'BODY_NAMES'),
 'CONTACT_BODY_INDEX': ('somaforge_core.motion_contracts', 'CONTACT_BODY_INDEX'),
 'CONTACT_PARTS': ('somaforge_core.motion_contracts', 'CONTACT_PARTS'),
 'InteractionBoundary': ('somaforge_core.motion_contracts', 'InteractionBoundary'),
 'SparseKeyframe': ('somaforge_core.motion_contracts', 'SparseKeyframe'),
 'TeacherSegment': ('somaforge_core.motion_contracts', 'TeacherSegment'),
 'ConstrainedKeypointResult': ('somaforge_core.motion_trajectory', 'ConstrainedKeypointResult'),
 'FullBodyCollisionAudit': ('somaforge_core.motion_trajectory', 'FullBodyCollisionAudit'),
 'FullBodyTeacherSegment': ('somaforge_core.motion_trajectory', 'FullBodyTeacherSegment'),
 'FullBodyTrajectory': ('somaforge_core.motion_trajectory', 'FullBodyTrajectory'),
 'KeypointTrajectory': ('somaforge_core.motion_trajectory', 'KeypointTrajectory'),
 'G1MechanicalProjector': ('contact_solver.mechanical', 'G1MechanicalProjector'),
 'MechanicalConstraintError': ('contact_solver.mechanical', 'MechanicalConstraintError'),
 'CanonicalG1CollisionPoints': ('contact_solver.collision_geometry', 'CanonicalG1CollisionPoints'),
 'CanonicalG1ForwardKinematics': ('somaforge_core.g1_kinematics', 'CanonicalG1ForwardKinematics'),
 'full_geometry_box_collision_penalty': ('contact_solver.collision_geometry',
                                         'full_geometry_box_collision_penalty'),
 'full_geometry_box_ground_penetration': ('contact_solver.collision_geometry',
                                          'full_geometry_box_ground_penetration'),
 'full_geometry_box_penetration': ('contact_solver.collision_geometry', 'full_geometry_box_penetration'),
 'full_geometry_contact_surface_penalty': ('contact_solver.collision_geometry',
                                           'full_geometry_contact_surface_penalty'),
 'TeacherDataset': ('generator.teacher_data', 'TeacherDataset'),
 'InteractionProjectionResult': ('contact_solver.trajectory_projection', 'InteractionProjectionResult'),
 'UnifiedInteractionLoss': ('generator.unified_interaction', 'UnifiedInteractionLoss'),
 'UnifiedInteractionPrediction': ('somaforge_core.prediction_contracts', 'UnifiedInteractionPrediction'),
 'UnifiedInteractionPredictor': ('generator.unified_interaction', 'UnifiedInteractionPredictor'),
 'contact_rotation_losses': ('contact_solver.trajectory_projection', 'contact_rotation_losses'),
 'project_interaction_q_trajectory': ('contact_solver.trajectory_projection',
                                      'project_interaction_q_trajectory'),
 'rollout_dense_collision_loss': ('contact_solver.trajectory_projection', 'rollout_dense_collision_loss'),
 'rollout_ground_collision_loss': ('contact_solver.trajectory_projection', 'rollout_ground_collision_loss'),
 'unified_interaction_loss': ('generator.unified_interaction', 'unified_interaction_loss'),
 'TrajectoryPrediction': ('somaforge_core.prediction_contracts', 'TrajectoryPrediction')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

# Support historical scripts that add only packages/climb00_pipeline to sys.path.
from pathlib import Path
import sys
_packages = Path(__file__).resolve().parents[2]
for _name in ("somaforge_core", "motion_edit", "contact_solver", "generator"):
    _path = _packages / _name
    if (_path / _name / "__init__.py").is_file() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
