"""Motion predictors, datasets, and prediction training."""
from importlib import import_module

_EXPORTS = {'ContactQPrediction': ('generator.contact_q_generator', 'ContactQPrediction'),
 'ContactTransitionPlan': ('generator.contact_q_generator', 'ContactTransitionPlan'),
 'SequentialContactQPredictor': ('generator.contact_q_generator', 'SequentialContactQPredictor'),
 'TeacherDataset': ('generator.teacher_data', 'TeacherDataset'),
 'UnifiedInteractionLoss': ('generator.unified_interaction', 'UnifiedInteractionLoss'),
 'UnifiedInteractionPredictor': ('generator.unified_interaction', 'UnifiedInteractionPredictor'),
 'unified_interaction_loss': ('generator.unified_interaction', 'unified_interaction_loss')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
