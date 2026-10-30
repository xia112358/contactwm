"""Default curriculum manager configurations."""

from holosoma.config_values.wbt.g1.curriculum import (
    g1_29dof_wbt_curriculum,
    g1_29dof_wbt_tracking_precision_curriculum_10k,
)

none = None

DEFAULTS = {
    "none": none,
    "g1_29dof_wbt_curriculum": g1_29dof_wbt_curriculum,
    "g1_29dof_wbt_tracking_precision_curriculum_10k": g1_29dof_wbt_tracking_precision_curriculum_10k,
}
