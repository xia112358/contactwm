"""Default command manager configurations."""

from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_baseline_29_command,
    g1_29dof_wbt_baseline_single_command,
    g1_29dof_wbt_contact_force_command,
)

none = None

DEFAULTS = {
    "none": none,
    "g1_29dof_wbt_baseline_single": g1_29dof_wbt_baseline_single_command,
    "g1_29dof_wbt_baseline_29": g1_29dof_wbt_baseline_29_command,
    "g1_29dof_wbt_contact_force": g1_29dof_wbt_contact_force_command,
}
