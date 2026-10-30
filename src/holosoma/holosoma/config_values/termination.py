"""Default termination manager configurations."""

from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_a2a_pure_termination,
    g1_29dof_wbt_proto_termination,
    g1_29dof_wbt_termination,
    g1_29dof_wbt_timeout_only_termination,
    g1_29dof_wbt_php_student_termination,
)

none = None

DEFAULTS = {
    "none": none,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_a2a_pure": g1_29dof_wbt_a2a_pure_termination,
    "g1_29dof_wbt_proto": g1_29dof_wbt_proto_termination,
    "g1_29dof_wbt_timeout_only": g1_29dof_wbt_timeout_only_termination,
    "g1_29dof_wbt_php_student": g1_29dof_wbt_php_student_termination,
}
