"""Default randomization manager configurations."""

from holosoma.config_values.wbt.g1.randomization import (
    g1_29dof_wbt_randomization,
    g1_29dof_wbt_randomization_domain_rand,
    g1_29dof_wbt_php_student_randomization,
)

none = None

DEFAULTS = {
    "none": none,
    "g1_29dof_wbt": g1_29dof_wbt_randomization,
    "g1_29dof_wbt_domain_rand": g1_29dof_wbt_randomization_domain_rand,
    "g1_29dof_wbt_php_student": g1_29dof_wbt_php_student_randomization,
}
