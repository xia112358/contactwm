"""Default reward manager configurations."""

from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_a2a_reward,
    g1_29dof_wbt_contact_force_reward,
    g1_29dof_wbt_proto_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_sparse_climb_reward,
    g1_29dof_wbt_native_force_reward,
    g1_29dof_wbt_native_sparse_reward,
    g1_29dof_wbt_native_root_contact_reward,
)

none = None

DEFAULTS = {
    "wbt_native_force": g1_29dof_wbt_native_force_reward,
    "wbt_native_sparse": g1_29dof_wbt_native_sparse_reward,
    "wbt_native_root_contact": g1_29dof_wbt_native_root_contact_reward,
    "none": none,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_a2a": g1_29dof_wbt_a2a_reward,
    "g1_29dof_wbt_contact_force": g1_29dof_wbt_contact_force_reward,
    "g1_29dof_wbt_sparse_climb": g1_29dof_wbt_sparse_climb_reward,
    "g1_29dof_wbt_proto": g1_29dof_wbt_proto_reward,
}
