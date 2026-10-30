from typing import Annotated

import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_baseline_29,
    g1_29dof_wbt_baseline_29_future_ref,
    g1_29dof_wbt_baseline_29_future_ref_no_qdot,
    g1_29dof_wbt_sparse_climb00_3k,
    g1_29dof_wbt_sparse_climb00_root_contact_resume_3250,
    g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k,
    g1_29dof_wbt_baseline_single,
    g1_29dof_wbt_contact_force,
    g1_29dof_wbt_climb00_php_expert,
    g1_29dof_wbt_climb00_php_student,
)
from holosoma.config_values.wbt.g1.climb_reference_goal import (
    g1_29dof_climb00_reference_goal_24event,
    g1_29dof_climb00_single_full_noref_imitation_2k,
)

DEFAULTS = {
    "g1_29dof_wbt_baseline_single": g1_29dof_wbt_baseline_single,
    "g1_29dof_wbt_baseline_29": g1_29dof_wbt_baseline_29,
    "g1_29dof_wbt_baseline_29_future_ref": g1_29dof_wbt_baseline_29_future_ref,
    "g1_29dof_wbt_baseline_29_future_ref_no_qdot": g1_29dof_wbt_baseline_29_future_ref_no_qdot,
    "g1_29dof_wbt_sparse_climb00_3k": g1_29dof_wbt_sparse_climb00_3k,
    "g1_29dof_wbt_sparse_climb00_root_contact_resume_3250": (
        g1_29dof_wbt_sparse_climb00_root_contact_resume_3250
    ),
    "g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k": (
        g1_29dof_wbt_future_ref_no_qdot_tracking_curriculum_10k
    ),
    "g1_29dof_wbt_contact_force": g1_29dof_wbt_contact_force,
    "g1_29dof_wbt_climb00_php_expert": g1_29dof_wbt_climb00_php_expert,
    "g1_29dof_wbt_climb00_php_student": g1_29dof_wbt_climb00_php_student,
    "g1_29dof_climb00_reference_goal_24event": g1_29dof_climb00_reference_goal_24event,
    "g1_29dof_climb00_single_full_noref_imitation_2k": (
        g1_29dof_climb00_single_full_noref_imitation_2k
    ),
}

from holosoma.config_values.wbt.g1.modular import DEFAULTS as MODULAR_WBT_DEFAULTS

DEFAULTS.update(MODULAR_WBT_DEFAULTS)

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{key.replace('_', '-')}": value for key, value in DEFAULTS.items()}
        )
    ),
]
