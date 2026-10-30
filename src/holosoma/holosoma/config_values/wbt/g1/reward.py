"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_ankle_roll_sphere_[1-5]_link$)(?!right_ankle_roll_sphere_[1-5]_link$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)

g1_29dof_wbt_contact_force_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_knee_link$)(?!right_knee_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_ankle_roll_sphere_[1-5]_link$)"
                    "(?!right_ankle_roll_sphere_[1-5]_link$).+$"
                ),
            },
            weight=-0.5,
        ),
        "motion_contact_force_relative_vector_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_contact_force_relative_vector_error_exp",
            params={
                "sigma": 0.5,
                "force_floor": 50.0,
                "contact_threshold": 10.0,
                "part_indices": (0, 1, 2, 3, 4, 5, 6, 7),
            },
            weight=1.0,
        ),
        "motion_contact_force_unexpected_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_contact_force_unexpected_contact",
            params={"contact_threshold": 10.0, "part_indices": (0, 1, 2, 3, 4, 5, 6, 7)},
            weight=-0.5,
        ),
    }
)


_SPARSE_CLIMB_CONTACT_ENDPOINTS = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "right_knee_link",
)

_SPARSE_CLIMB_ENDPOINT_LABELS = ("lf", "rf", "lh", "rh", "lk", "rk")

_sparse_climb_endpoint_position_terms = {
    f"sparse_climb_position_{label}": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:motion_relative_selected_body_position_error_exp",
        params={"sigma": 0.3, "body_names": (body_name,)},
        weight=0.25,
    )
    for label, body_name in zip(_SPARSE_CLIMB_ENDPOINT_LABELS, _SPARSE_CLIMB_CONTACT_ENDPOINTS, strict=True)
}

_sparse_climb_endpoint_orientation_terms = {
    f"sparse_climb_orientation_{label}": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:motion_relative_selected_body_orientation_error_exp",
        params={"sigma": 0.4, "body_names": (body_name,)},
        weight=0.25,
    )
    for label, body_name in zip(_SPARSE_CLIMB_ENDPOINT_LABELS, _SPARSE_CLIMB_CONTACT_ENDPOINTS, strict=True)
}

g1_29dof_wbt_sparse_climb_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Knees are commanded contact-capable parts in the sparse interface;
        # do not simultaneously classify their intended contacts as undesired.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_knee_link$)(?!right_knee_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_ankle_roll_sphere_[1-5]_link$)"
                    "(?!right_ankle_roll_sphere_[1-5]_link$).+$"
                ),
            },
            weight=-0.5,
        ),
        # Per-part kernels prevent one badly tracked active limb from being
        # hidden by five easy support limbs in a single averaged error.
        **_sparse_climb_endpoint_position_terms,
        **_sparse_climb_endpoint_orientation_terms,
        "sparse_climb_contact_match": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_sparse_contact_match",
            params={"contact_threshold": 10.0},
            weight=1.0,
        ),
    }
)

_A2A_REPLACED_TRACKING_TERMS = {
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_lin_vel",
    "motion_global_body_ang_vel",
}

_A2A_PART_NAMES = ("LF", "RF", "LH", "RH")

g1_29dof_wbt_a2a_reward = RewardManagerCfg(
    terms={
        **{
            name: term
            for name, term in g1_29dof_wbt_reward.terms.items()
            if name not in _A2A_REPLACED_TRACKING_TERMS
        },
        **{
            f"motion_relative_body_position_error_exp_active_{part_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp_active_part",
                params={"sigma": 0.3, "part_index": part_index},
                weight=0.25,
            )
            for part_index, part_name in enumerate(_A2A_PART_NAMES)
        },
        **{
            f"motion_relative_body_orientation_error_exp_active_{part_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp_active_part",
                params={"sigma": 0.4, "part_index": part_index},
                weight=0.25,
            )
            for part_index, part_name in enumerate(_A2A_PART_NAMES)
        },
        **{
            f"motion_global_body_lin_vel_active_{part_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel_active_part",
                params={"sigma": 1.0, "part_index": part_index},
                weight=0.25,
            )
            for part_index, part_name in enumerate(_A2A_PART_NAMES)
        },
        **{
            f"motion_global_body_ang_vel_active_{part_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel_active_part",
                params={"sigma": 3.14, "part_index": part_index},
                weight=0.25,
            )
            for part_index, part_name in enumerate(_A2A_PART_NAMES)
        },
        **{
            f"a2a_support_contact_match_{part_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:a2a_support_contact_match_part",
                params={"threshold": 10.0, "part_index": part_index},
                weight=0.25,
            )
            for part_index, part_name in enumerate(_A2A_PART_NAMES)
        },
        "a2a_support_slip": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:a2a_support_slip",
            params={"contact_threshold": 10.0, "include_ang_vel": False},
            weight=-0.2,
        ),
        "a2a_support_force_validity": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:a2a_support_force_validity",
            params={"min_force": 10.0, "max_force": 1500.0, "force_reduce": "max"},
            weight=-0.2,
        ),
        "a2a_unexpected_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:a2a_unexpected_contact",
            params={"threshold": 10.0},
            weight=-0.2,
        ),
    }
)

g1_29dof_wbt_proto_reward = RewardManagerCfg(
    terms={
        "proto_functional": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ProtoFunctionalReward",
            params={
                "root_goal_offset": (0.8, 0.0, 0.45),
                "active_region_center_offset": (0.65, 0.0, 0.75),
                "active_region_half_extents": (0.35, 0.35, 0.10),
                "active_region_from_terrain": True,
                "active_region_grid_size": 5,
                "active_region_surface_mode": "max_height",
                "support_set": ("left_foot", "right_foot"),
                "active_set": ("left_hand", "right_hand"),
                "w_root": 2.0,
                "w_support": 0.5,
                "w_active": 0.75,
                "alpha_forward": 1.0,
                "alpha_height": 1.0,
                "forward_axis": (1.0, 0.0),
                "progress_clip": 0.25,
                "goal_radius_xy": 0.20,
                "goal_radius_z": 0.15,
                "goal_bonus": 5.0,
                "stable_frames_required": 10,
                "root_vel_threshold": 0.35,
                "torso_rp_threshold": 0.5,
                "contact_threshold": 10.0,
                "support_contact_bonus": 1.0,
                "support_lost_penalty": 1.0,
                "support_slip_threshold": 0.20,
                "active_reach_weight": 1.0,
                "active_contact_bonus": 1.0,
                "active_slip_threshold": 0.20,
                "active_valid_contact_frames": 3,
                "use_motion_part_masks": True,
            },
            weight=1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_knee_link$)(?!right_knee_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_ankle_roll_sphere_[1-5]_link$)"
                    "(?!right_ankle_roll_sphere_[1-5]_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)

g1_29dof_wbt_native_force_reward = RewardManagerCfg(terms={
    **g1_29dof_wbt_reward.terms,
    "native_force_tracking": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt_experiment:force_tracking",
        params={"sigma": 100.0}, weight=1.0,
    ),
})

g1_29dof_wbt_native_sparse_reward = RewardManagerCfg(terms={
    **{name: term for name, term in g1_29dof_wbt_sparse_climb_reward.terms.items()
       if name != "sparse_climb_contact_match"},
    "native_contact_match": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt_experiment:contact_match", weight=1.0),
})

g1_29dof_wbt_native_root_contact_reward = RewardManagerCfg(terms={
    **{name: term for name, term in g1_29dof_wbt_native_sparse_reward.terms.items()
       if name in {"motion_global_ref_position_error_exp", "motion_global_ref_orientation_error_exp",
                   "action_rate_l2", "limits_dof_pos", "undesired_contacts"}},
    "native_contact_match": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt_experiment:contact_match", weight=10.0),
})

__all__ = [
    "g1_29dof_wbt_a2a_reward",
    "g1_29dof_wbt_contact_force_reward",
    "g1_29dof_wbt_sparse_climb_reward",
    "g1_29dof_wbt_proto_reward",
    "g1_29dof_wbt_reward",
]
