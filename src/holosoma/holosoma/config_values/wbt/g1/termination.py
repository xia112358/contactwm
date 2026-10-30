"""Whole Body Tracking termination presets for the G1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:start_probe_aware_timeout_exceeded",
            is_timeout=True,
            params={
                "probe_margin_steps": 2,
            },
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.3,
                "bad_ref_ori_threshold": 0.5,
                "bad_motion_body_pos_threshold": 0.15,
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

g1_29dof_wbt_a2a_pure_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": g1_29dof_wbt_termination.terms["bad_tracking"],
        "a2a_support_slip": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:A2ASupportSlip",
            params={
                "contact_threshold": 10.0,
                "slip_threshold": 0.5,
                "part_indices": (0, 1, 2, 3),
                "use_xy_only": True,
            },
        ),
        "a2a_support_contact_mismatch": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:A2ASupportContactMismatch",
            params={
                "contact_threshold": 10.0,
                "grace_steps": 5,
                "part_indices": (0, 1, 2, 3),
            },
        ),
    }
)

g1_29dof_wbt_proto_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "base_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:base_height_below_threshold",
            params={
                "min_height": 0.25,
            },
        ),
        "undesired_contacts": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_knee_link$)(?!right_knee_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_ankle_roll_sphere_[1-5]_link$)(?!right_ankle_roll_sphere_[1-5]_link$).+$"
                ),
            },
        ),
    }
)

g1_29dof_wbt_timeout_only_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
    }
)

g1_29dof_wbt_php_student_termination = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_a2a_pure_termination",
    "g1_29dof_wbt_proto_termination",
    "g1_29dof_wbt_php_student_termination",
    "g1_29dof_wbt_timeout_only_termination",
]
