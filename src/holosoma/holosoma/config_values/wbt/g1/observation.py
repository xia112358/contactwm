"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
            scale=1.0,
            noise=0.25,
        ),
        "pelvis_global_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:pelvis_global_pos",
            scale=1.0,
            noise=0.0,
        ),
        "pelvis_global_lin_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:pelvis_global_lin_vel",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "robot_body_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "robot_body_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "base_lin_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_lin_vel",
            scale=1.0,
            noise=0.0,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "terrain_height_scan": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:terrain_height_scan",
            scale=1.0,
            noise=0.0,
            clip=(-1.0, 1.0),
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "pelvis_global_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:pelvis_global_pos",
        scale=1.0,
        noise=0.0,
    ),
    "pelvis_global_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:pelvis_global_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "terrain_height_scan": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:terrain_height_scan",
        scale=1.0,
        noise=0.05,
        clip=(-1.0, 1.0),
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

FUTURE_REF_OFFSETS = (1, 2, 4, 8)
CURRENT_REF_TERM_NAMES = ("motion_command", "motion_ref_pos_b", "motion_ref_ori_b")

future_ref_terms = {}
for offset in FUTURE_REF_OFFSETS:
    # Keep each future frame identical to the current reference contract and
    # adjacent in the flattened observation: [q, qd, torso_pos, torso_rot6d].
    future_ref_terms.update(
        {
            f"future_t{offset}_motion_command": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:future_motion_command",
                params={"offset": offset},
                scale=1.0,
                noise=0.0,
            ),
            f"future_t{offset}_motion_ref_pos_b": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:future_motion_ref_pos_b",
                params={"offset": offset},
                scale=1.0,
                noise=0.25,
            ),
            f"future_t{offset}_motion_ref_ori_b": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:future_motion_ref_ori_b",
                params={"offset": offset},
                scale=1.0,
                noise=0.05,
            ),
        }
    )

actor_obs_future_ref_terms = {
    "00_reference_window": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:policy_reference_window",
        params={"include_current": False, "add_noise": True},
        scale=1.0,
        noise=0.0,
    ),
    **{name: term for name, term in actor_obs_shared.terms.items() if name not in CURRENT_REF_TERM_NAMES},
}

critic_obs_future_ref_terms = {
    "00_reference_window": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:policy_reference_window",
        params={"include_current": True, "add_noise": False},
        scale=1.0,
        noise=0.0,
    ),
    **{name: term for name, term in critic_obs_shared_terms.items() if name not in CURRENT_REF_TERM_NAMES},
}

actor_obs_future_ref_no_qdot_terms = {
    "00_reference_window": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:policy_reference_window",
        params={"include_current": False, "add_noise": True, "include_joint_velocity": False},
        scale=1.0,
        noise=0.0,
    ),
    **{name: term for name, term in actor_obs_shared.terms.items() if name not in CURRENT_REF_TERM_NAMES},
}

critic_obs_future_ref_no_qdot_terms = {
    "00_reference_window": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:policy_reference_window",
        params={"include_current": True, "add_noise": False, "include_joint_velocity": False},
        scale=1.0,
        noise=0.0,
    ),
    **{name: term for name, term in critic_obs_shared_terms.items() if name not in CURRENT_REF_TERM_NAMES},
}

sparse_climb_actor_terms = {
    "00_sparse_climb_reference_window": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:sparse_climb_reference_window",
        params={"add_noise": True},
        scale=1.0,
        noise=0.0,
    ),
    # Current full-body geometry and terrain remain single-frame observations.
    **{
        name: actor_obs_shared.terms[name]
        for name in (
            "pelvis_global_pos",
            "pelvis_global_lin_vel",
            "robot_body_pos_b",
            "robot_body_ori_b",
            "terrain_height_scan",
        )
    },
}

sparse_climb_actor_proprio_history_terms = {
    "projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity",
        scale=1.0,
        noise=0.05,
    ),
    **{
        name: actor_obs_shared.terms[name]
        for name in ("base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel", "actions")
    },
}

a2a_limb_state_term = {
    "limb_state": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:limb_state",
        scale=1.0,
        noise=0.0,
    ),
}

contact_force_ref_term = {
    "contact_force_ref_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:contact_force_ref_b",
        params={"force_scale": 300.0},
        scale=1.0,
        noise=0.0,
    ),
}

actor_obs_a2a_terms = {
    **actor_obs_shared.terms,
    **a2a_limb_state_term,
}

critic_obs_a2a_terms = {
    **critic_obs_shared_terms,
    **a2a_limb_state_term,
}

actor_obs_contact_force_terms = {
    **actor_obs_shared.terms,
    **contact_force_ref_term,
}

critic_obs_contact_force_terms = {
    **critic_obs_shared_terms,
    **contact_force_ref_term,
}

pure_a2a_actor_obs_terms = {
    "a2a_root_ref": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:a2a_root_ref",
        scale=1.0,
        noise=0.0,
    ),
    "a2a_active_limb_ref": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:a2a_active_limb_ref",
        scale=1.0,
        noise=0.0,
    ),
    **a2a_limb_state_term,
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

pure_a2a_critic_obs_terms = pure_a2a_actor_obs_terms

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_a2a_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=actor_obs_a2a_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_a2a_terms,
        ),
    },
)

g1_29dof_wbt_future_ref_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=actor_obs_future_ref_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_future_ref_terms,
        ),
    },
)

g1_29dof_wbt_future_ref_no_qdot_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=actor_obs_future_ref_no_qdot_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_future_ref_no_qdot_terms,
        ),
    },
)

g1_29dof_wbt_sparse_climb_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=sparse_climb_actor_terms,
        ),
        "actor_proprio_history": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=5,
            terms=sparse_climb_actor_proprio_history_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_future_ref_no_qdot_terms,
        ),
    },
)

g1_29dof_wbt_contact_force_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=actor_obs_contact_force_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_contact_force_terms,
        ),
    },
)

g1_29dof_wbt_a2a_pure_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=pure_a2a_actor_obs_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=pure_a2a_critic_obs_terms,
        ),
    },
)

php_student_actor_terms = {
    "velocity_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:php_velocity_command",
        params={"command": (1.0, 0.0)},
    ),
    "projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", noise=0.05
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel", noise=0.2
    ),
    "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", noise=0.01),
    "dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", noise=0.5),
    "actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions"),
}

php_expert_actor_terms = {
    name: actor_obs_shared.terms[name]
    for name in (
        "motion_command",
        "motion_ref_pos_b",
        "motion_ref_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "terrain_height_scan",
        "dof_pos",
        "dof_vel",
        "actions",
    )
}

g1_29dof_wbt_php_expert_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=php_expert_actor_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

g1_29dof_wbt_php_student_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=php_student_actor_terms
        ),
        "depth_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "depth": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt:PHPDepthImage",
                    params={
                        "max_distance": 5.0,
                        "offset_range": (-0.03, 0.03),
                        "gaussian_std": 0.03,
                        "delay_steps": (3, 4),
                    },
                    scale=0.2,
                    clip=(0.0, 1.0),
                )
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
        "teacher_actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=php_expert_actor_terms
        ),
    }
)

# Independently composable native-contact observation blocks. Reference targets
# occur only in expert blocks; all student input functions are checked at launch.
g1_29dof_wbt_sparse_native_observation = replace(
    g1_29dof_wbt_sparse_climb_observation,
    groups={
        **g1_29dof_wbt_sparse_climb_observation.groups,
        "actor_obs": replace(
            g1_29dof_wbt_sparse_climb_observation.groups["actor_obs"],
            terms={
                **g1_29dof_wbt_sparse_climb_observation.groups["actor_obs"].terms,
                "00_sparse_climb_reference_window": ObsTermCfg(
                    func="holosoma.managers.observation.terms.wbt_experiment:sparse_reference",
                    params={"add_noise": True},
                ),
            },
        ),
    },
)

g1_29dof_wbt_native_force_observation = replace(
    g1_29dof_wbt_observation,
    groups={name: replace(group, terms={**group.terms,
        "reference_force": ObsTermCfg(func="holosoma.managers.observation.terms.wbt_experiment:reference_force"),
        "actual_contacts": ObsTermCfg(func="holosoma.managers.observation.terms.wbt_experiment:actual_contacts"),
    }) for name, group in g1_29dof_wbt_observation.groups.items()},
)

g1_29dof_wbt_sparse_native_force_observation = replace(
    g1_29dof_wbt_sparse_native_observation,
    groups={**g1_29dof_wbt_sparse_native_observation.groups,
        "actor_obs": replace(g1_29dof_wbt_sparse_native_observation.groups["actor_obs"], terms={
            **g1_29dof_wbt_sparse_native_observation.groups["actor_obs"].terms,
            "reference_force": ObsTermCfg(func="holosoma.managers.observation.terms.wbt_experiment:reference_force"),
            "actual_contacts": ObsTermCfg(func="holosoma.managers.observation.terms.wbt_experiment:actual_contacts"),
        })},
)


def student_contact_observation(mode):
    terms = {**php_student_actor_terms,
        "velocity_command": ObsTermCfg(func="holosoma.managers.observation.terms.wbt_experiment:velocity_command",
            params={"source": "dataset", "command": (1., 0.)}),
    }
    if mode != "none":
        terms["actual_contacts"] = ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt_experiment:actual_contacts", params={"mode": mode})
    return ObservationManagerCfg(groups={
        "actor_obs": ObsGroupCfg(terms=terms, enable_noise=True),
        "depth_obs": g1_29dof_wbt_php_student_observation.groups["depth_obs"],
        "critic_obs": g1_29dof_wbt_observation.groups["critic_obs"],
    })


g1_29dof_wbt_student_contact_state_observation = student_contact_observation("state")
g1_29dof_wbt_student_contact_force_observation = student_contact_observation("state_force")
g1_29dof_wbt_student_proprio_depth_observation = student_contact_observation("none")

__all__ = [
    "CURRENT_REF_TERM_NAMES",
    "FUTURE_REF_OFFSETS",
    "g1_29dof_wbt_a2a_observation",
    "g1_29dof_wbt_a2a_pure_observation",
    "g1_29dof_wbt_contact_force_observation",
    "g1_29dof_wbt_future_ref_observation",
    "g1_29dof_wbt_future_ref_no_qdot_observation",
    "g1_29dof_wbt_sparse_climb_observation",
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_php_student_observation",
    "g1_29dof_wbt_php_expert_observation",
]
