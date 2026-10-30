"""Whole Body Tracking curriculum presets for the G1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

g1_29dof_wbt_tracking_precision_curriculum_10k = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        **g1_29dof_wbt_curriculum.setup_terms,
    },
    reset_terms={},
    step_terms={
        "tracking_precision": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.wbt:TrackingPrecisionCurriculum",
            params={
                "probe_quantile": 0.90,
                "probe_history_size": 100,
                "min_probe_episodes": 10,
                "base_body_threshold": 0.30,
                "body_threshold_bounds": (0.05, 0.30),
                "base_root_pos_threshold": 0.50,
                "root_pos_threshold_bounds": (0.10, 0.50),
                "base_root_ori_threshold": 0.50,
                "root_ori_threshold_bounds": (0.15, 0.50),
                "base_position_sigma": 0.15,
                "position_sigma_bounds": (0.05, 0.15),
                "base_reset_noise_scale": 1.0,
                "reset_noise_scale_bounds": (1.0 / 6.0, 1.0),
            },
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_curriculum",
    "g1_29dof_wbt_tracking_precision_curriculum_10k",
]
