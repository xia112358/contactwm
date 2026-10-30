"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    motion_manifest: str = ""
    """JSON/YAML manifest that binds motion files to terrain ids.
    When non-empty, takes precedence over motion_dir and motion_file."""

    # motion sampling related
    reset_sampler: Literal[
        "uniform",
        "adaptive",
        "failure_window",
        "completion_recent30_failure_window",
        "completion_ema_failure_window",
        "hotspot_failure_window",
    ] = "uniform"
    """Reset timestep sampler: uniform RSI, adaptive bins, or failure-window variants."""

    canonicalize_motion_order_on_load: bool = False
    """Reorder motion tensors to simulator body/joint order during load instead of at each property access."""

    fixed_motion_assignment_in_eval: bool = False
    """Assign eval environment i to motion i % motion_count, including resets.
    Independent complete recordings do not require segment chaining.
    """

    touchdown_lift_min_window_frames: int = 2
    """Minimum proto bin length, in motion frames, to use for reset sampling."""

    failure_window_pre_frames: int = 50
    """Number of frames before a bad-tracking failure frame to sample from."""

    failure_window_post_frames: int = 20
    """Number of frames after a bad-tracking failure frame to sample from."""

    failure_window_before_prob: float = 0.7
    """Probability of sampling from the pre-failure side of the failure window."""

    failure_window_log_bin_frames: int = 25
    """Frame width used only for logging failure-frame histogram stats."""

    failure_window_success_horizon_frames: int = 50
    """Internal-reset a failure-window retry after it runs this many frames past the original failure frame."""

    hotspot_failure_uniform_mix: float = 0.3
    """Uniform-RSI mixture for completion-EMA failure-window resets."""

    hotspot_failure_decay: float = 0.995
    """Deprecated compatibility field; completion-EMA replay does not accumulate hotspot weights."""

    hotspot_failure_min_count: float = 1.0
    """Deprecated compatibility field; completion-EMA replay starts after probe statistics exist."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    use_completion_learning_sampler: bool = False
    """Adapt normal motion sampling using ordinary start-to-end training episodes instead of reserved probe envs."""

    completion_success_streak_threshold: int = 3
    """Number of consecutive start-to-end completions required before a motion is treated as learned."""

    completion_learned_replay_weight: float = 0.1
    """Relative sampling weight retained for learned motions so they are replayed enough to resist forgetting."""

    completion_weight_beta: float = 0.05
    """EMA update rate for normal-env motion weights derived from completion-learning learned masks."""

    use_start_probe_envs: bool = True
    """Reserve motion-balanced probe environments for multi-motion commands.
    Probe environments always reset from each motion's first frame and are used to estimate per-motion
    start-to-end success rates and adapt normal-env motion sampling. Single-motion commands ignore this default."""

    use_group_probe_envs: bool = False
    """Reserve group-balanced probe environments instead of per-motion start probes.
    Group probes estimate difficulty for a terrain/source-motion group and sample a random motion inside the group on
    each reset, which is cheaper for motion-edit jitter sets than assigning probes to every generated motion."""

    group_probe_by: Literal["terrain_id", "climb_id"] = "terrain_id"
    """Grouping key for group probes. terrain_id uses the motion manifest binding; climb_id parses climb_XX names."""

    probe_env_per_group: int = 8
    """Number of fixed from-zero probe environments assigned to each motion group when use_group_probe_envs is True."""

    group_variant_sample_count: int = 0
    """When positive, sample from at most this many random motions per group in each reset batch.
    This keeps full motion-edit manifests loaded while limiting simultaneous variant diversity per terrain group."""

    chain_motion_segments: bool = False
    """When a sampled motion segment ends, continue into the next consecutive segment without resetting the robot.
    This preserves natural motion continuity for normal training environments. Probe environments still reset to
    their assigned segment starts so they can estimate per-motion completion."""

    hold_at_motion_end_in_eval: bool = False
    """During evaluation only, hold commands at the final motion frame instead of resetting when the motion ends."""

    local_motion_segment_reference: bool = False
    """During evaluation, treat each motion segment as a local template anchored at the robot state on segment entry."""

    relative_reference_rotation: Literal["yaw", "full"] = "yaw"
    """Rotation alignment used when adapting motion body targets to the current robot reference frame."""

    require_chain_boundary_success: bool = True
    """When chaining consecutive motion segments, require boundary tracking/contact checks before switching segments."""

    chain_gate_before_transition: bool = False
    """If True, require boundary success before switching to the next chained motion segment."""

    chain_boundary_contact_threshold: float = 10.0
    """Minimum contact force for an expected support limb to count as established at a chained segment boundary."""

    chain_boundary_tracking_threshold: float = 0.25
    """Maximum z tracking error for key limbs at a chained segment boundary."""

    chain_transition_grace_steps: int = 8
    """Number of steps after chaining into the next segment allowed for contact/tracking to settle."""

    chain_transition_success_window: int = 3
    """Number of consecutive successful post-chain checks required to accept the transition."""

    chain_require_touchdown_completion: bool = False
    """If True, hold at a chained segment tail until time, tracking/contact, and expected touchdown are satisfied."""

    chain_completion_exit_window_steps: int = 5
    """Number of frames before segment end where expected touchdown events may be latched."""

    chain_completion_max_hold_steps: int = 10
    """Maximum frames to hold at a segment tail waiting for touchdown completion before resetting."""

    chain_touchdown_free_window_steps: int = 3
    """A part must be contact-free for this many recent frames before a touchdown edge can latch."""

    chain_touchdown_stable_steps: int = 2
    """A part must be in contact for this many consecutive frames to count as touchdown."""

    event_token_plan: str = ""
    """Optional contact event-token plan used for event-triggered primitive switching."""

    event_token_contact_threshold: float = 10.0
    """Minimum contact force for a token target limb rising edge to trigger the next token."""

    event_token_timeout_margin_frames: int = 10
    """Additional frames after the observed token end before timeout fallback switches tokens."""

    probe_env_per_motion: int = 3
    """Number of fixed from-zero probe environments assigned to each motion when use_start_probe_envs is True."""

    probe_completion_alpha: float = 0.02
    """EMA rate for the separate ordinary completion-learning sampler.
    Start/group probes use a rolling window of 30 finished episodes.
    """

    probe_weight_beta: float = 0.05
    """EMA update rate for normal-env motion sampling weights derived from start-to-end probe success."""

    probe_uniform_mix: float = 0.4
    """Uniform mixture in adapted normal-env motion weights. The rest prioritizes probe difficulty."""

    probe_priority_temperature: float = 0.2
    """Softmax temperature for converting per-motion probe difficulty into sampling priority."""

    probe_fail_bin_count: int = 32
    """Number of phase bins used to record from-zero probe failure locations."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
