from __future__ import annotations

from dataclasses import field
from typing import Annotated, Any, List

import tyro

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    global_average_pool: bool = False
    """Apply spatial global-average pooling after the final convolution."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    max_noise_std: float | None = None
    """Maximum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    export_onnx: bool = True
    """Export an ONNX policy artifact whenever a checkpoint is saved."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    actor_finetune_mode: str = "none"
    """Actor update scope: none, ref_q, ref_qd, last_layers, or full_actor."""

    freeze_actor_obs_normalizer: bool = False
    """Keep actor observation normalization statistics fixed during training."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    adaptive_schedule_start_iter: int = 0
    """Learning iteration at which adaptive KL learning-rate control starts."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""
    full_checkpoints: bool = False
    """Deprecated compatibility option; checkpoints always use the ordinary format."""
    warm_start: bool = False
    """Start a new reference experiment from policy/optimizer weights, without old environment state."""

    anchor_kl_checkpoint: str | None = None
    """Optional checkpoint for a frozen reference actor used as an anchor KL prior."""

    anchor_kl_coef: float = 0.0
    """Coefficient for KL(current policy || frozen reference policy) during actor updates."""

    exec_consistency_coef: float = 0.0
    """Coefficient for fixed-scale old-reference to execution-reference action-mean consistency."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    action_clip: float = 10.0
    """Symmetric policy action bound shared by rollout, evaluation, and export."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None


@dataclass(frozen=True)
class KLEarlyStopPPOConfig(PPOConfig):
    """PPO configuration with per-rollout actor KL safeguards."""

    actor_early_stop_kl: float = 0.02
    """Stop actor updates for the current rollout at or above this KL."""

    actor_rollback_kl: float = 0.04
    """Roll back the last actor update at or above this post-update KL."""

    reshuffle_minibatches_each_epoch: bool = True
    """Draw a fresh minibatch permutation for every learning epoch."""


@dataclass(frozen=True)
class TeacherActorConfig:
    """One manually supplied expert and its observation groups in the student environment."""

    checkpoint: str
    manifest: str
    observation_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class DistillConfig:
    """Optional frozen-teacher distillation settings for DistillPPO."""

    enable_kl: bool = False
    """Whether to apply teacher KL/MSE regularization."""

    teacher_checkpoint_path: str | None = None
    """Checkpoint containing a frozen teacher actor state dict."""

    teachers: Annotated[tuple[TeacherActorConfig, ...], tyro.conf.Suppress] = ()
    """Explicit experts for RoutedDistillPPO; no checkpoint discovery or selection."""

    teacher_spec_file: str | None = None
    """JSON list of manually supplied {checkpoint, manifest} experts."""

    motion_schedule_start: int = 1000
    """Iteration at which routed student motion allocation becomes adaptive."""

    probes_per_motion: int = 1
    """Full-motion probes per motion after adaptive allocation starts."""

    lambda_kl_init: float = 0.0
    """Initial teacher-prior curriculum weight."""

    lambda_kl_final: float = 0.0
    """Final teacher-prior curriculum weight after annealing."""

    lambda_kl_anneal_iters: int = 0
    """Number of learning iterations used for teacher-prior curriculum annealing."""

    lambda_kl_anneal_schedule: str = "linear"
    """Teacher-prior curriculum schedule: "linear", "cosine", or "php_parkour"."""

    use_mean_mse_fallback: bool = False
    """Use mean-action MSE instead of Gaussian KL."""

    distill_type: str = "kl"
    """Teacher prior type: "kl", "mse_dagger", or "php_dagger_ppo"."""

    dagger_coef: float = 1.0
    """Multiplier applied to MSE DAgger loss before the DAgger curriculum weight."""

    pure_distill_iters: int | None = None
    """Optional pure-distillation phase, then fixed final DAgger/PPO weights.

    Iterations [0, pure_distill_iters) update only the teacher-supervised actor;
    PPO actor and critic gradients are excluded. None preserves the curriculum.
    """

    teacher_prior_coef: float = 1.0
    """Constant multiplier applied to the teacher prior after the curriculum weight."""

    lambda_dagger_init: float | None = None
    """Initial DAgger curriculum weight. If None, uses lambda_kl_init."""

    lambda_dagger_final: float | None = None
    """Final DAgger curriculum weight. If None, uses lambda_kl_final."""

    lambda_ppo_init: float | None = None
    """Initial PPO actor-loss curriculum weight. If None, uses 1 - lambda_dagger."""

    lambda_ppo_final: float | None = None
    """Final PPO actor-loss curriculum weight. If None, uses 1 - lambda_dagger."""

    kl_action_slice: tuple[int | None, int | None] | None = None
    """Optional action slice used for teacher KL, e.g. (0, 15). Defaults to all shared actions."""

    kl_std_min: float = 1e-6
    """Minimum std used in teacher KL computation."""

    teacher_obs_keys: tuple[str, ...] | None = None
    """Observation keys for the teacher. Defaults to student actor keys."""

    teacher_init_noise_std: float | None = None
    """Teacher actor std fallback if checkpoint does not provide log_std."""

    teacher_hidden_dims: tuple[int, ...] | None = None
    """Optional teacher actor MLP hidden dims. Use when student and teacher architectures differ."""

    teacher_module_type: str | None = None
    """Optional frozen-teacher module type, e.g. ``MLP`` for a CNN student."""

    dagger_disable_on_term_names: tuple[str, ...] = ()
    """Termination terms that mark a sample outside the teacher-valid region for DAgger."""

    dagger_valid_use_bad_tracking: bool = False
    """Use the expert bad-tracking test as a DAgger valid-region mask without terminating the student."""

    dagger_bad_ref_pos_threshold: float = 0.5
    """Expert-valid root position threshold used by the DAgger mask."""

    dagger_bad_ref_ori_threshold: float = 0.8
    """Expert-valid root orientation threshold used by the DAgger mask."""

    dagger_bad_motion_body_pos_threshold: float = 0.25
    """Expert-valid tracked-body position threshold used by the DAgger mask."""

    dagger_bad_motion_body_pos_body_names: tuple[str, ...] = ()
    """Tracked body names checked by the expert-valid DAgger mask."""

    php_student_termination_anneal_iters: int = 0
    """Iterations over which the PHP student's tracking boundary doubles."""


@dataclass(frozen=True)
class DistillPPOConfig(PPOConfig):
    """PPO config variant for standalone teacher distillation."""

    distill: DistillConfig = field(default_factory=DistillConfig)


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class KLEarlyStopPPOAlgoConfig:
    """Configuration wrapper for the KL-guarded PPO comparison."""

    _target_: str
    _recursive_: bool
    config: KLEarlyStopPPOConfig


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = PPOConfig | KLEarlyStopPPOConfig | DistillPPOConfig | FastSACConfig

AlgoConfig = PPOAlgoConfig | KLEarlyStopPPOAlgoConfig | FastSACAlgoConfig
