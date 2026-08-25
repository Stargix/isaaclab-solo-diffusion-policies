"""Validated configuration for diffusion-policy PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DPPOConfig:
    """Hyperparameters whose units match one executed action chunk."""

    inference_steps: int = 10
    finetune_denoising_steps: int = 5
    exec_horizon: int = 4
    min_denoising_std: float = 0.10
    # Finite episodic task with bounded difference rewards. Gamma=1 preserves
    # their exact endpoint semantics instead of adding an implicit pace cost.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    gamma_denoising: float = 0.99
    actor_lr: float = 1.0e-5
    critic_lr: float = 1.0e-3
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 0.0
    clip_ratio_base: float = 1.0e-3
    clip_ratio: float = 1.0e-2
    clip_ratio_rate: float = 3.0
    value_clip: float | None = None
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 5
    minibatch_size: int = 8192
    critic_minibatch_size: int = 4096
    critic_warmup_iterations: int = 10
    target_kl: float = 0.02
    # Optional transition-kernel KL to the immutable actor loaded at the start
    # of a run.  Zero preserves canonical DPPO/backwards compatibility; the
    # path-speed v3 experiment enables it explicitly to prevent cumulative
    # drift away from the stable path_1 gait.
    reference_kl_coef: float = 0.0
    # Keep rare terminal outcomes intact. In a large vectorized batch, a
    # symmetric 1 % quantile clip can erase every success/fall when its event
    # rate is below 1 %.
    advantage_clip_quantile: float = 0.0

    def validate(self, *, prediction_horizon: int, execution_offset: int) -> None:
        if self.inference_steps < 2:
            raise ValueError("DPPO requires at least two denoising steps.")
        if not 1 <= self.finetune_denoising_steps <= self.inference_steps:
            raise ValueError("finetune_denoising_steps must be in [1, inference_steps].")
        if not 1 <= self.exec_horizon <= prediction_horizon - execution_offset:
            raise ValueError("exec_horizon exceeds the checkpoint's executable horizon.")
        if self.min_denoising_std <= 0.0:
            raise ValueError("min_denoising_std must be positive so every trained transition has a density.")
        if self.gamma != 1.0:
            raise ValueError(
                "gamma must be 1.0: the bounded difference rewards encode endpoint objectives."
            )
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if not 0.0 < self.gamma_denoising <= 1.0:
            raise ValueError("gamma_denoising must be in (0, 1].")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive.")
        if not 0.0 < self.clip_ratio_base <= self.clip_ratio:
            raise ValueError("clip_ratio_base must be positive and no larger than clip_ratio.")
        if self.clip_ratio_rate <= 0.0:
            raise ValueError("clip_ratio_rate must be positive.")
        if self.value_clip is not None and self.value_clip <= 0.0:
            raise ValueError("value_clip must be positive when enabled.")
        if self.value_coef <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("value_coef and max_grad_norm must be positive.")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive.")
        if self.reference_kl_coef < 0.0:
            raise ValueError("reference_kl_coef must be non-negative.")
        if self.update_epochs < 1 or self.minibatch_size < 1 or self.critic_minibatch_size < 1:
            raise ValueError("update epochs and minibatch sizes must be positive.")
        if self.critic_warmup_iterations < 0:
            raise ValueError("critic_warmup_iterations must be non-negative.")
        if not 0.0 <= self.advantage_clip_quantile < 0.5:
            raise ValueError("advantage_clip_quantile must be in [0, 0.5).")

    def to_dict(self) -> dict:
        return asdict(self)
