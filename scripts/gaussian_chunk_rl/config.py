"""Independent Gaussian PPO settings; no diffusion likelihood hyperparameters."""

from dataclasses import asdict, dataclass
import math


@dataclass
class GaussianPPOConfig:
    exec_horizon: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    initial_std: float = 0.04
    min_std: float = 0.01
    max_std: float = 0.2
    actor_lr: float = 1e-4
    adaptive_actor_lr: bool = False
    min_actor_lr: float = 1e-7
    kl_probe_size: int = 8192
    critic_lr: float = 1e-3
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    update_epochs: int = 5
    minibatch_size: int = 8192
    critic_minibatch_size: int = 4096
    critic_warmup_iterations: int = 10
    max_grad_norm: float = 1.0
    value_coef: float = 0.5

    def validate(self, *, prediction_horizon: int, execution_offset: int) -> None:
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("PPO configuration must be finite.")
        if not 1 <= self.exec_horizon <= prediction_horizon - execution_offset:
            raise ValueError("exec_horizon exceeds the executable checkpoint horizon.")
        if self.gamma != 1.0 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Use gamma=1 for the endpoint reward and lambda in [0,1].")
        if not 0 < self.min_std <= self.initial_std <= self.max_std:
            raise ValueError("Require 0 < min_std <= initial_std <= max_std.")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0,1).")
        for name in ("actor_lr", "critic_lr", "target_kl", "max_grad_norm", "value_coef"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(self.adaptive_actor_lr, bool):
            raise ValueError("adaptive_actor_lr must be a boolean.")
        if self.min_actor_lr <= 0 or (self.adaptive_actor_lr and self.min_actor_lr > self.actor_lr):
            raise ValueError("Require a positive min_actor_lr no larger than the adaptive starting LR.")
        for name in ("exec_horizon", "update_epochs", "minibatch_size", "critic_minibatch_size", "kl_probe_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.critic_warmup_iterations, int) or self.critic_warmup_iterations < 0:
            raise ValueError("critic_warmup_iterations must be a nonnegative integer.")

    def to_dict(self) -> dict:
        return asdict(self)
