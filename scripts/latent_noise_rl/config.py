"""Pilot-specific settings kept separate from the numeric PPO dataclass."""

from dataclasses import asdict, dataclass
import math

from scripts.gaussian_chunk_rl.config import GaussianPPOConfig


@dataclass(frozen=True)
class LatentPolicyConfig:
    hidden_dims: tuple[int, int] = (256, 256)
    noise_rows: int = 12
    noise_dim: int = 12
    noise_scale: float = 1.5
    initial_std: float = 1.0
    min_std: float = 0.1
    max_std: float = 2.0
    inference_steps: int = 10
    eta: float = 0.0

    def validate(self, *, prediction_horizon: int, execution_offset: int, action_dim: int) -> None:
        if self.noise_rows != execution_offset + 4 or self.noise_dim != action_dim:
            raise ValueError("Latent dimensions must cover the causal prefix through the four executed actions.")
        if not 0 < self.noise_rows < prediction_horizon:
            raise ValueError("The latent prefix must leave a non-empty unused suffix.")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise_scale must be finite and positive.")
        if not 0 < self.min_std <= self.initial_std <= self.max_std:
            raise ValueError("Require 0 < min_std <= initial_std <= max_std.")
        if self.inference_steps < 1 or self.eta != 0.0:
            raise ValueError("The pilot requires a positive DDIM step count and eta=0.")
        if len(self.hidden_dims) != 2 or any(width < 1 for width in self.hidden_dims):
            raise ValueError("The pilot actor must have two positive hidden widths.")

    def to_dict(self) -> dict:
        return asdict(self)


def pilot_ppo_config(*, smoke: bool = False) -> GaussianPPOConfig:
    """Return the frozen, explicit PPO budget for the latent pilot."""
    return GaussianPPOConfig(
        exec_horizon=4,
        gamma=1.0,
        gae_lambda=0.95,
        initial_std=1.0,
        min_std=0.1,
        max_std=2.0,
        actor_lr=1.0e-5,
        adaptive_actor_lr=True,
        min_actor_lr=1.0e-7,
        kl_probe_size=8192,
        critic_lr=1.0e-3,
        clip_ratio=0.2,
        target_kl=0.02,
        update_epochs=5,
        minibatch_size=8192,
        critic_minibatch_size=4096,
        critic_warmup_iterations=0 if smoke else 10,
        max_grad_norm=1.0,
        value_coef=0.5,
    )
