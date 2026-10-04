"""Left/right augmentation of the route-conditioned diffusion MDP.

This uses the augmented-behaviour-policy ratio (Mittal et al., Eq. 6):
the denominator and advantages belong to the ORIGINAL sampled transition.
It does not recompute a tiny old density on a counterfactual reflected state.
As in symmetry PPO, ignoring the state-distribution ratio is an approximation;
the reflected-policy mismatch must be monitored, especially for pretrained actors.
"""

from __future__ import annotations

import torch

from scripts.diffusion_policy.train.data.symmetry import (
    LEFT_RIGHT_PERM,
    LEFT_RIGHT_SIGN,
    apply_symmetry,
)


class LeftRightAugmentation:
    """Transform raw histories and every latent action token consistently.

    With reflection-closed min/max ranges, action normalization commutes with
    this signed permutation. Hence the latent map is orthogonal, |det|=1,
    Gaussian noise is preserved and clipping commutes. Reject other ranges
    rather than silently translating/scaling noisy diffusion latents.
    """

    def __init__(self, policy):
        self.policy = policy
        cfg = policy.cfg
        if (cfg.proprio_dim, cfg.action_hist_dim, cfg.action_dim) != (30, 12, 12):
            raise ValueError("Mirror DPPO requires Solo12 raw 30-D proprioception and 12-D actions.")
        if cfg.goal_dim not in (12, 16):
            raise ValueError("Mirror DPPO requires geometric goal12 or height-profile goal16.")
        stats = policy.normalizer_stats
        if stats is None:
            raise ValueError("Load checkpoint normalizers before constructing mirror DPPO.")
        low = torch.as_tensor(stats.action.min, dtype=torch.float64)
        high = torch.as_tensor(stats.action.max, dtype=torch.float64)
        if low.shape != (12,) or high.shape != (12,):
            raise ValueError("Mirror DPPO requires twelve action normalization ranges.")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not (high > low).all():
            raise ValueError("Invalid action normalization ranges for mirror DPPO.")
        perm, sign = LEFT_RIGHT_PERM, LEFT_RIGHT_SIGN.to(low)
        reflected_low = torch.minimum(low[perm] * sign, high[perm] * sign)
        reflected_high = torch.maximum(low[perm] * sign, high[perm] * sign)
        if not (
            torch.allclose(low, reflected_low, atol=1e-6, rtol=1e-5)
            and torch.allclose(high, reflected_high, atol=1e-6, rtol=1e-5)
        ):
            raise ValueError(
                "Action ranges are not left/right reflection-closed: signed permutation "
                "would not preserve the normalized denoising process. Do not alter saved stats."
            )

    @staticmethod
    def reflect_latent(value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 12:
            raise ValueError("Expected twelve action coordinates in diffusion latents.")
        return value[..., LEFT_RIGHT_PERM.to(value.device)] * LEFT_RIGHT_SIGN.to(value)

    @staticmethod
    def reflect_conditions(proprio, actions, goals):
        p, a, g, _ = apply_symmetry(
            proprio, actions, goals, actions, index=1, mode="mirror"
        )
        return p, a, g

    def reflect_critic(self, proprio, actions, goals, critic_observation):
        # Layout is declared by Solo12DPPOEnv.get_critic_features(). CTE is an
        # unsigned distance; progress, tangent speed and time are invariants.
        condition_dim = self.policy.cfg.history * (
            self.policy.cfg.proprio_dim + self.policy.cfg.action_hist_dim + self.policy.cfg.goal_dim
        )
        if critic_observation.shape[-1] != condition_dim + 14:
            raise ValueError("Mirror critic requires the declared fourteen route features.")
        privileged = critic_observation[..., -14:].clone()
        privileged[..., 6] *= -1.0  # sin(tangent_yaw - robot_yaw)
        privileged[..., 10] *= -1.0  # projected gravity y
        p, a, g = self.reflect_conditions(proprio, actions, goals)
        return self.policy.critic_condition(p, a, g, privileged)
