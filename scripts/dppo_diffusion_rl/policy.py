"""Hybrid frozen/pruned DDPM actor with exact DPPO transition likelihoods.

The environment action is the denormalized slice of ``x_0``.  PPO does not
pretend that this marginal has a tractable density: it optimizes the Gaussian
reverse transitions that generated it, as in Ren et al. (ICLR 2025).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.distributions import Normal

from scripts.diffusion_policy.model.solo12_diffusion_policy import (
    Solo12DiffusionPolicy,
    Solo12DiffusionPolicyConfig,
)
from scripts.diffusion_policy.train.data.normalization import (
    NormalizerStats,
    denormalize_minmax,
    normalize_minmax,
    normalize_zscore,
)

from .config import DPPOConfig


@dataclass
class DPPOSample:
    """One physical policy sample and its trainable reverse-diffusion chain."""

    normalized_trajectory: torch.Tensor
    denormalized_trajectory: torch.Tensor
    chain: torch.Tensor
    old_logprobs: torch.Tensor


def _as_int(timestep: int | torch.Tensor) -> int:
    return int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)


class DPPODiffusionPolicy(torch.nn.Module):
    """A trainable last-steps actor plus an immutable pretrained DDPM prior."""

    def __init__(self, policy_cfg: Solo12DiffusionPolicyConfig, dppo_cfg: DPPOConfig):
        super().__init__()
        dppo_cfg.validate(
            prediction_horizon=policy_cfg.prediction_horizon,
            execution_offset=policy_cfg.execution_offset,
        )
        if policy_cfg.prediction_type != "epsilon" or policy_cfg.variance_type != "fixed_small":
            raise ValueError("This exact DPPO implementation requires epsilon prediction and fixed_small variance.")
        if policy_cfg.guidance_scale != 1.0:
            raise ValueError("DPPO training does not support classifier-free guidance mixing.")
        policy_cfg.num_inference_steps = dppo_cfg.inference_steps
        self.policy = Solo12DiffusionPolicy(policy_cfg)
        self.base_model = copy.deepcopy(self.policy.model)
        self.dppo_cfg = dppo_cfg
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        # Attention dropout was useful for BC, but it is an unmodelled random
        # variable in PPO. Keep both denoisers in inference mode while allowing
        # gradients through the fine-tuned model.
        self.policy.model.eval()

    @property
    def cfg(self) -> Solo12DiffusionPolicyConfig:
        return self.policy.cfg

    @property
    def normalizer_stats(self) -> NormalizerStats | None:
        return self.policy.normalizer_stats

    @property
    def trainable_parameters(self):
        return self.policy.model.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.policy.model.eval()
        self.base_model.eval()
        return self

    def load_pretrained(self, state_dict: dict[str, torch.Tensor], stats: NormalizerStats | dict) -> None:
        self.policy.load_state_dict(state_dict)
        self.base_model.load_state_dict(self.policy.model.state_dict())
        self.policy.set_normalizer_stats(stats)
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self.policy.model.eval()

    def load_dppo_actor(
        self,
        actor_policy_state_dict: dict[str, torch.Tensor],
        base_model_state_dict: dict[str, torch.Tensor],
        stats: NormalizerStats | dict,
    ) -> None:
        self.policy.load_state_dict(actor_policy_state_dict)
        self.base_model.load_state_dict(base_model_state_dict)
        self.policy.set_normalizer_stats(stats)
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self.policy.model.eval()

    def actor_policy_state_dict(self) -> dict[str, torch.Tensor]:
        return self.policy.state_dict()

    def base_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self.base_model.state_dict()

    def _normalized_condition(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be loaded before DPPO sampling.")
        return (
            normalize_zscore(proprio_hist, self.normalizer_stats.proprio),
            normalize_minmax(action_hist, self.normalizer_stats.action),
            normalize_zscore(goal_hist, self.normalizer_stats.goal),
        )

    def critic_condition(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        privileged_features: torch.Tensor,
    ) -> torch.Tensor:
        """Build a normalized asymmetric-critic input without affecting the actor contract."""

        proprio_n, action_n, goal_n = self._normalized_condition(
            proprio_hist, action_hist, goal_hist
        )
        return torch.cat(
            (
                proprio_n.flatten(1),
                action_n.flatten(1),
                goal_n.flatten(1),
                privileged_features,
            ),
            dim=1,
        )

    def _model_output(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        proprio_n: torch.Tensor,
        action_n: torch.Tensor,
        goal_n: torch.Tensor,
        *,
        trainable: bool,
    ) -> torch.Tensor:
        model = self.policy.model if trainable else self.base_model
        return model(sample, proprio_n, action_n, goal_n, timestep)

    def _transition_mean_std(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        *,
        apply_exploration_floor: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror ``diffusers.DDPMScheduler.step`` without drawing its noise."""

        scheduler = self.policy.noise_scheduler
        t = _as_int(timestep)
        prev_t = _as_int(scheduler.previous_timestep(timestep))
        alpha_prod_t = scheduler.alphas_cumprod[t].to(sample)
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t].to(sample) if prev_t >= 0 else scheduler.one.to(sample)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if scheduler.config.thresholding:
            pred_original = scheduler._threshold_sample(pred_original)
        elif scheduler.config.clip_sample:
            pred_original = pred_original.clamp(
                -float(scheduler.config.clip_sample_range),
                float(scheduler.config.clip_sample_range),
            )
        coeff_x0 = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        coeff_xt = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        mean = coeff_x0 * pred_original + coeff_xt * sample
        variance = scheduler._get_variance(timestep).to(sample)
        std = variance.sqrt()
        if apply_exploration_floor:
            std = std.clamp_min(self.dppo_cfg.min_denoising_std)
        return mean, std

    def _reward_relevant_logprob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        next_sample: torch.Tensor,
        final_transition: bool | torch.Tensor,
    ) -> torch.Tensor:
        """Likelihood of stochastic tokens that can still cause the reward.

        The trajectory contains a generated history prefix before its
        executable slice. Because the denoiser is causal, that prefix affects
        executable tokens while another reverse step remains. At the final
        transition the newly sampled prefix is never consumed again, so only
        the executed slice belongs to the reward's stochastic computation
        graph. Tokens after the executed chunk are always irrelevant.
        """

        start = self.cfg.execution_offset
        end = self.cfg.execution_offset + self.dppo_cfg.exec_horizon
        elementwise = Normal(mean, std).log_prob(next_sample).clamp(-5.0, 2.0)
        # The paper's implementation averages rather than sums the high-
        # dimensional action likelihood and clips rare tails before the PPO
        # ratio. This avoids exponential ratios for action chunks.
        # Keep one common denominator across denoising transitions. The final
        # transition omits irrelevant prefix scores; it must not thereby give
        # each executed coordinate ``end / exec_horizon`` times more weight.
        denominator = float(end * elementwise.shape[-1])
        causal = elementwise[:, :end].sum(dim=(-1, -2)) / denominator
        executed = elementwise[:, start:end].sum(dim=(-1, -2)) / denominator
        if isinstance(final_transition, bool):
            return executed if final_transition else causal
        return torch.where(final_transition.bool(), executed, causal)

    @torch.no_grad()
    def sample(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        training_exploration: bool = True,
    ) -> DPPOSample:
        proprio_n, action_n, goal_n = self._normalized_condition(proprio_hist, action_hist, goal_hist)
        scheduler = self.policy.noise_scheduler
        scheduler.set_timesteps(self.dppo_cfg.inference_steps, device=proprio_n.device)
        trajectory = torch.randn(
            (proprio_n.shape[0], self.cfg.prediction_horizon, self.cfg.action_dim),
            device=proprio_n.device,
            dtype=proprio_n.dtype,
            generator=generator,
        )
        trainable_start = self.dppo_cfg.inference_steps - self.dppo_cfg.finetune_denoising_steps
        chain: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        for index, timestep in enumerate(scheduler.timesteps):
            trainable = index >= trainable_start
            if trainable and not chain:
                chain.append(trajectory.clone())
            t_batch = torch.full(
                (trajectory.shape[0],), _as_int(timestep), device=trajectory.device, dtype=torch.long
            )
            output = self._model_output(
                trajectory, t_batch, proprio_n, action_n, goal_n, trainable=trainable
            )
            mean, std = self._transition_mean_std(
                trajectory,
                output,
                timestep,
                apply_exploration_floor=training_exploration,
            )
            if training_exploration or _as_int(timestep) > 0:
                noise = torch.randn(
                    trajectory.shape,
                    device=trajectory.device,
                    dtype=trajectory.dtype,
                    generator=generator,
                )
                next_trajectory = mean + std * noise
            else:
                next_trajectory = mean
            if trainable:
                if training_exploration:
                    logprobs.append(
                        self._reward_relevant_logprob(
                            mean,
                            std,
                            next_trajectory,
                            final_transition=index == self.dppo_cfg.inference_steps - 1,
                        )
                    )
                else:
                    logprobs.append(torch.zeros(trajectory.shape[0], device=trajectory.device))
                chain.append(next_trajectory.clone())
            trajectory = next_trajectory
        if len(chain) != self.dppo_cfg.finetune_denoising_steps + 1:
            raise RuntimeError("The stored reverse chain does not match finetune_denoising_steps.")
        # Keep the stochastic chain unmodified for an exact likelihood, while
        # respecting the BC action support at the environment boundary.
        normalized = trajectory.clamp(-1.0, 1.0)
        denormalized = denormalize_minmax(normalized, self.normalizer_stats.action)
        return DPPOSample(
            normalized_trajectory=normalized,
            denormalized_trajectory=denormalized,
            chain=torch.stack(chain, dim=1),
            old_logprobs=torch.stack(logprobs, dim=1),
        )

    def transition_logprob(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        chain_previous: torch.Tensor,
        chain_next: torch.Tensor,
        local_denoising_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate selected on-policy transitions under the current actor."""

        proprio_n, action_n, goal_n = self._normalized_condition(proprio_hist, action_hist, goal_hist)
        scheduler = self.policy.noise_scheduler
        scheduler.set_timesteps(self.dppo_cfg.inference_steps, device=proprio_n.device)
        offset = self.dppo_cfg.inference_steps - self.dppo_cfg.finetune_denoising_steps
        global_index = local_denoising_index + offset
        timestep_values = scheduler.timesteps[global_index].to(device=chain_previous.device, dtype=torch.long)
        output = self._model_output(
            chain_previous,
            timestep_values,
            proprio_n,
            action_n,
            goal_n,
            trainable=True,
        )

        means = torch.empty_like(chain_previous)
        stds = torch.empty(
            (len(chain_previous), 1, 1), device=chain_previous.device, dtype=chain_previous.dtype
        )
        # A minibatch normally contains all K' values. Grouping avoids relying
        # on private scheduler methods accepting a vector timestep.
        for index in torch.unique(local_denoising_index, sorted=True):
            mask = local_denoising_index == index
            timestep = scheduler.timesteps[int(index.item()) + offset]
            mean, std = self._transition_mean_std(chain_previous[mask], output[mask], timestep)
            means[mask] = mean
            stds[mask] = std
        final_transition = local_denoising_index == self.dppo_cfg.finetune_denoising_steps - 1
        logprob = self._reward_relevant_logprob(
            means, stds, chain_next, final_transition=final_transition
        )
        entropy = Normal(means, stds).entropy()
        start = self.cfg.execution_offset
        end = self.cfg.execution_offset + self.dppo_cfg.exec_horizon
        denominator = float(end * entropy.shape[-1])
        causal_entropy = entropy[:, :end].sum(dim=(-1, -2)) / denominator
        executed_entropy = entropy[:, start:end].sum(dim=(-1, -2)) / denominator
        return logprob, torch.where(final_transition, executed_entropy, causal_entropy)

    @torch.no_grad()
    def predict_action(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if guidance_scale not in (None, 1.0):
            raise ValueError("DPPO hybrid inference supports guidance_scale=1 only.")
        return self.sample(
            proprio_hist,
            action_hist,
            goal_hist,
            generator=generator,
            training_exploration=False,
        ).normalized_trajectory

    @torch.no_grad()
    def predict_action_denormalized(self, *args, **kwargs) -> torch.Tensor:
        guidance_scale = kwargs.pop("guidance_scale", None)
        if guidance_scale not in (None, 1.0):
            raise ValueError("DPPO hybrid inference supports guidance_scale=1 only.")
        return self.sample(*args, **kwargs, training_exploration=False).denormalized_trajectory

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int | None = None) -> torch.Tensor:
        requested = self.dppo_cfg.exec_horizon if num_actions is None else num_actions
        if requested != self.dppo_cfg.exec_horizon:
            raise ValueError(
                "DPPO must execute the same horizon used by its likelihood objective: "
                f"requested {requested}, trained {self.dppo_cfg.exec_horizon}."
            )
        return self.policy.executable_chunk(trajectory, requested)

    def denoising_clip(self, local_denoising_index: torch.Tensor) -> torch.Tensor:
        """Step-dependent PPO epsilon from the official DPPO implementation."""

        if self.dppo_cfg.finetune_denoising_steps == 1:
            return torch.full_like(local_denoising_index, self.dppo_cfg.clip_ratio, dtype=torch.float)
        fraction = local_denoising_index.float() / (self.dppo_cfg.finetune_denoising_steps - 1)
        numerator = torch.exp(self.dppo_cfg.clip_ratio_rate * fraction) - 1.0
        denominator = math.exp(self.dppo_cfg.clip_ratio_rate) - 1.0
        return self.dppo_cfg.clip_ratio_base + (
            self.dppo_cfg.clip_ratio - self.dppo_cfg.clip_ratio_base
        ) * numerator / denominator
