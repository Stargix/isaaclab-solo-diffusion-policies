# Copyright (c) 2021-2025 ETH Zurich and NVIDIA CORPORATION.
# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO fine-tuning with a frozen action reference for the original jump expert.

The update follows RSL-RL 3.1.2 PPO and adds a mean-action MSE to the frozen
expert on each sampled state. This is deliberately kept separate from generic
PPO and from the original pronk-v1 experiment.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO


class PronkPPO(PPO):
    """PPO whose actor stays near a frozen pretrained gait unless task reward helps."""

    def __init__(
        self,
        policy,
        reference_checkpoint: str,
        reference_checkpoint_sha256: str,
        reference_loss_coef: float = 1.0,
        **kwargs,
    ):
        installed_rsl_version = version("rsl-rl-lib")
        if installed_rsl_version != "3.1.2":
            raise RuntimeError(
                "PronkPPO currently mirrors the RSL-RL 3.1.2 PPO update; "
                f"found rsl-rl-lib {installed_rsl_version}."
            )
        if kwargs.get("rnd_cfg") is not None:
            raise ValueError("PronkPPO does not support RND; use the standard PPO for that task.")
        if not torch.isfinite(torch.tensor(reference_loss_coef)) or reference_loss_coef <= 0.0:
            raise ValueError("reference_loss_coef must be finite and positive.")
        super().__init__(policy, **kwargs)

        checkpoint_path = Path(reference_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen pronk reference not found: {checkpoint_path}")
        import hashlib

        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest.lower() != reference_checkpoint_sha256.lower():
            raise ValueError(
                f"Frozen reference SHA256 mismatch: got {digest}, expected {reference_checkpoint_sha256}."
            )
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")

        self.reference_policy = deepcopy(policy).to(self.device)
        self.reference_policy.load_state_dict(state, strict=True)
        self.reference_policy.eval()
        self.reference_policy.requires_grad_(False)
        # Initialize the trainable mean actor and its input coordinates from
        # exactly this reference. Critic, optimizer and exploration stay fresh.
        self.policy.actor.load_state_dict(self.reference_policy.actor.state_dict(), strict=True)
        self.policy.actor_obs_normalizer.load_state_dict(
            self.reference_policy.actor_obs_normalizer.state_dict(), strict=True
        )
        self.reference_loss_coef = float(reference_loss_coef)
        self.reference_checkpoint = str(checkpoint_path)
        self.maximum_learning_rate = self.learning_rate
        print(
            f"[INFO] Frozen pronk reference loaded: {checkpoint_path} "
            f"(SHA256 {digest}, action-MSE coefficient {self.reference_loss_coef:g})."
        )

    def process_env_step(self, obs, rewards, dones, extras):
        # The warm-start actor and the reference must keep the same observation
        # coordinate system. Continue updating the critic's separate normalizer.
        actor_normalization = self.policy.actor_obs_normalization
        self.policy.actor_obs_normalization = False
        try:
            super().process_env_step(obs, rewards, dones, extras)
        finally:
            self.policy.actor_obs_normalization = actor_normalization

    def update(self) -> dict[str, float]:
        if self.policy.is_recurrent:
            raise ValueError("Frozen-reference pronk PPO currently supports only feed-forward actors.")
        if self.rnd is not None:
            raise ValueError("Frozen-reference pronk PPO does not support RND.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_reference_loss = 0.0
        mean_symmetry_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                augment = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = augment(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                augmentation_count = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(augmentation_count, 1)
                target_values_batch = target_values_batch.repeat(augmentation_count, 1)
                advantages_batch = advantages_batch.repeat(augmentation_count, 1)
                returns_batch = returns_batch.repeat(augmentation_count, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                        / (2.0 * sigma_batch.square())
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(self.maximum_learning_rate, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (value_batch - returns_batch).square(), (value_clipped - returns_batch).square()
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).square().mean()

            with torch.no_grad():
                reference_actions = self.reference_policy.act_inference(obs_batch)
            reference_loss = F.mse_loss(self.policy.action_mean, reference_actions)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.reference_loss_coef * reference_loss
            )

            symmetry_loss = torch.zeros((), device=self.device)
            if self.symmetry:
                augment = self.symmetry["data_augmentation_func"]
                if not self.symmetry["use_data_augmentation"]:
                    obs_batch, _ = augment(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                student_augmented_actions = self.policy.act_inference(obs_batch.detach().clone())
                original_actions = student_augmented_actions[:original_batch_size]
                _, symmetric_targets = augment(obs=None, actions=original_actions, env=self.symmetry["_env"])
                symmetry_loss = F.mse_loss(
                    student_augmented_actions[original_batch_size:],
                    symmetric_targets.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_reference_loss += reference_loss.item()
            mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "reference_action_mse": mean_reference_loss / num_updates,
            "symmetry": mean_symmetry_loss / num_updates,
        }
