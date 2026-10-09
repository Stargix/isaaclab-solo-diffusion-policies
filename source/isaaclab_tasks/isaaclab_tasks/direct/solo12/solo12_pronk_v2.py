# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pronk fine-tuning that explicitly preserves the jumpy_safe reference gait."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from .pronk_ppo import PronkPPO

from .solo12_pronk_env import (
    Solo12PronkEnvCfg,
    Solo12PronkPPORunnerCfg,
    compute_pronk_left_right_symmetry,
)

# Follow the existing SOLO12 custom-module registration pattern. This config
# is imported by training and playback, so both runners resolve the algorithm.
on_policy_runner_module.PronkPPO = PronkPPO


@configclass
class PronkReferencePpoCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PronkPPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.001
    desired_kl: float = 0.005
    max_grad_norm: float = 0.5
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    rnd_cfg = None
    reference_checkpoint: str = "checkpoints/jumpy_safe.pt"
    reference_checkpoint_sha256: str = "6496d831f4d3b8fec8793eb740351edbe6581198787d713ac7010371a0ff4183"
    reference_loss_coef: float = 1.0


@configclass
class Solo12PronkV2EnvCfg(Solo12PronkEnvCfg):
    """Fine-tune in the slow-to-moderate interval selected from the source audit."""

    command_lin_vel_x_range = (0.25, 0.65)
    tracking_std = 0.5
    pronk_cycle_reward_scale = 0.0


@configclass
class Solo12PronkV2PPORunnerCfg(Solo12PronkPPORunnerCfg):
    max_iterations = 1000
    save_interval = 25
    run_name = "pronk_v2_reference_preserving_seed42"
    algorithm: PronkReferencePpoCfg = PronkReferencePpoCfg()

    def __post_init__(self):
        # Install symmetry before the v1 parent validates the PPO configuration.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=compute_pronk_left_right_symmetry,
        )
        super().__post_init__()
        self.policy.init_noise_std = 0.20
        # The v1 parent assigns its own optimization defaults in __post_init__;
        # re-apply the v2 pilot values after it has configured symmetry.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.005
        self.algorithm.reference_loss_coef = 1.0
