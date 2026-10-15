"""Latent pilot loop reusing the DPPO collector and Gaussian PPO math."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.dppo_diffusion_rl.checkpointing import task_config_from_env_cfg
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer

from .checkpointing import save_latent_checkpoint, source_semantic_digest
from .policy import ALGORITHM


class LatentNoisePPOTrainer(DPPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_sha = self.policy.source_sha256
        self.source_digest = source_semantic_digest(self.policy.decoder.prior)

    def _save(self, filename, iteration, metrics):
        save_latent_checkpoint(
            self.output_dir / filename,
            source=self.source_checkpoint,
            source_path=self.source_checkpoint_path,
            source_sha=self.source_sha,
            source_digest=self.source_digest,
            policy=self.policy,
            critic=self.critic,
            updater=self.updater,
            iteration=iteration,
            total_control_transitions=self.total_physics_steps,
            metrics=metrics,
            best_score=self.best_score,
            task_config=task_config_from_env_cfg(self.raw_env.cfg),
        )

    def _write_metrics(self, iteration, metrics):
        record = {"iteration": iteration, "total_control_transitions": self.total_physics_steps,
                  **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.logger is not None:
            self.logger.log(record, step=iteration)

    def run(self, iterations):
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self._save("initial.pt", -1, {"checkpoint_phase": "pre_update_initial"})
        self.env.reset()
        self._seed_history()
        last_metrics = {"checkpoint_phase": "post_update_unscored"}
        for iteration in range(iterations):
            started = time.perf_counter()
            rollout, collected = self.collect()
            score = self._selection_score(collected)
            if score > self.best_score:
                self.best_score = score
                self._save("best.pt", iteration, {**collected, "checkpoint_phase": "pre_update"})
            actor_enabled = iteration >= self.policy.ppo_cfg.critic_warmup_iterations
            update = self.updater.update(rollout, update_actor=actor_enabled)
            elapsed = time.perf_counter() - started
            last_metrics = {
                **collected, **update,
                "Policy/critic_warmup_active": float(not actor_enabled),
                "Performance/iteration_seconds": elapsed,
                "Performance/control_transitions_per_second": (
                    self.num_envs * self.rollout_chunks * self.policy.ppo_cfg.exec_horizon
                    / max(elapsed, 1.0e-6)
                ),
            }
            self._write_metrics(iteration, last_metrics)
            if iteration % self.save_interval == 0:
                self._save(f"model_{iteration}.pt", iteration,
                           {**last_metrics, "checkpoint_phase": "post_update_unscored"})
            print(
                f"[Latent PPO {iteration:05d}] success={collected['Episode/success_rate']:.3f} "
                f"fall={collected['Episode/base_contact_rate']:.3f} "
                f"speed_err={collected['Episode/mean_speed_error_abs_mps']:.3f} "
                f"actor_updates={update['Policy/actor_updates']:.0f} "
                f"raw_std={update['Policy/raw_noise_std_mean']:.4f} "
                f"kl={update['Policy/max_observed_kl']:.5f}", flush=True,
            )
        self._save("last.pt", iterations - 1,
                   {**last_metrics, "checkpoint_phase": "post_update_unscored"})
