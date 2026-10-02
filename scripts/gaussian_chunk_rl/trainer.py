"""Reuse the tested physical collector, histories, episode metrics and GAE."""

import os
import time
import torch
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer
from scripts.dppo_diffusion_rl.checkpointing import (
    DPPO_TASK_CONTRACT_VERSION, sha256_file, task_config_from_env_cfg,
)
from .policy import ALGORITHM


class GaussianPPOTrainer(DPPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_sha = sha256_file(self.source_checkpoint_path)

    def _save(self, filename, iteration, metrics):
        source = self.source_checkpoint
        state = {name: source[name] for name in
                 ("schema_version", "policy_kind", "config", "normalizer_stats", "split_manifest")
                 if name in source}
        state.update({
            "algorithm": ALGORITHM, "policy_backend": "deterministic_chunk",
            "gaussian_ppo_checkpoint_version": 1, "gaussian_ppo_config": self.policy.ppo_cfg.to_dict(),
            # Legacy evaluator names describe the shared task, not the algorithm.
            "dppo_task_contract_version": DPPO_TASK_CONTRACT_VERSION,
            "dppo_task_config": task_config_from_env_cfg(self.raw_env.cfg),
            "ema_model_state_dict": self.policy.policy.state_dict(),
            "model_state_dict": self.policy.policy.state_dict(),
            "ema_is_training_actor": True, "gaussian_log_std": self.policy.log_std.detach(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.updater.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.updater.critic_optimizer.state_dict(),
            "iteration": iteration, "total_physics_steps": self.total_physics_steps,
            "metrics": metrics, "best_score": self.best_score,
            "source_checkpoint_sha256": self.source_sha,
            "source_checkpoint_path": self.source_checkpoint_path,
            "evaluation_sampling": "clipped_mean_no_exploration",
            "denoising_steps": 0, "forward_passes_per_chunk": 1,
            "resume_supported": False,
        })
        path = self.output_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def run(self, iterations):
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.env.reset()
        self._seed_history()
        for iteration in range(iterations):
            started = time.perf_counter()
            rollout, collected = self.collect()
            score = self._selection_score(collected)
            # These outcomes belong to the PRE-update actor. Select exactly that
            # actor, rather than attaching its score to untested updated weights.
            if score > self.best_score:
                self.best_score = score
                self._save("best.pt", iteration, {**collected, "checkpoint_phase": "pre_update"})
            actor_enabled = iteration >= self.policy.ppo_cfg.critic_warmup_iterations
            update = self.updater.update(rollout, update_actor=actor_enabled)
            elapsed = time.perf_counter() - started
            metrics = {**collected, **update, "Policy/critic_warmup_active": float(not actor_enabled),
                       "Performance/iteration_seconds": elapsed,
                       "Performance/physics_steps_per_second": self.num_envs * self.rollout_chunks
                       * self.policy.ppo_cfg.exec_horizon / max(elapsed, 1e-6)}
            self._write_metrics(iteration, metrics)
            if iteration % self.save_interval == 0:
                self._save(f"model_{iteration}.pt", iteration,
                           {**metrics, "checkpoint_phase": "post_update_unscored"})
            print(f"[Gaussian PPO {iteration:05d}] success={collected['Episode/success_rate']:.3f} "
                  f"fall={collected['Episode/base_contact_rate']:.3f} "
                  f"speed_err={collected['Episode/mean_speed_error_abs_mps']:.3f} "
                  f"actor_updates={update['Policy/actor_updates']:.0f} "
                  f"std={update['Policy/std_mean']:.4f} kl_max={update['Policy/max_observed_kl']:.5f} "
                  f"lr={update['Policy/actor_lr']:.2e}->{update['Policy/actor_lr_next']:.2e}", flush=True)
        self._save("last.pt", iterations - 1, {**metrics, "checkpoint_phase": "post_update_unscored"})
