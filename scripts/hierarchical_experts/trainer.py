"""Collect four physical steps per high-level PPO decision."""

import os
import time
import torch

from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.checkpointing import task_config_from_env_cfg
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer, EpisodeAccumulator


class HierarchyTrainer(DPPOTrainer):
    def __init__(self, *, experts, hashes, **kwargs):
        super().__init__(**kwargs)
        self.experts, self.hashes = experts, hashes

    @torch.no_grad()
    def collect(self):
        names = ("proprio", "action_history", "goals", "critic_observation", "chains",
                 "old_logprobs", "old_values", "rewards", "dones")
        stored = {key: [] for key in names}
        episodes = EpisodeAccumulator()
        crouch_sum = saturation_sum = switch_sum = switch_count = categorical_entropy_sum = 0.0
        low_prob_sum = high_prob_sum = low_count = high_count = 0.0
        previous_skill = None
        previous_done = None
        for _ in range(self.rollout_chunks):
            p, a, g = self.proprio_history.clone(), self.action_history.clone(), self.goal_history.clone()
            critic_obs = self._critic_observation()
            values = self.critic(critic_obs)
            latent, logprob = self.policy.sample(p, a, g)
            _, categorical = self.policy.distribution(p, a, g)
            categorical_entropy_sum += float(categorical.entropy().mean())
            # Only clear posture plateaus: intermediate targets are not classified.
            target = self.raw_env._route_state.target_height
            low, high = target < 0.20, target > 0.25
            crouch_prob = categorical.probs[:, 1]
            low_prob_sum += float(crouch_prob[low].sum())
            high_prob_sum += float(crouch_prob[high].sum())
            low_count += float(low.sum())
            high_count += float(high.sum())
            skill, command = self.policy.command(latent)
            if previous_skill is not None:
                valid = ~previous_done
                switch_sum += float(((skill != previous_skill) & valid).sum())
                switch_count += float(valid.sum())
            previous_skill = skill.clone()
            crouch_sum += float(skill.float().mean())
            saturation_sum += float((latent[:, :3].tanh().abs() > 0.95).float().mean())
            total = torch.zeros(self.num_envs, device=self.device)
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_boundary = torch.zeros_like(done)
            for step in range(self.policy.dppo_cfg.exec_horizon):
                active = ~done
                before, goal = self.raw_env.get_proprioception().clone(), self.raw_env.get_goal().clone()
                action = self.experts.action(self.raw_env, skill, command)
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                _, reward, terminated, truncated, extras = self.env.step(action)
                newly_done = (terminated | truncated) & active
                total += reward * active.float()
                self.running_episode_return += reward * active.float()
                if torch.any(newly_done):
                    episodes.returns.extend(self.running_episode_return[newly_done].cpu().tolist())
                    self.running_episode_return[newly_done] = 0.0
                episodes.add_extras(extras, active)
                self._advance_history(before, goal, action, active, newly_done)
                done |= newly_done
                if step + 1 < self.policy.dppo_cfg.exec_horizon:
                    reset_boundary |= newly_done
                self.total_physics_steps += self.num_envs
            if torch.any(reset_boundary):
                self.raw_env.reset_after_action_chunk(torch.nonzero(reset_boundary).squeeze(-1))
            self._seed_history(torch.nonzero(done).squeeze(-1))
            previous_done = done
            for key, value in zip(names, (p, a, g, critic_obs, latent[:, None], logprob[:, None], values, total, done)):
                stored[key].append(value)
        rollout = RolloutBatch(**{k: torch.stack(v) for k, v in stored.items()})
        rollout.compute_gae(self.critic(self._critic_observation()), gamma=self.policy.dppo_cfg.gamma,
                            gae_lambda=self.policy.dppo_cfg.gae_lambda)
        metrics = {"Rollout/reward_mean": float(rollout.rewards.mean()),
                   "Rollout/categorical_entropy": categorical_entropy_sum / self.rollout_chunks,
                   "Rollout/crouch_prob_low_target": low_prob_sum / max(low_count, 1),
                   "Rollout/crouch_prob_high_target": high_prob_sum / max(high_count, 1),
                   "Rollout/low_target_samples": low_count,
                   "Rollout/high_target_samples": high_count,
                   "Rollout/crouch_fraction": crouch_sum / self.rollout_chunks,
                   "Rollout/skill_change_fraction": switch_sum / max(switch_count, 1.0),
                   "Rollout/command_saturation": saturation_sum / self.rollout_chunks, **episodes.metrics()}
        return rollout, metrics

    def _save(self, filename, iteration, metrics):
        state = {"algorithm": "hierarchical_expert_ppo", "format_version": 2,
                 "actor_state_dict": self.policy.state_dict(), "critic_state_dict": self.critic.state_dict(),
                 "actor_optimizer_state_dict": self.updater.actor_optimizer.state_dict(),
                 "critic_optimizer_state_dict": self.updater.critic_optimizer.state_dict(),
                 "categorical_entropy_coef": self.updater.categorical_entropy_coef,
                 "ppo_config": self.policy.ppo_cfg.to_dict(), "expert_sha256": self.hashes,
                 "dppo_task_config": task_config_from_env_cfg(self.raw_env.cfg),
                 "iteration": iteration, "total_physics_steps": self.total_physics_steps,
                 "best_score": self.best_score, "metrics": metrics,
                 "resume_supported": False}
        path = self.output_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def run(self, iterations):
        self.env.reset()
        self._seed_history()
        for iteration in range(iterations):
            start = time.perf_counter()
            rollout, collected = self.collect()
            score = self._selection_score(collected)
            if score > self.best_score:
                self.best_score = score
                self._save("best.pt", iteration, {**collected, "checkpoint_phase": "pre_update"})
            update = self.updater.update(rollout, update_actor=True)
            update["Policy/joint_latent_entropy"] = update.pop("Policy/gaussian_entropy")
            metrics = {**collected, **update, "Performance/iteration_seconds": time.perf_counter() - start}
            self._write_metrics(iteration, metrics)
            if iteration % self.save_interval == 0:
                self._save(f"model_{iteration}.pt", iteration, {**metrics, "checkpoint_phase": "post_update_unscored"})
            print(f"[Hierarchy {iteration:04d}] success={collected['Episode/success_rate']:.3f} "
                  f"crouch={collected['Rollout/crouch_fraction']:.3f} "
                  f"updates={update['Policy/actor_updates']:.0f}", flush=True)
        self._save("last.pt", iterations - 1, {**metrics, "checkpoint_phase": "post_update_unscored"})
