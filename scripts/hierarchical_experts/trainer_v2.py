"""Budget-based trainer and resumable checkpoints for hierarchy V2."""

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.dppo_diffusion_rl.checkpointing import task_config_from_env_cfg
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer, EpisodeAccumulator


@dataclass
class HybridRolloutV2:
    proprio: torch.Tensor
    action_history: torch.Tensor
    goals: torch.Tensor
    critic_observation: torch.Tensor
    skills: torch.Tensor
    command_latents: torch.Tensor
    old_discrete_logprobs: torch.Tensor
    old_continuous_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def compute_gae(self, last_value, *, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros_like(last_value)
        for step in reversed(range(self.rewards.shape[0])):
            next_value = last_value if step == self.rewards.shape[0] - 1 else self.old_values[step + 1]
            nonterminal = (~self.dones[step]).float()
            delta = self.rewards[step] + next_value * nonterminal - self.old_values[step]
            gae = delta + gae_lambda * nonterminal * gae
            advantages[step] = gae
        self.advantages = advantages.reshape(-1)
        self.returns = (advantages + self.old_values).reshape(-1)

    def flatten(self):
        if self.advantages is None or self.returns is None:
            raise RuntimeError("GAE must be computed before flattening.")
        def merge(value):
            return value.reshape(-1, *value.shape[2:])
        return SimpleNamespace(**{
            name: merge(getattr(self, name)) for name in (
                "proprio", "action_history", "goals", "critic_observation", "skills",
                "command_latents", "old_discrete_logprobs", "old_continuous_logprobs",
                "old_values", "rewards", "dones")
        }, advantages=self.advantages, returns=self.returns)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_v2_resume_contract(state, *, architecture, expert_hashes, task_config,
                                 run_config, interaction_budget):
    if state.get("algorithm") != "hierarchical_expert_ppo_v2" or state.get("format_version") != 3:
        raise ValueError("Resume requires a recognized hierarchy V2 checkpoint.")
    if state.get("architecture") != architecture:
        raise ValueError("Checkpoint architecture does not match the V2 policy.")
    if state.get("metrics", {}).get("checkpoint_phase") == "pre_update":
        raise ValueError("Resume only from post-update checkpoints (last.pt or model_update_*.pt).")
    if "route_sampler_curriculum_state" not in state:
        raise ValueError("Resume checkpoint is missing curriculum sampler progress.")
    if state.get("expert_sha256") != expert_hashes:
        raise ValueError("Cannot resume with changed frozen expert weights.")
    if state.get("dppo_task_config") != task_config:
        raise ValueError("Cannot resume after changing the underlying task contract.")
    for key, value in run_config.items():
        if state.get("run_config", {}).get(key) != value:
            raise ValueError(f"Cannot resume after changing run setting {key}.")
    if int(state.get("planned_interaction_budget", -1)) != int(interaction_budget):
        raise ValueError("Resume must preserve the configured total interaction budget.")


class HierarchyTrainerV2(DPPOTrainer):
    def __init__(self, *, experts, hashes, interaction_budget, curriculum_fraction=0.20,
                 config, start_iteration=0, **kwargs):
        super().__init__(**kwargs)
        if interaction_budget < 1 or not 0.0 < curriculum_fraction < 1.0:
            raise ValueError("Interaction budget and curriculum fraction must be positive.")
        self.experts, self.hashes = experts, hashes
        self.interaction_budget = int(interaction_budget)
        self.curriculum_fraction = float(curriculum_fraction)
        self.curriculum_boundary = int(self.interaction_budget * self.curriculum_fraction)
        self.start_iteration = int(start_iteration)
        self.config = dict(config)
        self.resume_rng = None
        self.resume_route_sampler_state = None
        self.update_steps = self.start_iteration
        self.total_env_transitions = 0
        self.wall_time_seconds = 0.0
        self.rollout_samples = self.num_envs * self.rollout_chunks
        self.rollout_transitions = self.rollout_samples * self.policy.execution_steps
        if self.rollout_samples % self.updater.minibatch_size:
            raise ValueError("Rollout sample count must divide evenly into actor minibatches.")
        if self.rollout_samples % self.updater.critic_minibatch_size:
            raise ValueError("Rollout sample count must divide evenly into critic minibatches.")
        self.raw_env.cfg.hierarchy_curriculum_enabled = True
        self.raw_env.cfg.hierarchy_curriculum_stage = 0

    def _set_curriculum_stage(self, steps):
        stage = int(steps >= self.curriculum_boundary)
        self.raw_env.cfg.hierarchy_curriculum_stage = stage
        return stage

    @torch.no_grad()
    def collect(self):
        keys = ("proprio", "action_history", "goals", "critic_observation", "skills",
                "command_latents", "old_discrete_logprobs", "old_continuous_logprobs",
                "old_values", "rewards", "dones")
        stored = {key: [] for key in keys}
        episodes = EpisodeAccumulator()
        skill_crouch = switch_sum = switch_count = entropy_d = command_saturation = 0.0
        p_low_sum = p_high_sum = n_low = n_high = 0.0
        previous_skill = previous_done = None
        stage_samples = [0, 0]
        curriculum_counts_before = self.raw_env._routes.straight_curriculum_sample_counts.clone()
        for _ in range(self.rollout_chunks):
            stage = self._set_curriculum_stage(self.total_env_transitions)
            stage_samples[stage] += self.num_envs
            p, a, g = self.proprio_history.clone(), self.action_history.clone(), self.goal_history.clone()
            critic_obs = self._critic_observation()
            values = self.critic(critic_obs)
            sample = self.policy.sample(p, a, g)
            skill, command = sample.skill, self.policy.command(sample.skill, sample.command_latent)
            command_saturation += float((command.abs() / self.policy.command_limits[skill] > 0.95).float().mean())
            feature = self.policy.network(self.policy.features(p, a, g))
            probs = torch.softmax(self.policy.skill_head(feature), dim=-1)[:, 1]
            target = self.raw_env._route_state.target_height
            low, high = target < 0.20, target > 0.25
            p_low_sum += float(probs[low].sum()); n_low += float(low.sum())
            p_high_sum += float(probs[high].sum()); n_high += float(high.sum())
            entropy_d += float(torch.distributions.Categorical(logits=self.policy.skill_head(feature)).entropy().mean())
            skill, cmd = sample.skill, command
            if previous_skill is not None:
                valid = ~previous_done
                switch_sum += float(((skill != previous_skill) & valid).sum())
                switch_count += float(valid.sum())
            previous_skill = skill.clone()
            skill_crouch += float((skill == 1).float().mean())
            total = torch.zeros(self.num_envs, device=self.device)
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_boundary = torch.zeros_like(done)
            for step_index in range(self.policy.execution_steps):
                active = ~done
                before = self.raw_env.get_proprioception().clone()
                goal = self.raw_env.get_goal().clone()
                action = self.experts.action(self.raw_env, skill, cmd)
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
                if step_index + 1 < self.policy.execution_steps:
                    reset_boundary |= newly_done
                self.total_env_transitions += self.num_envs
            if torch.any(reset_boundary):
                self.raw_env.reset_after_action_chunk(torch.nonzero(reset_boundary).squeeze(-1))
            self._seed_history(torch.nonzero(done).squeeze(-1))
            previous_done = done
            row = (p, a, g, critic_obs, sample.skill, sample.command_latent,
                   sample.discrete_logprob, sample.continuous_logprob, values, total, done)
            for key, value in zip(keys, row):
                stored[key].append(value)
        rollout = HybridRolloutV2(**{key: torch.stack(value) for key, value in stored.items()})
        rollout.compute_gae(self.critic(self._critic_observation()))
        metrics = {"Rollout/reward_mean": float(rollout.rewards.mean()),
                   "Rollout/categorical_entropy": entropy_d / self.rollout_chunks,
                   "Rollout/crouch_prob_low_target": p_low_sum / max(n_low, 1),
                   "Rollout/crouch_prob_high_target": p_high_sum / max(n_high, 1),
                   "Rollout/low_target_samples": n_low, "Rollout/high_target_samples": n_high,
                   "Rollout/crouch_fraction": skill_crouch / self.rollout_chunks,
                   "Rollout/command_saturation": command_saturation / self.rollout_chunks,
                   "Rollout/skill_change_fraction": switch_sum / max(switch_count, 1),
                   "Curriculum/straight_stage_fraction": stage_samples[0] / (self.num_envs * self.rollout_chunks),
                   "Curriculum/final_stage_fraction": stage_samples[1] / (self.num_envs * self.rollout_chunks),
                   **episodes.metrics()}
        sampled_profile_counts = self.raw_env._routes.straight_curriculum_sample_counts - curriculum_counts_before
        for profile in range(4):
            for speed_index in range(2):
                metrics[f"Curriculum/straight_profile_{profile}_speed_{speed_index}_routes"] = int(
                    sampled_profile_counts[profile, speed_index])
        return rollout.flatten(), metrics

    def save_checkpoint(self, path, iteration, metrics):
        state = {"algorithm": "hierarchical_expert_ppo_v2", "format_version": 3,
                 "architecture": self.policy.architecture,
                 "actor_state_dict": self.policy.state_dict(), "critic_state_dict": self.critic.state_dict(),
                 "actor_optimizer_state_dict": self.updater.actor_optimizer.state_dict(),
                 "critic_optimizer_state_dict": self.updater.critic_optimizer.state_dict(),
                 "expert_sha256": self.hashes,
                 "dppo_task_config": task_config_from_env_cfg(self.raw_env.cfg),
                 "run_config": self.config, "iteration": int(iteration),
                 "update_steps": self.update_steps, "total_50hz_env_transitions": self.total_env_transitions,
                 "cumulative_training_wall_time_s": self.wall_time_seconds,
                 "planned_interaction_budget": self.interaction_budget,
                 "curriculum_boundary_steps": self.curriculum_boundary,
                 "curriculum_stage": int(self.raw_env.cfg.hierarchy_curriculum_stage),
                 "route_sampler_curriculum_state": self.raw_env._routes.curriculum_state_dict(),
                 "best_score": self.best_score, "metrics": metrics, "rng_state": _rng_state(),
                 "resume_semantics": "optimizer and global RNG restored; simulator state and rollout histories restart, so continuation is not bitwise-equivalent"}
        path = Path(path)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temp)
        os.replace(temp, path)

    def load_checkpoint(self, path, expected_config):
        state = torch.load(path, map_location=self.device, weights_only=False)
        validate_v2_resume_contract(state, architecture=self.policy.architecture,
            expert_hashes=self.hashes, task_config=task_config_from_env_cfg(self.raw_env.cfg),
            run_config=expected_config, interaction_budget=self.interaction_budget)
        self.policy.load_state_dict(state["actor_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        self.updater.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.updater.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        self.total_env_transitions = int(state["total_50hz_env_transitions"])
        self.wall_time_seconds = float(state["cumulative_training_wall_time_s"])
        self.start_iteration = int(state["iteration"]) + 1
        self.update_steps = int(state["update_steps"])
        self.best_score = float(state["best_score"])
        self.resume_rng = state["rng_state"]
        self.resume_route_sampler_state = state["route_sampler_curriculum_state"]
        self._set_curriculum_stage(self.total_env_transitions)
        self._reconcile_metrics(state)
        return state

    def _reconcile_metrics(self, state):
        """Recover the final metrics row if interrupted after checkpoint rename."""
        records = []
        if self.metrics_path.exists():
            records = [json.loads(line) for line in self.metrics_path.read_text(encoding="utf-8").splitlines()]
        if any(int(row.get("update_steps", 0)) > self.update_steps for row in records):
            raise ValueError("Metrics contain updates newer than the resume checkpoint; choose last.pt.")
        recorded = max((int(row.get("update_steps", 0)) for row in records), default=0)
        if recorded < self.update_steps:
            metrics = state.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("Resume checkpoint cannot recover its latest metrics row.")
            record = {"iteration": int(state["iteration"]), "update_steps": self.update_steps,
                      "total_50hz_env_transitions": self.total_env_transitions, **metrics}
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_metrics(self, iteration, metrics):
        record = {"iteration": int(iteration), "update_steps": self.update_steps,
                  "total_50hz_env_transitions": self.total_env_transitions, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.logger is not None:
            self.logger.log(record, step=iteration)

    def run_budget(self):
        self.env.reset()
        self._seed_history()
        if self.resume_route_sampler_state is not None:
            self.raw_env._routes.load_curriculum_state_dict(self.resume_route_sampler_state)
            self.resume_route_sampler_state = None
        if self.resume_rng is not None:
            _restore_rng_state(self.resume_rng)
            self.resume_rng = None
        if self.metrics_path.exists() and self.start_iteration == 0:
            raise FileExistsError(f"Refusing to append to existing metrics: {self.metrics_path}")
        metrics = {}
        if self.total_env_transitions >= self.interaction_budget:
            raise ValueError("The run has already reached its configured interaction budget.")
        while self.total_env_transitions < self.interaction_budget:
            stage = self._set_curriculum_stage(self.total_env_transitions)
            started = time.perf_counter()
            batch, collected = self.collect()
            score = self._selection_score(collected)
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint(self.output_dir / "best.pt", self.start_iteration,
                                     {**collected, "checkpoint_phase": "pre_update"})
            update = self.updater.update(batch)
            elapsed = time.perf_counter() - started
            self.wall_time_seconds += elapsed
            metrics = {**collected, **update, "Curriculum/stage_at_update": float(stage),
                       "Performance/iteration_seconds": elapsed,
                       "Performance/50hz_env_transitions_per_second": self.rollout_transitions / max(elapsed, 1e-6)}
            self.update_steps += 1
            checkpoint_metrics = {**metrics, "checkpoint_phase": "post_update_unscored"}
            self.save_checkpoint(self.output_dir / "last.pt", self.start_iteration, checkpoint_metrics)
            self._write_metrics(self.start_iteration, metrics)
            if self.update_steps % self.save_interval == 0:
                self.save_checkpoint(self.output_dir / f"model_update_{self.update_steps}.pt",
                                     self.start_iteration, checkpoint_metrics)
            self.start_iteration += 1
            print(f"[HierarchyV2 {self.update_steps:05d}] env_steps={self.total_env_transitions} "
                  f"success={collected['Episode/success_rate']:.3f} "
                  f"crouch={collected['Rollout/crouch_fraction']:.3f} "
                  f"stage={stage} kl_d={update['Policy/discrete_approximate_kl']:.4g} "
                  f"kl_c={update['Policy/continuous_approximate_kl']:.4g}", flush=True)
        self.save_checkpoint(self.output_dir / "last.pt", self.start_iteration - 1,
                             {**metrics, "checkpoint_phase": "post_update_unscored",
                              "actual_interaction_budget": self.total_env_transitions,
                              "budget_overshoot": self.total_env_transitions - self.interaction_budget})


__all__ = ["HierarchyTrainerV2", "HybridRolloutV2", "validate_v2_resume_contract"]
