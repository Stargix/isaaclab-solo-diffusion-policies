"""On-policy collection and optimization loop for Solo12 DPPO."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .buffer import RolloutBatch
from .checkpointing import save_checkpoint
from .critic import ValueCritic
from .policy import DPPODiffusionPolicy
from .ppo import DPPOUpdater


@dataclass
class EpisodeAccumulator:
    count: float = 0.0
    success: float = 0.0
    base_contact: float = 0.0
    corridor_failure: float = 0.0
    terminal_overshoot: float = 0.0
    arrival_failure: float = 0.0
    progress_fraction: float = 0.0
    mean_speed_error_abs_mps: float = 0.0
    terminal_distance_m: float = 0.0
    returns: list[float] = field(default_factory=list)

    def add_extras(self, extras: dict[str, Any], active: torch.Tensor) -> None:
        event = extras.get("dppo_episode")
        if not event:
            return
        env_ids = event["env_ids"].long()
        keep = active[env_ids]
        if not torch.any(keep):
            return
        count = float(keep.sum())
        self.count += count
        for name in (
            "success",
            "base_contact",
            "corridor_failure",
            "terminal_overshoot",
            "arrival_failure",
            "progress_fraction",
            "mean_speed_error_abs_mps",
            "terminal_distance_m",
        ):
            setattr(self, name, getattr(self, name) + float(event[name][keep].sum()))

    def metrics(self) -> dict[str, float]:
        divisor = max(self.count, 1.0)
        return {
            "Episode/completed": self.count,
            "Episode/success_rate": self.success / divisor,
            "Episode/base_contact_rate": self.base_contact / divisor,
            "Episode/corridor_failure_rate": self.corridor_failure / divisor,
            "Episode/terminal_overshoot_rate": self.terminal_overshoot / divisor,
            "Episode/arrival_failure_rate": self.arrival_failure / divisor,
            "Episode/progress_fraction": self.progress_fraction / divisor,
            "Episode/mean_speed_error_abs_mps": self.mean_speed_error_abs_mps / divisor,
            "Episode/terminal_distance_m": self.terminal_distance_m / divisor,
            "Episode/return": sum(self.returns) / max(len(self.returns), 1),
        }


class DPPOTrainer:
    def __init__(
        self,
        *,
        env,
        policy: DPPODiffusionPolicy,
        critic: ValueCritic,
        updater: DPPOUpdater,
        source_checkpoint: dict[str, Any],
        source_checkpoint_path: str | Path,
        output_dir: str | Path,
        rollout_chunks: int,
        start_iteration: int = 0,
        initial_total_physics_steps: int = 0,
        save_interval: int = 25,
        logger=None,
    ):
        if rollout_chunks < 1 or save_interval < 1:
            raise ValueError("rollout_chunks and save_interval must be positive.")
        self.env = env
        self.raw_env = env.unwrapped
        self.policy = policy
        self.critic = critic
        self.updater = updater
        self.source_checkpoint = source_checkpoint
        self.source_checkpoint_path = str(source_checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_chunks = rollout_chunks
        self.start_iteration = start_iteration
        self.save_interval = save_interval
        self.logger = logger
        self.num_envs = self.raw_env.num_envs
        self.device = torch.device(self.raw_env.device)
        self.history = self.policy.cfg.history
        self.proprio_history = torch.zeros(
            self.num_envs, self.history, self.policy.cfg.proprio_dim, device=self.device
        )
        self.action_history = torch.zeros(
            self.num_envs, self.history, self.policy.cfg.action_hist_dim, device=self.device
        )
        self.goal_history = torch.zeros(
            self.num_envs, self.history, self.policy.cfg.goal_dim, device=self.device
        )
        self.previous_applied_action = torch.zeros(
            self.num_envs, self.policy.cfg.action_dim, device=self.device
        )
        self.running_episode_return = torch.zeros(self.num_envs, device=self.device)
        self.total_physics_steps = int(initial_total_physics_steps)
        continuing_same_output = (
            Path(source_checkpoint_path).resolve().parent == self.output_dir.resolve()
        )
        saved_best = source_checkpoint.get("best_score") if continuing_same_output else None
        if saved_best is not None:
            self.best_score = float(saved_best)
        elif source_checkpoint.get("algorithm") == "dppo" and continuing_same_output:
            try:
                self.best_score = self._selection_score(source_checkpoint.get("metrics", {}))
            except KeyError:
                self.best_score = float("-inf")
        else:
            self.best_score = float("-inf")
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def _critic_observation(self) -> torch.Tensor:
        return self.policy.critic_condition(
            self.proprio_history,
            self.action_history,
            self.goal_history,
            self.raw_env.get_critic_features(),
        )

    @staticmethod
    def _slide(buffer: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        buffer[mask, :-1] = buffer[mask, 1:].clone()
        buffer[mask, -1] = value[mask]

    def _advance_history(
        self,
        proprio_before: torch.Tensor,
        goal_before: torch.Tensor,
        applied_action: torch.Tensor,
        active: torch.Tensor,
        newly_done: torch.Tensor,
    ) -> None:
        self._slide(self.proprio_history, proprio_before, active)
        self._slide(self.action_history, self.previous_applied_action, active)
        self._slide(self.goal_history, goal_before, active)
        self.previous_applied_action[active] = applied_action[active]
        if torch.any(newly_done):
            self.proprio_history[newly_done] = 0.0
            self.action_history[newly_done] = 0.0
            self.goal_history[newly_done] = 0.0
            self.previous_applied_action[newly_done] = 0.0

    def _seed_history(self, env_ids: torch.Tensor | None = None) -> None:
        """Reproduce the Phase-A padded-start observation contract."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        proprio = self.raw_env.get_proprioception()[env_ids]
        goals = self.raw_env.get_goal()[env_ids]
        self.proprio_history[env_ids] = proprio[:, None, :].expand(-1, self.history, -1)
        self.goal_history[env_ids] = goals[:, None, :].expand(-1, self.history, -1)
        self.action_history[env_ids] = 0.0
        self.previous_applied_action[env_ids] = 0.0

    @torch.no_grad()
    def collect(self) -> tuple[RolloutBatch, dict[str, float]]:
        stored: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "proprio",
                "action_history",
                "goals",
                "critic_observation",
                "chains",
                "old_logprobs",
                "old_values",
                "rewards",
                "dones",
            )
        }
        episodes = EpisodeAccumulator()
        saturation_sum = 0.0
        for _ in range(self.rollout_chunks):
            proprio = self.proprio_history.clone()
            action_history = self.action_history.clone()
            goals = self.goal_history.clone()
            critic_observation = self._critic_observation()
            values = self.critic(critic_observation)
            sample = self.policy.sample(proprio, action_history, goals)
            chunk = self.policy.executable_chunk(sample.denormalized_trajectory)
            saturation_sum += float((sample.normalized_trajectory.abs() >= 0.999).float().mean())
            macro_reward = torch.zeros(self.num_envs, device=self.device)
            macro_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            needs_boundary_reset = torch.zeros_like(macro_done)
            for index in range(self.policy.dppo_cfg.exec_horizon):
                active = ~macro_done
                proprio_before = self.raw_env.get_proprioception().clone()
                goal_before = self.raw_env.get_goal().clone()
                action = torch.where(active[:, None], chunk[:, index], torch.zeros_like(chunk[:, index]))
                _, reward, terminated, truncated, extras = self.env.step(action)
                done_now = (terminated | truncated) & active
                macro_reward += reward * active.float()
                self.running_episode_return += reward * active.float()
                if torch.any(done_now):
                    episodes.returns.extend(
                        self.running_episode_return[done_now].detach().cpu().tolist()
                    )
                    self.running_episode_return[done_now] = 0.0
                episodes.add_extras(extras, active)
                self._advance_history(proprio_before, goal_before, action, active, done_now)
                macro_done |= done_now
                if index + 1 < self.policy.dppo_cfg.exec_horizon:
                    needs_boundary_reset |= done_now
                self.total_physics_steps += self.num_envs

            if torch.any(needs_boundary_reset):
                self.raw_env.reset_after_action_chunk(
                    torch.nonzero(needs_boundary_reset, as_tuple=False).squeeze(-1)
                )
            self._seed_history(torch.nonzero(macro_done, as_tuple=False).squeeze(-1))

            stored["proprio"].append(proprio)
            stored["action_history"].append(action_history)
            stored["goals"].append(goals)
            stored["critic_observation"].append(critic_observation)
            stored["chains"].append(sample.chain)
            stored["old_logprobs"].append(sample.old_logprobs)
            stored["old_values"].append(values)
            stored["rewards"].append(macro_reward)
            stored["dones"].append(macro_done)

        last_value = self.critic(self._critic_observation())
        rollout = RolloutBatch(**{name: torch.stack(values) for name, values in stored.items()})
        rollout.compute_gae(
            last_value,
            gamma=self.policy.dppo_cfg.gamma,
            gae_lambda=self.policy.dppo_cfg.gae_lambda,
        )
        assert rollout.advantages is not None and rollout.returns is not None
        metrics = {
            "Rollout/reward_mean": float(rollout.rewards.mean()),
            "Rollout/reward_std": float(rollout.rewards.std(unbiased=False)),
            "Rollout/advantage_mean": float(rollout.advantages.mean()),
            "Rollout/return_mean": float(rollout.returns.mean()),
            "Rollout/action_saturation_fraction": saturation_sum / self.rollout_chunks,
            **episodes.metrics(),
        }
        return rollout, metrics

    @staticmethod
    def _selection_score(metrics: dict[str, float]) -> float:
        if metrics["Episode/completed"] <= 0.0:
            return float("-inf")
        return (
            metrics["Episode/success_rate"]
            - metrics["Episode/base_contact_rate"]
            - 0.5 * metrics["Episode/corridor_failure_rate"]
            - 0.5 * metrics["Episode/terminal_overshoot_rate"]
            - 0.5 * metrics.get("Episode/arrival_failure_rate", 0.0)
            + 0.10 * metrics["Episode/progress_fraction"]
            - 0.50 * metrics["Episode/mean_speed_error_abs_mps"]
            - 0.01 * metrics["Episode/terminal_distance_m"]
        )

    def _write_metrics(self, iteration: int, metrics: dict[str, float]) -> None:
        record = {"iteration": iteration, "total_physics_steps": self.total_physics_steps, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.logger is not None:
            self.logger.log(record, step=iteration)

    def _save(self, filename: str, iteration: int, metrics: dict[str, float]) -> None:
        save_checkpoint(
            self.output_dir / filename,
            source_checkpoint=self.source_checkpoint,
            source_path=self.source_checkpoint_path,
            policy=self.policy,
            critic=self.critic,
            updater=self.updater,
            iteration=iteration,
            total_physics_steps=self.total_physics_steps,
            metrics=metrics,
            best_score=self.best_score,
            task_config={
                "speed_budget_max_mps": float(self.raw_env.cfg.speed_budget_max_mps),
                "route_stage": int(self.raw_env.cfg.route_stage),
                "route_speed_max_mps": self.raw_env.cfg.route_speed_max_mps,
                "episode_length_s": float(self.raw_env.cfg.episode_length_s),
            },
        )

    def run(self, iterations: int) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.env.reset()
        self._seed_history()
        for iteration in range(self.start_iteration, self.start_iteration + iterations):
            started = time.perf_counter()
            rollout, rollout_metrics = self.collect()
            actor_enabled = iteration >= self.policy.dppo_cfg.critic_warmup_iterations
            update_metrics = self.updater.update(rollout, update_actor=actor_enabled)
            elapsed = time.perf_counter() - started
            metrics = {
                **rollout_metrics,
                **update_metrics,
                "Policy/critic_warmup_active": float(not actor_enabled),
                "Performance/iteration_seconds": elapsed,
                "Performance/physics_steps_per_second": (
                    self.num_envs
                    * self.rollout_chunks
                    * self.policy.dppo_cfg.exec_horizon
                    / max(elapsed, 1.0e-6)
                ),
            }
            self._write_metrics(iteration, metrics)
            score = self._selection_score(metrics)
            if score > self.best_score:
                self.best_score = score
                self._save("best.pt", iteration, metrics)
            if iteration % self.save_interval == 0:
                self._save(f"model_{iteration}.pt", iteration, metrics)
            print(
                f"[DPPO {iteration:05d}] reward={metrics['Rollout/reward_mean']:+.3f} "
                f"success={metrics['Episode/success_rate']:.3f} "
                f"arrival_fail={metrics['Episode/arrival_failure_rate']:.3f} "
                f"fall={metrics['Episode/base_contact_rate']:.3f} "
                f"progress={metrics['Episode/progress_fraction']:.3f} "
                f"speed_err={metrics['Episode/mean_speed_error_abs_mps']:.3f} "
                f"kl={metrics['Policy/approximate_kl']:.5f} "
                f"ref_kl={metrics['Policy/reference_kl']:.5f} "
                f"sat={metrics['Rollout/action_saturation_fraction']:.3f}"
            )
        final_iteration = self.start_iteration + iterations - 1
        self._save("last.pt", final_iteration, metrics)
