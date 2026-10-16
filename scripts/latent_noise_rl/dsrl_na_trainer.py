"""Online DSRL-NA collection and checkpointing for the Solo12 vector task."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from scripts.dppo_diffusion_rl.checkpointing import task_config_from_env_cfg
from scripts.dppo_diffusion_rl.trainer import EpisodeAccumulator
from .checkpointing import source_semantic_digest
from .dsrl_na import DSRLNA, DSRLNoiseReplay, save_dsrl_na_checkpoint


class DSRLNATrainer:
    def __init__(self, *, env, agent: DSRLNA, source, source_path, source_sha,
                 output_dir, rollout_chunks=2, save_interval=100, logger=None):
        if rollout_chunks < 1 or save_interval < 1:
            raise ValueError("rollout_chunks and save_interval must be positive.")
        self.env, self.raw_env, self.agent = env, env.unwrapped, agent
        if rollout_chunks != agent.cfg.train_freq:
            raise ValueError("rollout_chunks must match the pinned DSRL train_freq.")
        self.source, self.source_path, self.source_sha = source, str(source_path), source_sha
        self.output_dir, self.rollout_chunks = Path(output_dir), rollout_chunks
        self.save_interval, self.logger = save_interval, logger
        self.num_envs, self.device = self.raw_env.num_envs, torch.device(self.raw_env.device)
        self.agent.to(self.device)
        self.history = agent.history
        self.proprio_history = torch.zeros(self.num_envs, self.history, agent.proprio_dim, device=self.device)
        self.action_history = torch.zeros(self.num_envs, self.history, agent.action_hist_dim, device=self.device)
        self.goal_history = torch.zeros(self.num_envs, self.history, agent.goal_dim, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, agent.action_dim, device=self.device)
        self.running_return = torch.zeros(self.num_envs, device=self.device)
        self.control_steps = 0
        self.vector_chunks = 0
        self.replay = DSRLNoiseReplay(
            agent.cfg.replay_capacity,
            (self.history, agent.proprio_dim),
            (self.history, agent.action_hist_dim),
            (self.history, agent.goal_dim), agent.executed_dim, self.device,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prior_digest = source_semantic_digest(agent.decoder.prior)

    @staticmethod
    def _slide(buffer, value, mask):
        if torch.any(mask):
            buffer[mask, :-1] = buffer[mask, 1:].clone()
            buffer[mask, -1] = value[mask]

    def _seed_history(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        if len(ids) == 0:
            return
        proprio = self.raw_env.get_proprioception()[ids]
        goals = self.raw_env.get_goal()[ids]
        self.proprio_history[ids] = proprio[:, None].expand(-1, self.history, -1)
        self.goal_history[ids] = goals[:, None].expand(-1, self.history, -1)
        self.action_history[ids] = 0.0
        self.previous_action[ids] = 0.0

    def _advance(self, proprio_before, goal_before, action, active, done):
        self._slide(self.proprio_history, proprio_before, active)
        self._slide(self.action_history, self.previous_action, active)
        self._slide(self.goal_history, goal_before, active)
        self.previous_action[active] = action[active]
        if torch.any(done):
            self.proprio_history[done] = 0.0
            self.action_history[done] = 0.0
            self.goal_history[done] = 0.0
            self.previous_action[done] = 0.0

    @torch.no_grad()
    def collect_batch(self, *, use_base_noise=False, chunks=None, episode_accumulator=None,
                      progress_label=None):
        episodes = episode_accumulator if episode_accumulator is not None else EpisodeAccumulator()
        chunk_count = self.rollout_chunks if chunks is None else chunks
        for chunk_index in range(chunk_count):
            p0, a0, g0 = self.proprio_history.clone(), self.action_history.clone(), self.goal_history.clone()
            if use_base_noise:
                noise = self.agent.base_noise(self.num_envs)
                action_chunk = self.agent.decode_noise(p0, a0, g0, noise)
            else:
                noise, action_chunk, _ = self.agent.act(p0, a0, g0)
            macro_reward = torch.zeros(self.num_envs, device=self.device)
            macro_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_ids = torch.zeros_like(macro_done)
            for step in range(self.agent.cfg.exec_horizon):
                active = ~macro_done
                p_before, g_before = self.raw_env.get_proprioception().clone(), self.raw_env.get_goal().clone()
                action = torch.where(active[:, None], action_chunk[:, step], torch.zeros_like(action_chunk[:, step]))
                _, reward, terminated, truncated, extras = self.env.step(action)
                done_now = (terminated | truncated) & active
                # Official ActionChunkWrapper sums rewards; gamma applies once
                # per decoded action chunk in the SAC Bellman target.
                macro_reward += reward * active.float()
                self.running_return += reward * active.float()
                if torch.any(done_now):
                    episodes.returns.extend(self.running_return[done_now].cpu().tolist())
                    self.running_return[done_now] = 0.0
                episodes.add_extras(extras, active)
                self._advance(p_before, g_before, action, active, done_now)
                macro_done |= done_now
                if step + 1 < self.agent.cfg.exec_horizon:
                    reset_ids |= done_now
                self.control_steps += self.num_envs
            if torch.any(reset_ids):
                self.raw_env.reset_after_action_chunk(torch.nonzero(reset_ids, as_tuple=False).squeeze(-1))
            self._seed_history(torch.nonzero(macro_done, as_tuple=False).squeeze(-1))
            self.replay.add_batch(
                p0, a0, g0, action_chunk.flatten(1), macro_reward, macro_done,
                self.proprio_history, self.action_history, self.goal_history,
            )
            self.vector_chunks += 1
            if progress_label and ((chunk_index + 1) % 100 == 0 or chunk_index + 1 == chunk_count):
                print(f"[DSRL-NA {progress_label}] chunks={chunk_index + 1}/{chunk_count} "
                      f"episodes={episodes.count:.0f} replay={self.replay.size}", flush=True)
        return episodes.metrics()

    def save(self, name, iteration, metrics):
        if source_semantic_digest(self.agent.decoder.prior) != self.prior_digest:
            raise RuntimeError("The pretrained diffusion prior changed during DSRL-NA training.")
        save_dsrl_na_checkpoint(
            self.output_dir / name, source=self.source, source_path=self.source_path,
            source_digest=self.prior_digest,
            source_sha=self.source_sha, agent=self.agent, iteration=iteration,
            total_control_transitions=self.control_steps, metrics=metrics,
            task_config=task_config_from_env_cfg(self.raw_env.cfg),
        )

    def run(self, iterations):
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        self.env.reset()
        self._seed_history()
        self.save("initial.pt", -1, {"phase": "untrained_actor"})
        stats = self.collect_batch(use_base_noise=True, chunks=self.agent.cfg.initial_rollout_chunks,
                                   progress_label="initialization")
        (self.output_dir / "baseline_metrics.json").write_text(
            json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"[DSRL-NA base rollout] success={stats['Episode/success_rate']:.3f} "
              f"fall={stats['Episode/base_contact_rate']:.3f} "
              f"episodes={stats['Episode/completed']:.0f}", flush=True)
        if self.replay.size < self.agent.cfg.batch_size:
            raise RuntimeError("Initial standard-noise rollouts did not fill a DSRL minibatch.")
        metric_path = self.output_dir / "metrics.jsonl"
        window_episodes = EpisodeAccumulator()
        for iteration in range(iterations):
            started = time.perf_counter()
            episodes = self.collect_batch(use_base_noise=False, episode_accumulator=window_episodes)
            update_metrics = self.agent.update(self.replay)
            metrics = {
                **episodes, **update_metrics,
                "iteration": iteration,
                "replay_size": self.replay.size,
                "total_control_transitions": self.control_steps,
                "total_macro_transitions": self.vector_chunks * self.num_envs,
                "episode_metric_window_iterations": iteration % 100 + 1,
                "iteration_seconds": time.perf_counter() - started,
                "alpha": float(self.agent.log_alpha.detach().exp()),
            }
            with metric_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
            if self.logger is not None:
                self.logger.log(metrics, step=iteration)
            print(f"[DSRL-NA {iteration:04d}] success={episodes['Episode/success_rate']:.3f} "
                  f"fall={episodes['Episode/base_contact_rate']:.3f} "
                  f"return={episodes['Episode/return']:.3f} "
                  f"alpha={metrics['alpha']:.4f} replay={self.replay.size} "
                  f"qA={metrics.get('DSRL/q_action_loss', float('nan')):.4g} "
                  f"qW={metrics.get('DSRL/q_noise_loss', float('nan')):.4g}", flush=True)
            if (iteration + 1) % self.save_interval == 0:
                self.save(f"model_{iteration + 1}.pt", iteration, metrics)
            if (iteration + 1) % 100 == 0:
                window_episodes = EpisodeAccumulator()
        self.save("last.pt", iterations - 1, metrics)
