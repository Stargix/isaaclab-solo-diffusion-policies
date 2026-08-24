from __future__ import annotations

from types import SimpleNamespace
import math

import torch

from scripts.dppo_diffusion_rl.policy import DPPOSample
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer


class _FakeRawEnv:
    def __init__(self) -> None:
        self.proprio = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.goal = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    def get_proprioception(self) -> torch.Tensor:
        return self.proprio

    def get_goal(self) -> torch.Tensor:
        return self.goal


def test_seed_history_matches_padded_start_contract() -> None:
    trainer = object.__new__(DPPOTrainer)
    trainer.raw_env = _FakeRawEnv()
    trainer.num_envs = 2
    trainer.history = 3
    trainer.device = torch.device("cpu")
    trainer.proprio_history = torch.zeros(2, 3, 2)
    trainer.goal_history = torch.zeros(2, 3, 2)
    trainer.action_history = torch.ones(2, 3, 2)
    trainer.previous_applied_action = torch.ones(2, 2)

    trainer._seed_history(torch.tensor([1]))

    torch.testing.assert_close(
        trainer.proprio_history[1], torch.tensor([[3.0, 4.0]]).expand(3, -1)
    )
    torch.testing.assert_close(
        trainer.goal_history[1], torch.tensor([[7.0, 8.0]]).expand(3, -1)
    )
    torch.testing.assert_close(trainer.action_history[1], torch.zeros(3, 2))
    torch.testing.assert_close(trainer.previous_applied_action[1], torch.zeros(2))
    torch.testing.assert_close(trainer.proprio_history[0], torch.zeros(3, 2))


class _ChunkEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.num_envs = 2
        self.device = "cpu"
        self.state = torch.zeros(2, 1)
        self.substep = 0
        self.boundary_resets: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def get_proprioception(self) -> torch.Tensor:
        return self.state.clone()

    def get_goal(self) -> torch.Tensor:
        return self.state + 100.0

    def get_critic_features(self) -> torch.Tensor:
        return torch.zeros(2, 1)

    def step(self, action: torch.Tensor):
        self.actions.append(action.clone())
        self.state += 1.0
        terminated = torch.tensor([self.substep == 0, False])
        truncated = torch.zeros(2, dtype=torch.bool)
        if terminated[0]:
            # Isaac auto-reset state. Remaining actions in this macro step are
            # padding and must not become the next policy observation.
            self.state[0] = 10.0
        self.substep += 1
        return {}, torch.ones(2), terminated, truncated, {}

    def reset_after_action_chunk(self, env_ids: torch.Tensor) -> None:
        self.boundary_resets.append(env_ids.clone())
        self.state[env_ids] = 20.0


class _ChunkPolicy:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            history=2,
            proprio_dim=1,
            action_hist_dim=1,
            goal_dim=1,
            action_dim=1,
        )
        self.dppo_cfg = SimpleNamespace(exec_horizon=4, gamma=1.0, gae_lambda=0.95)

    def critic_condition(self, proprio, action, goal, privileged):
        return torch.cat(
            (proprio.flatten(1), action.flatten(1), goal.flatten(1), privileged), dim=1
        )

    def sample(self, proprio, action, goal) -> DPPOSample:
        trajectory = torch.ones(2, 4, 1)
        return DPPOSample(
            normalized_trajectory=trajectory,
            denormalized_trajectory=trajectory,
            chain=torch.zeros(2, 2, 4, 1),
            old_logprobs=torch.zeros(2, 1),
        )

    def executable_chunk(self, trajectory):
        return trajectory


class _ZeroCritic:
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(observation))


def test_collect_discards_post_terminal_chunk_padding(tmp_path) -> None:
    env = _ChunkEnv()
    trainer = DPPOTrainer(
        env=env,
        policy=_ChunkPolicy(),
        critic=_ZeroCritic(),
        updater=object(),
        source_checkpoint={},
        source_checkpoint_path="source.pt",
        output_dir=tmp_path,
        rollout_chunks=1,
        save_interval=1,
    )
    trainer._seed_history()
    rollout, _ = trainer.collect()

    torch.testing.assert_close(rollout.rewards[0], torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(rollout.dones[0], torch.tensor([True, False]))
    assert len(env.boundary_resets) == 1
    torch.testing.assert_close(env.boundary_resets[0], torch.tensor([0]))
    # Inactive env zero-actions were applied only as simulator padding.
    assert all(float(action[0]) == 0.0 for action in env.actions[1:])
    torch.testing.assert_close(trainer.proprio_history[0], torch.full((2, 1), 20.0))
    torch.testing.assert_close(trainer.goal_history[0], torch.full((2, 1), 120.0))


def test_selection_score_penalizes_overshoot_and_terminal_distance() -> None:
    common = {
        "Episode/completed": 1.0,
        "Episode/success_rate": 0.0,
        "Episode/base_contact_rate": 0.0,
        "Episode/corridor_failure_rate": 0.0,
        "Episode/progress_fraction": 1.0,
        "Episode/mean_speed_error_abs_mps": 0.1,
    }
    near = DPPOTrainer._selection_score(
        {**common, "Episode/terminal_overshoot_rate": 0.0, "Episode/terminal_distance_m": 0.2}
    )
    far = DPPOTrainer._selection_score(
        {**common, "Episode/terminal_overshoot_rate": 0.0, "Episode/terminal_distance_m": 2.0}
    )
    overshoot = DPPOTrainer._selection_score(
        {**common, "Episode/terminal_overshoot_rate": 1.0, "Episode/terminal_distance_m": 0.2}
    )
    assert near > far > overshoot


def test_selection_score_prefers_accurate_first_arrival() -> None:
    common = {
        "Episode/completed": 1.0,
        "Episode/success_rate": 0.0,
        "Episode/base_contact_rate": 0.0,
        "Episode/corridor_failure_rate": 0.0,
        "Episode/terminal_overshoot_rate": 0.0,
        "Episode/progress_fraction": 1.0,
        "Episode/terminal_distance_m": 0.1,
    }
    accurate = DPPOTrainer._selection_score(
        {
            **common,
            "Episode/arrival_failure_rate": 0.0,
            "Episode/mean_speed_error_abs_mps": 0.02,
        }
    )
    early = DPPOTrainer._selection_score(
        {
            **common,
            "Episode/arrival_failure_rate": 1.0,
            "Episode/mean_speed_error_abs_mps": 0.20,
        }
    )
    assert accurate > early


def test_historical_best_score_is_only_reused_in_place(tmp_path) -> None:
    source = {"algorithm": "dppo", "best_score": 3.0}
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    new_trainer = DPPOTrainer(
        env=_ChunkEnv(), policy=_ChunkPolicy(), critic=_ZeroCritic(), updater=object(),
        source_checkpoint=source, source_checkpoint_path=old_dir / "last.pt",
        output_dir=tmp_path / "new", rollout_chunks=1, save_interval=1,
    )
    assert math.isinf(new_trainer.best_score) and new_trainer.best_score < 0.0
    same_trainer = DPPOTrainer(
        env=_ChunkEnv(), policy=_ChunkPolicy(), critic=_ZeroCritic(), updater=object(),
        source_checkpoint=source, source_checkpoint_path=old_dir / "last.pt",
        output_dir=old_dir, rollout_chunks=1, save_interval=1,
    )
    assert same_trainer.best_score == 3.0
