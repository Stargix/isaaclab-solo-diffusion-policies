"""Runtime and symmetry contract test for ``solo12-bound-v3``."""

from __future__ import annotations

import argparse
import copy

from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.solo12.bound_symmetry import (
    compute_bound_left_right_symmetry,
    reflect_bound_actions_left_right,
    reflect_bound_observations_left_right,
)
from isaaclab_tasks.direct.solo12.solo12_bound_v3_env import Solo12BoundV3EnvCfg
from isaaclab_tasks.direct.solo12.solo12_env_cfg import SOLO12_TRICKY_TERRAINS_CFG
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _test_symmetry_contract(device: str) -> None:
    observations = torch.randn(5, 50, device=device)
    actions = torch.randn(5, 12, device=device)
    reflected_observations = reflect_bound_observations_left_right(observations)
    reflected_actions = reflect_bound_actions_left_right(actions)

    assert torch.allclose(reflect_bound_observations_left_right(reflected_observations), observations)
    assert torch.allclose(reflect_bound_actions_left_right(reflected_actions), actions)
    assert torch.equal(reflected_observations[:, 48:50], observations[:, 48:50])

    observations_augmented, actions_augmented = compute_bound_left_right_symmetry(
        obs=observations, actions=actions
    )
    assert observations_augmented.shape == (10, 50)
    assert actions_augmented.shape == (10, 12)
    assert torch.allclose(observations_augmented[:5], observations)
    assert torch.allclose(actions_augmented[:5], actions)


def main() -> None:
    agent_cfg = load_cfg_from_registry("solo12-bound-v3", "rsl_rl_cfg_entry_point")
    assert agent_cfg.experiment_name == "solo12_rsl_rl_bound_v3_runs"
    assert agent_cfg.algorithm.learning_rate == 3.0e-4

    cfg = Solo12BoundV3EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or cfg.sim.device
    cfg.episode_length_s = 0.08

    terrain_generator = copy.deepcopy(SOLO12_TRICKY_TERRAINS_CFG)
    terrain_generator.curriculum = False
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.sub_terrains = {"flat": terrain_generator.sub_terrains["flat"]}
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = terrain_generator

    env = gym.make("solo12-bound-v3", cfg=cfg)
    try:
        observations, _ = env.reset()
        raw_env = env.unwrapped
        _test_symmetry_contract(raw_env.device)

        assert observations["policy"].shape == (args.num_envs, 50)
        assert torch.all(raw_env._commands[:, 0] >= 0.90)
        assert torch.all(raw_env._commands[:, 0] <= 1.15)
        assert torch.count_nonzero(raw_env._commands[:, 1:]) == 0

        expected_speed_stages = ((0.90, 1.15), (1.00, 1.30), (1.10, 1.50))
        for stage, expected_range in enumerate(expected_speed_stages):
            raw_env._set_max_velx_range_curriculum_level(stage)
            assert raw_env.cfg.command_lin_vel_x_range == expected_range
        raw_env._set_max_velx_range_curriculum_level(0)

        for _ in range(args.steps):
            actions = torch.zeros((args.num_envs, 12), device=raw_env.device)
            observations, reward, terminated, truncated, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 50)
            assert torch.all(torch.isfinite(observations["policy"]))
            assert torch.all(torch.isfinite(reward))
            assert torch.all(reward >= 0.0)
            assert torch.all(reward <= 3.5 * raw_env.step_dt + 1.0e-6)
            assert torch.count_nonzero(raw_env._commands[:, 1:]) == 0
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)

        required_metrics = {
            "Metrics/direction_quality",
            "Metrics/forward_tracking_exp_score",
            "Metrics/forward_progress_score",
            "Metrics/front_action_mirror_mae",
            "Metrics/rear_action_mirror_mae",
            "Metrics/forward_tracking_score",
            "Metrics/bound_pair_pattern_fraction",
            "Metrics/diagonal_pattern_fraction",
        }
        assert required_metrics.issubset(raw_env.extras["log"])
        print("[PASS] solo12-bound-v3 environment and symmetry contracts", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
