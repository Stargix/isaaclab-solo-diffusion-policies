"""Runtime contract test for ``solo12-bound-v2`` without remote terrain assets."""

from __future__ import annotations

import argparse
import copy

# On Windows, load tensordict before Kit starts (same ordering as the RSL train script).
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
from isaaclab_tasks.direct.solo12.bound_gait import (
    bound_pair_desynchronization,
    diagonal_contact_probability,
)
from isaaclab_tasks.direct.solo12.solo12_bound_v2_env import Solo12BoundV2EnvCfg
from isaaclab_tasks.direct.solo12.solo12_env_cfg import SOLO12_TRICKY_TERRAINS_CFG
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _test_contact_topology_helpers() -> None:
    perfect_bound = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    diagonal_trot = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(bound_pair_desynchronization(perfect_bound), torch.zeros(2))
    assert torch.allclose(diagonal_contact_probability(perfect_bound), torch.zeros(2))
    assert torch.allclose(bound_pair_desynchronization(diagonal_trot), torch.ones(2))
    assert torch.allclose(diagonal_contact_probability(diagonal_trot), torch.ones(2))


def main() -> None:
    _test_contact_topology_helpers()

    agent_cfg = load_cfg_from_registry("solo12-bound-v2", "rsl_rl_cfg_entry_point")
    assert agent_cfg.experiment_name == "solo12_rsl_rl_bound_v2_runs"
    assert agent_cfg.algorithm.learning_rate == 3.0e-4

    cfg = Solo12BoundV2EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or cfg.sim.device
    # Exercise timeout and inherited reset logging during this short test.
    cfg.episode_length_s = 0.08

    terrain_generator = copy.deepcopy(SOLO12_TRICKY_TERRAINS_CFG)
    terrain_generator.curriculum = False
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.sub_terrains = {"flat": terrain_generator.sub_terrains["flat"]}
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = terrain_generator

    env = gym.make("solo12-bound-v2", cfg=cfg)
    try:
        observations, _ = env.reset()
        raw_env = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 50)
        assert torch.all(raw_env._commands[:, 0] >= 0.0)
        assert torch.count_nonzero(raw_env._commands[:, 1:]) == 0

        expected_speed_stages = ((0.60, 1.00), (0.70, 1.25), (0.80, 1.50))
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
            "Metrics/forward_tracking_score",
            "Metrics/lateral_speed_abs_mps",
            "Metrics/heading_error_abs_rad",
            "Metrics/exact_desired_contact_fraction",
            "Metrics/bound_pair_pattern_fraction",
            "Metrics/diagonal_pattern_fraction",
            "Metrics/pair_desynchronization",
        }
        assert required_metrics.issubset(raw_env.extras["log"])
        print("[PASS] solo12-bound-v2 reset/observation/reward/step contract", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
