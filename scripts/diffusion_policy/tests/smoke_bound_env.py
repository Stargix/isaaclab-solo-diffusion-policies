"""Runtime smoke test for ``solo12-bound-v0`` without remote terrain assets."""

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
from isaaclab_tasks.direct.solo12.solo12_bound_env import Solo12BoundEnvCfg
from isaaclab_tasks.direct.solo12.solo12_env_cfg import SOLO12_TRICKY_TERRAINS_CFG
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    agent_cfg = load_cfg_from_registry("solo12-bound-v0", "rsl_rl_cfg_entry_point")
    assert agent_cfg.num_steps_per_env == 32
    assert agent_cfg.max_iterations == 3000
    assert agent_cfg.algorithm.learning_rate == 3.0e-4

    cfg = Solo12BoundEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or cfg.sim.device
    # Exercise the complete timeout -> reset path during this short test.
    cfg.episode_length_s = 0.08

    # TerrainImporter "plane" references a remote NVIDIA USD. A one-tile flat
    # mesh exercises identical contacts while keeping this smoke test offline.
    terrain_generator = copy.deepcopy(SOLO12_TRICKY_TERRAINS_CFG)
    terrain_generator.curriculum = False
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.sub_terrains = {"flat": terrain_generator.sub_terrains["flat"]}
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = terrain_generator

    env = gym.make("solo12-bound-v0", cfg=cfg)
    try:
        observations, _ = env.reset()
        policy_obs = observations["policy"]
        assert policy_obs.shape == (args.num_envs, 50), policy_obs.shape
        assert torch.all(env.unwrapped._commands[:, 0] >= 0.0)
        assert torch.all(env.unwrapped._commands[:, 0] <= 1.5)

        expected_speed_stages = ((0.60, 1.00), (0.70, 1.25), (0.80, 1.50))
        for stage, expected_range in enumerate(expected_speed_stages):
            env.unwrapped._set_max_velx_range_curriculum_level(stage)
            assert env.unwrapped.cfg.command_lin_vel_x_range == expected_range
        env.unwrapped._set_max_velx_range_curriculum_level(0)

        for _ in range(args.steps):
            actions = torch.zeros((args.num_envs, 12), device=env.unwrapped.device)
            observations, reward, terminated, truncated, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 50)
            assert torch.all(torch.isfinite(observations["policy"]))
            assert torch.all(torch.isfinite(reward))
            assert torch.all(reward >= 0.0)
            assert torch.all(reward <= 3.5 * env.unwrapped.step_dt + 1.0e-6)
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)

        required_metrics = {
            "Metrics/velocity_tracking_score",
            "Metrics/bound_contact_score",
            "Metrics/front_pair_sync",
            "Metrics/rear_pair_sync",
            "Metrics/front_rear_opposition",
        }
        assert required_metrics.issubset(env.unwrapped.extras["log"])

        print("[PASS] solo12-bound-v0 reset/observation/reward/step contract", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
