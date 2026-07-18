# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to find the average height of a trained RSL-RL policy (100% native PyTorch, zero rsl_rl/tensordict dependency)."""

import argparse
import sys
import os

# Import torch first before launching AppLauncher to prevent Windows access violations
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Find the average height of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="solo12-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
parser.add_argument("--steps", type=int, default=500, help="Number of steps to evaluate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Force headless to avoid GUI popups and run faster
sys.argv = sys.argv + ["--headless"]

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class SimpleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(48, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 12)
        )
        self.mean = nn.Parameter(torch.zeros(1, 48), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1, 48), requires_grad=False)

    def forward(self, obs):
        obs = (obs - self.mean) / (self.std + 1e-8)
        obs = torch.clamp(obs, -5.0, 5.0)
        return self.actor(obs)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    device = torch.device(env_cfg.sim.device)
    model = SimpleActor().to(device)

    # Load state dict
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state_dict[k[6:]] = v
        elif k == "actor_obs_normalizer._mean":
            model.mean.copy_(v)
        elif k == "actor_obs_normalizer._std":
            model.std.copy_(v)

    model.actor.load_state_dict(actor_state_dict)
    model.eval()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    raw_env = env.unwrapped

    heights = []
    print("[INFO] Simulating to collect base heights...")
    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = model(obs)
            if agent_cfg.clip_actions is not None:
                actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
            # Collect z-coordinate of root_pos_w of all envs
            z_heights = raw_env._robot.data.root_pos_w[:, 2]
            heights.extend(z_heights.cpu().numpy().tolist())

    if len(heights) > 0:
        avg_height = sum(heights) / len(heights)
        print(f"\n==========================================")
        print(f"RESULT: Average base height over {args_cli.steps} steps is: {avg_height:.4f} meters")
        print(f"==========================================\n")
    else:
        print("[ERROR] No heights collected.")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
