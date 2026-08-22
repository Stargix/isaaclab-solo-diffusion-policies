"""Short simulator check for the 4-D high-level / 12-D low-level contract."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_inference_steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg


task = "solo12-hierarchical-diffuseloco-v0"
cfg = parse_env_cfg(task, device=args.device, num_envs=2)
cfg.diffuseloco_checkpoint = args.checkpoint
cfg.diffuseloco_inference_steps = args.num_inference_steps
env = gym.make(task, cfg=cfg)
observation, _ = env.reset()
raw = env.unwrapped
assert observation["policy"].shape == (2, cfg.observation_space)
assert raw.single_action_space.shape == (4,)
assert raw._actions.shape == (2, 12)
action = torch.zeros((2, 4), device=raw.device)
for step_index in range(8):
    observation, reward, terminated, truncated, _ = env.step(action)
    assert torch.isfinite(observation["policy"]).all()
    assert torch.isfinite(reward).all()
    assert terminated.shape == truncated.shape == (2,)
print(
    f"[PASS] obs={tuple(observation['policy'].shape)} external_action=4 "
    f"joint_action={raw._actions.shape[1]} reward={reward.tolist()}",
    flush=True,
)
env.close()
simulation_app.close()
