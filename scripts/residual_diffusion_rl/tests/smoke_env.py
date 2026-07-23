"""One-step Isaac Sim smoke test for the registered B1 task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    task = "solo12-residual-diffusion-rl-v0"
    print("[SMOKE] importing task and parsing config", flush=True)
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1)
    cfg.spatial_diffusion_checkpoint = str(Path(args.checkpoint).resolve())
    cfg.diffusion_inference_steps = 2
    env = gym.make(task, cfg=cfg)
    try:
        observation, _ = env.reset()
        assert observation["policy"].shape == (1, 77)
        action = torch.zeros((1, 12), device=env.unwrapped.device)
        observation, reward, terminated, truncated, _ = env.step(action)
        assert observation["policy"].shape == (1, 77)
        assert torch.isfinite(observation["policy"]).all()
        assert torch.isfinite(reward).all()
        # stderr remains visible after Kit replaces its stdout logger.
        print("[PASS] B1 task registered, reset and stepped with finite tensors", file=sys.stderr, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
