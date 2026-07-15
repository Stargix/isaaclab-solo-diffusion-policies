# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play back a collected trajectory in the Isaac Sim GUI.
Runs a single environment (num_envs=1) and replays recorded states/actions.

Example usage:
python scripts/diffusion_policy/data/play_trajectory.py --task="solo12-v0" --dataset scripts/diffusion_policy/data/datasets/walk_raw.hdf5 --demo demo_0
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Any

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

# Launch omniverse app with GUI enabled
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded expert trajectory visually.")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. solo12-v0).")
parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset file.")
parser.add_argument("--demo", type=str, default=None, help="Name of the demo to replay (e.g. demo_0). Defaults to first demo.")
parser.add_argument("--mode", type=str, choices=["action", "kinematic"], default="kinematic", 
                    help="Playback mode: open-loop 'action' replay or 'kinematic' state replay.")
parser.add_argument("--num_loops", type=int, default=1, help="Number of times to loop the playback.")

# Add standard app launcher CLI arguments (GUI is enabled by default here)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Strip parsed arguments from sys.argv so Hydra doesn't crash on them
sys.argv = [sys.argv[0]] + hydra_args

# Force headless = False to show the GUI window unless explicitly overridden
args_cli.headless = False if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import libraries requiring active sim
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any):
    # Normalize paths
    dataset_path = os.path.abspath(args_cli.dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # Read trajectory actions and states from HDF5
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        selected_demo = args_cli.demo if args_cli.demo else demos[0]
        if selected_demo not in demos:
            raise KeyError(f"Demo '{selected_demo}' not found in dataset. Available: {demos[:10]} ...")
            
        print(f"[INFO] Loading '{selected_demo}' from {dataset_path}")
        demo_grp = f[f"data/{selected_demo}"]
        actions_seq = demo_grp["actions"][:]
        commands_seq = demo_grp["obs/command_speed"][:]
        
        # Load physical states for kinematic replay
        root_pos_w = demo_grp["obs/root_pos_w"][:]
        
        root_quat_w = demo_grp["obs/root_quat_w"][:]
        joint_pos_seq = demo_grp["obs/joint_pos"][:]
        joint_vel_seq = demo_grp["obs/joint_vel"][:]

    # Override environment settings for single visual environment
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = 1.0e9  # Disable automatic resets
    
    # Disable randomizations for clean playback
    if getattr(env_cfg, "events", None):
        env_cfg.events = None
    if hasattr(env_cfg, "enable_observation_corruption"):
        env_cfg.enable_observation_corruption = False
    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)

    # Build environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    device = torch.device(vec_env.unwrapped.device)
    actions_tensor = torch.tensor(actions_seq, dtype=torch.float32, device=device)
    num_steps = actions_tensor.shape[0]

    # Convert logged states to torch tensors for kinematic playback
    root_pos_t = torch.tensor(root_pos_w, dtype=torch.float32, device=device)
    root_quat_t = torch.tensor(root_quat_w, dtype=torch.float32, device=device)
    joint_pos_t = torch.tensor(joint_pos_seq, dtype=torch.float32, device=device)
    joint_vel_t = torch.tensor(joint_vel_seq, dtype=torch.float32, device=device)

    # Control rates
    dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt
    print(f"[INFO] Environment built. Replaying {num_steps} steps at {1/dt:.1f}Hz in '{args_cli.mode}' mode...")

    # Visual camera follow setup
    camera_look_at = np.array([0.0, 0.0, 0.35])
    camera_offset = np.array([-2.0, 0.0, 0.8])

    for loop in range(args_cli.num_loops):
        print(f"\n🔁 Starting Loop {loop + 1}/{args_cli.num_loops}...")
        
        # Reset env to starting pose
        obs = vec_env.reset()
        # Capture the initial starting position from the reset state
        start_pos = raw_env._robot.data.root_pos_w[0].clone()
        time.sleep(0.5)

        for t in range(num_steps):
            if not simulation_app.is_running():
                break
                
            if args_cli.mode == "action":
                # Step environment using recorded expert actions (open-loop PD control)
                action_step = actions_tensor[t : t + 1] # shape (1, 12)
                obs, _, _, _ = vec_env.step(action_step)
            else:
                # Kinematic playback (override simulator state to match recorded physical trajectory)
                curr_root_pos = root_pos_t[t] - root_pos_t[0] + start_pos
                root_pose = torch.cat([curr_root_pos, root_quat_t[t]], dim=-1).unsqueeze(0) # (1, 7)
                j_pos = joint_pos_t[t].unsqueeze(0) # (1, 12)
                j_vel = joint_vel_t[t].unsqueeze(0) # (1, 12)
                
                # Force state in simulation (specifying the exact joint IDs)
                raw_env._robot.write_root_pose_to_sim(root_pose, env_ids=torch.tensor([0], device=device))
                raw_env._robot.write_joint_state_to_sim(
                    j_pos, j_vel, joint_ids=raw_env._joint_ids, env_ids=torch.tensor([0], device=device)
                )
                
                # Write back buffers and render the visual frame without stepping physics
                raw_env.scene.write_data_to_sim()
                raw_env.sim.render()
                raw_env.scene.update(dt=raw_env.physics_dt)
            
            # Update camera to follow the robot
            try:
                root_pos = raw_env._robot.data.root_pos_w[0].cpu().numpy()
                eye_pos = root_pos + camera_offset
                target_pos = root_pos + camera_look_at
                raw_env.sim.set_camera_view(eye=eye_pos, target=target_pos)
            except Exception:
                pass
            
            # Print status periodically
            if t % 100 == 0:
                cmd = commands_seq[t]
                print(f"Step {t:03d}/{num_steps} | Active Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")

            # Keep real-time speed
            time.sleep(dt)

    print("[INFO] Playback finished successfully.")
    vec_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
