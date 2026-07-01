# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to collect expert demonstrations from trained RSL-RL policies.
Supports single-policy collection (Dataset A) and skill-chaining transitions (Dataset B).

Example usage:
# Single mode (Dataset A)
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" --checkpoint "checkpoints/walk_safe.pt" --num_envs 64 --num_steps 50000 --output_name walk_raw.hdf5 --headless

# Chained mode (Dataset B)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt --num_envs 128 --num_steps 100000 --output_name chained_raw.hdf5 --headless
"""

import argparse
import os
import sys
import torch
import numpy as np
import h5py
import random
from pathlib import Path
from typing import Dict, List, Any

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

# Launch omniverse app first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect expert trajectory demonstrations in parallel.")
parser.add_argument("--mode", type=str, choices=["single", "chained"], default="single", 
                    help="Collection mode: single policy (Dataset A) or skill chaining (Dataset B).")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. solo12-v0).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to expert checkpoint (required for single mode).")
parser.add_argument("--checkpoints", type=str, nargs="+", default=[], 
                    help="Paths to expert checkpoints: walk crouch jump (required for chained mode).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=50000, help="Total number of timesteps to collect across all envs.")
parser.add_argument("--output_name", type=str, default="raw_dataset.hdf5", help="Output file name.")
parser.add_argument("--command_resample_time_s", type=float, default=4.0, help="Time interval (in s) to resample speed commands.")
parser.add_argument("--min_demo_len", type=int, default=100, help="Minimum step length to save an episode.")
parser.add_argument("--min_steps_per_skill", type=int, default=150, help="Minimum steps to run a skill in chained mode before transition.")
parser.add_argument("--max_steps_per_skill", type=int, default=300, help="Maximum steps to run a skill in chained mode before transition.")
parser.add_argument("--survival_check_steps", type=int, default=100, help="Steps to survive after a transition for the episode to be kept.")

# Add standard app launcher CLI arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Strip parsed arguments from sys.argv so Hydra doesn't crash on them
sys.argv = [sys.argv[0]] + hydra_args

# Forces headless if not explicitly set to False, as data collection is usually run headless
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import libraries requiring active sim
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

def resample_env_command(env_idx: int, policy_name: str, commands_tensor: torch.Tensor, device: torch.device):
    """Generates a policy-specific speed command for a single environment."""
    if policy_name == "walk":
        vx = random.uniform(-1.2, 1.2)
        vy = random.uniform(-0.6, 0.6)
        wz = random.uniform(-0.8, 0.8)
    elif policy_name == "crouch":
        vx = random.uniform(-0.4, 0.4)
        vy = random.uniform(-0.15, 0.15)
        wz = random.uniform(-0.4, 0.4)
    elif policy_name == "jump":
        # Encourage forward leaps
        vx = random.uniform(0.3, 1.0)
        vy = 0.0
        wz = random.uniform(-0.2, 0.2)
    else:
        vx, vy, wz = 0.0, 0.0, 0.0
        
    commands_tensor[env_idx, 0] = vx
    commands_tensor[env_idx, 1] = vy
    commands_tensor[env_idx, 2] = wz

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any):
    # Setup directories
    output_dir = _THIS_DIR / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args_cli.output_name

    # Check mode args
    if args_cli.mode == "single" and not args_cli.checkpoint:
        raise ValueError("Single mode requires the --checkpoint argument.")
    if args_cli.mode == "chained" and len(args_cli.checkpoints) < 2:
        raise ValueError("Chained mode requires at least two paths in --checkpoints.")

    # Override environment settings for clean data collection
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Episode timeout: 20 seconds = 1000 steps at 50Hz
    max_episode_steps = 1000
    env_cfg.episode_length_s = 20.0  
    
    # We will handle command resampling manually in this script to align with active policies
    env_cfg.command_resampling_time_s = 1.0e9  # Disable automatic resampling
    env_cfg.standing_env_prob = 0.0
    
    # Disable training randomizations for cleaner trajectories
    if getattr(env_cfg, "events", None):
        env_cfg.events = None
    if hasattr(env_cfg, "enable_observation_corruption"):
        env_cfg.enable_observation_corruption = False
    if hasattr(env_cfg, "reset_base_lin_vel_range"):
        env_cfg.reset_base_lin_vel_range = (0.0, 0.0)
    if hasattr(env_cfg, "reset_base_ang_vel_range"):
        env_cfg.reset_base_ang_vel_range = (0.0, 0.0)
    if hasattr(env_cfg, "flexed_initial_joint_pos_noise_range"):
        env_cfg.flexed_initial_joint_pos_noise_range = (0.0, 0.0)
    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)
    if hasattr(env_cfg, "forces_applied_to_base_curriculum"):
        env_cfg.forces_applied_to_base_curriculum = []
    
    # Synchronize gains (important for crouch policy in direct/solo12_crouch_env.py)
    if hasattr(env_cfg, "kp") and hasattr(env_cfg, "kd"):
        print(f"[INFO] Initializing environment with Kp={env_cfg.kp}, Kd={env_cfg.kd}")

    # Build environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    device = torch.device(vec_env.unwrapped.device)

    # Load policies
    policies: Dict[str, Any] = {}
    if args_cli.mode == "single":
        resume_path = os.path.abspath(args_cli.checkpoint)
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading expert checkpoint from: {resume_path}")
        runner.load(resume_path)
        policies["walk"] = runner.get_inference_policy(device=device)
    else:
        # Chained mode: load multiple policies
        # Map checkpoints by descriptive names based on filenames
        for cp_path in args_cli.checkpoints:
            abs_cp = os.path.abspath(cp_path)
            name = Path(abs_cp).stem.lower()
            if "walk" in name:
                key = "walk"
            elif "crouch" in name:
                key = "crouch"
            elif "jump" in name:
                key = "jump"
            else:
                key = name
                
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading policy '{key}' from: {abs_cp}")
            runner.load(abs_cp)
            policies[key] = runner.get_inference_policy(device=device)

    policy_names = list(policies.keys())

    # Trajectory buffers and tracking states
    trajectory_buffers: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(args_cli.num_envs)]
    
    # Skill tracking variables for each env
    current_policies = [random.choice(policy_names) for _ in range(args_cli.num_envs)]
    steps_left_in_policy = np.random.randint(args_cli.min_steps_per_skill, args_cli.max_steps_per_skill, size=args_cli.num_envs)
    steps_since_resample = np.zeros(args_cli.num_envs, dtype=int)
    
    # Transition logs to verify survival filtering
    last_transition_step = np.zeros(args_cli.num_envs, dtype=int)
    has_transitioned = np.zeros(args_cli.num_envs, dtype=bool)

    # Initialize environment commands
    for i in range(args_cli.num_envs):
        resample_env_command(i, current_policies[i], raw_env._commands, device)

    # HDF5 file setup
    f = h5py.File(output_path, "w")
    data_group = f.create_group("data")
    demo_counter = 0
    total_steps_saved = 0

    print(f"[INFO] Starting data collection ({args_cli.mode} mode). Saving to {output_path}")
    obs = vec_env.get_observations()
    
    steps_collected = 0
    resample_interval_steps = int(round(args_cli.command_resample_time_s / raw_env.step_dt))
    
    while total_steps_saved < args_cli.num_steps:
        # In single-policy mode, query the single active policy
        if args_cli.mode == "single":
            with torch.inference_mode():
                actions = policies["walk"](obs)
                obs, _, dones, _ = vec_env.step(actions)
        else:
            # In chained mode, batch observations and run active policies
            actions = torch.zeros((args_cli.num_envs, vec_env.num_actions), device=device)
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            
            for name, policy_fn in policies.items():
                env_mask = torch.tensor([cp == name for cp in current_policies], dtype=torch.bool, device=device)
                if env_mask.any():
                    with torch.inference_mode():
                        actions[env_mask] = policy_fn(policy_obs[env_mask])
            
            obs, _, dones, _ = vec_env.step(actions)

        # Query low-level states
        joint_pos = raw_env._robot.data.joint_pos[:, raw_env._joint_ids].cpu().numpy()
        joint_vel = raw_env._robot.data.joint_vel[:, raw_env._joint_ids].cpu().numpy()
        base_ang_vel = raw_env._robot.data.root_ang_vel_b.cpu().numpy()
        projected_gravity = raw_env._robot.data.projected_gravity_b.cpu().numpy()
        root_pos_w = raw_env._robot.data.root_pos_w.cpu().numpy()
        root_quat_w = raw_env._robot.data.root_quat_w.cpu().numpy()
        command = raw_env._commands[:, :3].cpu().numpy()
        actions_np = actions.cpu().numpy()

        # Update environment tracking steps
        steps_since_resample += 1
        if args_cli.mode == "chained":
            steps_left_in_policy -= 1

        # Process each environment transitions and resets
        for i in range(args_cli.num_envs):
            # 1. Save step data to buffer
            trajectory_buffers[i].append({
                "joint_pos": joint_pos[i],
                "joint_vel": joint_vel[i],
                "base_ang_vel": base_ang_vel[i],
                "projected_gravity": projected_gravity[i],
                "root_pos_w": root_pos_w[i],
                "root_quat_w": root_quat_w[i],
                "command": command[i],
                "actions": actions_np[i]
            })

            # 2. Command resampling (periodic speed changes)
            if steps_since_resample[i] >= resample_interval_steps:
                resample_env_command(i, current_policies[i], raw_env._commands, device)
                steps_since_resample[i] = 0

            # 3. Skill Chaining (Chained mode only)
            if args_cli.mode == "chained" and steps_left_in_policy[i] <= 0:
                old_policy = current_policies[i]
                new_policy = random.choice([p for p in policy_names if p != old_policy])
                current_policies[i] = new_policy
                
                # Resettle steps left and resample speed commands
                steps_left_in_policy[i] = random.randint(args_cli.min_steps_per_skill, args_cli.max_steps_per_skill)
                resample_env_command(i, new_policy, raw_env._commands, device)
                steps_since_resample[i] = 0
                
                # Log transition
                last_transition_step[i] = len(trajectory_buffers[i])
                has_transitioned[i] = True

            # 4. Check resets and apply Survival Filtering
            if dones[i]:
                traj = trajectory_buffers[i]
                trajectory_buffers[i] = [] # Reset buffer
                
                # Fall is defined as a reset happening before the timeout (capped at 1000 steps)
                is_fall = len(traj) < 950
                
                # Survival condition:
                # - In single mode: discard if it fell.
                # - In chained mode: discard if it fell within survival_check_steps after a transition.
                survived = True
                if is_fall:
                    if args_cli.mode == "single":
                        survived = False
                    elif has_transitioned[i]:
                        steps_since_last_transition = len(traj) - last_transition_step[i]
                        if steps_since_last_transition < args_cli.survival_check_steps:
                            survived = False
                            print(f"[FILTER] Discarded env {i} due to fall {steps_since_last_transition} steps after transition.")
                
                # Save if survived and has minimum length
                if survived and len(traj) >= args_cli.min_demo_len and total_steps_saved < args_cli.num_steps:
                    demo_name = f"demo_{demo_counter}"
                    demo_grp = data_group.create_group(demo_name)
                    
                    # Convert list of dicts to numpy arrays
                    keys = traj[0].keys()
                    traj_data = {key: np.array([step[key] for step in traj]) for key in keys}
                    
                    obs_grp = demo_grp.create_group("obs")
                    obs_grp.create_dataset("joint_pos", data=traj_data["joint_pos"])
                    obs_grp.create_dataset("joint_vel", data=traj_data["joint_vel"])
                    obs_grp.create_dataset("base_ang_vel", data=traj_data["base_ang_vel"])
                    obs_grp.create_dataset("projected_gravity", data=traj_data["projected_gravity"])
                    obs_grp.create_dataset("root_pos_w", data=traj_data["root_pos_w"])
                    obs_grp.create_dataset("root_quat_w", data=traj_data["root_quat_w"])
                    obs_grp.create_dataset("command_speed", data=traj_data["command"])
                    
                    demo_grp.create_dataset("actions", data=traj_data["actions"])
                    demo_grp.create_dataset("dones", data=np.array([False] * (len(traj) - 1) + [True], dtype=bool))
                    demo_grp.attrs["num_samples"] = len(traj)
                    
                    demo_counter += 1
                    total_steps_saved += len(traj)
                    print(f"Saved {demo_name} (len={len(traj)}, mode={args_cli.mode}). Total steps: {total_steps_saved}/{args_cli.num_steps}")

                # Reset environment tracking states for new episode
                current_policies[i] = random.choice(policy_names)
                steps_left_in_policy[i] = random.randint(args_cli.min_steps_per_skill, args_cli.max_steps_per_skill)
                steps_since_resample[i] = 0
                last_transition_step[i] = 0
                has_transitioned[i] = False
                resample_env_command(i, current_policies[i], raw_env._commands, device)

        steps_collected += 1
        if steps_collected % 500 == 0:
            print(f"[STATUS] Total steps saved: {total_steps_saved}/{args_cli.num_steps}")

    # Close and write HDF5 file
    f.close()
    print(f"[SUCCESS] Data collection complete! Total demos saved: {demo_counter}. Total steps: {total_steps_saved}")
    vec_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
