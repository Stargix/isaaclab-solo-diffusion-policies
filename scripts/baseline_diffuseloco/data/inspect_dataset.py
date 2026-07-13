# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to inspect and visualize trajectories inside a collected HDF5 dataset.
Plots base height, command speeds, projected gravity, and joint positions.

Example usage:
python scripts/baseline_diffuseloco/data/inspect_dataset.py --dataset scripts/baseline_diffuseloco/data/datasets/walk_raw.hdf5 --demo demo_0
"""

import argparse
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Disable HDF5 file locking to avoid Win32 GetLastError() = 33 / locked files on Windows
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def main():
    parser = argparse.ArgumentParser(description="Inspect and plot trajectories in an HDF5 dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset file.")
    parser.add_argument("--demo", type=str, default=None, help="Name of the demo to inspect (e.g. demo_0). If None, selects a random demo.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the inspection plot. Defaults to dataset directory.")
    args = parser.parse_args()

    args.dataset = os.path.abspath(args.dataset)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found at: {args.dataset}")

    # Open HDF5 file
    with h5py.File(args.dataset, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 file does not contain a 'data' group. Is this a RoboMimic-formatted file?")

        demos = list(f["data"].keys())
        num_demos = len(demos)
        print(f"\n==================================================")
        print(f"INSPECTING DATASET: {args.dataset}")
        print(f"==================================================")
        print(f"Total number of demos: {num_demos}")
        
        if num_demos == 0:
            print("[WARN] No demos found in this dataset.")
            return

        # Select demo
        selected_demo = args.demo
        if selected_demo is None:
            selected_demo = random_choice = np.random.choice(demos)
            print(f"No specific --demo provided. Selected random demo: '{selected_demo}'")
        elif selected_demo not in demos:
            print(f"[ERROR] Demo '{selected_demo}' not found. Available demos: {demos[:10]} ...")
            return
        
        demo_grp = f[f"data/{selected_demo}"]
        num_samples = demo_grp.attrs.get("num_samples", len(demo_grp["dones"]))
        print(f"\nProperties of '{selected_demo}':")
        print(f"  - Length (steps): {num_samples} (or {num_samples * 0.02:.2f} seconds at 50Hz)")
        
        # Read keys
        obs_grp = demo_grp["obs"]
        print(f"  - Observation fields: {list(obs_grp.keys())}")
        
        # Load arrays to memory
        joint_pos = obs_grp["joint_pos"][:]
        base_ang_vel = obs_grp["base_ang_vel"][:]
        projected_gravity = obs_grp["projected_gravity"][:]
        root_pos_w = obs_grp["root_pos_w"][:]
        root_quat_w = obs_grp["root_quat_w"][:]
        command_speed = obs_grp["command_speed"][:]
        actions = demo_grp["actions"][:]
        dones = demo_grp["dones"][:]

    # Create visualization plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    time = np.arange(num_samples) * 0.02  # 50Hz control rate

    # Plot 1: Base Height Z
    heights = root_pos_w[:, 2]
    axs[0].plot(time, heights, label="Base Height (Z)", color="blue", linewidth=2)
    # Add target references
    axs[0].axhline(y=0.24, color="green", linestyle="--", alpha=0.7, label="Nominal Walk Height (24cm)")
    axs[0].axhline(y=0.16, color="red", linestyle="--", alpha=0.7, label="Nominal Crouch Height (16cm)")
    axs[0].set_ylabel("Height [m]")
    axs[0].set_title(f"Trajectory Visualization - {selected_demo} ({args.dataset})")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper left", ncol=3, framealpha=0.5, fontsize=8)

    # Plot 2: Speed Commands
    axs[1].plot(time, command_speed[:, 0], label="Cmd Vx (forward)", color="red")
    axs[1].plot(time, command_speed[:, 1], label="Cmd Vy (sideways)", color="orange")
    axs[1].plot(time, command_speed[:, 2], label="Cmd Wz (turn)", color="purple")
    axs[1].set_ylabel("Speed Cmds [m/s, rad/s]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper left", ncol=3, framealpha=0.5, fontsize=8)

    # Plot 3: Projected Gravity (indicates body orientation / tilt)
    axs[2].plot(time, projected_gravity[:, 0], label="Gravity X (pitch)", color="brown")
    axs[2].plot(time, projected_gravity[:, 1], label="Gravity Y (roll)", color="pink")
    axs[2].plot(time, projected_gravity[:, 2], label="Gravity Z (yaw plane)", color="olive")
    axs[2].set_ylabel("Proj Gravity [g]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper left", ncol=3, framealpha=0.5, fontsize=8)

    # Plot 4: Front Left (FL) Leg Joint Positions
    axs[3].plot(time, joint_pos[:, 0], label="FL Hip", color="teal")
    axs[3].plot(time, joint_pos[:, 1], label="FL Thigh", color="magenta")
    axs[3].plot(time, joint_pos[:, 2], label="FL Calf", color="cyan")
    axs[3].set_ylabel("Joint Positions [rad]")
    axs[3].set_xlabel("Time [seconds]")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper left", ncol=3, framealpha=0.5, fontsize=8)

    # Adjust layout (restored to full width)
    plt.tight_layout()
    
    # Save the figure
    dataset_path = Path(args.dataset)
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        # Default to a plots directory beside this baseline utility.
        save_dir = dataset_path.parent.parent / "plots"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{dataset_path.stem}_{selected_demo}_inspection.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nInspection plot successfully saved to:\n   {save_path}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
