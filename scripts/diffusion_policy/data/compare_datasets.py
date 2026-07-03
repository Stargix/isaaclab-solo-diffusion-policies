# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to compare multiple HDF5 datasets collected with different blending configurations.
Example usage:
python scripts/diffusion_policy/data/compare_datasets.py --datasets scripts/diffusion_policy/data/datasets/chained_abrupt_raw.hdf5 scripts/diffusion_policy/data/datasets/chained_blend_raw.hdf5 --labels "Abrupto" "Cosine 8"
"""

import argparse
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Disable HDF5 file locking to prevent issues on Windows
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def extract_metrics(dataset_path, label, window_before=15, window_after=25):
    """
    Extracts global and transition-aligned metrics from a single dataset.
    """
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        print(f"[WARN] Dataset not found: {dataset_path}")
        return None

    print(f"\nProcessing: {os.path.basename(dataset_path)} ({label})")
    
    global_action_deltas = []
    global_joint_jerks = []
    
    transition_action_deltas = []
    transition_base_ang_vels = []
    transition_base_tilts = []
    transition_joint_vel_deltas = []
    
    dt = 0.02  # 50Hz control rate
    
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            print(f"[ERROR] 'data' group not found in {dataset_path}")
            return None
            
        demos = list(f["data"].keys())
        print(f"  - Found {len(demos)} demos")
        
        for demo in demos:
            demo_grp = f[f"data/{demo}"]
            obs_grp = demo_grp["obs"]
            
            # Load demo data
            actions = demo_grp["actions"][:]
            joint_pos = obs_grp["joint_pos"][:]
            joint_vel = obs_grp["joint_vel"][:]
            base_ang_vel = obs_grp["base_ang_vel"][:]
            projected_gravity = obs_grp["projected_gravity"][:]
            command_speed = obs_grp["command_speed"][:]
            
            num_steps = len(actions)
            if num_steps < 50:
                continue
                
            # 1. Compute global metrics
            act_diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
            global_action_deltas.extend(act_diffs)
            
            vel = np.diff(joint_pos, axis=0) / dt
            acc = np.diff(vel, axis=0) / dt
            jerk = np.diff(acc, axis=0) / dt
            jerk_norm = np.linalg.norm(jerk, axis=1)
            global_joint_jerks.extend(jerk_norm)
            
            # 2. Extract transition-aligned windows
            cmd_changed = np.any(np.diff(command_speed, axis=0) != 0, axis=1)
            transition_indices = np.where(cmd_changed)[0] + 1
            
            demo_action_diffs = np.zeros(num_steps)
            demo_action_diffs[1:] = act_diffs
            
            joint_vel_diffs = np.zeros(num_steps)
            joint_vel_diffs[1:] = np.linalg.norm(np.diff(joint_vel, axis=0), axis=1)
            
            base_ang_vel_rp = np.linalg.norm(base_ang_vel[:, :2], axis=1)
            
            g_xy_norm = np.linalg.norm(projected_gravity[:, :2], axis=1)
            g_xy_norm = np.clip(g_xy_norm, 0.0, 1.0)
            base_tilts_deg = np.arcsin(g_xy_norm) * (180.0 / np.pi)
            
            for idx in transition_indices:
                if idx - window_before >= 0 and idx + window_after < num_steps:
                    window_range = slice(idx - window_before, idx + window_after)
                    
                    transition_action_deltas.append(demo_action_diffs[window_range])
                    transition_base_ang_vels.append(base_ang_vel_rp[window_range])
                    transition_base_tilts.append(base_tilts_deg[window_range])
                    transition_joint_vel_deltas.append(joint_vel_diffs[window_range])
                    
    metrics = {
        "global_action_deltas": np.array(global_action_deltas),
        "global_joint_jerks": np.array(global_joint_jerks),
        "transition_action_deltas": np.array(transition_action_deltas),
        "transition_base_ang_vels": np.array(transition_base_ang_vels),
        "transition_base_tilts": np.array(transition_base_tilts),
        "transition_joint_vel_deltas": np.array(transition_joint_vel_deltas),
        "label": label
    }
    
    print(f"  - Extracted {len(transition_action_deltas)} transition events.")
    return metrics

def plot_comparison(all_metrics, save_path, window_before=15, window_after=25):
    """
    Plots comparative statistics for multiple datasets in a 3x2 grid.
    """
    print(f"\nGenerating comparative plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(3, 2, figsize=(15, 14))
    
    # Modern, clean color palette for up to 6 curves
    colors = ["#ff5757", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    
    steps_x = np.arange(-window_before, window_after)
    
    # Helper for temporal profiles
    def plot_metric_profile(ax, key, title, ylabel, show_legend=True, scale="linear"):
        for i, m in enumerate(all_metrics):
            if len(m[key]) == 0:
                continue
            mean = np.mean(m[key], axis=0)
            std = np.std(m[key], axis=0)
            color = colors[i % len(colors)]
            ax.plot(steps_x, mean, color=color, label=m["label"], linewidth=2.2)
            ax.fill_between(steps_x, mean - 0.15*std, mean + 0.15*std, color=color, alpha=0.1)
            
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Switch Inst" if show_legend else None)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Steps Relative to Transition", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_yscale(scale)
        if show_legend:
            ax.legend(loc="best", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.4)

    # 1. Action jump
    plot_metric_profile(axs[0, 0], "transition_action_deltas", 
                        "Action Jump Magnitude (||a_t - a_{t-1}||) at Transition", "Action Delta [L2 Norm]")
    
    # 2. Base angular velocity
    plot_metric_profile(axs[0, 1], "transition_base_ang_vels", 
                        "Base Angular Velocity (Roll/Pitch) at Transition", "Velocity [rad/s]")

    # 3. Base tilt
    plot_metric_profile(axs[1, 0], "transition_base_tilts", 
                        "Base Tilt Angle (Stumble Indicator) at Transition", "Tilt Angle [Degrees]")

    # 4. Joint velocity jump
    plot_metric_profile(axs[1, 1], "transition_joint_vel_deltas", 
                        "Joint Velocity Jump (||v_t - v_{t-1}||) at Transition", "Velocity Delta [rad/s]")

    # 5. Global Action Boxplots
    action_data = [m["global_action_deltas"] for m in all_metrics]
    labels = [m["label"] for m in all_metrics]
    
    bplot1 = axs[2, 0].boxplot(action_data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(bplot1['boxes'], colors[:len(all_metrics)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bplot1['medians']:
        median.set(color='black', linewidth=1.5)
    axs[2, 0].set_title("Global Distribution of Action Step Deltas", fontsize=11, fontweight='bold')
    axs[2, 0].set_ylabel("Action Change Magnitude [L2 Norm]", fontsize=9)
    axs[2, 0].grid(True, alpha=0.4)

    # 6. Global Jerk Boxplots (Log Scale)
    jerk_data = [m["global_joint_jerks"] for m in all_metrics]
    bplot2 = axs[2, 1].boxplot(jerk_data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(bplot2['boxes'], colors[:len(all_metrics)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bplot2['medians']:
        median.set(color='black', linewidth=1.5)
    axs[2, 1].set_title("Global Distribution of Joint Jerks", fontsize=11, fontweight='bold')
    axs[2, 1].set_ylabel("Joint Jerk [rad/s³] (Log Scale)", fontsize=9)
    axs[2, 1].set_yscale('log')
    axs[2, 1].grid(True, which="both", alpha=0.4)

    plt.suptitle("Dataset Locomotion Transition & Smoothness Comparison", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Comparative plots successfully saved to:\n   {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare arbitrary HDF5 datasets dynamically.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="List of HDF5 paths to analyze.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Legend labels for each dataset.")
    parser.add_argument("--output_path", type=str, default=None, help="Output plot filename.")
    args = parser.parse_args()
    
    # Resolve labels
    if args.labels is None:
        labels = [Path(p).stem for p in args.datasets]
    else:
        labels = args.labels
        if len(labels) < len(args.datasets):
            labels.extend([Path(p).stem for p in args.datasets[len(labels):]])
            
    # Resolve output path
    if args.output_path:
        save_path = Path(args.output_path)
    else:
        save_path = Path(__file__).resolve().parent / "plots" / "dataset_transition_comparison.png"
        
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process all datasets
    all_metrics = []
    for path, label in zip(args.datasets, labels):
        m = extract_metrics(path, label)
        if m is not None:
            all_metrics.append(m)
            
    if not all_metrics:
        print("[ERROR] No datasets could be successfully processed.")
        return
        
    plot_comparison(all_metrics, save_path)

if __name__ == "__main__":
    main()
