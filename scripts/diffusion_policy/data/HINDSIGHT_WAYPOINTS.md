# Crouch: hindsight waypoint baseline

This is the minimal goal-conditioned behaviour-cloning baseline.  Collect
ordinary clean crouch demonstrations with random high-level commands; no
route tracker, reference path, terminal-hold generator or separate guidance
field is required.  At train time the existing dataset loader relabels every
state with three future achieved positions and its achieved terminal pose.
Only the three intermediate waypoint observations receive bounded 3 cm XY
jitter; the terminal pose and action labels stay clean.

Use an existing clean crouch HDF5 file, or collect one with the normal profile:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-crouch-v0 `
  --checkpoint checkpoints/crouch_final.pt --skill_name crouch `
  --desired_base_height 0.1705 --route_profile random_velocity `
  --num_envs 256 --num_steps 1000000 `
  --output_name crouch_hindsight_waypoints_train_v1.hdf5 --headless
```

Train:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --config scripts/diffusion_policy/train/configs/crouch_hindsight_waypoints_k10.json `
  --datasets scripts/diffusion_policy/data/datasets/crouch_hindsight_waypoints_train_v1.hdf5 `
  --output_dir scripts/diffusion_policy/runs/crouch_hindsight_waypoints_k10_v1 `
  --run_name crouch_hindsight_waypoints_k10_v1 `
  --device cuda --wandb_project solo12-diffusion-policy
```

The `waypoint_noise_*` options only apply to this achieved/path11 baseline and
are training-only.  Set `waypoint_noise_std_m` to `0.0` for a completely clean
hindsight ablation.
