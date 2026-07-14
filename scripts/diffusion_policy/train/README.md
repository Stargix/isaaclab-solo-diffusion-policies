# Spatial diffusion policy (waypoints + terminal goal)

Run this after the command-only baseline. It keeps the same tested generative
backbone and changes only the conditioning/data problem.

## Schema v3 contract

- 50 Hz control and `H=8` observation tokens.
- State condition: `s[t-8:t]`; delayed action condition: `a[t-9:t-1]`.
- Rolling 11D goal at every history state:
  `[p_xy(+0.5s), p_xy(+1.0s), p_xy(+1.5s), target_xy(+2.0s), target_z_abs, target_dyaw, v_req]`.
- Hindsight endpoint: exactly 100 simulator steps (2 s) ahead by default.
- Denoising target: 16 actions `a[t-8:t+8]`; execution begins at token 8 (`a[t]`).
- DDPM K=10, cosine schedule, epsilon prediction; CFG is disabled by default.
- Whole-episode train/validation split; normalizers use training episodes only.
- Bounded-memory z-score fitting and exact action extrema after symmetry.

Training builds a rolling achieved-future temporal preview for each historical state. This
matches deployment, where the local plan moves at every control tick. The old
fixed-endpoint history and post-step buffer update are no longer used.

The preview is temporal rather than fixed at `0.4/0.8/1.2 m`. At training time
it reads poses reached 25/50/75/100 ticks later. At deployment time it samples
the planned path at approximately `speed * time_offset` metres. This preserves
the meaning of every feature across speeds and avoids duplicated far waypoints
when a slow trajectory travels less than 1.2 m in the terminal horizon.

## Train

Use the validated separate-skill walk+crouch dataset for the single main spatial
train:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --output_dir scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3 `
  --config scripts/diffusion_policy/train/configs/compact_k10.json `
  --symmetry_mode quadruped
```

A walk-only train is a diagnostic only if this run fails to learn path geometry.

## Evaluate

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/play_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3/best.pt `
  --path_file scripts/diffusion_policy/paths/s_curve_crouch.npy `
  --exec_horizon 1 `
  --guidance_scale 1.0
```

`goal_horizon_steps`, prediction horizon and execution offset are loaded from
the checkpoint. Overriding the goal horizon is an explicit distribution-shift
ablation, not a normal deployment option.

## Pre-flight checks

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/tests/test_contracts.py
conda run --no-capture-output -n env_isaaclab python -m compileall -q scripts/diffusion_policy
```

Spatial checkpoints before schema v3 are deliberately rejected.
