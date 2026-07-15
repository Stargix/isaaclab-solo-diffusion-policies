# Spatial diffusion policy (waypoints + terminal goal)

Run this after the command-only baseline. It keeps the same tested generative
backbone and changes only the conditioning/data problem.

## Phase A: walk-only reference-path tracker

The first valid waypoint experiment is **walk only**. It does not train on
achieved-future hindsight positions. Collection integrates the commanded
`[vx, vy, wz]` in world coordinates and stores `reference_pos_w`,
`reference_yaw_w` and `reference_command` alongside the measured robot state.
Thus the target remains a desired route when the expert has tracking error.

The collector also records a zero-command reset hold, starts, stops, arcs and
lateral phases. The Phase-A dataset loader includes padded reset samples and
oversamples them modestly; this makes the all-zero action history used at
deployment an explicit training condition instead of an accidental OOD state.

### Collect and preflight

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt --skill_name walk `
  --desired_base_height 0.2932 `
  --route_profile phase_a --command_resample_time_s 1.0 `
  --startup_hold_steps 25 --include_warmup_frames `
  --num_envs 128 --num_steps 1500000 `
  --physics_dr_mode light --seed 42 `
  --output_name walk_phase_a_reference_v1.hdf5 --headless

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/validate_phase_a_dataset.py `
  --dataset scripts/diffusion_policy/data/datasets/walk_phase_a_reference_v1.hdf5 `
  --output_dir scripts/diffusion_policy/data/coverage/walk_phase_a_reference_v1

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/analyze_spatial_coverage.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_phase_a_reference_v1.hdf5 `
  --goal_source reference --include_padded_starts --startup_sample_multiplier 16 `
  --output_dir scripts/diffusion_policy/data/coverage/walk_phase_a_reference_v1_goals
```

Do not merge this file with crouch data. Posture transitions are Phase B and
need a teacher that actually executes a height schedule within each episode.

### Train Phase A

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_phase_a_reference_v1.hdf5 `
  --output_dir scripts/diffusion_policy/runs/phase_a_walk_reference_k10_v1 `
  --config scripts/diffusion_policy/train/configs/phase_a_walk_reference_k10.json `
  --symmetry_mode quadruped `
  --run_name phase_a_walk_reference_k10_v1_seed42 `
  --wandb_project solo12-diffusion-policy
```

The resulting checkpoint uses schema v4 / `spatial_reference_path_ddpm`; it is
intentionally incompatible with the old achieved-hindsight run. Evaluate it on
a constant-height straight `walk` route first, then on gentle curves.

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

## Legacy achieved-hindsight run (diagnostic only)

Use the validated separate-skill walk+crouch dataset for the single main spatial
train:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --output_dir scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3 `
  --config scripts/diffusion_policy/train/configs/compact_k10.json `
  --symmetry_mode quadruped `
  --batch_size 1024 --epochs 30 --save_every 5 `
  --run_name spatial_walk_crouch_time_preview_v3_seed42 `
  --wandb_project solo12-diffusion-policy
```

A walk-only train is a diagnostic only if this run fails to learn path geometry.
Thirty epochs give approximately the same optimization budget as the successful
baseline run stopped at epoch 25; do not default to a blind 200-epoch run.

Resume an interrupted epoch-boundary checkpoint with the same data/config flags
and:

```powershell
--resume scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3/checkpoint_epoch_0010.pt
```

## Evaluate

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/play_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3/best.pt `
  --default_path_mode walk --desired_speed 0.4 `
  --exec_horizon 8 --torchscript_denoiser `
  --guidance_scale 1.0
```

Do not evaluate `s_curve_crouch.npy` or `--default_path_mode walk_to_crouch`
with the current merged separate-skill dataset: it contains no real
walk-to-crouch transition windows.  Those routes are reserved for the next
dataset iteration, which must record an expert/reference height schedule.

`goal_horizon_steps`, prediction horizon and execution offset are loaded from
the checkpoint. Overriding the goal horizon is an explicit distribution-shift
ablation, not a normal deployment option.

## Pre-flight checks

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/tests/test_contracts.py
conda run --no-capture-output -n env_isaaclab python -m compileall -q scripts/diffusion_policy
```

Spatial checkpoints before schema v3 are deliberately rejected.

## Dataset-support audit

Before interpreting a spatial loss, check whether the route goals and reset
history used at deployment exist in the offline data:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/audit_spatial_dataset.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --checkpoint scripts/diffusion_policy/runs/spatial_walk_crouch_time_preview_v3/best.pt `
  --output_dir scripts/diffusion_policy/data/audits/spatial_epoch7_design
```

The audit is diagnostic, not a substitute for closed-loop tracking.  In
particular, an all-zero reset action history must be represented explicitly by
the demonstrations (or bootstrapped by a teacher) before it is considered an
in-distribution policy state.
