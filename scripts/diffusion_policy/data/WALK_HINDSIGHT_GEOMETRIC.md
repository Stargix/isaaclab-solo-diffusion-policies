# Walk hindsight geometry baseline

This branch keeps the successful Phase-A walk diffusion architecture and
replaces only its temporal path condition.  It has no route tracker, no
teacher feedback, no injected waypoint noise and no DPPO stage.

For every state in a clean walk rollout, the dataset relabels the next 2 s of
the *achieved* base trajectory as:

```text
[waypoint@25% XY, waypoint@50% XY, waypoint@75% XY,
 terminal XY, sin/cos(terminal yaw), terminal height, average path speed]
```

Waypoints are sampled by arc-length fraction, not by prescribed arrival time.
Average speed is achieved arc length divided by the two-second horizon.

## First training: reuse the existing Phase-A walk data

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/analyze_spatial_coverage.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_phase_a_reference_v1.hdf5 `
  --goal_source achieved --goal_representation hindsight_geom_avg12 `
  --include_padded_starts --startup_sample_multiplier 16 `
  --output_dir scripts/diffusion_policy/data/coverage/walk_hindsight_geom_avg12_v1

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --config scripts/diffusion_policy/train/configs/walk_hindsight_geom_avg12_k10.json `
  --datasets scripts/diffusion_policy/data/datasets/walk_phase_a_reference_v1.hdf5 `
  --output_dir scripts/diffusion_policy/runs/walk_hindsight_geom_avg12_k10_v1 `
  --run_name walk_hindsight_geom_avg12_k10_v1 `
  --device cuda --wandb_project solo12-diffusion-policy
```

Evaluate only within the measured walk support before changing any collection:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/evaluate_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/diffusion_policy/runs/walk_hindsight_geom_avg12_k10_v1/best.pt `
  --speeds 0.2 0.4 0.6 --repeats 5 --duration_s 20 `
  --output_dir scripts/diffusion_policy/evaluations/walk_hindsight_geom_avg12_k10_v1
```

Do not evaluate 0.8 or 1.0 m/s with the existing walk dataset: its achieved
speed support ends near 0.65 m/s.

## Optional recollection after the baseline

The existing file is valid for the first comparison, but it has only 0.806% of
two-second windows with an achieved stop.  For a final walk model that must
arrive *and stop*, recollect at the current envelope with the new explicit
2.5-second holds.  This remains the same simple Phase-A collector: random
velocity commands drive the frozen walk checkpoint; there is no route-aware
teacher.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt --skill_name walk `
  --desired_base_height 0.2932 --route_profile phase_a `
  --include_warmup_frames --startup_hold_steps 125 --phase_a_stop_hold_s 2.5 `
  --command_resample_time_s 1.0 `
  --num_envs 1024 --num_steps 1000000 --physics_dr_mode light --seed 42 `
  --output_name walk_phase_a_geom_stop_v1.hdf5 --headless
```

Only try a higher-speed walk dataset after the baseline tracks 0.2/0.4/0.6 m/s
well. First run this 150k-step probe and inspect coverage/survival before a
full collection:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt --skill_name walk `
  --desired_base_height 0.2932 --route_profile phase_a `
  --include_warmup_frames --startup_hold_steps 125 --phase_a_stop_hold_s 2.5 `
  --command_resample_time_s 1.0 `
  --phase_a_forward_speed_max 0.80 --phase_a_yaw_rate_abs_max 0.70 `
  --num_envs 128 --num_steps 150000 --physics_dr_mode light --seed 314159 `
  --output_name walk_phase_a_higher_speed_probe_v1.hdf5 --headless
```

Do not promote 0.8 m/s into the train dataset solely because it was requested:
first verify the frozen walk checkpoint survives the probe and that the
achieved `average_path_speed` coverage reaches the intended values.  If it
does, rerun the same command with `--num_steps 1000000`, seed `42`, and output
name `walk_phase_a_higher_speed_train_v1.hdf5`.
