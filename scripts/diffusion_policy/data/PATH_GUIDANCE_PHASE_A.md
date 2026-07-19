# Phase A path-guidance ablation

This experiment is parallel to, and does not replace, `phase_a_holonomic`.

## What is being compared

`phase_a_holonomic` conditions the student on timed SE(2) tokens containing
nominal `vx/vy/wz`. `phase_a_path_guidance` conditions it on geometry only:

```text
7 guide tokens: [direction_x, direction_y, log1p(distance), arc_offset]
terminal token: [x, y, sin(yaw), cos(yaw), height, time_to_go, terminal_phase, guide_error]
```

The path-guidance vector has 36 dimensions and never exposes the nominal route
velocity or the feedback command consumed by the locomotion expert.

The design is inspired by path-conditioned RL, but it is intentionally not
presented as the same algorithm. Haro et al. train a high-level policy with PPO
and a goal-reaching reward, with no explicit path-following reward. Here the
student is an offline action policy. A valid BC adaptation therefore keeps
three objects separate:

1. the clean, dynamically feasible waypoint task followed by the teacher;
2. the imperfect guide shown to the student;
3. the clean terminal pose, which remains authoritative.

Noise is sampled before the expert acts and stored in the HDF5. Actions are
never relabelled after collection. Adding arbitrary noise to offline goals
while retaining unrelated actions would create contradictory supervision.

## Data generation

The executable route is sampled endpoint-first in waypoint space. A smooth
minimum-jerk schedule joins a continuously sampled curve/S bridge, an
independent final yaw is added, and proposals are rejected unless their exact
body-frame twist and translational acceleration remain inside the declared
crouch envelope. The final 50 ticks are a fixed zero-velocity pose.

The guide is a second dense curve obtained by independently corrupting future
waypoints. The start anchor is kept exact so the current route error remains
observable. By default 25% of episodes have a clean guide; the rest contain
bounded corruption. The HDF5 stores `reference_pos_w`, `guidance_pos_w` and
`root_pos_w` separately.

Collect train data:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-crouch-v0 `
  --checkpoint checkpoints/crouch_final.pt --skill_name crouch `
  --desired_base_height 0.1705 --route_profile phase_a_path_guidance `
  --include_warmup_frames --startup_hold_steps 25 `
  --route_vx_max 0.45 --route_vy_abs_max 0.30 --route_wz_abs_max 0.50 `
  --route_accel_abs_max 0.45 `
  --path_waypoint_count_min 6 --path_waypoint_count_max 9 `
  --path_waypoint_spread_m 0.16 --path_waypoint_spread_clip_m 0.28 `
  --guidance_noise_std_m 0.06 --guidance_noise_clip_m 0.14 `
  --guidance_clean_probability 0.25 `
  --path_terminal_hold_steps 50 --path_final_yaw_abs_max_rad 1.20 `
  --route_initial_lateral_offset_m 0.06 --route_initial_yaw_offset_rad 0.12 `
  --num_envs 128 --num_steps 1000000 --physics_dr_mode light --seed 42 `
  --output_name crouch_phase_a_path_guidance_train_v1.hdf5 --headless
```

Collect the evaluation file with the same command but `--seed 314159`,
`--num_steps 150000` and output name
`crouch_phase_a_path_guidance_eval_v1.hdf5`.

## Preflight and plots

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/validate_phase_a_dataset.py `
  --dataset scripts/diffusion_policy/data/datasets/crouch_phase_a_path_guidance_train_v1.hdf5 `
  --output_dir scripts/diffusion_policy/data/coverage/crouch_phase_a_path_guidance_train_v1

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/plot_phase_a_dataset.py `
  --dataset scripts/diffusion_policy/data/datasets/crouch_phase_a_path_guidance_train_v1.hdf5 `
  --output_dir scripts/diffusion_policy/data/coverage/crouch_phase_a_path_guidance_train_v1 `
  --num_demos 12 --seed 42

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/analyze_spatial_coverage.py `
  --datasets scripts/diffusion_policy/data/datasets/crouch_phase_a_path_guidance_train_v1.hdf5 `
  --output_dir scripts/diffusion_policy/data/coverage/crouch_phase_a_path_guidance_train_v1 `
  --goal_source reference --goal_representation path_guidance_se2_36
```

Expected outputs include:

- `phase_a_preflight.json`;
- `phase_a_routes_overview.png`;
- `path_guidance_routes_overview.png`;
- `path_guidance_goal_coverage.png`;
- `coverage_summary.json` and `goal_samples.csv`.

Do not train if survival, tracker error, teacher saturation, terminal hold, clean
vs noisy mixture, or guide-noise coverage fails the preflight.

## Train

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --config scripts/diffusion_policy/train/configs/phase_a_crouch_path_guidance_k10.json `
  --datasets scripts/diffusion_policy/data/datasets/crouch_phase_a_path_guidance_train_v1.hdf5 `
  --output_dir scripts/diffusion_policy/runs/crouch_phase_a_path_guidance_k10_v1 `
  --run_name crouch_phase_a_path_guidance_k10_v1 `
  --device cuda --wandb_project solo12-diffusion-policy
```

For a fair model comparison, keep K, Transformer size, history, action horizon,
optimizer, symmetry and seed equal to the holonomic run. The data distributions
are intentionally different because this experiment adds an explicit terminal
task; report that fact rather than treating it as a conditioning-only ablation.

Primary evaluation should replay a separately collected reference file. For
the current six-second episodes, use a duration that fits the file, for example:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/evaluate_policy.py `
  --task solo12-crouch-v0 `
  --checkpoint scripts/diffusion_policy/runs/crouch_phase_a_path_guidance_k10_v1/best.pt `
  --reference_replay_dataset scripts/diffusion_policy/data/datasets/crouch_phase_a_path_guidance_eval_v1.hdf5 `
  --reference_replay_demos 12 --duration_s 5.8 `
  --output_dir scripts/diffusion_policy/evaluations/crouch_phase_a_path_guidance_k10_v1
```

The important comparison is not only circle/S tracking. Report terminal
position/yaw error, speed during the last hold, time to settle, survival and
tracking error under both clean and noisy guidance.
