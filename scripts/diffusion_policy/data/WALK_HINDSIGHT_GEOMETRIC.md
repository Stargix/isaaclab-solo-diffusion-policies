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
  --speeds 0.2 0.4 0.6 --path_shapes straight s_curve circle right_angle random_polyline `
  --repeats 5 --duration_s 20 --exec_horizon 8 --num_inference_steps 10 `
  --output_dir scripts/diffusion_policy/evaluations/walk_hindsight_geom_avg12_k10_v1
```

The analytic evaluator creates all shape/speed/repeat scenarios as parallel
Isaac environments.  ``circle`` now starts exactly at the robot; ``right_angle``
tests connected 90-degree corners and ``random_polyline`` tests deterministic
but irregular connected segments.

Do not evaluate 0.8 or 1.0 m/s with the existing walk dataset: its achieved
speed support ends near 0.65 m/s.

## Balanced recollection for the joint walk + crouch model

For the joint model, collect one clean file per frozen expert.  `phase_a` is
not a teacher: it samples a body-velocity command, holds it for two seconds,
and records the trajectory actually produced by that expert.  Hindsight then
turns each recorded future into spatial waypoints.  The policy therefore sees
achievable local geometry, rather than a schedule of waypoint deadlines.

Use the same duration and number of environments for both files so neither
posture dominates merely by having more frames.  Stops are deliberately rare
(8%) but held for 2.1 s, which is enough to make valid final-stop windows
without making most of training stationary.

### Walk collection

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_final.pt --skill_name walk `
  --desired_base_height 0.2932 --route_profile phase_a `
  --include_warmup_frames --startup_hold_steps 25 `
  --command_resample_time_s 5.0 `
  --phase_a_forward_speed_min 0.20 --phase_a_forward_speed_max 1.00 `
  --phase_a_arc_forward_speed_max 0.80 `
  --phase_a_reverse_speed_abs_max 1.00 `
  --phase_a_lateral_speed_abs_max 0.50 --phase_a_lateral_forward_speed_max 0.65 `
  --phase_a_yaw_rate_abs_max 0.50 `
  --phase_a_stop_probability 0.08 --phase_a_stop_hold_s 2.0 `
  --num_envs 4096 --num_steps 1000000 --physics_dr_mode light --seed 42 `
  --output_name walk_phase_a_balanced_slow_v2.hdf5 --headless
```

### Crouch collection (intentionally slower)

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-crouch-v0 `
  --checkpoint checkpoints/crouch_final.pt --skill_name crouch `
  --desired_base_height 0.1705 --route_profile phase_a `
  --include_warmup_frames --startup_hold_steps 25 `
  --command_resample_time_s 5.0 `
  --phase_a_forward_speed_min 0.10 --phase_a_forward_speed_max 0.45 `
  --phase_a_arc_forward_speed_max 0.50 `
  --phase_a_reverse_speed_abs_max 0.50 `
  --phase_a_lateral_speed_abs_max 0.25 --phase_a_lateral_forward_speed_max 0.35 `
  --phase_a_yaw_rate_abs_max 0.35 `
  --phase_a_stop_probability 0.08 --phase_a_stop_hold_s 2.0 `
  --num_envs 1024 --num_steps 1000000 --physics_dr_mode light --seed 43 `
  --output_name crouch_phase_a_balanced_slow_v2.hdf5 --headless
```

First merge the two single-skill files.  The merge remaps ``skill_idx`` to a
single ``[walk, crouch]`` table; passing the two source files directly to the
training dataset is intentionally rejected because their local skill tables
are different.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/merge_datasets.py `
  --inputs scripts/diffusion_policy/data/datasets/walk_phase_a_waypoint_v3.hdf5 scripts/diffusion_policy/data/datasets/crouch_phase_a_waypoint_v3.hdf5 `
  --output scripts/diffusion_policy/data/datasets/walk_crouch_phase_a_waypoint_v3.hdf5 `
  --shuffle_seed 45
```

Then train the merged dataset:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/train/train.py `
  --config scripts/diffusion_policy/train/configs/walk_crouch_hindsight_geom_avg12_balanced_k10.json `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_phase_a_waypoint_v3.hdf5 `
  --output_dir scripts/diffusion_policy/runs/walk_crouch_hindsight_geom_balanced_k10_v1 `
  --run_name walk_crouch_hindsight_geom_balanced_k10_v1 `
  --device cuda --wandb_project solo12-diffusion-policy
```

This creates one policy that can reproduce walk-like and crouch-like local
motions because height is part of the goal.  It does **not** yet teach an
in-motion walk-to-crouch transition: that requires a later, separate dataset
of those transitions.  First establish that each posture works reliably in
isolation with this simpler mixed dataset.
