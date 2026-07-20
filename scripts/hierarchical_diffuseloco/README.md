# Hierarchical DiffuseLoco experiment

This is an isolated research line on branch `hierarchical-diffuseloco`.  It tests whether a
high-level policy can exploit the continuous command manifold of a frozen, command-plus-height
DiffuseLoco policy for route, final-pose, clearance and deadline objectives.

## What is learned

The high-level action is the absolute target

```text
[vx, vy, wz, desired_base_height]
```

and is passed directly to the frozen diffusion policy.  Only a physical safety clip is applied.
There is deliberately no command ramp, EMA, slew-rate limiter or interpolation.  Gradual
descent into a crouch is therefore learned because the return contains first- and second-order
command-change penalties.  `rewards.py` exposes every component for logging and ablations.

The observation is a versioned 42-vector: eight route-preview samples `(x,y,yaw,clearance)`,
final `(x,y,yaw)`, remaining time, required height, and five proprioceptive state values.
`route.py` makes clearance explicit: the central section of the benchmark route requires the
lower body height.  This avoids the degenerate solution of crouching for the whole episode.
For turns and multi-section studies, set `route_file` to an `.npy` file with columns
`x,y[,yaw[,required_height]]`; this keeps geometry and clearance labels versioned with each
evaluation seed.

## Task registration and training

The task is registered automatically when `isaaclab_tasks` imports the new direct package.  Use a
validated walk+crouch checkpoint (the current sprint policy is not a valid prerequisite):

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py `
  --task solo12-hierarchical-diffuseloco-v0 --num_envs 1024 `
  env_cfg.diffuseloco_checkpoint="scripts/baseline_diffuseloco/runs/walk_crouch_posture_conditioned/best.pt"
```

Hydra may require an absolute checkpoint path on Windows.  Before long runs, use
`--max_iterations 2 --num_envs 8` to validate the simulator/checkpoint integration.

## Required ablations

1. Classical path tracker (`baselines.py`).
2. Discrete walk/crouch selector (`DiscreteSkillSelector`).
3. Continuous PPO over the frozen policy.
4. Continuous PPO plus the support-risk term.
5. The same PPO with `smooth_first=smooth_second=0`.
6. Direct joint-action PPO only as a capacity/control baseline.

Report route success, survival, final-pose error, deadline miss, command total variation,
height-transition overshoot, and the fraction of commands outside the empirical support envelope.
Keep route seeds, checkpoint, inference steps, execution horizon and simulator randomization fixed
across the table.  Do not call diffusion denoising error an OOD detector without a separate
calibration study; `support_risk` is intentionally an empirical, auditable proxy.

## Offline tests

```powershell
conda run --no-capture-output -n env_isaaclab python -m compileall -q scripts/hierarchical_diffuseloco
conda run --no-capture-output -n env_isaaclab python -m unittest scripts.hierarchical_diffuseloco.tests.test_core
```
