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

The policy emits a normalized four-vector in `[-1,1]`, which is mapped affinely to the frozen
policy's observed command bounds.  Its 70-dimensional observation is eight robot-frame preview
samples `(x,y,sin(yaw),cos(yaw),max_height,valid)`, final `(x,y,sin(yaw),cos(yaw))`, remaining
time, base state and the two previous normalized commands.  The command history makes the
smoothness objective Markov; it does not impose smoothing. `max_height` is a command-space
clearance proxy (not physical collision geometry): the central section only permits the crouch
range, avoiding the degenerate solution of crouching throughout.  Routes use continuous arc
projection and are resampled when loaded from `.npy` files with columns
`x,y[,yaw[,max_command_height]]`.
For turns and multi-section studies, set `route_file`; keeping those files with the evaluation
seed versions geometry and clearance labels together.

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
4. The same PPO with `smooth_first=smooth_second=0`.
5. Direct joint-action PPO only as a capacity/control baseline.

Report route success, survival, final-pose error, deadline miss, command total variation,
height-transition overshoot, and the fraction of clearance violations.
Keep route seeds, checkpoint, inference steps, execution horizon and simulator randomization fixed
across the table.  The task deliberately does not claim an OOD penalty from a rectangular command
envelope: the walk/crouch data has only endpoint heights, so a capability map must be measured
before adding such a term.

## Offline tests

```powershell
conda run --no-capture-output -n env_isaaclab python -m compileall -q scripts/hierarchical_diffuseloco
conda run --no-capture-output -n env_isaaclab python -m unittest scripts.hierarchical_diffuseloco.tests.test_core
```
