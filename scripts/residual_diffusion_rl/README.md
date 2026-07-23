# Phase B1 — Residual PPO over spatial Diffusion Policy

This folder is intentionally independent from `hierarchical_diffuseloco`.
B1 asks whether online RL can correct a low-data spatial diffusion prior; the
high-level velocity/skill planner remains a later and separate experiment.

## Architecture

```text
path + target height + nominal speed
              │
              ▼
   frozen Phase-A goal-conditioned diffusion ──► base joint action
              │                                      │
robot state ──┴──────────────► residual PPO ─────────┤ + bounded residual
                                                     ▼
                                             Solo12 joint targets
```

The target height is piecewise constant. There is no command ramp, filter, or
hand-written transition controller. Smooth transitions can only emerge from
task reward, vertical-motion cost, and residual-rate regularization.

The task reward is average-speed-aware. Progress is rewarded only while
tangential speed is close to the request. Path, speed, schedule, yaw and height
use zero-centred Huber costs: they are quadratic near the target and keep
growing linearly for large errors instead of saturating. There is no alive
bonus, time cost or separate deadline command.

At elapsed time `t`, the schedule error is
`route_progress - requested_speed * t`. At the route end, success requires
`route_progress / t` to match requested speed. Thus requested average speed
defines traversal time without adding a second, redundant timing objective.
Success also requires Euclidean final-position, yaw, final-height and stopped
velocity tolerances held for five control steps. Passing through or beyond the
last route sample is not success.
Termination logs distinguish `route_success`, true `base_contact`, corridor
failure and timeout; the inherited Solo12 `base_contact` aggregate is not used.

`residual_scale=0.10` means at most 10% of each joint's demonstrated action
half-range; it is not a fixed radian offset shared by unequal joints.

The default samples the complete stage-2 route distribution from the start.
This is intentional: B1 refines an already competent frozen prior, so an
easy-only curriculum changes the training distribution without being needed
for exploration. The former three-stage curriculum remains available as an
ablation with `env.use_route_curriculum=true`.

## Train

PowerShell, from the repository root:

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task solo12-residual-diffusion-rl-v0 --num_envs 256 --headless env.spatial_diffusion_checkpoint="checkpoints_iri/real_walk_crouch_hindsight.pt"
```

Linux/Slurm:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task solo12-residual-diffusion-rl-v0 \
  --num_envs 256 --headless \
  env.spatial_diffusion_checkpoint=checkpoints_iri/real_walk_crouch_hindsight.pt
```

For the strict frozen-prior control, use the same environment with
`env.residual_scale=0.0`. Logs are written below
`logs/rsl_rl/solo12_residual_diffusion_rl`.

## Play a trained residual

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-residual-diffusion-rl-v0 --num_envs 1 --checkpoint="logs/rsl_rl/solo12_residual_diffusion_rl/<run>/model_<iteration>.pt" env.spatial_diffusion_checkpoint="checkpoints_iri/real_walk_crouch_hindsight.pt"
```

Useful ablations need no code fork:

- prior only: `env.residual_scale=0.0`
- larger correction ablation: `env.residual_scale=0.20`
- fixed straight/constant-height suite: `env.route_stage=0`
- staged route curriculum: `env.use_route_curriculum=true`

The environment logs progress, cross-track, instantaneous and final mean-speed
error, schedule error, terminal distance, height error, outcome rates and
residual RMS to RSL-RL/W&B.

## Paired checkpoint evaluation

`evaluate_residual.py` executes the deterministic actor mean and writes a
`summary.json` with route success, true base contact, corridor failure, timeout,
terminal mean-speed error, terminal distance, episode reward and duration,
both globally and separated by route family. Run each checkpoint with the same
`--seed`, `--stage` and frozen diffusion checkpoint before comparing them.
Evaluation route sampling is stratified, so stage 2 covers straight, S-curve,
right-angle and random-curve episodes instead of relying on a lucky random
draw; training sampling remains random.

The observation contract is now 77D (`residual_route_state77_average_speed_v2`).
Old 75D residual checkpoints are historical baselines and must not be resumed
under this MDP.
