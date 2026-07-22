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

`residual_scale=0.20` means at most 20% of each joint's demonstrated action
half-range; it is not a fixed radian offset shared by unequal joints.

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
- smaller correction: `env.residual_scale=0.10`
- no intermediate-height curriculum: set
  `env.curriculum_stage2_steps` above the total training steps.

The environment logs progress, cross-track, speed/height error, survival,
success and residual RMS to RSL-RL/W&B.
