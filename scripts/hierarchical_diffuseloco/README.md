# High-level PPO over frozen DiffuseLoco

This branch contains only experiment **H1**:

```text
route preview + final pose + progress schedule + robot state
                              |
                         PPO (4-D)
                              |
                  [vx, vy, wz, target height]
                              |
                  frozen velocity-height DiffuseLoco
                              |
                       12 joint actions
```

The frozen checkpoint must be `diffuseloco_velocity_height_ddpm` with `goal_dim=4`.
The intended prerequisite is:

```text
checkpoints_iri/diffuseloco_robust_walk_crouch.pt
sha256 2C41034B8ACF7E92AC28EF157F9F483F54C5EDA6B32976B8FBCCEA6E321676C0
```

The high-level action is absolute and is sent directly to the low-level policy. There is
no EMA, ramp, slew limiter or hand-written interpolation. A small action-difference cost
makes smooth commands preferable only when the learned return supports them.

## Task distribution

- Fixed horizon: 8 s.
- Families: straight, S-curve and rounded 90-degree turn.
- Desired mean speed: 0.30--0.42 m/s.
- Route length is exactly `desired_mean_speed * 8 s`.
- Interleaved high, intermediate and crouched target-height sections.
- No inherited velocity/force curriculum, pushes, action delay or terrain randomization.

The action envelope is deliberately forward-only in H1 (`vx in [0, 0.65]`) because all
routes are forward. Consequently the initial zero-mean actor maps to a useful 0.325 m/s
command rather than the trivial standing solution. Reverse navigation is out of scope for
this first falsifiable experiment.

## Reward

`rewards.py` uses normalized Huber costs and potential-based scheduled-progress shaping:

```text
s*(t) = min(v_desired * t, route_length)
Phi = -Huber((s - s*) / 0.30 m)
r_schedule = 4 * (gamma * Phi_next - Phi_previous)
```

The remaining terms penalize cross-track error, heading error, physical base-height error,
command variation, terminal pose error and falls. There is no alive reward, raw progress
bonus, per-step time penalty, instantaneous-speed reward or early success termination.
Final position over the fixed horizon is therefore the primary measurement of mean-speed
tracking, while scheduled progress supplies its dense learning signal.

## Train

PowerShell, local smoke (one optimizer update):

```powershell
conda run --no-capture-output -n env_isaaclab cmd /c isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task solo12-hierarchical-diffuseloco-v0 --num_envs 2 --max_iterations 1 --headless env.diffuseloco_checkpoint=checkpoints_iri/diffuseloco_robust_walk_crouch.pt env.diffuseloco_inference_steps=2
```

Cluster training:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task solo12-hierarchical-diffuseloco-v0 \
  --num_envs 4096 \
  --headless \
  env.diffuseloco_checkpoint=checkpoints_iri/diffuseloco_robust_walk_crouch.pt
```

RSL-RL uses PPO with Adam, observation normalization, learning rate `1e-4`, initial action
noise `0.35` and checkpoints every 50 iterations. Do not enable a reward curriculum for H1.

Inference on a held-out route bank (single PowerShell line):

```powershell
conda run --no-capture-output -n env_isaaclab cmd /c isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-hierarchical-diffuseloco-v0 --checkpoint logs/rsl_rl/solo12_hierarchical_diffuseloco/RUN/model_N.pt --num_envs 1 env.diffuseloco_checkpoint=checkpoints_iri/diffuseloco_robust_walk_crouch.pt env.route_seed=10017
```

## Offline verification

```powershell
conda run --no-capture-output -n env_isaaclab python -m compileall -q scripts/hierarchical_diffuseloco source/isaaclab_tasks/isaaclab_tasks/direct/hierarchical_diffuseloco
conda run --no-capture-output -n env_isaaclab python -m unittest scripts.hierarchical_diffuseloco.tests.test_core
conda run --no-capture-output -n env_isaaclab cmd /c isaaclab.bat -p scripts/hierarchical_diffuseloco/registration_smoke.py --headless
conda run --no-capture-output -n env_isaaclab cmd /c isaaclab.bat -p scripts/hierarchical_diffuseloco/smoke_env.py --checkpoint checkpoints_iri/diffuseloco_robust_walk_crouch.pt --headless
```

The full research decision, evaluation gate and future DPPO line are documented in
[`RESEARCH_DECISION.md`](RESEARCH_DECISION.md).
