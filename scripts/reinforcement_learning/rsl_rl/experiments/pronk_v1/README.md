# SOLO12 pronk expert fine-tuning

Task: `solo12-pronk-v0`. Warm start: `checkpoints/jumpy_safe.pt`
(SHA256 `6496d831f4d3b8fec8793eb740351edbe6581198787d713ac7010371a0ff4183`).

This experiment stabilizes and steers an existing four-foot hopping actor for
a prospective third-behaviour demonstration library. The checkpoint's original
run configuration is missing, so this is a new, documented acquisition run.
It does not establish that pronking is optimal at low speed or that the later
diffusion policy will preserve this gait.

## Control and training

- The actor keeps 48 observations, 12 actions, MLP layers 256/128/64 and the
  `safe` joint reference. Actor weights and its normalizer are imported.
  Critic, optimizer, exploration noise and iteration start fresh.
- PD gains are explicitly applied to the actuators: kp=9, kd=0.2, action
  scale 0.25. Physics runs at 200 Hz and the actor at 50 Hz.
- XY and yaw tracking use width 0.20 (the base task uses 0.50), making failure
  to track small commands meaningfully less rewarding.
- Commands: vx 0.15–1.0 m/s, vy ±0.15 m/s, yaw rate ±0.35 rad/s,
  resampled every 3 s. The base standing probability of 2% is retained.
  There are no reverse commands. Augmentation is identity plus left/right
  reflection, using the existing SOLO12 maps.
- The current base-task robustness setup is retained: observation noise,
  delays of 0–3 physics steps, randomized reset velocities, and startup
  randomization of friction, base mass, joint friction, inertia and base COM.
  Pushes use XY components in [-5,5] N and Z in [-10,10] N, intervals
  10–15 s, and durations 0.5–2 s. These amplitudes apply from the start;
  the inherited one-entry force setting is not a progressive curriculum.
  This matches the current base config, not a recovered original walk or
  crouch run. Crouch's current config disables observation noise, pushes and
  delay, while inheriting the startup physical randomization.
- Ground is a local generated flat mesh. Contact timers update at physics
  rate rather than only when lazily read at the control rate.

## Completed-cycle reward

The standard velocity, contact, orientation, torque and action-rate rewards
remain. The extra bonus is paid **once at a valid completed landing**:

1. A separate 0.5 s warm-up clock and prior upright supported stance are
   required. This clock resets with the physical state and is independent of
   RSL-RL's randomized initial episode counters.
2. All four cylindrical foot colliders must actually clear the flat ground
   by at least 3 mm, with no detected foot contact. Collider transforms,
   radius and height are read from the actual USD; link origins are not used
   as foot surfaces.
3. Valid flight lasts 0.025–0.20 s. All four feet must land within a 0.04 s
   window. A changed command, base/thigh contact, tilt above 25 degrees,
   or body height outside the broad 0.18–0.45 m band invalidates the cycle.
   These are bonus eligibility bounds, not an instantaneous height target.
4. The bonus is 0.25 times takeoff/landing synchrony, cycle-average XY
   tracking and actual progress in the commanded direction. Zero, reverse
   or purely sideways progress earns zero. Longer flights do not increase
   the per-cycle bonus; a failed flight earns none.

The timing width is 0.02 s. All these reward coefficients and eligibility bounds
are engineering choices informed by the local jumpy audit, not a proof of
optimality. There is no gait clock, phase observation or skill ID. Existing
walk/crouch/PPO defaults are unchanged.

Monitor `Metrics/pronk_completed_cycles_per_step`,
`Metrics/pronk_physical_flight_fraction`, actual speed, survival, and measured
foot forces. Progress/tracking diagnostics average only cycles completed in
that step and report zero when none completes, not a fictitious perfect score
for inactive trackers. A higher return alone does not show preservation of pronk.
The expert training range is not a promise that all its speeds are usable for
demonstrations; restrict collection to the range that passes evaluation.

## Launch

From the cluster repository root, after pulling the reviewed commit and copying
the checkpoint:

```bash
sbatch scripts/reinforcement_learning/rsl_rl/experiments/pronk_v1/train_cluster.sbs
```

The launcher checks the checkpoint digest and committed state of the experiment
and its shared training dependencies. It runs the regression tests in this
directory and an 8-env, 2-iteration smoke test before training 4096 environments
for 2500 iterations (seed 42). Regression tests live here rather than inside the
globally ignored `tests/` directories.

Output: `logs/rsl_rl/solo12_rsl_rl_pronk_runs/<timestamp>_<run_name>/`.
RSL-RL saves periodic `model_*.pt` files every 50 iterations; it does not select
a gait-validated `best.pt`. Keep checkpoints, logs and archives out of Git.

For a returned checkpoint, the existing command UI supports this task:

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-pronk-v0 --checkpoint "PATH_TO_MODEL.pt" --num_envs 1 --command_ui --command 0.6 0.0 0.0
```

Before collecting data, compare measured pronk and walk at 0.3 and 0.6 m/s,
steering in both directions, and stop. Inspect actual contacts/forces, survival
and velocity error. Treat acquisition as successful only if stability improves
while four-foot synchronization and physical flight remain visible.

## Local verification (2026-10-09)

- Ten regression tests pass, covering physical clearance, prior support,
  excessive flight, invalid posture, staggered landing, command changes,
  per-environment resets, actual progress and diagonal takeoff rejection.
- Python compilation and Bash launcher syntax checks pass.
- Isaac Lab completed an 8-environment, two-iteration actor warm-start/PPO
  smoke with left/right augmentation and the new collider-based reward.
- A separate deterministic-actor check, with the configured physical/noise/delay
  randomization retained, detected 2 valid cycles at 0.3 m/s over 10 s across
  16 environments. Rewards were finite. This is detector integration evidence,
  not a gait success rate: the actor had 36 reset events during that check.
- The corresponding 0.6 m/s check detected 1 valid cycle, with finite rewards
  and 79 reset events. The sparse bonus and unstable warm-start under this
  robustness setup are acquisition risks, not evidence of convergence.

The full acquisition training is not locally validated for convergence. Inspect
its saved checkpoints for both stability and gait preservation before using
them as demonstrations; do not select a checkpoint solely by return.
