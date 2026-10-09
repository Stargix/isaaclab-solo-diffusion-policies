# Pronk v2: preserve the jumpy_safe motion while fine-tuning stability

## Intent

The v1 experiment learned synchronized four-foot hops, but changed the desired
spring-like motion into short, fast bounces. V2 is a separate task and PPO
algorithm. It starts from the same frozen `jumpy_safe.pt`, removes the per-hop
reward, restores the base velocity-tracking width, and adds a direct action-mean
penalty to the frozen expert. V1 remains available unchanged for comparison.

This is a targeted fine-tuning experiment, not a guarantee of zero falls. Its
commands cover 0.25–0.65 m/s forward, with the existing modest lateral/yaw
commands and left-right augmentation. Wider speeds should only be added after
the source gait is retained and stability is established in this range.

## What the reference penalty does

For each PPO minibatch, the frozen checkpoint predicts its mean joint action
for the same observations. V2 adds

`reference_loss_coef * mean((student_action_mean - jumpy_safe_action_mean)^2)`

to the PPO loss. The actor observation normalizer stays at the checkpoint's
statistics, matching the frozen reference; the critic normalizer continues to
adapt. The reference actor is loaded and SHA-256 checked by the algorithm.
The ordinary PPO desired-KL schedule still only controls update-to-update
changes; the new reference term is what constrains cumulative drift from the
original motion.

Initial coefficient is 1.0, learning rate 1e-4, action exploration std 0.20,
and duration 1000 iterations. Adaptive learning rate is capped at 1e-4.
These are pilot settings, not tuned
or guaranteed optima. Review the measured reference-action MSE alongside fall
rate, speed tracking, jump height excursion, cadence, flight duration, and
four-foot timing. Do not select the checkpoint by return alone.

The physical-cycle tracker remains for evaluation metrics, but its reward scale
is zero. It cannot reward extra hop frequency. The environment retains the base
velocity, orientation, contact, torque, and action-rate terms, plus the v1
physical validity checks for measuring a completed pronk cycle.

## Cluster launch

Copy the verified checkpoint to `checkpoints/jumpy_safe.pt`, pull the commit,
then from the repository root:

```bash
sbatch scripts/reinforcement_learning/rsl_rl/experiments/pronk_v2/train_cluster.sbs
```

The launcher checks the checkpoint digest and committed source, runs the
existing cycle-tracker regressions, performs an 8-environment/2-iteration PPO
smoke run, then starts the 4096-environment training. The learned actor and
frozen reference are both initialized from the exact same checkpoint inside
`PronkPPO`; no separate warm-start flag is needed. Critic, optimizer, action
noise and PPO iteration are fresh. The task config registers the algorithm
for both training and playback. RSL-RL is pinned to the reviewed 3.1.2 API.

The new task is `solo12-pronk-v2-v0`; outputs use
`logs/rsl_rl/solo12_rsl_rl_pronk_runs/`. V1 task and checkpoints are not
overwritten.

## Evaluation gate

Before calling this a usable expert, evaluate at least 0.30 and 0.60 m/s with
multiple resets in the **cluster's Isaac Sim 4.5 environment** and record:

- survival and completed physical hops per episode;
- actual forward/lateral/yaw tracking;
- base vertical excursion and hop cadence against `jumpy_safe.pt` at matched
  commands;
- four-foot takeoff and landing synchrony;
- performance with the intended reset randomization and pushes.

The local Isaac Sim 5.1 full-randomization audit is not a substitute for this
gate because the cluster uses Isaac Sim 4.5. A high training return or a single
successful viewer episode is insufficient.

## Implementation review (2026-10-09)

A CPU PPO update with the actual shared SOLO12 left-right transform passed.
It checks exact initial action-mean equality with the reference, a fresh critic
and noise, unchanged actor-normalizer buffers, updated critic-normalizer
statistics, an unchanged reference without gradients, changed student weights,
finite losses, and the learning-rate ceiling. Python compilation and Bash syntax
checks passed. The task configuration was instantiated in local Isaac Sim.

The local GPU smoke could not complete its rollout because of GPU/Windows
virtual-memory allocation errors; no completed simulator update is claimed.
The launcher therefore runs the simulator smoke on the cluster before allowing
the longer train. This implementation review does not establish convergence or
retention of the original motion after training. The reference coefficient 1.0
is a pilot value whose effect must be measured against stability and gait.
