# Solo12 bound expert v3

## Why this iteration exists

`bound_v2_1.pt` demonstrated that contact-level shaping alone was insufficient.
At 0.8 and 1.0 m/s it under-tracked speed, accumulated 0.52--0.59 m lateral
drift in five to six seconds, and produced strongly asymmetric left/right joint
commands. It never left curriculum stage 0 and its best return was only 7.80.

`solo12-bound-v3` is isolated from v1/v2 and addresses the measured causes:

- it trains from scratch instead of inheriting the asymmetric trot-like v1 gait;
- its curriculum starts where a fast bound is meaningful;
- directional errors no longer multiply into a nearly zero PPO signal;
- PPO uses clock-aware left/right augmentation and mirror consistency;
- front/back symmetry is deliberately excluded because it would require a
  half-cycle clock shift and negative forward commands.

## Reward

The contact and stability gates are unchanged from v2. Only command quality is
made denser:

```text
forward_score = 0.65 * exp_tracking_score
              + 0.35 * clamp(actual_vx / commanded_vx, 0, 1)

direction_quality = 0.55
                  + 0.15 * lateral_score
                  + 0.15 * yaw_rate_score
                  + 0.15 * heading_score

task_score = forward_score * direction_quality
reward     = 3.5 * task_score * gait_gate * stability_gate * dt
```

The progress component provides a linear acquisition signal from zero velocity,
but cannot reward backward motion and saturates at the command. The narrow
exponential makes exact tracking strictly better than both underspeed and
overspeed motion. Forward tracking remains mandatory, so standing cannot collect
direction reward. Perfect tracking still has the same 3.5-per-second ceiling as
v1/v2. Heading, lateral velocity, and yaw rate each affect the result without
any one term annihilating the learning signal.

The gait gate still measures phase/contact agreement, pair XOR, diagonal-contact
probability, pair foot-height agreement, and pair vertical-velocity agreement.
The mirror objective complements rather than duplicates this: it trains
`policy(mirror(state)) = mirror(policy(state))`, allowing asymmetric recovery
when the observed state is asymmetric while rejecting a permanently crooked gait.

## Curriculum

```text
stage 0: vx in [0.90, 1.15] m/s
stage 1: vx in [1.00, 1.30] m/s
stage 2: vx in [1.10, 1.50] m/s
then:    light lateral pushes up to 2 N
```

The intended diffusion dataset remains `vx=[1.0,1.5]`, `vy=wz=0`.

## Cluster training

Train from scratch. Do not pass a v1/v2 checkpoint:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task solo12-bound-v3 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 42 \
  --run-name solo12_bound_v3_seed42 \
  --symmetry-mode both \
  --symmetry-loss-coeff 0.001 \
  --headless \
  --device cuda:0
```

`both` applies the valid left/right sample reflection and the mirror loss. The
coefficient matches the conservative default of this repository's RSL-RL entry
point. Do not use the generic Solo12 four-way symmetry for this task.

## Playback

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-bound-v3 --checkpoint checkpoints_iri/checkpoints_bound/bound_v3.pt --num_envs 1 --real-time --command_ui --command 1.25 0.0 0.0 --device cuda:0
```

## Acceptance

At 1.0, 1.25, and 1.5 m/s require:

- forward-speed RMSE below 0.12 m/s;
- mean absolute lateral speed below 0.03 m/s;
- mean absolute heading error below 5 degrees;
- bound-pair pattern above 0.70 and diagonal pattern below 0.05;
- front/rear stance Jaccard above 0.85;
- front/rear action mirror MAE substantially below `bound_v2_1`;
- fewer than 1% base-contact terminations.

Do not collect diffusion data from a checkpoint that fails these behavioral
criteria even if `Train/mean_reward` is high.
