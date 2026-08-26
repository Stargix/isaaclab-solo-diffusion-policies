# Solo12 bound expert (0.8--1.5 m/s)

## Scope

`solo12-bound-v0` trains one visibly distinct high-speed expert without changing
`solo12-v0`, `solo12-crouch-v0`, or the historical `solo12-sprint-v0` experiment.
The target gait is a **symmetric bound**: FL/FR are synchronized, RL/RR are
synchronized, and the front and rear pairs are separated by half a cycle. It is
not labelled gallop because a gallop is asymmetric and would add a leading-leg
choice that is unnecessary for the first sprint skill.

The training and collection ceiling is 1.5 m/s. Training at 2 m/s and later
discarding that region would expose PPO to an irrelevant, harder actuator regime
and spend capacity outside the downstream dataset.

## Evidence behind the environment

- [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf)
  represents gait through a timing clock and phase offsets, conditions the policy
  on that clock, and combines positive task reward with auxiliary gait/stability
  terms. Its bound offsets synchronize front and rear pairs separately.
- [Robust High-speed Running](https://arxiv.org/abs/2103.06484) shows that bound
  and gallop are natural high-speed solutions, but also warns that joint-space
  policies can exploit simulator dynamics. This task therefore keeps dynamics
  randomization and explicit contact-quality metrics.
- [Solo12 deep RL locomotion](https://www.nature.com/articles/s41598-023-38259-7)
  supports joint-position targets, velocity tracking, smoothness/energy costs,
  randomization, and curriculum learning on this robot. It does not prescribe a
  0.25 m sprint height; here 0.25 m is only a soft preference.

## MDP and reward

The action remains the repository's 12 joint-position offsets through PD control.
The normal 48-D proprioceptive/command observation receives a 2-D
`[sin(phase), cos(phase)]` clock. Two values are used because a memoryless MLP
cannot disambiguate the rising and falling halves of a lone sine.

The task reward is always positive:

```text
r = velocity_score * (3.0 + 0.5 * yaw_score)
    * exp(-0.5 * auxiliary_cost) * dt
```

Gating yaw by velocity prevents a stationary policy from collecting a useful
yaw-only reward. `auxiliary_cost` contains bounded contact-schedule error,
swing contact force, stance slip, roll, pitch outside an 18-degree corridor,
soft height error, action rate, torque, thigh contact, and a weak vertical-speed
cost. Pitch and vertical motion are not forced to zero because they are intrinsic
to a bound. Falling terminates the stream of positive task reward; there is no
large terminal penalty that could destabilize PPO.

The speed curriculum is forward-only:

```text
stage 0: vx in [0.60, 1.00] m/s
stage 1: vx in [0.70, 1.25] m/s
stage 2: vx in [0.80, 1.50] m/s
then:    light lateral pushes up to 2 N
```

This also fixes the old sprint task's hidden mismatch where the inherited
curriculum replaced a nominally positive command range with `[-v, +v]`.

## Cluster training

Use the repository's extended RSL-RL entry point so the run saves
`best_model.pt` and one best checkpoint per curriculum stage:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task solo12-bound-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 42 \
  --run-name solo12_bound_1p5_seed42 \
  --symmetry-mode none \
  --headless \
  --device cuda:0
```

Do not resume a 48-D walk/sprint checkpoint: the bound policy has a 50-D input
because of its phase clock. Training from scratch preserves a clean optimizer and
normalizer state. A transfer experiment would require an explicit 48-to-50 input
adapter and is not part of this baseline.

## Visual check

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task solo12-bound-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/solo12_rsl_rl_bound_runs/<run>/best_model.pt \
  env.command_lin_vel_x_range='[1.25,1.25]' \
  env.command_lin_vel_y_range='[0.0,0.0]' \
  env.command_ang_vel_z_range='[0.0,0.0]'
```

## Acceptance before collecting data

Do not select a checkpoint from mean reward alone. At curriculum stage 2 verify:

- `Episode_Termination/base_contact` is below 1% of completed episodes;
- velocity tracking is tested separately at 0.8, 1.0, 1.25, and 1.5 m/s;
- `Metrics/front_pair_sync` and `Metrics/rear_pair_sync` are above 0.90;
- `Metrics/front_rear_opposition` is near 0.8 (the theoretical value is below
  one because duty factor 0.40 deliberately includes flight windows);
- height oscillates around the 0.25 m preference without a persistent roll;
- contact traces visibly show two front strikes alternating with two rear strikes.

If tracking is good but contact topology is wrong, do not compensate by raising
the height or fall penalty. Diagnose contact order and the phase metrics first.

## Diffusion-data collection

Both data collectors recognize the explicit `bound` skill and sample only
`vx=[1.0,1.5]`, `vy=[-0.1,0.1]`, `wz=[-0.25,0.25]`. The expert itself trains
down to 0.6--0.8 m/s for starts and recovery, but those samples are excluded to
avoid conflicting with the existing walk/crouch data in their overlap region.

```bash
./isaaclab.sh -p scripts/diffusion_policy/data/collect_data.py \
  --mode single \
  --task solo12-bound-v0 \
  --checkpoint logs/rsl_rl/solo12_rsl_rl_bound_runs/<run>/best_model.pt \
  --skill_name bound \
  --desired_base_height 0.25 \
  --num_envs 256 \
  --num_steps 1500000 \
  --command_resample_time_s 2.0 \
  --physics_dr_mode light \
  --seed 42 \
  --output_name bound_raw_v1.hdf5 \
  --headless
```
