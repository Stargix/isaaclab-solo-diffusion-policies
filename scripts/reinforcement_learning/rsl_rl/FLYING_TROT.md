# Solo12 flying-trot expert

## Objective

Train a fast expert useful to the later multiskill diffusion policy without
relearning a fragile contact topology. The policy retains the 48-D observation
and 12-D joint-position action contracts of `walk_final.pt`.

The final command distribution is:

- forward velocity: 1.0--1.5 m/s;
- lateral velocity: -0.15--0.15 m/s;
- yaw rate: -0.35--0.35 rad/s.

Forward tracking has most of the reward weight. Lateral and yaw tracking are
independent secondary objectives, so their errors cannot suppress the forward
gradient. A small clock-free diagonal-contact score protects the walk policy's
trot topology. Air time is rewarded only on touchdown; there is no per-step
flight reward that could be exploited by jumping.

The speed curriculum is 0.75--1.0, 0.85--1.25 and 1.0--1.5 m/s. Full Solo12
quadruped augmentation applies identity, left/right reflection, front/back
reflection and their 180-degree composition to observations, commands and
actions.

## Train

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task solo12-flying-trot-v0 \
  --warm_start_checkpoint checkpoints/walk_final.pt \
  --num_envs 4096 \
  --max_iterations 2500 \
  --headless \
  --device cuda:0 \
  --run_name flying_trot_1p5_v1
```

`--warm_start_checkpoint` deliberately loads only the actor and its observation
normalizer. The critic, PPO optimizer, exploration noise and iteration counter
start fresh because the reward and command distribution changed.

The first run deliberately uses clean flat simulation, low exploration noise
and gentle resets. Domain randomization is not mixed into gait acquisition: it
can be introduced conservatively only after a checkpoint passes the acceptance
criteria below.

Do not use `--resume` together with `--warm_start_checkpoint`. Use `--resume`
only to continue a flying-trot run from its own log directory.

## Compare and visualize

The gait comparison script supports the `walk_flying_trot` preset. Both
policies are evaluated in the flying-trot environment, so their 48-D actor
contracts are checked explicitly before rollout:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py \
  --comparison walk_flying_trot \
  --walk_checkpoint checkpoints/walk_final.pt \
  --flying_trot_checkpoint checkpoints_iri/checkpoints_bound/flying_trot.pt \
  --speed 1.25 --duration_s 8 --warmup_s 2 --num_envs 16 \
  --output_dir scripts/reinforcement_learning/rsl_rl/evaluations/walk_vs_flying_trot_125 \
  --headless --device cuda:0
```

For interactive inspection, `play.py --command_ui` exposes vx, vy and yaw
controls for this task (omit `--headless`):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task solo12-flying-trot-v0 \
  --checkpoint checkpoints_iri/checkpoints_bound/flying_trot.pt \
  --num_envs 1 --command_ui --command 1.25 0.0 0.0
```

## Acceptance criteria

Evaluate deterministic checkpoints at 1.0, 1.25 and 1.5 m/s. Promote the skill
only if it reaches all of the following rather than selecting by training reward
alone:

- survival >= 95%;
- forward-speed RMSE <= 0.12 m/s;
- mean absolute lateral tracking error <= 0.05 m/s;
- mean absolute yaw-rate tracking error <= 0.10 rad/s;
- diagonal-trot pattern fraction >= 0.60;
- flight fraction between 0.08 and 0.25;
- pitch RMS <= 0.10 rad.

## Design evidence

- Aractingi et al., *Controlling the Solo12 quadruped robot with deep
  reinforcement learning*: make commanded velocity the main positive reward
  and introduce physical penalties after task acquisition.
- Tan et al., *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*:
  agile gaits can emerge from simple task rewards; explicit references are most
  useful when an exact gait must be imposed.
- Margolis and Agrawal, *Walk These Ways*: frequency, phase and duty factor are
  valid gait descriptors, but they are not added to the policy contract here
  because the proven walk prior already contains a diagonal phase manifold.
