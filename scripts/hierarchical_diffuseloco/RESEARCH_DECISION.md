# Decision record: high-level composition now, DPPO later

## Decision

We will test both ideas, but as two independent experiments:

1. **H1, current branch `hierarchical-diffuseloco`:** train a high-level PPO that outputs
   `[vx, vy, wz, height]` over a frozen, command-conditioned DiffuseLoco policy.
2. **D1, future branch:** fine-tune the direct path-conditioned diffusion checkpoint with
   Diffusion Policy Policy Optimization (DPPO). D1 will not reuse the H1 environment,
   observations, PPO checkpoint or reward implementation implicitly.

This separation is methodological, not cosmetic. H1 asks whether planning can emerge by
composing a locomotion manifold; D1 asks whether online RL can improve the denoising policy
that already maps path geometry to joints. Mixing them would make any gain impossible to
attribute.

## Evidence used

The low-level prerequisite selected for H1 is
`checkpoints_iri/diffuseloco_robust_walk_crouch.pt` (SHA256
`2C41034B8ACF7E92AC28EF157F9F483F54C5EDA6B32976B8FBCCEA6E321676C0`). Its checkpoint
contract is `diffuseloco_velocity_height_ddpm`, `goal_dim=4`, trained explicitly on
`[vx, vy, wz, desired_height]`. The existing evaluation
`scripts/baseline_diffuseloco/evaluations/diffuseloco_robust_k4_h4/summary.json` contains
45 eight-second scenarios and reports survival 1.0. This is the correct abstraction for a
high-level command policy.

The direct checkpoint `checkpoints_iri/real_walk_crouch_hindsight.pt` remains a frozen D1
baseline. Its hindsight geometric goal is derived from achieved future motion. Random expert
commands generated diverse trajectories, but the command labels are not the path policy's
inputs. It is therefore useful as a direct path-conditioned prior, but using another network
to invert its systematic path-to-speed bias would be a patch rather than hierarchical skill
composition.

## H1 research question

> Can an online high-level policy exploit useful continuous combinations of a frozen
> velocity/posture diffusion prior to satisfy route geometry, final pose and desired mean
> speed, while preserving the prior's locomotion stability?

The contribution is not “DiffuseLoco on Solo12.” It is the combination of:

- offline diffusion distilled from pre-existing locomotion experts and randomly sampled
  commands;
- continuous online composition in a semantic four-dimensional command space;
- simultaneous route and posture requirements;
- a comparison against direct path-conditioned diffusion and a classical tracker.

This differs from discrete skill selection: the high-level can request intermediate speeds,
lateral/yaw mixtures and intermediate heights. It also differs from end-to-end locomotion RL:
PPO never outputs joint targets and the diffusion prior is frozen.

## H1 architecture and scope

Observation: eight robot-frame preview samples
`(x, y, sin(yaw), cos(yaw), target_height, valid)`, final local pose, remaining horizon,
scheduled-progress error, desired mean speed, base velocity/orientation/height state, tracking
error and the previous high-level command.

Action: normalized four-vector mapped to the measured low-level support:

```text
vx     [0.00, 0.65] m/s
vy    [-0.45, 0.45] m/s
wz    [-0.50, 0.50] rad/s
height [0.1705, 0.2932] m
```

H1 is forward-only and flat-ground. It uses one seeded distribution containing straight,
S-shaped and rounded right-angle routes with interleaved posture targets. The route length is
`v_desired * 8 s`, so final longitudinal displacement directly defines mean-speed accuracy.
There is no curriculum or oracle.

## Why this reward

For progress `s`, desired mean speed `v*` and horizon `T=8 s`:

```text
s*(t) = min(v* t, L), where L = v* T
Phi(s,t) = -Huber((s - s*(t)) / 0.30)
r_progress = 4 [gamma Phi(s_next,t_next) - Phi(s,t)]
```

This potential makes falling behind negative and catching up positive without paying an
unbounded penalty merely for remaining alive. The rest of the reward is:

```text
- 0.60 Huber(cross_track / 0.30) dt
- 0.25 Huber(heading_error / 0.50) dt
- 0.45 Huber((physical_height - target_height) / height_range) dt
- 0.015 mean((command - previous_command)^2)
- 4.00 terminal Huber(final_x, final_y, final_yaw) only on done
- 12.0 if the base falls
```

There is deliberately no alive reward, raw progress bonus, instantaneous-speed reward,
standstill penalty, separate time penalty or early route-success termination. The fixed
horizon plus final pose enforces mean speed; scheduled progress only makes that terminal
objective learnable. Height uses the measured base height, not merely the requested height.
Smoothness is learned from return: no controller filters the command.

## Minimum acceptance gate

Do not judge H1 from mean reward alone. On held-out route-bank seeds, compare:

- frozen DiffuseLoco with the classical preview tracker;
- learned continuous high-level PPO;
- direct path-conditioned `real_walk_crouch_hindsight.pt` as an architectural baseline.

Primary metrics are survival, route-success rate at 8 s, final position error, mean-speed
error and cross-track RMSE. Secondary metrics are physical height RMSE and command total
variation. H1 is successful only if PPO improves route/final-speed tracking over the classical
tracker without materially reducing survival. One development seed is enough to reject a
broken design; use three seeds only for the final reported result.

## D1: future DPPO branch

D1 starts from `real_walk_crouch_hindsight.pt`, not from the velocity-height checkpoint and
not from an H1 checkpoint. Its action is the diffusion denoising trajectory that ultimately
produces joint actions. Implementing it correctly requires:

1. a stochastic DDPM sampling path whose transition log-probability is available at every
   denoising step;
2. PPO ratios/clipping on those denoising transitions, with advantage attached to the
   executed physical chunk;
3. a frozen reference policy and a small KL penalty to constrain updates near the pretrained
   model;
4. frozen normalizer and goal representation, short execution chunks, conservative learning
   rate and frequent checkpoints;
5. the same outcome metrics as the direct frozen baseline, plus KL, denoising-ratio clipping
   fraction and action-distribution drift.

The D1 reward should express outcomes—survival, route/final pose, mean speed and physical
height—not denoising loss. Start with the same fixed-horizon objective concept as H1, adapted
to the direct policy's observation contract. Do not call the existing residual joint-action
PPO “DPPO”: residual PPO optimizes an additive 12-D correction and has neither denoising
log-probabilities nor a diffusion-policy likelihood ratio.

D1 is attempted after H1 because it has more implementation and optimization risk. If H1
fails while the classical tracker succeeds, the likely issue is high-level PPO/observation
design. If both H1 and the tracker fail, the low-level command capability is insufficient. If
the direct frozen path checkpoint is already strongest but has a consistent correctable bias,
D1 becomes the most valuable next experiment.

## Literature basis

- DiffuseLoco: diffusion as a multimodal locomotion controller conditioned on task commands
  ([paper](https://arxiv.org/abs/2404.19264)).
- Skill-Nav: navigation by composing reusable low-level skills rather than relearning joint
  control end to end ([paper](https://arxiv.org/abs/2506.21853)).
- DPPO: policy-gradient fine-tuning by treating diffusion denoising as the policy's stochastic
  decision process ([paper](https://arxiv.org/abs/2409.00588)).
- Residual RL: additive corrections are useful when the prior is close, but optimize a
  different policy class from diffusion fine-tuning
  ([paper](https://arxiv.org/abs/1812.03201)).
- DAgger: why offline imitation can fail under the state distribution induced by its own
  errors ([paper](https://proceedings.mlr.press/v15/ross11a.html)).
