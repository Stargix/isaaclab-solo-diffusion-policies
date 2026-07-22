# Phase B1 research scope

## Research question

Can a small online residual policy improve route, orientation, speed and height
tracking of a low-data, goal-conditioned diffusion locomotion prior on novel
compositions, without destroying the behaviours already represented by that
prior?

The claim is deliberately narrower than “general navigation”. Phase A learned
walk/crouch endpoints from 6,690 demonstrations and already extrapolates to new
path geometry and intermediate height goals, but its failures increase when
route curvature and height transitions are composed out of distribution. B1
tests correction of that learned action manifold through interaction.

## Relation to the original TFG proposal

The original proposal is not a selector between independently deployed skills.
It uses `walk`, `crouch` and later `sprint` experts only to measure capability,
generate stable transitions and produce capability-aware demonstrations. The
final student removes skill IDs, velocity commands and the teacher schedule and
receives only robot state, future path, terminal pose and remaining time. Its
main question is whether one direct policy can infer the useful locomotion mode
and transition timing from those constraints.

Phase A already provides the first reduced instance of that idea: the
walk/crouch hindsight diffusion policy has demonstrated interpolation between
the endpoint behaviours. Therefore “skill interpolation is pending” would be
incorrect. What remains untested is the larger original claim: whether a direct
policy can choose when to exploit that interpolation (and, eventually, sprint)
from path geometry and deadline, without being given a speed/skill schedule.

B1 is an online-correction extension, not the original main contribution. Its
current route sampler supplies an average-speed value in `goal12`, so it tests
local robustness of an already conditioned prior rather than pure implicit
mode selection. This makes B1 a controlled adaptation study; it must not be
reported as evidence that the original path/deadline student has already been
solved.

## Why this interface

[Skill-Nav](https://arxiv.org/abs/2506.21853) motivates a waypoint interface:
it is more directly compatible with a planner than instantaneous velocity
commands, and trains first on fixed waypoints before randomized waypoints.
[Hierarchical RL for Quadruped Locomotion](https://arxiv.org/abs/1905.08926)
shows why reusing a low-level policy can make path-level adaptation efficient,
but a discrete or latent skill interface can restrict the planner to behaviours
encoded by its experts and makes the semantics of the interface harder to
audit. Here the interface is explicit SE(2) path geometry, terminal pose,
absolute height and average speed.

The frozen prior is a diffusion policy because action-sequence diffusion can
represent multimodal, high-dimensional behaviour and supports receding-horizon
execution, as established by
[Diffusion Policy](https://roboticsproceedings.org/rss19/p026.html). In this
project that is useful for retaining different walk/crouch strategies and their
interpolations in one policy instead of selecting one expert per skill. It does
not eliminate limited-data OOD errors; those are precisely B1's target.

[Residual RL](https://arxiv.org/abs/1812.03201) provides the appropriate first
online step: preserve the structured baseline and learn only the part it does
not solve. This is cheaper and safer to diagnose than RL from scratch, and it
gives a clean control through `residual_scale=0`. Direct diffusion fine-tuning
with [DPPO](https://arxiv.org/abs/2409.00588) is a justified later comparison
because it can explore on the diffusion manifold, but it changes the entire
prior and introduces denoising-step credit assignment and substantially higher
compute. It is Phase B2, not silently mixed into B1.

## B1 MDP

- **Frozen base policy:** Phase-A `spatial_hindsight_geometry_ddpm`; exact
  delayed proprio/action/goal histories are preserved. Its action history uses
  the action actually executed after residual correction.
- **Observation:** proprioception (30), geometric goal12, proposed diffusion
  action (12), previous residual (12), body velocity and route errors (9): 75D.
- **Action:** 12D PPO residual clipped to `[-1,1]`, scaled by 0.20 of each
  joint's demonstrated action half-range, added to the diffusion action and
  finally clipped to the demonstration action range.
- **Objective:** progress plus bounded rewards for path, tangent speed, yaw and
  height; fall/corridor failure and terminal success; small residual magnitude,
  residual-rate, tilt and vertical-velocity costs.
- **Curriculum:** (0) straight and constant endpoint heights; (1) straight,
  S-curve and right-angle paths with binary height sections; (2) random smooth
  curves and intermediate heights. Difficulty expands, but the MDP contract
  does not change.

No controller smooths height commands. The future terminal height is visible
in `goal12`, while the current section height is rewarded. Thus anticipation
and smoothness must be learned.

## Required comparisons and metrics

Use identical seeds and route suites for:

1. frozen Phase-A prior (`residual_scale=0`);
2. residual PPO (B1);
3. residual PPO with half action scale;
4. later DPPO fine-tuning;
5. RL from scratch only as a compute-matched negative control, not the primary
   baseline.

Report route success, survival, normalized progress, cross-track RMSE, tangent
speed MAE, height MAE/RMSE, terminal yaw error and residual RMS. Also rerun the
Phase-A in-distribution suite: improvement on OOD routes is not success if the
online layer regresses original walk/crouch performance.

## What B1 does not claim

B1 contains no perception, obstacle map, physical ceiling/clearance model,
terrain-conditioned planning, discrete skill selector, or learned OOD
estimator. It does not yet contain the original capability-aware expert data
generation, bridge transitions, deadline oracle or final direct
path/time-conditioned student. Those are the next main-TFG steps. Keeping them
out of B1 makes the causal result—whether local online correction fixes Phase-A
composition errors—clear.
