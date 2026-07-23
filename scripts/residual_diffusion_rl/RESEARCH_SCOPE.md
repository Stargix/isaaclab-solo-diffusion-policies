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
It uses `walk`, `crouch` and later `sprint` experts as data sources, but the
experts, a schedule and an oracle are not required at deployment. The final
student removes skill IDs and receives only robot state, future path, terminal
pose and remaining time. Its main question is whether one policy can infer the
useful locomotion behaviour and transition timing from those constraints.

Phase A already provides the first reduced instance of that idea: the
walk/crouch hindsight diffusion policy has demonstrated interpolation between
the endpoint behaviours. Therefore “skill interpolation is pending” would be
incorrect. What remains untested is the larger original claim: whether a direct
policy can choose when to exploit that interpolation (and, eventually, sprint)
from path geometry and deadline, without being given a speed/skill schedule.

B1 is an online-correction extension, not the original main contribution. Its
current route sampler supplies an average-speed value in `goal12`, so it tests
local robustness of an already conditioned prior rather than pure implicit
mode selection. The important research premise is deliberately data-light:
Phase A uses randomized hindsight conditions rather than a hand-designed
schedule, yet already generalizes to most tested path compositions and
transitions. A capability oracle may be used later as an evaluation upper
bound, but it should not be a required training component unless random data
fails on a clearly identified case. B1 must not be reported as evidence that
the original path/deadline student has already been solved.

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
  action (12), previous residual (12), body velocity and route errors (9), plus
  schedule and mean-progress-speed errors (2): 77D. The two timing features
  are required because PPO is feed-forward; omitting them makes an
  average-speed objective partially observable.
- **Action:** 12D PPO residual clipped to `[-1,1]`, scaled by 0.10 of each
  joint's demonstrated action half-range, added to the diffusion action and
  finally clipped to the demonstration action range. The conservative default
  reflects that the prior already solves most trajectories.
- **Objective:** speed-gated progress and terminal success, with
  non-saturating Huber costs for path, instantaneous tangent speed, progress
  schedule, yaw and height; fall/corridor/timeout failure; small residual
  magnitude, residual-rate, tilt and vertical-velocity costs. There is no
  alive reward or independent time cost.
- **Terminal contract:** the robot must be within 0.12 m of the actual final
  point, satisfy final yaw/height, be nearly stopped, and have final
  `progress / elapsed_time` within 0.05 m/s of requested average speed for five
  consecutive control steps. Projection onto the last route sample alone is
  insufficient.
- **Route distribution:** the default trains directly on stage 2 (straight,
  S-curve, right-angle and random smooth paths with endpoint/intermediate
  heights). A staged curriculum is retained only as an ablation because the
  frozen prior is already competent on the target distribution.

No controller smooths height commands. The future terminal height is visible
in `goal12`, while the current section height is rewarded. Thus anticipation
and smoothness must be learned.

## Average-speed objective and reward

Let `s_t` be monotone arc-length progress, `L` route length, `v*` requested
average speed and `t` elapsed episode time. B1 derives timing from the speed
command:

```text
expected_progress(t) = min(v* t, L)
schedule_error(t)    = s_t - expected_progress(t)
mean_speed_error(t)  = s_t / t - v*
```

`schedule_error` is the dense learning signal. It rules out both sprinting
ahead and lagging behind while remaining symmetric around the requested
schedule. `mean_speed_error` is observed by the feed-forward actor and is a
hard terminal condition. It is not a second time/deadline command.

The step objective is:

```text
+ progress_delta * Gaussian(tangent_speed_error)
- Huber(cross_track, tangent_speed, schedule, yaw, height)
- L2(residual, residual_rate, tilt, vertical_velocity)
+ terminal_success
- task_failure
- extra_fall_cost
```

Huber losses are zero at the target, quadratic for small errors and linear for
large errors. The old bounded exponential error cost saturated at exactly the
point where sprinting should become increasingly undesirable. Schedule error
has weight 0.1 because it persists throughout an episode; this keeps a fully
failed route on the same return scale as the other task terms without removing
its non-saturating gradient. Speed-gating progress also makes a metre travelled
at the requested speed more valuable than a metre obtained by exploiting route
projection at excessive speed.

There is no per-step survival reward and no time penalty. A correct trajectory
therefore does not score more merely because it lasts longer; route duration is
fixed by `L / v*`. Height commands remain piecewise constant. Smooth lowering
must emerge from future-goal anticipation, the height objective,
vertical-velocity cost and residual-rate cost, not a hand-written ramp.

The PPO defaults are deliberately local: residual authority 0.10, initial
action standard deviation 0.10, clipping 0.10 and fixed learning rate `1e-4`.
The fixed rate prevents the adaptive scheduler from increasing step size while
the target distribution changes. `gamma=0.999` retains terminal information
over a 10--22 second route at 50 Hz; `gamma=0.99` discounts that horizon almost
completely. The environment uses a separate zero-based task clock for timing
errors. RSL-RL may therefore randomize its internal initial episode length to
desynchronize timeout/reset waves without changing route time; shortened
startup rollouts are treated as neutral truncations rather than task failures.

## Required comparisons and metrics

Use identical seeds and route suites for:

1. frozen Phase-A prior (`residual_scale=0`);
2. residual PPO (B1);
3. residual PPO with half action scale;
4. later DPPO fine-tuning;
5. RL from scratch only as a compute-matched negative control, not the primary
   baseline.

Report route success, survival, normalized progress, cross-track RMSE, tangent
speed MAE, final mean-speed MAE, height MAE/RMSE, terminal position/yaw error
and residual RMS, globally and by route family. Also rerun the
Phase-A in-distribution suite: improvement on OOD routes is not success if the
online layer regresses original walk/crouch performance.

The termination report separates route success, physical base contact, corridor
exit and timeout. The upstream Solo12 aggregate labels all non-timeout resets
as base contact, which is not valid once route success is also terminal.

## What B1 does not claim

B1 contains no perception, obstacle map, physical ceiling/clearance model,
terrain-conditioned planning, discrete skill selector, or learned OOD
estimator. It also does not claim to solve the complete direct
path/time-conditioned student. Crucially, it does not depend on a capability
oracle: the main comparison is sparse randomized hindsight data versus the
same prior with online correction. Extra experts and harder transitions can be
added incrementally, testing whether the learned manifold and the residual
layer scale without redesigning the planner. Keeping these factors separate
makes the causal result—whether local online correction fixes Phase-A
composition errors—clear.
