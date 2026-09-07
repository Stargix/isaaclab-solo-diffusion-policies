# Final project and paper plan: skill-agnostic path, pose and speed conditioning

Date: 2026-09-07
Status: final research scope and experiment order; Phase A profile implementation prepared on `path-pose-profile-v2`

## Executive decision

Sprint remains in the intended paper, but it is not allowed to block the core
result and it is not treated as already solved by the current
`dppo_trot_ab.pt` checkpoint.

The paper has two nested questions:

1. **Core composition:** can one diffusion action policy follow a local path and
   transition between walk and crouch from spatial path/pose/height constraints,
   without receiving a skill ID?
2. **Overlapping-skill extension:** can the same interface incorporate a fast
   locomotion expert and use its action manifold when route geometry and the
   requested average speed make that additional capability useful, again
   without a skill ID or a gait reward?

This preserves the original project objective. The network is told what the
robot must achieve, not which expert generated the desired behavior.

Sprint is scientifically valuable because it turns a two-mode posture problem
into a harder capability-composition problem. It is not, by itself, the
novelty: DiffuseLoco already learns several locomotion skills from multimodal
offline data, and LocoDiff studies multi-skill interpolation. The differentiating
combination here is:

- pretrained RL policies reused as heterogeneous data sources;
- small random-command datasets with hindsight geometric relabeling;
- robot-centric path and spatial pose constraints;
- no skill label at inference;
- a route-level average-speed objective that can require local slowing and
  later recovery;
- optional direct online refinement of the diffusion distribution with DPPO.

Relevant related work:

- [DiffuseLoco](https://proceedings.mlr.press/v270/huang25a.html)
- [LocoDiff](https://arxiv.org/abs/2411.08832)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [DPPO](https://openreview.net/forum?id=mEpqHvbD2h)
- [Kimodo constraint representation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html)

## Final research question

> Can a skill-unlabelled diffusion policy distil independently trained Solo12
> locomotion experts into a single path-, pose- and average-speed-conditioned
> controller, compose their capabilities along unseen routes, and be refined
> online without losing the modes learned offline?

The corresponding sub-questions are:

1. Does a multi-point route preview outperform a terminal local target?
2. What route-preview horizon is sufficient for anticipation without causing
   premature posture changes?
3. Can walk-crouch transitions emerge from disjoint expert datasets, or are
   explicit transition demonstrations required?
4. Does adding the fast expert expand the feasible path-speed envelope rather
   than merely changing the visual gait?
5. Can DPPO improve first-arrival timing and robustness without degrading
   survival, sustained height tracking or mode retention?

## Explicit non-goals

The final main method does **not** add:

- `skill_idx` to the actor input;
- a trot/diagonal-contact/duty-factor reward;
- a high-level skill selector;
- a hand-written height interpolator or smooth transition controller;
- pitch, roll, bipedal walking or further locomotion skills;
- difficult terrain as a prerequisite for the main claim.

`skill_idx` remains dataset provenance only. Contact and gait statistics are
evaluation measurements only. They never provide gradients to the policy.

## Why the current fast checkpoint is not the final sprint result

The current `dppo_trot_ab.pt` can reach routes and track some speed targets, but
it does not demonstrate reliable fast-skill selection:

- sustained-height evaluation shows a low-posture compromise;
- the old actor receives only the height at the end of its two-second preview,
  while the reward evaluates the height at current route progress;
- online stage-2 routes sample heights from
  `[0.2932, 0.25, 0.21, 0.1705]` and omit the fast expert's nominal `0.28 m`;
- height and speed are sampled independently, producing unsupported
  high-speed crouch and intermediate-height combinations;
- the fast and walk experts overlap in speed and both use diagonal locomotion;
- the DPPO run without reference KL was free to find a low, stable compromise.

The plot where the actual height anticipates every required-height transition
has a direct explanation. At `0.4 m/s`, a two-second lookahead covers `0.8 m`,
exactly the old height-section length. The scalar future height therefore
describes nearly the next section for most of the current one. Anticipation is
desirable, but replacing the current requirement with the future one is not.

The current checkpoint is retained as a negative/diagnostic result. It must not
be presented as evidence that the fast skill is already used.

## Final conditioning contract

The final actor must observe a spatial posture profile, not a single terminal
height. The proposed fixed-slot main representation is:

```text
(x1, y1, h1)
(x2, y2, h2)
(x3, y3, h3)
(x4, y4, h4)
h_required_now
sin(yaw_terminal), cos(yaw_terminal)
average_speed_required
```

This is a 16-D goal when the fourth path token is also the local terminal
position. The tokens are at `25, 50, 75, 100%` of the local look-ahead arc.
For the nominal constant-speed two-second horizon these correspond approximately
to `0.5, 1.0, 1.5, 2.0 s`, without assigning a deadline to each waypoint.

Properties of this contract:

- the current requirement and future preparation are simultaneously
  observable;
- the policy can learn when to start a transition from demonstrations;
- lowering before a clearance section does not erase the requirement to remain
  low until its exit;
- speed stays a route-level performance request, not a skill name;
- no deployment controller forces smoothness.

The target should remain robot-centric and local. A distant global target is
not mixed with centimetre-scale local constraints.

## Role of sprint in the final task

Sprint must be tested as a **functional capability**, not as a visual label.

The desired behavior on a mixed route is:

1. use conservative locomotion through high curvature or a low-height section;
2. preserve route and clearance safety;
3. use the additional high-speed support on open sections when necessary to
   recover the requested route-average speed;
4. arrive with the correct terminal pose and mean speed.

The route does not say `walk` or `sprint`. Curvature is present in the path
preview, height is present in the posture profile, and timing is represented by
the requested route-average speed or remaining speed budget.

For this experiment to be identifiable, the requested task must sometimes lie
outside the comfortable envelope of the walk-crouch dataset but inside the
three-skill dataset. A comparison limited to `<=0.6 m/s` cannot show the value
of the fast data. Conversely, `1.5 m/s` on every path is outside the current
safe closed-loop envelope and is a stress test, not the main domain.

The initial main sprint domain should be approximately:

- average route speed: `0.65-0.9 m/s`;
- fast local straight support: up to approximately `1.1-1.2 m/s`;
- lower local speeds around tight curvature and crouch sections;
- `1.5 m/s` reported separately as OOD/high-speed stress.

Whether the executed actions resemble the original fast expert is checked after
the task evaluation using contact, duty-factor, flight-fraction and action-space
metrics. Those metrics do not appear in the reward.

## Dataset sequence

### Dataset A-WC: disjoint walk and crouch

Reuse the frozen clean walk/crouch demonstrations. Recompute the new 16-D goals
without recollecting data. There are no transition windows.

Purpose: test whether the new observable posture profile permits zero-shot
composition of disjoint manifolds.

Implementation and the immutable cluster recipe are in
`phase_a_pose_profile_v2/`. The new checkpoint contract is schema version 8;
schema-7 `hindsight_geom_avg12` checkpoints and their inference path are not
modified.

### Dataset B-WC: size-matched walk-crouch transitions

Add survival-filtered expert switches and replace approximately `10-20%` of
ordinary A-WC windows with transition windows. Keep the total number of windows,
optimizer updates, architecture and sampling budget matched to A-WC.

Purpose: isolate the value of transition content rather than extra data or
training time. No action blending is used as a deployment mechanism.

### Dataset C-WCS: add the fast expert

Merge the already audited fast dataset with the best walk-crouch dataset and
train from scratch for the primary comparison. Use balanced sampling and report
the exposure per skill.

Purpose: test expansion of the feasible path-speed envelope and implicit use of
the fast manifold.

The first C-WCS model need not contain walk-fast transition demonstrations. If
it preserves walk/crouch but cannot enter or leave the fast regime, one final
predeclared variant adds a small, size-controlled set of walk-fast transitions.
No further data variant is allowed after that without changing the paper scope.

### Support-aware online routes

Before DPPO, route conditions must respect the demonstrated joint support:

- crouch-like height around `0.1705 m`: conservative speeds;
- walk-like height around `0.2932 m`: normal locomotion speeds;
- fast expert height around `0.28 m`: high forward speed and limited
  lateral/yaw demand;
- intermediate heights: a separate interpolation split, not the dominant
  online training distribution.

Random commands remain important. The correction is to sample randomly within
feasible joint regions instead of independently sampling every marginal.

## Route, waypoint and horizon ablations

Four different horizons must not be conflated:

| Quantity | Meaning | Final treatment |
|---|---|---|
| goal/lookahead horizon | future route visible to actor | central ablation |
| waypoint count/spacing | route representation resolution | central ablation |
| action prediction horizon | generated action trajectory length | fixed at 16 |
| execution horizon | open-loop actions before replanning | deployment ablation |

### Waypoint count and separation

At a fixed two-second goal horizon, compare:

| Variant | Future tokens | Nominal temporal spacing |
|---|---:|---:|
| terminal-only | 1 | 2.0 s endpoint only |
| sparse preview | 2 | 1.0 s |
| main preview | 4 | 0.5 s |

The hypotheses are predeclared:

- terminal-only aliases paths that share an endpoint but have different local
  curvature;
- two tokens recover a coarse direction change;
- four tokens should represent curves and posture intervals without excessive
  input size.

Eight tokens are not part of the initial paper sweep. They are only justified
if four tokens demonstrably alias the held-out path set.

### Goal horizon

With four uniformly spaced tokens, compare:

```text
T_goal = 1 s, 2 s, 3 s
```

- `1 s` tests insufficient anticipation;
- `2 s` is the main expected compromise;
- `3 s` tests excessive anticipation and irrelevant distant constraints.

The paper uses a one-factor-at-a-time design around `N=4, T=2 s`; it does not
train the full Cartesian product. The five offline screening configurations are:

```text
N=4, T=2 s   main
N=1, T=2 s
N=2, T=2 s
N=4, T=1 s
N=4, T=3 s
```

Before policy training, a no-simulation geometric audit measures curvature
aliasing and path reconstruction error for these representations. This provides
an independent explanation of waypoint spacing at negligible training cost.

### Height-section length

Height-section length is a task property, not waypoint spacing. Training must
randomize transition positions and section lengths, initially around
`1.2-2.4 m`, instead of repeating a fixed `0.8 m` period.

At `0.4 m/s`, evaluate:

- `2.4 m / 6 s`: easy sustained section;
- `1.6 m / 4 s`: main section;
- `0.8 m / 2 s`: short stress test.

The final paper reports transition timing as both metres and seconds so results
are not silently speed-dependent.

### Execution horizon and denoising steps

After selecting the final checkpoint, evaluate without retraining:

- `H_exec = 1, 4, 8` actions;
- `K = 10` original DDPM chain and `K = 5` accelerated inference.

Report tracking, survival, transition response, inference p95/p99 and deadline
margin. Prediction horizon 16 and history 8 remain fixed because they are not
central research variables.

## Metrics and checkpoint selection

### Safety and route

- survival and base-contact failure;
- first-arrival rate and time;
- route completion ratio;
- cross-track RMSE and p95;
- terminal position and yaw error;
- mean-speed error measured at first arrival.

### Posture and transitions

- sustained height MAE per route section;
- fraction of path within height tolerance;
- maximum clearance violation in low sections;
- signed anticipation distance/time at each boundary;
- settling distance/time;
- premature rise before a low section ends;
- retained high/low plateau duration;
- vertical oscillation and action-delta RMS.

### Fast-capability analysis

- local speed by curvature and height section;
- amount of accumulated timing debt and subsequent recovery;
- task envelope with and without fast data;
- distance in action/statistics space to each source expert;
- diagonal contact correlation, duty factor and flight fraction as diagnostics.

### Lexicographic checkpoint selection

`best.pt` is not selected by scalar training return alone:

1. reject unsafe checkpoints;
2. reject checkpoints that lose sustained posture causality;
3. among the remaining checkpoints, maximize first-arrival task success;
4. use mean-speed and cross-track error as tie-breakers;
5. for C-WCS, also require no material regression on the frozen WC suite.

Indicative in-support gates are:

| Metric | Gate |
|---|---:|
| survival | `>=95%` |
| base contact | `<=2%` |
| arrival | `>=90%` |
| cross-track RMSE | `<=0.08-0.10 m` |
| terminal position error | `<=0.20 m` |
| mean-speed error | `<=0.08 m/s` |
| sustained height MAE | `<=0.025-0.03 m` |
| crouch-section compliance | `>=95%` |
| safe transition rate | `>=90%` |

The final paper reports confidence intervals and per-condition results rather
than presenting the gates as universal physical thresholds.

## Final model comparison

The minimum defensible comparison is:

| Model | Scientific purpose |
|---|---|
| source experts | capability and gait reference |
| deterministic predictor, same data/goal/actions | test whether diffusion adds value |
| diffusion A-WC | transition-free composition |
| diffusion B-WC | value of transition demonstrations |
| diffusion C-WCS | value of the fast expert and scalability |
| best diffusion + DPPO | value and cost of online refinement |
| existing hierarchical command policy | alternative interface and negative result |

The main actor never receives a skill label. A skill-conditioned actor is not a
required experiment for this project.

## DPPO scope

DPPO is run only after the selected offline checkpoint passes the safety,
geometry and sustained-height gates.

Its question is narrow:

> Can online fine-tuning improve first-arrival timing, route completion and
> robustness while preserving the capability composition learned offline?

The reward remains functional:

- route progress and schedule improvement;
- cross-track and yaw error;
- current height/clearance requirement;
- terminal pose;
- requested average speed at first arrival;
- physical failure;
- conservative reference regularization.

There is no gait reward. A small or adaptive reference KL is used to prevent
the cumulative mode drift observed in the no-KL fast run. DPPO is positive only
if it improves arrival/timing without more than a small survival regression and
without worsening sustained-height or WC retention.

## Experiment order and hard stop rules

### Phase 1: contract and evaluator

1. Implement the 16-D posture-profile condition in a new schema/version.
2. Unit-test agreement between dataset builder, online route builder, player
   and evaluator.
3. Add sustained-height, boundary-timing and fast-capability plots.
4. Freeze development and final-test route seeds.

No training starts until all four consumers produce byte/numerically equivalent
goals for the same synthetic route.

### Phase 2: WC core and architecture screening

1. Rebuild A-WC goals from existing data.
2. Run the five one-seed waypoint/horizon screening models.
3. Select one representation on development routes only.
4. Train/evaluate the deterministic baseline.
5. Build and evaluate size-matched B-WC.

If neither A-WC nor B-WC passes sustained-height and safety gates, do not add
sprint or DPPO. Diagnose the failed contract once; at most one correction is
allowed before freezing a negative conclusion.

### Phase 3: sprint extension

1. Train C-WCS from scratch using the audited fast data and selected contract.
2. Evaluate WC retention before testing high speed.
3. Compare WC and WCS on identical mixed routes at `0.65-0.9 m/s`.
4. Measure whether WCS slows in constrained/curved sections and recovers speed
   on open sections.
5. Inspect gait/contact metrics only after task metrics are known.

If C-WCS cannot switch into/out of the fast regime but retains the core task,
run exactly one transition-data variant. If that fails, report the overlapping-
skill limitation and stop; do not introduce skill IDs or gait rewards.

### Phase 4: online refinement

1. Run one DPPO pilot from the best offline model.
2. Cancel on safety/posture regression even if return increases.
3. If positive, run the remaining training seeds.
4. Evaluate the locked test set once after checkpoint selection.

### Phase 5: confirmation and deployment

- one training seed is sufficient for architecture screening;
- the central A/B and final DPPO comparisons use three training seeds;
- multiple rollout seeds do not substitute for training seeds;
- evaluate `H_exec/K` only on the final selected actor;
- use a conservative WC route for first hardware validation.

The project is considered complete when A/B, waypoint/horizon, deterministic
baseline and offline/DPPO comparisons have conclusive results. Sprint may be a
positive secondary claim or a documented overlapping-skill limit; either is a
valid completion, provided the experiment follows the predeclared gate.

## Paper structure and claims

### Primary claims

1. Random-command expert data plus hindsight relabeling can train a local
   path/pose-conditioned locomotion diffusion policy.
2. Spatial posture profiles enable skill-unlabelled walk-crouch composition.
3. The A/B comparison identifies whether explicit transition coverage is
   necessary.
4. Waypoint resolution and preview horizon have measurable anticipation versus
   aliasing trade-offs.

### Sprint claim, if successful

> Adding a partially overlapping fast expert expands the feasible path-time
> envelope, and the unified policy uses fast-manifold behavior selectively on
> locally compatible route sections without receiving a skill label.

This is stronger than claiming that a robot can visually trot. It connects the
extra mode to route-level utility.

### Sprint claim, if unsuccessful

> Spatial task conditioning composes capabilities separated by a strong
> physical constraint such as body height, but does not reliably identify
> partially overlapping locomotion modes unless the task makes their advantage
> observable and the data cover entry/exit transitions.

This is still a useful boundary result and explains why an unconstrained DPPO
actor collapsed to a low-posture compromise.

### DPPO claim

DPPO is presented as an offline-to-online refinement ablation, not as the source
of locomotion skills. Its benefit must be measured against retention of the
offline behavior.

## Expected paper artifacts

1. Method diagram from source experts to hindsight dataset, unified diffusion
   actor and optional DPPO refinement.
2. Dataset support plot for walk, crouch and fast locomotion.
3. XY held-out route plots.
4. Required/preview/actual height plots with boundary timing.
5. A-WC versus B-WC transition figure.
6. Waypoint-count and goal-horizon ablation table.
7. WC versus WCS feasible path-speed envelope.
8. Local speed versus curvature/height plot showing slow-down and recovery.
9. Offline versus DPPO safety/task/retention table.
10. Execution-horizon/denoising latency-quality Pareto plot.
11. Video with curves, sharp direction changes, a crouch interval and an open
    section where the timing target can benefit from the fast capability.

## Estimated remaining work

Assuming normal cluster availability:

- conditioning contract, tests and evaluator: `2-3 days`;
- A-WC screening and B-WC experiment: `4-7 days`;
- sprint extension and its one permitted fallback: `3-6 days`;
- DPPO pilot and confirmation seeds: `3-5 days`;
- locked evaluation, figures and writing: `4-7 days`.

A corrected WC result should be available in about one week. A simulation-only
paper package with a conclusive sprint result is realistically a `3-4 week`
target. Hardware validation requires a separate safety/calibration margin.

The existing `dppo_path_4.pt` result remains the protected fallback and can
support a WC-focused I2R demonstration while the sprint extension is completed.
