# Final clean-start experiment: supported procedural DPPO

## Question

Can online DPPO refine a data-poor, multi-expert diffusion policy into a
single path/pose/time policy without a skill index, gait reward, reference
trajectory or high-level velocity controller?

This is the definitive primary experiment. It starts directly from the pure
Phase-A imitation checkpoint trained on random expert commands. No previous
DPPO actor, critic or optimizer state is reused.

## What previous runs got wrong

The Phase-A dataset contains constant demonstrated posture endpoints:
walk at 0.2932 m, crouch at 0.1705 m and the fast expert near 0.28 m. It does
not contain 0.25/0.21 m demonstrations or within-episode height switches. The
profile16 actor therefore saw only constant vectors `[h,h,h,h,h]` offline.
The old Stage-2 task simultaneously introduced unseen mixed profiles,
intermediate heights, harder geometry and faster commands. Restarting from a
DPPO actor already adapted to that confounded task could not answer the
original research question.

The replacement changes the online task distribution, not the policy
architecture or DPPO mathematics:

- every reset samples a fresh smooth curvature field; there is no finite list
  of polygons to memorize;
- posture classes are balanced: constant walk, constant crouch,
  walk-to-crouch and crouch-to-walk;
- only the two well-separated demonstrated posture endpoints are commanded;
  intermediate measured heights remain an outcome of learned transitions;
- transition position is uniform in 1.6--2.4 m on a 4 m route;
- the 0.25 m interval on each side of a discontinuous requirement is omitted
  from the plateau-height score, because no physical policy can realize a
  discontinuous body-height change; actions are never blended or overridden;
- requested route-average speed is sampled continuously inside a support-aware
  envelope: crouched sections <=0.4 m/s, curved high sections <=0.6 m/s and
  nearly straight high sections up to the run cap of 0.8 m/s. Mixed-route
  limits use the harmonic mean of their section caps.

Thus the goal remains task-level `[path preview with required height, terminal
yaw, average speed]`. There is no skill ID and no reward for diagonal legs,
flying phases or selecting the fast expert. If the diffusion model exploits
its fast manifold on safe high/straight sections, that behavior emerges only
because it improves route completion and average timing.

## Reward contract

The existing bounded, finite-route reward is retained. Positive dense reward
is forward route progress. Cross-track, tangent-yaw, height and stability
costs are paid per metre rather than per second. Timing is the potential
difference of `-|progress - desired_speed * time|`; there is no second clock
objective and waiting cannot improve it. Terminal success jointly requires
position, yaw, terminal height, route-average speed and distance-weighted
plateau-height MAE. Falls, corridor exits and terminal overshoot remain hard
failures.

The profile-height weight is restored to 0.75 instead of changing the task and
also increasing it to 2.0. At worst it costs 0.75 per travelled metre, while
progress contributes 3.0 per metre. This removes the observed incentive to
sacrifice locomotion for height without inventing a new reward term.

## Optimizer decision

The primary run uses the successful clean-start DPPO settings: actor LR 1e-5,
five trainable late denoising steps, PPO clipping and no reference-policy KL.
Reference KL was not part of the original successful clean-start result and
the 0.05 continuation failed to improve arrival. It remains a later ablation,
not an extra assumption in the primary run. Training is capped at 150
iterations because the project repeatedly learned by roughly 50--140 and then
drifted. `best.pt` uses the joint task score and checkpoints are saved every
10 iterations.

## Run

From the cluster repository root:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_procedural_v1/train_cluster.sbs
```

An explicit Phase-A path can be passed as the first argument. The launcher
uses the original run checkpoint first and the copied checkpoint only as a
fallback. `--require_phase_a_source` aborts if either file is actually DPPO.

## Held-out evaluation

Select `best.pt`, but report it against the untouched Phase-A checkpoint with
the same seeds. Use at least 50 independent procedural route draws and both
transition directions. Also retain straight, S, circle, right-angle and random
polyline families as out-of-family stress tests. Report joint success plus its
separate position, yaw, terminal-height, average-speed and plateau-height
gates, survival/base contact, cross-track RMSE and results stratified by
constant/transition profile and requested speed.

The primary claim is supported only if DPPO improves joint task success over
Phase A without materially reducing survival. Fast-manifold use is a secondary
analysis based on kinematics/action similarity to the expert datasets, never a
training label or success condition.

Canonical held-out transition command (run once for `best.pt` and once for the
Phase-A checkpoint, changing checkpoint and output directory only):

```bash
./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_wct_supported_procedural_v1/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_wct_supported_procedural_v1_transitions --speeds 0.2 0.35 0.45 --path_shapes procedural --route_length_m 4.0 --transition_fractions 0.4 0.5 0.6 --transition_directions both --profile_transition_margin_m 0.25 --repeats 50 --duration_s 24.0 --seed 142 --save_timeseries --headless --require_empty_output_dir
```

Use seed 142 rather than the train seed 42. Constant walk and crouch are
evaluated separately so unsupported speed/posture cross-products are not
silently included:

```bash
./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_wct_supported_procedural_v1/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_wct_supported_procedural_v1_walk --speeds 0.2 0.4 0.6 0.8 --path_shapes procedural --path_height 0.2932 --route_length_m 4.0 --repeats 50 --duration_s 24.0 --seed 142 --headless --require_empty_output_dir
./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_wct_supported_procedural_v1/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_wct_supported_procedural_v1_crouch --speeds 0.2 0.3 0.4 --path_shapes procedural --path_height 0.1705 --route_length_m 4.0 --repeats 50 --duration_s 24.0 --seed 142 --headless --require_empty_output_dir
```

## Literature basis

- [DPPO](https://arxiv.org/abs/2409.00588): PPO fine-tuning of the last reverse
  diffusion steps and a non-zero denoising noise floor.
- [DiffuseLoco](https://arxiv.org/abs/2404.19264): source-agnostic tuples permit
  reuse and composition of heterogeneous locomotion experts.
- [Skill-Nav](https://arxiv.org/abs/2506.21853): randomized waypoint tasks and
  irregular held-out routes rather than memorizing one fixed path.
- [Path-conditioned RL](https://arxiv.org/abs/2603.13888): procedural route
  generation and perturbation provide broad path coverage during training.

Those works motivate the algorithm and task sampling. They do not determine
the numerical reward weights; those are fixed from the project's stable
finite-route formulation and disclosed above rather than presented as
universal constants.
