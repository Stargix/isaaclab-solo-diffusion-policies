# Phase B-DPPO: direct online correction of a path-conditioned Diffusion Policy

## Scoped research question

Can online policy-gradient fine-tuning correct the terminal, average-speed and
robustness failures of a path/pose-conditioned diffusion policy while retaining
the geometric and height behavior learned from small random-command datasets?

This branch tests direct DPPO. It does not test residual PPO and it does not add
a high-level velocity policy. The input remains the Phase-A history plus the
checkpoint-declared geometric path/terminal-pose/average-speed condition. The
legacy actor uses 12 dimensions; the profile actor uses 16 dimensions by
attaching required height to all four spatial samples and adding the current
height requirement. The output remains a 16-step joint-action trajectory, of
which indices 8:12 are executed.

The implementation follows [DPPO (Ren et al., ICLR 2025)](https://openreview.net/forum?id=mEpqHvbD2h)
and its [official implementation](https://github.com/irom-princeton/dppo).

## Starting checkpoints and their real limitations

The profile16 experiment starts directly from
`checkpoints_iri/checkpoints_dp/wct_diffusion_policy_baseline.pt`. This is the
schema-8 Phase-A actor: DPPO selects the 16-D route contract from its checkpoint
metadata and does not transplant weights or introduce a skill index.

For comparison, the earlier scalar-height v2 experiment started from
`checkpoints_iri/checkpoints_dppo/dppo_path_1.pt`, itself fine-tuned from
`real_walk_crouch_hindsight.pt`. That was deliberately a warm start rather than
a return to Phase A: its evaluation already showed about
98.7% survival, 93.3% arrival and 0.042 m cross-track RMSE, so rewriting the
actor condition or discarding its online geometric correction would add risk
without addressing the observed failure.

Its limitation is specific and measurable. At requested speeds 0.2 and 0.4
m/s it commonly reached the endpoint at the pretrained gait speed and waited
until `L/t` entered the success tolerance. The old terminal predicate allowed
this because it checked for joint satisfaction at every later step. Contract
v2 preserves the actor but makes first entry into the goal region absorbing;
critic and optimizer state are reset because they estimate the old return.

The original frozen Phase-A capability audit recorded roughly:

- 0.807 overall survival;
- 0.106 m cross-track RMSE;
- 0.030 m mean height error;
- 0.884 mean absolute speed-ratio error;
- 0.030 finite-route success.

Therefore DPPO is scientifically justified but not guaranteed to succeed. It
tests whether online, on-manifold exploration can repair a partly competent
conditional policy. If it cannot improve survival and terminal success without
destroying path/height tracking, the conclusion is that the Phase-A support is
too weak and must be repaired before further online optimization.

There is one deliberate observability limitation. The Phase-A goal stores the
average speed of the local lookahead segment. It equals the route command away
from the endpoint but tapers when that lookahead is clipped by the route end.
The asymmetric critic additionally sees route speed and elapsed fraction; the
actor does not. Changing the actor input would invalidate the pretrained
checkpoint contract, so this iteration does not hide that change inside DPPO.
If terminal timing alone remains unlearnable, an explicit speed/time adapter is
the next architecture ablation rather than another reward rewrite.

The profile16 extension removes a separate observability limitation found in
the height-transition audit. A single terminal height aliases routes that end
at the same posture but require different posture along the lookahead. Schema 8
therefore supplies `[x,y,h_required]` at 25/50/75/100% of the spatial horizon
plus `h_now`. This is task geometry, not a skill label: no gait identity or
expert index is exposed. DPPO replaces only the final average-speed scalar with
the remaining-route pace budget, exactly as for schema 7.

## Relationship to adjacent work

[Skill-Nav](https://arxiv.org/abs/2506.21853) supports the broader hierarchy:
a navigation policy can compose pretrained skills instead of relearning motor
control end to end. This branch asks a narrower prior question: can the existing
path-conditioned diffusion distribution itself be corrected online? A
high-level skill/velocity policy remains a separate comparison and is not mixed
into the DPPO result.

[NVIDIA Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) is useful
for representing sparse trajectory and pose constraints, but its two-stage
root/body model generates kinematic motion from a large motion dataset. Its
[constraint mechanism](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html)
overwrites selected motion components and provides masks. Solo12 here requires
closed-loop physical balance and joint actions; copying Kimodo's output or hard
constraint overwrite would change the problem and could force the smooth
height transition that is meant to be learned. We retain only the general idea
of explicit path/terminal constraints in the condition and evaluation.

## Mathematical contract

For physical condition `c_t`, the DDPM produces a chain

`x_t^K -> x_t^(K-1) -> ... -> x_t^0`,

with reverse transitions

`p_theta(x^(k-1) | x^k, c_t) = Normal(mu_theta(x^k, c_t, k), sigma_k^2 I)`.

The initial noise distribution is independent of the actor and is not included
in the policy ratio. The environment sees only the denormalized executable slice
of `x^0`. The PPO ratio is evaluated on stored reverse transitions:

`rho = exp(log p_theta(x^(k-1)|x^k,c) - log p_old(x^(k-1)|x^k,c))`.

As in the official DPPO implementation, per-coordinate log probabilities are
clipped to `[-5, 2]` before reduction; this avoids exponentially ill-conditioned
chunk ratios. The denoiser is a causal transformer, so the likelihood support
cannot always be only the executed slice. For every non-final fine-tuned reverse
transition it includes tokens `0:(execution_offset + exec_horizon)`: generated
prefix tokens can affect the later executed tokens. On the final transition,
the newly sampled prefix is never consumed and only the executed slice is used.
Both cases share the same denominator, so the final reverse transition is not
artificially upweighted. Tokens after the executed chunk are excluded because
causality makes them reward-irrelevant. PPO epsilon increases from 0.001 at the
first fine-tuned step to 0.01 at the last step. A target approximate KL of 0.02
terminates the actor epoch early.

GAE is computed once on the physical chunk sequence:

`delta_t = r_t + gamma V(c_(t+1)) (1-d_t) - V(c_t)`

`A_t = delta_t + gamma lambda (1-d_t) A_(t+1)`.

The same physical advantage supervises each selected reverse transition with
the inner discount `gamma_denoising^(K' - k - 1)`. The critic is updated once
per physical transition rather than once per denoising transition.

Here `gamma = 1`. The finite task uses potential differences, so this preserves
their exact endpoint semantics. With `gamma < 1`, the shaping would need the
form `gamma Phi(s') - Phi(s)`; discounting the current differences directly
would introduce an unintended occupancy/pace preference, following the policy-
invariance condition of [Ng, Harada and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
`lambda = 0.95` still
controls the GAE bias/variance tradeoff. Symmetric advantage quantile clipping
is disabled because a 1% clip can erase every rare success or fall in a large
batch. Value clipping is optional and disabled by default, matching the
official DPPO configuration; it is always disabled during critic-only warm-up
so distant terminal targets retain a gradient.

The critic receives ten warm-up iterations before the first actor update. At
32 chunks per iteration this spans approximately one 24 s route, so its initial
targets include real timeouts and failures rather than only short prefixes.
This is conservative for a transformer actor and avoids using an arbitrary
initial value baseline to move the pretrained policy.

## Frozen and trainable parts

The pretrained denoiser is copied twice:

- the first `K-K' = 5` reverse steps use an immutable frozen copy;
- the last `K' = 5` steps use the trainable actor initialized identically.

Although the transformer was behavior-cloned with attention dropout 0.3, both
copies stay in evaluation mode during rollout and optimization. Gradients still
flow through the trainable copy. Enabling dropout would introduce a latent
random variable absent from the transition likelihood and invalidate PPO's
importance ratio.

Training raises the reverse standard deviation to at least 0.10, including the
last transition, to give every optimized transition a non-degenerate density
and structured exploration. Evaluation removes that exploration floor and uses
the checkpoint's original DDPM posterior variance. The environment-bound
normalized action is clipped to `[-1,1]`; the stored stochastic chain is not
clipped, so its likelihood remains well-defined.

## Reward and termination

Let `s_t` be monotone arc-length progress and let
`B(e;a) = 1 - exp(-Huber(|e|/a))`, a smooth cost bounded in `[0,1]`. The dense
reward is

`3 Delta s + Delta[-|min(s,L)-v*t|] + Delta Phi_pose`

`- Delta s [1.25 B(e_perp;0.12) + 0.25 B(e_psi;0.40) + w_h B(e_h;0.04) + stability]`,

where `w_h=2.0` for profile16 contract v4 and remains 0.75 for the legacy
12-D contract.

The schedule potential is not capped in time: arriving either early or late
leaves `|L-v*t|>0`. The episode terminates at the first entry into the goal
region, so an early arrival cannot wait until `t=L/v` and retroactively satisfy
the average-speed condition. `Phi_pose <= 0` is a bounded final-position/yaw/
height potential. Its gate turns on only when the endpoint is inside the same
two-second geometric lookahead visible to the actor. This gives dense terminal
credit without rewarding an unobservable global target.

Tracking costs are integrated per metre of forward progress, not per second.
Standing still therefore cannot accumulate an unbounded penalty, and falling
early cannot improve return by avoiding future occupancy costs. There is no
alive bonus, independent clock penalty, instantaneous-speed target, action-rate
penalty, residual penalty, or external smoothing controller.

Terminal outcomes are:

- route success: `+10`;
- arrival with invalid yaw, height or mean speed: `-10`;
- timeout: `-10`;
- corridor/overshoot: a route- and horizon-dependent penalty larger than the
  maximum positive dense progress return, plus a safety margin;
- base contact: that hard-failure penalty plus `-10`.

Success requires first arrival within 0.15 m, yaw within 0.40 rad, height
within 0.05 m and `|L/t_arrival-v_des| <= 0.08 m/s`. Desired mean speed thus
defines timing without a separate deadline or instantaneous-speed reward.
For profile16 it additionally requires the full-route height MAE defined below
to be at most 0.04 m. Timeout remains a failed episode, not a source of value
bootstrapping.

Smooth height transitions are not hard-coded. The network receives future
height through its geometric goal and is rewarded against the height required
at its current route progress. Smoothness must remain an emergent property of
the pretrained diffusion manifold and its online refinement.

## Task contract v4: sustained height-profile success

Job 3471 exposed a mismatch between the stated task and task contract v3.  Its
iteration-100 actor substantially repaired the unstable Phase-A profile actor:
the matched long-horizon audit reached about 81.5% survival, and a separate
mixed-profile audit reached 90%.  It also retained a usable high posture on
constant-height trials (approximately 0.269 m achieved for 0.2932 m requested).
The actor therefore contains both postures and is a valid online warm start.
However, on held-out binary profiles it obtained 0.055 m distance-weighted
height MAE and spent only about 49% of travelled distance within 0.03 m of the
requirement.  Qualitatively, it often descended and then remained crouched.

This was an objective-specification failure, not evidence that DPPO could not
represent the transition.  Contract v3 required only terminal height for
success, and checkpoint selection did not include sustained height.  Moreover,
the old dense height cost had weight 0.75 and was bounded per metre.  On a 4 m
route its absolute upper bound was only 3; remaining crouched through a typical
half-high profile cost roughly 1.4, much less than the progress and terminal
success return.  A stable crouch followed by a late terminal correction was
therefore rational under the implemented objective.

Contract v4 keeps the actor input, action output, route distribution, average-
speed potential and DPPO likelihood unchanged.  For each episode it accumulates

`E_h = sum_t |h_t-h*_t| Delta s_t / sum_t Delta s_t`

and

`C_h = sum_t 1[|h_t-h*_t| <= 0.04] Delta s_t / sum_t Delta s_t`.

Both are weighted by monotone route progress rather than time. Waiting cannot
reduce the error and early failure cannot avoid a per-second occupancy cost.
For `hindsight_geom_profile16`, first-arrival success now additionally requires
`E_h <= 0.04 m`; terminal-height tolerance remains a separate 0.05 m condition.
The dense height weight is 2.0 for profile16 only. Its full-route cost remains
bounded by 8 on a 4 m route, below the explicit hard-failure cost, while an
always-crouched half-high route costs roughly 3.7 before its failed-arrival
penalty. Legacy `hindsight_geom_avg12` training retains weight 0.75 and does not
activate the profile gate.

`Episode/profile_height_mae_m`,
`Episode/profile_height_within_tolerance_fraction`, and
`Episode/profile_height_success_rate` are logged. Joint success remains the
primary checkpoint criterion; normalized profile MAE contributes at most 0.2
as a bounded tiebreaker. Timeout rate is now also logged and penalized in the
selection score, matching its explicit terminal reward and preventing a
no-progress policy from looking preferable to a near-complete profile attempt.
This prevents `best.pt` from selecting a visibly collapsed or stationary
strategy when task outcomes are otherwise similar.

No target ramp, height controller, gait label, gait-specific reward or action-
rate term is introduced. The future height samples remain observations, and
the diffusion actor must learn when and how to transition smoothly. Falls and
corridor exits remain hard terminations; profile error is evaluated over the
route rather than terminating valid transition attempts. This separation is
consistent with the constraint-as-termination treatment of physical safety in
[CaT](https://arxiv.org/abs/2403.18765), while retaining the potential-based
shaping invariance described above for task progress.

The preregistered repair run starts from Job 3471 `model_100.pt`, resets critic
and Adam state, anchors reference KL to that stable actor with coefficient
0.05, and trains only stage 1 for at most 150 iterations. The value 0.05 and all
other optimizer settings are retained from the successful path-v3/v4 runs;
this is not a hyperparameter sweep. Checkpoints are saved every 10 iterations.
Sprint allocation remains deferred until the held-out gate reaches: survival
at least 90%, base-contact at most 5%, joint success at least 75%, profile MAE
at most 0.04 m, speed MAE at most 0.05 m/s, and cross-track RMSE at most 0.10 m.

## What counts as a positive result

Checkpoint selection is not based on episodic return alone. The primary paired
comparison against the frozen Phase-A checkpoint is:

1. survival and base-contact rate;
2. joint task success and terminal position/pose error;
3. strict endpoint arrival rate and mean-speed error up to arrival;
4. cross-track RMSE;
5. height error, including random walk/crouch/intermediate transitions.

DPPO is positive only if terminal success and mean-speed obedience improve
without a material regression in survival, path tracking or height causality.
The evaluator rebuilds the correct DPPO hybrid sampler. It reports full-horizon
displacement separately from finite-route arrival timing, never includes
IsaacLab's post-termination auto-reset sample, and evaluates route-required
height separately from the future height preview sent to the actor.

## Deferred alternatives

- Full `K'=10` fine-tuning is an ablation, not the default, because it removes
  the frozen coarse prior.
- Canonical DPPO without a reference loss remains the v1/v2 ablation. The
  paired evaluation showed that its small per-update PPO KL still accumulated
  a large gait drift while correcting speed. The v3 primary run therefore uses
  the exact conditional Gaussian KL between the trainable reverse kernel and
  an immutable `path_1` actor at the same diffusion state. This tests whether
  online refinement can correct timing without leaving the stable locomotion
  manifold; it does not modify the task reward.
- Residual PPO remains a separate baseline. It has stronger action-level safety
  but cannot reshape the diffusion distribution itself.
- A velocity-policy plus high-level path planner remains the alternative if the
  direct path checkpoint is shown to be structurally unrepairable.

One known estimator limitation is retained for scope: when an environment ends
inside a four-action chunk, the stored denoising likelihood still covers the
configured chunk. Post-terminal actions and rewards are masked and the new
episode fragment is discarded at the boundary, but shortening likelihood
support per environment would require a separate variable-horizon DPPO
derivation. The official implementation also uses a fixed reward horizon; this
is documented rather than patched with an unverified mask.
