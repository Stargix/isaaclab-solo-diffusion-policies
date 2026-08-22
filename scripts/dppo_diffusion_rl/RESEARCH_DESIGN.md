# Phase B-DPPO: direct online correction of a path-conditioned Diffusion Policy

## Scoped research question

Can online policy-gradient fine-tuning correct the terminal, average-speed and
robustness failures of a path/pose-conditioned diffusion policy while retaining
the geometric and height behavior learned from small random-command datasets?

This branch tests direct DPPO. It does not test residual PPO and it does not add
a high-level velocity policy. The input remains the Phase-A history plus the
12-D geometric path/terminal-pose/average-speed condition. The output remains a
16-step joint-action trajectory, of which indices 8:12 are executed.

The implementation follows [DPPO (Ren et al., ICLR 2025)](https://openreview.net/forum?id=mEpqHvbD2h)
and its [official implementation](https://github.com/irom-princeton/dppo).

## Why this checkpoint, and its real limitation

The starting artifact is `checkpoints_iri/real_walk_crouch_hindsight.pt`. It is
the right checkpoint structurally: it already responds causally to path geometry
and height, interpolates intermediate heights, and exposes the requested average
speed in the same condition used at deployment.

It is not a solved base policy. The frozen capability audit recorded roughly:

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
clipped to `[-5, 2]` and averaged over the actually executed horizon and action
dimensions. Summing 48 likelihood terms would make the ratio exponentially
ill-conditioned. PPO epsilon increases from 0.001 at the first fine-tuned step
to 0.01 at the last step. A target approximate KL of 0.02 terminates the actor
epoch early.

GAE is computed once on the physical chunk sequence:

`delta_t = r_t + gamma V(c_(t+1)) (1-d_t) - V(c_t)`

`A_t = delta_t + gamma lambda (1-d_t) A_(t+1)`.

The same physical advantage supervises each selected reverse transition with
the inner discount `gamma_denoising^(K' - k - 1)`. The critic is updated once
per physical transition rather than once per denoising transition.

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

Let `s_t` be monotone arc-length progress, `e_perp` cross-track error,
`e_h` height error, `e_psi` tangent-yaw error and

`e_vavg = s_t / elapsed_t - v_desired`.

The startup value of `e_vavg` is neutral for 0.5 s. The step reward is

`3 Delta s - dt [1.25 H(e_perp/0.12) + 0.25 H(e_vavg/0.10) + 0.25 H(e_psi/0.40) + 0.75 H(e_h/0.04) + stability] + terminal`.

`H` is normalized Huber loss. `3 Delta s` is a potential difference and
telescopes over the route, so living longer does not manufacture reward. There
is no alive bonus, independent time penalty, instantaneous-speed target,
action-rate penalty, residual penalty, or external smoothing controller.

Terminal values are:

- route success: `+10`;
- ordinary timeout/corridor/overshoot failure: `-10`;
- base contact: an additional `-10`.

Success requires final position within 0.15 m, yaw within 0.40 rad, height
within 0.05 m and route-average speed within 0.08 m/s. Desired mean speed thus
defines timing without a separate deadline reward. Timeout remains a failed
episode, not a source of value bootstrapping.

Smooth height transitions are not hard-coded. The network receives future
height through its geometric goal and is rewarded for physical height tracking;
smoothness must remain an emergent property of the pretrained diffusion
manifold and its online refinement.

## What counts as a positive result

Checkpoint selection is not based on episodic return alone. The primary paired
comparison against the frozen Phase-A checkpoint is:

1. survival and base-contact rate;
2. finite-route success and terminal position error;
3. mean absolute average-speed error;
4. cross-track RMSE;
5. height error, including random walk/crouch/intermediate transitions.

DPPO is positive only if terminal success and mean-speed obedience improve
without a material regression in survival, path tracking or height causality.
The existing evaluator is used unchanged at the metric level and now rebuilds
the correct DPPO hybrid sampler.

## Deferred alternatives

- Full `K'=10` fine-tuning is an ablation, not the default, because it removes
  the frozen coarse prior.
- A reference-policy KL or BC auxiliary loss is deferred. Partial denoising
  fine-tuning plus PPO trust-region controls already tests canonical DPPO; an
  extra regularizer would change the hypothesis.
- Residual PPO remains a separate baseline. It has stronger action-level safety
  but cannot reshape the diffusion distribution itself.
- A velocity-policy plus high-level path planner remains the alternative if the
  direct path checkpoint is shown to be structurally unrepairable.
