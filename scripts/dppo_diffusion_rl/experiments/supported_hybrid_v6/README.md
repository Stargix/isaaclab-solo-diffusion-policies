# Final consolidation: `supported_hybrid_v6`

## Decision

V5 proved the important mechanism: without a skill id or local speed target,
the policy allocated more speed to the upright section and less to crouch, and
substantially improved fast crouch-to-walk.  It was not Pareto-safe:

| Paired benchmark | v3 | v5 | Change |
|---|---:|---:|---:|
| ID constant success | 100.0% | 95.3% | -4.8 pp |
| ID transition success | 99.8% | 95.1% | -4.7 pp |
| fast C->W success | 71.0% | 85.2% | +14.2 pp |
| fast W->C success | 78.4% | 77.0% | -1.4 pp |
| OOD fast C->W success | 75.1% | 88.2% | +13.1 pp |
| OOD fast W->C success | 81.6% | 70.9% | -10.7 pp |

The raw paired traces locate the regressions:

1. V5 arrived systematically too fast on ordinary routes (roughly +3--6
   cm/s signed mean-speed error instead of v3's +0--2 cm/s).
2. At 1.0 m/s W->C, the robot could attain crouch height after the boundary,
   but lowered early: pre-boundary upright MAE was about 4 cm while
   post-boundary crouch MAE remained about 1.5 cm.
3. Survival remained 97.7--100%, so neither issue is a fall-reward failure.

Therefore v6 does **not** change rewards, actor inputs, preview geometry,
success tolerances, or feasibility rules.  It is a conservative second DPPO
stage from the exact benchmarked v3 actor.

## Two minimal corrections

1. **Speed calibration anchors.** The v5 fast set remains 25% of resets, but
   each transition direction now contains 25% targets in 0.65--0.70 m/s and
   75% at the feasible upper frontier.  V5 trained almost all fast examples
   near the upper frontier, despite the benchmark explicitly testing 0.65.
2. **Replay-only reference KL.** The frozen v3 actor regularizes only
   `coverage_class == 0`, the exact v3 replay half.  Fast and repeated-profile
   samples have no reference penalty and can improve.  The class is private
   optimizer metadata; it is not an observation, reward, gait label, or skill
   index.

This follows PPO's proximal-update principle and DPPO's reverse-transition
likelihood formulation, while using policy distillation only where retention
is required.  Relevant primary sources:

- Schulman et al., [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
- Ren et al., [Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588).
- Kaplanis et al., [Policy Consolidation for Continual RL](https://proceedings.mlr.press/v97/kaplanis19a.html).
- Li and Hoiem, [Learning without Forgetting](https://arxiv.org/abs/1606.09282).

The KL coefficient is 0.01: five times below the earlier globally anchored
0.05 experiment, and active on only half of samples.  The actor LR is halved
to `5e-6` because this is consolidation of a proven actor, not acquisition
from Phase A.  No other PPO hyperparameter changes.

## Source checkpoint contract

The only admitted source is the exact v3 checkpoint used by the paired final
benchmark:

```text
SHA256 5258FCF239A391BCFE6E4A951944B037B94441113D137B782AD5ADE8341FA55F
scripts/dppo_diffusion_rl/runs/dppo_wct_supported_hybrid_v3/best.pt
```

The launcher verifies this hash and aborts on any other actor.  Critic and
Adam are restarted; the loaded actor becomes the immutable KL reference.

## Train

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v6/train_cluster.sbs
```

An explicit copy of the exact checkpoint may be passed as argument 1.

## Preregistered decision rule

Evaluate `best.pt` with the unchanged `wct_final_evaluation_v1` frozen banks.
V6 is the final actor only if, relative to paired v3 routes:

- ID constant and ID transition lose no more than 1 percentage point;
- fast C->W and fast W->C both improve, including at 0.65 and 0.85 m/s;
- OOD W->C no longer shows v5's regression;
- survival remains at least 97%;
- CTE RMSE <= 0.10 m and profile-height MAE <= 0.04 m.

If these gates fail, do not tune the reward again.  Report v3 as the balanced
main model and v5 as the temporal-expansion ablation; that is the honest
scientific conclusion supported by the current evidence.

