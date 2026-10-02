# Gaussian PPO seed-42 training audit and second-pilot decision

## Evidence

Local source: `scratch/gaussian_ppo_seed42_metrics.jsonl`, 150 rows, iterations
0--149. Selected pre-update actor is iteration 147, SHA256
`ae6b5794221c2e96999ee0575d6e920c4c1437ff11a6df8949446a91af6ae38e`.
Logged iteration time totals 27.05 minutes; this is not total Slurm wall time.
The full nominal physics budget is 78,643,200 steps.

- Iterations 0--9: critic-only warmup, no actor updates.
- Every one of the 140 subsequent iterations: one actor update and KL stop.
- Possible actor passes: 140 x 80 = 11,200; actual: 140 (1.25%). The selected
  pre-update iter147 contains 137 preceding actor steps, consistent with Adam.
- Observed minibatch KL: minimum .6260, median 2.2227, maximum 30.1333, against
  target .02. These are approximately 31--1507 times that target.
- Actor pre-clipping gradient norm: 17.54--52.30, median 31.55; clipping was
  active, but Adam step size still caused large joint-distribution changes.
- Learned mean std remained .039994--.040012, close to initialization .04.
- Logged accepted-only approximate KL was zero in every actor-enabled row.
  The first likelihood recomputation is on-policy and yields zero; subsequent
  oversized changes were excluded from that average. This is a telemetry
  defect, not zero gradient or frozen weights. The max-KL log already exposed it.

## Behaviour over training

| Iteration | Train success | Base contact | Profile height MAE | Mean speed error | Episode return |
|---|---:|---:|---:|---:|---:|
| 10 | 0.39% | 43.27% | 1.63 cm | .221 m/s | -19.41 |
| 30 | 1.87% | 25.86% | 2.09 cm | .166 m/s | -18.24 |
| 50 | 5.39% | 55.89% | 3.80 cm | .255 m/s | -29.61 |
| 75 | 3.51% | 1.23% | 5.63 cm | .193 m/s | -5.05 |
| 100 | 0.06% | .19% | 4.68 cm | .298 m/s | -3.45 |
| 147 | 16.23% | .16% | 6.76 cm | .142 m/s | -.19 |
| 149 | 12.89% | 0% | 6.62 cm | .152 m/s | -1.50 |

Train success and return peak at 147. Improving survival explains much of the
return improvement, not mastery of the conditioning. Early height averages
are partly censored by short failed episodes; do not interpret them as direct
proof that a competent full-route posture controller was subsequently lost.
The frozen evaluation independently establishes posture collapse: actual
height about .17 m even when walk requests .2932 m. ID transition success is
0%, fast success 0%, ID constant success 9.42%, despite >99% survival.

## Interpretation

The single-update/KL-stop hypothesis is confirmed, not merely suggested by
Adam counters. A fixed actor LR 1e-4 with joint 48-coordinate Gaussian density
and std .04 was an unsuitable optimization recipe for this prior. Correct
summed likelihood must not be changed to an average to conceal its KL scale.

The reward is not proved optimal, and this does not prove that optimization is
the sole cause of failure. However, modifying reward or weakening height/pace
gates now would confound a comparison whose current optimization is plainly
poorly calibrated. The functioning WC-DPPO arm shows the task is solvable with
the existing contract. This pilot cannot prove generic PPO incapacity or
diffusion necessity. It is pretrained BC + PPO, not RL from scratch.

## Minimal second pilot

Restart from the same pure deterministic BC checkpoint, with fresh critic and
optimizers. Keep the environment, reward, route generator, joint likelihood,
exploration distribution, minibatches, clipping and nominal interaction
budget unchanged. Change actor LR to 1e-5 and enable the documented bounded
KL scheduler. This scheduler is optional, preserving the old fixed-LR recipe.

The next-iteration LR reacts to observed stopping KL and post-update analytic
Gaussian KL on a deterministic rollout subset. Factor 1.5 and half/twice-target
bands follow the general adaptive-KL pattern in official RSL-RL PPO; this
implementation retains early stopping and uses a lower LR ceiling because
the audited pilot is a pretrained chunk controller, not fresh per-step PPO.
It does not roll back accepted steps or enforce a hard KL constraint.

Sources:
- [Official RSL-RL PPO](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py): KL-driven adaptive learning rate.
- [DPPO](https://arxiv.org/html/2409.00588v3), Appendices E.7/E.8, Table A11: pretrained Gaussian baselines and task-specific tuning; a .04 exploration example uses 1e-5 actor LR, not this failed pilot's 1e-4. Manipulation settings are motivation, not optimal locomotion parameters.

## Acceptance and publication limits

The next pilot is meaningful if measured distribution change is controlled
and the actor is no longer systematically restricted to one oversized step.
Many updates alone are not success. Evaluate complete route/posture/pace
compliance on the unchanged frozen bank; retain poor cases and initialization
failures. If this also fails with competent optimization, report that result
with the declared tuning effort and one-seed limitation. Do not repeatedly
retune against individual held-out failures or silently discard this pilot.

Only the second seed-42 pilot is planned here. Replication and any new
DPPO/gait experiment remain separate decisions.

The new run is `ppo_wc_supported_hybrid_v3_kl_adaptive_v2_seed42`; the original
run, checkpoint, datasets, and WC/WCT-DPPO implementations remain unchanged.

## Local validation of this patch

- 110 tests passed: 22 Gaussian-PPO cases and the complete existing DPPO suite.
- New tests cover rejected-minibatch KL telemetry, adaptive LR bounds,
  no adaptation during critic warmup, and analytic joint KL over all 48
  executable coordinates (not an incorrectly averaged action density).
- Bash syntax and scoped Git whitespace checks passed.
- A synthetic CUDA update using the actual pinned BC checkpoint produced
  finite parameters, actor/critic updates, and lowered LR after a large KL.
  This is a mechanical check with artificial observations/advantages, not
  simulator evidence that 1e-5 is calibrated or that the task will converge.
- No local simulation training or frozen re-evaluation was run. The cluster
  launcher retains its real simulator smoke before the new full pilot.
