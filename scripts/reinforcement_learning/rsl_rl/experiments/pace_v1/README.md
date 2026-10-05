# SOLO12 pace expert v1

This experiment trains a clock-free ipsilateral pace velocity expert as a
candidate source of offline demonstrations. It does not modify the walk,
crouch, DPPO, or paper checkpoints. The actor uses the same 48-observation,
12-action MLP contract as `checkpoints/walk_final.pt`; only actor and actor
normalizer are warm-started. The critic, PPO optimizer, and exploration noise
are initialized fresh.

The intended demonstration library uses pace for slow motion and reuses the
existing walk for fast motion. Expert training keeps commands vx in [-1, 1]
m/s; the pace demonstration speeds will be restricted later to the validated
slow region. The training range is not the intended range of collected pace
demonstrations.

Pace is defined from ordered foot contacts FL, FR, RL, RR. The synchronized
pairs are FL-RL and FR-RR; the opposite side must be out of phase. A bounded,
clock-free timer reward adapted from Isaac Lab's Spot `GaitReward` is combined
additively with velocity, height, and stability rewards. The extra topology
and long-contact guard are engineering choices for this expert and must be
judged from measured contacts and tracking; their coefficients are not claimed
to be literature-optimal.

## Before training

1. On local Windows, run the pure unit tests and the 2-environment, 1-iteration
   Isaac smoke test shown in `PACE_EXPERT_IMPLEMENTATION_HANDOFF_2026-10-06.md`.
   The Slurm launcher uses an 8-environment, 2-iteration smoke test first.
2. Confirm the exact warm-start SHA256 in the launcher matches the local
   `checkpoints/walk_final.pt` file on the cluster.
3. Inspect the saved env/agent YAML to confirm the command ranges, 48D actor,
   no domain randomization, and quadruped symmetry are active.
4. Run the Slurm script. It runs the smoke test first, then the 2500-iteration
   training job only when the smoke test succeeds.

## Evaluation gate

Use `evaluate_gait_comparison.py --comparison walk_pace` with the same flat
environment for both actors. The evaluator writes JSON, compressed time series,
and a contact/velocity plot. Evaluate forward commands 0.4, 0.7, and 1.0 m/s;
stop; reverse -0.7 m/s; and mirrored lateral/yaw commands. Use distinct output
directories. The 0.4 and 0.7 m/s trials are the primary acquisition gate; 1.0
m/s is a boundary test. Interpret contact raster, ipsilateral-versus-diagonal
pattern rates, realized velocity error, height, and survival together. A high
requested speed alone does not establish a pace gait.

Engineering gate from the handoff: at 0.4 and 0.7 m/s, at least 95% survival,
forward RMSE <= 0.10 m/s, mean-height bias <= 0.025 m, ipsilateral contact
Jaccard at least 0.20 above diagonal Jaccard, and >=3 touchdowns per foot in
8 seconds. The sample size is 16 per actor, so 95% means all 16 survive. Report
the actual denominator and the entire contact raster. Failure at 1.0 m/s alone
limits the demonstrated support; it is not grounds to silently change rewards.

## Output and provenance

RSL-RL writes under
`logs/rsl_rl/solo12_rsl_rl_pace_runs/<timestamp>_pace_v1_seed42_job<jobid>/`.
The Slurm log records Git status/revision and checkpoint digest. Keep run output,
checkpoints, and evaluation archives out of Git. Do not collect pace data or
start a WC+pace Phase A until this gate has been reviewed.

See `scripts/diffusion_policy/experiments/wc_wct_ablation_v1/PACE_EXPERT_IMPLEMENTATION_HANDOFF_2026-10-06.md` for equations, tests, commands, and the follow-on plan.
