# Data-reuse comparison

This suite prepares three connected development experiments. It does not change rewards, expert weights, task inputs, control intervals, or the final holdout. Run from the repository root inside the installed Isaac Lab environment, on an allocated GPU node. It does not submit scheduler jobs or assume an account/partition.

## Experiments

1. Existing H42/H43/H44 and WC42 on the same core routes and conditions. Keep the historical 25-step stand-hold plus 50-step policy posture preparation. Save full traces and report full active CTE, initial distance to the route and CTE after 0.5 m progress. The latter is a diagnostic window conditional on reaching that progress, not a replacement success metric. Optional `startup_sensitivity: true` adds separately labelled stage-zero core jobs; this also changes warmup conditioning and is not an isolated staging intervention. Default false avoids another diagnostic sweep.
2. Three online seeds for each of two Gaussian PPO initializations: BC mean weights versus random mean weights. Both retain exactly the source architecture and dataset-derived normalization/action ranges; exploration, critic, optimizer, warmup and interaction budget are shared. This estimates the effect of learned BC weights conditional on those statistics, not learning without demonstration information and not the causal effect of hindsight versus another labelling scheme. The random arm receives the constructor initialization without loading BC weights. Subsequent RNG state is identical for a paired seed before environment/critic construction.
3. All actors on repeated posture profiles, both starting heights, four changes every 0.8 m on open 4 m routes. Native heights 0.2932/0.1705; requested means 0.35/0.65/0.85. Report height error both including transitions and excluding the fixed +/-0.25 m transition margin. This tests posture composition, not contact-pattern/gait retention.

Core: 14 geometries (two draws from each of seven existing families), 0.25/0.50/0.85 m/s, two constant postures and both single-change directions: 168 conditions per actor. Composition: a separate 14-geometry bank, three speeds, two starting heights: 84 per actor. Geometry, not each crossed condition, is the independent route unit. Every process has at most 36 conditions. These are development banks, not an enlarged final test. Default ten actors imply 2520 evaluated conditions in 120 evaluator processes, plus six Gaussian training runs. No hierarchy retraining is scheduled.

Gaussian runs: 4096 environments x 150 iterations x 32 chunks x 4 control steps = 78,643,200 transitions per arm, including ten critic-only warmup iterations. Evaluate `last.pt` for all newly trained arms at the fixed budget; do not select epochs using this evaluation. Existing H/WC checkpoints retain their historical selection and are labelled separately; the causal initialization comparison is within the two newly trained Gaussian arms. Both arms use one fixed BC normalizer/prior source, not three independently collected datasets. Learning curves use the same interaction axis. A failure of this particular random-weight recipe does not establish that RL from scratch cannot solve the task; no scratch-specific tuning is included. The small std/learning-rate recipe originated in BC refinement and may be conservative for random learning.

## Cluster commands

Required artifacts: the existing deterministic BC prior, WC42 task/reference actor, walk/crouch experts and three hierarchy `best.pt` files. The prior must be pure `deterministic_chunk_bc`, not a refined Gaussian checkpoint. Known deterministic prior SHA256: `c874b2ad0a44cfaa69b0400ec354ec17fd3cad6d303923e1f55a91cc52305352`. These weights are distributed separately; the launcher never downloads or fabricates them.

```bash
export ARTIFACT_ROOT=/absolute/path/to/artifacts
export HIERARCHY_RUN_ROOT=/absolute/path/to/hierarchy_v2_runs
export PYTHONPATH="$PWD:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"

# Create once, after checkout/install. This hashes sources/inputs and materializes banks; no GPU used.
python -m scripts.research_comparison.suite prepare \
  --config scripts/research_comparison/example.json \
  --output scratch/data_reuse_comparison

# Run sequentially inside an allocation. This is six full trains, not a smoke.
python -m scripts.research_comparison.suite run \
  --protocol scratch/data_reuse_comparison --group train

# After trains, evaluate existing and newly trained actors with identical banks.
python -m scripts.research_comparison.suite run \
  --protocol scratch/data_reuse_comparison --group eval

# Refuses missing jobs, checksum changes or unmatched conditions; writes analysis.json.
python -m scripts.research_comparison.analyze --protocol scratch/data_reuse_comparison
```

For one train/allocation, append `--job gaussian_bc_42` or `--job gaussian_random_42` (seeds 43/44 likewise). Exact evaluator job IDs/arguments are in `protocol.json`. No parallel jobs against the same protocol directory: the runner holds an exclusive `state/RUNNING` lock and allows only one local simulator process. Six train jobs can be scheduled separately but must not run simultaneously under this sequential runner. Prepare separate protocols only if deliberately changing the experiment, not to select favourable random outcomes.

Completed jobs are skipped only if stored output checksums still match. Partial/failed outputs are preserved and cause an explicit refusal to overwrite. Gaussian training does not support resume; do not pass its `last.pt` as a BC source. After preemption, preserve the failed run and decide a new documented attempt before rerunning; do not silently append metrics. A leftover RUNNING lock after a hard kill must be checked against the actual process/allocation before manually removing it. Never remove another active job's lock.

Hashes are frozen at prepare time. Changing code, weights or banks afterwards stops execution. Prepare after merging/pulling the final code. A zero exit code alone is not accepted: row counts, traces, summaries and training updates/budget are checked before writing completion records. The main reports retain failures in success denominators, arrival-error denominators separately, and per-episode outputs for analysis by family/speed/profile; aggregate means alone do not prove generalization or statistical significance.

## Local verification

CPU tests cover initialization isolation, RNG pairing, raw Gaussian likelihood and updates, checkpoint selection, route materialization, geometric distances and manifest condition counts. Run:

```bash
python -m pytest scripts/research_comparison scripts/gaussian_chunk_rl/tests scripts/hierarchical_experts/tests scripts/diffusion_policy/evaluation/test_route_bank.py -q
```

Local checks on 2026-11-14: 55 CPU tests passed. With the actual deterministic prior, the random-initialized arm completed two local CPU-physics updates (256 transitions), actor/critic updates, checkpoint save and reload; learned tensors were finite (attention-mask infinities are intentional). The BC arm executed an actor/critic update in CPU physics but checkpoint saving failed when C: ran out of space. Logs and metrics are preserved locally; redundant smoke weights were removed to free space. GPU PhysX startup failed with allocation/kernel errors while host virtual memory was almost exhausted. A successful minimal CUDA allocation does not certify GPU physics. The GPU smoke and complete BC save/reload are **not** recorded as passed. Do not treat these wiring checks as performance evidence or approval of a full cluster allocation. After freeing disk and virtual memory, finish both checks with fresh outputs:

```bash
python -m scripts.gaussian_chunk_rl.train \
  --checkpoint "$ARTIFACT_ROOT/weights/deterministic_prior.pt" \
  --initialization random --output_dir scratch/comparison_random_smoke \
  --num_envs 4 --iterations 2 --rollout_chunks 8 --minibatch_size 32 \
  --critic_minibatch_size 32 --update_epochs 1 --seed 42 --smoke --headless --device cuda:0
```

Use a different fresh output for BC. This checks wiring only, not convergence. No real-robot work is part of this suite.
