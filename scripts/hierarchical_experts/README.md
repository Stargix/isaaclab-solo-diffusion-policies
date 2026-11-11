# Hierarchical expert baseline

PPO chooses walk or crouch and a body-frame velocity command. The frozen expert runs at 50 Hz; the high-level decision is held for four steps. The actor receives the same eight-step proprioception, previous-action and 16-dimensional task history as the path-conditioned controller. The critic additionally receives the task's privileged critic features. The actor is trained from scratch; the WC-v3 checkpoint is used only to verify the environment contract.

The experts also use body-frame base linear velocity, which is not part of the 30-dimensional high-level proprioception used by the diffusion actor. In simulation this is a true state measurement; on the robot it would require an estimator. This observation difference must be considered when interpreting any comparison.

The default configured command limits are walk `(±2.0, ±1.0, ±1.0)` and crouch `(±1.0, ±0.5, ±1.0)`. These are configured ranges, not experimentally established operating limits. `--command_range_mode shared` reproduces the previous common limits. Both experts receive the last action actually sent to the robot, including after a switch.

The shared limits can restrict walk more than its configured range. Measure each checkpoint's tracking and stability before selecting per-expert limits; the configured ranges alone do not establish reliable operation. The adaptive actor learning-rate option is rejected: its inherited Gaussian-only KL calculation does not support the hybrid action distribution.

The velocity distribution is Gaussian before `tanh`; the skill is categorical. PPO uses the joint likelihood of the sampled latent velocity and skill. The logged entropy is the joint *latent* entropy, not the entropy of bounded commands. The trainer saves `best.pt` only when a completed episode is available; its score belongs to the pre-update actor. `last.pt` is the post-update actor and has not yet been scored. Optimizer states are saved for inspection, but resume is not implemented.

For a diagnostic pilot, `--categorical_entropy_coef` adds an entropy bonus only to the skill distribution (default 0 preserves the earlier objective). `Rollout/categorical_entropy` reports selector entropy separately. `--command_range_mode shared` tests the common command envelope. Neither is a proven fix: compare independent runs and task-conditioned evaluation, without tuning on the final holdout. The coefficient is recorded in `run_config.json`; checkpoint resume remains unsupported. Entropy regularization follows the PPO objective, while the expert hierarchy is inspired by, not a reproduction of, ANYmal Parkour: https://arxiv.org/abs/1707.06347 and https://arxiv.org/abs/2306.14874.

`Rollout/crouch_prob_low_target` and `Rollout/crouch_prob_high_target` are mean sampled-policy probabilities at route target heights below 0.20 m and above 0.25 m, respectively; the sample counts are logged alongside them. Intermediate targets are excluded. These are not episode-level success or deterministic evaluation scores. The shared command envelope is a diagnostic restriction, not a measured safety guarantee.

For an ablation, set `HIERARCHY_COMMAND_RANGE` to `configured` or `shared` and `HIERARCHY_CATEGORICAL_ENTROPY` to 0 or a small coefficient such as 0.01 when calling `run_cluster.sh`. Use a distinct `HIERARCHY_RUN_ROOT` for each condition and seed. These settings leave the reward and frozen experts unchanged. Compare equal interaction budgets and evaluate saved checkpoints on the same development cases; reserve the final holdout. A single pilot or this coefficient cannot establish that the collapse has been resolved.

`--execution_steps` (or `HIERARCHY_EXECUTION_STEPS` in the launcher) can test holding the entire high-level decision for 4, 8, 12 or 16 low-level steps. This holds velocity as well as skill and changes the effective GAE time scale, so it is not a skill-duration-only ablation. The history still advances at 50 Hz. Match physical steps rather than iteration counts when comparing intervals, and evaluate with the same interval used in training. The default remains 4.

## Smoke test

From the repository root, with Isaac Lab activated and checkpoints downloaded into `scratch/artifacts`:

```bash
python scripts/hierarchical_experts/train.py \
  --walk_checkpoint scratch/artifacts/experts/walk_final.pt \
  --crouch_checkpoint scratch/artifacts/experts/crouch_final.pt \
  --reference_checkpoint scratch/artifacts/weights/wc_seed42.pt \
  --output_dir scratch/hierarchy_smoke \
  --num_envs 4 --iterations 2 --rollout_chunks 4 --smoke --headless
```

The output directory must not exist. A smoke run only checks integration; it is not a performance result. For a pilot, choose a new output directory and remove `--smoke`; the defaults are 4096 environments, 32 decisions per environment and 150 updates. These are starting settings, not an established optimal budget. Record seeds, actual simulator steps, training time and compute when comparing methods.

A smaller first pilot, for example on a workstation, can start with the same command but use `--output_dir scratch/hierarchy_pilot_seed42 --num_envs 256 --iterations 150 --rollout_chunks 32` and omit `--smoke`. Check available GPU memory and throughput first; this smaller batch changes the optimization regime, so it is not automatically equivalent to the 4096-environment setup.

Commit the code before a scientific run, so the saved commit and dirty-tree flag identify the exact implementation. The checkpoint files are external artifacts; their SHA-256 hashes are saved in the run configuration and model checkpoint.

For a separate wiring check, replace `--smoke` with `--sanity_steps 200`. This runs a simple scripted tracker with a height-based expert switch and writes `sanity.json`; it does not update the policy and is not the hierarchical RL baseline.

The sanity report groups completed episodes by the target-height plateaus actually observed before termination, and records separate arrival, contact, corridor, overshoot and timeout outcomes. This grouping does not identify the complete intended route after an early failure. The scripted tracker caps its command at 0.6 m/s, so its results do not validate faster or arbitrary commands, nor replace a paired evaluation.

The existing route evaluator accepts `--hierarchy_checkpoint` plus both expert paths. Its required `--checkpoint` argument supplies the WC-v3 task schema only; diffusion weights are not used for actions. The adapter checks task settings and expert hashes, uses deterministic skill argmax and velocity means, and recomputes expert actions at every low-level tick, including posture staging. Output identifies both the evaluated hierarchy and the reference schema checkpoint. Diffusion CUDA graphs and oracle fixtures are rejected for this backend.

Before interpreting a comparison, use identical route banks, staging, conditions and metrics across actors. This baseline has a different low-level interface and initialization from DPPO, so results alone do not isolate every design choice. The short bridge check is not a performance benchmark.

## Cluster sequence

Use the same repository revision and the `isaaclab` environment on an allocated GPU node. The external artifact root must contain `experts/walk_final.pt`, `experts/crouch_final.pt` and `weights/wc_seed42.pt`. Output directories must not already exist.

```bash
export ARTIFACT_ROOT="$PWD/artifacts"
export HIERARCHY_RUN_ROOT="$PWD/scratch/hierarchy_runs"
bash scripts/hierarchical_experts/run_cluster.sh smoke
bash scripts/hierarchical_experts/run_cluster.sh probe
bash scripts/hierarchical_experts/run_cluster.sh pilot 42
bash scripts/hierarchical_experts/run_cluster.sh eval 42 "$HIERARCHY_RUN_ROOT/pilot_seed42/last.pt"
```

Inspect `probe_seed42/probe.json` first. It measures forward tracking and height for constant skills and both switch directions, and records resets. It is a native-expert diagnostic, not the route benchmark, and does not automatically certify bounds. Inspect pilot learning curves and mode switching before the full budget:

The local 20-second seed-42 diagnostic had no resets for the two switch directions at 0.25/0.50/0.75 m/s, or constant commands up to 1.0 m/s. Constant walk at 1.5 m/s had one reset. These are single cases on flat terrain, not reliability estimates. The probe does not establish the full lateral/yaw envelope or walk at 2 m/s. The configured-range pilot must therefore be inspected before treating it as a final comparison.

```bash
for seed in 42 43 44; do
  bash scripts/hierarchical_experts/run_cluster.sh train "$seed"
done
```

The evaluation helper is a small straight-route diagnostic. For paper evaluation use the existing evaluator's frozen-bank and condition flags with `--hierarchy_checkpoint`; do not tune on the historical final holdout. These commands do not configure cluster-specific partitions, accounts or containers.
