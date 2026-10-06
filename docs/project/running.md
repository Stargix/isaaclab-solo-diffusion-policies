# Training, evaluation and visual demos

[Overview](../../README.md) · [Method](method.md) · [Results](results.md)

Run commands from the repository root. The project uses Python 3.11, PyTorch and Isaac Sim 5.1 through this Isaac Lab fork. Follow [the framework installation guide](../../README_ISAACLAB.md#getting-started) first. The examples assume the Isaac Lab environment and SOLO12 task packages are installed.

Datasets and checkpoints are not bundled. Supply your expert, imitation and refined checkpoints at the example paths. These commands reproduce interfaces and parameter recipes; they do not download the reported actors.

## Visual demo

A standing-to-crouching route with path, goal and actual-trajectory overlays:

```bash
./isaaclab.sh -p scripts/diffusion_policy/play_policy.py \
  --checkpoint checkpoints/policy.pt --num_envs 1 \
  --default_path_mode walk_to_crouch --transition_fraction 0.5 \
  --desired_speed 0.45 --exec_horizon 4 \
  --visualize_path --visualize_preview --visualize_goal \
  --visualize_actual_path --visualization_style clean --device cuda:0
```

PowerShell equivalent on one line:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/play_policy.py --checkpoint checkpoints/policy.pt --num_envs 1 --default_path_mode walk_to_crouch --transition_fraction 0.5 --desired_speed 0.45 --exec_horizon 4 --visualize_path --visualize_preview --visualize_goal --visualize_actual_path --visualization_style clean --device cuda:0
```

Red is the reference, blue the measured trace, orange the policy preview. The small yellow goal and heading marker display the current preview target.

The `clean` preset uses shaded lines and draws each reference segment once, avoiding red/orange overpainting. Heights and target coordinates are unchanged. Add `--visualize_height` for sparse vertical guides, or omit `--visualization_style clean` to retain the original debug-line appearance.

For a custom route, use `--path_file path.npy --path_file_frame robot`. Arrays are `[N,3]` for XYZ or `[N,4]` for XYZ/yaw. Robot-frame routes are anchored to the reset pose; use `world` only for world-frame coordinates.

## Offline imitation

The [data tools](../../scripts/diffusion_policy/data/) collect trajectories, audit support and construct datasets. The reported recipe uses achieved-motion hindsight rather than a path-conditioned teacher.

```bash
./isaaclab.sh -p scripts/diffusion_policy/train/train.py \
  --datasets datasets/walk_crouch.hdf5 \
  --config scripts/diffusion_policy/train/configs/walk_crouch_hindsight_geom_profile16_a0_faithful_k10.json \
  --output_dir scripts/diffusion_policy/runs/wc_imitation \
  --run_name wc_imitation_seed42 --device cuda
```

The config fixes `goal_source=achieved`, `goal_representation=hindsight_geom_profile16`, ten denoising steps and quadruped augmentation. Splits are by episode, with normalization fitted on training episodes.

For the matched deterministic control, use `walk_crouch_hindsight_geom_profile16_deterministic_chunk_v1.json` and a separate output directory.

## Online diffusion refinement

Start from pure imitation. Critic and optimizer initialize fresh; this is not RL from scratch.

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py \
  --checkpoint scripts/diffusion_policy/runs/wc_imitation/best.pt \
  --require_phase_a_source \
  --output_dir scripts/dppo_diffusion_rl/runs/wc_refined_seed42 \
  --run_name wc_refined_seed42 \
  --route_distribution supported_hybrid_v3 \
  --route_speed_max_mps 1.0 --speed_budget_max_mps 1.5 \
  --transition_boundary_min_m 0.8 --transition_boundary_max_m 3.2 \
  --path_reward_weight 2.0 --route_cte_rmse_tolerance_m 0.10 \
  --profile_height_reward_weight 0.75 --profile_height_mae_tolerance_m 0.04 \
  --reference_kl_coef 0.0 \
  --num_envs 4096 --iterations 150 --rollout_chunks 32 \
  --save_interval 10 --seed 42 --headless --device cuda:0
```

Use seeds 43 and 44 and distinct output directories for the reported replication structure. The evaluated recipe does not enable the optional online symmetry pilot or later preview variants.

## Matched Gaussian PPO

Start from deterministic imitation:

```bash
./isaaclab.sh -p scripts/gaussian_chunk_rl/train.py \
  --checkpoint scripts/diffusion_policy/runs/wc_deterministic/best.pt \
  --output_dir scripts/gaussian_chunk_rl/runs/wc_refined_seed42 \
  --run_name wc_gaussian_seed42 \
  --actor_lr 0.00001 --adaptive_actor_lr --min_actor_lr 0.0000001 \
  --initial_std 0.04 --clip_ratio 0.2 --target_kl 0.02 \
  --num_envs 4096 --iterations 150 --rollout_chunks 32 \
  --save_interval 10 --seed 42 --headless --device cuda:0
```

The trainer shares the frozen route/reward contract and four-action execution. Use seeds 43 and 44 in separate output directories. See [the baseline](../../scripts/gaussian_chunk_rl/) and [comparison results](results.md#matched-gaussian-ppo-comparison).

## Evaluation

A small paired diagnostic over the four ordinary families:

```bash
./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py \
  --checkpoint checkpoints/policy.pt \
  --route_bank scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/route_banks/wct_id_routes_v1.npz \
  --path_shapes procedural coherent_smooth rounded_waypoint hard_waypoint \
  --path_heights 0.1705 0.2932 --speeds 0.2 0.35 0.45 \
  --route_length_m 4 --duration_s 24 --repeats 10 \
  --output_dir scratch/route_evaluation --headless --device cuda:0
```

This is a subset, not the full benchmark. Keep route bank, seed and conditions paired across actors. More speeds do not create more independent geometries.

A [geometry-difficulty bank](../../scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v2/route_banks/) is also stored. Transitions, repeated profiles and temporal controls need separate paired suites. See the [frozen protocol notes](../../scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/README.md).

Existing traces can be analyzed without another rollout:

```bash
./isaaclab.sh -p scripts/diffusion_policy/evaluation/diagnostic_v2.py scratch/route_evaluation
```

## Source and local outputs

Weights, HDF5 datasets, evaluation traces, videos and logs are local artifacts. Queue settings, certificates and server paths belong in private launchers. Public Python entry points and configs define the experiment; the algorithms do not require a scheduler script.

The [preserved Isaac Lab README](../../README_ISAACLAB.md) covers framework installation and licences.
