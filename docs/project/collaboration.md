# Getting started

This page covers the current walk/crouch controller and the files needed to run it. For training recipes and evaluation commands, see the [running guide](running.md).

## Set up the project

Install Isaac Lab and this repository's task packages as described in the [framework guide](../../README_ISAACLAB.md#getting-started). The experiments used Python 3.11 and Isaac Sim 5.1; the [artifact guide](artifacts.md) records the historical software versions. Git LFS is needed for the robot assets.

Model weights and datasets are stored separately. Place the artifact bundle in a directory accessible to the machine running Isaac Lab. To check the files needed for the demo:

```bash
python scripts/tools/verify_artifacts.py --root artifacts --profile demo
```

Here `artifacts` is the root of the extracted bundle. The two required files are `weights/wc_seed42.pt` and `metadata/wc_seed42.json`. The checkpoint already contains the normalizer and configuration used by the actor. The metadata is for inspection.

## Run the controller

From the repository root, with the artifact directory above:

```bash
./isaaclab.sh -p scripts/diffusion_policy/play_policy.py \
  --checkpoint artifacts/weights/wc_seed42.pt --num_envs 1 \
  --default_path_mode walk_to_crouch --transition_fraction 0.5 \
  --desired_speed 0.45 --exec_horizon 4 \
  --visualize_path --visualize_preview --visualize_goal \
  --visualize_actual_path --visualization_style clean --device cuda:0
```

The [running guide](running.md#visual-demo) also has a PowerShell version and instructions for custom routes. This demo is a simulation, not a hardware timing test.

## Code and experiments

| Directory | Contents |
|---|---|
| [`scripts/diffusion_policy/`](../../scripts/diffusion_policy/) | Data collection, hindsight labels, imitation and evaluation |
| [`scripts/dppo_diffusion_rl/`](../../scripts/dppo_diffusion_rl/) | Task distribution, rewards and diffusion policy refinement |
| [`scripts/gaussian_chunk_rl/`](../../scripts/gaussian_chunk_rl/) | Deterministic imitation and Gaussian PPO comparison |
| [`source/isaaclab_tasks/isaaclab_tasks/direct/solo12/`](../../source/isaaclab_tasks/isaaclab_tasks/direct/solo12/) | SOLO12 tasks and expert environments |

The existing Gaussian comparison starts from an imitation checkpoint; it is not PPO from scratch. The next comparisons under consideration are a high-level controller using the original walk/crouch experts and a controller trained directly on the path-conditioned task. Their task inputs, reward, evaluation conditions and interaction budgets need to be recorded alongside each run.

The collection presets name `checkpoints/walk_final.pt` and `checkpoints/crouch_final.pt`; see [`phase_a_collection_presets.json`](../../scripts/diffusion_policy/data/phase_a_collection_presets.json) for their commands and task names. Check the weights and their observation contracts before using them in a hierarchy.

## Reproducing the reported results

The final evaluation used a frozen actor and source snapshot. The current source tree is a cleaned derivative; the [validation notes](validation.md) and [artifact guide](artifacts.md) explain how to identify the historical files.

The local archive contains the selected WC actor, Gaussian seed-42 actor, offline priors, evaluation tables and the frozen final evaluation bundle. Some original comparison checkpoints and training records still need to be recovered. Until then, the archived tables can be checked, but the complete set of comparison rollouts cannot be reproduced from this package alone.

Keep new development routes separate from the historical evaluation banks. Choose checkpoints using development results and reserve a fresh test bank for the eventual paper comparison.
