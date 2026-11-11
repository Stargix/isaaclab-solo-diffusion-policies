# Transfer and launch

This worktree contains V1 and the versioned V2 implementation; it contains no
weights. V2 also adds opt-in curriculum handling to the existing DPPO route
environment, so transfer the complete reviewed Git revision (recommended), not
only the V1 `scripts/hierarchical_experts` overlay described in older notes.
The hierarchy baseline originally started from Git revision
`5c71ca867be7ad44096d7f09714ee0115999df69`.

Copy `hierarchy_cluster_overlay.tar.gz` to your cluster home directory, then
prepare a separate checkout (the base revision must already be available):

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git worktree add -b codex/hierarchy-cluster ../isaaclab-hierarchy 5c71ca867be7ad44096d7f09714ee0115999df69
cd ../isaaclab-hierarchy
tar -xzf ~/hierarchy_cluster_overlay.tar.gz
git add scripts/hierarchical_experts scripts/diffusion_policy/evaluate_policy.py
git commit -m "Add frozen-expert hierarchical PPO baseline and evaluation bridge"
```

Activate the existing Isaac Lab environment on an allocated GPU node. If
editable packages or robot assets are missing in this checkout, apply the
repository's normal installation/LFS procedure first. The launcher pins the
Python source roots to this checkout.

```bash
conda activate isaaclab
export ARTIFACT_ROOT="$HOME/i2r/isaaclab-solo-diffusion-policies/artifacts"
export HIERARCHY_RUN_ROOT="$PWD/scratch/hierarchy_runs"
bash scripts/hierarchical_experts/run_cluster.sh smoke
bash scripts/hierarchical_experts/run_cluster.sh probe
bash scripts/hierarchical_experts/run_cluster.sh pilot 42
bash scripts/hierarchical_experts/run_cluster.sh eval 42 "$HIERARCHY_RUN_ROOT/pilot_seed42/last.pt"
```

Change `ARTIFACT_ROOT` if your downloaded checkpoint folder is elsewhere.
All outputs use new directories and reject existing runs. The smoke exercises
at least one full episode horizon and checks that episode resets occurred.
The probe and pilot must be inspected; they do not automatically authorize
a final scientific comparison. See README for the full-budget commands.

These commands deliberately leave Slurm partitions, accounts and container
setup to the cluster's existing allocation procedure. No job has been submitted.

## V2 final-training commands

The V2 code has passed CPU contract tests, a local GPU smoke through the 24 s
episode boundary and both curriculum stages, and a short run through the
existing route-evaluation adapter. These checks establish wiring, not learned
performance. There is no need to spend a cluster allocation on another V1
pilot or diagnostic sweep. Before training, make the reviewed V2 revision
available in this checkout and verify the three artifact hashes shown in the
local `run_config.json`.

```bash
conda activate isaaclab
export ARTIFACT_ROOT="$HOME/i2r/isaaclab-solo-diffusion-policies/artifacts"
export HIERARCHY_RUN_ROOT="$PWD/scratch/hierarchy_v2_runs"

bash scripts/hierarchical_experts/run_cluster.sh v2train 42
# Inspect seed 42's training curves and development evaluation before spending
# the remaining budget; if the recipe remains viable, run the matching seeds:
bash scripts/hierarchical_experts/run_cluster.sh v2train 43
bash scripts/hierarchical_experts/run_cluster.sh v2train 44
```

Each seed uses a fresh `v2_seed${seed}` directory and an 80,000,000-transition
budget. That is a predeclared starting budget, not a promise of convergence;
inspect selector probabilities by target height, command saturation, separate
discrete/continuous KLs, episode outcomes and development evaluation before
making a performance claim. The launcher does not submit a scheduler request
or assume a cluster partition or account.

If preempted after a saved post-update checkpoint, resume the same seed without
changing the run configuration or total budget:

```bash
bash scripts/hierarchical_experts/run_cluster.sh v2resume 42 \
  "$HIERARCHY_RUN_ROOT/v2_seed42/last.pt"
```

Use `last.pt`, which is atomically refreshed after each optimizer update. If
the budget completed, use it for evaluation rather than resuming. Do not resume from `best.pt`: it is selected before the
optimizer update and is for evaluation, not continuation. Resume restores
optimizer/global RNG/curriculum sampler state but starts a fresh simulator
state and fresh rollout histories; it is not bitwise identical.

For a quick development-only adapter check after a seed completes:

```bash
bash scripts/hierarchical_experts/run_cluster.sh v2eval 42 \
  "$HIERARCHY_RUN_ROOT/v2_seed42/best.pt"
```

This uses the existing evaluation pipeline, a 4 m straight route and requested
0.25/0.50 m/s constant-height tasks. It is not a substitute for paired route
and profile evaluation or the reserved final holdout.
