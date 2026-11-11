# Transfer and launch

Use the complete hierarchy branch, which contains both V1 and V2 and the
environment changes required by V2. Checkpoints are distributed separately.
Activate the Isaac Lab environment on an allocated GPU node and follow the
repository's installation procedure if packages or robot assets are missing.
These commands do not configure cluster-specific accounts or scheduling.

## V2 training

The V2 code has passed CPU contract tests, a local GPU smoke through the 24 s
episode boundary and both curriculum stages, and a short run through the
existing route-evaluation adapter. These checks establish wiring, not learned
performance. There is no need to spend a cluster allocation on another V1
pilot or diagnostic sweep. Before training, verify the three checkpoint hashes
against the distributed artifact manifest. Change `ARTIFACT_ROOT` below if the
checkpoints are stored elsewhere.

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
