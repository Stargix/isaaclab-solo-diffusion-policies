# Transfer and launch

The local overlay contains only the hierarchy scripts and its guarded integration
in the route evaluator. It contains no weights. Its base is Git revision
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
