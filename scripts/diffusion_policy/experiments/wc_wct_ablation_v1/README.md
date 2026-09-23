# WC versus WCT causal ablation

This experiment tests whether adding the fast-trot demonstrations expands the
capability of the path/pose-conditioned diffusion prior.  The intervention is
the demonstration set only:

- **WC:** frozen walk + crouch demonstrations;
- **WCT:** the same walk + crouch demonstrations plus the validated fast-trot
  demonstrations.

There is no skill index, gait label, gait reward, transition controller, model
change, or warm start.  Both Phase-A actors use schema 8
(`hindsight_geom_profile16`) and are trained from scratch with the same seed,
architecture, optimizer, diffusion schedule, symmetry augmentation and number
of epochs.  Keeping 50 epochs gives every demonstration the same expected
number of exposures.  WCT consequently receives more optimizer steps because
adding expert data is the experimental intervention; an update-matched WC run
is a secondary compute-control, not the primary comparison.

## First run: paired Phase-A WC

The WCT actor already exists.  The first new run reconstructs its paired WC
control from the exact frozen WC source dataset:

```bash
sbatch scripts/diffusion_policy/experiments/wc_wct_ablation_v1/train_phase_a_wc_cluster.sbs
```

Expected output:

```text
scripts/diffusion_policy/runs/wc_wct_ablation_v1_phase_a_wc_k10/best.pt
```

The launcher intentionally fails before training if:

- the WC dataset, WCT dataset, or shared config has a different SHA-256;
- the Git worktree is dirty;
- the output directory is non-empty.

It also records the Git commit and artifact hashes in `provenance.txt` inside
the run directory.  The training code is not modified.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| WC dataset `walk_crouch_phase_a_waypoint_v3.hdf5` | `00aea50b71d7ce3fb69c05996f51ae43f6a56623ac8a3314f7b684ae903e3a3f` |
| WCT dataset `walk_crouch_sprint_phase_a_waypoint_v4.hdf5` | `4e323bd426ccb6f503e70ee255b397899e0da2f847eb60fa6a21f147f7a6b730` |
| Shared schema-8 config | `df066b2a27be2c56c3d59b051ebf6f2f04b6f217ef990758c6967d47fc803a87` |
| Exact Phase-A ancestor of the published WCT-DPPO line | `2f28fd4beb963c74293948507eecd5e7061f34a1436d8014626bd60f2d73da54` |

The existing WCT checkpoint itself confirms the shared contract: history 8,
prediction horizon 16, execution offset 8, 100-step spatial look-ahead,
transformer 128/4/4, DDPM K=10, batch 4096, AdamW at `1e-4`, 50 epochs, EMA
`0.9999`, quadruped symmetry and seed 42.

## What comes after this run

Do not compare only Phase-A validation loss: WC and WCT are fitted to different
data distributions.  After WC Phase A passes its constant walk/crouch sanity
gate, its own DPPO actor must be trained with the same staged protocol used by
WCT.  The final four-way comparison uses the frozen route banks and
`exec_horizon=4`:

1. Phase-A WC;
2. DPPO WC;
3. Phase-A WCT;
4. DPPO WCT-v6.

The primary causal outcome is supported fast-route success (up to 0.85 m/s),
with survival, CTE, height error, mean-speed error and OOD generalization
reported jointly.  Contact-pattern similarity is diagnostic only.

## Frozen continuation protocol

The following order is part of the experiment. Do not tune a reward or alter a
route distribution after seeing an intermediate result.

### 1. Phase-A WC integrity and sanity gate

Use `best.pt`, not the final epoch, and first verify that its embedded config,
dataset path, schema, seed and split manifest match this experiment. Run the
same frozen-bank evaluator used for Phase-A WCT with `exec_horizon=4`.

This gate is not expected to solve the downstream task: Phase A WCT itself was
weak before DPPO. It only rejects a technically invalid WC fit (non-finite
losses, corrupt checkpoint, incompatible schema, unstable native walk/crouch,
or an obvious training failure). Validation losses between WC and WCT are not
a causal performance comparison because their data distributions differ.

### 2. DPPO WC acquisition stage

```bash
sbatch scripts/dppo_diffusion_rl/experiments/wc_wct_ablation_v1/train_dppo_wc_v3_cluster.sbs
```

Expected output:

```text
scripts/dppo_diffusion_rl/runs/dppo_wc_supported_hybrid_v3/best.pt
```

The admitted source is `checkpoints_iri/checkpoints_dp/wc_wct_phase_a.pt`
(SHA-256 `b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd`).
The launcher falls back to the Phase-A WC run `best.pt` if that alias is
absent, and aborts on any other hash. Do not reuse or relax
`supported_hybrid_v3/train_cluster.sbs`.

It copies the v3 training contract exactly:

- `supported_hybrid_v3`, 4096 environments, 150 iterations and seed 42;
- route maximum 1.0 m/s and speed budget 1.5 m/s;
- transition boundary 0.8--3.2 m;
- path weight 2.0, CTE tolerance 0.10 m;
- profile-height weight 0.75, height tolerance 0.04 m;
- reference KL 0, 32 rollout chunks and save interval 10.

No skill id, contact target, gait reward or WC-specific feasibility rule is
allowed. This stage tests whether online optimization alone can recover the
fast capability when its Phase-A prior never saw fast-trot demonstrations.

Evaluate the selected WC-v3 `best.pt` on the same frozen banks after the
train job finishes. Submit with Slurm `afterok` so the eval reads the final
checkpoint rather than a mid-run file:

```bash
sbatch --dependency=afterok:<train_jobid> \
  scripts/dppo_diffusion_rl/experiments/wc_wct_ablation_v1/evaluate_dppo_wc_v3_cluster.sbs
```

Expected output:

```text
scripts/dppo_diffusion_rl/evaluations/wc_wct_ablation_v1_dppo_wc_v3
```

The launcher does not hash-lock the actor; it records the SHA of whatever
`best.pt` exists at eval start. Do not point it at Phase-A WC or any WCT
actor.

### 3. DPPO WC consolidation stage

Starting from the selected WC-v3 actor, repeat the v6 consolidation contract
in a separate WC launcher:

- `supported_hybrid_v6`, 120 iterations, seed 42;
- restart critic and optimizer;
- replay-only reference KL 0.01 on update group 0;
- actor LR `5e-6`, 4096 environments and save interval 5;
- every remaining route/reward parameter identical to WCT-v6.

The WCT-v6 launcher has an intentionally hard-coded WCT-v3 checksum. Do not
edit or relax it. The WC launcher must instead freeze the checksum of the
newly selected WC-v3 source.

### 4. Four-way final evaluation

Evaluate Phase-A WC, DPPO WC, the exact Phase-A WCT ancestor and DPPO WCT-v6
with the same code commit, frozen ID/OOD route banks, seeds, conditions and
`exec_horizon=4`. Report route-clustered confidence intervals and keep these
groups separate:

- ordinary constant and transition routes;
- repeated height profiles;
- supported fast routes at 0.65, 0.75 and 0.85 m/s, split C->W and W->C;
- 0.95--1.0 m/s as boundary/stress only;
- ID and OOD geometries.

Primary metrics are task success, arrival, survival, CTE RMSE, height MAE and
mean-speed error. Local speed allocation along the route tests whether time is
recovered by accelerating in safe sections and slowing near corners/crouch.
Gait/contact plots are mechanism diagnostics, never success criteria.

### 5. Preregistered interpretation

Evidence that fast demonstrations add causal capability requires:

- at least +10 percentage points supported-fast success for DPPO WCT over
  DPPO WC;
- survival at least 95% and no more than 2 percentage points regression on
  ordinary navigation;
- the advantage to remain visible on OOD routes and in speed allocation, not
  only in a gait-similarity statistic.

If WC matches WCT, the correct conclusion is that DPPO, rather than fast-trot
data, supplied the fast capability. If both fail, do not tune the reward again:
the prior/support hypothesis is not established. If WCT wins clearly, repeat
the decisive WC and WCT training conditions with at least three total seeds
before a paper claim; route bootstrap intervals do not measure train-seed
variance.

A 75-epoch WC Phase-A run is reserved as a secondary equal-update control
because WCT has 1.5 times as many demonstrations. It is justified only after a
clear primary WCT advantage; it must not replace the primary equal-exposure
comparison.

## Conditional-multimodality audit

The following CPU-only diagnostic tests whether different expert futures remain
after matching the complete Phase-A conditioning, rather than assuming that a
multi-skill dataset is automatically conditionally multimodal:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/experiments/wc_wct_ablation_v1/audit_conditional_multimodality.py `
  --wc_dataset scripts/diffusion_policy/data/datasets/walk_crouch_phase_a_waypoint_v3.hdf5 `
  --wct_dataset scripts/diffusion_policy/data/datasets/walk_crouch_sprint_phase_a_waypoint_v4.hdf5 `
  --output_dir scripts/diffusion_policy/data/audits/wc_wct_conditional_multimodality_v2 `
  --samples_per_skill 5000 `
  --projection_dimensions 64 `
  --candidate_neighbors 64 `
  --temporal_exclusion_steps 50 `
  --seed 42
```

Skill labels are used only after matching for diagnosis. They are never policy
inputs. The audit compares only future executable tokens and excludes trivial
nearby frames from the same episode.

## Matched deterministic action-chunk control

`train_phase_a_wc_deterministic_cluster.sbs` is the registered one-seed
control for the diffusion-necessity question. It uses the same WC HDF5,
schema-8 profile conditioning, split, quadruped augmentation, 16-token target
and execution offset as the Phase-A diffusion prior. Only the output objective
changes: a direct Transformer predicts the normalized 16-action chunk in one
forward pass instead of DDPM noise prediction and reverse denoising. It does
not add a skill label, gait loss, smoothness loss, reward term or DPPO update.

Before submitting it, commit the implementation so the launcher's clean-tree
gate can pass. The launcher checks the frozen dataset and config hashes and
writes its exact provenance. The subsequent evaluator must use the same frozen
route banks and `--exec_horizon 4` as the clean WC Phase-A benchmark.
