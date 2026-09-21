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
