# Pronk-only nominal imitation

Next authorized experiment, 2026-10-18. This is the first step, before the
optional `pronk_recovery_v1` augmentation experiment. No expert retraining,
new data collection, mixture with walk/crouch, DPPO, gait reward or skill input.
WC/WCT and the final publication actor/holdout are unchanged.

## Question

Can the existing diffusion architecture faithfully reproduce the frozen pronk
V8 expert when pronk is the only source? The objective is motion reproduction,
not merely reaching a route endpoint or moving at the requested average speed.
This separates isolated imitation failure from interference in a mixed prior
and from later changes under RL. One seed is an exploratory diagnostic, not a
new comparative publication claim.

## Frozen recipe

- Only `walk_crouch_pronk_v1_aligned/pronk_full.hdf5`, SHA256
  `2d4c3947e2403841883e8c734d1a44fec41908e855e7df9f829e8b887e489ca4`.
- 3345 accepted whole nominal episodes, 299 frames each, 1,000,155 frames.
  No pilots, raw reset-indexing error, speed-partition index or mixed HDF5.
- This is the final repaired file from the existing collection. Do not apply
  reset repair again or modify its samples. The exact hash is checked.
- Retain all 669,000 padded action anchors, including startup and stops.
  There is no impulse, so the later recovery experiment's 92-anchor mask does
  **not** apply. This raw HDF5 is self-contained; no linked walk/crouch files
  or expert checkpoint are needed to train offline.
- Same profile16 hindsight (2 s), history 8, prediction horizon 16/offset 8,
  128/4/4 Transformer, K10 DDPM, EMA and quadruped augmentation.
- Same full original Phase-A JSON, byte hash pinned in `protocol.json`:
  50 epochs, batch 4096, LR 1e-4, warmup 10000, seed42. No LR/warmup changes,
  model enlargement, new phase features or initialization from another policy.
- Episode-level split: 3178 train /167 validation. Normalization fits train
  episodes only, using the existing trainer. About 621 updates/epoch, 31050
  updates total. Four symmetry transforms are augmentation, not independent
  episodes. No optimizer runs during `--preflight-only`.

The constant 0.2932-m label retains the existing nominal upright annotation,
not the instantaneous bouncing height. Expert median actual height was around
0.260 m. Offline BC has no height reward: the targets are the expert actions.
Do not interpret this train as proof of exact 0.2932-m height tracking.

## Run

After publishing this experiment directory to the research branch and placing
the one dataset at its recorded path on the cluster, use `env_isaaclab`:

```bash
python -m pytest scripts/diffusion_policy/experiments/pronk_imitation_v1/test_protocol.py -q
python scripts/diffusion_policy/experiments/pronk_imitation_v1/train.py --preflight-only
# Allocated GPU + clean research checkout, never a login-node train:
python scripts/diffusion_policy/experiments/pronk_imitation_v1/train.py --device cuda:0 --wandb
```

Output: `scripts/diffusion_policy/runs/pronk_imitation_v1_k10_seed42/`.
The wrapper checks data/config hashes, schema, all delayed actions, finite
observations, skill identity, episode lengths and nominal-height labels. It
rejects a nonempty/existing run, wrong dataset, dirty Git or missing CUDA.
It records dataset/config/executable-source hashes and actual Git revision.
No Isaac process or V8 task registration is required for this offline train.
The native expert task will be required later for a matched physical gait audit.

## Evaluate the behaviour, not only the validation loss

Preserve best.pt, the final checkpoint, config.json, split_manifest.json,
experiment_provenance.json and training log. The existing trainer selects
best.pt by EMA validation denoising loss, not physical hopping quality.

After training, audit best and final at .20/.35/.50 m/s on straight paths,
then gentle steering, with the same native control/PD/physical contract as the
expert and K10. Start with the existing execution horizon4. Compare:

1. all-attempt survival, including startup failures;
2. genuine four-foot clearance, flight intervals and completed supported
   landings, with the expert's physical cycle detector;
3. foot coordination/contact traces and base-height oscillation, to distinguish
   pronk from flat walking or another hopping pattern;
4. actual signed speed and turning, so bouncing in place is not a success.

Contact-force unloading, PCA separation, a low denoising loss or height
oscillation alone are insufficient evidence. Compare against the same expert,
not an arbitrary numerical cadence target. Generic path-evaluation plots may
help diagnose motion but do not substitute for the matched flight/cycle audit.
Do not evaluate this new pilot on the paper's sealed final holdout.

Decisions: if nominal imitation reproduces pronk, test mixture separately. If
it reproduces the gait but falls under small deviations, run recovery-augmented
imitation next. If it walks or loses the gait already in isolation, investigate
observability, sequence/phase and closed-loop sampling before another mixed
DPPO. None of those outcomes automatically validates three-skill selection.

The earlier six-environment recovery smoke exhausted memory during reset. It
does not block this simulator-free offline train and is not counted as a
successful physical validation of this new student.
