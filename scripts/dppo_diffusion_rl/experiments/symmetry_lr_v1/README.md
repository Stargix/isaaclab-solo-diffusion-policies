# WC/WCT DPPO: left/right augmentation pilot

Decision recorded 2026-10-05 on `research/wct-dppo`. This is an optional pilot,
not a replacement of the paper's existing actors or three-seed results.

## Question and fixed contract

Does left/right experience augmentation improve task tracking and reduce
directional artifacts without degrading path, posture, timing or survival?
It does **not** force leg synchronization, a gait identity or a skill index.

Both runs start directly from the **same Phase-A priors as historical v3**.
They do not initialize from the trained v3 actors. Critic and optimizers start
fresh. Keep supported_hybrid_v3 geometry, reward, yaw objective, transition
sampling and all optimizer settings. Seed 42; 4096 environments; 150 iterations;
32 rollout chunks; 10 denoising steps, five fine-tuned, execution horizon four.
Route mean-speed sampling reaches 1.0 m/s; remaining-pace input cap is 1.5 m/s.
Reference KL remains zero, as in the paired historical v3 experiments.

Only identity and left/right reflection are used. Although SOLO12's morphology
is front/back symmetric, the current **directed path/yaw task is not invariant
under the existing front/back transform**. Backward locomotion would not
preserve the tangent-heading and terminal-yaw objective. We therefore leave
legacy offline quadruped augmentation untouched and do not enable it online.

## Implementation and limitations

- Reflect raw proprioception, previous actions and geometric/height goals
  **before** normalization, using the existing Phase-A left/right transform.
- Reflect every action coordinate of both states of each trained denoising
  transition, including chronological history-prefix tokens. Do not reverse time.
- Require reflection-closed saved action ranges: normalized latents then transform
  by an orthogonal signed permutation. No altered normalizers or hidden Jacobian.
- Preserve original advantages, returns and old transition scores. Following
  [Mittal et al., section III-B, Eq. 6](https://arxiv.org/html/2403.04359v1#S3.SS2),
  the mirrored PPO ratio compares the new reflected score with the original
  behavior score, not a recomputed counterfactual old score.
- Average original and mirrored gradients at the same parameters, with one Adam
  step per original minibatch. Sequential backward passes limit activation-memory
  growth. Actor/critic update budgets and environment interactions are unchanged;
  update compute increases because both copies are evaluated.
- Reflect critic conditioning consistently; only privileged yaw-error sine and
  gravity-y change sign. Cross-track distance is unsigned.
- Estimate update-KL/early stopping from actual identity samples. Log reflected
  ratio and absolute logratio separately under `Symmetry/`.

This is a symmetry-PPO approximation applied to the reverse-diffusion MDP,
**not an exact likelihood-ratio theorem for arbitrary asymmetric priors**.
The state-distribution correction is omitted, as in the cited method. Large
reflected mismatch would weaken this approximation. Existing DPPO per-coordinate
log-score averaging/clipping remains unchanged; it is not a new exact joint-density
estimator. [DPPO](https://arxiv.org/abs/2409.00588) supplies the denoising-MDP framework.
Neither paper establishes in advance that this particular robot will improve.

An additional, explicit checkpoint-selection correction is enabled by
`--select_rollout_actor`: save `best.pt` **before** the update scored by the
collected rollout, rather than attach that score to post-update weights. Episode
statistics can still include episodes spanning previous updates, so the training
score remains a proxy, not a frozen evaluation. Numbered checkpoints and `last.pt`
are marked post-update/unscored. A pre-update best resumes without skipping its
unapplied update. Historical selection and augmentation remain the default.
Consequently a historical-v3 comparison is a **practical pilot comparison**, not
a strictly isolated causal estimate of augmentation alone. A causal replication
would give a no-augmentation control the same selection rule.

## Pin the actual source, not its filename

| Prior | Required original file SHA256 |
| --- | --- |
| WC | `b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd` |
| WCT | `2f28fd4beb963c74293948507eecd5e7061f34a1436d8014626bd60f2d73da54` |

The local `wct_diffusion_policy_baseline.pt` currently has a different hash;
using its name alone would silently change the WCT prior. `prepare_source.py`
first looks for an exact original file. If unavailable, it recovers only the
immutable Phase-A denoiser from the **hash-verified historical v3 archive**.
It strips trained actor, critic, optimizer and iteration information, validates
the resulting pure Phase-A policy and mirror-compatible ranges, and records
original-source, archive and snapshot hashes in `source_provenance.json`.
Serialization of a recovered snapshot changes the file hash, not the recovered
weights. Failure to find either verified source aborts before training.

Historical archive SHA256:

- WC: `2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868`.
- WCT: `5258fcf239a391bcfe6e4a951944b037b94441113d137b782ad5ade8341fa55f`.

## Cluster commands

After these changes are committed and pushed:

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git switch research/wct-dppo
git pull --ff-only origin research/wct-dppo
sbatch --array=0-1%1 scripts/dppo_diffusion_rl/experiments/symmetry_lr_v1/train_cluster.sbs
```

Index 0 is WC; index 1 is WCT. `%1` runs them sequentially on one GPU; remove
`%1` if two simultaneous GPU jobs are preferable. Existing output directories
are never overwritten. No historical launcher or checkpoint is modified.

Outputs under `scripts/dppo_diffusion_rl/runs/`:

- `dppo_wc_supported_hybrid_v3_lr_seed42/`.
- `dppo_wct_supported_hybrid_v3_lr_seed42/`.

Once training finishes, evaluate each **best.pt** with the existing frozen
benchmark, in new directories (do not compare train returns alone):

```bash
sbatch scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  scripts/dppo_diffusion_rl/runs/dppo_wc_supported_hybrid_v3_lr_seed42/best.pt \
  scripts/dppo_diffusion_rl/evaluations/wc_v3_lr_seed42_frozen
sbatch scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  scripts/dppo_diffusion_rl/runs/dppo_wct_supported_hybrid_v3_lr_seed42/best.pt \
  scripts/dppo_diffusion_rl/evaluations/wct_v3_lr_seed42_frozen
```

Inspect joint success and survival, CTE, height MAE, average-speed error and
arrival failures by geometry, speed and transition direction. Check repeated
crouch-start and fast/OOD cases explicitly. Keep current models unless the
frozen results show a useful improvement without important regressions.
One seed is a pilot, not sufficient evidence to replace three-seed paper tables.

## Validation

`python -m pytest scripts/dppo_diffusion_rl/tests -q`: **104 passed**.
Tests cover raw/latent reflection, Gaussian transition-score equivariance,
causal prefix, normalized-action compatibility, critic layout, original-score
ratio, finite actor/critic updates, unchanged Adam step budget, warm-up,
source recovery and pre/post-update checkpoint selection/resume.

A direct comparison with HEAD's original updater with symmetry `none` gives
bitwise-identical actor/critic weights and all historical update metrics on the
same seeded batch. Actual WC source recovery equals its original EMA tensors;
actual WCT recovery equals all 94 immutable source tensors. Launcher passes
`bash -n` and uses LF line endings.

Local Isaac Lab / CUDA smoke checks completed successfully for **both** priors:
16 environments, two iterations, actor and critic updates enabled, finite metrics
and the expected mirrored-minibatch count. WC used two chunks per iteration;
WCT used 32 to exercise completed-episode selection as well. WCT's actual
`best.pt` was verified tensor-for-tensor to contain the pre-update actor at
iteration 0; loading it reports the correct resume iteration. Smoke uses one
update epoch and small minibatches; the real launcher retains historical defaults
and ten critic-only warm-up iterations. These tests validate execution, **not**
convergence or improved task performance. Runtime logs/checkpoints are in ignored
`scratch/dppo_lr_*_20261005*`, not part of the repository publication.
