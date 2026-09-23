# Matched deterministic action-chunk baseline: implementation specification

Status: implementation hand-off, 2026-09-23. The implementer must follow this
document literally and leave scientific decisions unchanged for review.

## 1. Scientific question

Does DDPM training provide a useful Phase-A prior for the SOLO12
path/posture/time task, or can a direct deterministic Transformer trained on the
same action chunks achieve comparable closed-loop behaviour?

This experiment isolates the **output distribution/training objective** as far
as practical. It is not an MLP-versus-Transformer comparison, not a one-step
versus action-chunk comparison, and not an RL-from-scratch experiment.

The conditional-multimodality audit found no strict walk/crouch ambiguity in WC
and only 0.65% strict cross-skill overlap in WCT, with no strictly matched pairs
exceeding the within-skill p95 action-divergence threshold. Therefore do not
claim that a deterministic model must fail through mode averaging.

## 2. Frozen primary comparison

Implement and train **WC deterministic BC, seed 42 only** first.

| Contract item | Frozen value |
|---|---|
| Dataset | `walk_crouch_phase_a_waypoint_v3.hdf5` |
| Dataset SHA-256 | `00aea50b71d7ce3fb69c05996f51ae43f6a56623ac8a3314f7b684ae903e3a3f` |
| Diffusion comparator | `wc_wct_ablation_v1_phase_a_wc_k10/best.pt` |
| Comparator SHA-256 | `b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd` |
| Goal contract | schema 8, `hindsight_geom_profile16` |
| History | 8 |
| Prediction horizon | 16 |
| Execution offset | 8 |
| Evaluation execution horizon | 4 |
| Goal horizon | 100 simulator steps |
| Symmetry | `quadruped` |
| Episode split | 5%, seed 42, existing `split_episode_indices` |
| Backbone size | `d_model=128`, 4 heads, 4 decoder layers |
| Optimizer | AdamW, LR `1e-4`, weight decay `1e-3`, betas `(0.9, 0.95)` |
| Schedule | 10,000-step warmup then cosine |
| Batch / epochs | 4096 / 50 |
| EMA | existing `EMAModel`, max decay `0.9999` |

Do not train WCT, additional seeds, PPO, DPPO or a smaller MLP in this
implementation. Those are later decision branches.

## 3. Required architecture

Add a direct action-chunk Transformer. It must consume exactly:

- normalized `proprio_hist`: `[B, 8, 30]`;
- normalized delayed `action_hist`: `[B, 8, 12]`;
- normalized `goal_hist`: `[B, 8, 16]`.

It must output one normalized joint-target trajectory `[B, 16, 12]` and expose
the same runtime surface used by the evaluator:

```python
compute_loss(batch) -> scalar
predict_action(proprio_hist, action_hist, goal_hist, ...) -> [B, 16, 12]
predict_action_denormalized(...) -> [B, 16, 12]
executable_chunk(trajectory, num_actions) -> [B, num_actions, 12]
set_normalizer_stats(stats)
configure_optimizers(...)
```

### 3.1 Backbone constraints

Create, do not retrofit, a small dedicated module such as:

- `scripts/diffusion_policy/model/transformer_chunk_policy.py`;
- `scripts/diffusion_policy/model/solo12_deterministic_chunk_policy.py`.

Reuse the proven design choices from `TransformerForDiffusion`:

- separate two-layer Mish MLPs for I/O and goal tokens;
- learned conditioning positional embeddings;
- the same `TransformerDecoderLayer` configuration, GELU, norm-first and
  dropout;
- a learned sequence of 16 action-query embeddings decoded through
  cross-attention to the conditioning memory;
- final LayerNorm and linear 12-dimensional action head;
- the existing AdamW decay/no-decay grouping semantics.

Do **not** feed ground-truth or previous future actions to the decoder. Do not
reuse the DDPM noisy-action input with an all-zero tensor: that would leave its
input projection weights permanently inactive and create a handicapped control.
Learned query tokens are the clean deterministic analogue.

Use causal visibility consistent with the existing decoder:

- query token `j < history` sees I/O and goal history through `j`;
- future query tokens see the complete eight-token observed history;
- no future ground-truth token is available.

The deterministic model may have slightly fewer parameters because it has no
diffusion-time encoder. Log exact trainable parameter counts for both models.
The implementer must not resize either model to force exact equality. Flag the
comparison for review if the deterministic count is outside 90--110% of the
diffusion count.

### 3.2 Normalization, loss and inference

Use the existing `NormalizerStats` and functions:

- z-score proprio and goals;
- min-max normalize action history and target actions;
- train with ordinary mean squared error over **all 16 action tokens and 12
  joints**.

Also log, but do not optimize separately:

- total normalized MSE;
- executable-future MSE over tokens `8:16`;
- per-joint future MSE.

Do not introduce smoothness, gait, velocity, height or auxiliary losses. Do not
weight future tokens differently. The purpose is to compare DDPM noise
prediction with direct action regression, not to tune a stronger bespoke BC
objective.

The raw network output is used for the training MSE. At inference only, clamp
normalized actions to `[-1, 1]` before denormalization, matching the bounded
DDPM sample support. Do not use `tanh` inside the training forward pass.

`guidance_scale` and random `generator` arguments may be accepted for API
compatibility, but `guidance_scale != 1` must raise a clear error. Prediction
must be deterministic in `eval()` mode.

## 4. Integration with the existing training pipeline

Prefer a narrow backward-compatible extension of
`scripts/diffusion_policy/train/train.py` over copying its full training loop.

Add CLI:

```text
--policy_backend {diffusion,deterministic_chunk}
```

with default `diffusion`. Existing commands and checkpoints must behave exactly
as before when the flag is omitted.

Required branching points only:

1. `make_config`: use the distinct policy kind
   `spatial_hindsight_height_profile_deterministic_chunk_bc` for the new backend.
2. Policy construction: instantiate the appropriate class.
3. Checkpoint serialization: write `noise_scheduler_config` only for diffusion;
   write top-level `algorithm="deterministic_chunk_bc"` and
   `policy_backend="deterministic_chunk"` for the new policy.
4. Logging: print backend, total parameters, total loss and executable-future
   loss.

Everything else must remain shared:

- `SpatialHindsightDataset`;
- episode-level split;
- normalizer fitted only on training episodes;
- quadruped augmentation;
- DataLoaders;
- AdamW, scheduler, AMP, clipping and EMA;
- best-checkpoint selection by the same validation objective;
- split manifest and provenance.

Do not edit the WC/WCT HDF5 files, goal builder, symmetry transforms, diffusion
policy implementation, DPPO code, reward code or evaluation metrics.

Create a config copied from
`walk_crouch_hindsight_geom_profile16_a0_faithful_k10.json`, changing no
hyperparameter. A filename such as
`walk_crouch_hindsight_geom_profile16_deterministic_chunk_v1.json` is expected.
The backend is selected by CLI, not hidden inside an unrelated diffusion field.

## 5. Checkpoint and loader contract

The deterministic checkpoint must retain:

- `schema_version=8`;
- the existing height-profile `goal_schema`;
- full config and dataset paths;
- `model_state_dict` and `ema_model_state_dict`;
- optimizer, scheduler, scaler and EMA state;
- epoch, global step, normalizer stats, validation loss and split manifest;
- `algorithm` and `policy_backend` tags above.

Extend `load_training_checkpoint` only enough to validate this explicit policy
kind. Never allow a deterministic checkpoint to be silently constructed as a
diffusion policy or vice versa. Resume must require identical dataset/model
sections and backend.

## 6. Evaluator integration

Modify `scripts/diffusion_policy/evaluate_policy.py` minimally:

1. Inspect `policy_kind`/`policy_backend` after loading.
2. Build `Solo12DeterministicChunkPolicy` for the new kind and the existing
   policy for every old kind.
3. Load the EMA weights in both cases.
4. Keep the same history buffers, goal construction, `executable_chunk`, route
   metrics and plots.
5. Record `policy_backend` and actual sampler calls in `evaluation_summary.json`.

For deterministic checkpoints, `num_inference_steps` is not meaningful. Set the
recorded value to `null` and print a warning if the generic launcher supplies
it; do not pretend ten deterministic passes occurred. The frozen comparison
must explicitly pass `--exec_horizon 4`.

Do not fork or copy the 2,000-line evaluator. The compatible policy interface
exists specifically to avoid duplicated evaluation logic.

## 7. Tests required before any cluster train

Add focused CPU tests under
`scripts/diffusion_policy/model/test_deterministic_chunk_policy.py` or the
repository's established test location.

Required tests:

1. **Shapes:** batch sizes 1 and 4 produce `[B,16,12]`.
2. **Finite loss and gradients:** normalized synthetic batch, backward pass,
   at least one conditioning encoder, query token and decoder parameter has a
   finite non-zero gradient.
3. **Condition dependence:** changing only the goal changes the prediction.
4. **Determinism:** repeated `eval()` calls are bitwise equal on CPU.
5. **Bounded deployment:** deliberately large network outputs are clamped only
   in `predict_action`, not in the training loss path.
6. **Execution offset:** horizon 4 returns exactly tokens `8:12`; invalid sizes
   raise.
7. **Checkpoint round trip:** saved EMA weights and normalizer reproduce the
   same prediction after reload.
8. **Backend rejection:** loading a deterministic checkpoint as diffusion and
   vice versa fails loudly.
9. **Legacy regression:** construct the existing diffusion policy and run one
   loss and prediction call unchanged.
10. **Parameter count:** emit both counts and enforce the 90--110% review gate.

Commands the implementer must run locally:

```powershell
conda run --no-capture-output -n env_isaaclab python -m py_compile `
  scripts/diffusion_policy/model/transformer_chunk_policy.py `
  scripts/diffusion_policy/model/solo12_deterministic_chunk_policy.py `
  scripts/diffusion_policy/train/train.py `
  scripts/diffusion_policy/evaluate_policy.py

conda run --no-capture-output -n env_isaaclab python -m pytest `
  scripts/diffusion_policy/model/test_deterministic_chunk_policy.py -q
```

Then run a CPU/CUDA-independent one-batch smoke command. Add a dedicated
`--max_train_steps 1` developer-only option if no safe equivalent exists; it
must default to `None` and must not enter the canonical config. Verify that a
checkpoint can be loaded and produces a finite trajectory.

## 8. Cluster launcher

Add one launcher under this experiment, for example
`train_phase_a_wc_deterministic_cluster.sbs`. It must:

- require a clean worktree;
- verify the frozen WC dataset hash above;
- verify the deterministic config hash recorded after review;
- refuse a non-empty output directory;
- record Git commit, dirty state, dataset/config hashes, Slurm ID, GPU and exact
  command in `provenance.txt`;
- use a new output directory such as
  `scripts/diffusion_policy/runs/wc_wct_ablation_v1_phase_a_wc_deterministic_v1`;
- pass `--policy_backend deterministic_chunk` explicitly;
- use seed 42 and no warm start.

No canonical train may be launched until the reviewer approves the diff and the
smoke test.

## 9. Frozen evaluation protocol

First run cheap gates:

1. offline validation total/future MSE;
2. finite deterministic rollout;
3. native constant walk and crouch sanity at 0.20, 0.35 and 0.45 m/s;
4. survival, action smoothness and absence of joint-limit saturation.

If technically valid, run the same frozen Phase-A benchmark used for the clean
WC diffusion prior, with identical route banks, evaluator seed, reset mode,
conditions and `exec_horizon=4`. Do not compare validation MSE numerically to
DDPM epsilon-prediction loss because they are different objectives.

Report side by side:

- ID constant and transition success;
- OOD constant and transition success;
- repeated walk-start and crouch-start;
- supported-fast only as diagnostic, since neither Phase-A model was optimized
  online for that task;
- survival, arrival, CTE RMSE, height MAE, mean-speed error and action delta;
- parameter count, checkpoint size, training wall time and inference latency.

## 10. Preregistered decision after one seed

Use route-level paired results, not validation loss alone.

- **Deterministic clearly worse:** if it has a material and consistent closed-
  loop deficit (suggested gate: at least 10 percentage points task success, or
  materially worse survival/CTE on multiple ID/OOD suites), retain diffusion as
  an empirically useful offline prior. Inspect failures before attributing the
  gap specifically to multimodality.
- **Comparable:** if differences are within 5 percentage points and continuous
  errors overlap substantially, Phase A does not justify diffusion necessity.
  The next experiment is one matched online comparison, designed separately;
  do not automatically launch three deterministic seeds.
- **Deterministic better:** report diffusion as a compatible DPPO
  parameterization rather than a necessary imitation architecture. Keep the
  main contribution around hindsight conditioning and online acquisition.
- **Technically invalid:** NaNs, broken loading, mismatched split or immediate
  failures do not count as a scientific loss. Repair implementation only; do
  not tune architecture or loss after seeing benchmark results.

The 5--10 percentage-point bands are decision aids, not confidence intervals.
Final claims require route-clustered uncertainty and, only if competitive,
additional train seeds.

## 11. Explicitly out of scope

The implementer must not add:

- skill IDs, gait labels or contact-pattern targets;
- auxiliary/smoothness losses;
- reward changes, curriculum or DPPO modifications;
- a smaller/larger architecture search;
- WCT training;
- an MLP one-step policy;
- PPO fine-tuning;
- changes to route generation, success criteria or frozen banks;
- commits, pushes or removal of existing files unless separately authorized.

## 12. Handoff checklist

The implementation handoff must contain:

- concise `git diff --stat` and list of modified files;
- parameter counts and their ratio;
- all test outputs;
- smoke command and output;
- confirmation that a legacy diffusion checkpoint still loads;
- proposed cluster command, but no launched job;
- any deviation from this document called out explicitly.

The reviewer will inspect normalization, temporal alignment, mask visibility,
checkpoint dispatch and evaluator parity before approving the cluster run.
