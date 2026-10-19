# Pronk imitation with expert takeovers

Small exploratory continuation of the isolated nominal pronk student. This is
intervention-based imitation/data aggregation, not a reproduction of original
DAgger, DART, PDP or Diff-DAgger. It does not alter WC/WCT, the paper's actor,
reward, architecture, skill conditioning or sealed publication holdout.

## Decision

The nominal learner fails under its own feedback, including with expert
initialization, shorter execution chunks and an achieved-path oracle. Native
expert disagreement grows before falls. Therefore collect corrections from
learner-induced deviations rather than repeat successful nominal trajectories.

References: [DAgger](https://proceedings.mlr.press/v15/ross11a.html),
[PDP](https://arxiv.org/html/2406.00960v2). PDP motivates corrective physical
motion data; our perturbations are short learner prefixes, not its noise recipe.
This pilot does not prove that observation or model limitations are absent.

## Collection and labels

Frozen student: nominal pronk-only EMA, SHA `2098f85b...` (full SHA in protocol.py).
Expert: native V8 model_2999, SHA `73c572ff...`. No new gait expert or RL updates.
Native V8 nominal physics, original PD/joint order, no observation noise, delay,
pushes or physics randomization. This isolates nominal reproduction, NOT robust
locomotion under the complete expert training distribution.

Each six-second attempt has .5 s expert-controlled zero-command startup,
then .4/.6/.8 s student control, followed by full expert takeover WITHOUT reset.
Commands are fixed forward .20/.35/.50 m/s after startup. No action blending.
The student receives current-heading straight geometric profiles during
collection; this is a gait experiment, not evidence of global route tracking.

First cohort: 18 attempts (six per speed; two per prefix duration). Every speed
must have >=80% of attempts survive, complete >=2 verified landing cycles after
2 s, and have mean speed error <=.10 m/s. These are engineering gate choices,
not thresholds from a paper. Failed attempts stay in collection.json. If the
gate fails, collection stops and fine-tuning refuses the data.

Four cohorts total, 72 attempts proposed; only full surviving, gait/speed-valid
attempts are training data. Cohorts 0–2 are training; cohort 3 is held out.
Selection and failed attempt counts are explicit. Do not report accepted-data
survival as all-attempt success. Terminating HDF5 dones indicate the deliberate
6-s collection boundary, not a physical fall. Failed resets are never stitched
into accepted episodes.

HDF5 actions are the ACTUAL executed actions; last_action[t] is actions[t-1].
No hypothetical expert labels are passed off as executed learner actions.
Recovery anchors are retained only when executable future actions and the
remaining 2-s outcome are expert-controlled. Past context may include learner
states/actions, and past target tokens describe that same actual past. Hindsight
goals follow the actual recovery trajectory. No fabricated clean history.
Height stays at the historical nominal .2932-m task annotation, not the
instantaneous bounce and not a skill code.

## Paired bounded fine-tuning

Two arms start from the SAME nominal EMA, with new Adam and EMA schedules:

- nominal: 100% original nominal data;
- recovery: 50% original nominal, 50% qualified takeover data.

Both: 600 updates, batch128, LR1e-5, 50-update warmup, seed42, original
quadruped augmentation and K10 diffusion loss. No architecture or reward change.
These pilot budgets are declared choices, not a claim of optimal hyperparameters.
Keep the ORIGINAL normalizer fixed to preserve the checkpoint's input/output
mapping. The first local audit found two out-of-range target elements among
111360 eligible expert action elements. After that audit, the exploratory
protocol added an explicit mask for unsupported 16-token target windows (past
AND future), with every removed anchor listed in provenance. Stop if >1% of
windows need masking. This choice is post-audit, not a preregistered paper
comparison. Raw recordings are retained; never clip corrective labels, widen
the sampler or refit stats silently. Physical evaluation still counts every
attempt, including states requiring unsupported corrections.

Use 32 original training episodes and eight original validation episodes,
chosen deterministically from the saved episode split. Cache at most 2048
training windows per source and 128 validation windows per source. Sampling
reuses windows; it does not create independent experimental evidence.
Whole-episode/cohort splits prevent window leakage. Both arms are evaluated on
the same held-out nominal/recovery buffers. Primary checkpoint is last.pt after
exactly 600 updates, not the model selected by the nicest physical rollout.
This is a fine-tuning pilot, not an optimizer resume of the 50-epoch run.

## Local/allocated-GPU commands

From repository root with env_isaaclab (PowerShell single-line syntax):

```powershell
python -m pytest scripts/diffusion_policy/experiments/pronk_intervention_v1/test_protocol.py -q
python scripts/diffusion_policy/experiments/pronk_intervention_v1/collect.py --rounds 4 --headless --device cuda:0
python scripts/diffusion_policy/experiments/pronk_intervention_v1/finetune.py --arm recovery --preflight_only
python scripts/diffusion_policy/experiments/pronk_intervention_v1/finetune.py --arm nominal --device cuda:0
python scripts/diffusion_policy/experiments/pronk_intervention_v1/finetune.py --arm recovery --device cuda:0
```

All outputs default to ignored scratch/pronk_intervention_v1_20261019. Existing
outputs are rejected, not overwritten. Supply --output/--collection for a new
declared run. No SBS, checkpoints or generated data belong in this directory.
The physical script requires the registered optional native V8 task and the
existing runtime/controller.py history utility in this research checkout; the
public main checkout alone may not contain those optional dependencies.

Student-only screens: same 18-attempt cohort/seed543, six seconds, K10/chunk4,
no expert startup or correction, startup failures count. Compare all three:

```powershell
python scripts/diffusion_policy/experiments/pronk_intervention_v1/collect.py --evaluate --rounds 1 --seed 543 --checkpoint best.pt --output scratch/pronk_intervention_v1_20261019/eval_initial --headless --device cuda:0
python scripts/diffusion_policy/experiments/pronk_intervention_v1/collect.py --evaluate --rounds 1 --seed 543 --checkpoint scratch/pronk_intervention_v1_20261019/fit_nominal/last.pt --output scratch/pronk_intervention_v1_20261019/eval_nominal --headless --device cuda:0
python scripts/diffusion_policy/experiments/pronk_intervention_v1/collect.py --evaluate --rounds 1 --seed 543 --checkpoint scratch/pronk_intervention_v1_20261019/fit_recovery/last.pt --output scratch/pronk_intervention_v1_20261019/eval_recovery --headless --device cuda:0
```

Evaluate survival, speed and completed physical cycles together, not loss or
force unloading alone. Passing six seconds is only a preview: a promising
student needs longer episodes, repeat seeds, gentle steering and the original
physical perturbations before collecting a new mixed-skill dataset. If neither
arm improves, do not send a long run automatically: next isolate feedback/state
representation instead. No local pilot justifies a guarantee of success.
