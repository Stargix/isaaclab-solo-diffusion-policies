# Iterative pronk imitation

## Question and scope

Can repeated expert corrections of learner-induced states make the existing
pronk diffusion student reproduce its frozen expert? This is an isolated gait
experiment, **not** new WC/WCT training, DPPO, a skill classifier or a gait reward.
It uses the original architecture, diffusion objective, normalization, action
history, quadruped augmentation and native V8 PD parameters.

The earlier one-round intervention pilot collected useful recoveries but neither
600-update arm survived any of its 18 unassisted attempts. Lower denoising loss
did not establish physical improvement. That result does not reject iterative
data aggregation. This experiment fixes the one-round and tiny cached-data
limitations without simultaneously changing the model or observation space.

## Frozen budget

- Start from the original nominal pronk EMA (`BASE_SHA` in the pilot protocol),
  not a selected pilot continuation. Expert and nominal data are also SHA-pinned.
- Three rounds, four cohorts of 18 attempts per round, each six seconds.
  Speeds: 0.20, 0.35 and 0.50 m/s; six attempts per speed per cohort.
- Half the attempts expose the student from reset; half have 0.5 seconds of
  expert startup. Commands remain zero for the initial 0.5 seconds in both.
  Student exposure lasts 0.4/0.6/0.8 seconds in round 0, increasing by 0.2 seconds
  per round. Expert then takes over, without reset or action blending.
- The first cohort must qualify in **every** speed group: >=80% survive,
  complete >=2 verified flight/landing cycles after two seconds and achieve
  mean speed error <=0.10 m/s. Failure stops the experiment, not a hidden retry.
  Each collection must also contain >=30 accepted training and >=10 validation
  episodes; failure stops before fitting.
- Aggregate all preceding recovery collections. Cohorts 0-2 train; cohort 3
  stays held out forever. Original nominal episode splits are preserved.
- Each arm gets 4,000 Adam updates per round, batch 128, LR 1e-5, 100-step warmup,
  gradient clipping 1.0 and the original EMA implementation. Initialize each
  round from its own preceding EMA with a fresh optimizer. Total: 12,000 updates
  per arm, 24,000 overall. Always evaluate the fixed-budget `last.pt`.
- Recovery batches: 50% nominal, 25% all corrective windows, 25% first 16
  corrective anchors. Collections are sampled uniformly, then windows within
  each collection. Nominal control: 100% original nominal windows with the same
  update budget. No fixed 2,048-window cache.
- Never refit old normalization or clip expert labels silently. Existing range
  audit explicitly excludes rare unsupported raw target windows and aborts if
  more than 1% would be excluded. Reports retain the excluded anchor list.

All numbers above are declared engineering choices, not hyperparameters proved
optimal by literature. There is no adaptive uncertainty detector in this version.
Scheduled exposure covers both reset and established gait without introducing
an unvalidated trigger based on pronk's naturally oscillating height.

## Label validity and limits

Only actual executed actions enter history. At a correction anchor the executable
action future and the two-second hindsight outcome must be expert-controlled.
Earlier history may contain student actions, which are recorded as executed,
not replaced with hypothetical teacher labels. First corrective anchors receive
extra weight so recovered steady gait does not dominate the corrective signal.
Failed episodes remain in reports but do not supply supervised trajectories.
This acceptance filter is a limitation, not proof the expert can recover every state.

The expert is velocity-conditioned, not a path-aware navigation oracle. Training
uses achieved hindsight; collection requests current-heading straight previews.
Therefore this is **DAgger-inspired intervention imitation**, not an exact
reproduction of full DAgger or Diff-DAgger, and not a solution to arbitrary-route
expert queries. No skill identifier is added. Native nominal physics deliberately
disables DR, pushes, observation corruption and delay: robustness comes later,
only if unassisted nominal reproduction works.

## Evaluation and decision

After each round, evaluate both arms from reset with no expert help, seed 543,
18 attempts. This diagnostic bank never selects checkpoints or changes budgets.
After all three rounds, evaluate original / nominal / recovery on fresh seeds
9543, 10543, 11543: 54 attempts per arm, no seed selection.

Report all failures, verified cycles, speed and survival; denoising loss alone
is not a success criterion. Validation also reports sampled next-four-action
RMSE in raw action units, matching the executed chunk rather than noise targets.
These action errors are diagnostics: alternative actions can still be valid.
Small nominal screens do not establish robust locomotion or mixed-skill success.

If correction helps physically and the final screen qualifies, proceed to a
separate longer/perturbed validation before mixing skills. If nominal replay
improves equally, attribute improvement to additional training, not DAgger.
If neither qualifies after this finite budget, stop: inspect observation/context
limitations and label compatibility instead of launching another blind extension.
Even a successful imitation result does not guarantee DPPO preserves pronk.

The runner saves source/input hashes, Git dirtiness, per-stage logs and outcomes.
It checks required artifacts because Kit can close without a trustworthy exit
status. Existing output directories are never overwritten. There is deliberately
no automatic resume or automatic new variant after a failed gate.

## Run

From the repository root, activated Isaac Lab environment (PowerShell):

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/experiments/pronk_dagger_v1/run.py
```

This is **preflight only**: checks frozen inputs/dependencies, no Kit or training.
To run the bounded experiment locally:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/experiments/pronk_dagger_v1/run.py --execute --device cuda:0 --output scratch/pronk_dagger_v1
```

Linux uses `./isaaclab.sh -p` instead. On another machine, provide `--checkpoint`,
`--expert` and `--nominal` paths explicitly; file hashes must still match. The
native V8 task, runtime helper and both experiment folders must exist there.
No Slurm launcher or generated data is added to the public repository.

### Source publication prerequisite

V8 inherits historical configurations; publishing only its final file is not
sufficient. Its required local additions are `solo12_pronk_v4.py` through
`solo12_pronk_v8.py`, `pronk_curriculum.py`, `pronk_schedule.py`, and the matching
changes in `solo12_pronk_env.py` and `pronk_gait.py`. V2 and `pronk_ppo.py` are
already tracked. V3 is not in V8's import chain. Do not publish every historical
launcher or every task registration to solve this dependency problem.

The collector registers V8 privately when needed; the global `solo12/__init__.py`
does not need any change. The new `dependencies.py` checks transitive local
Python imports without starting Kit or requiring data:

```powershell
python scripts/diffusion_policy/experiments/pronk_dagger_v1/dependencies.py
```

The runner refuses sources absent from Git or different from HEAD, before
loading data or starting Kit. Use the committed checkout on the cluster. A
research working directory with unpublished changes to imported sources
(including package initializers) intentionally fails this check. Unrelated
documentation, launcher and data changes do not block it. Third-party packages,
Isaac versions and physical behavior still need an actual cluster smoke check;
the AST dependency audit does not prove those are compatible.

Tests without Kit:

```powershell
python -m pytest scripts/diffusion_policy/experiments/pronk_dagger_v1/test_dagger_protocol.py scripts/diffusion_policy/experiments/pronk_intervention_v1/test_protocol.py -q -p no:cacheprovider
```

Outputs live under ignored `scratch/`: `experiment.json`, stage logs, collection
HDF5/NPZ, validation metrics, fixed-budget checkpoints, `RESULTS.md` and a final
comparison plot. Do not put these outputs in Git.

## Literature basis

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates recollecting
expert corrections under successive learner-induced state distributions rather
than assuming nominal demonstration coverage is sufficient.
[Diff-DAgger](https://arxiv.org/html/2410.14868v4) demonstrates intervention-based
aggregation with diffusion policies; its uncertainty gate is not implemented here.
[PDP](https://arxiv.org/html/2406.00960v2) supports investigating corrective physical
motion data. It does not validate our thresholds or guarantee pronk reproduction.
