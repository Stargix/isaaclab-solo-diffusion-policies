# Frozen-prior latent-noise PPO pilot

> The section below documents the earlier PPO pilot. It is **not** the
> paper's original RL optimizer. The DSRL-NA implementation and launch recipe
> are documented at the end of this file.

This is one exploratory run on the WCP Phase-A imitation checkpoint. PPO changes
the first 12 rows of the diffusion sampler's initial noise, while a deterministic
10-step DDIM decoder and its learned prior remain frozen. It does not add gait
labels, gait rewards, a new route generator, or a new locomotion expert.

The design follows the latent steering construction in [Steering Your Diffusion
Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799),
but the optimizer here is PPO. The paper's DSRL-SAC/DSRL-NA results do not transfer
to this implementation. The sampler is DDIM eta=0; it differs from the DDPM chain
used by the prior's historical playback and DPPO. Comparisons with DPPO therefore
compare complete pipelines, not only whether the weights are frozen.

The prior is pinned to SHA256
`40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886`.
The physical task is the existing WCP nominal-window task. The actor receives only
the existing normalized observation histories. No latent coordinate is interpreted
as a gait command, and a good task score does not prove pronk retention.

## Commands

The isolated worktree does not contain the locally ignored Isaac Sim installation.
On Windows, run from the main repository root and invoke its launcher with the
script path under `.worktrees/latent-steering`. This keeps the local simulator
available while `train.py` resolves the research code from the isolated branch.
Activate `env_isaaclab` first. Checkpoints are ignored by Git and are not copied
automatically when the worktree is created; pass the verified source by absolute
path. Preflight checks that source, the scheduler, causal masks and output path
without starting Kit:

```powershell
python .worktrees/latent-steering/scripts/latent_noise_rl/train.py --checkpoint "C:\absolute\path\to\walk_crouch_pronk_phase_a.pt" --output_dir .worktrees/latent-steering/scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42 --preflight-only --headless --device cuda:0
```

Run the short integration smoke in a new, separate directory:

```powershell
.\isaaclab.bat -p .worktrees/latent-steering/scripts/latent_noise_rl/train.py --checkpoint "C:\absolute\path\to\walk_crouch_pronk_phase_a.pt" --output_dir .worktrees/latent-steering/scripts/latent_noise_rl/runs/wcp_ddim_ppo_smoke_seed42 --smoke --headless --device cuda:0
```

Only after a successful smoke, the fixed full pilot is:

```powershell
.\isaaclab.bat -p .worktrees/latent-steering/scripts/latent_noise_rl/train.py --checkpoint "C:\absolute\path\to\walk_crouch_pronk_phase_a.pt" --output_dir .worktrees/latent-steering/scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42 --headless --device cuda:0
```

The smoke runs for only 0.32 simulated seconds, so it is expected not to finish
an episode or produce a task-selected `best.pt`; it verifies finite PPO updates,
frozen-prior integrity and `initial.pt`/`last.pt` round-tripping instead.

After the branch is checked out normally on the cluster, run from that checkout:

```bash
./isaaclab.sh -p scripts/latent_noise_rl/train.py --checkpoint "$PWD/scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/best.pt" --output_dir scripts/latent_noise_rl/runs/wcp_ddim_ppo_smoke_seed42 --smoke --headless --device cuda:0
```

Only after a successful smoke, use the same command with
`--output_dir scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42` and remove
`--smoke`. The full budget is fixed at 4096 environments, 32 chunks/update, four
physical controls/chunk, 150 iterations and seed 42: 78,643,200 control
transitions, with four 5-ms physics substeps per control transition.

```bash
./isaaclab.sh -p scripts/latent_noise_rl/train.py --checkpoint "$PWD/scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/best.pt" --output_dir scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42 --headless --device cuda:0
```

For a one-off play, use the task contract saved in the actor:

```powershell
.\isaaclab.bat -p .worktrees/latent-steering/scripts/diffusion_policy/play_policy.py --task solo12-v0 --checkpoint .worktrees/latent-steering/scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42/best.pt --num_envs 1 --latent_reference_mode checkpoint --desired_speed 0.5 --visualize_path --visualize_goal --visualize_preview --visualize_actual_path --visualize_height --device cuda:0
```

## Frozen evaluation matrix

Use `scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/route_banks/wct_id_routes_v1.npz`,
the four declared ID families, repeats 0 and 1, four metres, 24 seconds, seed 4242,
deterministic resets, 25 warm-up steps, 50 posture-settling steps, timeseries on,
and the protocol defaults recorded in the paper validation. Construct the nine
exact evaluation commands for modes `checkpoint`, `initial`, and `gaussian` with:

```powershell
python .worktrees/latent-steering/scripts/latent_noise_rl/experiments/wcp_pilot.py --launcher .\isaaclab.bat --checkpoint .worktrees/latent-steering/scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42/best.pt --output_root .worktrees/latent-steering/scripts/latent_noise_rl/evaluations/wcp_pilot
```

From the branch checkout on Linux:

```bash
python -m scripts.latent_noise_rl.experiments.wcp_pilot --launcher ./isaaclab.sh --checkpoint scripts/latent_noise_rl/runs/wcp_ddim_ppo_seed42/best.pt --output_root scripts/latent_noise_rl/evaluations/wcp_pilot
```

This command builder prints commands; it does not launch them. Run the printed
commands inside a GPU allocation after training. Each mode/case has a separate
output directory.
Evaluation and playback use `solo12-v0`, the same physical robot/environment
without the training task's separate random route bank and route terminations.
The evaluator applies the saved nominal-window posture contract to its own paths.
For constant height, run upright 0.2932 m at 0.2/0.35/0.5/0.65/0.85 m/s and
crouch 0.1705 m at 0.2/0.35/0.45 m/s. For transitions, run both directions at
0.35/0.65 m/s and midpoint transition. The previous WCP nominal-window DPPO actor
can be added through its normal backend if its SHA is supplied and verified.

Each condition uses the saved four-action execution horizon. Do not override the
DDIM step count, guidance, task contract, or the post-selection paper bank. The
mode `gaussian` uses direct standard-normal initial noise. `initial` uses the
zero-mean, unit-standard-deviation actor with the fixed `1.5*tanh` transform;
these references are intentionally different.

After the paired runs, aggregate them without Isaac:

```powershell
python -m scripts.latent_noise_rl.analyze --run checkpoint=PATH_TO_CHECKPOINT_EVAL --run initial=PATH_TO_INITIAL_EVAL --run gaussian=PATH_TO_GAUSSIAN_EVAL --output_dir scripts/latent_noise_rl/analysis/wcp_pilot
```

The analyzer writes a per-scenario CSV and two compact diagnostic figures. It
labels contact unloading as unloading; contact forces alone do not establish
geometric flight.

## DSRL-NA follow-up

`train_dsrl_na.py` implements the paper's Noise-Aliased DSRL (DSRL-NA) update,
not the PPO pilot above. The default experiment is WC (`--prior wc`), initialized
from the same Phase-A checkpoint as archived WC-v3 DPPO, SHA256
`b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd`.
The Phase-A diffusion prior and deterministic DDIM
decoder stay frozen. SAC's bounded actor chooses the initial latent noise;
the action critic learns from replayed decoded action chunks, and the twin
noise critic is repeatedly distilled on fresh `N(0, I)` latents using the
action critic's values. SAC then updates the noise actor through that learned
noise critic. The decoder is never differentiated through.

WC uses the archived DPPO task: pointwise (`instantaneous`) height tracking,
0.25-m transition margin and vertical-velocity weight 0.025, alongside the
same supported_hybrid_v3 geometry, timing and tracking settings. `--prior wcp`
is an explicit follow-up for gait preservation; it keeps the WCP prior SHA
and its nominal-window posture task, zero transition margin and zero vertical
velocity cost. The loader rejects a checkpoint from the other prior family.
Changing only `--checkpoint` must not silently switch physical task contracts.
Comparison with historical DPPO still compares complete pipelines because the
frozen DSRL decoder uses DDIM whereas historical DPPO uses its DDPM chain.

The update sequence follows the authors' [DSRL-NA implementation](https://github.com/ajwagen/stable-baselines3-dsrl/blob/master/stable_baselines3/dsrl/dsrl.py).
The adapter sums the rewards of the four executed 50-Hz controls and applies
`gamma=1.0` once per chunk. Unlike the authors' Gym benchmark, this task uses
undiscounted endpoint-potential differences; see the job-4197 audit below. Each collection
block has two vectorized macro-actions followed by 20 SAC updates. Ten
noise-critic updates follow that complete block, rather than each SAC update.

The actor, its log probability and the noise critic use normalized tanh noise
coordinates. Scaling noise before the frozen decoder does not change the
entropy objective. Default noise bounds are `[-1,1]`, matching the normalized
action bounds inherited by the official NA launcher; its `action_magnitude=1.5`
configuration is used by the DSRL-SAC environment wrapper. Actor and critic
networks use Tanh and LayerNorm; width 512 is a project setting for the larger
history/conditioning input, not a claim that the paper used this width.

The training checkpoint records the Phase-A prior, actor, both critics, target
critic, temperature, scheduler and task contract. Evaluation/play sample the
learned noise distribution with an explicit seeded generator and the same
10-step DDIM decoder. `--dsrl_sampling deterministic` is an optional separate
actor-mean diagnostic; the default is `--dsrl_sampling stochastic`, matching
the authors' primary evaluation. Checkpoint version 2 records the corrected
entropy and discount conventions and rejects the earlier draft's version 1.
There is no `best.pt` chosen from training return; retain periodic checkpoints
and select a model only with the frozen evaluation protocol.

On the cluster, after checking out this branch and confirming the pinned Phase-A
checkpoint exists, first run the short integration smoke in its own directory:

```bash
./isaaclab.sh -p scripts/latent_noise_rl/train_dsrl_na.py \
  --prior wc --checkpoint checkpoints_iri/checkpoints_dp/wc_wct_phase_a.pt \
  --output_dir scripts/latent_noise_rl/runs/wc_dsrl_na_smoke_seed42 \
  --run_name wc_dsrl_na_smoke_seed42 --initial_alpha 0.01 --smoke --headless --device cuda:0
```

Only if the smoke reports `SMOKE_OK`, use the corrected pilot command in the
job-4197 audit below. It uses four environments, as in the authors' locomotion
configuration. With
batch 256 and 20 updates per two vectorized macro-actions, it uses 640 replay
samples per newly collected transition, compared with 0.625 in the rejected
4096-environment draft. The 2001 standard-noise initialization macro-actions
per environment cover 160.08 simulated seconds before the first actor update;
the previous 0.32-second warmup did not cover complete routes.

The historical 1000-iteration pilot collected 8004 initialization and 8000 optimization macro-transitions
(64,016 total control transitions), and performed 20,000 SAC updates plus 10,000
noise-critic updates. This was a shorter feasibility budget than the original
experiments. It does not establish preserved pronk or generalization; those
require the frozen multi-route evaluation. The run metadata reports the sample
ratio and all update counts explicitly. Do not increase `num_envs` to PPO scale
without redesigning this ratio. Generated checkpoints are ignored local artifacts.

Regression checks cover replay column shapes/wraparound, terminal masking,
undiscounted chunk rewards, the exact optimizer-block order, entropy coordinate
invariance, frozen-prior integrity, checkpoint round-tripping and the evaluation
action-chunk API:

```bash
python -m pytest scripts/latent_noise_rl/tests/test_dsrl_na.py scripts/latent_noise_rl/tests/test_latent_noise.py -q
```

The Isaac smoke runs for only 0.64 simulated seconds per environment. It checks
updates and seeded checkpoint round-tripping, not task success or gait retention;
zero completed episodes in that smoke must not be interpreted as a success rate.

## Limits

- One training seed and a small development route bank characterize a pilot; they
  do not establish generalization or superiority over DPPO.
- The Phase-A WCP prior was unstable upright in earlier closed-loop checks, so
  success would show that latent steering can improve task behavior from that
  prior, not recovery of a demonstrated stable pronk skill.
- The task's nominal height target and reward can still favor flat walking.
- DDIM changes the sampler relative to historical DDPM evaluation.
- Inference still executes ten denoising passes and has no real-robot qualification.
- A low-foot-load interval is not proof of flight or pronk.

## WC job 4197 audit and corrected follow-up

The completed job is a failed pilot, not evidence against diffusion steering.
Its baseline completed 85 routes: 76.5% position arrival, 2.35% timing success,
21.2% base contact and 1.18% joint success. Height MAE was 1.13 cm and CTE RMSE
4.10 cm. The optimized actor had zero successes across 83 completed episodes
(ten non-overlapping 100-iteration metric windows). This is not 1000 independent
evaluations. Baseline and actor visited different routes, so these figures are
not a paired generalization comparison.

One concrete implementation mismatch was copying SAC's `gamma=0.99` into this
task. `dppo_diffusion_rl/config.py` explicitly requires 1.0: schedule and terminal
pose rewards are potential differences. With discounting, their sum depends on
intermediate states and arrival time, rather than only the intended endpoints.
At 0.08 s per macro-action, gamma=0.99 weights a signal 20 s later by only
`0.99**250 = 0.081`; the same signal 4 s later retains `0.99**50 = 0.605`.
The new trainer rejects discounted task returns. Old checkpoints remain readable
for evaluation and are not resumable; their recorded gamma is preserved.

The reported `qA` and `qW` were MSE losses, not Q values. Mean Bellman target
peaked around 506 and later decreased, so the logs do not establish numerical
Q divergence. They do show initial targets around 89 despite initially small
random critic predictions. The 144-dimensional entropy term is a plausible
major contributor; the old logs did not record its decomposition. New logs
separately record task reward, entropy bonus, bootstrap Q, current Q, actor Q
and terminal fraction, and label losses unambiguously.

For the next pilot, keep the source, task, architecture and update schedule
fixed. Use gamma=1 (the correction) and explicitly lower initial alpha to 0.01
(a declared tuning hypothesis, not an original-paper default). Automatic
temperature adaptation remains enabled with target entropy 0. This only reduces
the initial entropy scale; it neither removes SAC's entropy objective nor proves
that it will remain well balanced. The default alpha remains 1 for reproducibility.
An entropy-regularized finite-horizon objective can still favour extra steps,
even with the task's discount repaired. Do not interpret this as a guarantee of
matching the DPPO objective exactly.

After the smoke, use a fresh output directory, never the failed run's directory:

```bash
./isaaclab.sh -p scripts/latent_noise_rl/train_dsrl_na.py \
  --prior wc --checkpoint checkpoints_iri/checkpoints_dp/wc_wct_phase_a.pt \
  --output_dir scripts/latent_noise_rl/runs/wc_dsrl_na_endpoint_v2_seed42 \
  --run_name wc_dsrl_na_endpoint_v2_seed42 --num_envs 4 --iterations 5000 \
  --initial_alpha 0.01 --initial_rollout_chunks 2001 --rollout_chunks 2 \
  --save_interval 500 --seed 42 --headless --device cuda:0
```

This collects 40,000 optimization macro-transitions, five times the previous
pilot's online experience, plus 8,004 initialization transitions. The increased
budget is not itself a bug fix or a claim of sufficient convergence. Inspect
periodic checkpoints using a fixed development route bank, with identical seeds
for the prior, initial actor and trained actor. Evaluate stochastic sampling as
primary and the actor mean separately. Keep the paper's final holdout untouched.
If timing does not improve after meaningful completed-episode counts, stop this
optional experiment rather than adding gait rewards or changing the main task.

References: [DSRL paper](https://arxiv.org/abs/2506.15799),
[authors' implementation](https://github.com/ajwagen/stable-baselines3-dsrl/blob/master/stable_baselines3/dsrl/dsrl.py).
The gamma and initial-temperature choices above are project adaptations, not a
claim to reproduce the original experimental configuration exactly.

Local validation (2026-10-16, Isaac Sim 5.1): the integration smoke completed,
including frozen-prior checks and seeded checkpoint round-trip. An additional
four-environment diagnostic collected 9 baseline episodes, then 2 actor episodes
in five completed optimizer blocks. It was manually interrupted during the
following block to bound local test cost, not because it converged or failed.
Its initialization used 256 rather than 2001 chunks and is not a training-result
comparison. The logged Q losses were finite; this establishes integration only.
Both diagnostics are ignored artifacts under `scripts/latent_noise_rl/runs/`.
