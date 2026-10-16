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

## Paper-faithful DSRL-NA follow-up

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
`gamma=0.99` once per chunk, as in their `ActionChunkWrapper`. Each collection
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
  --run_name wc_dsrl_na_smoke_seed42 --smoke --headless --device cuda:0
```

Only if the smoke reports `SMOKE_OK`, start the pilot below. This deliberately
uses the four environments of the authors' locomotion configuration. With
batch 256 and 20 updates per two vectorized macro-actions, it uses 640 replay
samples per newly collected transition, compared with 0.625 in the rejected
4096-environment draft. The 2001 standard-noise initialization macro-actions
per environment cover 160.08 simulated seconds before the first actor update;
the previous 0.32-second warmup did not cover complete routes.

```bash
./isaaclab.sh -p scripts/latent_noise_rl/train_dsrl_na.py \
  --prior wc --checkpoint checkpoints_iri/checkpoints_dp/wc_wct_phase_a.pt \
  --output_dir scripts/latent_noise_rl/runs/wc_dsrl_na_seed42 \
  --run_name wc_dsrl_na_seed42 --num_envs 4 --iterations 1000 \
  --initial_rollout_chunks 2001 --rollout_chunks 2 --save_interval 100 \
  --seed 42 --headless --device cuda:0
```

The pilot collects 8004 initialization and 8000 optimization macro-transitions
(64,016 total control transitions), and performs 20,000 SAC updates plus 10,000
noise-critic updates. This is a shorter feasibility budget than the original
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
