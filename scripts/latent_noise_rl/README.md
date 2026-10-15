# Frozen-prior latent-noise PPO pilot

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
