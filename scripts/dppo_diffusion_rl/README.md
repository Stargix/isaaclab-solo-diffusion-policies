# Solo12 path-conditioned DPPO

This directory is an independent Phase-B implementation of Diffusion Policy
Policy Optimization (DPPO).  The actor is the existing geometric-hindsight
Diffusion Policy; it is not a residual policy and it does not contain a
high-level velocity controller.

The environment selects its actor observation contract from checkpoint
metadata. Both the original `hindsight_geom_avg12` checkpoints and the new
`hindsight_geom_profile16` checkpoints are supported; mixing a policy kind,
goal name and dimension is rejected before simulation starts. In schema 8,
the four route samples are `[x,y,h_required]` tokens and `h_now` is appended
before terminal yaw and average speed. Task contract v4 adds sustained profile
tracking to schema-8 success and checkpoint selection without changing DPPO's
likelihood mathematics.

## Train

Next controlled run: Stage-1 geometry with the four-level height curriculum.
Job 3475 showed that contract v4 alone cannot improve intermediate-height
tracking when Stage 1 samples only walk/crouch targets. Geometry and height
difficulty are now independent: `--route_stage 1 --height_profile_stage 2`
keeps the solved geometry/speed distribution while exposing 0.2932, 0.25,
0.21 and 0.1705 m profiles. The run starts from Job 3471 `model_100`, which
dominates Job 3475 `iter110` on the matched 4 m evaluation. Critic and Adam
state are reset and the KL reference is re-anchored to `model_100`:

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wct_100_potxo.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_wct_profile16_multilevel_c4_v1 --run_name dppo_wct_profile16_multilevel_c4_v1 --restart_optimization --reference_kl_coef 0.05 --profile_height_reward_weight 2.0 --profile_height_mae_tolerance_m 0.04 --num_envs 4096 --iterations 150 --rollout_chunks 32 --route_stage 1 --height_profile_stage 2 --route_speed_max_mps 0.5 --speed_budget_max_mps 0.8 --save_interval 5 --seed 44 --headless --device cuda:0 --wandb
```

The profile reward flags are explicit for provenance although they equal the
v4 defaults. `--height_profile_stage 2` changes only target-height sampling;
omitting it preserves every historical run's coupling to `--route_stage`.
Legacy 12-D actors keep their old 0.75 height weight and terminal-only success
semantics. Do not start stage-2/sprint allocation until evaluation demonstrates at least
90% survival, at most 5% base contact, 75% joint success, profile MAE at most
0.04 m, speed MAE at most 0.05 m/s, and cross-track RMSE at most 0.10 m.

Historical path-v4 continuation from `dppo_path_3`. Task-contract v3 kept the
12-D geometric goal but replaces its final local `v_avg` scalar with the
closed-loop remaining-route speed budget
`(remaining distance) / (remaining target time)`.  Geometry is still previewed
at the nominal requested speed.  The reward is unchanged: this only makes
accumulated timing debt observable to the actor.

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_path_3.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_4_speed_budget --run_name dppo_path_4_speed_budget --restart_optimization --reference_kl_coef 0.05 --num_envs 4096 --iterations 500 --rollout_chunks 32 --route_stage 2 --route_speed_max_mps 0.6 --speed_budget_max_mps 0.8 --save_interval 25 --headless --device cuda:0 --wandb
```

`--restart_optimization` is mandatory when loading path3: its critic and Adam
states estimate task-contract v2.  The actor remains the path3 actor, and the
immutable reference used by the KL is deliberately re-anchored to that loaded
actor rather than to an older reference stored inside the checkpoint.

Recommended v3 after the paired path_1/path_2 audit. It retains the stable
`path_1` actor as an immutable transition-kernel reference while optimizing the
first-arrival speed objective. The task reward weights are intentionally
unchanged:

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_path_1.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_reference_v3 --restart_optimization --reference_kl_coef 0.05 --num_envs 4096 --iterations 1000 --rollout_chunks 32 --route_stage 2 --route_speed_max_mps 0.6 --save_interval 25 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_reference_v3
```

`Policy/reference_kl` measures cumulative drift from that frozen actor; it is
different from `Policy/approximate_kl`, which only compares one PPO update to
its rollout behavior. `0.05` is the preregistered initial coefficient, not a
new reward term. Keep it fixed for the primary run and compare periodic
checkpoints on the paired held-out benchmark.

Warm-start task-contract v2 from the successful v1 DPPO actor (PowerShell,
single line). The actor is retained; critic, Adam states, counters and critic
warm-up restart because the terminal return changed:

```powershell
.\isaaclab.bat -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_path_1.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2 --restart_optimization --num_envs 4096 --iterations 1000 --rollout_chunks 32 --route_stage 2 --route_speed_max_mps 0.6 --save_interval 25 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_first_arrival_v2
```

Linux/cluster (single line):

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_path_1.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2 --restart_optimization --num_envs 4096 --iterations 1000 --rollout_chunks 32 --route_stage 2 --route_speed_max_mps 0.6 --save_interval 25 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_first_arrival_v2
```

Start at 4096 environments. Increase it only after checking GPU memory and
throughput; the rollout stores `K' + 1` action trajectories per physical
decision, so DPPO memory does not scale like ordinary PPO.

Resume a v2 run with the same likelihood and task contract (without
`--restart_optimization`):

```powershell
.\isaaclab.bat -p scripts/dppo_diffusion_rl/train.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2/last.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2 --num_envs 4096 --iterations 500 --rollout_chunks 32 --route_stage 2 --route_speed_max_mps 0.6 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_first_arrival_v2
```

Changing `inference_steps`, `finetune_denoising_steps`, `exec_horizon`, or
`min_denoising_std` while resuming is rejected because it changes the stored
transition likelihood. Changing `gamma`, `gae_lambda`, or `gamma_denoising` is
also rejected because it changes the objective attached to the stored rollout.
Optimizer learning rates may be changed explicitly and are reapplied after
loading AdamW state. Restoring critic/Adam while changing route stage, height
curriculum, speed range, profile tolerance/weight or episode duration is also
rejected; use `--restart_optimization` and a new output directory. Reusing a
populated output directory is allowed only when the checkpoint belongs to that
same run; this prevents accidental log mixing.

## Evaluate

The existing evaluator detects DPPO metadata and reconstructs the frozen-early
/ fine-tuned-late sampler. Do not strip the DPPO keys from the checkpoint.

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_wct_profile16_multilevel_c4_v1/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_wct_profile16_multilevel_c4_v1_finite4m --speeds 0.2 0.35 0.5 --path_shapes straight circle s_curve right_angle random_polyline --route_length_m 4.0 --height_profile random --height_cycle 0.2932 0.25 0.21 0.1705 --height_segment_m 0.8 --repeats 3 --duration_s 24 --seed 42 --save_timeseries --headless --require_empty_output_dir
```

For finite routes, use `route_arrival_speed_ratio` together with
`route_arrived`; the old full-horizon speed is only a displacement diagnostic
after the robot reaches the endpoint. `task_success` evaluates final yaw,
height, route-average speed and (for profile16) distance-weighted full-route
height MAE at the first valid entry into the goal region; waiting cannot repair
an early arrival. `--route_length_m 4.0` matches DPPO's finite route length, so
low-speed cases also reach an evaluable endpoint. CSV/JSON report the four gates
separately. Height reports distinguish the physical requirement at current route
progress from the future height preview supplied to the policy. The evaluator
also rejects non-finite or clearly non-physical
simulator states (base outside 0.08--0.50 m, planar speed above 5 m/s, or tilt
above 60 degrees), so a PhysX escape cannot count as survival or dominate the
height plots. These are evaluation sanity bounds, not rewards.

Interactive right-angle transition:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/play_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2/best.pt --default_path_mode right_angle_walk_to_crouch --desired_speed 0.4 --exec_horizon 4 --device cuda:0
```

`best.pt` is created only after at least one episode has completed. Selection
prioritizes joint route/profile success and fall avoidance, penalizes timeout,
failed arrival, corridor and overshoot, and then uses progress, mean-speed
error, terminal distance and a bounded profile-height MAE tiebreaker. Its score
is saved and restored on an in-place resume. `last.pt`, periodic `model_N.pt`,
`metrics.jsonl`, and `run_config.json` are always produced.

The first 10 iterations train only the critic. With 32 chunks per iteration,
this covers approximately one full 24 s route before the transformer actor is
allowed to move. Override `--critic_warmup_iterations` only as an explicit
ablation.

## Tests

```powershell
conda run --no-capture-output -n env_isaaclab python -m pytest scripts/dppo_diffusion_rl/tests scripts/diffusion_policy/evaluation/test_metrics.py -q -p no:cacheprovider
```

The suite checks the reverse DDPM transition against `diffusers`, exact
behavior-logprob reproduction, the denoising clip schedule, base-network
immutability, reset/chunk handling, checkpoint resume, a complete synthetic PPO
update, reward invariants, and finite-route evaluation semantics.
