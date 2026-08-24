# Solo12 path-conditioned DPPO

This directory is an independent Phase-B implementation of Diffusion Policy
Policy Optimization (DPPO).  The actor is the existing geometric-hindsight
Diffusion Policy; it is not a residual policy and it does not contain a
high-level velocity controller.

## Train

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
loading AdamW state. Reusing a populated output directory is allowed only when
the checkpoint belongs to that same run; this prevents accidental log mixing.

## Evaluate

The existing evaluator detects DPPO metadata and reconstructs the frozen-early
/ fine-tuned-late sampler. Do not strip the DPPO keys from the checkpoint.

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_path_pose_speed_first_arrival_v2 --speeds 0.2 0.4 0.6 --path_shapes straight circle s_curve right_angle random_polyline --height_profile random --height_cycle 0.2932 0.25 0.21 0.1705 --height_segment_m 0.8 --repeats 3 --duration_s 50 --seed 42 --headless --require_empty_output_dir
```

For finite routes, use `route_arrival_speed_ratio` together with
`route_arrived`; the old full-horizon speed is only a displacement diagnostic
after the robot reaches the endpoint. `task_success` evaluates final yaw,
height and route-average speed at the first valid entry into the goal region;
waiting can no longer repair an early arrival. Height reports separate the physical
requirement at current route progress from the future height preview supplied
to the policy.

Interactive right-angle transition:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/play_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_first_arrival_v2/best.pt --default_path_mode right_angle_walk_to_crouch --desired_speed 0.4 --exec_horizon 4 --device cuda:0
```

`best.pt` is created only after at least one episode has completed. Selection
prioritizes route success and fall avoidance, penalizes corridor/overshoot, and
then uses progress, mean-speed error and terminal distance. Its score is saved
and restored on an in-place resume. `last.pt`, periodic `model_N.pt`,
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
