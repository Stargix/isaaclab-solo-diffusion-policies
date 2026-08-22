# Solo12 path-conditioned DPPO

This directory is an independent Phase-B implementation of Diffusion Policy
Policy Optimization (DPPO).  The actor is the existing geometric-hindsight
Diffusion Policy; it is not a residual policy and it does not contain a
high-level velocity controller.

## Train

PowerShell (single line):

```powershell
.\isaaclab.bat -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/real_walk_crouch_hindsight.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1 --num_envs 4096 --iterations 1000 --rollout_chunks 32 --save_interval 25 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_v1
```

Linux/cluster (single line):

```bash
./isaaclab.sh -p scripts/dppo_diffusion_rl/train.py --checkpoint checkpoints_iri/real_walk_crouch_hindsight.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1 --num_envs 4096 --iterations 1000 --rollout_chunks 32 --save_interval 25 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_v1
```

Start at 4096 environments. Increase it only after checking GPU memory and
throughput; the rollout stores `K' + 1` action trajectories per physical
decision, so DPPO memory does not scale like ordinary PPO.

Resume with the same likelihood contract:

```powershell
.\isaaclab.bat -p scripts/dppo_diffusion_rl/train.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1/last.pt --output_dir scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1_resume --num_envs 4096 --iterations 500 --rollout_chunks 32 --headless --device cuda:0 --wandb --run_name dppo_path_pose_speed_v1_resume
```

Changing `inference_steps`, `finetune_denoising_steps`, `exec_horizon`, or
`min_denoising_std` while resuming is rejected because it changes the stored
transition likelihood.

## Evaluate

The existing evaluator detects DPPO metadata and reconstructs the frozen-early
/ fine-tuned-late sampler. Do not strip the DPPO keys from the checkpoint.

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1/best.pt --output_dir scripts/dppo_diffusion_rl/evaluations/dppo_path_pose_speed_v1 --speeds 0.2 0.4 0.6 0.8 1.0 --path_shapes straight circle s_curve right_angle random_polyline --height_profile random --height_cycle 0.2932 0.25 0.21 0.1705 --height_segment_m 0.8 --repeats 3 --duration_s 50 --exec_horizon 4 --seed 42 --headless
```

Interactive right-angle transition:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/play_policy.py --checkpoint scripts/dppo_diffusion_rl/runs/dppo_path_pose_speed_v1/best.pt --default_path_mode right_angle_walk_to_crouch --desired_speed 0.4 --exec_horizon 4 --device cuda:0
```

`best.pt` is created only after at least one episode has completed. Selection
prioritizes route success and fall avoidance, then progress and mean-speed
error. `last.pt`, periodic `model_N.pt`, `metrics.jsonl`, and `run_config.json`
are always produced.

The first 10 iterations train only the critic. With 32 chunks per iteration,
this covers approximately one full 24 s route before the transformer actor is
allowed to move. Override `--critic_warmup_iterations` only as an explicit
ablation.

## Tests

```powershell
conda run --no-capture-output -n env_isaaclab python -m pytest scripts/dppo_diffusion_rl/tests -q -p no:cacheprovider
```

The suite checks the reverse DDPM transition against `diffusers`, exact
behavior-logprob reproduction, the denoising clip schedule, base-network
immutability, a complete synthetic PPO update, and reward invariants.
