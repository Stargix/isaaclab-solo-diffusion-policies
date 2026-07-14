# DiffuseLoco command baseline

This is the first experiment to run. It deliberately removes hindsight,
waypoints and terminal pose from the problem. The policy is conditioned on the
expert command `[vx, vy, wz, desired_height]`. The fourth value is an explicit
desired height label, never the measured base height.

For the currently working walk/crouch experts, training uses two fixed
posture-reference labels (walk `0.2932`, crouch `0.1705`). Closed-loop evaluation
shows monotonic, stable interpolation at unseen values in between. This is
emergent interpolation, not dense supervision of continuous height tracking.

## Versioned contract (schema v3)

- Control rate: 50 Hz.
- Observation history: `H=8` (`s[t-8:t]`).
- Delayed action history: `a[t-9:t-1]`; the latest value is `a[t-2]`.
- Denoising trajectory: 16 actions, `a[t-8:t+8]`.
- Deployment starts at token 8, which is exactly `a[t]`.
- DDPM: 10 train/inference steps, cosine schedule, epsilon prediction.
- Default first run: compact network, 4 layers, width 128, 4 heads. The 6-layer,
  width-256 network remains a capacity ablation/reference to DiffuseLoco.
- No classifier-free guidance dropout by default.
- Split and normalizer fitting are performed on complete training episodes only.

Old checkpoints are rejected because they do not implement this contract.

## Train

Regenerate data first: schema-v2 velocity-only files are rejected by design.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/train/train.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --output_dir scripts/baseline_diffuseloco/runs/velocity_height_compact_k10 `
  --config scripts/baseline_diffuseloco/train/configs/compact_k10.json `
  --symmetry_mode quadruped
```

For a cheap pipeline smoke test, add `--epochs 1 --batch_size 32 --step_stride 20`.

## Evaluate

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/play_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/baseline_diffuseloco/runs/walk_crouch_posture_conditioned/best.pt `
  --command 0.4 0.0 0.0 `
  --desired_height 0.2932 `
  --command_ui `
  --exec_horizon 8 `
  --torchscript_denoiser
```

The floating window controls velocity and height. Keyboard control remains
available through `--interactive_commands`. The camera follows env 0 in the
robot yaw frame; use `--no_camera_follow` to disable it.

The K=10, `exec_horizon=1` reference misses the 20 ms wall-clock deadline on the
measured Windows setup. K=10 with TorchScript and `exec_horizon=8` has a 160 ms
chunk deadline and passed the complete 45-scenario grid with 100% survival. It
also improved smoothness and tracking relative to replanning stochastically at
every action.

## Reproduce the evaluation

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/evaluate_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/baseline_diffuseloco/runs/walk_crouch_posture_conditioned/best.pt `
  --output_dir scripts/baseline_diffuseloco/evaluations/walk_crouch_epoch25_k10_exec1 `
  --repeats 3 --duration_s 8 --settling_s 2 --dynamic_segment_s 2 `
  --latency_samples 100 --exec_horizon 1 --headless

conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/evaluate_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/baseline_diffuseloco/runs/walk_crouch_posture_conditioned/best.pt `
  --output_dir scripts/baseline_diffuseloco/evaluations/walk_crouch_epoch25_k10_exec8_ts `
  --repeats 3 --duration_s 8 --settling_s 2 --latency_samples 100 `
  --exec_horizon 8 --torchscript_denoiser --skip_dynamic --headless
```

Run the regression checks before every long training:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/train/tests/test_contracts.py
```
