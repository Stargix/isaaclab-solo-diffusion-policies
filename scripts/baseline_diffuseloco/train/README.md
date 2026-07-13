# DiffuseLoco command baseline

This is the first experiment to run. It deliberately removes hindsight,
waypoints, terminal pose and height from the problem. The policy is conditioned
only on the expert command `[vx, vy, wz]`.

## Versioned contract (schema v2)

- Control rate: 50 Hz.
- Observation history: `H=8` (`s[t-8:t]`).
- Delayed action history: `a[t-9:t-1]`; the latest value is `a[t-2]`.
- Denoising trajectory: 16 actions, `a[t-8:t+8]`.
- Deployment starts at token 8, which is exactly `a[t]`.
- DDPM: 10 train/inference steps, cosine schedule, epsilon prediction.
- Default reference network: 6 layers, width 256, 8 heads.
- No classifier-free guidance dropout by default.
- Split and normalizer fitting are performed on complete training episodes only.

Old checkpoints are rejected because they do not implement this contract.

## Train

The existing aligned `walk_raw.hdf5` can be reused directly; it does not have to
live under this folder.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/train/train.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_raw.hdf5 `
  --output_dir scripts/baseline_diffuseloco/runs/walk_k10 `
  --config scripts/baseline_diffuseloco/train/configs/large_k10.json `
  --symmetry_mode quadruped
```

For a cheap pipeline smoke test, add `--epochs 1 --batch_size 32 --step_stride 20`.

## Evaluate

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/play_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/baseline_diffuseloco/runs/walk_k10/best.pt `
  --command 0.4 0.0 0.0 `
  --exec_horizon 1
```

Run the regression checks before every long training:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/train/test_contracts.py
```
