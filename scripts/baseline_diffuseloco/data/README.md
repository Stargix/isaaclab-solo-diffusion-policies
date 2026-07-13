# Data for the command baseline

Use a single stable walking expert. Mixing crouch/jump experts while conditioning
only on velocity creates an ambiguous target and is intentionally disabled in
this baseline collector.

The required HDF5 alignment is:

- `obs[t]` is captured before applying `actions[t]`;
- `last_action[t] = actions[t-1]` and is zero at `t=0`;
- `command_speed[t] = [vx, vy, wz]` is the command seen by the expert;
- the final `dones` element of every saved demonstration is true;
- `control_rate_hz = 50`.

The loader validates these conditions and fails before training if they are not
present. Collection uses a deterministic command-sampling seed and stores it in
`data.attrs["collection_seed"]`.

## Regenerate the walking data

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/data/collect_data.py `
  --mode single `
  --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt `
  --num_envs 128 `
  --num_steps 1500000 `
  --command_resample_time_s 2.0 `
  --physics_dr_mode light `
  --seed 42 `
  --output_name walk_raw.hdf5 `
  --headless
```

Inspect it before training:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/data/inspect_dataset.py `
  --dataset scripts/baseline_diffuseloco/data/datasets/walk_raw.hdf5
```

The spatial fields `root_pos_w` and `root_quat_w` remain in the raw schema so a
single collection can also be audited or reused later, but the command baseline
does not consume them.
