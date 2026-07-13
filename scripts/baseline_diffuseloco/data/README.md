# Data for the command baseline

Collect each stable expert separately, then merge. The baseline takes
`[vx, vy, wz, desired_height]`, so the explicit height makes walk/crouch
selection identifiable without leaking the measured height.

The required HDF5 alignment is:

- `obs[t]` is captured before applying `actions[t]`;
- `last_action[t] = actions[t-1]` and is zero at `t=0`;
- `command_speed[t] = [vx, vy, wz]` is the command seen by the expert;
- `desired_base_height[t]` is the explicit, constant expert target height;
- the final `dones` element of every saved demonstration is true;
- `control_rate_hz = 50`.

The loader validates these conditions and fails before training if they are not
present. Collection uses a deterministic command-sampling seed and stores it in
`data.attrs["collection_seed"]`.

## Regenerate expert data

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/data/collect_data.py `
  --mode single `
  --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt `
  --skill_name walk `
  --num_envs 128 `
  --num_steps 1500000 `
  --desired_base_height 0.2932 `
  --command_profile shared_height `
  --command_resample_time_s 2.0 `
  --physics_dr_mode light `
  --seed 42 `
  --output_name walk_raw_v3.hdf5 `
  --headless
```

Repeat with `--task solo12-crouch-v0`, the crouch checkpoint,
`--skill_name crouch`, `--desired_base_height 0.1705` and the same
`--command_profile shared_height`, then merge the two files with
`scripts/diffusion_policy/data/merge_datasets.py`. Do not use a chained file:
it has no explicit per-skill height schedule in schema v3.

The spatial fields `root_pos_w` and `root_quat_w` remain in the raw schema so a
single collection can also be audited or reused later, but the command baseline
does not consume them. Instantiating `DiffuseLocoCommandDataset` performs the
mandatory schema and temporal-alignment checks before training.
