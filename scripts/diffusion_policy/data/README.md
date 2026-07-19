# Expert data for the spatial policy

The velocity-free waypoint/terminal-pose ablation is documented in
[`PATH_GUIDANCE_PHASE_A.md`](PATH_GUIDANCE_PHASE_A.md). It is additive and does
not change the existing holonomic collection or training contract.

The collector records aligned pairs: `obs[t]` is captured before `actions[t]`
and `last_action[t] = actions[t-1]`. It also stores world base pose for hindsight
relabeling and the expert velocity command for baseline reuse.

The spatial v3 loader checks all of the following before training:

- 50 Hz control and the exact alignment convention;
- equal lengths for every required field;
- no NaN/Inf values;
- `last_action[t] == actions[t-1]` over every sample;
- no intermediate `done`, normalized WXYZ quaternions and a final `done`;
- matching skill tables when more than one HDF5 is loaded.

Collection command/skill sampling is seeded and the seed is stored in the HDF5.

## Single-expert collection

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_safe.pt --skill_name walk `
  --num_envs 128 --num_steps 1500000 `
  --command_resample_time_s 2.0 `
  --physics_dr_mode light --seed 42 `
  --desired_base_height 0.2932 --command_profile shared_height `
  --output_name walk_raw_v3.hdf5 --headless
```

Collect every expert separately. Do not use chained data in the first baseline
or first spatial run. It is reserved for a later, explicit transition ablation.

## Merge compatible single-expert files

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/merge_datasets.py `
  --inputs scripts/diffusion_policy/data/datasets/walk_raw_v3.hdf5 `
           scripts/diffusion_policy/data/datasets/crouch_raw_v3.hdf5 `
  --output scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --shuffle_seed 42
```

Schema-v3 merge rejects files without `desired_base_height`, so regenerate
walk/crouch data before the first multi-skill baseline. The current archived raw
data have robust post-warmup median base heights of 0.2932 m (walk) and 0.1705 m
(crouch). Use these as fixed **posture-reference labels**, not as a claim that
the existing walk expert was trained for continuous height tracking.
`shared_height` keeps velocity commands in the same support, so the label—not
disjoint command ranges—must select the demonstrated posture.

## Spatial preflight

Generate the goal coverage table and figure before launching the spatial train:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/analyze_spatial_coverage.py `
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_v3.hdf5 `
  --output_dir scripts/diffusion_policy/data/coverage/walk_crouch_time_preview_v3 `
  --max_samples 50000 --seed 42
```

This samples the exact 0.5/1.0/1.5/2.0-second hindsight representation used by
training and reports joint XY, yaw, speed and target-height support.
