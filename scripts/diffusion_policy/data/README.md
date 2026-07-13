# Expert data for the spatial policy

The collector records aligned pairs: `obs[t]` is captured before `actions[t]`
and `last_action[t] = actions[t-1]`. It also stores world base pose for hindsight
relabeling and the expert velocity command for baseline reuse.

The v2 loader checks all of the following before training:

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
  --mode single --task solo12-sprint-v0 `
  --checkpoint checkpoints/sprint_safe.pt --skill_name sprint `
  --num_envs 128 --num_steps 1500000 `
  --command_resample_time_s 2.0 `
  --physics_dr_mode light --seed 42 `
  --desired_base_height 0.25 --command_profile shared_height `
  --output_name sprint_raw_v3.hdf5 --headless
```

Collect every expert separately. Do not use chained data in the first baseline
or first spatial run. It is reserved for a later, explicit transition ablation.

## Merge compatible single-expert files

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/merge_datasets.py `
  --inputs scripts/diffusion_policy/data/datasets/sprint_raw_v3.hdf5 `
           scripts/diffusion_policy/data/datasets/crouch_raw_v3.hdf5 `
  --output scripts/diffusion_policy/data/datasets/sprint_crouch_v3.hdf5 `
  --shuffle_seed 42
```

Schema-v3 merge rejects files without `desired_base_height`, so regenerate
sprint/crouch data before the first multi-skill baseline. The current expert
configs provide a defensible pair: sprint is trained with a 0.25 m height reward
and crouch with a 0.16 m height reward. `shared_height` keeps their velocity
commands in the same support, so height—not disjoint command ranges—must select
the skill.
