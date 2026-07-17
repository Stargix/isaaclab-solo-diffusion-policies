# Datos de LocoDiff

Se recogen por separado un millón de pasos de walk y un millón de crouch, como
en el paper. Cada episodio pertenece a una sola habilidad. `desired_base_height`
se conserva como metadato del profesor, pero el modelo no lo consume: la
habilidad se representa con one-hot.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/data/collect_data.py `
  --mode single --task solo12-v0 `
  --checkpoint checkpoints/walk_final.pt --skill_name walk `
  --desired_base_height 0.2932 --command_profile shared_height `
  --num_envs 128 --num_steps 1000000 --command_resample_time_s 2.0 `
  --physics_dr_mode light --seed 42 --output_name locodiff_walk_v1.hdf5 --headless

conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/data/collect_data.py `
  --mode single --task solo12-crouch-v0 `
  --checkpoint checkpoints/crouch_final.pt --skill_name crouch `
  --desired_base_height 0.1705 --command_profile shared_height `
  --num_envs 128 --num_steps 1000000 --command_resample_time_s 2.0 `
  --physics_dr_mode light --seed 43 --output_name locodiff_crouch_v1.hdf5 --headless

conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/merge_datasets.py `
  --inputs scripts/baseline_locodiff/data/datasets/locodiff_walk_v1.hdf5 `
           scripts/baseline_locodiff/data/datasets/locodiff_crouch_v1.hdf5 `
  --output scripts/diffusion_policy/data/datasets/locodiff_walk_crouch_v1.hdf5 `
  --shuffle_seed 42
```

El loader comprueba alineación temporal, finitud, `last_action[t]=actions[t-1]`,
50 Hz, habilidades por episodio y la presencia de twist lineal y angular. No se
usan secuencias chained ni recompensas.
