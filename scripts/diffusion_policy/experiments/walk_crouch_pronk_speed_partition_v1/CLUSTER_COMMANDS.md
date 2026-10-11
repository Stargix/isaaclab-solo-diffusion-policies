# Lanzamiento corregido — no usar el SBS de la mezcla solapada

Las tres fuentes originales no cambian. Si ya enviaste `walk_full.hdf5`,
`pronk_full.hdf5` a `walk_crouch_pronk_v1_aligned/` y crouch a su ubicación
histórica, **no repetir su transferencia**. Los hashes los verifica el launcher.
Si todavía no los enviaste, usar el bloque de fuentes del documento
`../walk_crouch_pronk_v1/CLUSTER_COMMANDS.md`, pero no lanzar su SBS anterior.

## PowerShell local

```powershell
cd "C:\Users\11ser\OneDrive\Escritorio\Coding\Projectes\Isaac\isaaclab-solo"
ssh sflores@10.4.26.33 "mkdir -p /home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/walk_crouch_pronk_speed_partition_v1 /home/sflores/i2r/sbs/walk_crouch_pronk_speed_partition_v1 /home/sflores/i2r/slurm_logs"
$wcpSelected = "scripts/diffusion_policy/data/datasets/walk_crouch_pronk_speed_partition_v1"
scp "$wcpSelected/walk_crouch_pronk_speed_partition.hdf5" "$wcpSelected/manifest.json" sflores@10.4.26.33:/home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/walk_crouch_pronk_speed_partition_v1/
scp scratch/walk_crouch_pronk_speed_partition_v1/train_phase_a_cluster.sbs sflores@10.4.26.33:/home/sflores/i2r/sbs/walk_crouch_pronk_speed_partition_v1/
```

Solo transferir estos archivos definitivos; no los índices intermedios,
pilotos, plots, checkpoints expertos ni logs. Mantener la estructura de
carpetas para que resuelvan los links relativos.

## Cluster, después de publicar el código

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git pull --ff-only origin research/wct-dppo
conda activate env_isaaclab
python scripts/diffusion_policy/experiments/walk_crouch_pronk_speed_partition_v1/train.py --preflight-only
```

Si el preflight pasa y el checkout está limpio:

```bash
sbatch ~/i2r/sbs/walk_crouch_pronk_speed_partition_v1/train_phase_a_cluster.sbs
```

Output nuevo:
`scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/`.
Phase A desde cero, profile16, K=10, quadruped, batch4096, 50 epochs, seed42.
SBS privado: Lovelace, 8 CPU, 50 GB, 72 h. No necesita Isaac Sim ni importar
las tareas del experto para entrenar offline. No hay fast-trot, resume ni DPPO.

Conservar `data_provenance.json` y los logs. Al terminar, traer `best.pt` y
`config.json`; evaluar la retención antes de lanzar otro entrenamiento.
