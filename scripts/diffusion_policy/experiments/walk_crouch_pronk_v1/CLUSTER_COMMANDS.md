# Transferencia y próximo train — solo después del gate de datos

**Actualización 2026-10-12:** conservar este bloque solo para transferir las
fuentes. El train actual usa la
[selección temporal corregida](../walk_crouch_pronk_speed_partition_v1/CLUSTER_COMMANDS.md).
No lanzar el SBS solapado de este documento para la ablation lento/rápido.

Durante esta preparación no se ha ejecutado Phase A ni reentrenado expertos. No
hace falta Isaac Sim para entrenar esta Phase A: el trainer offline usa PyTorch.
Tampoco necesita copiar los checkpoints RL ni sus referencias; sí todos los HDF5
a los que apunta el índice.

## 1. Transferir desde PowerShell, en la raíz local

Los paths siguientes conservan los links relativos. La carpeta raíz remota
es la habitual del proyecto. No usar `\` como continuación de línea en PowerShell.

```powershell
ssh sflores@10.4.26.33 "mkdir -p /home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned"
scp scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/walk_full.hdf5 scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/pronk_full.hdf5 scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/walk_crouch_pronk_linked.hdf5 scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/manifest.json sflores@10.4.26.33:/home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/
ssh sflores@10.4.26.33 "mkdir -p /home/sflores/i2r/sbs/walk_crouch_pronk_v1 /home/sflores/i2r/slurm_logs"
scp scratch/walk_crouch_pronk_v1/train_phase_a_cluster.sbs sflores@10.4.26.33:/home/sflores/i2r/sbs/walk_crouch_pronk_v1/
```

Crouch ya puede existir allí, pero su hash debe coincidir con el manifiesto. Si
no existe o es otro archivo, transferir el congelado local a la ubicación
esperada **tras comprobar que no se sobrescribe un artefacto diferente**:

```powershell
ssh sflores@10.4.26.33 "sha256sum /home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/crouch_phase_a_waypoint_v3.hdf5"
# Si ya coincide con 492eaddc081efc3a2554fcc83080e094a126d2268d4853b8a3e45faf4c258bbe, no copiar.
# Si falta, copiar. Si difiere, conservar el archivo remoto y resolverlo antes.
scp scripts/diffusion_policy/data/datasets/crouch_phase_a_waypoint_v3.hdf5 sflores@10.4.26.33:/home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/diffusion_policy/data/datasets/
```

## 2. Verificación en cluster

Actualizar el código de este experimento después de su commit/push autorizado;
este documento no afirma que esos pasos ya estén hechos. Usar la misma rama
`research/wct-dppo`, entorno `env_isaaclab`, y checkout limpio para la procedencia
del run. No hacer checkout forzado ni borrar modificaciones del cluster.

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git pull --ff-only origin research/wct-dppo
conda activate env_isaaclab
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/run_phase_a.py --preflight-only
```

El índice por sí solo no basta: sin las fuentes los links estarán rotos. El
hash del índice tampoco demuestra integridad de sus fuentes; por eso `verify`
comprueba los cuatro HDF5 (tres fuentes y el índice), protocolo y config por separado.
El índice final incorpora también sus hashes de fuentes en `linked_sources_json`;
guardar el manifiesto con el checkpoint permite verificar sus targets reales.
El preflight es lectura/CPU; el train solo se ejecuta dentro del job GPU. No
necesita repetir la auditoría estadística completa ya cerrada en local.

## 3. Comando de entrenamiento dentro de una GPU asignada por Slurm

El SBS privado reutiliza los recursos anteriores (Lovelace, 8 CPU, 50 GB, 72 h),
activa `env_isaaclab` y llama al launcher. No añadir checkpoint/resume/KL/DPPO:
es **imitation desde cero**. No ejecutar el train en el login node.

```bash
sbatch ~/i2r/sbs/walk_crouch_pronk_v1/train_phase_a_cluster.sbs
```

El launcher registra SHA del checkout, versiones Python/PyTorch/CUDA, job ID,
comando y manifiesto completo en `data_provenance.json` junto a los checkpoints.
Exige checkout limpio y un output nuevo. No sobrescribe runs anteriores. El
SBS está en `scratch/` local y fuera del repo remoto: no se publica en GitHub.

Equivalente directo **solo dentro de una GPU asignada**, con las mismas guardas:

```bash
python -u scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/run_phase_a.py --device cuda:0
```

Comando subyacente del trainer (el launcher lo construye; no hace falta lanzarlo
otra vez):

```bash
test ! -e scripts/diffusion_policy/runs/walk_crouch_pronk_phase_a_v1_k10 || { echo "Output already exists; do not overwrite"; exit 1; }
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/diffusion_policy/train/train.py \
  --datasets scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/walk_crouch_pronk_linked.hdf5 \
  --config scripts/diffusion_policy/train/configs/walk_crouch_hindsight_geom_profile16_a0_faithful_k10.json \
  --output_dir scripts/diffusion_policy/runs/walk_crouch_pronk_phase_a_v1_k10 \
  --run_name walk_crouch_pronk_phase_a_v1_k10 \
  --policy_backend diffusion \
  --device cuda:0 \
  --wandb_project solo12-diffusion-policy
```

Antes de lanzar, conservar fuera de Git el SHA del checkout, manifiesto,
versiones de Python/PyTorch/CUDA y stdout/stderr. No hace falta publicar SBS,
logs, HDF5 ni checkpoints en GitHub.

Después: traer `best.pt` y evaluar tres comportamientos + arranques + soporte
compartido. **No lanzar automáticamente DPPO** hasta comprobar retención del
pronk y el conflicto de altura descrito en el README.
