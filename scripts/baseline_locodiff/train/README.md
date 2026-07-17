# Baseline LocoDiff sin return guidance

Esta variante reproduce el modelo generativo de *Offline Adaptation of
Quadruped Locomotion using Diffusion Models* y elimina únicamente la rama de
return/reward conditioning. El condicionamiento que queda es el del expert
model de la tabla I: historial de estado, comando real de velocidad y habilidad.

Contrato de schema 4:

- estado: posición y velocidad articular, velocidad lineal y angular del chasis
  en body frame y gravedad proyectada;
- historial de ocho estados, sin historial de acciones;
- condición por paso `[vx, vy, wz, walk, crouch]`;
- target puramente futuro `a[t:t+16]`;
- Transformer decoder con self-attention causal sobre acciones y cross-attention
  a todo el historial observado;
- precondicionamiento EDM, sigmas log-logísticos y loss directa sobre `x0`;
- probability-flow ODE con tres pasos; por defecto Euler;
- se ejecuta `a[t]` y se replantea en el siguiente control step.

Los checkpoints de schema 3 se rechazan. También se rechazan los HDF5 antiguos
que no guardan `base_lin_vel`: reconstruir twist por diferencias finitas no sería
la misma entrada del paper.

## Comprobación previa

Después de generar y unir los datos, ejecutar primero:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/train/train.py `
  --config scripts/baseline_locodiff/train/configs/paper_command_skill_sde.json `
  --datasets scripts/diffusion_policy/data/datasets/locodiff_walk_crouch_v1.hdf5 `
  --output_dir scripts/baseline_locodiff/runs/_preflight_v1 `
  --run_name locodiff_preflight_v1 `
  --step_stride 50 --batch_size 16 --max_stats_samples 2000 `
  --preflight_only --no_amp
```

El preflight valida datos, split, normalización, formas, loss, backward y un
muestreo completo de tres pasos. No entrena epochs.

## Entrenamiento

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/train/train.py `
  --config scripts/baseline_locodiff/train/configs/paper_command_skill_sde.json `
  --datasets scripts/diffusion_policy/data/datasets/locodiff_walk_crouch_v1.hdf5 `
  --output_dir scripts/baseline_locodiff/runs/locodiff_command_skill_sde_v1 `
  --run_name locodiff_command_skill_sde_v1 `
  --wandb_project solo12-diffusion `
  --symmetry_mode quadruped
```

No se ofrece una garantía de locomoción cerrada antes de entrenar: eso depende
de optimización y cobertura de datos. Sí quedan cerrados los errores de contrato
que invalidaban el run anterior. La primera evaluación debe usar `exec_horizon=1`
y tres solver steps; cambiar esos dos valores ya sería una ablation.

El paper no publica anchura, número de capas ni parámetros de la log-logística.
Los valores del JSON son por tanto explícitos y versionados, pero no pueden
presentarse como hiperparámetros oficiales. La otra diferencia deliberada es
Solo12 a 50 Hz frente a ANYmal a 25 Hz.

## Ablación: altura continua

`configs/velocity_height_sde.json` conserva el mismo SDE/EDM, estado de 33D,
horizonte, arquitectura y datos, pero reemplaza el one-hot por
`[vx, vy, wz, desired_base_height]`. Los HDF5 ya guardan esa columna: no hay
que recolectar de nuevo. El checkpoint se etiqueta
`locodiff_sde_velocity_height_v1` y queda separado de la reproducción fiel al
paper.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/train/train.py `
  --config scripts/baseline_locodiff/train/configs/velocity_height_sde.json `
  --datasets scripts/diffusion_policy/data/datasets/locodiff_walk_crouch_v1.hdf5 `
  --output_dir scripts/baseline_locodiff/runs/locodiff_velocity_height_sde_v1 `
  --run_name locodiff_velocity_height_sde_v1 `
  --wandb_project solo12-diffusion `
  --symmetry_mode quadruped
```
