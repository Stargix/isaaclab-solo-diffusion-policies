# Evaluación de LocoDiff

La referencia científica replantea en cada step y usa tres pasos del solver:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_locodiff/evaluate_policy.py `
  --task solo12-v0 `
  --checkpoint scripts/baseline_locodiff/runs/locodiff_command_skill_sde_v1/best.pt `
  --output_dir scripts/baseline_locodiff/evaluations/locodiff_command_skill_sde_v1 `
  --num_inference_steps 3 --exec_horizon 1 `
  --repeats 3 --duration_s 15 --settling_s 2 --latency_samples 100 --headless
```

Las alturas extremas del grid equivalen exactamente a walk y crouch. Las
intermedias mezclan los dos one-hot y se reportan como prueba de interpolación,
no como supervisión de altura. `benchmark_runtime.py` queda reservado para
ablation posterior; no debe usarse para cambiar la configuración antes de saber
si la reproducción de referencia es estable.
