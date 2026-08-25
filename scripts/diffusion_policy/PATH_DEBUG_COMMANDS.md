# Rutas geométricas para visualizar la política

Estas rutas son diagnósticos de reproducción en Isaac Sim. Los archivos `.npy`
generados por `generate_test_path.py` están en el marco local del robot: el
primer punto es `(x, y) = (0, 0)` y el cargador de `play_policy.py` lo ancla a
la posición y yaw iniciales del robot mediante `--path_file_frame robot`.

Las alturas `z` son valores absolutos de la base del robot, no desplazamientos.
No usar `--headless` si se quiere ver el viewport.

## Generar la ruta con curvas y vértices en V

Esta ruta mide 5.5 m e intercala curvas suaves con dos V de esquinas bruscas
(aproximadamente 100° y 70° de cambio de rumbo). Los vértices no se redondean,
para comprobar cambios de dirección no graduales.

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/generate_test_path.py `
  --shape sharp_v_curves `
  --length 5.5 `
  --step_size 0.05 `
  --z_start 0.2932 `
  --z_end 0.2932 `
  --transition_x 999 `
  --output scripts/diffusion_policy/paths/_debug_sharp_v_curves.npy
```

## Visualizar la ruta V + curvas a 0.5 m/s

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/play_policy.py `
  --task solo12-v0 `
  --checkpoint checkpoints_iri/checkpoints_dppo/dppo_path_4.pt `
  --path_file scripts/diffusion_policy/paths/_debug_sharp_v_curves.npy `
  --path_file_frame robot `
  --desired_speed 0.5 `
  --num_envs 1 `
  --visualize_path `
  --visualize_goal `
  --visualize_preview `
  --visualize_actual_path `
  --visualize_height
```

## Otras rutas disponibles

Cambiar únicamente el archivo de `--path_file` para probar las variantes
anteriores:

```text
scripts/diffusion_policy/paths/_debug_s_curve_multiheight.npy
scripts/diffusion_policy/paths/_debug_circle_multiheight.npy
scripts/diffusion_policy/paths/_debug_right_angle_multiheight.npy
scripts/diffusion_policy/paths/_debug_mixed_turn_curve.npy
scripts/diffusion_policy/paths/_debug_sharp_v_curves.npy
```

Para una prueba geométrica limpia, usar el archivo `sharp_v_curves`, que tiene
altura constante. Las variantes `*_multiheight` añaden además cambios de altura
y sirven para probar simultáneamente geometría y transición de postura.

Si un `.npy` ya contiene coordenadas absolutas del mundo, usar
`--path_file_frame world`; para los archivos de esta carpeta se debe usar
`robot`.
