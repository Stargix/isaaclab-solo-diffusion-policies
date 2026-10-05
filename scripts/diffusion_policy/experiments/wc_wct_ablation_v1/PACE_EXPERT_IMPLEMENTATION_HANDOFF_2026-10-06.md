# Handoff: experto pace de SOLO12, sin cambiar el controlador de tarea

Fecha: 2026-10-06. Estado: **implementado y validado localmente; ver sección 13**.
Base revisada: `research/wct-dppo`, HEAD `e37979c`.

## 1. Qué implementar y por qué

Implementar y preparar para entrenamiento **un experto de locomoción pace**:
parejas ipsilaterales FL–RL y FR–RR, alternando entre lados. El experto walk
actual ya presenta coordinación diagonal; otra política que sólo aumente la
velocidad no proporciona el contraste de movimientos que queremos investigar.

Este experto será una fuente de demostraciones para una futura biblioteca
walk+crouch+pace. **No** será una high-level policy ni un selector durante la
ejecución de la difusión. El objetivo final sigue siendo conditioning de
geometría, altura y velocidad media, sin gait ID.

Reparto deseado confirmado por el usuario: pace aporta la habilidad lenta y
se reaprovecha el walk existente para la rápida. Mantener el rango de train
del experto pace en `vx=(-1.0,1.0)`; en la futura colección se recortan sus
comandos al soporte lento que pase la evaluación. No reducir el rango de train
a 0.6 m/s sólo por ese papel en la biblioteca. El recorte de datos queda
pendiente de medir la calidad del gait y el tracking.

La revisión de simetría LR no ha justificado sustituir los actores del paper.
Conservarlos. El siguiente experimento añade una fuente de movimiento, no intenta
reparar aquellos pilotos ni demostrar superioridad frente a RL de tarea desde cero.

Documento de alcance vigente:
`PATH_CONDITIONED_MULTISKILL_PLAN_2026-10-04.md`. Las prioridades de documentos
anteriores que pedían PPO de tarea desde cero no rigen este trabajo.

### Alcance cerrado de esta implementación

- Nuevo task `solo12-pace-v0`, config, runner PPO, integración mínima en play.
- Reward pace probado matemáticamente, warm-start estricto y smoke real.
- Extensión pequeña del evaluador existente para comparar walk/pace.
- README y launcher Slurm listos para revisión y posterior lanzamiento.

**No implementar todavía** colección, nueva Phase A, nuevos rewards DPPO,
generadores de rutas, gait IDs, otra arquitectura, sim2sim o modificaciones del
paper. No ejecutar un train largo ni commit/push sin autorización posterior.
No cambiar de branch, borrar archivos o limpiar el worktree para implementar esto.

## 2. Lectura obligatoria y reutilización

Leer estos archivos antes de editar:

1. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py`
2. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env_cfg.py`
3. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_flying_trot_env.py`
4. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/rsl_rl_ppo_cfg.py`
5. `source/scripts/skrl/solo12_symmetry.py`
6. `scripts/reinforcement_learning/rsl_rl/warm_start.py`, `train.py`, `play.py`
7. `scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py`
8. `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py`, clase `GaitReward`.

Reutilizar `Solo12Env` y su tracking, acciones, reset y contacto de base.
Reutilizar el runner simétrico y el warm-start existente. Flying-trot sirve como
ejemplo de registro/config, **no** como padre del nuevo entorno ni como reward
que copiar íntegro. No reutilizar el reloj de bound, su observación de 50D ni
su gating multiplicativo de toda la tarea.

El archivo `diffusion_policy/evaluation/pace_intervention.py` analiza el ritmo
temporal de la tarea: no es el experto de gait pace y no debe reutilizarse por
coincidir el nombre.

## 3. Diseño que debe mantenerse

### 3.1 Contrato del actor

- Observaciones y orden: los mismos **48 valores** de `solo12-v0`.
- Acciones: las mismas 12 acciones y mismo orden articular.
- Sin reloj, embeddings, estados extra, altura nueva como etiqueta o skill ID.
- Actor/critic MLP `[256, 128, 64]`, ELU y normalización existentes.
- `kp=9.0`, `kd=0.2`, action scale `0.25`; misma articulación, límites y masas.
- Física `dt=1/200`, decimation `4`: control a 50 Hz.
- Altura erguida deseada `0.2932 m`, comparable a walk.

No cambiar la pose de reset para conseguir visualmente otro gait. La altura
objetivo no equivale a la altura de root usada al reset: conservar el arranque
físico existente y evaluar su estabilización por separado.

### 3.2 Inicialización

Partir del actor y normalizador de `checkpoints/walk_final.pt` mediante
`--warm_start_checkpoint`; **critic, Adam y ruido de exploración nuevos**.
No `--resume`, ni usar `--checkpoint` como sustituto del warm-start.

Archivo local verificado:

```text
SHA256 e42e1ddd6645ec0b1ddb622cc064e84a521f57fbb3c244259dbae1675db9111a
actor.0.weight: [256, 48]
actor.2.weight: [128, 256]
actor.4.weight: [64, 128]
actor.6.weight: [12, 64]
```

No fallback silencioso a otro checkpoint o a inicialización aleatoria. Si el
cluster usa otro archivo, parar y explicar la diferencia. Warm-start facilita
adaptar una locomoción existente; no garantiza escapar de su coordinación diagonal.

### 3.3 Simetría del experto, no del task DPPO

Reutilizar las cuatro transformaciones existentes del runner quadruped:
identidad, izquierda/derecha, delante/detrás y composición. Este experto recibe
velocidades positivas **y negativas**, no tiene objetivo de heading absoluto,
y ambas parejas pace se conservan bajo esas permutaciones.

Comprobar signos de velocidades polares, velocidades angulares axiales,
comandos, acciones y offsets articulares. No crear otra copia de esas matrices.
La pose por defecto y los límites deben ser compatibles con las transformaciones.
Las pruebas deben verificar esta compatibilidad, no limitarse a afirmar que el
robot es simétrico. Si falla, reportar y parar: no corregir todo el módulo común
ni cambiar silenciosamente el experimento.

Esto **no reactiva augmentation online en DPPO**. En una futura tarea dirigida
o forward-only, la simetría delante/detrás no se deduce de este experto.

## 4. Reward: fórmulas y límites de la fundamentación

El mecanismo de sincronización usa la implementación oficial `GaitReward` de
Isaac Lab: tiempos actuales de contacto/vuelo, sincronización de las parejas y
oposición entre parejas. Es clock-free y no fija una frecuencia o duty factor.
Adaptar las operaciones tensoriales a nuestro `DirectRLEnv`; no introducir un
`CommandManager` ni instanciar `ManagerTermBase` aquí.

La topología instantánea, el límite de duración y los pesos siguientes son
**decisiones de ingeniería iniciales**, no constantes óptimas demostradas por
un paper. Deben validarse con contactos y tracking. No prometer un gait estable
en un único entrenamiento sólo por tener estos términos.

### 4.1 Orden de patas y sincronización

Resolver cuerpos con `preserve_order=True`, exigir exactamente:

```text
[FL_calf, FR_calf, RL_calf, RR_calf] = [0, 1, 2, 3]
sync:  (0,2), (1,3)
async: (0,1), (0,3), (2,1), (2,3)
```

Usar `current_air_time` y `current_contact_time`, no `last_air_time` ni tiempos
derivados del máximo de un historial. Para cada pata, sean `a_i` y `c_i` esos
tiempos en segundos. Con `M=0.20 s`:

```text
E_sync(i,j) = min((a_i-a_j)^2, M^2) + min((c_i-c_j)^2, M^2)
E_async(i,j) = min((a_i-c_j)^2, M^2) + min((c_i-a_j)^2, M^2)
T = exp(-(sum(E_sync) + sum(E_async)) / timing_error_scale_s2)
timing_error_scale_s2 = 0.10
```

El denominador tiene unidades de segundos cuadrados. No volver a elevarlo al
cuadrado: el parámetro upstream llamado `std` ya divide errores cuadrados.
Sumar los exponentes equivale al producto de sus seis scores.

### 4.2 Evitar reward por estar quieto o sostener sólo un lado

Los timers pueden estar todos a cero y dar `T=1`. Añadir un factor de topología
instantánea. Fuerza de contacto **actual**, no máximo histórico:

```text
p_i = sigmoid((norm(net_forces_w[i]) - 1.0 N) / 0.25 N)
sync = 1 - 0.5*(abs(p_0-p_2) + abs(p_1-p_3))
opposition = abs(0.5*(p_0+p_2) - 0.5*(p_1+p_3))
P = clamp(sync*opposition, 0, 1)
```

Para contactos binarios, pace `1010` y `0101` puntúa 1. Diagonales `1001` y
`0110`, bound `1100`/`0011`, cuatro apoyos y vuelo completo puntúan 0.
Las probabilidades suaves se aproximan a esos casos; no exigir igualdad exacta
para fuerzas finitas. No añadir una recompensa de vuelo que incentive saltos.

Una pareja inmóvil apoyada y la otra suspendida podría recibir sincronización.
Por ello añadir:

```text
cycle_ok = max_i(max(a_i,c_i)) <= max_mode_time_s
max_mode_time_s = 0.50
```

Es un guard inicial contra modos estáticos prolongados, **no** una frecuencia
prescrita ni un resultado de la literatura. Puede limitar cadencias muy lentas;
no afirmar cobertura arbitraria a bajas velocidades antes de medirla. En stop
el término gait está desactivado. Registrar cuántos samples elimina este guard.

### 4.3 Integración aditiva con la tarea

Con velocidades corporales reales `v` y comando `u`:

```text
q_xy = exp(-sum((u_xy-v_xy)^2) / 0.30^2)
q_y  = exp(-((u_y-v_y)/0.12)^2)
moving = norm(u_xy) > 0.10
r_pace = step_dt * 1.0 * q_xy * T * P * moving * cycle_ok
r_lateral_extra = step_dt * 0.65 * q_y
r_total = r_base + r_pace + r_lateral_extra
```

`r_base` incluye ya `4*q_xy`, tracking yaw, altura y regularización. El extra
lateral estrecha el tracking de `vy` sin sustituir `q_xy`. **No** multiplicar
el reward entero por gait, estabilidad o tracking: el robot debe poder mejorar
el movimiento aunque todavía no sincronice pace. No penalización de tiempo ni
premio terminal de ruta en este experto de velocidades.

### 4.4 Configuración inicial exacta

Asignar sólo en `Solo12PaceEnvCfg`; no cambiar defaults de otros tasks.

| Campo existente | Valor |
|---|---:|
| `tracking_std` | `0.30` |
| `track_lin_vel_xy_reward_scale` | `4.0` |
| `track_ang_vel_z_reward_scale` | `0.65` |
| `base_z_desired` | `0.2932` |
| `base_height_exp_scale` | `1 / 0.035**2` |
| `track_base_height_reward_scale` | `0.50` |
| `lin_vel_z_reward_scale` | `-0.05` |
| `ang_vel_xy_reward_scale` | `-0.05` |
| `joint_torque_reward_scale` | `-0.10e-3` |
| `action_rate_reward_scale` | `-0.02` |
| `undesired_contact_reward_scale` | `-2.25` |
| `base_tilt_penalty_reward_scale` | `-0.25` |
| `foot_contact_reward_scale` | `-0.5e-3` |
| `joint_accel_reward_scale` | `0.0` |
| `feet_air_time_reward_scale` | `0.0` |
| `force_transmited_through_joints_reward_scale` | `0.0` |

Mantener el nombre histórico `force_transmited...` aunque tenga una errata.
Los parámetros nuevos anteriores deben ser configurables con prefijo `pace_`,
unidades en sus nombres/docstrings y validación de denominadores positivos.
La sigma de yaw utiliza el mismo `tracking_std` existente; no inventar una sigma
distinta mientras se llama al reward base.

No añadir clearance calculado con el centro de `calf`: no es la altura del
punto distal del pie. Mantener el detector de caída/contacto de base existente.

### 4.5 Contabilidad y logs

Llamar `super()._get_rewards()` exactamente una vez. Añadir dos claves a
`_episode_sums` sin reemplazar las claves del padre: `pace_gait` y
`pace_lateral_tracking`. Sumar únicamente sus extras a esos acumuladores y a
`_episode_reward_sums`; devolver base+extras. No multiplicar `dt` dos veces.

Actualizar `self.extras['log']` mediante `update`, preservar las claves base y
sobrescribir `RewardsPerStep/total` con el total real. Logs mínimos adicionales:
score temporal, topología, fracción bloqueada por duración, errores absolutos
vx/vy/yaw, altura y contribuciones pace/lateral. No obtener observaciones para
logging dentro del reward: `_get_observations()` modifica previous-actions y
podría borrar la penalización de action rate.

El logger histórico de episodios puede mostrar valores absolutos de términos.
No cambiarlo globalmente; los logs por paso nuevos deben conservar el signo.

## 5. Distribución de comandos y PPO

Sin curriculum inicial: separar adquisición de gait y robustez. No avances de
curriculum disparados por un return al que acabamos de añadir gait reward.

```text
command_lin_vel_x_range = (-1.0, 1.0)
command_lin_vel_y_range = (-0.15, 0.15)
command_ang_vel_z_range = (-0.35, 0.35)
command_resampling_time_s = 3.0
standing_env_prob = 0.10
opposite_direction_cmd_prob = 0.0
max_velx_range_curriculum = ()
forces_applied_to_base_curriculum = (0.0,)
base_push_force_xy_range = (0.0, 0.0)
base_push_force_z_range = (0.0, 0.0)
enable_observation_corruption = False
events = None
actuation_delay_range = (0, 0)
flexed_initial_joint_pos_noise_range = (-0.02, 0.02)
reset_base_lin_vel_range = (-0.05, 0.05)
reset_base_ang_vel_range = (-0.05, 0.05)
tricky_terrain = False
episode_length_s = 20.0
```

Terreno plano generado como trimesh local (sin referencia a USD de Nucleus),
200 x 200 m, con orígenes de entorno en la cuadrícula del scene espaciados
2.5 m. El tamaño cubre la cuadrícula de 4096 entornos. Copias independientes de configs mutables:
instanciar pace no debe modificar robot/terrain/runner de walk.

Runner nuevo heredando `Solo12PPORunnerWithSymmetryCfg`:

```text
num_steps_per_env = 32
max_iterations = 2500
save_interval = 50
experiment_name = solo12_rsl_rl_pace_runs
run_name = pace_v1_seed42
policy.init_noise_std = 0.35
algorithm.learning_rate = 3e-4
algorithm.schedule = adaptive
algorithm.desired_kl = 0.01
algorithm.entropy_coef = 0.002
```

Resto heredado: 5 epochs, 4 minibatches, clip 0.2, gamma 0.99, lambda 0.95,
max grad norm 0.5 y normalización. No reescribir PPO ni crear un optimizador.
Son parámetros del primer experimento, no garantía de convergencia ni un sweep.

## 6. Mapa de cambios mínimo

### Archivos nuevos

1. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/pace_gait.py`
   Funciones tensoriales puras para score temporal, topología y guard. Sólo
   PyTorch; sin imports de Isaac/Kit/Gym. Aceptar batches y dispositivo arbitrarios.
2. `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_pace_env.py`
   `Solo12PaceEnvCfg`, `Solo12PacePPORunnerCfg`, `Solo12PaceEnv(Solo12Env)`.
   Resolver pies, extender logs/acumuladores y sumar dos rewards; no duplicar
   el reward base, reset o stepping.
3. `scripts/reinforcement_learning/rsl_rl/tests/test_pace_gait.py`
   Tests puros descritos abajo. Import por filepath si el paquete activa Isaac.
4. `scripts/reinforcement_learning/rsl_rl/experiments/pace_v1/README.md`
   Hipótesis, receta, límites, comandos y cómo seleccionar/evaluar checkpoints.
5. `scripts/reinforcement_learning/rsl_rl/experiments/pace_v1/train_cluster.sbs`
   Launcher reproducible, sin modificar scripts globales del cluster.

### Modificaciones estrechas

- `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/__init__.py`: registrar
  `solo12-pace-v0` con entry points de entorno, config y runner nuevos. Usar el
  registro por strings existente. No reparar registros históricos ajenos.
- `scripts/reinforcement_learning/rsl_rl/play.py`: añadir pace a la allowlist
  de `--command_ui` y límites vx ±1, vy ±0.15, yaw ±0.35. Reutilizar ventana y
  refresco actuales; no construir otra UI.
- `scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py`: añadir
  preset `walk_pace`, `--pace_checkpoint`, `--lateral_speed` y `--yaw_rate`
  (los dos últimos default 0). Las salidas antiguas mantienen nombres/contratos;
  las métricas nuevas y correcciones se identifican como protocolo pace v1.

No modificar `warm_start.py` o `train.py` salvo incompatibilidad demostrada y
reportada. No añadir dependencias. No tocar ningún `.pt`, dataset, reward DPPO,
modelo de difusión o script de colección.

## 7. Comparación walk/pace sin artefactos

### Entorno común

Los dos actores se ejecutan en **`solo12-pace-v0`, 48D**, bajo el mismo entorno físico,
PD y suelo plano generado localmente, sin ruido/DR/delay/push/curriculum. El reward
del wrapper no afecta la inferencia de ninguno de los actores. Así no confundimos actor
con física. Grupos de 16 entornos por actor, seed 42, arranque 2 s y
medición 8 s. Comandos fijos más largos que el episodio; no resampling oculto.

Validar normalización e inferencia del `_CheckpointActor` del evaluador contra
la función de inference del runner instalado con los mismos inputs. No asumir
que una epsilon copiada a mano reproduce cualquier versión de RSL-RL.

### Correcciones necesarias para este preset

1. Tras reset y escribir el comando, refrescar observaciones **antes** de la
   primera inferencia. No permitir que el primer action vea un comando aleatorio.
2. Supervivencia desde el primer reset. No reiniciar el mask de fallos al
   terminar warmup ni contar como sano un robot que cayó y se autoreseteó.
3. No registrar post-reset como continuación del ciclo previo. Mantener
   validez persistente, marcar/truncar fallos y contar transiciones sólo entre
   muestras consecutivas válidas.
4. Orden explícito de patas, mismo threshold 1 N para ambos actores.
5. Hz de contactos **por pata**: si se agregan cuatro patas dividir también
   por cuatro. No etiquetar el conteo total de cuatro sensores como frecuencia
   de ciclo; touchdowns por segundo y cambios de contacto no son lo mismo.
6. Correlación indefinida de contacto constante -> null/NaN con soporte
   declarado, no cero ni evidencia de un gait perfecto.
7. Comandos de lateral/yaw distintos de cero -> error respecto al comando,
   no penalización respecto a cero.

Si una corrección compartida cambia métricas antiguas, mantener el campo viejo
o versionar la salida; no sobrescribir evaluaciones archivadas. No hace falta
una refactorización completa del evaluador para corregir el preset nuevo.

### Métricas y plots suficientes

- Supervivencia, fallos en arranque/medición y duración válida.
- RMSE vx/vy/yaw; altura media/MAE respecto a 0.2932 m; velocidad realizada.
- Duty factor y touchdowns/ciclos por pata; Jaccard de FL–RL/FR–RR frente a
  FL–RR/FR–RL; fracciones de patrones binarios ipsilateral y diagonal.
- Correlaciones sólo como medida secundaria, junto a raster de contactos.
- Figura compacta: raster walk, raster pace y velocidades reales/comandadas.
  Usar env 0 predeclarado; si falla mostrarlo truncado. No escoger el mejor
  superviviente para que parezca pace. Los números agregan todos los entornos.
- JSON con config, seeds, checkpoints/hash, unidades, número de entornos válidos;
  NPZ con contactos, velocidad, altura, comandos, tiempo y masks para replotear.

No distinguir gait sólo por velocidad pedida. Una comparación a velocidades
realizadas muy diferentes no certifica conservación de otro patrón. No usar un
único valor de correlación o una imagen seleccionada como gate.

### Banco compacto y decisión antes de colección

Evaluar comandos `(vx,vy,wz)`:

```text
(0.4,0,0), (0.7,0,0), (1.0,0,0), (0,0,0), (-0.7,0,0)
(0.7,+0.10,0), (0.7,-0.10,0), (0.7,0,+0.25), (0.7,0,-0.25)
```

No es una batería publicable ni requiere nuevas seeds. Es el gate de adquisición.
Targets de aceptación iniciales, declarados como ingeniería:

- A 0.4 y 0.7: al menos 95% supervivencia (con 16 esto exige 16/16), RMSE vx
  ≤0.10 m/s y sesgo de altura media ≤2.5 cm. Informar MAE además del sesgo.
- Jaccard ipsilateral medio supera al diagonal medio en ≥0.20 en esos dos
  comandos; cada pata tiene ≥3 touchdowns durante los 8 s. El raster debe
  confirmar alternancia, no un apoyo lateral estático o pares suspendidos.
- Steering/reverse: mismos límites vx y seguridad, RMSE vy ≤0.06 m/s, yaw
  ≤0.15 rad/s; mostrar ambos signos. No exigir el mismo Jaccard en pleno giro
  que en marcha recta, pero tampoco esconder que pierde la coordinación.
- Stop: velocidad planar media en los últimos 2 s ≤0.08 m/s, sin caída.
- 1.0 m/s es una prueba del borde. Si falla pero 0.4/0.7 pasan, no subir reward
  automáticamente: la colección futura se limita al soporte validado.

No recoger datos si no hay pace reconocible o si el tracking/seguridad fallan
en el soporte principal. Entregar el fallo observable antes de inventar otra
versión. Estos límites no prometen soporte continuo ni robustez sim2real.

## 8. Tests y orden de implementación

### Paso A — helpers y config

Tests puros, CPU y CUDA si está disponible:

- Pace ideal frente a diagonal, bound, todos apoyados/todos en vuelo.
- Fórmula temporal igual a producto upstream, clipping correcto, parámetros
  positivos, batch 1/N, finite/range [0,1], device/dtype preservados.
- Timers cero no generan bonus completo; modo fijo de 1 s bloqueado; stop
  desactiva sólo gait, no reward base de tracking.
- Permutaciones LR/FB conservan los scores pace; transformación doble retorna
  observaciones/acciones originales. Comandos y offsets respetan signos reales.
- Peso y `dt` aplicados una sola vez. No doble acumulación de episodios.

No importar simulator para tests matemáticos. Los tests de config/runner sí
pueden requerir AppLauncher. Instanciar walk antes/después de pace y comprobar
que sus defaults/actuadores no cambiaron por alias de objetos mutables.

### Paso B — integración real y warm-start

- Registro carga el nuevo task y ambos actores mantienen 48D/12D.
- Warm-start copia actor/normalizador exactamente; critic/noise siguen nuevos.
- Feet IDs/orden correctos y sensor con `track_air_time=True`.
- Smoke real de 8 envs, 2 iteraciones: obs/reward/loss finitos, reset y guardado.
- Reward total/log/episode sums coherentes con base+dos extras.
- Preservar import RSL-RL/tensordict antes de Kit en Windows: orden ya resuelto
  en scripts existentes. No introducir imports tardíos que recuperen el crash.

No presentar mocks o tests tensoriales como prueba de un train en Isaac Lab.
Si el smoke no puede correr, explicar el error; no marcarlo aprobado.

### Paso C — evaluador y launcher

Tests de comando inicial fresco, máscara persistente en warmup, Hz por pata y
correlación constante indefinida. Puede probarse el preset con walk en ambos
slots para verificar paridad; **no** presentarlo como validación de pace.
Después preparar README/SBS, entregar diff y resultados, y parar para revisión.

Sólo después de revisar el código se autorizarán push/train largo. Tras train,
aplicar el banco de la sección 7 al checkpoint candidato antes de colección.

## 9. Comandos que deben funcionar después de implementar

Estos comandos son el contrato del código pendiente, **no ejecutarlos hoy como
si el task existiese**. En PowerShell usar una línea o backticks, no `\` de Bash.

### Local: tests, smoke y UI

Para comandos `isaaclab.bat`, activar primero el entorno en esta terminal:

```powershell
conda activate env_isaaclab
```

```powershell
& C:\Users\11ser\miniconda3\envs\env_isaaclab\python.exe -m unittest discover -s scripts/reinforcement_learning/rsl_rl/tests -p 'test_pace*.py'
```

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task solo12-pace-v0 --warm_start_checkpoint checkpoints/walk_final.pt --num_envs 2 --max_iterations 1 --seed 42 --logger tensorboard --headless --device cuda:0
```

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-pace-v0 --checkpoint "<checkpoint-pace-real>" --num_envs 1 --command_ui --command 0.7 0.0 0.0
```

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py --comparison walk_pace --walk_checkpoint checkpoints/walk_final.pt --pace_checkpoint "<checkpoint-pace-real>" --speed 0.7 --lateral_speed 0 --yaw_rate 0 --num_envs 16 --warmup_s 2 --duration_s 8 --output_dir scratch/pace_v1/vx_070 --headless --device cuda:0
```

Documentar el banco anterior con una pequeña lista/bucle de nueve comandos y
directorios distintos. No otro framework ni parser de configuración nuevo.

### Cluster: launcher futuro

Raíz `/home/sflores/i2r/isaaclab-solo-diffusion-policies`.
SBS: 1 nodo/task/GPU lovelace, 8 CPUs, 50G, 24 h; logs bajo
`/home/sflores/i2r/slurm_logs`. Activar `env_isaaclab` como launchers actuales.
EGL/PHYSX/OMNI GPU index 0 dentro de dispositivos visibles, threads CPU=1,
TMPDIR bajo `${HOME}/i2r/tmp` y PYTHONPATH del repo. No cambiar CUDA_VISIBLE_DEVICES
asignado por Slurm. Heredar configuración SSL existente si necesaria para W&B.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task solo12-pace-v0 \
  --warm_start_checkpoint checkpoints/walk_final.pt \
  --num_envs 4096 --max_iterations 2500 --seed 42 \
  --run_name pace_v1_seed42 --logger wandb \
  --log_project_name solo12-pace --headless --device cuda:0
```

SBS debe exigir checkpoint existente y SHA esperado, guardar SHA Git,
`git status` (incluyendo untracked), hash del checkpoint y comando. No llamar
limpio a un repo con código untracked por verificar sólo `git diff`. Si hay
suciedad relevante a este experimento, fallar con diagnóstico; no auto-stash,
checkout, add o borrado. README aclara que launcher/docs ajenos no demuestran
cambio de algoritmo, pero su procedencia sigue registrándose.

Salida estándar de RSL-RL:
`logs/rsl_rl/solo12_rsl_rl_pace_runs/<timestamp>_pace_v1_seed42/`.
No asumir que existe `best.pt`: comprobar qué guarda la versión instalada y
usar `model_<iter>.pt` real. Guardar configs YAML con mecanismos ya existentes.
El candidato debe seleccionarse por gait+tracking observado, no sólo máximo
return. No crear un selector de checkpoint complejo para este primer experto.

## 10. Después del experto: contexto que no implementar ahora

Si el experto pasa el gate:

1. Añadir un preset nuevo de colección, preservando los grupos walk/crouch y
   la receta A0 de `../phase_a_fast_trot_v1/FINAL_DATA_GATE.md`: resampling 3 s,
   stops 10%, frames iniciales incluidos y hindsight de 2 s. No copiar defaults
   históricos de 5 s/8% ni reescribir alineación pre-action/reset.
2. Medir soporte realizado de pace; combinar un volumen comparable de ventanas
   válidas por fuente. No confundir comando aleatorio con velocidad alcanzada.
3. Nueva Phase A WC+pace con split/normalizadores propios y arquitectura/receta
   actual; declarar presupuesto extra de updates. Sin gait ID como input.
4. Confirmar retención diagonal e ipsilateral **antes** de DPPO. Misma altura
   no identifica unívocamente qué gait debe elegir: no prometer switching
   determinado por tiempo si ambos satisfacen la tarea igualmente bien.
5. Refinar con contrato v3 inicialmente congelado, sin gait reward/ID ni online
   symmetry pilot. Auditar antes/después y conservar categoría mixed/unknown.
6. Ampliación de rutas y sim2sim son trabajos separados; no cambiar rutas,
   experto y objetivo temporal simultáneamente para atribuir luego mejoras a pace.

## 11. Referencias y qué justifican exactamente

- [Isaac Lab, Spot GaitReward](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py):
  fuente de sincronización/oposición por timers. Adaptamos parejas para pace;
  no trasladamos sus pesos de Spot ni asumimos rendimiento en SOLO12.
- [Walk These Ways](https://arxiv.org/abs/2212.03238) y su
  [implementación](https://github.com/Improbable-AI/walk-these-ways): alternativa
  de control de gait con parámetros de fase. No adoptada en este piloto para
  conservar interfaz 48D y warm-start; no implica que los clocks sean malos.
- [Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/abs/2403.04359):
  distinguir simetría morfológica de simetría de tarea. No prueba que augmentation
  siempre mejore un prior asimétrico ni valida transformaciones sin tests.
- DiffuseLoco y DPPO: fundamento del proyecto y del refinamiento posterior,
  no evidencia de que el nuevo experto pace vaya a ser retenido automáticamente.

El resultado válido puede ser que pace se aprenda como experto y se pierda en
imitation o refinamiento. Eso localizaría un límite de composición/retención;
no debe ocultarse con labels o rewards de gait añadidos al controlador final.

## 12. Entrega exigida al implementador

- Diff limitado a los archivos de la sección 6, con explicación de desviaciones.
- Lista de tests ejecutados, outputs y tests no ejecutados con motivo.
- Smoke real distinguido de tests puros; checkpoint de smoke fuera de Git.
- Comandos Windows/cluster, rutas de salidas y README coherentes con el parser.
- Ningún train largo, colección, actualización del paper, commit/push o branch
  switch sin revisión/autorización. No tocar cambios previos del usuario.

**Prompt para el modelo implementador:**

> Lee este documento completo y los archivos de la sección 2. Implementa sólo
> el experto pace y los pasos A–C de la sección 8, respetando fórmulas, contratos
> y alcance. Reutiliza el código existente y no hagas refactors ajenos. Ejecuta
> tests puros y el smoke real si el entorno lo permite; reporta cualquier fallo
> sin esconderlo ni convertir el smoke en un train largo. Entrega diff, tests y
> comandos, y para antes de commit/push, colección o nuevas fases de entrenamiento.

## 13. Implementación y revisión local (2026-10-06)

- Implementados el task aislado `solo12-pace-v0`, los helpers tensoriales,
  registro, UI, comparación walk/pace y launcher Slurm.
- El runner reutiliza la augmentation quadruped existente (cuatro
  transformaciones); esto es distinto del piloto DPPO izquierda-derecha.
- Siete tests tensoriales aprobados, incluidos invariancia de reflejos,
  rechazo de patrones estáticos y equivalencia con la fórmula de timers.
- Compilación Python y sintaxis Bash del launcher aprobadas.
- Smoke Isaac local: dos entornos, una iteración PPO, warm-start estricto y
  rewards finitos. Ocho entornos fallaron por `bad allocation` en esta máquina;
  el launcher del cluster conserva su smoke de ocho entornos/dos iteraciones.
- Smoke del evaluador con walk en ambos lados: JSON/NPZ/PNG generados con
  backend Agg. Valida el evaluador, no la adquisición de pace.
- El SBS se publica con finales LF. El test debe incluirse explícitamente
  en Git pese a la regla general `tests/` del `.gitignore`.
- El límite de comandos pace se aplica solamente a `walk_pace`, preservando
  los rangos de las comparaciones anteriores.

El siguiente paso es el train Slurm del experto y su gate walk/pace. No se ha
realizado entrenamiento largo ni se han recogido demostraciones de pace.
