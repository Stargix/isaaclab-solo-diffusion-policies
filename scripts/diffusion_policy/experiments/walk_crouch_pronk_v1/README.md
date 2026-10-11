# Phase A: crouch + pronk lento + walk de mayor alcance

**2026-10-12: no lanzar esta mezcla solapada para estudiar selección por ritmo.**
Se conserva como artefacto histórico/fuentes. El lanzamiento actual está en
[walk_crouch_pronk_speed_partition_v1](../walk_crouch_pronk_speed_partition_v1/README.md):
walk rápido con floor real de 0,6 y sin arranques walk desde cero.

Decisión del 2026-10-11. Este experimento sustituye la propuesta pace y **no
utiliza el experto fast-trot, sprint ni sus datasets**. Se conserva el experimento
WC/WCT anterior; no se cambian sus actores, rewards, configuración ni datos.

**Preparación completa:** 10.035 episodios / 3.000.465 frames, auditoría y
hashes verificados. Resultado final y límites: [DATA_GATE.md](DATA_GATE.md).
Los datos válidos están en `data/datasets/walk_crouch_pronk_v1_aligned/`.

## Qué queremos comprobar

Una misma difusión, condicionada por geometría, altura requerida y velocidad
media, debe poder conservar tres manifolds de movimiento sin recibir un ID de
skill. El pronk aporta un movimiento visualmente distinto a velocidades bajas;
el walk aporta las distancias mayores para un mismo horizonte temporal; crouch
aporta la postura baja. Esto es una extensión exploratoria, no una sustitución
automática del resultado principal WC ni una promesa de selección de gait.

No añadimos una tercera altura para identificar pronk, recompensa de gait en
DPPO, etiquetado de fases ni mezclas artificiales de acciones. `skill_idx` existe
solo para balance/auditoría y no aparece en los inputs de la red.

## Fuentes

La receta ejecutable está en [protocol.json](protocol.json). Los hashes de los
checkpoints walk y pronk se verifican antes de recoger. El manifiesto de los
datos termina de fijar los hashes de cada HDF5 y la configuración del trainer.

| Experto | Fuente | Comandos propuestos, no velocidades garantizadas | Altura requerida |
|---|---|---|---|
| walk | `checkpoints/walk_final.pt`, tarea `solo12-v0` | forward 0.20–1.50; arcos hasta 1.0; reverse hasta 0.70; lateral hasta 0.50; yaw hasta 0.45 rad/s | 0.2932 m |
| crouch | HDF5 congelado `crouch_phase_a_waypoint_v3.hdf5` | forward 0.08–0.50; arcos hasta 0.40; reverse hasta 0.30; lateral hasta 0.40; yaw hasta 0.30 rad/s | 0.1705 m |
| pronk | V8 `model_2999.pt`, SHA `73c572ff…`, tarea `solo12-pronk-v8-v0` | forward 0.15–0.50; arcos hasta 0.45; reverse hasta 0.35; lateral hasta 0.15; yaw hasta 0.20 rad/s | **0.2932 m, igual a walk** |

La política pronk sigue entrenada en vx [-0.65, 0.65] y vy/wz ±0.25. Solo se
recopila un subconjunto más estable; **no se ha reducido el rango del train del
experto**. Walk y pronk se solapan deliberadamente: no se fuerza una frontera
de gait mediante bins disjuntos.

Se reutiliza el crouch congelado para no recollectar ni modificar datos que
funcionaron. Walk se recoge de nuevo porque queremos su alcance alto, no el del
fast-trot. Por ello este experimento no aísla únicamente «añadir pronk» frente
al WC antiguo: también amplía la propuesta de comandos walk. No presentarlo
como esa comparación causal estricta sin un control con el mismo nuevo walk.

## Tiempos: misma receta, no un horizonte diferente para cada skill

| Componente | Valor | Motivo |
|---|---:|---|
| Control / física | 50 Hz / 200 Hz | Compatibilidad con expertos y dataset original |
| Episodio | 6 s; normalmente 299 frames guardados | Evitar clips largos con deriva acumulada; nunca concatenar episodios |
| Arranque | 25 ticks = 0.5 s, comando cero | Frames conservados; crouch histórico registra un tick adicional (0.52 s), declarado en DATA_GATE |
| Comandos en movimiento | Cada 150 ticks = 3 s | Mismo protocolo A0/v4; horizonte de 2 s tiene soporte dentro de fases y cruza algunos cambios reales |
| Stop propuesto | Probabilidad 0.10, hold 100 ticks = 2 s | Mantener arranque/frenado y segmentos casi estacionarios |
| Horizonte hindsight | 100 ticks = **2 s para todos** | El tiempo no debe codificar la identidad del experto |
| Historia de la red | 8 ticks = 0.16 s | Se conserva la arquitectura probada |
| Secuencia de acciones | 16 ticks, offset 8; 8 acciones futuras | Mismo objetivo de entrenamiento y alineamiento |
| Ejecución posterior | 4 ticks = 80 ms entre replans | Contrato de despliegue ya medido; no confundir con denoising K=10 |

«Versión larga con walk» significa mayor **distancia alcanzada en esos mismos
2 s**, no alargar el hindsight solo para walk. Idealmente 0.3 m/s recorre 0.6 m
y 1.0 m/s recorre 2 m. La etiqueta utiliza la longitud XY **realmente recorrida**
dividida por 2 s; no `comando × 2 s`. No se supone que el walk alcance 3 m por
pedirle 1.5 m/s.

Importante: **profile16 usa puntos al 25/50/75/100% del arco**, no puntos a
0.5/1/1.5 s. Esos offsets siguen en la configuración heredada por compatibilidad,
pero no convierten los puntos geométricos en una velocidad local obligatoria.
La altura de los cinco tokens procede de `desired_base_height`, no del Z
instantáneo; el rebote de pronk no se relabela como cambios de postura.

El 10% de stops es probabilidad por sorteo, **no 10% de frames ni ventanas con
velocidad cero**. Un hold de 2 s permite cubrir el horizonte nominal, pero su
frenada consume parte del tiempo y no garantiza ventanas completamente
estacionarias. Se mide el soporte casi cero; no se fabrica una parada perfecta.

Se solicitan 1,000,000 frames aceptados por experto. Con episodios de 299 ticks
se obtienen 3,345 demos / 1,000,155 frames por fuente. Hay 665,655 anclajes de
estado con futuro exacto de 2 s. El loader con arranques padded cuenta 669,000
anclajes de acción por fuente: la diferencia es el token de estado t−1 y el
padding inicial, no un desfase entre expertos. Las transformaciones quadruped
se aplican después, igual que en el train anterior.

## Recopilación y selección

Se reutiliza `data/collect_data.py` con cambios acotados: reconocer pronk,
activar explícitamente la configuración final de su curriculum antes de fijar
el perfil de recopilación, diagnósticos opcionales fuera de `obs`, límite de
pasos y protección contra sobrescritura. No se reescribe el muestreador.

- Expertos independientes, comandos aleatorios, no catálogo de rutas ni
  tracker nuevo. Mezcla heredada de movimiento: 37.5% straight, 37.5% arc,
  15% lateral, 10% reverse, multiplicada por 0.90; 0.10 de stops.
- Este muestreador tiene mínimos heredados: en pronk los arcos solicitan ±0.20
  rad/s y los modos laterales ±0.15 m/s. Otros modos añaden comandos continuos
  pequeños. No afirmar que todas las componentes son uniformes globalmente;
  la geometría hindsight sigue variando con vx, duración, seguimiento y yaw.
- Perfil físico light; observaciones sin ruido, sin pushes/delay, resets limpios.
  Los PD y límites físicos del experto no se alteran. La fricción articular
  permanece desactivada en pronk, como en su contrato local V8; walk usa el
  perfil light heredado. No fingir equivalencia exacta entre contratos distintos.
- Nuevos datos locales: Isaac Sim 5.1, RSL-RL 3.1.2. Crouch es un artefacto
  histórico congelado. Este mix de procedencias queda declarado; no equivale
  a una demostración de sim2sim.
- Se descarta el **episodio completo** que termina por caída/guardrail. No se
  cortan frames «bonitos» justo antes del fallo. Los nuevos expertos usan un
  guardrail de tilt 30 grados y se conserva todo el arranque.
- Estadísticas de recolección = episodios finalizados hasta llenar el cupo;
  los pendientes al cerrar están censurados. No son un benchmark congelado de
  supervivencia ni tasas de éxito de una tarea de ruta.

Patas estrechas: no equivalen por sí solas a un gait inválido. El piloto registra
centros de pies, torques, límites y vuelo confirmado; además hay un piloto
separado con self-collisions activadas. Esto comprueba el modelo de colisión USD,
no garantiza ausencia de toda interferencia mecánica en el robot real.

## Evidencia y gate completados

Usar las cifras finales de [DATA_GATE.md](DATA_GATE.md), no los resultados del
primer lote excluido. Ver las correcciones de registro y su trazabilidad en
[COLLECTOR_SYNCHRONIZATION_FIX.md](COLLECTOR_SYNCHRONIZATION_FIX.md).

Los resultados completos se guardan junto a los datos en
`data/audits/walk_crouch_pronk_v1/`; logs y procedencia en
`scratch/walk_crouch_pronk_v1/`. Los archivos grandes y plots no se publican en Git.

- Walk full: 3.345/5.652 episodios aceptados (59.18%). En el bin de comando
  1.25–1.50, vx mediano 1.143 m/s y p95 1.325. Soporte rápido real pero filtrado;
  **no un experto fiable que clave 1.5 m/s**.
- Pronk full: 3.345/4.093 (81.72%). El piloto corregido (100/101) tiene vuelo
  confirmado mediano ~120 ms y cero violaciones blandas, con separación de
  centros de pies mediana ~8.1 cm. No extrapolar ese vuelo a todos los frames.
- Self-collision piloto corregido: 16/20 (80%); conserva vuelo y postura
  similares. Check pequeño, con distinta secuencia aleatoria; no prueba
  estadística causal ni hardware. No forma parte del train.

**Limitación importante de altura:** pronk tiene una altura real mediana ~0.260
m frente a ~0.298 en walk aunque ambos reciben 0.2932 m de condición nominal.
El experto pronk no recibe esa altura como input ni tiene un target instantáneo
en su reward (`track_base_height_reward_scale=0`); 0.2932 es la anotación
compartida «de pie» de este experimento, no una altura que el experto haya
aprendido a clavar. Este supuesto queda explícito y es aproximado.
Mantener la etiqueta compartida evita esconder un ID, pero no convierte ese
error en seguimiento perfecto. Antes de DPPO habrá que verificar que su
criterio de postura no penaliza el rebote/nivel medio hasta eliminar el gait.
No se ha cambiado ahora el reward para favorecer pronk. No prometer retención
de tres gaits únicamente porque el HDF5 sea correcto.

Decisión confirmada antes de Phase A: mantener la anotación nominal compartida,
no relabelar el sinus instantáneo de Z como una ruta de altura, ni añadir ahora
una tercera altura exclusiva de pronk. Las acciones contienen el rebote real;
los objetivos de altura representan postura de tarea. Es una prueba exploratoria
de retención sin pista explícita de fase/gait, no prueba de tracking exacto de
altura media. Seguridad/clearance sí deben considerar el máximo instantáneo.

## Preparación reproducible

Desde la raíz, usando Python de `env_isaaclab`:

```powershell
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/test_protocol.py
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/prepare_data.py --stage pilot
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/prepare_data.py --stage collision_pilot
# Solo después de revisar el piloto:
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/prepare_data.py --stage full
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/prepare_data.py --stage merge
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/prepare_data.py --stage verify
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/audit_data.py --stage full
python scripts/diffusion_policy/experiments/walk_crouch_pronk_v1/audit_temporal.py
```

Los stages ejecutados no se relanzan sobre sus archivos: fallan explícitamente
antes de sobrescribir. `--skill walk` o `--skill pronk` permite recopilar una
fuente pendiente. Los pilotos no forman parte del train.

El índice `walk_crouch_pronk_linked.hdf5` contiene links relativos para evitar
duplicar el GB de datos con poco espacio en C:. Se deben transferir **los dos
HDF5 nuevos, el crouch congelado, índice y manifiesto**, manteniendo su estructura.
Ver [CLUSTER_COMMANDS.md](CLUSTER_COMMANDS.md).

## Siguiente train y punto de decisión

Phase A desde cero, configuración profile16 A0 existente byte-idéntica, misma
red/normalización/quadruped/K=10/batch4096/50 epochs/seed42. Más demos implican
más updates; no interpretar una mejora como ventaja con igual cómputo.
Nada de inicializar desde WCT ni cargar fast-trot por fallback.

Orden: (1) auditoría de datos → (2) train en cluster → (3) medir retención de
pronk, walk rápido, crouch y arranques con conditioning compartido → (4) decidir
si procede DPPO. El diagnóstico de gait debe usar vuelos/contactos/ciclos,
comparado con los expertos, no solo una proyección PCA ni una impresión visual.
Si pierde pronk, diagnosticar antes la altura, cobertura y preferencia de modo;
no añadir silenciosamente un ID o recompensa diagonal.

La base es coherente con [DiffuseLoco](https://arxiv.org/abs/2404.19264)
(datos offline de comportamientos heterogéneos en una difusión) y
[Diffusion Policy](https://arxiv.org/abs/2303.04137) (secuencias de acciones
multimodales). Ninguno garantiza que esta nueva mezcla retenga un gait concreto.
El problema de distribución entre experto y learner señalado por
[DAgger](https://proceedings.mlr.press/v15/ross11a.html) sigue existiendo aun
filtrando todas las caídas del dataset.
