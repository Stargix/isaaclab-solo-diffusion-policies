# Phase A: pronk lento, walk rápido, crouch — selección temporal explícita

Corrección del 2026-10-12. **Este es el train que debe lanzarse ahora**, no
el launcher de `walk_crouch_pronk_v1` con soporte solapado. Se conserva ese
experimento y sus fuentes: no se borran ni se sobrescriben datos anteriores.

## Pregunta original y por qué cambia la selección

Queremos comprobar si una difusión con geometría, postura y ritmo como inputs,
sin ID ni fase de gait, puede representar tres comportamientos: pronk lento,
walk rápido y crouch bajo. En el dataset inicial, el 72,37% de ventanas walk
de 2 s estaba a ≤0,5 m/s; pronk tenía el 99,12% en esa región. Esa mezcla era
válida para estudiar multimodalidad, pero no favorecía que pedir un ritmo lento
seleccionase pronk: el modelo tenía muchos ejemplos walk para la misma tarea.

Esta nueva ablation especializa deliberadamente el soporte de demostraciones
según una variable real de tarea, la velocidad conseguida. **No es prueba de
que el robot descubra espontáneamente un gait óptimo.** Tampoco demuestra que
DPPO vaya a conservar pronk: el reward de tarea no distingue gaits equivalentes.
No se añade un ID oculto mediante una altura exclusiva ni una onda de altura.
Walk/pronk conservan el mismo 0,2932 nominal; crouch, 0,1705. La altura media
real de pronk (~0,26) sigue siendo una limitación declarada, no tracking perfecto.

## Qué significa exactamente el mínimo walk de 0,6

No cambiamos ese límite a 0,55 ni lo aplicamos solo al comando:

1. **Todos los ocho tokens históricos** de velocidad hindsight deben ser
   ≥0,6 m/s, calculados del arco XY conseguido en sus siguientes 100 ticks.
2. La **velocidad XY local suavizada sobre 10 ticks = 0,20 s** debe ser
   ≥0,6 en toda la unión de historia y futuros hindsight. Así, una media de
   0,6 que esconde un arranque desde cero o una frenada lenta queda fuera.
3. Comando planar ≥0,6 en todo ese contexto; no incluir coasting tras pedir stop.
4. Excluir los primeros 25 ticks (0,50 s) tras el último cambio de comando
   previo al contexto. Excluir cualquier tick con velocidad XY ≤0,02 m/s.
5. No permitir arranques walk padded. Comprobar también el tick anterior a la
   primera observación, usado por la historia de acciones retardada.
6. Mantener el target de 16 acciones y sus estados exactos. No recortar clips
   y fingir un reset nuevo; no desplazar estados ni poner last_action a cero.

La suavización no obliga a cada instante de 20 ms a ser ≥0,6: la marcha tiene
oscilaciones normales de velocidad. El contrato excluye el arranque/parada y
un régimen local lento, no suprime la dinámica del gait ni prescribe velocidad
constante. Se audita también la velocidad instantánea mínima para no ocultarlo.

Un arranque desde cero **con un goal rápido** no es automáticamente una
demostración de tarea lenta. Retirarlo aquí aísla el soporte rápido, pero
sacrifica ejemplos útiles de arranque. El problema original era el 72% de
**goals lentos** walk, no la existencia de aceleraciones por sí sola. No
confundir una oscilación instantánea del cuerpo con una orden de ritmo lento.

Pronk conserva las ventanas cuyos **ocho tokens** tienen velocidad media ≤0,5,
incluidos arranques/stops nativos. Crouch conserva su soporte. La banda
0,5–0,6 de pie no se rellena con etiquetas fabricadas: es una banda de
generalización a evaluar, no evidencia de que ya se hayan aprendido transiciones.

El floor es velocidad planar, no vx con signo. Las curvas/reverse rápidos
pueden participar si cumplen el contrato; augmentation quadruped se mantiene.

## Datos preparados y balance

| Experto | Episodios con anchors elegibles | Anchors únicos | Anchors por epoch antes de simetría |
|---|---:|---:|---:|
| walk | 556 | 25.375 | 669.000 |
| crouch | 3.345 | 669.000 | 669.000 |
| pronk | 3.345 | 662.534 | 669.000 |

Walk: velocidad del último goal mediana 0,786, mínimo 0,611 y p95 1,092 m/s.
El mínimo **del selector** sigue siendo 0,6; los datos supervivientes pueden
estar por encima. Pronk: mediana 0,284, máximo 0,500 m/s. Estos números se
refieren a las ventanas seleccionadas, no a la velocidad máxima del experto.

No igualamos pérdidas dejando walk con 25k frente a ~660k de los otros:
cada anchor walk tiene multiplicidad 26–27, pronk 1–2 y crouch 1. Se preservan
todos los anchors elegibles y el resto del cupo se reparte determinísticamente
(seed50). Son **pesos de muestreo**, no nueva evidencia ni nuevos episodios.
Así se conservan 2.007.000 anchors ponderados, 50 epochs y el presupuesto de
warmup de la receta anterior; reducir todo a 25k por skill dejaría menos updates
que los 10.000 pasos de warmup heredados. No cambiar silenciosamente el LR.

La repetición fuerte aumenta riesgo de overfit en walk. Sus 556 episodios
independientes dan una primera prueba razonable, no cobertura universal.
La separación train/validation sigue siendo por episodio: todas las ventanas
y repeticiones de un rollout van al mismo lado. Balance total exacto no implica
balance exacto tras dividir episodios; la auditoría publica ambos conteos.

## Implementación mínima y compatibilidad

El nuevo HDF5 es un índice de links relativos a las **mismas tres fuentes**.
Solo añade `training_anchor_steps` por episodio, ordenados, con repeticiones
como peso. El loader respeta esa lista cuando existe; sin ella conserva su
indexado anterior. Skill metadata y listas de anchors no entran en la red.

La normalización de estos datos se ajusta solo con estados/acciones realmente
usados por sus ventanas y solo con episodios de train. Los frames walk lentos
del archivo fuente no reaparecen por la normalización. Fuentes originales,
alineación, geometría, horizonte 2 s, Transformer, K=10, quadruped y config
A0 siguen intactos. No ejecutar la selección como un cropper de episodios.

Scripts:

- `prepare.py`: prepara y sella índice/manifiesto, o `--verify` sin escribir.
- `selection.py`: criterio puro, testeable sin simulador.
- `test_selection.py`: floor, arranque, frenada, borde retardado, loader,
  normalización y compatibilidad sin metadata.
- `audit.py`: comprueba todos los anchors; loader real, split por episodios y
  un forward/loss CPU finito, **sin optimizador ni train local**; genera el plot.
- `train.py`: reutiliza el launcher verificado de Phase A, con nuevo índice
  y output. Exige CUDA asignada, checkout limpio y no sobreescribe runs.

Preparación/verificación local:

```powershell
python scripts/diffusion_policy/experiments/walk_crouch_pronk_speed_partition_v1/test_selection.py
python scripts/diffusion_policy/experiments/walk_crouch_pronk_speed_partition_v1/prepare.py --verify
python scripts/diffusion_policy/experiments/walk_crouch_pronk_speed_partition_v1/train.py --preflight-only
```

Datos: `data/datasets/walk_crouch_pronk_speed_partition_v1/`.
Auditoría/plots: `data/audits/walk_crouch_pronk_speed_partition_v1/`.
Ni datos, plots, logs ni SBS van a Git. [Comandos de cluster](CLUSTER_COMMANDS.md).

## Gate final medido

Ocho tests de selección y siete de compatibilidad pasan. Auditoría exhaustiva
de todos los anchors y de **todos los tokens reales del trainer**: walk mínimo
0,611271 m/s; pronk máximo 0,499999 m/s. Velocidad local mínima walk (0,20 s):
0,6000005; comando planar mínimo 0,600241. Primer tick físico usado: 50 (1 s);
cero arranques walk padded.

La velocidad instantánea mínima de un tick de 20 ms es 0,0861 m/s: no afirmar
que cada instante es ≥0,6. Se conservan oscilaciones rápidas del gait, no un
arranque desde cero ni segmentos locales de 0,2 s por debajo del floor.

Loader real: 2.007.000 anchors ponderados / 8.028.000 con simetría. Split seed42:
train walk/crouch/pronk = 640.116 / 637.400 / 634.102; validation = 28.884 /
31.600 / 34.898. Cero episodios compartidos. Un forward/loss CPU con los tres
expertos da un valor finito; **cero pasos de optimizador**. Ese loss de smoke
no mide la calidad del modelo ni predice el resultado del train.

Fuentes originales mantienen exactamente sus hashes. Índice seleccionado:
`aa45ce00ffeb72a16dbd4ea3145b179503b04def6be1c70db072a25e24c1729b`.
Los archivos intermedios quedan ignorados y no deben transferirse. El plot
`speed_partition_support.png` se revisó visualmente: gris = movimiento original
no supervisado; azul = contexto walk utilizado.

## Qué comprobar tras este train

1. Retención física de pronk con targets lentos de pie, frente al experto
   (contactos/vuelo/ciclos, no solo PCA o velocidad).
2. Walk rápido ≥0,6 y crouch nativo; evaluar alturas reales y supervivencia.
3. Arranque desde reposo pidiendo rápido: **ahora se prueba generalización**;
   no afirmar que hay una demostración walk de ese arranque. Pronk sí aporta
   arranque, pero todavía no demostraciones de transición pronk→walk.
4. Misma geometría/altura variando únicamente el ritmo, incluida banda
   0,5–0,6. Una transición estable no queda garantizada por soporte disjunto.
5. Solo después decidir DPPO y su semántica de altura. No penalizar el rebote
   hasta borrarlo ni añadir gait rewards/IDs para fabricar una conclusión.

Esto corrige el diseño del experimento, no garantiza que una red nunca genere
walk lento: puede extrapolar fuera del soporte, especialmente tras RL.
