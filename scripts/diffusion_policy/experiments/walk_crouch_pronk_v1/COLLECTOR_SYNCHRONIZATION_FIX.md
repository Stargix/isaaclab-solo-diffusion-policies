# Corrección antes del nuevo train

Fecha: 2026-10-11. Detectada al revisar el colector, antes de aprobar el dataset.

## Problema concreto

El modo `phase_a` mantenía `obs` devuelto por `vec_env.step`. Después podía:

1. cambiar `raw_env._commands` al alcanzar el contador de 3 s;
2. poner comando cero después de un timeout;
3. resetear manualmente un robot por el guardrail y refrescar solo el estado
   físico guardado en `cached_state`.

La siguiente acción podía consumir la observación antigua. En un cambio de
comando esto introducía un tick de retraso. En un reset manual, el input del
experto podía corresponder al estado anterior al reset, aunque el estado físico
registrado ya fuera nuevo. Los tokens del archivo no bastan para detectar esto:
`last_action[t] = actions[t-1]` y la integración de la referencia pueden pasar.

## Corrección acotada

Antes de cada inferencia en `phase_a`, refrescar la observación del experto y
copiar el comando activo al estado a registrar. El perfil cerrado ya hacía ese
refresh. No cambiar el muestreador, rewards, expertos ni definición hindsight.
La nueva fuente declara `policy_input_timing=synchronous_pre_action_v2`.

El test reproduce un reset y una escritura de comando con un wrapper falso y
comprueba que input/registro coinciden y que el array guardado no es una vista
mutable del comando. Los tests adicionales validan alturas nominales,
dimensiones del batch, remapeo de skills y relocación del índice HDF5.

## Artefactos

- `data/datasets/walk_crouch_pronk_v1/`: primera recopilación **excluida**, se
  conserva para trazabilidad. Pronk full fue interrumpido y puede estar incompleto.
- `data/datasets/walk_crouch_pronk_v1_aligned/`: nuevas fuentes y manifiesto que
  se utilizarán para el siguiente train; el preparador rechaza fuentes nuevas
  sin la marca de sincronización.
- Crouch histórico permanece congelado; no afirmar retroactivamente que se
  recogió con este nuevo código. Su contrato de tiempos y pares físicos
  pre-acción se revalida. Su metadata registra cero episodios descartados, de
  modo que no tuvo la rama de reset manual por caída. El hindsight geométrico
  se obtiene del movimiento ejecutado, no de una supuesta obediencia perfecta
  al comando. Esto no convierte su procedencia en idéntica a la de las fuentes
  nuevas ni invalida automáticamente los experimentos WC anteriores.

Las cifras del primer piloto no son resultados del colector corregido ni una
ablación del fix: cambiar la secuencia de acciones puede cambiar los resets y
qué sorteos aleatorios corresponden a cada episodio. Utilizar la auditoría final
para decidir soporte, no comparar dos porcentajes de harvest como si fuesen un
benchmark emparejado.

## Segundo error: registro de articulaciones en un reset individual

La consulta `joint_pos[sel, joint_ids]` combina dos índices avanzados: con una
sola fila seleccionada devuelve `(12,)`, no `(1, 12)`. El posterior `[0]`
podía distribuir el primer hip (cero) por las doce articulaciones del frame
inicial. La consulta se corrige a `joint_pos[sel][:, joint_ids]`, también para
velocidad. Un test comprueba una y varias filas seleccionadas.

Este error afecta al cache **registrado**, no a las observaciones físicas
refrescadas que usa el experto después del primer fix. Por ello no cambió las
acciones ni la trayectoria de la recopilación completa corregida.

La recuperación de `reset_row_repair.py` está limitada a estos nuevos HDF5.
Verifica el hash original, colección cerrada, inputs sincronizados, postura
SAFE exacta en las filas intactas, velocidad cero, comando cero durante 25
ticks y Z inicial 0.35 m. Solo recupera `obs/joint_pos[0]` cuando contiene los
doce ceros de esa firma conocida. Cualquier otra postura aborta. No interpola
estados ni cambia acciones, otros frames o la fuente crouch congelada.

En los datasets completos se recuperaron **1.070/3.345 filas iniciales walk**
y **83/3.345 pronk**. El reset sin ruido escribe determinísticamente
`[0,.4,-.8, 0,.4,-.8, 0,-.4,.8, 0,-.4,.8]`; no es una pose inferida a partir
del movimiento. Los sidecars y el manifiesto guardan hashes originales/finales
y nombres afectados. El test verifica que acciones y todas las otras filas
permanecen idénticas. Los hashes del colector durante recopilación y durante
finalización se distinguen explícitamente; no se atribuye retroactivamente el
segundo fix a una colección que ya estaba en marcha.

## Tick de comandos del crouch histórico

La auditoría completa encuentra cambios de comando registrados en 26/176
(y 126/226/276 cuando hay stops), frente a 25/175 en las fuentes nuevas.
Es un tick de 20 ms de la procedencia antigua; no se desplazan estados ni
acciones para esconderlo. Todos mantienen holds móviles de 150 ticks y stops
de 100 ticks; la referencia registrada integra sus comandos con error máximo
5.35e-6 m y el `last_action` coincide exactamente con la acción anterior.
El conditioning usado en este experimento procede de la **trayectoria física
achieved**, no de esos cambios de comando/reference. Por eso ese tick no cambia
el horizonte hindsight compartido de 100 ticks ni introduce un target de
velocidad local. Se declara la diferencia, sin afirmar colección idéntica.

El antiguo `validate_phase_a_dataset.py` exige giros estrictamente mayores que
0.2 rad/s: rechaza pronk, cuyo cap declarado es precisamente 0.2. El check
específico `audit_temporal.py` no desactiva validación: comprueba todos los
holds, integración, stops, giros no nulos y alineamiento acción/estado en las
tres fuentes, con el tick histórico explícito y sin modificar ese validador.
