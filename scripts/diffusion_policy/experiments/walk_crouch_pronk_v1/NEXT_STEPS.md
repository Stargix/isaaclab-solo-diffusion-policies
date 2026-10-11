# Estado y próximos pasos — 2026-10-11

Actualización 2026-10-12: el siguiente train es la
[selección temporal corregida](../walk_crouch_pronk_speed_partition_v1/README.md),
no la mezcla solapada v1. Fuentes preservadas, walk ≥0,6 sin arranque/reposo
en sus ventanas. Retención y transición siguen pendientes de evaluar.

## Base que se conserva

- [x] WC/WCT anteriores y sus evaluaciones con seeds 42/43/44.
- [x] Baseline determinista BC + Gaussian PPO y sus réplicas evaluadas.
- [x] Protocolo de geometría, postura, tiempo y OOD; análisis de asignación temporal.
- [x] Auditorías de gait y de coste de inferencia; no confundirlas con transferencia física.
- [x] Artículo y material explicativo existentes, sin reemplazar sus resultados por este piloto.
- [x] Pace/bound/fast-trot estudiados; no reutilizar fast-trot en la mezcla nueva.
- [x] Pronk V8 entrenado, probado visualmente y con screen congelado de 6 s.

## Extensión de tres comportamientos

- [x] Definir pronk lento + walk de mayor alcance + crouch sin ID de gait.
- [x] Recuperar y fijar el protocolo temporal real A0: 6 s / 3 s / 2 s hindsight.
- [x] Mantener misma altura nominal para walk/pronk; documentar el desfase real de pronk.
- [x] Preparar colector, manifiesto verificable e índice ligero sin borrar datasets anteriores.
- [x] Detectar/corregir refresh de inputs tras resampling/reset; tests con loader real.
- [x] Recoger pilotos corregidos y comprobar vuelo/límites/self-collisions.
- [x] Cerrar datasets completos: 3.000.465 frames, balance exacto, validador real,
      auditoría temporal y hashes correctos; ver [DATA_GATE.md](DATA_GATE.md).
- [x] Recuperar las filas iniciales de reset mediante su estado exacto conocido;
      no cambiar acciones/trayectorias. Documentar el tick histórico del crouch.
- [x] Launcher de Phase A con preflight de hashes, guardas anti-sobrescritura,
      train desde cero y procedencia. SBS privado en scratch, fuera de Git.
- [x] Autorizar publicación selectiva de esta preparación; excluir datos, logs,
      SBS y cambios ajenos. Actualizar el cluster con el commit publicado.
- [ ] Transferir datos verificados y ejecutar **Phase A desde cero en el cluster**.
- [ ] Evaluar el checkpoint offline: crouch nativo; pronk a 0.15/0.2/0.35/0.5;
      walk de mayor alcance; stops/arranques; curvas y steering moderado.
- [ ] Comprobar selección/retención sin ID, en particular a igual altura y velocidad solapada.
      Medir gait con contactos/vuelos/ciclos, no inferirlo del PCA o de la rapidez.
- [ ] Solo entonces preparar DPPO y revisar el significado del criterio de altura:
      un criterio instantáneo demasiado estricto puede eliminar el rebote de pronk.
      No añadir una recompensa de gait solo para obtener la conclusión deseada.
- [ ] Medir transiciones y asignación temporal tras DPPO; mantener fallos y no elegir
      únicamente vídeos donde cambian los tres comportamientos.

Primero basta una seed exploratoria. Las réplicas y un control con el **mismo
nuevo walk** son necesarios si después se reclama una ventaja causal de añadir
pronk; no son prerequisites para comprobar si el experimento funciona siquiera.
El WC antiguo no es ese control emparejado porque cambia el soporte walk, el
número de ejemplos y, con 50 epochs, el número de updates.

No está programado RL de tarea desde cero ni otro jerárquico. No son el objetivo
de esta extensión. Tampoco toca retocar familias de rutas a la vez que expertos:
mantener el contrato para saber qué se está probando.

## Trabajo paralelo y cierre

- [ ] Optimizar/medir inferencia del actor que finalmente se vaya a desplegar, sin
      prometer que una red más pequeña o menos denoising conservará el resultado.
- [ ] Completar sim2sim físico con interfaces/PD/orden de articulaciones/tiempos
      verificados; el replay del runtime no equivale a este experimento.
- [ ] Robot real con puesta en marcha y seguridad incremental; no presentar
      resultados de hardware antes de medirlos.
- [ ] Integrar en el I2R el método probado y sus límites; la extensión de tres
      comportamientos es una ablation/piloto hasta tener evidencia suficiente.
- [ ] Para paper: añadir solo conclusiones demostradas, controles emparejados si
      se reclaman efectos causales y transferencia si el alcance la incluye.

La fase de datos no demuestra por sí sola que DPPO vaya a usar pronk. El objetivo
inmediato es construir una base legible y reproducible y resolver esa pregunta
con Phase A, sin modificar reward/arquitectura para forzar el resultado.
