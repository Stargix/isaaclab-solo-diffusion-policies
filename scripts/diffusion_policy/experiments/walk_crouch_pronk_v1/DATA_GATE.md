# Gate final de datos — 2026-10-11

**Resultado: preparado y verificado para un primer train offline exploratorio.**
No se ha entrenado Phase A aquí. No hay fast-trot: las fuentes son walk_final,
crouch congelado y pronk V8 model_2999. No se alteran los experimentos WC/WCT.

## Artefactos que usar

Dataset de entrada:
`scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/walk_crouch_pronk_linked.hdf5`.
Es un índice de links relativos; necesita los dos `*_full.hdf5` en su carpeta
y `../crouch_phase_a_waypoint_v3.hdf5`. El manifiesto verifica las cuatro
fuentes/índice, protocolo y configuración. Los pilotos **no** entran al train.

| Fuente | Demos aceptadas | Frames | Ventanas de estado de 2 s | Aceptadas/finalizadas |
|---|---:|---:|---:|---:|
| walk | 3.345 | 1.000.155 | 665.655 | 3.345/5.652 = 59,18% |
| crouch histórico | 3.345 | 1.000.155 | 665.655 | 3.345/3.345 = 100% registrado |
| pronk | 3.345 | 1.000.155 | 665.655 | 3.345/4.093 = 81,72% |
| Total | 10.035 | 3.000.465 | 1.996.965 | No agregar como benchmark |

Balance exacto en episodios, frames y ventanas. El loader usa 669.000 anchors
de acción padded por fuente (2.007.000 antes de augmentation). Se rechazan
episodios completos que caen; los pendientes al cerrar están censurados.
Estos porcentajes describen cosecha filtrada, **no** supervivencia general.

## Tiempos comprobados, no solo configurados

- 50 Hz de control, 200 Hz física; episodios nominales de 6 s y 299 filas
  pre-acción guardadas. Último estado registrado a 5,96 s; no concatenar clips.
- Walk/pronk: primeros 25 frames con comando cero; primer cambio a 0,50 s.
  Crouch histórico registra 26 (0,52 s), un tick de su colector anterior.
- Holds móviles: **150 ticks = 3 s**. Stops sorteados: **100 ticks = 2 s**.
  Último hold puede quedar truncado por el final del episodio. Stops seguidos
  se observan como un hold unido; no se interpretan como resampling ausente.
- Hindsight: **100 ticks = 2 s para las tres fuentes**. Walk aporta mayor
  distancia en ese tiempo, no un horizonte diferente que codifique la skill.
- Perfil geométrico: cuatro puntos al 25/50/75/100% del **arco espacial**,
  no a instantes prefijados. Velocidad media = arco XY real / 2 s.
- Historia 8 ticks, secuencia 16/offset 8, ejecución 4 ticks y denoising K=10:
  configuración A0/profile16 existente sin modificación.

`temporal_alignment.json` verifica todos los episodios: holds, inicio cero,
soporte de giros/stops, integración de referencia y `last_action[t]=action[t-1]`.
Error máximo de integración XY ≤5,35e-6 m; yaw ≤2,39e-7 rad;
alineamiento de last_action **exacto**. Stops de comando ocupan 16,68% walk,
15,33% crouch y 15,10% pronk incluyendo arranques: no son el 10% por sorteo.
Comando cero tampoco implica ausencia de oscilación o rebote físico.

## Soporte realmente conseguido

Los valores hindsight siguientes vienen de 10.000 ventanas muestreadas por
fuente (seed49); alturas y velocidades por bins usan todos los clips aceptados.

| Fuente | Velocidad media hindsight mediana / p95 / máximo [m/s] | Altura real mediana [m] | Altura nominal [m] |
|---|---|---:|---:|
| walk | 0,390 / 0,780 / 1,337 | 0,2979 | 0,2932 |
| crouch | 0,241 / 0,413 / 0,496 | 0,1590 | 0,1705 |
| pronk | 0,287 / 0,451 / 0,560 | 0,2601 | 0,2932 |

Walk contiene alcance rápido real. En comandos 1,25–1,50, vx conseguido
tras 0,5 s de cada cambio tiene mediana **1,143** y p95 **1,325 m/s**
(9.687 frames). La cola es escasa y filtrada: no venderlo como experto que
clava 1,5 m/s. No sobreponderar esa cola ni fabricar etiquetas de comandos.

Walk y pronk comparten altura nominal y se solapan en velocidad. No se entrega
`skill_idx`, fase de vuelo ni diagnóstico al modelo. **El conditioning no
identifica de forma única pronk**: geometría, historia y distribución de acciones
pueden preservar distintos modos, pero no garantizan preferencia por uno.

En el piloto corregido pronk se midió vuelo confirmado mediano **120 ms**,
~20,7% de frames con vuelo físico, sin violaciones de límites blandos. Los
centros de pies tienen separación izquierda-derecha mediana ~8,1 cm. El piloto
separado con self-collisions conservó vuelo, pero solo aceptó 16/20 episodios:
es una comprobación pequeña con otra secuencia aleatoria, no prueba de hardware.
El full no guarda esos diagnósticos adicionales; no extrapolar una tasa de
vuelo del piloto a todos sus frames. Los plots muestran el rebote de altura.

## Límites y supuestos que siguen abiertos

1. **Altura pronk:** el experto no fue entrenado con reward instantáneo de
   altura (`track_base_height_reward_scale=0`). Su anotación 0,2932 significa
   nominalmente «de pie», no altura clavadísima. No codificar el gait con una
   tercera altura. Un DPPO de altura estricta puede borrar ese comportamiento;
   revisar retención y semántica de postura antes de lanzarlo.
2. El pronk filtrado no es perfecto: harvest 81,72% y algunos transitorios
   físicos siguen presentes. Seleccionar episodios enteros no demuestra que
   un learner nunca caerá. La mezcla es una prueba exploratoria, no un nuevo
   resultado principal ya validado.
3. Cuatro puntos no resuelven todo el arco XY denso. En ventanas con arco
   >0,10 m, fracción de longitud no resuelta mediana / p95: walk 0,40%/12,10%,
   crouch 0,71%/10,69%, pronk 0,22%/10,86%. No aparece un desfase específico
   enorme de pronk; las colas incluyen inversión de dirección, curvas y wiggle.
   Mantener la definición heredada de velocidad achieved, declarando esta
   aproximación. No confundir arco recorrido con desplazamiento neto al final.
4. Crouch es un artefacto histórico; nuevos expertos usan Isaac Sim 5.1 y un
   contrato light. No son procedencias idénticas ni evidencia de sim2sim.
5. El nuevo walk amplía soporte y se añade una tercera fuente. A 50 epochs hay
   más updates. Sin control con el mismo walk y presupuesto emparejado no se
   puede atribuir causalmente una mejora solo a añadir pronk.

## Integridad y errores corregidos

- Validador real del trainer: todos los campos y **todos los frames** de las
  tres fuentes pasan. Auditoría estructural del combinado: contract_valid=true,
  frame_balance=1.0; no datos incompletos ni archivos sobrescritos.
- Siete tests sin simulación pasan, incluido un batch real del loader profile16
  con augmentation quadruped y relocación/remapeo de los links externos.
  El launcher también se prueba: difusión desde cero, sin resume/fast-trot ni
  overrides que alteren la configuración congelada. Preflight local y sintaxis
  Python/Bash verificados; no se ha ejecutado un train como parte de esos checks.
- Inputs del experto refrescados tras cambios/reset. Primer lote desalineado
  excluido. Corrección exacta y trazable de las filas iniciales registradas
  incorrectamente: 1.070 walk y 83 pronk, sin modificar acciones/movimiento.
  Ver [detalle de ambos errores](COLLECTOR_SYNCHRONIZATION_FIX.md).
- Archivo walk del primer lote excluido archivado sin pérdida en
  `data/datasets/walk_crouch_pronk_v1/excluded_walk_full.tar.xz`; eliminado solo
  su duplicado sin comprimir, recuperable. Ninguna fuente anterior se borró.

SHA256 finales:

```text
walk    67d4f34bde03e76df5a65adb8520c18df1a1094a093be0624ca38c5df57e38d9
crouch  492eaddc081efc3a2554fcc83080e094a126d2268d4853b8a3e45faf4c258bbe
pronk   2d4c3947e2403841883e8c734d1a44fec41908e855e7df9f829e8b887e489ca4
index   ca21d9ba93a62e5c4d3ecbc1b3f1191a2b05d74fc8085401da3a587310b91363
```

Plots locales:
`scripts/diffusion_policy/data/audits/walk_crouch_pronk_v1/full/`:
`hindsight_multiskill_support.png` (condiciones/alcance/balance),
`expert_episode_timing.png` (comando versus movimiento y altura),
`raw_multiskill_support.png` (acciones/distribuciones; PCA no demuestra gait).
Todos se generaron de los datos finales y se revisaron visualmente. Datos,
plots y logs están ignorados por Git; documentos/código son pequeños.

## Acción siguiente

Transferir los datos sellados y lanzar **Phase A desde cero** en cluster con
la configuración A0 conservada: [CLUSTER_COMMANDS.md](CLUSTER_COMMANDS.md).
Actualizar el checkout del cluster con el commit de esta preparación antes
de lanzar; los HDF5 y el SBS se transfieren por separado, no mediante Git.
Al volver el checkpoint, medir retención de pronk, walk rápido, crouch y
arranques antes de preparar DPPO. No cambiar reward para forzar retrospectivamente
el resultado. [Lista completa](NEXT_STEPS.md).
