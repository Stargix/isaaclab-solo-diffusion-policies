# Checkpoint Phase A recibido: diagnóstico y decisión antes de DPPO

**Actualización posterior (2026-10-13):** la recomendación exploratoria de
piloto con reward sin cambios queda sustituida por el contrato opt-in de
[postura nominal WCP](../../../dppo_diffusion_rl/experiments/wcp_nominal_posture_v1/README.md).
La telemetría adicional del experto mostró sensibilidad a la fase de salto
en la altura terminal instantánea; el nuevo contrato separa postura media de
seguridad instantánea, sin rewards de gait ni IDs. Los resultados del preview
de este documento no cambian y los experimentos WC/WCT anteriores se conservan.

## Identidad y alcance

Checkpoint: `scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/best.pt`.
SHA256: `40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886`.
Procedencia: job 4071, Git limpio `9003dc399bc20159aec7a7ccdedb88ba4df08f7c`,
Phase A desde cero, dataset seleccionado `aa45ce00ffeb72a16dbd4ea3145b179503b04def6be1c70db072a25e24c1729b`.
Best: epoch 48, update 89616, loss de validación 0.00341553. El paquete recibido
incluye config y procedencia, **no el log completo ni last.pt**: no permite
reconstruir las curvas ni certificar el cierre de las 50 epochs.

Pesos y normalizadores sin NaN ni infinitos inesperados; rangos de acciones
válidos; 6884 episodios de train y 362 de validación, sin intersección de nombres
de episodio/fuente. Los `-inf` de las máscaras causales de atención son intencionales.
Schema 8 / profile16 / Transformer 128 / K10 / hindsight 2 s / quadruped coinciden
con la preparación. Las rutas Linux del checkpoint son procedencia, no impiden
inferir aquí: el actor usa sus normalizadores guardados, no necesita abrir esos paths.

No se modificó ni entrenó el actor. No se cambiaron rewards, arquitectura,
datasets ni algoritmos. Scripts de diagnóstico, logs y plots quedan ignorados
por Git, bajo `scratch/wcp_partition_preview_20261012/` y
`scripts/diffusion_policy/evaluations/wcp_partition_preview_20261012/`.

## Pruebas ejecutadas localmente

Isaac Sim 5.1, RSL-RL 3.1.2, CUDA; nominal sin ruido, eventos, pushes ni delay.
No son una evaluación de robustez ni sustituyen al benchmark congelado.

1. **48 escenarios:** recta y fragmento curvo de 4 m, dos repeticiones, alturas
   0.2932/0.1705, velocidades 0.2/0.35/0.5/0.65/0.85/1.0, duración 12 s,
   K10 y ejecución de 4 acciones. El fragmento truncado del generador `s_curve`
   no es una S completa ni una prueba de OOD agresivo.
2. **16 escenarios:** recta, 0.2/0.35/0.5/0.85, ambas alturas, dos repeticiones,
   6 s; solo cambia ejecución a sus 8 acciones futuras nativas.
3. **8 escenarios:** de pie, mismas cuatro velocidades, 6 s, ejecución 8;
   stand-hold inicial reducido de 25 a 9 ticks. No arregla el resultado.
4. **Experto pronk V8:** 4 robots a 0.2 y 4 a 0.5, 6 s scored después de
   0.5 s de warmup. 8/8 sobreviven; vuelo físico y ciclos verificados por el
   evaluador nativo, no inferidos de PCA o velocidad.
5. **Traspaso experto → alumno:** 12 robots, cuatro a cada 0.2/0.35/0.5.
   Dos segundos de experto, luego seis segundos del alumno, con historias
   de estado/acción/goal actualizadas y sin teleportar estados. Llegan vivos
   al traspaso 3/4, 4/4 y 3/4: los dos fallos de priming no se ocultan.
   **Los diez que entran vivos también terminan fallando tras el traspaso.**
   Este priming solo diagnostica; no cuenta como actuación autónoma del alumno.
6. **Imitación teacher-forced:** ocho episodios held-out por experto, tres
   anchors por episodio y tres semillas de denoising. No se usa la validación
   para entrenar, escoger ejemplos exitosos ni ajustar normalizadores.

Se conserva cada intento. Las métricas de movimiento excluyen reset continuations
y muestras posteriores a primera llegada; las de gait excluyen el primer segundo.
El tiempo de los estados post-step se corrige a `(step+1)*dt`. Los contactos de
la evaluación general usan fuerzas >1 N: intervalos descargados >=60 ms no son,
por sí solos, prueba geométrica de vuelo. El experto/traspaso sí registran ese
diagnóstico geométrico. No confundir supervivencia con éxito de tarea ni con timing.

## Resultado principal

| Condición de pie | Intentos sin fallo antes de llegada/fin | Mediana de tiempo al fallo |
|---|---:|---:|
| 0.2 m/s | 0/4 | 0.95 s |
| 0.35 m/s | 0/4 | 1.03 s |
| 0.5 m/s | 0/4 | 1.38 s |
| 0.65 m/s | 0/4 | 4.26 s |
| 0.85 m/s | 0/4 | 2.80 s |
| 1.0 m/s | 0/4 | 3.16 s |

Crouch: 20/20 sin fallo en el bloque 0.2–0.85; 3/4 a 1.0.
CTE RMSE mediano ~0.9–1.5 cm hasta 0.85, pero no clava el tiempo:
por ejemplo en recta pide 0.2 y llega con velocidad media sobre progreso
~0.35–0.37; pide 0.35 y llega con ~0.48. La velocidad local rápida observada
en crouch no convierte su soporte de demostraciones en soporte nativo a 1.0.
El bloque crouch >0.5 es exploratorio, no un gate ID de entrenamiento.

Con ejecución 8, los ocho casos upright de la prueba correspondiente vuelven
a fallar; los ocho crouch sobreviven. Cambiar replanning no es la solución
observada. Con warmup corto tampoco sobreviven los ocho upright.

Experto V8: vuelo físico ~22–26% de muestras y ciclos validados con mediana
de duración 120–140 ms a 0.2/0.5. El alumno, tras priming correcto, tiene
algo de vuelo antes de fallar, pero su altura media en esos prefijos baja
a ~0.223–0.229 m. **No mantiene el gait estable**, aunque no lo haya borrado
por completo de su representación.

Teacher-forced: error RMS de targets articulares futuros ~0.038–0.041 rad
(action scale 0.25) en los tres expertos. La pérdida epsilon media por experto
es ~0.0034–0.0046. Esto acredita una aproximación a acciones en contextos reales,
no estabilidad al cerrar el lazo. Una política multimodal no se juzga únicamente
por error contra una única acción objetivo, y pérdida epsilon no equivale a CTE.

## Qué puede y qué no puede concluirse

- No hay evidencia de checkpoint corrupto, mezcla equivocada o schema incompatible.
- Pronk original sí funciona en esta máquina: culpar a que SOLO12 no puede
  saltar o al backend local no explica por sí solo los fallos del alumno.
- Fallan tanto entradas nominales como continuación desde pronk experto.
  El problema no se resuelve solo añadiendo warmup ni usando ejecución 8.
- La acumulación de errores fuera de los historiales observados es consistente
  con estas pruebas; **no se ha aislado una única causa matemática** ni demostrado
  que toda la explicación sea tamaño de red, arquitectura o duración del train.
- No hay RL en este checkpoint: el reward DPPO no puede haber causado este fallo
  offline. Sí puede borrar modos de locomoción en la fase siguiente si ofrecen
  menos retorno para la misma tarea.
- Crouch es utilizable como modo representado; los tres modos autónomos todavía
  no están validados. Más epochs de la misma BC no son una solución demostrada.

## Decisión sobre DPPO

**Compatible como inicialización de un piloto DPPO, no certificado como una
base estable de tres gaits ni como experimento final listo para publicación.**
La comparación anterior Phase A → DPPO del proyecto justifica probar recuperación
online; no garantiza que la recuperación preserve pronk en vez de fabricar walk lento.

No hace falta cambiar la red, el loader o las matemáticas de DPPO por estos
resultados. Tampoco se justifica subir automáticamente las penalizaciones de caída
o altura. Lo siguiente es preparar un **run nuevo y corto (p. ej. 50 iteraciones)**
desde este best, sin tocar WC/WCT existentes, con el contrato v3 que funcionó:

- Actor de este Phase A; critic/optimizer nuevos; `--require_phase_a_source`.
- `supported_hybrid_v3`, geometrías existentes, velocidades objetivo hasta 1.0,
  presupuesto hasta 1.5, CTE 0.10, path weight 2.0; no cambiar familias a la vez.
- Altura **explícita 0.75**, tolerancia MAE 0.04 y terminal 0.05, como el launcher
  v3. **El default actual de profile-height weight es 2.0**, no omitir la bandera
  y fingir que se está repitiendo la receta anterior.
- KL de referencia inicialmente 0, como la recuperación anterior; un ancla fuerte
  al actor que se cae puede frenar su corrección. No afirmar que KL garantiza gait.
- Ejecución 4, K10 y cinco denoising steps ajustables: protocolo anterior; la
  prueba de ejecución 8 no mostró una ventaja que justifique otro cambio simultáneo.
- Metas lentas y rápidas y ambos sentidos de postura presentes. No ID de skill,
  fase gait, reward diagonal ni bonus de vuelo. Si el timing usa 0.2, 4 m requieren
  unos 20 s: el límite existente de 24 s es compatible; no reducirlo a 8 s.

**Gate tras ese piloto:** recuperar estabilidad en upright lento y rápido;
medir contactos/vuelo, no solo return/success; probar misma ruta y altura con
solo el ritmo distinto y observar continuación/transiciones. Antes de declarar
tres comportamientos, ampliar réplicas y verificar vuelo geométrico estable.
Si solo aprende walk lento, es recuperación de tarea, **no éxito de esta ablation**.
No lanzar cientos de iteraciones o otra Phase A automáticamente ante ese resultado.

### Precaución con la altura

El experto pronk tiene altura media ~0.262 y MAE instantáneo respecto a 0.2932
~3.0–3.2 cm. Solo ~55–58% de sus muestras están dentro de 4 cm instantáneos.
Ese porcentaje **no es** el gate de MAE de ruta: no convertirlo en un fracaso
del experto ni decir que viola inevitablemente el contrato. La media anterior
es temporal, mientras que el entrenamiento agrega por distancia.

La altura nominal compartida no es su media real. Height tracking puntual y
vertical-velocity cost favorecen movimientos menos oscilatorios a igualdad de
tarea, pero todavía no se ha demostrado que sean el cuello de botella dominante.
No endurecerlos. Si desaparece vuelo al mejorar la tarea, revisar explícitamente
la semántica de altura nominal/clearance y la retención del prior; no introducir
otra altura exclusiva de pronk o un reward de gait para fabricar la selección.

## Plots y reproducción

Artefactos relativos al repositorio (ignorados, no publicados):

- `scripts/diffusion_policy/evaluations/wcp_partition_preview_20261012/constants/preview_paths.png`
- `scripts/diffusion_policy/evaluations/wcp_partition_preview_20261012/constants/preview_contacts_height_speed.png`
- `scripts/diffusion_policy/evaluations/wcp_partition_preview_20261012/expert_handover/expert_to_diffusion.png`
- Resúmenes `compact_results.json`, `preview_gait_summary.json` y `summary.json`.
- Auditorías de integridad/BC y scripts de reproducción en
  `scratch/wcp_partition_preview_20261012/`.

## Estado actualizado

- [x] Datos completos, alineados, congelados y seleccionados por soporte real.
- [x] Phase A entrenada; checkpoint/config/procedencia recuperados.
- [x] Integridad, separación por episodios e imitación teacher-forced comprobadas.
- [x] Preview closed-loop, alternativas de ejecución/warmup y experto de control.
- [x] Diagnóstico de continuación desde gait experto.
- [ ] Recuperar log si se necesitan curvas de convergencia o certificar 50/50.
- [ ] Preparar/lanzar piloto DPPO con flags explícitos y output independiente.
- [ ] Verificar estabilidad **y retención** tras DPPO; después transiciones y timing.
- [ ] Solo si pasa: réplicas y control WC con exactamente el mismo soporte walk.
- [ ] Sim2sim físico, optimización de inferencia y puesta en marcha real siguen
  pendientes por separado; estos resultados no sustituyen a esa transferencia.

## Base bibliográfica y límite del argumento

[DiffuseLoco](https://arxiv.org/abs/2404.19264) fundamenta aprender secuencias
de locomoción multiskill offline y ejecutarlas en horizonte recedente; no implica
que todo experto con oscilaciones se transfiera automáticamente a un alumno.
[DPPO](https://arxiv.org/abs/2409.00588) y su
[implementación oficial](https://github.com/irom-princeton/dppo) fundamentan
ajustar el denoising con policy gradients desde un prior preentrenado. Exploración
estructurada no es una restricción que garantice conservar cada modo de gait.
Los datos locales, no esos papers, respaldan el diagnóstico de este checkpoint.
