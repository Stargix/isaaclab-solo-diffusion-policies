# WCT final evaluation v2: diagnostic protocol

This is an **evaluation-only** protocol for the frozen WC DPPO deployment
candidate. It does not modify rewards, route sampling, checkpoints, or the
training distribution. Its purpose is to make the existing result
interpretable before deciding whether a new training experiment is warranted.

## Questions and controls

1. **Posture transitions:** requirement changes are aligned by travelled route
   distance, not by an arbitrary percentage or a single straight path. The
   output separates both directions and requested speeds. It shows normalized
   posture response and CTE separately from tangent speed and schedule debt.
   Bands are descriptive route variability (10--90%), not confidence
   intervals.
2. **Intermediate posture:** `0.20, 0.23, 0.26 m` are held-out heights. They
   are evaluated separately from endpoint walk/crouch conditions, including
   routes that transition into and out of the intermediate values.
3. **Geometry boundary:** an independent bank varies S curvature/reversals,
   hard turns, rounded turns and compound turns. It reports descriptors, not a
   single pooled "OOD" percentage.
4. **Temporal allocation:** the actor is evaluated with its normal closed-loop
   remaining-route pace condition, an initial-pace freeze, and a deterministic
   shuffled pace assignment. These are counterfactual diagnostics only; the
   task, checkpoint and reference route remain unchanged.

`dynamic` is the actor's normal mode. `frozen` tests whether it needs the
closed-loop pace signal. `shuffled` preserves the distribution of pace values
but breaks their pairing to route conditions. The latter is deliberately
off-task and must be reported as an intervention, not a performance baseline.

The plotted "schedule debt" is a diagnostic against uniform progress,
`min(v_target * t, route_length) - route_progress`; positive means behind that
reference. Uniform progress is **not** imposed as a local target or reward. The
task requirement remains average traversal speed, so slowing before a difficult
turn and recovering later is valid.

## Frozen route bank

Create once, inspect the gallery, then commit the `.npz` plus manifest before
cluster evaluation:

```bash
python scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v2/prepare_route_banks.py
```

The v1 ID/OOD banks are reused for the transition and intermediate-height
suites. The v2 bank is *only* the geometry sweep, so the new diagnostics stay
directly comparable with v1.

## Outputs

For every trace suite run:

```bash
python scripts/diffusion_policy/evaluation/diagnostic_v2.py <suite_dir>
```

This writes `route_diagnostics.csv`, `height_transition_events.csv`,
`height_transition_event_aligned.png` and
`temporal_allocation_event_aligned.png`,
`intermediate_height_calibration.png`, `geometry_speed_capability.png`, and
`trajectory_speed_colored.png`. Independent geometry is still
`(path_shape, repeat)`; speed/height settings are repeated conditions on it.

The transition CSV reports thresholded diagnostics whose definitions are also
stored in `diagnostic_manifest.json`:

- anticipation distance: first sustained 20% response over 10 cm of route,
  measured relative to the achieved pre-event plateau and restricted to the
  final 0.8 m before the boundary;
- overshoot: maximum excursion past the new height in the transition
  direction;
- settling distance: first point within 2 cm for a sustained 20 cm of route.

These thresholds are operational definitions for this study, not universal
locomotion standards. For repeated schedules, pre/post metrics are clipped at
the neighbouring transition so one event cannot contaminate another.

After all three pace interventions, `summarize_diagnostic_v2.py` first averages
repeated speed/height/direction conditions within each materialized route and
then produces route-stratified bootstrap 95% intervals. The resulting interval
does **not** represent checkpoint/training-seed uncertainty; those seeds remain
a separate level of evidence.

## Interpretation guardrails

- Do not turn a high-speed CTE error into a smaller loss with a speed-dependent
  sigmoid. The bounded Huber tracking error already protects optimization;
  speed-dependent tolerance would redefine the task as permitting larger
  corner-cutting at speed.
- Treat `0.95--1.0 m/s` on sharp routes as a boundary/stress condition, not an
  ID guarantee. Report the geometry descriptors and feasible envelope.
- A future reward or distribution change is justified only if this protocol
  finds a supported, feasible condition in which the actor allocates pace
  incorrectly despite having physical authority.
- Do not compare discrete-corner curvature directly with smooth-path
  curvature. Hard turns are parameterized by corner angle; smooth families by
  peak curvature. `geometry_speed_capability.png` therefore uses one panel per
  family rather than a pooled difficulty axis.
- The coloured trajectories are selected by a fixed median-difficulty rule at
  the requested speed nearest 0.85 m/s. They are qualitative illustrations,
  not selected best cases and not statistical evidence.

## Relation to prior evaluation practice

This protocol follows the useful parts of prior work without implying that the
tasks are identical:

- [DiffuseLoco](https://arxiv.org/abs/2404.19264) reports stability and velocity
  tracking alongside skill execution and validates design choices through
  ablations. That motivates keeping survival, path, posture and timing metrics
  separate here.
- [LocoDiff](https://arxiv.org/abs/2411.08832) visualizes the realized robot
  state during skill transitions. That motivates event-aligned achieved
  posture rather than reporting only transition success percentages.
- [DPPO](https://arxiv.org/abs/2409.00588) evaluates online refinement relative
  to a fixed diffusion prior. Here the checkpoint and materialized routes are
  frozen across the temporal-conditioning interventions.
- [Reliable RL evaluation](https://arxiv.org/abs/2108.13264) cautions against
  conclusions from point estimates with few runs. Route-bootstrap intervals
  are therefore included, while explicitly not being presented as a substitute
  for the already planned checkpoint-seed comparison.
- [Skill-Nav](https://arxiv.org/abs/2506.21853) and
  [path-conditioned RL](https://leggedrobotics.github.io/rl-path-following/)
  motivate evaluating continuously sampled waypoint/path families and explicit
  robustness conditions rather than only a short catalogue of named routes.

The path-conditioned navigation literature motivates waypoint/path robustness
tests, but its objectives are not silently imported: this project explicitly
measures path adherence, posture and average-speed timing, so those quantities
remain distinct in both tables and plots.
