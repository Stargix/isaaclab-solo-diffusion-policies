# WCT final evaluation v2: diagnostic protocol

This is an **evaluation-only** protocol for the frozen WC DPPO deployment
candidate. It does not modify rewards, route sampling, checkpoints, or the
training distribution. Its purpose is to make the existing result
interpretable before deciding whether a new training experiment is warranted.

## Questions and controls

1. **Posture transitions:** requirement changes are aligned by travelled route
   distance, not by an arbitrary percentage or a single straight path. The
   output shows height error, CTE, tangent speed and schedule debt in both
   directions, with median and 10--90% route bands.
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
`geometry_speed_capability.png`. Independent geometry is still
`(path_shape, repeat)`; speed/height settings are repeated conditions on it.

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
