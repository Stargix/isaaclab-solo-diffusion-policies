# WCT final evaluation v1

This protocol qualifies the current walk--crouch--fast-trot (`WCT`) policy
before a walk--crouch (`WC`) ablation is trained or compared.  It changes no
policy input, reward, checkpoint or skill representation.  In particular,
neither the policy nor the evaluator supplies a skill index.

## Questions

1. Does WCT follow previously unseen paths, rather than a small set of reused
   geometries?
2. Is cross-track error bounded during walk/crouch transitions instead of
   being hidden by post-arrival samples?
3. Does the fast expert expand the temporal envelope, and does the policy
   accelerate and brake as route constraints change?
4. How far does the result transfer beyond the geometry distribution used by
   `supported_hybrid_v2`?

The protocol is intentionally an evaluation iteration.  A failed gate locates
the next intervention; it is not permission to change several reward terms at
once.

## Statistical unit and frozen paths

The route, not an episode condition, is the independent unit.  The checked-in
banks contain:

- ID: 50 paths from each of `procedural`, `coherent_smooth`,
  `rounded_waypoint` and `hard_waypoint` (200 paths);
- geometric OOD: 50 randomized paths from each of `ood_arc`, `ood_s_curve`
  and `ood_corner` (150 paths).

Speeds, heights, transition directions and transition locations reuse these
paths.  `summarize_benchmark.py` first averages all conditions belonging to a
`(path_shape, repeat)` pair and only then bootstraps routes.  It therefore does
not turn 3,600 correlated conditions into 3,600 independent samples.

The bank files use a versioned, compressed, pickle-free NumPy schema.  Their
SHA-256 hashes are recorded both in `route_banks/manifest.json` and every
evaluation's `evaluation_summary.json`.  Regeneration is deliberately refused
unless `--overwrite` is given:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/prepare_route_banks.py
```

The ID bank samples the exact registered hybrid families but uses test-only
seeds.  OOD is not a renamed ID set:

- `ood_arc`: sustained curvature and accumulated heading beyond the declared
  train envelope;
- `ood_s_curve`: higher-frequency, peak curvature `0.85--1.10 rad/m`;
- `ood_corner`: a continuously sampled hard turn between 90 and 120 degrees.

OOD and ID are always reported separately.  Straight/circle/S/right-angle
legacy templates remain a small backwards-comparable screen; they are not
treated as independent statistical evidence.

## Suites

| suite | purpose | independent routes | repeated conditions |
|---|---|---:|---|
| `id_constant` | basic walk and crouch | 200 | speeds 0.20/0.35/0.45 |
| `id_transition` | W->C and C->W composition | 200 | fractions 0.40/0.50/0.60, three speeds |
| `fast_crouch_to_walk` | leave a 0.8 m restricted section | 200 | 0.65/0.75/0.85/0.95/1.00 m/s |
| `fast_walk_to_crouch` | brake into a final 0.8 m restricted section | 200 | same speeds |
| `ood_constant` | geometric transfer | 150 | two heights, three speeds |
| `ood_transition` | composition under geometric shift | 150 | two fractions/directions, three speeds |
| `legacy_templates` | historical qualitative comparison | not a statistical bank | old templates |

`0.95` and `1.00 m/s` are boundary/OOD temporal tests.  With 0.8 m of crouch
near 0.4 m/s and an open-section support near 1.5 m/s, exactly 1.0 m/s average
can be physically infeasible.  Failure there does not invalidate the ID gate;
it identifies the capability boundary.

## Metrics

Canonical tracking metrics stop at first arrival, or at physical failure when
arrival never occurs:

- joint task success and strict arrival;
- full-horizon survival and base contact, kept separate from arrival;
- terminal position/yaw and arrival mean-speed error;
- active cross-track mean, RMSE, p95 and maximum;
- active height MAE, signed bias and p95;
- cross-track RMSE and height MAE before, within and after a 0.25 m spatial
  transition window.

Trajectory and height plots are also truncated at first arrival.  A gold
diamond marks the spatial height transition.  This prevents the robot's
post-task standing behavior from making path or height tracking look better.

The two fast suites additionally save foot-contact traces.  The offline
temporal report contains local tangent speed, schedule debt, duty factor,
flight fraction and diagonal/ipsilateral contact correlation per 0.4 m route
segment.  These are post-hoc diagnostics, not rewards and not gait labels.

## Running on the cluster

From the repository root, using the default checkpoint path:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs
```

Or provide the checkpoint and output root explicitly:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  /absolute/path/to/dppo_wct_hybrid_procedural.pt \
  /absolute/path/to/wct_final_evaluation_v1
```

Each evaluator output is protected by `--require_empty_output_dir`.  The
launcher can therefore be resumed only by deliberately choosing a new output
root or removing an incomplete directory after inspecting it.

## Reading the result

Primary engineering gates, fixed before looking at the result:

- ID joint success >= 90%;
- ID survival >= 95%;
- terminal position <= 0.15 m and arrival mean-speed absolute error <= 0.08
  m/s, matching the task contract;
- active height MAE <= 0.04 m;
- active cross-track RMSE near or below 0.06 m and route-clustered p95 <= 0.12
  m, the current path reward scale;
- no systematic transition-window spike that is hidden by the overall mean.

The 0.60 m termination corridor is a catastrophic-failure boundary, not a
tracking-quality threshold.

Fast-trot utilization is supported only when local evidence agrees: the robot
must gain speed in the open section, reduce it in the restricted section,
recover schedule debt when feasible, and show a contact regime
consistent with the fast expert.  Completing a fast route by uniformly
speeding up walk is useful performance but is not evidence of implicit skill
composition.

Outputs of interest:

- `<root>/summary/benchmark_summary.json`;
- `<root>/summary/aggregate_metrics.csv`;
- `<root>/summary/route_level_metrics.csv`;
- `<root>/summary/benchmark_overview.png`;
- `<root>/fast_*/temporal_allocation_summary.json`;
- `<root>/fast_*/temporal_allocation.png`.

Only after this WCT protocol passes should WC be evaluated on the exact same
banks.  The paired comparison then tests negative transfer on slow tasks and
capability expansion on the fast suites without changing geometry.
