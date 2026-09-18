# Final coverage correction: `supported_hybrid_v5`

V5 is the final preregistered WCT DPPO train. It starts from the pure Phase-A
imitation checkpoint. It does **not** fine-tune v3 or v4.

## Evidence behind the change

The complete frozen-bank evaluation of v3 already established that the actor
solves constant and single-transition ID/OOD routes (99.6--100% success), with
2.4--3.1 cm CTE RMSE and 1.5--1.8 cm height MAE. Its remaining failures were
localized:

- repeated binary height composition at 0.45 m/s;
- global deadlines above 0.75 m/s when a route contains a short crouch
  section, where speed and CTE gates fail but survival remains high.

V4 changed both task coverage and the observation at once. Its nominally fast
tier mostly sampled constant upright routes, while fast transitions remained
rare. Pace-consistent preview also changed the spatial discretization whenever
timing debt changed. The result was safer but slower: the exact paired preview
had speed ratios around 0.72--0.80 versus 0.95--0.97 for v3. V4 is therefore a
rejected ablation, not the base of this run.

## Single intervention

V5 restores v3 observation semantics and keeps its reward, PPO settings,
geometry generator, CTE gate and height gate unchanged. Only reset-time task
coverage changes:

- **50% v3 replay:** the exact v3 profile and speed draw is retained;
- **25% fast single transitions:** both directions are balanced, the crouch
  section is continuously sampled in `[0.8, 1.2]` m, and route-average speed
  is sampled from the upper feasible quartile. With
  `c = min(0.85, private ceiling)` and
  `l = 0.65 + 0.75 (c - 0.65)`, the target is `Uniform(l, c)` m/s;
- **25% repeated profiles:** both starting postures are balanced, binary
  section lengths are continuously randomized around the 0.8 m evaluation
  case, and speed lies in `[0.35, 0.45]` m/s.

Every bucket is balanced over smooth-v1, coherent-smooth, rounded-waypoint and
hard-waypoint geometry. Fast labels are assigned only where the private
Phase-A support model admits at least 0.65 m/s. If an asynchronous reset batch
has no feasible matched replacement, that sample falls back to v3 replay; an
impossible deadline is never inserted.

The upper-quartile draw is deliberate task-frontier sampling. The 50% v3
replay already retains the broad/easy speed distribution, while the dedicated
fast bucket must put useful mass near the measured 0.75--0.85 m/s failure
range. Sampling its whole feasible interval overrepresented already-solved
0.65--0.75 m/s cases. This changes only which feasible deadlines are sampled;
it neither supplies a local speed controller nor changes the actor inputs.

This is sampling, not a controller. The actor still sees only path preview,
required height profile, terminal pose and one global remaining-speed budget.
It receives no local speed schedule, curvature, feasibility ceiling, gait
label or skill index. It must discover where to use fast trot and where to
slow down while satisfying one route-average deadline.

## Deliberately unchanged

- v3 preview (`pace_consistent_preview=False`);
- path reward `2.0`, CTE success gate `0.10 m`;
- height reward `0.75`, profile MAE gate `0.04 m`;
- canonical DPPO optimizer and no reference KL;
- four hybrid geometry families and 4 m routes;
- recovery headroom of `1.5 m/s`.

Targets above `0.85 m/s` remain benchmark stress/OOD conditions. They are not
valid ID gates: the frozen feasibility audit showed that 0.95 and 1.0 m/s are
outside the conservative support for these composite routes.

## Local preflight

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/dppo_diffusion_rl/experiments/supported_hybrid_v5/plot_distribution.py
```

Inspect `route_generation/supported_hybrid_v5.png` and `summary.json`.

## Cluster train

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v5/train_cluster.sbs
```

Or provide the absolute pure Phase-A checkpoint:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v5/train_cluster.sbs /absolute/path/to/phase_a_best.pt
```

The launcher rejects a DPPO source, refuses to overwrite an existing run, and
saves every five iterations. Evaluate `best.pt` plus periodic checkpoints near
the peak; never use `last.pt` merely because it is last.

## Final decision rule

Select the checkpoint on a frozen validation bank, then run the untouched
`wct_final_evaluation_v1` test bank once. Compare Phase A, v3 and v5. V5 must:

1. preserve v3's near-perfect constant/single-transition ID and OOD behavior;
2. improve repeated-profile success at 0.45 m/s;
3. improve fast-transition speed and CTE at 0.65/0.75/0.85 m/s without a
   material survival regression.

The 0.95/1.0 m/s suites are reported as boundary stress tests, not optimized
or used for checkpoint selection. If v5 does not improve these prespecified
gaps, retain v3 and report the supported 0.65--0.75 m/s range rather than
retuning the reward on the test set.
