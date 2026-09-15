# Controlled geometry expansion: `supported_hybrid_v2`

This is the next controlled experiment after `supported_procedural_v1`. It
starts from the same pure Phase-A imitation checkpoint and changes only the
route geometry distribution. Reward, observations, DPPO optimizer, height
profiles, speed envelope, seed and training budget remain fixed.

The only proposed intervention relative to `supported_procedural_v1` is route
geometry:

- 25% unchanged v1 smooth random-curvature routes;
- 25% coherent smooth routes with continuously sampled curvature bias,
  sinusoidal amplitude, frequency and phase;
- 25% continuously sampled waypoint routes with locally rounded corners;
- 25% continuously sampled hard polylines.

The two smooth components remain one 50% top-level family.  The coherent half
adds sustained arcs and C/S curves that independent zero-centred v1 knots
under-sample, without introducing fixed `circle` or `s_curve` templates.
Its desired peak curvature is sampled directly from `[0.25, 0.8] rad/m`, so
there is no artificial probability mass at the upper curvature limit.

Every route is 4 m long.  Waypoint routes contain 2--5 segments, every segment
is at least 0.55 m, and every non-zero turn is sampled continuously between
15 and 90 degrees with an independent left/right sign.  Proper
self-intersections, endpoints closer than 1 m to the start and paths whose
heading accumulates beyond 135 degrees are rejected.  The latter removes
accidental near-U-turns from this first controlled expansion without removing
90-degree corners or alternating S-like turns.
The number of waypoint segments is sampled once from `2--5`; rejection of an
invalid candidate redraws only its lengths and turns.  Therefore the validity
filter does not preferentially replace complex paths with simpler ones.
The 16-cell stratification schedule crosses every geometry family with every
height-profile class, preventing geometry from leaking the requested posture.
The rounded family samples radii in `[0.12, 0.30]` m.  Height profiles, global
average-speed targets, observations, rewards and DPPO are inherited unchanged
from v1.

Generate the preregistered diagnostic plots from the repository root:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/dppo_diffusion_rl/experiments/supported_hybrid_v2/plot_route_distribution.py
```

Inspect `route_generation/route_gallery.png`, `route_coverage.png`,
`paired_corner_realizations.png` and `summary.json` before enabling the
distribution in the Isaac Lab environment.  The paired figure holds the
sampled segment lengths and turn angles fixed and changes only whether each
corner is hard or rounded.

## Training contract

The run deliberately starts from Phase A rather than from the v1 DPPO actor.
This makes the comparison answer one question: whether broader geometry
support improves DPPO, without transferring optimizer state or behavior from
an earlier online distribution. `--require_phase_a_source` rejects an
accidental DPPO checkpoint.

The run keeps the successful v1 settings: 4096 environments, 32 rollout
chunks, actor LR `1e-5`, no reference KL, height reward weight `0.75`, maximum
route speed `0.8 m/s`, remaining-speed budget headroom `1.2 m/s`, 150
iterations and checkpoints every 10 iterations. The hybrid generator is
recorded as contract version 1 in DPPO checkpoints; an optimizer resume with a
different contract is rejected.

From the cluster repository root:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v2/train_cluster.sbs
```

An explicit pure Phase-A checkpoint can be supplied as the first positional
argument:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v2/train_cluster.sbs /absolute/path/to/phase_a_best.pt
```

Output is written to
`scripts/dppo_diffusion_rl/runs/dppo_wct_supported_hybrid_v2`. The launcher
and trainer refuse to overwrite a populated run directory.
