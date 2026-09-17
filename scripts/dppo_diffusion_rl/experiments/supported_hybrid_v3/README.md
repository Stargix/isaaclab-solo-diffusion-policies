# Final train candidate: `supported_hybrid_v3`

This experiment is additive: it does not modify the reproducible
`supported_hybrid_v2` generator or any earlier run. It starts from the same
pure Phase-A WCT checkpoint and tests whether DPPO can learn precise path,
posture and arrival-time control without local speed targets or skill labels.

## Intervention

V3 preserves the four v2 geometry components and four posture-profile classes.
It changes only the following task-contract fields:

- transition boundaries cover `[0.8, 3.2] m` instead of `[1.6, 2.4] m`;
- transition contexts are 50% uniform, 25% low-curvature and 25%
  high-curvature, with no context label exposed to the policy;
- a private Phase-A support envelope rejects impossible route-average
  deadlines; only one global desired mean speed is exposed;
- route-average targets are capped at `1.0 m/s`, with `1.5 m/s` remaining-pace
  headroom so the actor can recover delay on open sections;
- path reward weight changes from `1.25` to `2.0`;
- joint success requires distance-weighted route CTE RMSE at most `0.10 m`.

The 60 cm corridor remains only a catastrophic termination. There is no local
speed profile, turn-risk observation, gait reward, action smoothing or skill
index. Progress, average-speed potential, height, yaw, safety, PPO and the
diffusion architecture are unchanged.

The private feasibility envelope reuses the established Phase-A caps from the
supported generators (`0.4/0.6/1.2 m/s`) and integrates them over route
distance. It is retained as an auditable reset diagnostic but is not part of
the actor goal, critic features or reward.

## Preflight plots

Generate the v3-only diagnostics locally:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/dppo_diffusion_rl/experiments/supported_hybrid_v3/plot_distribution.py
```

Inspect:

- `route_generation/transition_gallery.png`;
- `route_generation/speed_support.png`;
- `route_generation/summary.json`.

The unchanged geometry support remains documented by the v2 route-generation
figures.

## Cluster train

From the cluster repository root:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v3/train_cluster.sbs
```

An explicit pure Phase-A checkpoint may be supplied:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v3/train_cluster.sbs /absolute/path/to/phase_a_best.pt
```

The launcher rejects DPPO sources through `--require_phase_a_source` and
refuses to overwrite a populated output directory. Results are written to
`scripts/dppo_diffusion_rl/runs/dppo_wct_supported_hybrid_v3`.

## Preregistered gates

- ID joint success >= 98%;
- ID survival >= 99%;
- active CTE RMSE <= 0.05 m;
- transition CTE p95 <= 0.10 m;
- profile-height MAE <= 0.04 m;
- mean-speed absolute error <= 0.05 m/s;
- no regression on constant walk/crouch relative to v2.

Evidence of slowing in demanding sections and later recovery is a secondary
research outcome. It must be established from speed-versus-progress and foot
contact plots; it is not inferred from target speed or semantic gait labels.
