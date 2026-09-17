# Final fast-tracking candidate: `supported_hybrid_v4`

V4 is an additive experiment. V1--V3 and their checkpoint semantics remain
unchanged. It starts directly from the pure Phase-A WCT policy and keeps the
v3 reward, geometry, height profiles and transition-context distribution.

## Why this intervention

The v3 preview used the route's nominal mean speed to select its geometric
look-ahead, then replaced only the final `v_avg` scalar with the closed-loop
remaining pace. When the robot was early or late, this created a goal tuple
that never exists in Phase A: geometry for one speed and a scalar requesting
another. V4 uses the same remaining pace for preview length and `v_avg`.
This is a representation fix, not a local speed controller.

The v3 sampler also underrepresented demanding but feasible fast cases. V4
crosses its existing 16 geometry/profile cells with two private pace tiers:

- 50% samples span the complete feasible interval `[0.2, ceiling]`;
- 50% span its upper feasible quartile.

The ceiling is computed only at reset from Phase-A support: `0.4 m/s` crouch,
`0.6 m/s` strong turns, `1.2 m/s` moderate turns and `1.5 m/s` open upright
motion. A `0.25 m` reserve around posture boundaries models finite transition
distance. Neither the ceiling, pace tier, curvature nor a gait label enters
the actor, critic or reward. The task still exposes one route, height profile
and global arrival-time objective.

## Deliberately unchanged

- path reward weight `2.0` and CTE success gate `0.10 m`;
- profile-height reward `0.75` and MAE gate `0.04 m`;
- PPO/DPPO hyperparameters, no reference KL and no gait reward;
- route speed targets at most `1.0 m/s`, with `1.5 m/s` recovery headroom;
- the audited 25/25/25/25 hybrid geometry distribution.

Therefore the ablation is interpretable: any change relative to v3 comes from
coherent goal conditioning and better supported fast-task coverage, not reward
retuning.

## Local preflight

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/dppo_diffusion_rl/experiments/supported_hybrid_v4/plot_distribution.py
```

Inspect `route_generation/speed_support_v4.png` and `summary.json`. The script
checks exact 32-cell coverage and that no target exceeds its private ceiling.

## Cluster train

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v4/train_cluster.sbs
```

Or pass an explicit pure Phase-A checkpoint:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/supported_hybrid_v4/train_cluster.sbs /absolute/path/to/phase_a_best.pt
```

The launcher rejects DPPO sources and refuses to overwrite a populated run.
Use `best.pt`, never `last.pt`, for the preregistered final benchmark.

## Decision gates

Primary gates remain those of the final benchmark: joint success, survival,
CTE, height and global mean-speed error. The key v4 comparison is stratified
by route family and speed (`0.65`, `0.85`, `1.0 m/s`), especially hard turns
and walk-to-crouch. `1.0 m/s` on routes whose private ceiling is lower remains
an OOD stress test, not an ID failure criterion.
