# WCT DPPO v6 final evidence package

This package is a compact copy of the final evaluation generated at:

`scripts/dppo_diffusion_rl/evaluations/wct_final_evaluation_v1_hybrid_v6_20260920`

## Provenance

- code commit: `927bbf4440b9baf8024d9369d44accd7e67b0e03`
- checkpoint SHA-256: `e3614cff8fcc7483fa5fb6cd75cc2b7c562992032f72628eb2a13f5a726f470d`
- dataset SHA-256: `4e323bd426ccb6f503e70ee255b397899e0da2f847eb60fa6a21f147f7a6b730`
- ID route bank SHA-256: `aeb472364efcdef0f244769a3816a00b7590b5983f25616e705a3d2e59b2b138`
- OOD route bank SHA-256: `3abb2d5568a1943a2aaa01febd445c60622b6d722b4769b167156f6ebacdef96`
- policy: `spatial_hindsight_height_profile_ddpm`, schema 8
- inference: 10 denoising steps, execution horizon 4

The cluster benchmark metadata reports `git_dirty_before_evaluation=true`.
The controlled local gait run, using the same checkpoint hash and commit,
reports a clean worktree. Results are therefore retained with exact hashes, but
the cluster dirty flag must be disclosed rather than silently ignored.

## Main benchmark result

Suite rates below are route-level means over the path families in
`benchmark_summary.json`; confidence intervals and per-family values are in the
same file.

| Suite | Independent routes | Task success | Survival | CTE RMSE | Height MAE |
|---|---:|---:|---:|---:|---:|
| ID constant | 400 | 100.0% | 99.92% | 2.43 cm | 1.36 cm |
| ID transition | 400 | 100.0% | 99.83% | 2.81 cm | 1.72 cm |
| ID repeated, crouch start | 400 | 93.75% | 100.0% | 3.61 cm | 3.63 cm |
| ID repeated, walk start | 400 | 84.50% | 99.75% | 4.02 cm | 3.33 cm |
| OOD constant | 300 | 100.0% | 99.89% | 2.53 cm | 1.37 cm |
| OOD transition | 300 | 99.94% | 99.83% | 2.91 cm | 1.73 cm |
| Fast crouch-to-walk | 400 | 78.50% | 97.90% | 6.31 cm | 1.83 cm |
| Fast walk-to-crouch | 400 | 84.90% | 99.00% | 7.28 cm | 3.16 cm |
| OOD fast crouch-to-walk | 300 | 80.67% | 98.67% | 6.57 cm | 1.95 cm |
| OOD fast walk-to-crouch | 300 | 83.78% | 99.33% | 7.45 cm | 3.17 cm |

Interpretation: v6 is very strong on constant and ordinary-transition routes,
including held-out OOD geometry. The unresolved boundary is the coupled
high-speed transition regime, especially cross-track error and speed-gated
success. These suites must not be pooled into one headline success number.

## Controlled gait result

`controlled_gait/` contains a clean-worktree, straight-route check at 0.8 m/s.
It confirms 100% survival over 16 scenarios and 87.5% task success, but the
contact signature is not equivalent to the standalone flying-trot expert. The
v6 actor should therefore be described as exploiting the combined WCT action
manifold, not as provably selecting or reproducing a discrete trot skill.

## Contents

- `benchmark/`: aggregate and route-level CSV/JSON plus overview plots.
- `fast_crouch_to_walk/` and `fast_walk_to_crouch/`: representative transition,
  gait, height, speed, and trajectory plots.
- `controlled_gait/`: compact straight-route evaluation without raw time series.

Raw `.npz`, multi-megabyte per-suite CSV/JSON traces, and duplicate plots stay
in the ignored local evaluation directory and can be regenerated with the
tracked final-evaluation protocol.
