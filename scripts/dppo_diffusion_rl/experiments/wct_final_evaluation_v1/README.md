# Frozen route benchmark

[Running guide](../../../../docs/project/running.md#evaluation) · [Metrics](../../../../docs/project/results.md#what-success-means)

The `route_banks/` directory contains materialized ID and legacy stress references and their checksums. Geometry is reused across targets, postures, actors and online seeds.

One independently sampled geometry is the statistical unit. Multiple conditions on it are paired observations, not new geometries. Keep ordinary, legacy stress, fast-transition and repeated-profile suites separate.

Archived success omits the training CTE gate and excludes 0.25 m around height changes. Adherence success adds active time-weighted CTE RMSE ≤10 cm. Report safety, arrival and continuous errors alongside success.

Public Python entry points define the benchmark; site-specific allocation and Slurm launchers are local.
