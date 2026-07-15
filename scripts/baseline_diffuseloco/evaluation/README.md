# Baseline evaluation

`evaluate_policy.py` runs the frozen velocity-height DDPM in Isaac Lab and writes:

- per-scenario metrics in `static_summary.csv`;
- the height sweep trace in `dynamic_height_response.csv`;
- checkpoint/configuration/latency metadata in `summary.json`;
- height interpolation, velocity tracking, dynamic response and latency figures.

The default grid contains five heights, three commands and three stochastic
repeats. It uses one vectorized environment per condition, so it is not a list
of manual launches.

`--torchscript_denoiser` reduces denoiser dispatch overhead without changing the
checkpoint. `--exec_horizon 8` evaluates the real-time action-chunk deployment;
the scientific receding-horizon reference remains `--exec_horizon 1`.

## Runtime/quality sweep

`../benchmark_runtime.py` evaluates the factorial trade-off between the number
of actions executed per sample and the number of DDPM denoising steps. It runs
the same closed-loop evaluator for every point and aggregates survival,
velocity/height tracking, tilt, action smoothness, p95/p99 latency and deadline
margin into `runtime_quality_summary.csv`. It also writes
`runtime_tradeoff.png` and `quality_tradeoff.png`.

The recommended first sweep is deliberately small enough to be interpretable:

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/baseline_diffuseloco/benchmark_runtime.py `
  --checkpoint scripts/baseline_diffuseloco/runs/walk_crouch_posture_conditioned/best.pt `
  --output_dir scripts/baseline_diffuseloco/evaluations/runtime_sweep_k10_k5 `
  --exec_horizons 1 2 4 6 8 `
  --inference_steps 10 5 `
  --repeats 1 --duration_s 8 --settling_s 2 `
  --latency_samples 50 --torchscript_denoiser
```

The default sweep skips the dynamic height sequence to avoid multiplying the
runtime unnecessarily. After selecting the best two candidates, rerun the
individual evaluator with `--dynamic_segment_s 2` and no `--skip_dynamic`.
The comparison must not select the highest horizon only because it has the
lowest amortized milliseconds: it should first satisfy survival and tracking,
then use the shortest feedback horizon that meets the latency deadline.
