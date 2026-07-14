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
