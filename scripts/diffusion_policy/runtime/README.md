# Frozen WC-v3 inference and simulator receiver

The runtime loads the complete hybrid DPPO checkpoint and checks its SHA256,
schema, normalizers and timing contract. It preserves FP32, K10, five refined
reverse steps, delayed histories and four distinct actions per chunk.

| Module | Purpose |
|---|---|
| `contract` | Pinned actor/config validation |
| `controller` | Shared goal builder, delayed histories and action mapping |
| `sampler` | Cached coefficients for the original reverse chain |
| `cuda_graph` | Optional fixed-shape CUDA graph with eager ordered noise |
| `host_buffers` | Transfer a whole chunk, then consume it on the CPU |
| `fixtures`, `benchmark`, `optimize` | Oracle parity and inference timing |
| `worker`, `process_worker` | Single-owner inference with copied inputs and plans |
| `receiver_validation` | Independent paced MuJoCo plant and command receiver |
| `safety` | Fail-latched simulator command admission |

CUDA performance was measured on an external development laptop with an RTX
5070 Laptop GPU. Host buffers do not mean that the neural network runs on CPU.
The independent receiver experiment defaults to CPU; CUDA is an optional
external-computer configuration. The intended robot control computer,
communication and independent actuator watchdog have not been validated.

## Validate the interface

```powershell
python -m unittest discover -s scripts/diffusion_policy/runtime/tests -p 'test_*.py'
```

CUDA tests need the actual checkpoint and the explicit opt-in environment
variable specified in `tests/test_cuda_graph.py`. They otherwise skip.
Original captured fixtures can be replayed with `benchmark` and `optimize`;
the actor SHA, device and histories must match. Use `--help` for the complete
arguments. Do not widen tolerances or replace the actor to obtain a pass.

## Independent receiver experiment

```powershell
python -m scripts.diffusion_policy.runtime.receiver_validation --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --model <validated_solo12.xml> --validation <fk_mass_inertia_validation.json> --manifest <development_manifest.json> --dependency-dir <mujoco_dependency_directory> --output-dir output/receiver_trial --device cpu --duration-s 24
```

The supplied development manifest must contain `coherent_smooth_0.35_walk`.
The inference worker runs in a separate process by default, with a thread mode
available for diagnosis. It initializes its device context before the plant
starts and restores RNG after warmup. The receiver uses a 20-ms period and a fixed 15-ms dispatch
phase; physics advances independently in discrete 5-ms steps. This delay
changes execution timing, so it is a separate engineering experiment rather
than an equivalent replacement for the synchronous task evaluator. Results
record late plans, timestamps, dispatches and physical failures.

`--inject-delay-ms 90` deliberately delays inference. A missing or late plan
terminates the simulated trial without dispatching an old chunk. Python
threads on Windows are best-effort scheduling. Stopping a simulation is not a
hardware emergency action or a hard realtime guarantee. No robot connection
is opened by these modules.

The [final validation protocol](../../../docs/project/validation.md)
specifies the source/model freeze and post-selection route evaluation.
