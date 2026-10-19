# WC-v3 local runtime and parity gate

Implementation hand-off for closing sequential replay and adding a separate
physics backend: [sim2sim specification](../experiments/wc_wct_ablation_v1/SIM2SIM_IMPLEMENTATION_HANDOFF.md).
This is a staged plan, not a claim that sim2sim has already been implemented.

This is a small simulator-independent inference wrapper for the frozen DPPO
checkpoint `checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt`. Its runtime sampler
caches the fixed ten-step DDPM coefficients on the inference device and avoids
materializing PPO-only reverse-chain/log-probability tensors. This is an
algebra-preserving FP32 implementation, gated against the native evaluator by
same-backend oracle replay. It intentionally
accepts only the pinned WC-v3 contract (SHA-256 in `contract.py`): schema/task
v8, profile16, proprio/action/goal dimensions 30/12/16, history 8, DDPM K=10,
5 fine-tuned reverse steps, execution offset 8 and four actions per plan.
`checkpointing.build_inference_policy` restores the actual hybrid actor; the
Phase-A EMA alone is not used as a substitute.

The current module implements the software runtime and a same-Isaac-backend
oracle replay/latency benchmark. It is not a MuJoCo backend, hardware driver,
real-time guarantee or sim2sim result. The profile16 route goal uses the same
shared goal builder and v3 remaining-speed rule: requested mean speed forms the
geometric preview, while only the final speed scalar is replaced by remaining
budget. Runtime-generated q targets apply the evaluator's action clip followed
by `action_scale * env_action + q_offset` in the environment's recorded joint
order. Physical values are read from the live Isaac session, not inferred from
the checkpoint.

## Sequential transition replay gate (2026-10-03)

Phase A passed locally for one captured walk-to-crouch transition using the
pinned WC-v3 checkpoint. A 600-tick fixture (150 four-action chunks) was replayed
through both native and cached samplers. Both passed 150-snapshot parity and
sequential replay with zero reported error for the seeded histories, actions,
goals and joint targets; the runtime regression suite passes 10 tests.

Artifacts are local and ignored under `scratch/wc_deployment/`:
`s2s_gate_oracle_transition_01/` is the Isaac capture and
`s2s_gate_replay_transition_01/{native,cached}/` contains replay reports.
This is same-backend software replay against recorded observations, not a
closed-loop physics transfer, MuJoCo result or sim-to-real validation. The
remaining staged plan and evidence are tracked in the
[implementation handoff](../experiments/wc_wct_ablation_v1/SIM2SIM_IMPLEMENTATION_HANDOFF.md).

## Capture three development oracle rollouts

Run each command from the repo root, with `env_isaaclab` available. Each command
uses one environment, deterministic reset, the native checkpoint pace mode and
the exact normal evaluator loop. Fixture directories are local scratch and are
not tracked. The route set is development-only, not a held-out benchmark.

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --output_dir scratch/wc_deployment/eval_straight --runtime_fixture_output_dir scratch/wc_deployment/oracle_straight --runtime_fixture_max_samples 12 --path_shapes straight --speeds 0.35 --path_height 0.2932 --route_length_m 4 --duration_s 12 --repeats 1 --seed 42 --warmup_steps 25 --deterministic_resets --headless --device cuda:0
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --output_dir scratch/wc_deployment/eval_curve --runtime_fixture_output_dir scratch/wc_deployment/oracle_curve --runtime_fixture_max_samples 12 --path_shapes coherent_smooth --speeds 0.65 --path_height 0.2932 --route_length_m 4 --duration_s 12 --repeats 1 --seed 43 --warmup_steps 25 --deterministic_resets --headless --device cuda:0
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --output_dir scratch/wc_deployment/eval_transition --runtime_fixture_output_dir scratch/wc_deployment/oracle_transition --runtime_fixture_max_samples 12 --path_shapes coherent_smooth --speeds 0.35 --transition_fractions 0.5 --transition_directions walk_to_crouch --route_length_m 4 --duration_s 12 --repeats 1 --seed 44 --warmup_steps 25 --deterministic_resets --headless --device cuda:0
```

For a contiguous controller replay, capture 40 consecutive policy steps as well
as the distributed oracle snapshots (the sequential capture replaces the
distributed schedule in that fixture, so use a separate output directory):

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --output_dir scratch/wc_deployment/eval_sequential --runtime_fixture_output_dir scratch/wc_deployment/oracle_sequential --runtime_fixture_sequential_steps 40 --path_shapes coherent_smooth --speeds 0.35 --route_length_m 4 --duration_s 12 --repeats 1 --seed 45 --warmup_steps 25 --deterministic_resets --headless --device cuda:0
```

Stop if capture refuses a checkpoint/config, output directory or scenario. Do
not force a different checkpoint through by turning off the SHA check.

## Replay parity and profile the captured software path

```powershell
.\isaaclab.bat -p -m scripts.diffusion_policy.runtime.benchmark --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --fixture scratch/wc_deployment/oracle_straight scratch/wc_deployment/oracle_curve scratch/wc_deployment/oracle_transition scratch/wc_deployment/oracle_sequential --output-dir scratch/wc_deployment/runtime_ab_01 --device cuda:0 --warmup 20 --samples 200 --sampler both
```

For a reverse-order repeat, use a new output directory and add
`--sampler-order cached,native` (for example, `runtime_ab_02`).

`parity.json` is written first. A parity failure exits before latency profiling;
investigate it rather than widening thresholds. Native and cached outputs are
written into separate subdirectories and checked against the oracle. Sequential
fixtures additionally verify history shifts, replanning cadence, all four
actions per chunk, route-goal updates and joint targets. The command refuses to
write into a non-empty output directory; choose a fresh name for each run.
`latency_summary.json`, `latency_samples.csv` and `latency_ecdf.png` describe a
repeated-input software microbenchmark, not a closed-loop or end-to-end
controller timing. `sampler_comparison.json` records the serial native/cached
comparison; repeat it in reverse measurement order when drawing a latency
conclusion. Host wall time is synchronized around CUDA work; CUDA-event time is
separately reported and is not additive. The 80 ms reference is four 20 ms
policy ticks, not a safe blocking delay for an actuator loop. Viewer, physics,
state transport and a hard real-time servo are outside this benchmark.

## Focused tests

```powershell
conda run -n env_isaaclab python -m unittest scripts.diffusion_policy.runtime.tests.test_runtime
```

Do not export, quantize, compile or change denoising/chunk settings unless the
cached FP32 path still fails the target timing budget and a new candidate first
passes same-device oracle parity.

## Deployment closure (2026-10-13)

The candidate remains WC-v3 seed 42, with the SHA pinned in `contract.py`.
The current project decision and immutable inference/physics-reference values
are recorded in
[PROJECT_CLOSURE_AND_DEPLOYMENT_2026-10-13.md](../experiments/wc_wct_ablation_v1/PROJECT_CLOSURE_AND_DEPLOYMENT_2026-10-13.md).
Fresh gate reports and a source/asset hash manifest are local-only under
`scratch/wc_deployment/runtime_closure_20261013/`. Do not treat a successful
recorded-observation replay as physical transfer or real-time readiness.
The converted-model audit and nominal closed-loop physical bank are now
complete. See [the expanded results and safety limits](../experiments/wc_wct_ablation_v1/SIM2SIM_RUNTIME_AND_SAFETY_2026-10-13.md).

## Component profile (2026-10-09)

`profile_breakdown.py` separates CPU route-goal preparation, grouped
CPU-to-GPU history copies, GPU conditioning normalization, the five frozen
transformer forwards, the five fine-tuned transformer forwards, remaining
reverse-chain work, and the final 12-joint GPU-to-CPU copy. Reproduce it with:

```powershell
conda run -n env_isaaclab python -u -m scripts.diffusion_policy.runtime.profile_breakdown --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --fixture scratch/wc_deployment/s2s_gate_oracle_transition_01 --output scratch/wc_deployment/component_profile_20261009/profile_breakdown.json --device cuda:0 --warmup 20 --samples 40
```

On the RTX 5070 Laptop GPU, for the captured 0.35 m/s coherent-smooth
walk-to-crouch transition, medians were: route-goal preparation 0.48 ms on
CPU; three history transfers (1,856 bytes) 0.23 ms synchronized host wall;
conditioning normalization 0.89 ms CUDA; frozen/base transformer 58.47 ms;
fine-tuned transformer 57.89 ms; and the remaining denoising, random draws
and output denormalization 12.66 ms. The full ten-step sampler was 130.27 ms
CUDA / 130.38 ms synchronized wall; returning a 12-joint action to CPU was
0.042 ms median. Thus the transformer accounts for about 89.9% of median
sampler CUDA time, while CPU route preparation and data copies are small.

The profiler report is stored in ignored `scratch/wc_deployment/`; its
percentiles describe 40 repeated samples of one recorded state, with
instrumentation enabled. These are component attribution measurements, not a
closed-loop timing or a hardware-interface benchmark. The sampler alone is
already slower than the nominal 80 ms replanning period on this laptop. The
next optimization should therefore target transformer inference; reducing
route-computation or transfer overhead cannot close this gap.

## Opt-in CUDA Graph execution (2026-10-13)

`CudaGraphDPPOSampler` captures the same FP32 hybrid K10 reverse chain at a
fixed batch size. No weights, denoising steps, normalizers, goals or chunk slots
change. Ten random draws remain eager, with the original shape/order. Buffers
remain alive and outputs are cloned. Construct it before control, on a private
eval-mode policy; do not mutate weights or share it across threads. Automatic
causal-mask detection is replaced by the equivalent flag only after verifying
the pinned decoder mask. This is optional: the default sampler is unchanged.

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
python -m scripts.diffusion_policy.runtime.optimize --checkpoint checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt --fixture scratch/wc_deployment/oracle_straight_v3 scratch/wc_deployment/oracle_curve_v2 scratch/wc_deployment/s2s_gate_oracle_transition_01 --output-dir scratch/wc_deployment/runtime_graph_new --samples 200 --warmup 20 --order cached,graph
```

Repeat with `--order graph,cached` in a new directory. Both variants must pass
the original snapshot oracle and continuous-history replay before timing.
The local graph passes 174 snapshots and 600 ticks with zero observed absolute
action/target error. Across both measurement orders, sampler medians are
approximately 4.1--4.4 ms, versus 117--131 ms cached, on an RTX 5070 Laptop GPU. Timing excludes live
sensor/actuator transport and simulator contention; it is not a hard realtime
guarantee. No real robot may be enabled based on this benchmark alone.

Capture/warmup follow [PyTorch 2.7 CUDA Graph guidance](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs):
fixed shapes/storage, side-stream warmup and no CPU synchronization in capture.

## Evolving-loop validation (2026-10-14)

`live.py` is a simulator-only profiler/supervised loop, not a robot driver.
`HostBufferSampler` keeps histories/chunks in CPU FP32 and copies at replans;
all four slots and hybrid K10 remain unchanged. It passes the original600-tick
oracle with zero action/target/history error. The live nominal trajectory also
agrees exactly (1200 standalone ticks and580 load-prefix ticks).

Use `--host-buffers cpu`; default buffers remain GPU. The supervisor defaults
to enforcement and latches faults. `--supervision observe` is explicitly a
diagnostic bypass for this simulator only; never expose it in a robot adapter.
The response is to stop issuing targets and terminate physics, not to prescribe
a supposedly safe real motor torque or pose. No hardware fallback is validated.

Measured sensor-to-target median is2.02 ms standalone, but17/1200 release
deadlines exceed20 ms. Under PhysX the loop vetoes a43.1-ms stale state after580
accepted ticks. Do not claim a worst-case or hard-realtime guarantee. The plant
is synchronous; it does not advance independently while the model computes.
See [the full results](../experiments/wc_wct_ablation_v1/DEPLOYMENT_VALIDATION_RESULTS_2026-10-14.md)
for bank, faults, timings, hashes and reproducible commands.
