# Final simulation evidence and execution limits

The selected actor remains WC-v3 seed 42. The existing three-seed WC/WCT and
Gaussian-PPO comparisons remain development comparisons. A separate
post-selection bank validates the selected actor on new route draws.

## New route bank

The [result archive](../../research_artifacts/final_validation_v1/) contains
56 geometries and 672 conditions. The primary lower-speed ID task/adherence
rate is 98.4%, with CTE RMSE 1.97 cm. Fast crouch and some fast transitions
remain difficult. No additional training follows this check.

The selected actor SHA256 is
`2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868`.
Precision, K10 sampling, four-action execution and the task predicate remain
fixed. Height scoring retains a 0.25-m exemption on either side of each
transition. All initialization and physical failures remain in denominators.
Routes are the statistical unit, not the crossed speed/posture conditions.

To freeze and execute a separate reproduction from the repository root:

```bash
python -m scripts.diffusion_policy.evaluation.final_holdout freeze --checkpoint checkpoints/policy.pt --output-dir output/final_reproduction
python -m scripts.diffusion_policy.evaluation.final_holdout run --output-dir output/final_reproduction
python -m scripts.diffusion_policy.evaluation.final_holdout verify --output-dir output/final_reproduction
```

Supply the selected checkpoint separately. `freeze` rejects a different actor
and nonempty output; `run` rejects existing results and changed source. The
protocol records source, route and actor hashes before execution. This
recreates the fixed numeric bank, not a new unseen holdout for further tuning.
The exact original execution source is archived separately from later runtime
and reporting edits. Compact input hashes and the verified result table are
versioned; raw traces, checkpoint weights and temporary files are not.
The source inventory identifies which files are retained unchanged in the
clean public tree and which exploratory files remain only in the archive.
For exact historical replay, use the original snapshot rather than a later
Git revision. Weights, datasets and that snapshot are not bundled in Git.

## Inference computer

CUDA Graph measurements use the external development RTX 5070 Laptop GPU.
They do not measure or require a GPU inside the SOLO12. The actual robot
control computer and communication link still need to be specified and tested.
Host-memory buffering reduces GPU transfers; it is not CPU neural inference.

A local one-thread CPU sampler benchmark has median 60.9 ms and maximum
observed 67.9 ms over 20 fixed-input repeats. The neural block is recalculated
every 80 ms, but this benchmark excludes conditioning, transport and command
scheduling. It does not establish a sustained 20-ms receiver deadline.

The [runtime](../../scripts/diffusion_policy/runtime/) implements an independent
inference process and simulator receiver. It returns the same 600 original
action slots with matched recorded noise, but the declared paced receiver
trial fails sustained deadlines even with GPU inference. CPU and injected
90-ms-delay trials reject the first late plan. This is a recorded limitation,
not a safety pass. Rejection terminates the simulated trial and does not
implement an emergency action on physical hardware.

## Remaining deployment work

1. Identify the actual control computer, state-acquisition interface and link.
2. Measure complete latency and sustained command scheduling on that setup.
3. Use an independent receiver/actuator watchdog with a hardware-qualified
   emergency response and measured joint/torque limits.
4. Perform supervised progressive robot trials before making sim2real claims.

The simulation paper does not claim hard realtime, target-hardware safety,
optimal temporal allocation or reliable preservation of three distinct gaits.
