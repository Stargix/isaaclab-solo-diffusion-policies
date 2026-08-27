# Final A1 data and training gate

Date: 2026-08-27

Decision: **the corrected three-skill dataset is accepted for one from-scratch
A1 diffusion training run.** The earlier `*_v1`, `*_v2` and `*_v3` files are
diagnostic pilots and must not be used for this run.

## Exact artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `sprint_fast_trot_phase_a_waypoint_v4.hdf5` | 348,931,617 | `B45FA70FD3BCEF9946C7833ED62AD9048F197AD4EFD8D1786D0161DA663B45F5` |
| `walk_crouch_sprint_phase_a_waypoint_v4.hdf5` | 1,021,092,973 | `4E323BD426CCB6F503E70EE255B397899E0DA2F847EB60FA6A21F147F7A6B730` |

The merged file contains 10,035 demonstrations, 3,000,465 frames and exactly
665,655 valid 2 s hindsight windows for each of `walk`, `crouch` and `sprint`.
The original walk/crouch file is frozen and byte-identical to A0.

## Why v4 supersedes the pilots

The final sprint collection matches the real A0 temporal protocol: commands
change every 3 s, explicit stops have probability 0.10 and seed 46 is distinct
from the walk/crouch seeds. A 30 degree body-tilt guardrail rejected eight
unstable attempts; no simulator failures occurred.

The audit exposed and fixed a collector alignment bug. A manually reset
active-fall environment had been cached before reset, so its fallen terminal
state could become frame zero of the next demonstration when warmup frames were
retained. The collector now refreshes only the affected reset environment. In
v4 the maximum first-frame XY displacement is 2.8 mm and no stale reset
discontinuity remains.

## Evidence for accepting the support

- frame balance is exactly 1.0 and all structural checks pass;
- sprint hindsight speed is 0.283 / 0.964 / 1.327 m/s at p05/p50/p95 and
  reaches 1.499 m/s;
- sprint terminal height is 0.285 m at the median;
- maximum sampled body tilt remains below the 30 degree guardrail;
- only 0.0237% of sprint windows have `|terminal_y| > 1 m`; inspection shows
  continuous high-speed arcs, not reset jumps;
- the combined distribution has 1.902% near-stationary windows and only
  0.008% duplicate waypoints;
- walk and sprint overlap near their boundary while sprint extends the 2 s
  reach to about 3 m.

The hollow two-wing sprint PCA is retained intentionally: it is the
mirror-related, non-convex flying-trot limit cycle. Filling its interior would
add joint configurations the expert never executes.

Plots:

- [`raw_multiskill_support.png`](../../data/audits/fast_trot_v4/merged_final/raw_multiskill_support.png)
- [`hindsight_multiskill_support.png`](../../data/audits/fast_trot_v4/merged_final/hindsight_multiskill_support.png)
- [`hindsight_geom_avg12_coverage.png`](../../data/coverage/walk_crouch_sprint_fast_trot_v4/hindsight_geom_avg12_coverage.png)

## Training decision

The first A1 run starts from scratch. Initializing from the two-skill A0 model
would bias optimization toward the old walk/crouch manifold and confound the
effect of adding data. Warm-start remains a fallback, not the main experiment.

The new config comes from the embedded configuration of the successful A0
checkpoint, not the stale 1024-batch/30-epoch JSON. It preserves history 8,
trajectory 16, execution offset 8, 2 s geometric goal, quadruped symmetry,
DDPM K=10, batch 4096, 50 epochs, LR 1e-4, EMA 0.9999 and seed 42.

Keeping 50 epochs gives each example the same expected exposures as A0. The
larger dataset naturally produces 1.5 times more updates; reducing epochs to
equalize updates would under-train every skill.

No transition demonstrations, extra loss, skill label or trainer rewrite is
added. The research variable is only the new expert manifold.

The executable Slurm recipe is
[`train_a1_cluster.sbs`](train_a1_cluster.sbs). It verifies the merged HDF5
SHA-256 before importing the trainer and intentionally runs plain PyTorch
without launching Isaac Sim.
