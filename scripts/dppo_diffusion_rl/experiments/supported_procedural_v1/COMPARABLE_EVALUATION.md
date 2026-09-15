# Comparable evaluation protocol

This note prevents results from different task contracts from being compared
as if they measured the same capability.  In particular, the old 50 s suite
combines repeated height changes, intermediate height targets, long routes and
post-arrival stability.  It is useful as an OOD stress test, but it is not a
fair primary test for the 4 m `supported_procedural_v1` task.

## What each suite measures

1. **Primary in-distribution (ID):** independent smooth procedural routes,
   constant endpoint heights and one walk/crouch transition.  This matches the
   DPPO training distribution without replaying training routes.
2. **Legacy-geometry control:** the same 4 m task, speeds, transition locations
   and directions, but on straight, circle, S-curve, right-angle and random
   polyline routes.  This isolates geometry generalization, including sharp
   corners.
3. **Repeated-transition stress:** endpoint heights only, alternating every
   0.8 m.  This measures repeated switching without confounding it with unseen
   intermediate target heights.
4. **Intermediate-height stress:** the historical
   `0.2932, 0.25, 0.21, 0.1705, 0.21, 0.25` schedule.  This is explicitly OOD:
   the definitive Phase-A data contain constant experts at the endpoint
   heights, not demonstrations of sustained intermediate-height locomotion.
5. **Post-arrival stability:** report first-arrival/task success separately
   from survival at the end of a long rollout.  The DPPO task terminates on
   success, so failures after arrival are a distinct capability.

Never rank checkpoints by combining suites 3--5 into the primary score.

## Apples-to-apples legacy-geometry command

Run the following command unchanged for every checkpoint, modifying only
`--checkpoint` and `--output_dir`.  `--repeats 1` is a local screen (60
scenarios); use `--repeats 3` for the reported comparison (180 scenarios).

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint CHECKPOINT.pt --output_dir OUTPUT_DIR --speeds 0.2 0.35 0.45 --path_shapes straight circle s_curve right_angle random_polyline --route_length_m 4.0 --transition_fractions 0.4 0.6 --transition_directions both --profile_transition_margin_m 0.25 --repeats 3 --duration_s 24.0 --num_inference_steps 10 --exec_horizon 4 --seed 542 --save_timeseries --headless --require_empty_output_dir --device cuda:0
```

The fixed families are not a substitute for the procedural test.  A
`random_polyline` repeat samples a different corner sequence, while the
right-angle family provides an explicit discontinuous 90-degree control.

## Repeated endpoint-transition stress

This is the closest valid analogue of the old interleaved-height plots while
remaining within the demonstrated walk/crouch endpoint support:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint CHECKPOINT.pt --output_dir OUTPUT_DIR --speeds 0.2 0.35 0.45 --path_shapes straight circle s_curve right_angle random_polyline --route_length_m 4.0 --height_profile interleaved --height_segment_m 0.8 --height_cycle 0.2932 0.1705 --repeats 3 --duration_s 24.0 --num_inference_steps 10 --exec_horizon 4 --seed 642 --save_timeseries --headless --require_empty_output_dir --device cuda:0
```

Run the historical intermediate-height schedule only as a labelled OOD test:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py --checkpoint CHECKPOINT.pt --output_dir OUTPUT_DIR --speeds 0.2 0.4 0.6 0.8 --path_shapes straight circle s_curve right_angle random_polyline --height_profile random --height_segment_m 0.8 --height_cycle 0.2932 0.25 0.21 0.1705 --repeats 3 --duration_s 50.0 --num_inference_steps 10 --exec_horizon 4 --seed 42 --save_timeseries --headless --require_empty_output_dir --device cuda:0
```

## Metrics that remain comparable across checkpoint contracts

Use survival, base-contact failure, physical-sanity failure, route arrival,
arrival-speed ratio, cross-track RMSE, terminal-position error and raw height
absolute error.  Do not directly compare `task_success` across contract v3 and
v5: v5 additionally requires sustained plateau-height tracking.

## Local 60-scenario screening result (seed 542)

| checkpoint | survival | arrival | base contact | arrival speed ratio | cross-track RMSE | terminal error | height MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase A WCT | 18.3% | 18.3% | 8.3% | 1.185 | 7.2 cm* | 164.8 cm | 0.79 cm* |
| DPPO `potxo` iter 100 | 88.3% | 90.0% | 8.3% | 1.032 | 5.7 cm | 19.5 cm | 1.92 cm |
| DPPO procedural iter 141 | **93.3%** | **95.0%** | **1.7%** | **0.989** | 8.9 cm | 21.9 cm | **1.20 cm** |

`*` The Phase-A tracking errors are survivor-biased because most trajectories
terminate before covering the route; its terminal error is the meaningful
indicator here.

The screen supports a real improvement in safety, arrival, average-speed
tracking and height tracking over `potxo`, but not in geometric precision.
The old checkpoint remains better in cross-track RMSE.  Therefore the final
claim must report both success and tracking error rather than reducing the
comparison to one scalar score.

## Local repeated binary-transition screen (seed 642)

This screen uses five endpoint plateaus on each 4 m route (a change every
0.8 m), 45 scenarios in total:

| checkpoint | survival | arrival | base contact | arrival speed ratio | cross-track RMSE | raw height MAE |
|---|---:|---:|---:|---:|---:|---:|
| DPPO `potxo` iter 100 | 91.1% | 86.7% | 6.7% | **1.036** | **5.6 cm** | 4.06 cm |
| DPPO procedural iter 141 | 91.1% | **91.1%** | **4.4%** | 1.186 | 12.4 cm | **2.73 cm** |

The procedural actor can repeatedly switch endpoint heights; the failure of
the historical 50 s suite is therefore not evidence that all transition
behaviour was lost.  However, rapid switching exposes a real trade-off: the
new actor tracks height better and arrives more often, but overspeeds and loses
geometric precision.  This suite is deliberately kept as a stress result—the
actor was trained with one transition and 1.6--2.4 m plateaus, not 0.8 m
plateaus.
