# Phase A pose-profile v2: isolated walk/crouch experiment

This experiment changes only the goal observation. It keeps the demonstrated
actions, model size, optimizer, diffusion schedule, data split and symmetry of
the successful walk/crouch Phase A checkpoint.

## Question

Can a diffusion policy trained from disjoint walk and crouch demonstrations use
a spatial height profile to follow unseen routes and sustain the requested
height, without a skill label, gait reward or hand-written transition controller?

This is tested before adding sprint or DPPO. A failure here is evidence that
the disjoint dataset lacks transition support, not a reason to tune an online
reward.

## New goal contract

`hindsight_geom_profile16` (checkpoint schema version 8):

```text
[x25,  y25,  h25,
 x50,  y50,  h50,
 x75,  y75,  h75,
 x100, y100, h100,
 h_now, sin(yaw_terminal), cos(yaw_terminal), v_average]
```

The four tokens are sampled at 25/50/75/100% of the two-second look-ahead
arc. At constant speed they correspond approximately to 0.5/1.0/1.5/2.0 s;
arc fractions remain well defined during turns and variable local speed.

The training profile comes from `obs/desired_base_height`. Measured
`root_pos_w[:, 2]` is intentionally not used as a command because it contains
gait phase, tracking error and disturbances. Deployment uses the Z component
of the requested path. The fourth XY point remains the terminal local target;
the current height is separate so anticipation cannot erase the active
clearance/posture constraint.

## What is deliberately unchanged

- dataset: `walk_crouch_phase_a_waypoint_v3.hdf5`;
- from-scratch training;
- history 8, action horizon 16, execution offset 8;
- look-ahead 100 steps at 50 Hz;
- transformer 128/4 heads/4 layers;
- DDPM K=10, batch 4096, AdamW LR `1e-4`, 50 epochs;
- quadruped symmetry and episode-level validation split.

The old `hindsight_geom_avg12` code and checkpoints remain valid and provide
the paired baseline.

## Gates before any transition data or DPPO

Evaluate the best validation checkpoint against the old walk/crouch baseline
with identical seeds and route scenarios.

1. Constant walk/crouch routes: no material regression in survival or route
   completion; report XY RMSE, height MAE/P95 and fraction within 3 cm.
2. Interleaved and random height routes: report the same metrics plus the saved
   time series; inspect whether the posture is sustained throughout each
   section rather than only anticipated at its boundary.
3. Verify that all four profile slots change at the correct spatial progress
   and that `h_now` equals the route requirement at current progress.

Decision rule:

- if constant-height competence regresses, debug the contract/normalization;
- if constants pass but unseen transitions fail, collect one size-matched
  transition-data variant;
- only after a walk/crouch Phase A passes is sprint added with the same schema;
- DPPO is adapted to schema 8 only after the offline checkpoint passes.

No skill/gait signal is introduced at any stage. Contact and gait metrics are
evaluation-only diagnostics.

## Literature basis

- Diffusion Policy motivates receding-horizon conditional action sequences.
- DiffuseLoco demonstrates unified diffusion locomotion from heterogeneous
  offline skills and delayed state/action histories.
- LocoDiff and Gaitor support interpolation/transition hypotheses, while also
  motivating an explicit test of whether disjoint examples are sufficient.
- Kimodo-style path constraints motivate attaching task constraints to spatial
  path locations rather than exposing only a terminal scalar.
- DPPO is retained as the later online fine-tuning method, not used to conceal
  a missing offline observation contract.

See `FINAL_PROJECT_AND_PAPER_PLAN.md` for the full staged paper design.

## Commands

Cluster training from the repository root:

```bash
sbatch scripts/diffusion_policy/experiments/phase_a_pose_profile_v2/train_a_wc_cluster.sbs
```

After copying `best.pt` back, run the paired constant-height gate in PowerShell
(the backtick, not `\\`, is PowerShell's continuation character):

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py `
  --checkpoint checkpoints_iri/checkpoints_dp/walk_crouch_profile16_a_wc_v1.pt `
  --output_dir scripts/diffusion_policy/evaluations/walk_crouch_profile16_a_wc_v1_constant `
  --speeds 0.2 0.4 0.6 `
  --path_shapes straight circle s_curve right_angle random_polyline `
  --path_heights 0.1705 0.2932 `
  --repeats 3 --duration_s 20 --num_inference_steps 10 --exec_horizon 4 `
  --seed 42 --headless
```

Then run the held-out spatial-profile gate:

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/evaluate_policy.py `
  --checkpoint checkpoints_iri/checkpoints_dp/walk_crouch_profile16_a_wc_v1.pt `
  --output_dir scripts/diffusion_policy/evaluations/walk_crouch_profile16_a_wc_v1_interleaved `
  --speeds 0.4 0.6 `
  --path_shapes straight s_curve right_angle random_polyline `
  --height_profile interleaved --height_segment_m 0.8 `
  --height_cycle 0.2932 0.1705 0.2932 0.1705 `
  --repeats 3 --duration_s 20 --num_inference_steps 10 --exec_horizon 4 `
  --save_timeseries --seed 42 --headless
```

Only after this binary-support test, repeat it with intermediate heights
(`0.25`, `0.21`) as a separately labelled interpolation test. This avoids
confounding spatial composition with unseen scalar-height interpolation.
