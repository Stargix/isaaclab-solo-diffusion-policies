# WCP nominal-posture DPPO pilot

Decision: 2026-10-13. This is an **opt-in ablation**, not a replacement of
the paper's WC/WCT contracts, actors or tables. No skill ID, gait phase,
contact-pattern bonus or speed-conditioned gait reward is introduced.

## What is changed, and why

The upright task requests nominal base height 0.2932 m for both walk and
pronk. The native stable pronk expert oscillates: mean about 0.262 m,
p05/p95 about 0.233/0.294 m. About 29–30.5% of instantaneous samples lie
outside the terminal ±5 cm band. This is a phase-sensitivity diagnostic,
**not** a measured arrival failure rate. A 0.4 s causal mean reduced that
sample-level violation to zero in the recorded nominal expert traces,
while retaining the approximately 3 cm mean bias. It is a provisional
expert-calibrated task definition, not a parameter chosen from DPPO success.

Two changes apply equally to every posture and speed:

1. `posture_tracking_mode=nominal_window`, `posture_window_s=0.4`:
   average measured and required heights over the same last 20 control
   steps (50 Hz). Dense error is `mean(height) - mean(required_height)`.
   A matched transition has zero artificial filtering lag; a delayed
   transition still has error. Route-profile success uses the same
   distance-weighted absolute error and the original 4 cm tolerance.
   Terminal posture compares measured mean height to the **unsmoothed
   endpoint** requirement, with the original ±5 cm tolerance.
2. `vertical_velocity_reward_weight=0`: remove the explicit cost on
   vertical motion. There is **no positive reward for flight**.

Each scored step contributes one post-physics sample. Startup uses only
available samples; each environment's reset clears its history. Reading
observations/rewards never advances the observer. The immutable actor
inputs, goal representation, normalizers and action conventions are unchanged.
Three privileged critic features expose height mean, reference mean and
window fill; legacy critics retain their 14 route features.

## What is NOT changed

- Chassis-contact falls, corridor/overshoot failures, tilt cost and their
  terminal penalties remain instantaneous. No filter feeds physical safety.
- First arrival still determines traversal time; no free endpoint wait.
- Geometry, feasibility sampling and remaining-pace conditioning remain
  supported_hybrid_v3. No v4 pace-consistent preview is enabled.
- Historical defaults retain pointwise height and the 0.025 vertical cost.
- No artificial ceiling is inferred from nominal height. These are flat
  posture-reference scenarios, **not a clearance-under-obstacles experiment**.
  Future obstacle claims require instantaneous collision/clearance checks;
  nominal posture alone must never be reported as clearance safety.

The task/checkpoint contract is v9; model schema remains 8. Optimizer resume
checks the posture mode/window and vertical cost. Old missing fields map to
historical defaults; a changed task requires fresh optimization.

## Frozen training protocol

Start from pure Phase A, not a previous DPPO actor:
`scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/best.pt`.
SHA256: `40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886`.
The prior represents slow pronk, fast walk (selected anchors ≥0.6 m/s), and
crouch; no flying-trot source. Its closed-loop preview was poor upright;
small teacher-forced error is not evidence of autonomous stabilization.

Keep historical v3 settings: path weight 2, posture weight 0.75, CTE gate
0.10 m, mean-speed sampling up to 1 m/s, remaining pace cap 1.5 m/s,
transition boundaries 0.8–3.2 m, KL 0, no online symmetry, seed 42,
4096 envs, 150 iterations, 32 chunks, execution 4, K10 / 5 fine-tuned.
The first 10 iterations warm up the critic. Best selection explicitly uses
the **pre-update actor that generated the score**, not the unscored next actor.
This is a finite pilot, not a guarantee that pronk is preserved by RL.

The saved `transition_margin_m` is explicitly **0**, because v3's training
bank scores the entire transition (`height_tracking_weight=1`). This fixes
train/eval alignment for the new experiment: older v3 metadata could record
0.25 despite not masking training samples, causing the offline evaluator to
exclude them. Historical checkpoint evaluation is not silently rewritten.

```bash
python scripts/dppo_diffusion_rl/experiments/wcp_nominal_posture_v1/train.py --preflight-only
./isaaclab.sh -p scripts/dppo_diffusion_rl/experiments/wcp_nominal_posture_v1/train.py --device cuda:0 --wandb
```

Slurm wrapper stays private in `scratch/wcp_nominal_posture_v1/` locally,
then `/home/sflores/i2r/sbs/wcp_nominal_posture_v1/` on the cluster. No SBS,
logs, data or checkpoints need to be published. The wrapper and main trainer
both refuse a populated output; the experiment wrapper is for a new run.
Intentional resume uses the main trainer with all saved task/optimizer flags.

## Evaluation and interpretation (before another train)

Use the frozen evaluator on `best.pt`, with raw telemetry enabled. It reads
the saved posture contract automatically. Raw height is **never overwritten**;
nominal traces are extra NPZ fields `achieved_posture_m` and
`required_posture_m`. Raw MAE, raw first-arrival height error, transitions,
contact/survival and timing remain visible. Train logs include both
`Episode/profile_height_mae_m` (task metric) and
`Episode/raw_profile_height_mae_m` (instantaneous diagnostic).

Inspect straight and curved slow upright cases (0.2/0.35/0.5), fast walk
(0.65/0.85/1.0), crouch (0.2/0.35/0.45), and both walk↔crouch directions.
Report all failures. Separate raw tracking from nominal task success.
Check flight/contact cycles against the expert, not a speed-only label.
Good task completion with flattened walk is **not** successful pronk
retention, and failed completion is not proof that the robot cannot pronk.

Only after stable task behavior **and** recognizable slow pronk: evaluate
repeated transitions/OOD, then consider matched WC support control and seeds.
Compare priors/controllers on the **same new contract**; do not mix v9
success percentages into the historical pointwise-height paper tables.

## Basis and limitations

[DiffuseLoco](https://arxiv.org/abs/2404.19264) motivates multi-behavior
offline diffusion; [DPPO](https://arxiv.org/abs/2409.00588) and its
[official implementation](https://github.com/irom-princeton/dppo) motivate
fine-tuning the denoising MDP. Neither establishes that a generic task reward
preserves every demonstrated gait. The 0.4 s window is grounded in the
project's native expert telemetry, **not claimed as a literature-standard
optimal reward**. Averaging softens short transient height errors and can
cancel opposite-signed deviations; raw errors and transition delay must
therefore be retained, and no physical-clearance claim follows from it.

## Implementation verification (2026-10-13)

- **142 tests pass**: DPPO regression suite plus shared evaluator metrics.
  Covers partial startup, asynchronous reset, sinusoidal motion with retained
  DC bias, both matched/delayed posture transitions, unchanged arrival time,
  legacy precision, zero-vz-cost/no-flight-bonus, old checkpoint defaults,
  rejection of changed optimizer contracts, and the 17-feature mirror critic.
- Local Isaac Sim **5.1.0.0**, CUDA, **8 environments**: post-physics update,
  finite rewards/features, reads without extra window updates, selective
  resets, real Phase-A checkpoint inference and critic forward all pass.
  `SMOKE_OK` is recorded in ignored
  `scratch/wcp_nominal_posture_v1/smoke_final.log`. No training occurred.
- Prior SHA preflight passes; output mixing and wrong-prior checks are tested.
- Reapplying the observer to eight native expert traces gives 0 nominal
  ±5 cm violations after the first full window, versus approximately 27–32%
  raw violations per trace. This is calibration, not new policy performance.
- Python compilation, Git whitespace checks and private SBS shell syntax pass.

Cluster Isaac 4.5 has **not** been exercised by this local check. The code is
not a claim of guaranteed learning or sim-version equivalence. Before cluster
submission, publish just this revision; keep unrelated worktree edits and
all scratch/data/evaluation artifacts outside that commit.
