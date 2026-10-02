# Matched deterministic BC + Gaussian PPO pilot

This is NOT PPO from scratch, a hierarchical controller, or residual RL. It
loads the already evaluated deterministic WC Phase-A mean (SHA-256
`c874b2ad0a44cfaa69b0400ec354ec17fd3cad6d303923e1f55a91cc52305352`), adds
learned diagonal Gaussian exploration, and updates that same Transformer.
No additional demonstrations are required. WC diffusion/DPPO code and actors
remain unchanged. The existing evaluator receives three additive dispatch
changes so this actor also gets its trained four-action horizon and remaining
time feedback, rather than being mistaken for offline BC.

## Scientific protocol

The paired task is exactly `supported_hybrid_v3`: 4096 environments, 150
iterations, 32 chunks/update, execution horizon 4 at 50 Hz, gamma 1,
GAE lambda .95, ten critic-only warmup iterations, the same asymmetric critic,
route generator, feasibility filtering, history padding and terminal resets.
Path weight 2, CTE gate .10 m, height weight .75, height gate .04 m,
route speed max 1.0 m/s, remaining-pace cap 1.5 m/s, transition boundary
.8--3.2 m, original geometric preview, no reference KL or gait/symmetry reward.
The nominal sampled-physics budget is 78,643,200 env steps per arm; both
collectors retain the same post-terminal padding convention. Optimizer FLOPs
are NOT matched: DPPO optimizes multiple denoising transitions per chunk.

Gaussian-specific pilot settings: actor LR 1e-4, clipping .2, target KL .02,
initial normalized joint std .04, learned per joint with bounds [.01,.2],
five epochs/update, actor minibatch 8192, critic minibatch 4096 and LR 1e-3.
No entropy bonus or BC regularization in this first pilot. These are a declared
starting recipe, not claimed optimal hyperparameters or a literal copy of
DPPO's transition clipping. Attention dropout is disabled during collection
AND gradient updates. Evaluation uses the clipped mean with no exploration.

Following [PPO](https://arxiv.org/abs/1707.06347) and the Gaussian baseline in
[DPPO, sections 5.3 and E.7](https://arxiv.org/html/2409.00588v3#A5.SS7), we
fine-tune a pretrained mean and learn its exploration scale. Gaussian std is
measured in normalized action units, not metres/second. Unlike that paper's
extra 3-sigma draw clipping, we retain exact untruncated draws. Joint log
likelihood is the SUM over only the 4x12 executable coordinates. Store raw
draws before bounded deployment clipping; the bound is a deterministic action
mapping, not a Gaussian density assigned incorrectly to clipped actions.

The best actor is selected by the existing training outcome score, before
updating the weights that produced those outcomes. Historical DPPO saved
post-update weights with the pre-update score; we do not alter archived runs
or deliberately copy that mismatch. Periodic and last checkpoints are marked
post-update/unscored. `ema_model_state_dict` is a legacy inference key containing
the current mean, NOT an online EMA. No optimizer resume is supported in this
pilot; rerunning into a populated output fails explicitly.

This is one online seed on one offline prior. A failed pilot cannot establish
that non-diffusion policies cannot solve the task. Diagnose gradient flow,
exploration, clipping/KL and training stability first; any further tuning must
be declared. If competent, replicate seeds 43/44 before population claims.
The frozen benchmark is a development comparison, not a new sealed holdout.

## Cluster

First deploy this implementation to the cluster (same code revision); the
launcher refuses dirty Git or the wrong checkpoint. It uses the existing
cluster Phase-A path by default, not a guessed checkpoint alias.

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git pull --ff-only
sbatch scripts/gaussian_chunk_rl/train_cluster.sbs
```

The job runs unit tests, a real-checkpoint GPU preflight and a 64-environment,
two-update Isaac smoke test before the main train. Smoke disables critic-only
warmup and requires nonzero actor AND critic updates. Failure aborts the job.
For smoke alone: `sbatch scripts/gaussian_chunk_rl/train_cluster.sbs smoke`.
Optional positional arguments: `train|smoke`, seed, checkpoint path. No main
train or evaluation is launched by preparing these files locally.

Output: `scripts/gaussian_chunk_rl/runs/ppo_wc_supported_hybrid_v3_seed42/best.pt`.
Use the existing frozen benchmark, explicitly execution horizon 4:

```bash
sbatch --dependency=afterok:<TRAIN_JOB_ID> --export=ALL,RECORD_STAGING_FAILURES=1 \
  scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  "$PWD/scripts/gaussian_chunk_rl/runs/ppo_wc_supported_hybrid_v3_seed42/best.pt" \
  "$PWD/scripts/dppo_diffusion_rl/evaluations/wc_wct_gaussian_ppo_seed42"
```

The optional staging-failure retention flag counts unstable initializations
as failed attempts rather than aborting the population evaluation. It changes
neither successful trajectories nor tolerances; historical launch defaults
stay unchanged. Do not silently exclude those attempts from comparisons.

The historical eval launcher passes `--num_inference_steps 10`; it is explicitly
ignored for deterministic/mean inference. Do not change its banks or metrics.
Then compare with WC Phase A and WC DPPO v3 on those same banks, and report
physical tracking/safety/timing, not training return alone.

## KL-calibrated second pilot (2026-10-03)

The first seed-42 run completed 150 iterations, but all 140 actor-enabled
iterations stopped after exactly one minibatch. Observed stopping KL ranged
from .626 to 30.133 (median 2.223), versus target .02. The old
`Policy/approximate_kl` was zero because it only averaged accepted minibatches
before their optimizer steps; the rejected minibatch was only visible in
`Policy/max_observed_kl`. This was a misleading diagnostic, not zero policy
change. See [the audit](OPTIMIZATION_AUDIT_2026-10-03.md).

The original `train` mode and fixed-LR algorithm remain available. New `tuned`
mode starts again from the pinned pure BC prior (NOT the failed PPO actor),
uses actor LR 1e-5, and enables bounded KL-based LR adaptation. It saves to a
different run directory and does not overwrite the original experiment.

The scheduler divides LR by 1.5 if measured joint KL exceeds twice the target;
it can increase LR by 1.5 below half the target only if no early stop occurred.
Bounds are [1e-7, initial LR 1e-5]. Measurement is the maximum of the sampled
minibatch KL and the post-update analytic KL(old Gaussian || new Gaussian)
on a fixed, evenly spaced subset of at most 8192 rollout conditions. This
retains the correct SUM over 48 executable coordinates. It adjusts the next
rollout's LR, does not undo the previous update, and is NOT a hard trust-region
guarantee. KL-driven adaptation is motivated by
[RSL-RL PPO](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py);
the bound, probe and early-stop combination here are explicit implementation
choices, not a verbatim reproduction of that algorithm or proven optima.

Rewards, routes, prior, exploration initialization/bounds, clipping, critic,
150 iterations, seed, and physical-interaction budget remain unchanged.
New telemetry records LR used/next and final analytic KL. Sampled
`Policy/approximate_kl` now includes rejected minibatches; the old accepted-only
quantity is retained as `Policy/accepted_minibatch_kl`. Do not compare the old
and new approximate-KL curves without this definition change.

After deploying a clean committed revision to the cluster:

```bash
cd ~/i2r/isaaclab-solo-diffusion-policies
git pull --ff-only
TRAIN_JOB_ID=$(sbatch --parsable --job-name=ppo_wc_klv2 scripts/gaussian_chunk_rl/train_cluster.sbs tuned 42)
echo "Train job: ${TRAIN_JOB_ID}"

sbatch --dependency="afterok:${TRAIN_JOB_ID}" --export=ALL,RECORD_STAGING_FAILURES=1 \
  scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  "$PWD/scripts/gaussian_chunk_rl/runs/ppo_wc_supported_hybrid_v3_kl_adaptive_v2_seed42/best.pt" \
  "$PWD/scripts/dppo_diffusion_rl/evaluations/wc_wct_gaussian_ppo_kl_adaptive_v2_seed42"
```

The launcher tests and simulation smoke both precede the train, using the
selected recipe. Before interpreting this pilot, inspect actor update counts,
actual KL and LR, height/pace compliance and frozen task success. Early stops
are normal; the goal is to avoid the original orders-of-magnitude overshoot,
not to force every epoch to run. No automatic success/convergence claim or
automatic seed replication is made.

## Local checks

```powershell
conda run --no-capture-output -n env_isaaclab python -m pytest scripts/gaussian_chunk_rl/tests -q
conda run --no-capture-output -n env_isaaclab python -m scripts.gaussian_chunk_rl.preflight --checkpoint checkpoints_iri/checkpoints_dp/wc_wct_phase_a_deterministic.pt --device cuda:0
```

The independent shared collector tests also cover padding/reset/GAE. Local
unit success is not proof of convergence or cluster compatibility; simulation
smoke status must be recorded separately. While the pilot trains remotely,
audit the frozen WC runtime and proceed toward sim2sim without retuning it.

## Validation record (2026-10-02)

- 114 CPU tests passed (new Gaussian tests, the full existing DPPO suite,
  and the reporting-contract tests).
- Real BC checkpoint preflight passed on CUDA, with the pinned SHA above;
  1,238,668 mean-network parameters, history 8, horizon 16 and execution offset 8.
- Isaac Lab local smoke passed: 16 envs, two iterations, two chunks/iteration,
  actual actor/critic updates and checkpoint save. It is only .32 seconds per
  robot, NOT an episode-success or convergence test. Small-batch KL stopping
  occurred after one actor update; inspect `max_observed_kl` in the pilot.
- Saved task dictionary matched the actual frozen WC-v3 checkpoint exactly;
  saved mean reload and finite inference passed. Bash syntax and Python
  compilation passed. Isaac emitted local Kit/config warnings but exited 0.
- The actual existing Isaac evaluator successfully loaded the saved Gaussian
  checkpoint, executed a two-second straight-route compatibility screen and
  recorded `exec_horizon=4`, `pace_budget_mode=dynamic`. Its outcome is not a
  frozen benchmark or evidence of learning; it tests the deployment dispatch.
- No cluster training or new frozen evaluation has run. Cluster launch includes
  a larger 64-env smoke before the full pilot; successful launch is not a
  guarantee of scientific improvement.
