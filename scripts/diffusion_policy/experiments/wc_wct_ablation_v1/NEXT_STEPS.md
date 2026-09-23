# Roadmap after the WC/WCT ablation

Status date: 2026-09-23. This document freezes the next steps after the clean
seed-42 comparison. It is intended to prevent reward, task, or narrative
changes while the replication and evaluation protocol is in progress.

## Current checklist

- [x] Train and freeze Phase-A WC and Phase-A WCT priors.
- [x] Train matched `supported_hybrid_v3` WC/WCT actors with seed 42.
- [x] Run the clean frozen benchmark for both seed-42 actors.
- [x] Run the gait audit and establish that the learned fast behaviour is a
  hybrid regime, not reproduced flying-trot.
- [x] Train matched WC and WCT DPPO replicas with seeds 43 and 44.
- [x] Run the frozen benchmark for WC43, WC44, WCT43 and WCT44.
- [x] Aggregate the preregistered suite metrics across seeds.
- [x] Freeze the WC/WCT conclusion and select WC seed 42 as the current
  deployment candidate.
- [x] Audit whether the demonstrations are conditionally multimodal.
- [~] Train one matched non-diffusion action-chunk BC baseline (WC seed 42
  currently running on the cluster, 2026-09-24).
- [ ] Implement and evaluate the matched end-to-end RL-from-scratch baseline.
- [ ] Assemble final paper/I2R tables, plots, limitations and provenance.
- [ ] Profile/export the frozen candidate and begin sim2sim.
- [ ] Measure the transfer gap and add only evidence-driven robustness.
- [ ] Complete hardware-in-the-loop and staged SOLO12 tests.
- [ ] Consider pace/trot only as an optional follow-up after the core result.

## Question that is currently being tested

Can path-, posture-, and time-conditioned DPPO acquire fast locomotion from a
simple walk+crouch (`WC`) diffusion prior, and do explicit fast-trot
demonstrations in the walk+crouch+fast-trot (`WCT`) prior improve that
acquisition?

The experiment does **not** currently establish latent policy selection. The
gait audit shows that the learned fast behaviour is a new high-cadence hybrid
regime rather than a faithful reproduction of either the walk or flying-trot
expert. That distinction must remain explicit in every report.

## Evidence already frozen

- Clean Phase-A WC SHA256:
  `b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd`.
- Clean WCT Phase-A ancestor SHA256:
  `2f28fd4b...da54` (record the complete digest from the cluster before final
  publication).
- WC-v3 seed-42 SHA256:
  `2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868`.
- WC-v3 beat the matched WCT-v3 seed-42 actor on supported-fast and OOD-fast
  success. WC-v3's main failure is repeated crouch-start at 0.45 m/s, caused
  by upright-height recovery rather than falling or path error.
- Neither WC-v3 nor WCT-v6 reproduces the flying-trot expert. Call the learned
  behaviour a **fast hybrid locomotion regime**, not sprint or fast-trot.

## Step 1 — Finish the DPPO seed replications (complete)

Complete seeds 43 and 44 for both conditions:

1. WC Phase A -> DPPO `supported_hybrid_v3`.
2. WCT Phase A -> DPPO `supported_hybrid_v3`.

The four jobs must retain the seed-42 contract exactly: 4096 environments,
150 iterations, 32 rollout chunks, route maximum 1.0 m/s, speed budget
1.5 m/s, KL 0, actor LR 1e-5, horizon 4, and the unchanged v3 rewards and route
distribution. Only the Phase-A checkpoint and random seed may differ.

For every job preserve:

- SLURM log and W&B URL;
- selected `best.pt`, iteration, SHA256 and embedded source checkpoint SHA;
- Git commit and dirty status;
- selection metric and complete `metrics.jsonl`.

Do not select `last.pt` merely because it is the final checkpoint.

Job 3652 completed all four replicas sequentially with exit code 0. The
training-selection success rates were consistent but are not substitutes for
the frozen benchmark:

| Condition | Seed | Train success | `best.pt` SHA256 |
|---|---:|---:|---|
| WC | 42 | 98.6% | `2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868` |
| WC | 43 | 99.2% | `5d0ddba54dabf3baf7bfcc1cb63d3187a21944a443b28c3cc698c1ed870685e2` |
| WC | 44 | 98.4% | `b4a26ca20bf161a9c8da54f170a25c9f2726f61b1dd9e1d9bb5e8ea142032f6d` |
| WCT | 42 | 86.5% | `5258fcf239a391bcfe6e4a951944b037b94441113d137b782ad5ade8341fa55f` |
| WCT | 43 | 85.5% | `e985587c6e8d1a3bca61432cf480af1b3f0f5edc88761d206968c7499502ef90` |
| WCT | 44 | 88.1% | `59dee18f1bb3d30c5d9001920f5412cb211eeeefbae33fd580d5ab08e64fa9ff` |

The four new actors were produced from clean commit `f3df4df`; the seed-42
actors remained intact. No WC-v6 run was performed.

## Frozen three-seed WC/WCT result (2026-09-23)

The four additional evaluations used the same frozen route banks and evaluator
seed as seed 42. Their embedded checkpoint hashes match the selected WC/WCT
seed-43 and seed-44 checkpoints above.

Across-seed mean task success (minimum--maximum across the three train seeds):

| Suite | WC | WCT |
|---|---:|---:|
| ID constant | 100.0% [99.9, 100.0] | 99.4% [98.6, 100.0] |
| ID transition | 99.9% [99.9, 100.0] | 99.8% [99.6, 99.9] |
| OOD constant | 99.9% [99.7, 100.0] | 99.0% [98.3, 99.9] |
| OOD transition | 99.9% [99.9, 99.9] | 99.1% [98.6, 99.6] |
| Fast crouch-to-walk | 96.4% [94.1, 97.7] | 75.4% [66.3, 88.9] |
| Fast walk-to-crouch | 92.2% [89.1, 93.7] | 65.4% [51.1, 78.4] |
| OOD fast crouch-to-walk | 95.3% [92.2, 98.0] | 73.0% [61.1, 82.9] |
| OOD fast walk-to-crouch | 86.1% [80.7, 90.4] | 65.9% [52.2, 81.6] |
| Repeated walk start | 98.5% [95.5, 100.0] | 76.3% [58.5, 97.0] |
| Repeated crouch start | 54.6% [50.0, 59.3] | 84.0% [61.8, 99.3] |

The preregistered conclusion is therefore stable across train seeds: explicit
flying-trot demonstrations do not provide a reliable fast-task advantage under
this contract. WC is consistently better on every supported-fast and OOD-fast
aggregate, and WCT has substantially larger optimization variance. WCT retains
a real advantage on repeated transitions starting crouched, whereas WC has a
repeatable upright-height recovery failure there. This exception must remain in
the final report.

WC seed 42 is the current deployment candidate because it has the best overall
frozen-suite aggregate among the WC replicas, strong fast performance, and an
existing gait audit. This is an engineering selection, not evidence that seed
42 is statistically superior to WC seeds 43 or 44.

## Step 2 — Evaluate every selected seed on the frozen benchmark

Run `wct_final_evaluation_v1/evaluate_cluster.sbs` separately for the four new
`best.pt` files. Use unique, empty output directories. Do not regenerate route
banks and do not change evaluator seed, horizon, tolerances, speeds, or number
of repeats.

Example for one actor:

```bash
sbatch scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/evaluate_cluster.sbs \
  /absolute/path/to/best.pt \
  /home/sflores/i2r/isaaclab-solo-diffusion-policies/scripts/dppo_diffusion_rl/evaluations/<condition>_seed<seed>_clean
```

Evaluate the four actors as:

- `wc_seed43_clean`;
- `wc_seed44_clean`;
- `wct_seed43_clean`;
- `wct_seed44_clean`.

The gait audit does not need to be repeated for every seed unless the contact
statistics or visual behaviour differ substantially. Its purpose is mechanism
diagnosis, not model selection.

## Step 3 — Make the WC/WCT decision statistically

Aggregate route-level results without pooling ID and OOD. Report each seed and
the across-seed mean/range for:

- ordinary constant and transition success;
- supported-fast success at 0.65, 0.75 and 0.85 m/s, split C->W and W->C;
- OOD-fast success;
- survival, arrival, CTE RMSE, height MAE and mean-speed error;
- repeated walk-start and repeated crouch-start.

Interpret the replication before any new tuning:

- If WC wins fast consistently across seeds, conclude that DPPO acquires the
  fast capability without fast-trot demonstrations and that WCT introduces
  interference under this training contract.
- If the ranking varies strongly, conclude that fast demonstrations provide no
  reliable advantage and that optimization variance dominates the seed-42
  result.
- If WCT wins consistently on the new seeds, treat seed 42 as an outlier and
  retract the strong WC-superiority claim.

Do not train WC-v6 to improve the result. It would break the matched causal
comparison with WCT-v3.

## Step 4 — Run one honest RL-from-scratch baseline

Before this step, close the separate question of whether the diffusion
parameterization itself is warranted. The existing WC/WCT comparison tests the
value of fast demonstrations, not diffusion versus a conventional policy.

### Step 3.5 — Minimal diffusion-necessity control

Run two deliberately small controls before making a claim that diffusion is
needed:

1. **Conditional multimodality audit (no new simulator training).** On the WC
   and WCT datasets, find samples with similar observation history and route,
   posture and speed conditioning, and measure whether their future action
   chunks form multiple separated modes. Report expert/source labels only for
   diagnosis, never as policy inputs. If the conditional future is effectively
   unimodal, multimodality is not a defensible reason for diffusion in this
   experiment.
2. **Matched deterministic action-chunk BC.** Use the same observations,
   conditioning, dataset splits, history, prediction horizon and receding-
   horizon execution as Phase A, but replace iterative denoising with a direct
   deterministic action-chunk head trained by regression. Keep the conditioning
   encoder and parameter budget as close as practical. Evaluate it on the same
   frozen Phase-A banks. A one-step MLP with different inputs or no action
   horizon is not a matched control.

Start with one seed. Only replicate it if it is close enough to alter the
conclusion. This is cheaper and more diagnostic than immediately launching a
large collection of alternative architectures.

Interpretation is preregistered as follows:

- If diffusion clearly improves Phase-A stability, transitions or OOD tracking,
  retain the claim that its sequence distribution is a useful prior for DPPO.
- If deterministic BC matches Phase A, do not claim that diffusion was required
  for imitation. Describe it as a compatible prior that enabled the tested DPPO
  pipeline, and run one standard PPO fine-tuning comparison only if a
  diffusion-specific contribution is required for publication.
- If deterministic BC plus standard PPO also matches DPPO at comparable data and
  interaction cost, diffusion is an implementation choice rather than the
  scientific contribution. Reframe the contribution around hindsight task
  relabelling, path/posture/time conditioning and online acquisition beyond the
  demonstration support.

Do not use published DiffuseLoco ablations as a substitute for this control:
they justify the design in their datasets and robots, not in the current SOLO12
task.

#### Conditional-multimodality audit result (2026-09-23)

The reproducible audit is implemented in
`audit_conditional_multimodality.py`. It uses 5,000 raw samples per skill, the
exact 464-dimensional Phase-A conditioning contract, the 96-dimensional
executable future action chunk, a one-second same-episode exclusion, projected
candidate retrieval followed by exact full-context reranking, and a random
cross-skill control. Frozen outputs are under
`scripts/diffusion_policy/data/audits/wc_wct_conditional_multimodality_v2/`.

Median normalized context distances were 0.306 within skill, 1.161 for the
nearest other WC skill and 1.593 for random WC cross-skill pairs. No WC
walk/crouch cross-skill pair passed even the broad within-skill p95 similarity
gate. WC therefore provides no evidence that multiple expert actions remain
valid after the complete state/history/goal conditioning is specified.

For WCT, median distances were 0.229 within skill, 0.893 for the nearest other
skill and 1.482 for random cross-skill pairs. The broad p95 gate admitted 9.28%
of nearest cross-skill pairs, almost exclusively walk/sprint, but the strict
within-skill median gate admitted only 0.653%. None of those strict-overlap
pairs exceeded the within-skill p95 future-action divergence threshold. This is
weak boundary overlap, not strong evidence of conditional multimodality.

The audit is a nearest-neighbour diagnostic rather than a proof that the full
continuous conditional density is unimodal. Its defensible conclusion is
narrower: **multimodality is not currently an empirically established reason
for diffusion in this project.** The matched deterministic action-chunk BC
baseline remains necessary. Sequence modelling, receding-horizon execution and
the availability of DPPO remain valid design motivations, but they must be
distinguished from a demonstrated need for a multimodal output distribution.

Implementation is frozen in
`DETERMINISTIC_BASELINE_IMPLEMENTATION_PLAN.md`. The first implementation is WC
seed 42 only; it must stop after tests, local smoke and preparation of the
cluster launcher so the diff can be reviewed before training.

Only after the deterministic Phase-A decision is frozen, add an end-to-end
joint-action PPO baseline with
the same observations, route generator, reward, termination rules, action
frequency, environment count, and interaction budget. A randomly initialized
diffusion model optimized with DPPO is not the appropriate baseline because
DPPO is a diffusion fine-tuning algorithm and assumes a meaningful denoising
prior.

A randomly initialized DPPO actor is a narrower initialization ablation but not
the primary full run. It may receive one cheap smoke test: continue only if it
produces finite gradients, non-degenerate chunks, measurable progress and
improving survival. Its failure would be ambiguous because DPPO is designed for
fine-tuning a pretrained denoising policy; it would not by itself prove that
Phase A is fundamentally necessary. The primary practical comparison remains
direct joint-action PPO from scratch. The complete rationale and claim matrix
are frozen in `CAUSAL_BASELINES_AND_DECISION_TREE.md`.

Compare at least:

- learning curve versus environment interactions;
- falls and survival during training;
- final frozen-bank success and OOD generalization;
- CTE, height and timing errors;
- total simulation/GPU cost.

Run a smoke test first and then the same three seeds if the baseline is
technically valid. Do not give the scratch baseline extra observations,
curricula, demonstrations, or a different success definition.

This baseline determines the value of imitation learning. If PPO eventually
matches DPPO but needs substantially more interactions or is less stable, the
learned diffusion prior remains justified. If it matches DPPO at equal cost,
the contribution must be narrowed accordingly.

## Step 5 — Freeze the main paper/I2R result

After Steps 1–4, produce one final evidence package:

1. Phase-A WC and WCT performance.
2. DPPO WC and WCT across three seeds.
3. RL-from-scratch baseline.
4. ID/OOD, ordinary/fast, posture-transition, and timing breakdowns.
5. Gait/contact audit showing what behaviour DPPO actually learned.
6. Existing horizon-4 and waypoint-spacing evidence.
7. Limitations: repeated crouch-start recovery, boundary speed at 1.0 m/s,
   and absence of demonstrated latent gait selection.

The primary claim should be phrased around objective-conditioned online
capability acquisition beyond the effective offline support. It must not claim
that DPPO selects the demonstrated flying-trot unless later evidence directly
supports that mechanism.

## Step 6 — Optional pace/trot extension (not part of the core result)

Pace is relevant because the current walk and flying-trot experts are both
predominantly diagonal-trot behaviours. Their contact manifolds are too close
to make implicit skill switching identifiable.

Treat pace/trot as a separate extension and proceed only through these gates:

1. Train a pace expert alone.
2. Require robust speed tracking, low lateral drift, symmetry, survival and a
   clearly lateral contact-pair signature.
3. Confirm that it is quantitatively distinct from flying-trot using contact
   phase, duty factor, flight fraction and transition rate.
4. Train a small pace+trot Phase A before any DPPO.
5. Reject the extension if Phase A collapses both experts into one averaged
   gait or if speed alone trivially labels two disjoint datasets.

Do not add bipedal walking, gait rewards, skill IDs, or contact-pattern rewards
to rescue the current hypothesis. Those changes define a different project or
force the mechanism that the experiment is supposed to discover.

## Stop conditions

The research core is complete when:

- WC/WCT conclusions survive the three-seed comparison;
- the RL-from-scratch comparison is available;
- all final evaluations use frozen banks and clean Git provenance;
- claims distinguish task success from gait identity;
- tables, plots, checkpoint hashes and limitations are archived.

No additional reward version or locomotion skill should be introduced before
these conditions are evaluated.

## Deployment track — sim2sim and real SOLO12

The research-core stop conditions above do not mean that the complete project
is deployed. After freezing the actor and scientific comparisons, continue
with the following engineering stages. Keep their results separate from the
WC/WCT causal experiment: a deployment failure does not invalidate that
comparison, and deployment tuning must not silently change its checkpoints.

### D1 — Freeze and export the deployment candidate

Select one immutable checkpoint from the three-seed evidence, record its hash,
normalization statistics, observation ordering, action convention, denoising
steps, execution horizon, control frequency, PD gains and joint limits. Add a
deterministic replay test that detects differences between the training model
and its exported/runtime representation.

Do not choose the actor from a visually attractive rollout. Selection must use
the preregistered benchmark, and the deployment checkpoint must remain frozen
during the following measurements.

### D2 — Profile and optimize inference

Measure end-to-end latency on the target computer, including state ingestion,
conditioning, diffusion inference, action post-processing and communication.
Report median, p95 and worst-case latency, not only neural-network throughput.

First optimize implementation without changing the policy: inference mode,
buffer reuse, fixed tensor shapes, removal of logging allocations and an
appropriate exported runtime. Only if the deadline is still missed should the
number of denoising steps or model size change. Any such change requires
re-running the frozen benchmark because it changes the controller rather than
merely its implementation.

The gate is sustained real-time execution at the robot control deadline with
margin and without stale action chunks. Horizon 4 remains the starting point
because it was selected empirically; it is not reopened unless profiling shows
that deployment cannot meet the deadline.

### D3 — Sim2sim validation

Run the frozen controller in an independent simulator using the real robot's
control interface as closely as possible. Before tuning, audit coordinate
frames, joint order/signs, quaternion convention, units, action scaling,
control frequency, PD semantics, reset state and command timing. These
interface errors must be ruled out before attributing a gap to dynamics.

Evaluate a reduced but fixed deployment suite:

- standing and native walk/crouch;
- straight lines and gentle turns at low speed;
- posture changes in both directions;
- representative ID and OOD paths;
- fast conditions only after the earlier stages pass.

Use the same task metrics plus torque/current proxies, joint-limit margin,
foot slip, command latency and action discontinuity. Record failures rather
than tuning from individual videos.

### D4 — Measure the gap, then add minimal robustness

Compare Isaac Lab and sim2sim traces to identify the dominant discrepancies:
mass/inertia, friction, motor strength, actuator/communication delay, PD gains,
joint damping, sensor noise or contact modelling. Add system identification or
domain randomization only for discrepancies supported by those measurements.

Avoid broad randomization by default. It can hide interface bugs and degrade
the path/posture/timing behaviour. Every robustness change creates a new
checkpoint and must be compared against the frozen deployment candidate in
both simulators.

### D5 — Hardware-in-the-loop and safety layer

Before free locomotion, validate observation and action streams with motors
disabled or unloaded. Add independent joint position/velocity/torque limits,
command timeout, stale-observation detection, fall/orientation detection,
safe-stop posture, operator emergency stop and complete telemetry logging.

The safety layer may clip or stop unsafe commands; it must not act as an
unreported trajectory controller that supplies the behaviour claimed for the
learned policy.

### D6 — Staged real-robot validation

Progress only after each stage is repeatable:

1. suspended or tethered joint-direction and latency test;
2. standing and zero-command stability;
3. low-speed straight walk and crouch separately;
4. gentle curves and single posture transitions;
5. short ID routes with conservative timing;
6. OOD geometry and repeated transitions;
7. supported-fast routes last.

Use several trials per condition and external pose measurement when available.
Report success, falls/safety stops, CTE, height, timing, latency and energy or
current—not only successful videos. Preserve unsuccessful trials.

### D7 — Final optimization and deliverables

Optimize only bottlenecks observed in the real-time or robot data. Freeze the
final runtime, configuration and calibration; repeat the deployment suite; and
archive simulator and robot logs with checkpoint hashes. The final report must
separate three claims:

1. learning result in Isaac Lab;
2. robustness transfer demonstrated by sim2sim;
3. behaviours actually validated on the physical SOLO12.

If time is limited, a rigorous sim2sim result plus safe low-speed real-robot
validation is preferable to an unquantified full-speed demonstration.
