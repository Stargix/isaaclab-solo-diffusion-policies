# Causal baselines and final decision tree

Status: frozen reasoning record, 2026-09-24.

This document separates the questions that accumulated during the project. A
single baseline cannot answer all of them, and mixing them would make the final
claim difficult to defend.

## 1. What has already been established

The final task is path-, posture- and average-speed-conditioned locomotion on
SOLO12. Phase A uses hindsight relabelling of offline expert trajectories to
train a receding-horizon diffusion policy. DPPO then optimizes that diffusion
policy online without a skill ID, gait label or contact-pattern reward.

The matched three-seed WC/WCT experiment established that explicit fast-trot
demonstrations were not required for the fast capability obtained after DPPO.
WC was consistently better than WCT on supported-fast and OOD-fast suites,
whereas WCT retained an advantage on repeated profiles starting crouched. The
gait audit showed that the fast WC behaviour is a learned hybrid regime, not a
faithful selection of the demonstrated flying-trot.

The defensible mechanism claim is therefore:

> objective-conditioned online optimization can acquire useful locomotion
> behaviour beyond the effective support of a limited offline prior.

It is not currently defensible to claim implicit skill selection or faithful
recovery of the demonstrated fast gait.

## 2. The four separate causal questions

| Question | Required comparison | What it can establish |
|---|---|---|
| Do fast demonstrations add capability? | WC-DPPO vs WCT-DPPO | Value of the additional fast expert data |
| Does diffusion improve Phase A? | Diffusion BC vs matched deterministic action-chunk BC | Value of the DDPM objective and iterative sampling for the offline prior |
| Does Phase A help online learning? | Pretrained pipeline vs RL from scratch | Value of demonstrations and offline initialization |
| Is the complete pipeline better than conventional RL? | Diffusion+DPPO vs direct joint-action PPO | Practical value, sample efficiency and stability of the overall method |

Results from one row must not be used as evidence for another row.

## 3. Current experiment: matched deterministic Phase A

The running control is **not PPO**. It is offline behavior cloning from the
same WC demonstrations as the diffusion Phase A.

Both policies receive the same 8-step proprioceptive, delayed-action and
16-dimensional goal histories and predict the same 16-by-12 action target.
They use the same split, normalization, quadruped augmentation, Transformer
width/depth, AdamW schedule and EMA. Their trainable parameter counts are
1,238,412 for diffusion and 1,238,668 for the deterministic control.

The only intended difference is the output objective and sampler:

- diffusion predicts DDPM noise and performs ten denoising network calls per
  action chunk;
- deterministic BC directly regresses the complete normalized action chunk
  with MSE and performs one network call per chunk.

The frozen JSON retains diffusion fields for configuration parity, but the
deterministic checkpoint records `denoising_steps=0` and
`forward_passes_per_chunk=1`; those ten steps are not executed.

### Decision after evaluation

- If diffusion has a material, consistent closed-loop advantage across several
  ID/OOD suites, retain it as an empirically better offline prior. Do not
  attribute the advantage specifically to multimodality without new evidence.
- If the two are comparable, Phase A does not establish that diffusion is
  necessary. Describe diffusion as the parameterization used by the successful
  DPPO pipeline and continue to the online control.
- If deterministic BC wins, diffusion is not justified by Phase-A performance.
  The main contribution must move to hindsight conditioning and online
  acquisition unless an online comparison reverses the result.

One seed is sufficient for this decision gate. Additional deterministic seeds
are justified only if it is competitive enough to affect the conclusion.

## 4. Why this is not an MLP baseline

A one-step MLP would change architecture, parameter budget, temporal output,
receding-horizon execution and learning objective simultaneously. Losing to
diffusion would not identify which change mattered. The direct action-chunk
Transformer is deliberately close to the diffusion backbone, so the comparison
targets the DDPM objective and iterative sampler rather than model capacity.

No additional hierarchical policy is required. The failed velocity-command
hierarchy already addressed a different question and introduced a restrictive
intermediate action space and low-level response latency. Repeating it would
not close either causal question above.

## 5. RL from scratch: primary and optional controls

### 5.1 Primary practical baseline: direct joint-action PPO

The principal RL-from-scratch baseline should be a conventional policy that
maps the same task observations and conditioning to joint actions and is
trained with PPO using the frozen route generator, reward, terminations,
control rate and interaction budget.

This answers the practical reviewer question:

> Why use offline imitation plus diffusion fine-tuning if ordinary RL can solve
> the same task directly?

Compare learning curves by environment interactions, early falls, final
success, survival, CTE, height MAE, speed error, OOD generalization and total
compute. If PPO eventually matches but needs substantially more interaction or
is less stable, Phase A remains valuable. If it matches at equal cost, narrow
the contribution: the offline prior was not necessary under this simulator and
reward.

PPO is not a perfectly architecture-matched test of pretraining; it is the
standard system-level alternative. That limitation must be stated.

### 5.2 Optional diagnostic: randomly initialized DPPO

Starting the diffusion actor randomly and applying DPPO would isolate offline
initialization more narrowly because the online parameterization remains the
same. It is mathematically conceivable, but it is not the primary baseline:
DPPO is designed to fine-tune a meaningful pretrained denoising policy. From a
random denoiser, reward credit through the complete denoising chain may be too
poor for a fair or useful run, particularly with route success and physical
stability constraints.

Therefore:

1. do not make random-initialized DPPO mandatory for the paper;
2. run at most a cheap smoke experiment after the deterministic Phase-A
   decision and before allocating a full job;
3. continue only if it produces finite gradients, non-degenerate action chunks,
   measurable progress and improving survival;
4. interpret failure as an algorithm/applicability result, not proof that
   demonstrations are fundamentally necessary.

This resolves the apparent conflict between two reasonable intuitions: random
DPPO is the cleaner initialization ablation, while direct PPO is the stronger
practical alternative and the more interpretable full training run.

## 6. Required experimental order

1. Finish the running WC deterministic Phase-A seed-42 train.
2. Run its sanity gate and the exact frozen Phase-A benchmark.
3. Decide whether diffusion is better, comparable or worse; do not tune after
   seeing the benchmark.
4. Implement and smoke-test direct joint-action PPO from scratch under the
   frozen task contract.
5. Train the technically valid PPO baseline with the same interaction budget;
   use three seeds for any publication-level comparison.
6. Optionally smoke-test random-initialized DPPO if a narrower initialization
   ablation is still scientifically necessary.
7. Freeze paper tables, limitations and checkpoint provenance before sim2sim.

Do not insert another reward revision, locomotion skill, gait supervision,
hierarchical controller or architecture search into this sequence.

## 7. Final claim matrix

| Observed result | Permitted conclusion |
|---|---|
| Diffusion Phase A > deterministic BC | DDPM provides a more useful offline prior for this task |
| Diffusion Phase A ~= deterministic BC | Diffusion is compatible with DPPO but not shown necessary offline |
| Pretrained DPPO learns faster/more safely than PPO | Offline prior improves sample efficiency or optimization stability |
| PPO matches at equal interactions and quality | Full diffusion/IL pipeline is not necessary in this setting |
| WC-DPPO > WCT-DPPO | Extra fast-trot data does not cause the acquired fast capability |
| WCT improves crouch-start profiles | Additional demonstrations help a specific posture-recovery region |

None of these outcomes invalidates the engineering system. They determine how
narrowly the scientific contribution must be phrased.

## 8. Out-of-scope rescue attempts

Do not add skill IDs, gait labels, diagonal-contact rewards, bipedal walking or
pace merely to make the current mechanism easier to demonstrate. Pace may be a
later distinct-skill study, but it is not required to close the current
path-posture-time result. Likewise, do not train another velocity-command
hierarchy unless a future project explicitly studies hierarchical composition.

After the causal experiments, the independent deployment track remains:
inference profiling, export, sim2sim interface audit, measured robustness,
hardware-in-the-loop safety and staged real-robot validation.
