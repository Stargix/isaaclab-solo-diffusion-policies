# Phase A capability audit

This audit answers only the questions left unresolved by the causal preflight.
It does not train or modify the frozen diffusion policy.

`core_envelope` evaluates 135 paired scenarios over five path families, three
requested mean speeds, three constant heights and three stochastic diffusion
samples. `random_height_challenge` adds 18 scenarios on the three demanding
path families, with reproducible random height requirements every metre.

Speeds above 0.6 m/s are deliberately excluded: the preflight already shows a
66.7% survival rate at 0.6 m/s, so adding faster commands cannot establish the
minimum safe domain needed before online learning. Longer runs and additional
seeds belong in the final held-out evaluation, after the failure class is known.

Generate the two commands with:

```bash
python scripts/diffusion_policy/evaluation/prepare_protocol.py \
  --protocol scripts/diffusion_policy/experiments/phase_a_a0/capability_audit_v1.json \
  --platform linux \
  --results_root scripts/diffusion_policy/evaluations/phase_a_capability_audit_v1
```

After both runs finish, validate their checkpoint hash, code provenance and
protocol, then classify the failure with:

```bash
python -m scripts.diffusion_policy.evaluation.summarize_capability_audit \
  --protocol scripts/diffusion_policy/experiments/phase_a_a0/capability_audit_v1.json \
  --results_root scripts/diffusion_policy/evaluations/phase_a_capability_audit_v1 \
  --output scripts/diffusion_policy/evaluations/phase_a_capability_audit_v1/capability_decision.json
```

The classifier has three outcomes:

- `base_task_sufficient`: the frozen policy already meets the scoped task;
  adding RL is not justified.
- `command_timing_correction_candidate`: safety, geometry and height pass, but
  mean progress or endpoint timing fails; correct low-dimensional commands
  before considering joint residuals.
- `base_envelope_not_ready`: safety, geometry or pose transitions fail; repair
  or restrict the base distribution before online optimization.
