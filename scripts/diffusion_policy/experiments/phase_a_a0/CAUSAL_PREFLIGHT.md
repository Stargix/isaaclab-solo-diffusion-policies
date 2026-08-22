# Phase A causal preflight

This is the only authorized Phase 1 computation before selecting an execution
horizon.  It evaluates 54 vectorized scenarios per run: three paths, three
speeds, two endpoint heights and three stochastic policy repeats.

The environment reset is deterministic.  Diffusion samples remain stochastic;
causality is therefore assessed across paired condition groups rather than from
a single trajectory.

## 1. Verify A0

```bash
python scripts/diffusion_policy/experiments/verify_artifacts.py \
  scripts/diffusion_policy/experiments/phase_a_a0/manifest.json
```

Stop if verification fails.

## 2. Generate cluster commands

```bash
python scripts/diffusion_policy/evaluation/prepare_protocol.py \
  --protocol scripts/diffusion_policy/experiments/phase_a_a0/causal_preflight_v1.json \
  --platform linux
```

The generator emits one line per run.  These are evaluations, not training
jobs.  Run them from a clean committed checkout so provenance validation can
distinguish the code that produced the outputs.  Each command refuses a
non-empty result directory and explicitly checks that the summary exists,
because some Isaac/Hydra launcher failures can otherwise return exit code zero.

## 3. Select the execution horizon

```bash
python scripts/diffusion_policy/evaluation/summarize_preflight.py \
  --exec1 scripts/diffusion_policy/evaluations/phase_a_causal_audit_v1/preflight_exec1 \
  --exec4 scripts/diffusion_policy/evaluations/phase_a_causal_audit_v1/preflight_exec4 \
  --output scripts/diffusion_policy/evaluations/phase_a_causal_audit_v1/preflight_decision.json
```

The preflight deliberately has a weaker survival threshold (80%) than the final
90–95% gate.  Its purpose is to choose a valid inference contract cheaply.  If
both horizons fail survival or causal height response, the decision is
`base_not_ready` and Phase 1 stops: no transition suite and no RL are launched.
