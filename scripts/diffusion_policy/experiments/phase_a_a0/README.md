# Phase A / A0 frozen baseline

This directory freezes the last useful walk/crouch diffusion checkpoint before
the causal audit.  It does not duplicate the 20 MB checkpoint or the 680 MB
dataset; `manifest.json` identifies them by repository-relative path, byte size
and SHA-256.

The checkpoint-embedded configuration is the source of truth.  In particular,
this model was trained for 50 epochs with batch size 4096.  The similarly named
JSON currently under `train/configs` has changed since training and must not be
used to claim exact reproduction of A0.

## Verify A0

From the repository root:

```powershell
conda run --no-capture-output -n env_isaaclab python `
  scripts/diffusion_policy/experiments/verify_artifacts.py `
  scripts/diffusion_policy/experiments/phase_a_a0/manifest.json
```

Linux/cluster equivalent:

```bash
python scripts/diffusion_policy/experiments/verify_artifacts.py \
  scripts/diffusion_policy/experiments/phase_a_a0/manifest.json
```

The command is deliberately independent of Isaac Sim and PyTorch.  A non-zero
exit code means that the experiment is not A0 and evaluation must stop.

## Interpretation boundary

A0 is a partial-generalization baseline, not an RL-ready locomotion prior.
Historical evaluation outputs are retained under
`scripts/diffusion_policy/evaluations`, but they did not record git dirty state
and lack longitudinal completion and counterfactual-goal metrics.  They must not
be mixed with the canonical causal audit that follows.

The local dataset audit is generated under the ignored path
`scripts/diffusion_policy/data/audits/phase_a_a0`.  Its key immutable findings
are copied into the manifest: the dataset has no walk/crouch boundary within the
2 s conditioning horizon, and only 1.03% of sampled histories resemble the
all-zero deployment warm-up.
