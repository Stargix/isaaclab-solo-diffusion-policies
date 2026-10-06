# Matched deterministic BC + Gaussian PPO

[Results](../../docs/project/results.md#matched-gaussian-ppo-comparison) · [Commands](../../docs/project/running.md#matched-gaussian-ppo)

The baseline refines a pretrained deterministic transformer mean with a diagonal Gaussian policy. The route task, reward, critic input, four-action horizon and nominal online interaction budget match WC-DPPO.

Reported runs use actor learning rate `1e-5`, adaptive rate with floor `1e-7`, initial std `0.04`, clipping `0.2` and target KL `0.02`. Three online seeds share one fixed deterministic imitation prior.

Evaluation executes the mean. This is an imitation-plus-refinement comparison, rather than PPO from scratch or a small-MLP baseline. Different priors and optimizers prevent attributing the complete difference solely to diffusion.

Use the shared [evaluator](../diffusion_policy/evaluate_policy.py). Checkpoints, logs and scheduler scripts are local artifacts.
