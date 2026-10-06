# Path-conditioned diffusion policy

[Project overview](../../README.md) · [Method](../../docs/project/method.md) · [Results](../../docs/project/results.md) · [Running guide](../../docs/project/running.md)

This package turns velocity-expert trajectories into spatial and temporal conditioning through hindsight relabelling, then trains and evaluates a joint-action policy.

| Entry point | Role |
|---|---|
| [data/](data/) | Collect, audit and prepare expert trajectories |
| [train/](train/) | Diffusion imitation and matched deterministic action regression |
| [model/](model/) | Transformer policies and action-sequence generation |
| [play_policy.py](play_policy.py) | Interactive Isaac Lab playback and route overlays |
| [evaluate_policy.py](evaluate_policy.py) | Paired frozen-route rollouts |
| [evaluation/](evaluation/) | Offline diagnostics of geometry, posture, timing and gait |

Online refinement lives in [DPPO](../dppo_diffusion_rl/) and the [Gaussian comparison](../gaussian_chunk_rl/). Preserve each checkpoint's conditioning and execution contract when comparing models. Datasets, weights and raw outputs remain local.
