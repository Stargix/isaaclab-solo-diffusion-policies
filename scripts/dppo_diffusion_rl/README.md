# Diffusion policy refinement

[Method](../../docs/project/method.md) · [Commands](../../docs/project/running.md#online-diffusion-refinement) · [Results](../../docs/project/results.md)

DPPO refines the joint-action diffusion policy from offline imitation. The evaluated recipe uses `supported_hybrid_v3`, ten denoising steps with five trained reverse transitions, and four executed actions per plan.

The objective combines route adherence, native-height posture, average arrival speed, terminal pose and safety. Explicit remaining-speed feedback is supplied without a gait identifier, gait reward or required local speed profile.

| Module | Role |
|---|---|
| [policy.py](policy.py) | Reverse transitions and policy densities |
| [ppo.py](ppo.py) | Policy and critic updates |
| [rewards.py](rewards.py) | Task terms and outcomes |
| [hybrid_routes.py](hybrid_routes.py), [supported_hybrid_v3.py](supported_hybrid_v3.py) | Route generation and task sampling |
| [isaaclab_env.py](isaaclab_env.py) | Task state and conditioning |
| [checkpointing.py](checkpointing.py) | Source validation and restoration |
| [Evaluation v1](experiments/wct_final_evaluation_v1/) | Frozen benchmark banks |
| [Evaluation v2](experiments/wct_final_evaluation_v2/) | Geometry-difficulty diagnostics |

Historical task versions and opt-in symmetry/preview variants remain for existing checkpoints. Their presence does not make them validated improvements. Machine-specific scheduler launchers are local files.
