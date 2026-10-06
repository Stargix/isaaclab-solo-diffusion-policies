# Offline policy training

[Method](../../../docs/project/method.md) · [Commands](../../../docs/project/running.md#offline-imitation)

The public experiment uses [achieved-motion profile16 hindsight](configs/walk_crouch_hindsight_geom_profile16_a0_faithful_k10.json): history eight, prediction horizon sixteen, ten denoising steps, episode-wise validation and quadruped augmentation.

The [deterministic control](configs/walk_crouch_hindsight_geom_profile16_deterministic_chunk_v1.json) uses the same transformer and conditions with action regression. Its checkpoint supplies Gaussian PPO; a diffusion checkpoint supplies DPPO.

Other configs remain for checkpoint compatibility and exploration. A reference-path teacher dataset cannot replace achieved-hindsight data merely because its dimensions match. See the [dataset](data/dataset.py) and [goal builder](conditioning/goal_builder.py).
