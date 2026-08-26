"""Gym registration for the dense, mirror-regularized Solo12 bound expert."""

import gymnasium as gym


gym.register(
    id="solo12-bound-v3",
    entry_point="isaaclab_tasks.direct.solo12.solo12_bound_v3_env:Solo12BoundV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.solo12.solo12_bound_v3_env:Solo12BoundV3EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.direct.solo12_bound.rsl_rl_ppo_cfg:Solo12BoundV3PPORunnerCfg"
        ),
    },
)
