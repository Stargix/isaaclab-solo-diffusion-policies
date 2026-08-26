"""Gym registration for the strict straight Solo12 bound expert."""

import gymnasium as gym


gym.register(
    id="solo12-bound-v2",
    entry_point="isaaclab_tasks.direct.solo12.solo12_bound_v2_env:Solo12BoundV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.solo12.solo12_bound_v2_env:Solo12BoundV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.direct.solo12_bound.rsl_rl_ppo_cfg:Solo12BoundV2PPORunnerCfg"
        ),
    },
)
