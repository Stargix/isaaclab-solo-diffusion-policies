"""Gym registration for the isolated Solo12 bound expert task."""

import gymnasium as gym

gym.register(
    id="solo12-bound-v0",
    entry_point="isaaclab_tasks.direct.solo12.solo12_bound_env:Solo12BoundEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.solo12.solo12_bound_env:Solo12BoundEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.solo12_bound.rsl_rl_ppo_cfg:Solo12BoundPPORunnerCfg",
    },
)
