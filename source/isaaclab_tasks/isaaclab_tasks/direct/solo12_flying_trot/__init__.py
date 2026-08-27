"""Gym registration for the Solo12 flying-trot expert."""

import gymnasium as gym


gym.register(
    id="solo12-flying-trot-v0",
    entry_point="isaaclab_tasks.direct.solo12.solo12_flying_trot_env:Solo12FlyingTrotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.direct.solo12.solo12_flying_trot_env:Solo12FlyingTrotEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.direct.solo12.solo12_flying_trot_env:Solo12FlyingTrotPPORunnerCfg"
        ),
    },
)
