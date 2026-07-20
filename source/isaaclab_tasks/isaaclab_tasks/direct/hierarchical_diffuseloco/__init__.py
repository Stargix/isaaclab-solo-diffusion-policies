"""Gym registration for the frozen DiffuseLoco high-level experiment."""

import gymnasium as gym

from . import agents

gym.register(
    id="solo12-hierarchical-diffuseloco-v0",
    entry_point="scripts.hierarchical_diffuseloco.isaaclab_env:HierarchicalSolo12Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HierarchicalSolo12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HierarchicalPPOCfg",
    },
)
