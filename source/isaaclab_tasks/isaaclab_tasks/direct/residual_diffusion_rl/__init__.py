"""Gym registration for Phase B1 residual diffusion RL."""

from pathlib import Path
import sys

import gymnasium as gym

# ``isaaclab.bat -p path/to/script.py`` adds extension ``source`` folders but
# not the repository root. B1 intentionally keeps research code under
# ``scripts/residual_diffusion_rl``; make that package reachable without a
# machine-specific PYTHONPATH export.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[5])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from . import agents

gym.register(
    id="solo12-residual-diffusion-rl-v0",
    entry_point="scripts.residual_diffusion_rl.isaaclab_env:ResidualDiffusionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ResidualDiffusionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ResidualDiffusionPPOCfg",
    },
)
