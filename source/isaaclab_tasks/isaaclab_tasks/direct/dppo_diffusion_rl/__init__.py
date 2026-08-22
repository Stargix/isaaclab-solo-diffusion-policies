"""Gym registration for direct DPPO fine-tuning."""

from pathlib import Path
import sys

import gymnasium as gym

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[5])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

gym.register(
    id="solo12-dppo-diffusion-rl-v0",
    entry_point="scripts.dppo_diffusion_rl.isaaclab_env:DPPODiffusionEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:DPPODiffusionEnvCfg"},
)
