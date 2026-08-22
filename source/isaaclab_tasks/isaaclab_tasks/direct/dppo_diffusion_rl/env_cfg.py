"""Isaac Lab configclass bridge for the DPPO route task."""

from isaaclab.utils import configclass

from scripts.dppo_diffusion_rl.isaaclab_env import DPPODiffusionEnvCfg as _DPPODiffusionEnvCfg


@configclass
class DPPODiffusionEnvCfg(_DPPODiffusionEnvCfg):
    pass
