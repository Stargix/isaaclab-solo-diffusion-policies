"""Hydra/configclass bridge for the Phase B1 environment."""

from isaaclab.utils import configclass

from scripts.residual_diffusion_rl.isaaclab_env import ResidualDiffusionEnvCfg as _ResidualDiffusionEnvCfg


@configclass
class ResidualDiffusionEnvCfg(_ResidualDiffusionEnvCfg):
    pass

