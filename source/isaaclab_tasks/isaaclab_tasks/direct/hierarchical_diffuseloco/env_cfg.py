"""Hydra/configclass bridge for the research-only task."""

from isaaclab.utils import configclass

from scripts.hierarchical_diffuseloco.isaaclab_env import HierarchicalSolo12EnvCfg as _HierarchicalSolo12EnvCfg


@configclass
class HierarchicalSolo12EnvCfg(_HierarchicalSolo12EnvCfg):
    """Configclass wrapper so Isaac Lab can compose CLI overrides."""

    pass
