"""Hydra/configclass bridge for the research-only task."""

from pathlib import Path
import sys

# Isaac's Python launcher places the called script, not necessarily the repository
# root, on sys.path. Resolve the research package deterministically.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from isaaclab.utils import configclass

from scripts.hierarchical_diffuseloco.isaaclab_env import HierarchicalSolo12EnvCfg as _HierarchicalSolo12EnvCfg


@configclass
class HierarchicalSolo12EnvCfg(_HierarchicalSolo12EnvCfg):
    """Configclass wrapper so Isaac Lab can compose CLI overrides."""

    pass
