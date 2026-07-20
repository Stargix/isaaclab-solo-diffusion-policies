"""Research scaffold for hierarchical route/pose/time control over DiffuseLoco.

The package deliberately keeps the high-level contract independent from Isaac Lab.  This
makes the reward, route and policy tests runnable on a workstation without launching
Isaac Sim, while :mod:`isaaclab_env` contains the optional simulator adapter.
"""

from .contracts import ActionBounds, RouteObservation
from .rewards import RewardWeights, hierarchical_reward

__all__ = ["ActionBounds", "RouteObservation", "RewardWeights", "hierarchical_reward"]
