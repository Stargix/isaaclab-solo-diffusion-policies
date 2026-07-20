"""Run with ``isaaclab.bat -p`` to verify task discovery inside Isaac's Python."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  # registration side effect


spec = gym.spec("solo12-hierarchical-diffuseloco-v0")
assert spec.entry_point == "scripts.hierarchical_diffuseloco.isaaclab_env:HierarchicalSolo12Env"
print(f"registered: {spec.id} -> {spec.entry_point}")
