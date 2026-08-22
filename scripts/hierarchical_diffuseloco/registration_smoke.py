"""Run with ``isaaclab.bat -p`` to verify task discovery inside Isaac's runtime."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: E402,F401  # registration side effect


spec = gym.spec("solo12-hierarchical-diffuseloco-v0")
assert spec.entry_point == "scripts.hierarchical_diffuseloco.isaaclab_env:HierarchicalSolo12Env"
print(f"registered: {spec.id} -> {spec.entry_point}")
simulation_app.close()
