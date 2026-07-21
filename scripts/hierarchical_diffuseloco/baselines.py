"""Non-learning baselines used to make the RL claim falsifiable."""

from __future__ import annotations

import numpy as np

from .contracts import ActionBounds, RouteObservation


class ClassicalPathTracker:
    """A proportional path/deadline controller; it has no learned smoothness."""

    def __init__(self, bounds: ActionBounds = ActionBounds(), cruise_speed: float = 0.35,
                 lateral_gain: float = 1.2, yaw_gain: float = 1.0):
        self.bounds, self.cruise_speed = bounds, float(cruise_speed)
        self.lateral_gain, self.yaw_gain = float(lateral_gain), float(yaw_gain)

    def __call__(self, observation: RouteObservation) -> np.ndarray:
        target = observation.preview[0]
        yaw_error = float(np.arctan2(target[2], target[3]))
        command = np.asarray((self.cruise_speed, self.lateral_gain * target[1], self.yaw_gain * yaw_error,
                              target[4]), dtype=np.float32)
        return self.bounds.clip(command)


class DiscreteSkillSelector:
    """Selector baseline (walk/crouch/sprint prototypes) for the ablation table."""

    def __init__(self, skills: dict[str, np.ndarray], bounds: ActionBounds = ActionBounds()):
        if not skills:
            raise ValueError("At least one skill prototype is required.")
        self.skills = {name: bounds.clip(value) for name, value in skills.items()}
        self.bounds = bounds

    def __call__(self, observation: RouteObservation) -> np.ndarray:
        maximum = float(observation.clearance)
        feasible = [(maximum - float(cmd[3]), cmd) for cmd in self.skills.values() if cmd[3] <= maximum]
        return min(feasible or [(float("inf"), cmd) for cmd in self.skills.values()], key=lambda item: item[0])[1].copy()
