"""Opt-in nominal-posture observation; never a replacement for physical safety.

Both measured and required heights are averaged over the SAME causal window.
Thus a correctly tracked transition has zero error, without an artificial lag
in the reference. Terminal posture instead compares measured mean height with
the (unsmoothed) endpoint requirement. No gait or speed labels are used.
"""

from __future__ import annotations

import math

import torch


def posture_window_steps(mode: str, window_s: float, dt: float) -> int:
    if mode not in {"instantaneous", "nominal_window"}:
        raise ValueError("Unknown posture_tracking_mode.")
    if not all(math.isfinite(x) and x > 0 for x in (window_s, dt)):
        raise ValueError("posture_window_s and dt must be finite and positive.")
    if mode == "instantaneous":
        return 1
    steps = round(window_s / dt)
    if steps < 1 or not math.isclose(steps * dt, window_s, abs_tol=1e-8):
        raise ValueError("posture_window_s must be a positive integer number of control steps.")
    return steps


class CausalPostureWindow:
    """Batched boxcar with partial startup windows and independent episode reset.

    Call update exactly once per control step, AFTER physics. Reads never
    advance the window. Reset histories contain no fictitious initial samples.
    """

    def __init__(self, num_envs: int, steps: int, device: str | torch.device):
        if num_envs < 1 or steps < 1:
            raise ValueError("num_envs and steps must be positive.")
        self.steps = steps
        self.buffer = torch.zeros(steps, num_envs, 2, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = torch.zeros_like(self.count)
        self.ids = torch.arange(num_envs, device=device)
        self.mean = torch.zeros(num_envs, 2, device=device)

    def update(self, height: torch.Tensor, required: torch.Tensor) -> torch.Tensor:
        samples = torch.stack((height, required), dim=-1)
        self.buffer[self.cursor, self.ids] = samples
        self.count.add_(1).clamp_(max=self.steps)
        self.cursor.add_(1).remainder_(self.steps)
        # Summing the short ring directly avoids running-sum drift over long
        # GPU rollouts and mixed-length episodes (20 samples in this ablation).
        self.mean.copy_(self.buffer.sum(dim=0) / self.count[:, None])
        return self.mean

    def reset(self, env_ids: torch.Tensor) -> None:
        self.buffer[:, env_ids] = 0
        self.count[env_ids] = 0
        self.cursor[env_ids] = 0
        self.mean[env_ids] = 0


def causal_posture_trace(heights, required, *, mode: str, window_s: float, dt: float):
    """Offline counterpart using exactly the same ring as the environment.

    Inputs are one uninterrupted, post-physics episode, not concatenated
    resets. Returned NumPy arrays are independent of the original raw trace.
    """
    steps = posture_window_steps(mode, window_s, dt)
    height = torch.as_tensor(heights, dtype=torch.float32).reshape(-1)
    target = torch.as_tensor(required, dtype=torch.float32).reshape(-1)
    if height.numel() != target.numel():
        raise ValueError("Height and reference traces must have the same length.")
    if mode == "instantaneous":
        # Preserve legacy input precision; do not round old evaluations.
        import numpy as np
        return np.asarray(heights).copy(), np.asarray(required).copy()
    observer = CausalPostureWindow(1, steps, "cpu")
    output = torch.empty(height.numel(), 2)
    for index in range(height.numel()):
        output[index] = observer.update(height[index:index + 1], target[index:index + 1])[0]
    return output[:, 0].numpy(), output[:, 1].numpy()
