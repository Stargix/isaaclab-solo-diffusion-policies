"""Solo12 symmetry augmentation for diffusion-policy training samples."""

from __future__ import annotations

import torch

LEFT_RIGHT_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long)
FRONT_BACK_PERM = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=torch.long)
LEFT_RIGHT_SIGN = torch.tensor([-1.0, 1.0, 1.0] * 4)
FRONT_BACK_SIGN = torch.tensor([1.0, -1.0, -1.0] * 4)

VECTOR_REFLECT_X = torch.tensor([1.0, -1.0, 1.0])
VECTOR_REFLECT_Y = torch.tensor([-1.0, 1.0, 1.0])
PSEUDOVECTOR_REFLECT_X = torch.tensor([-1.0, 1.0, -1.0])
PSEUDOVECTOR_REFLECT_Y = torch.tensor([1.0, -1.0, -1.0])
GOAL_XY_REFLECT_X = torch.tensor([1.0, -1.0])
GOAL_XY_REFLECT_Y = torch.tensor([-1.0, 1.0])


def symmetry_count(mode: str) -> int:
    if mode == "none":
        return 1
    if mode == "mirror":
        return 2
    if mode == "quadruped":
        return 4
    raise ValueError(f"Unsupported symmetry mode: {mode}")


def symmetry_name(index: int, mode: str) -> str:
    names = {
        "none": ("identity",),
        "mirror": ("identity", "reflect_x"),
        "quadruped": ("identity", "reflect_x", "reflect_y", "rotate_180"),
    }
    return names[mode][index]


def _device_tensor(values: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return values.to(device=ref.device, dtype=ref.dtype)


def _transform_joint_data(data: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    perm = perm.to(device=data.device)
    sign = _device_tensor(sign, data)
    return data[..., perm] * sign


def _reflect_proprio_x(proprio_hist: torch.Tensor) -> torch.Tensor:
    proprio = proprio_hist.clone()
    proprio[..., 0:12] = _transform_joint_data(proprio[..., 0:12], LEFT_RIGHT_PERM, LEFT_RIGHT_SIGN)
    proprio[..., 12:24] = _transform_joint_data(proprio[..., 12:24], LEFT_RIGHT_PERM, LEFT_RIGHT_SIGN)
    proprio[..., 24:27] *= _device_tensor(PSEUDOVECTOR_REFLECT_X, proprio)
    proprio[..., 27:30] *= _device_tensor(VECTOR_REFLECT_X, proprio)
    return proprio


def _reflect_proprio_y(proprio_hist: torch.Tensor) -> torch.Tensor:
    proprio = proprio_hist.clone()
    proprio[..., 0:12] = _transform_joint_data(proprio[..., 0:12], FRONT_BACK_PERM, FRONT_BACK_SIGN)
    proprio[..., 12:24] = _transform_joint_data(proprio[..., 12:24], FRONT_BACK_PERM, FRONT_BACK_SIGN)
    proprio[..., 24:27] *= _device_tensor(PSEUDOVECTOR_REFLECT_Y, proprio)
    proprio[..., 27:30] *= _device_tensor(VECTOR_REFLECT_Y, proprio)
    return proprio


def _reflect_goal_x(goal_hist: torch.Tensor) -> torch.Tensor:
    goal = goal_hist.clone()
    if goal.shape[-1] == 3:
        goal *= _device_tensor(torch.tensor([1.0, -1.0, -1.0]), goal)
        return goal
    if goal.shape[-1] == 4:
        goal *= _device_tensor(torch.tensor([1.0, -1.0, -1.0, 1.0]), goal)
        return goal
    xy_sign = _device_tensor(GOAL_XY_REFLECT_X, goal)
    xyz_sign = _device_tensor(VECTOR_REFLECT_X, goal)
    goal[..., 0:2] *= xy_sign
    goal[..., 2:4] *= xy_sign
    goal[..., 4:6] *= xy_sign
    goal[..., 6:9] *= xyz_sign
    goal[..., 9] *= -1.0
    return goal


def _reflect_goal_y(goal_hist: torch.Tensor) -> torch.Tensor:
    goal = goal_hist.clone()
    if goal.shape[-1] == 3:
        goal *= _device_tensor(torch.tensor([-1.0, 1.0, -1.0]), goal)
        return goal
    if goal.shape[-1] == 4:
        goal *= _device_tensor(torch.tensor([-1.0, 1.0, -1.0, 1.0]), goal)
        return goal
    xy_sign = _device_tensor(GOAL_XY_REFLECT_Y, goal)
    xyz_sign = _device_tensor(VECTOR_REFLECT_Y, goal)
    goal[..., 0:2] *= xy_sign
    goal[..., 2:4] *= xy_sign
    goal[..., 4:6] *= xy_sign
    goal[..., 6:9] *= xyz_sign
    goal[..., 9] *= -1.0
    return goal


def apply_symmetry(
    proprio_hist: torch.Tensor,
    action_hist: torch.Tensor,
    goal_hist: torch.Tensor,
    actions: torch.Tensor,
    *,
    index: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one symmetry transform to unnormalized tensors."""

    name = symmetry_name(index, mode)
    if name == "identity":
        return proprio_hist, action_hist, goal_hist, actions
    if name == "reflect_x":
        return (
            _reflect_proprio_x(proprio_hist),
            _transform_joint_data(action_hist, LEFT_RIGHT_PERM, LEFT_RIGHT_SIGN),
            _reflect_goal_x(goal_hist),
            _transform_joint_data(actions, LEFT_RIGHT_PERM, LEFT_RIGHT_SIGN),
        )
    if name == "reflect_y":
        return (
            _reflect_proprio_y(proprio_hist),
            _transform_joint_data(action_hist, FRONT_BACK_PERM, FRONT_BACK_SIGN),
            _reflect_goal_y(goal_hist),
            _transform_joint_data(actions, FRONT_BACK_PERM, FRONT_BACK_SIGN),
        )
    if name == "rotate_180":
        proprio_x, action_x, goal_x, actions_x = apply_symmetry(
            proprio_hist, action_hist, goal_hist, actions, index=1, mode="quadruped"
        )
        return apply_symmetry(proprio_x, action_x, goal_x, actions_x, index=2, mode="quadruped")
    raise ValueError(f"Unsupported symmetry transform: {name}")
