"""Optional physical collection diagnostics, deliberately outside policy inputs."""

import torch


def capture_expert_diagnostics(env, selection, joint_ids):
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    data = env._robot.data
    q = data.joint_pos[:, joint_ids]
    limits = data.soft_joint_pos_limits[:, joint_ids]
    values = {
        "joint_limit_margin": torch.minimum(q - limits[..., 0], limits[..., 1] - q),
        "applied_torque": data.applied_torque[:, joint_ids],
    }
    if hasattr(env, "_pronk_robot_foot_ids"):
        ids = env._pronk_robot_foot_ids
        quat = data.body_link_quat_w[:, ids]
        offsets = env._pronk_foot_geometry[:, 0].unsqueeze(0).expand(env.num_envs, -1, -1)
        centers = data.body_link_pos_w[:, ids] + quat_apply(quat, offsets)
        relative = centers - data.root_pos_w[:, None]
        values["foot_centers_b"] = quat_apply_inverse(
            data.root_quat_w[:, None].expand(-1, 4, -1), relative)
        components = getattr(env, "_pronk_cycle_components", None)
        if components is not None:
            # Reward components can outlive an automatic/manual reset. Clear
            # such stale diagnostics on the first frame of each new episode.
            valid = env._pronk_cycle_tracker.age > 0
            for key in ("physical_flight", "completed", "flight_duration", "clearance_m"):
                value = components[key]
                mask = valid.reshape((-1,) + (1,) * (value.ndim - 1))
                values[key] = torch.where(mask, value, torch.zeros_like(value))
        else:
            values.update(physical_flight=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
                          completed=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
                          flight_duration=torch.zeros(env.num_envs, device=env.device),
                          clearance_m=torch.zeros((env.num_envs, 4), device=env.device))
    return {"diagnostics_" + key: value[selection].detach().cpu().numpy().copy()
            for key, value in values.items()}
