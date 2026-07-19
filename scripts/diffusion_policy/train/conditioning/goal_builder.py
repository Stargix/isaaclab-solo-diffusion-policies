"""Shared temporal-preview goal construction for hindsight and deployment."""

from __future__ import annotations

import math

import numpy as np

from .geometry import relative_yaw, transform_point_to_yaw_frame


# The vector layout is identical for achieved-hindsight and explicit-reference
# data.  The schema name carries the semantic distinction so checkpoints cannot
# silently train on one and deploy as the other.
GOAL_SCHEMA_NAME = "spatial_time_preview11_v1"
REFERENCE_GOAL_SCHEMA_NAME = "spatial_reference_path11_v1"
HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME = "holonomic_reference_se2_32_v1"
PATH_GUIDANCE_GOAL_SCHEMA_NAME = "path_guidance_se2_terminal36_v1"
WAYPOINT_TIME_OFFSETS_S = (0.5, 1.0, 1.5)
# The t=0 token makes the instantaneous cross-track/yaw error observable to
# the student, exactly as it is to the route-tracking teacher.  The remaining
# three previews retain the finite look-ahead needed for anticipatory control.
HOLONOMIC_TOKEN_TIMES_S = (0.0, 0.5, 1.0, 1.5)
PATH_GUIDANCE_FRACTIONS = tuple(float(value) for value in np.linspace(0.0, 1.0, 7))
REFERENCE_GOAL_REPRESENTATIONS = ("path11", "holonomic_se2_32", "path_guidance_se2_36")


def goal_dimension(goal_representation: str) -> int:
    """Return the flat conditioning dimension for a declared route contract."""

    if goal_representation == "path11":
        return 11
    if goal_representation == "holonomic_se2_32":
        return 32
    if goal_representation == "path_guidance_se2_36":
        return 36
    raise ValueError(f"Unsupported goal representation {goal_representation!r}.")


def goal_schema_name(goal_representation: str, *, reference: bool) -> str:
    """Map a goal representation to its checkpoint-visible schema name."""

    if goal_representation == "holonomic_se2_32":
        if not reference:
            raise ValueError("holonomic_se2_32 requires an explicit reference route.")
        return HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME
    if goal_representation == "path_guidance_se2_36":
        if not reference:
            raise ValueError("path_guidance_se2_36 requires an explicit reference route.")
        return PATH_GUIDANCE_GOAL_SCHEMA_NAME
    return REFERENCE_GOAL_SCHEMA_NAME if reference else GOAL_SCHEMA_NAME


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Build a yaw-only WXYZ quaternion."""

    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _wrap_to_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def closest_path_index(
    path_w: np.ndarray,
    robot_xy: np.ndarray,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> int:
    """Return the index of the closest point on a 3D path (uses XY only)."""

    path_xy = np.asarray(path_w, dtype=np.float32)[:, :2]
    robot_xy = np.asarray(robot_xy, dtype=np.float32).reshape(-1)[:2]
    start_idx = int(np.clip(start_idx, 0, len(path_xy) - 1))
    end_idx = len(path_xy) if end_idx is None else int(np.clip(end_idx, start_idx + 1, len(path_xy)))
    dists = np.linalg.norm(path_xy[start_idx:end_idx] - robot_xy[None, :], axis=-1)
    return start_idx + int(np.argmin(dists))


def advance_path_progress(
    path_w: np.ndarray,
    robot_xy: np.ndarray,
    progress_idx: int,
    *,
    search_back: int = 5,
    search_forward: int = 50,
) -> int:
    """Advance monotonically without jumping to a distant self-crossing branch."""

    progress_idx = int(np.clip(progress_idx, 0, len(path_w) - 1))
    low = max(0, progress_idx - max(0, int(search_back)))
    high = min(len(path_w), progress_idx + max(1, int(search_forward)) + 1)
    closest = closest_path_index(path_w, robot_xy, start_idx=low, end_idx=high)
    return max(int(progress_idx), closest)


def _validate_preview_times(preview_times_s: tuple[float, float, float], terminal_time_s: float) -> None:
    if len(preview_times_s) != 3:
        raise ValueError("The spatial11 schema requires exactly three preview times.")
    if any(time_s <= 0.0 for time_s in preview_times_s):
        raise ValueError("Preview times must be positive.")
    if tuple(sorted(preview_times_s)) != tuple(preview_times_s):
        raise ValueError("Preview times must be strictly ordered.")
    if len(set(preview_times_s)) != len(preview_times_s):
        raise ValueError("Preview times must be unique.")
    if preview_times_s[-1] >= terminal_time_s:
        raise ValueError("Every preview time must be earlier than the terminal horizon.")


def _training_temporal_preview(
    pos_w: np.ndarray,
    step_idx: int,
    end_idx: int,
    origin_w: np.ndarray,
    quat_origin: np.ndarray,
    *,
    dt: float,
    preview_times_s: tuple[float, float, float],
) -> np.ndarray:
    values: list[float] = []
    for time_s in preview_times_s:
        preview_idx = min(step_idx + int(round(time_s / dt)), end_idx)
        local = transform_point_to_yaw_frame(pos_w[preview_idx], origin_w, quat_origin)
        values.extend((float(local[0]), float(local[1])))
    return np.asarray(values, dtype=np.float32)


def build_holonomic_reference_goal_vector(
    pos_w: np.ndarray,
    step_idx: int,
    end_idx: int,
    origin_w: np.ndarray,
    quat_origin: np.ndarray,
    *,
    yaws_w: np.ndarray,
    reference_command_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build four SE(2) preview tokens shared by teacher and student.

    A path alone under-specifies a holonomic task: the same XY curve can be
    traversed while facing tangent, sideways or backwards.  Every token hence
    carries position, desired body yaw, nominal body-frame velocity and its
    timestamp.  These are reference derivatives, not the teacher's feedback
    command; at deployment they are supplied by the same route planner.
    """

    if reference_command_b.shape[-1] != 3:
        raise ValueError("reference_command_b must have three SE(2) command components.")
    terminal_time_s = max((end_idx - step_idx) * dt, dt)
    if HOLONOMIC_TOKEN_TIMES_S[-1] > terminal_time_s + 1.0e-6:
        raise ValueError("Holonomic token horizon exceeds the reference endpoint.")
    values: list[float] = []
    for time_s in HOLONOMIC_TOKEN_TIMES_S:
        index = min(step_idx + int(round(time_s / dt)), end_idx)
        local = transform_point_to_yaw_frame(pos_w[index], origin_w, quat_origin)
        dyaw = relative_yaw(quat_origin, yaw_to_quat_wxyz(float(yaws_w[index])))
        vx_ref, vy_ref, wz_ref = (float(value) for value in reference_command_b[index, :3])
        cos_yaw, sin_yaw = math.cos(float(dyaw)), math.sin(float(dyaw))
        # The reference command is expressed in the reference body frame.  The
        # policy condition is expressed in its current body/yaw frame.
        vx_local = cos_yaw * vx_ref - sin_yaw * vy_ref
        vy_local = sin_yaw * vx_ref + cos_yaw * vy_ref
        values.extend(
            (
                float(local[0]), float(local[1]),
                math.sin(float(dyaw)), math.cos(float(dyaw)),
                vx_local, vy_local, wz_ref, float(time_s),
            )
        )
    return np.asarray(values, dtype=np.float32)


def build_path_guidance_goal_vector(
    guidance_pos_w: np.ndarray,
    guidance_cumulative_xy: np.ndarray,
    reference_pos_w: np.ndarray,
    reference_yaw_w: np.ndarray,
    step_idx: int,
    terminal_idx: int,
    origin_w: np.ndarray,
    quat_origin: np.ndarray,
    *,
    dt: float,
    time_to_go_s: float | None = None,
) -> np.ndarray:
    """Encode fallible path context plus an authoritative terminal pose.

    Seven path tokens contain ``[dir_x, dir_y, log1p(distance), arc_offset]``.
    They are sampled uniformly in remaining guide arc length and deliberately
    contain no reference or teacher velocity.  The last eight scalars are the
    clean terminal SE(2) pose, height, time-to-go and stop/valid flags.
    """

    guidance = np.asarray(guidance_pos_w, dtype=np.float32)
    cumulative = np.asarray(guidance_cumulative_xy, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_pos_w, dtype=np.float32)
    yaws = np.asarray(reference_yaw_w, dtype=np.float32).reshape(-1)
    if guidance.shape != reference.shape or guidance.ndim != 2 or guidance.shape[1] != 3:
        raise ValueError("Guidance and reference positions must share shape (T,3).")
    if cumulative.shape != (len(guidance),) or yaws.shape != (len(reference),):
        raise ValueError("Guidance cumulative length and reference yaw must have length T.")

    step = int(np.clip(step_idx, 0, len(guidance) - 1))
    terminal = int(np.clip(terminal_idx, step, len(guidance) - 1))
    start_s, end_s = float(cumulative[step]), float(cumulative[terminal])
    values: list[float] = []
    for fraction in PATH_GUIDANCE_FRACTIONS:
        target_s = start_s + float(fraction) * max(0.0, end_s - start_s)
        index = int(np.searchsorted(cumulative, target_s, side="left"))
        index = int(np.clip(index, step, terminal))
        local = transform_point_to_yaw_frame(guidance[index], origin_w, quat_origin)[:2]
        distance = float(np.linalg.norm(local))
        if distance > 1.0e-6:
            direction = local / distance
        else:
            direction = np.zeros(2, dtype=np.float32)
        arc_offset = float(fraction) * max(0.0, end_s - start_s)
        values.extend((float(direction[0]), float(direction[1]), math.log1p(distance), arc_offset))

    terminal_local = transform_point_to_yaw_frame(reference[terminal], origin_w, quat_origin)
    terminal_dyaw = relative_yaw(quat_origin, yaw_to_quat_wxyz(float(yaws[terminal])))
    remaining = max(0.0, float(terminal - step) * float(dt)) if time_to_go_s is None else max(0.0, float(time_to_go_s))
    guidance_terminal_error = float(np.linalg.norm(guidance[terminal, :2] - reference[terminal, :2]))
    values.extend((
        float(terminal_local[0]),
        float(terminal_local[1]),
        math.sin(float(terminal_dyaw)),
        math.cos(float(terminal_dyaw)),
        float(reference[terminal, 2]),
        remaining,
        1.0 if remaining <= 1.0e-6 else 0.0,  # terminal/hold phase
        guidance_terminal_error,  # observable guide reliability at the endpoint
    ))
    return np.asarray(values, dtype=np.float32)


def _planned_temporal_preview(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    start_idx: int,
    end_idx: int,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    speed: float,
    preview_times_s: tuple[float, float, float],
) -> np.ndarray:
    values: list[float] = []
    start_s = float(cumulative_xy[start_idx])
    end_s = float(cumulative_xy[end_idx])
    for time_s in preview_times_s:
        target_s = min(start_s + max(0.0, float(speed)) * time_s, end_s)
        preview_idx = int(np.searchsorted(cumulative_xy, target_s, side="left"))
        preview_idx = int(np.clip(preview_idx, start_idx, end_idx))
        local = transform_point_to_yaw_frame(path_w[preview_idx], robot_pos_w, robot_quat_w)
        values.extend((float(local[0]), float(local[1])))
    return np.asarray(values, dtype=np.float32)


def resolve_end_idx(
    cumulative_xy: np.ndarray,
    start_idx: int,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
) -> int:
    """Pick an end index using arc-length lookahead (matches training scale).

    Training hindsight uses ``segment_steps in [50, 150]`` ticks at 50 Hz, i.e.
    about 1–3 s ahead. We map that to ``speed * goal_horizon_steps * dt`` meters
    along the reference path instead of jumping to the global terminus.
    """

    if len(cumulative_xy) == 0:
        return 0
    start_idx = int(np.clip(start_idx, 0, len(cumulative_xy) - 1))
    lookahead_m = max(float(speed) * float(goal_horizon_steps) * float(dt), float(dt) * float(speed))
    target_s = min(cumulative_xy[start_idx] + lookahead_m, cumulative_xy[-1])
    end_idx = int(np.searchsorted(cumulative_xy, target_s, side="left"))
    return int(np.clip(end_idx, start_idx, len(cumulative_xy) - 1))


def build_goal_vector(
    pos_w: np.ndarray,
    cumulative_xy: np.ndarray,
    step_idx: int,
    end_idx: int,
    origin_w: np.ndarray,
    quat_origin: np.ndarray,
    *,
    quat_w: np.ndarray | None = None,
    yaws_w: np.ndarray | None = None,
    dt: float = 0.02,
    waypoint_time_offsets_s: tuple[float, float, float] = WAYPOINT_TIME_OFFSETS_S,
    v_req_clip: float = 2.0,
    goal_representation: str = "path11",
    reference_command_b: np.ndarray | None = None,
    guidance_pos_w: np.ndarray | None = None,
    guidance_cumulative_xy: np.ndarray | None = None,
    terminal_idx: int | None = None,
) -> np.ndarray:
    """Build a declared goal vector shared by the DataLoader and inference."""

    if quat_w is None and yaws_w is None:
        raise ValueError("Either quat_w or yaws_w must be provided.")

    if goal_representation == "holonomic_se2_32":
        if yaws_w is None or reference_command_b is None:
            raise ValueError("holonomic_se2_32 requires reference yaw and nominal reference commands.")
        return build_holonomic_reference_goal_vector(
            pos_w, step_idx, end_idx, origin_w, quat_origin,
            yaws_w=yaws_w, reference_command_b=reference_command_b, dt=dt,
        )
    if goal_representation == "path_guidance_se2_36":
        if yaws_w is None or guidance_pos_w is None or guidance_cumulative_xy is None:
            raise ValueError("path_guidance_se2_36 requires reference yaw and a stored guidance path.")
        return build_path_guidance_goal_vector(
            guidance_pos_w,
            guidance_cumulative_xy,
            pos_w,
            yaws_w,
            step_idx,
            len(pos_w) - 1 if terminal_idx is None else terminal_idx,
            origin_w,
            quat_origin,
            dt=dt,
        )
    if goal_representation != "path11":
        raise ValueError(f"Unsupported goal representation {goal_representation!r}.")

    terminal_time_s = max((end_idx - step_idx) * dt, dt)
    _validate_preview_times(waypoint_time_offsets_s, terminal_time_s)
    waypoints = _training_temporal_preview(
        pos_w,
        step_idx,
        end_idx,
        origin_w,
        quat_origin,
        dt=dt,
        preview_times_s=waypoint_time_offsets_s,
    )
    target_w = pos_w[end_idx]
    target_rel = transform_point_to_yaw_frame(target_w, origin_w, quat_origin)

    if quat_w is not None:
        dyaw = relative_yaw(quat_origin, quat_w[end_idx])
    else:
        dyaw = relative_yaw(quat_origin, yaw_to_quat_wxyz(float(yaws_w[end_idx])))

    path_length_remaining = float(cumulative_xy[end_idx] - cumulative_xy[step_idx])
    time_remaining = max((end_idx - step_idx) * dt, dt)
    v_req = float(np.clip(path_length_remaining / (time_remaining + 1.0e-3), 0.0, v_req_clip))

    return np.asarray(
        [
            *waypoints.tolist(),
            float(target_rel[0]),
            float(target_rel[1]),
            float(target_w[2]),  # Absolute target base height (project-specific spatial goal)
            float(dyaw),
            v_req,
        ],
        dtype=np.float32,
    )


def build_goal_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    start_idx: int | None = None,
    waypoint_time_offsets_s: tuple[float, float, float] = WAYPOINT_TIME_OFFSETS_S,
    v_req_clip: float = 2.0,
) -> np.ndarray:
    """Build a single-env goal from a planned world-frame path."""

    terminal_time_s = float(goal_horizon_steps) * float(dt)
    _validate_preview_times(waypoint_time_offsets_s, terminal_time_s)
    if start_idx is None:
        start_idx = closest_path_index(path_w, robot_pos_w)
    start_idx = int(np.clip(start_idx, 0, len(path_w) - 1))
    end_idx = resolve_end_idx(
        cumulative_xy,
        start_idx,
        goal_horizon_steps=goal_horizon_steps,
        dt=dt,
        speed=speed,
    )
    path_length_remaining = float(cumulative_xy[end_idx] - cumulative_xy[start_idx])
    time_remaining = max(float(goal_horizon_steps) * dt, dt)
    v_req = float(np.clip(path_length_remaining / (time_remaining + 1.0e-3), 0.0, v_req_clip))

    waypoints = _planned_temporal_preview(
        path_w,
        cumulative_xy,
        start_idx,
        end_idx,
        robot_pos_w,
        robot_quat_w,
        speed=speed,
        preview_times_s=waypoint_time_offsets_s,
    )
    target_rel = transform_point_to_yaw_frame(path_w[end_idx], robot_pos_w, robot_quat_w)
    dyaw = relative_yaw(robot_quat_w, yaw_to_quat_wxyz(float(yaws_w[end_idx])))

    return np.asarray(
        [
            *waypoints.tolist(),
            float(target_rel[0]),
            float(target_rel[1]),
            float(path_w[end_idx, 2]),  # Absolute target base height
            float(dyaw),
            v_req,
        ],
        dtype=np.float32,
    )


def derive_reference_commands_from_path(
    path_w: np.ndarray,
    yaws_w: np.ndarray,
    cumulative_xy: np.ndarray,
    *,
    speed: float,
    dt: float,
) -> np.ndarray:
    """Derive a conservative tangent-frame schedule for legacy analytic paths.

    This compatibility helper is only for playback of an externally supplied
    XYZ(+yaw) path.  The holonomic collector stores its actual SE(2) schedule,
    which should be passed directly whenever it is available.
    """

    command = np.zeros((len(path_w), 3), dtype=np.float32)
    if len(path_w) < 2:
        return command
    for index in range(len(path_w) - 1):
        ds = max(float(cumulative_xy[index + 1] - cumulative_xy[index]), 1.0e-6)
        heading = math.atan2(
            float(path_w[index + 1, 1] - path_w[index, 1]),
            float(path_w[index + 1, 0] - path_w[index, 0]),
        )
        delta = _wrap_to_pi(heading - float(yaws_w[index]))
        command[index, 0] = float(speed * math.cos(delta))
        command[index, 1] = float(speed * math.sin(delta))
        command[index, 2] = float(_wrap_to_pi(float(yaws_w[index + 1] - yaws_w[index])) / max(dt, ds / max(speed, 1.0e-3)))
    command[-1] = command[-2]
    return command


def build_holonomic_goal_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    start_idx: int | None = None,
    reference_command_b: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 32D route condition for deployment from an SE(2) path plan."""

    if start_idx is None:
        start_idx = closest_path_index(path_w, robot_pos_w)
    start_idx = int(np.clip(start_idx, 0, len(path_w) - 1))
    end_idx = resolve_end_idx(
        cumulative_xy, start_idx, goal_horizon_steps=goal_horizon_steps, dt=dt, speed=speed
    )
    commands = (
        derive_reference_commands_from_path(path_w, yaws_w, cumulative_xy, speed=speed, dt=dt)
        if reference_command_b is None else np.asarray(reference_command_b, dtype=np.float32)
    )
    values: list[float] = []
    start_s, end_s = float(cumulative_xy[start_idx]), float(cumulative_xy[end_idx])
    for time_s in HOLONOMIC_TOKEN_TIMES_S:
        target_s = min(start_s + max(0.0, float(speed)) * time_s, end_s)
        index = int(np.clip(np.searchsorted(cumulative_xy, target_s, side="left"), start_idx, end_idx))
        local = transform_point_to_yaw_frame(path_w[index], robot_pos_w, robot_quat_w)
        dyaw = relative_yaw(robot_quat_w, yaw_to_quat_wxyz(float(yaws_w[index])))
        vx_ref, vy_ref, wz_ref = (float(value) for value in commands[index, :3])
        cos_yaw, sin_yaw = math.cos(float(dyaw)), math.sin(float(dyaw))
        values.extend((
            float(local[0]), float(local[1]), math.sin(float(dyaw)), math.cos(float(dyaw)),
            cos_yaw * vx_ref - sin_yaw * vy_ref,
            sin_yaw * vx_ref + cos_yaw * vy_ref,
            wz_ref, float(time_s),
        ))
    return np.asarray(values, dtype=np.float32)


def build_goal_batch_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    path_progress: np.ndarray | None = None,
    waypoint_time_offsets_s: tuple[float, float, float] = WAYPOINT_TIME_OFFSETS_S,
    v_req_clip: float = 2.0,
) -> np.ndarray:
    """Build goals for a batch of robots following the same planned path."""

    num_envs = robot_pos_w.shape[0]
    if path_progress is None:
        path_progress = np.zeros(num_envs, dtype=np.int32)

    goals = []
    for i in range(num_envs):
        start_idx = advance_path_progress(path_w, robot_pos_w[i], int(path_progress[i]))
        path_progress[i] = start_idx
        goals.append(
            build_goal_from_path(
                path_w,
                cumulative_xy,
                yaws_w,
                robot_pos_w[i],
                robot_quat_w[i],
                goal_horizon_steps=goal_horizon_steps,
                dt=dt,
                speed=speed,
                start_idx=start_idx,
                waypoint_time_offsets_s=waypoint_time_offsets_s,
                v_req_clip=v_req_clip,
            )
        )
    return np.stack(goals, axis=0).astype(np.float32)


def build_holonomic_goal_batch_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    path_progress: np.ndarray | None = None,
    reference_command_b: np.ndarray | None = None,
) -> np.ndarray:
    """Batch counterpart of :func:`build_holonomic_goal_from_path`."""

    num_envs = robot_pos_w.shape[0]
    if path_progress is None:
        path_progress = np.zeros(num_envs, dtype=np.int32)
    goals = []
    for index in range(num_envs):
        start_idx = advance_path_progress(path_w, robot_pos_w[index], int(path_progress[index]))
        path_progress[index] = start_idx
        goals.append(build_holonomic_goal_from_path(
            path_w, cumulative_xy, yaws_w, robot_pos_w[index], robot_quat_w[index],
            goal_horizon_steps=goal_horizon_steps, dt=dt, speed=speed, start_idx=start_idx,
            reference_command_b=reference_command_b,
        ))
    return np.stack(goals).astype(np.float32)


def build_path_guidance_goal_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    speed: float,
    start_idx: int | None = None,
) -> np.ndarray:
    """Deployment form of the velocity-free path-guidance contract."""

    if start_idx is None:
        start_idx = closest_path_index(path_w, robot_pos_w)
    start = int(np.clip(start_idx, 0, len(path_w) - 1))
    remaining_m = max(0.0, float(cumulative_xy[-1] - cumulative_xy[start]))
    time_to_go = remaining_m / max(abs(float(speed)), 1.0e-3)
    return build_path_guidance_goal_vector(
        path_w,
        cumulative_xy,
        path_w,
        yaws_w,
        start,
        len(path_w) - 1,
        robot_pos_w,
        robot_quat_w,
        dt=1.0,
        time_to_go_s=time_to_go,
    )


def build_path_guidance_goal_batch_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    speed: float,
    path_progress: np.ndarray | None = None,
) -> np.ndarray:
    """Batch counterpart of :func:`build_path_guidance_goal_from_path`."""

    num_envs = robot_pos_w.shape[0]
    if path_progress is None:
        path_progress = np.zeros(num_envs, dtype=np.int32)
    goals = []
    for index in range(num_envs):
        start_idx = advance_path_progress(path_w, robot_pos_w[index], int(path_progress[index]))
        path_progress[index] = start_idx
        goals.append(build_path_guidance_goal_from_path(
            path_w,
            cumulative_xy,
            yaws_w,
            robot_pos_w[index],
            robot_quat_w[index],
            speed=speed,
            start_idx=start_idx,
        ))
    return np.stack(goals).astype(np.float32)
