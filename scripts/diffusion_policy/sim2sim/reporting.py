"""Existing first-arrival metrics and bounded transfer plots; no reward changes."""
import json
from pathlib import Path
import numpy as np
from scripts.diffusion_policy.evaluation.metrics import (
    compute_first_task_success, compute_active_tracking_metrics,
    project_trajectory_to_polyline,
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def episode_metrics(condition, path, yaw, trace, start_position, failure, failure_time_s, manifest, task_config):
    valid = trace["valid"]
    positions = trace["position"][valid]
    result = dict(condition, valid_steps=int(valid.sum()), failure=failure, failure_time_s=failure_time_s,
                  arrived=False, task_success=False, adherence_success=False,
                  full_horizon_survival=failure is None, active_survival=failure is None)
    if not len(positions):
        return result
    start_arc = float(project_trajectory_to_polyline(np.asarray(start_position)[None, :2], path[:, :2])[0][0])
    length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
    target_progress = min(condition["speed_mps"] * manifest["duration_s"], max(length - start_arc, 0))
    if target_progress <= 0:
        result["failure"] = "invalid_active_origin"; result["active_survival"] = False
        return result
    task = compute_first_task_success(
        positions[:, :2], path[:, :2], yaws_rad=trace["yaw"][valid], heights_m=positions[:, 2],
        requested_speed_m_s=condition["speed_mps"], target_progress_m=target_progress,
        target_yaw_rad=float(yaw[-1]), target_height_m=float(path[-1, 2]), dt=manifest["dt_s"],
        start_position_xy=np.asarray(start_position)[:2], target_heights_m=trace["required_height"][valid],
        profile_height_mae_tolerance_m=task_config.get("profile_height_mae_tolerance_m", .04),
        path_heights_m=path[:, 2], profile_transition_margin_m=manifest["profile_transition_margin_m"],
        position_tolerance_m=.15, yaw_tolerance_rad=.4, height_tolerance_m=.05,
        mean_speed_tolerance_m_s=.08, corridor_half_width_m=.6,
    )
    active_steps = max(1, min(len(positions), round(task.time_s / manifest["dt_s"]))) if task.arrived else len(positions)
    tracking = compute_active_tracking_metrics(
        positions[:, :2], path[:, :2], heights_m=positions[:, 2],
        target_heights_m=trace["required_height"][valid], start_position_xy=np.asarray(start_position)[:2],
        active_steps=active_steps,
        transition_progress_m=length * .5 - start_arc if "_to_" in condition["profile"] else None,
        transition_margin_m=manifest["profile_transition_margin_m"],
    )
    result.update(arrived=task.arrived, task_success=task.success,
                  adherence_success=bool(task.success and tracking.cross_track_rmse_m <= .1),
                  active_survival=bool(failure is None or task.arrived),
                  task=task.to_dict(), tracking=tracking.to_dict(),
                  max_torque_nm=float(trace["torque_max"][valid].max()),
                  torque_saturation_fraction=float(trace["saturated_fraction"][valid].mean()))
    return result


def summarize(episodes):
    summary = {"attempts": len(episodes), "profiles": {}}
    for group, selected in [("all", episodes)] + [(p, [e for e in episodes if e["profile"] == p])
                                                   for p in sorted({e["profile"] for e in episodes})]:
        summary["profiles"][group] = {
            "n": len(selected),
            **{k: int(sum(bool(e[k]) for e in selected)) for k in
               ("arrived", "task_success", "adherence_success", "active_survival", "full_horizon_survival")},
            "failure_counts": {f: sum(e["failure"] == f for e in selected)
                               for f in sorted({e["failure"] for e in selected if e["failure"] is not None})},
        }
    return summary


def plots(output, conditions, paths, traces, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [i for i, c in enumerate(conditions) if c["id"] in manifest["smoke_ids"]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, i in zip(axes.flat, selected):
        c, path, t = conditions[i], paths[i], traces[i]
        points = t["position"][t["valid"]]
        ax.plot(path[:, 0], path[:, 1], "--", color=".25", label="Reference")
        if len(points):
            ax.plot(points[:, 0], points[:, 1], color="#0072B2", label="MuJoCo")
            ax.scatter(*points[-1, :2], marker="x", color="#D55E00")
        ax.set_title(f"{c['family']}: {c['profile']}, {c['speed_mps']:g} m/s", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(Path(output) / "geometry.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 2.2 * len(selected)), constrained_layout=True)
    for row, i in enumerate(selected):
        c, t = conditions[i], traces[i]; valid = t["valid"]
        times = (np.arange(len(valid)) + 1)[valid] * manifest["dt_s"]
        axes[row, 0].plot(times, t["position"][valid, 2], color="#0072B2", label="Actual")
        axes[row, 0].plot(times, t["required_height"][valid], "--", color=".25", label="Required")
        axes[row, 1].plot(times, t["tangent_speed"][valid], color="#0072B2", label="Local tangent speed")
        axes[row, 1].axhline(c["speed_mps"], color=".25", ls="--", label="Requested mean")
        for ax in axes[row]:
            ax.set_xlabel("Active time [s]"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        axes[row, 0].set_ylabel("Base height [m]"); axes[row, 1].set_ylabel("Speed [m/s]")
        axes[row, 0].set_title(c["id"], fontsize=9)
    fig.savefig(Path(output) / "height_speed.png", dpi=170); plt.close(fig)
